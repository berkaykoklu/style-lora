"""LoRA training for one style.

Diffusion training is one idea repeated: add a known amount of noise to a real
image, ask the model how much it thinks is there, and nudge it toward the right
answer. Generation is the same skill run backwards.

LoRA leaves the model's own weights frozen and trains a small pair of matrices
beside them. That is what makes the rest of this project possible: an adapter
has a strength dial, and a fully fine-tuned model does not.

Both styles are trained with identical settings. Tuning one and not the other
would put a second difference into a comparison designed to have exactly one.

The recipe follows `examples/text_to_image/train_text_to_image_lora.py` in
diffusers. An earlier version of this file diverged from it in two ways that
mattered -- a batch of one, and a floor on the sampled timestep -- and produced
adapters that barely moved. The floor was a patch for sd-turbo, which visits
only two timesteps at inference; SD 1.5 visits the whole range and the patch
would now leave half of it untrained.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, DDPMScheduler, StableDiffusionPipeline
from diffusers.training_utils import compute_snr
from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image

from stylelora import budget
from stylelora.data import CAPTIONS_FILE, FALLBACK_CAPTION, SIZE

BASE_MODEL = "sd-legacy/stable-diffusion-v1-5"

# Rank is the adapter's capacity. The reference trains a style at 62; the sweep
# here runs 8 to 64 and this is its middle.
RANK = 32

# Per sample, scaled by the effective batch below -- the reference does the same,
# so that changing the batch does not quietly change how hard each step pulls.
LR = 1e-5

STEPS = 1000
BATCH = 4
ACCUMULATION = 2

# Min-SNR weighting (arxiv 2303.09556).
#
# Every timestep is a different task, and left alone they do not contribute
# equally. The weight is min(SNR, gamma) / SNR, which is 1 wherever the
# signal-to-noise ratio is below gamma and falls away above it. Measured on this
# schedule:
#
#     t=10   SNR 103.5   weight 0.048
#     t=100  SNR   8.5   weight 0.591
#     t=300  SNR   1.4   weight 1.000
#     t=900  SNR   0.01  weight 1.000
#
# So it holds back the near-clean end of the schedule, where the latent is
# mostly signal and the epsilon objective produces outsized gradients, and
# leaves the noisy end untouched.
#
# The TIMESTEP_FLOOR this replaces was the same instinct swung much harder: it
# gave weight 1 above 499 and weight 0 below, throwing those steps away rather
# than damping them. That was defensible for a two-step sampler and is not for a
# twenty-five step one, which spends most of its steps down there.
SNR_GAMMA = 5.0

MAX_GRAD_NORM = 1.0
WEIGHTS_NAME = "pytorch_lora_weights.safetensors"

# How many images go through the VAE at once. Kept equal to the batch: the VAE
# runs inside the loop now rather than once up front, because the horizontal
# flip changes the image and a cached latent would be of the unflipped one.
FLIP_PROBABILITY = 0.5


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _dtype(device: str) -> torch.dtype:
    """Half precision on CUDA, full precision everywhere else.

    Half precision alone produced a loss of nan from the first step: Stable
    Diffusion's VAE overflows float16's range, and once a value reaches inf the
    gradient after it is nan and every weight past it is ruined. The fix was not
    to abandon float16 -- that was an over-correction, and it halves what fits
    in a batch -- but to use it the way the reference does: the frozen model in
    half precision, the trainable LoRA parameters in full, and an autocast scaler
    to keep the gradients in range.

    MPS has no GradScaler, so that machine stays in float32 and pays for it.
    """
    return torch.float16 if device == "cuda" else torch.float32


def _load_batch(
    paths: list[Path],
    captions: dict[str, str],
    rng: random.Random,
) -> tuple[torch.Tensor, list[str]]:
    """Images in [-1, 1], which is the range the VAE expects, and their captions.

    Half of them are mirrored. With sixty paintings per style the adapter sees
    the same canvas many times over, and a mirrored painting is the same style
    with a different composition -- the cheapest honest way to widen the set.
    Mirroring never changes what the caption says.
    """
    arrays, texts = [], []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((SIZE, SIZE))
        if rng.random() < FLIP_PROBABILITY:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        arrays.append(np.asarray(image, dtype=np.float32))
        texts.append(captions.get(path.name, FALLBACK_CAPTION))
    stacked = torch.from_numpy(np.stack(arrays))  # (n, H, W, 3)
    return stacked.permute(0, 3, 1, 2) / 127.5 - 1.0, texts


def training_scheduler() -> Any:
    """The forward process to fine-tune against.

    Deliberately not the pipeline's sampling scheduler. A sampler adds noise the
    way inference needs it and, handed raw timesteps, produces latents at a scale
    the UNet has never seen -- what the adapter then learns is how to compensate
    for them, and both styles come out as the same smeared texture.
    """
    return DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")


def check_finite(loss: torch.Tensor, step: int) -> None:
    """Stop the moment the loss stops being a number.

    A run that produces nan still finishes, still prints its progress and still
    writes a file. Every measurement taken from those weights afterwards would be
    measuring nothing, and nothing in the output would say so.
    """
    if not torch.isfinite(loss):
        raise RuntimeError(f"loss became {loss.item()} at step {step}; weights are not usable")


def snr_weights(scheduler: Any, timesteps: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per-sample loss weights that even out how hard each noise level is."""
    snr = compute_snr(scheduler, timesteps)
    clipped = torch.stack([snr, gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
    weights: torch.Tensor = clipped / snr
    return weights


def train(
    style: str,
    images: Path,
    out: Path,
    steps: int = STEPS,
    seed: int = 0,
    rank: int = RANK,
    limit: int | None = None,
    batch: int = BATCH,
    accumulation: int = ACCUMULATION,
) -> Path:
    """One adapter. `limit` takes the first N images; `rank` sets its capacity.

    Both are arguments rather than constants because the experiment sweeps them,
    and a sweep that edits a module between runs cannot say afterwards which
    value produced which weights.
    """
    paths = sorted(images.glob("*.png"))[:limit]
    if not paths:
        raise ValueError(f"no images in {images}")
    if limit is not None and len(paths) < limit:
        raise ValueError(f"{images} holds {len(paths)} images, {limit} asked for")

    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = _device()
    dtype = _dtype(device)
    if device == "mps":
        budget.apply()

    captions_path = images / CAPTIONS_FILE
    captions: dict[str, str] = (
        json.loads(captions_path.read_text()) if captions_path.exists() else {}
    )

    pipe = AutoPipelineForText2Image.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    unet, vae, text_encoder = pipe.unet, pipe.vae, pipe.text_encoder
    for module in (unet, vae, text_encoder):
        module.requires_grad_(False)

    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    # The frozen model is half precision; what is trained is not. Half-precision
    # gradients on a rank-32 matrix underflow long before the weights do.
    trainable = [p for p in unet.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.to(torch.float32)

    unet.enable_gradient_checkpointing()
    unet.train()
    vae.eval()
    text_encoder.eval()

    # Scaled by the effective batch, as the reference does: a bigger batch is a
    # steadier gradient, so it can take a proportionally longer stride.
    lr = LR * batch * accumulation
    optimiser = torch.optim.AdamW(trainable, lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=dtype == torch.float16)
    noise_scheduler = training_scheduler()
    horizon = noise_scheduler.config.num_train_timesteps

    report_every = max(steps // 10, 1)
    started = time.perf_counter()
    order: list[Path] = []

    for step in range(steps):
        for _ in range(accumulation):
            if len(order) < batch:
                order = list(paths)
                rng.shuffle(order)
                order = order * math.ceil(batch / len(order))
            chosen, order = order[:batch], order[batch:]

            pixels, texts = _load_batch(chosen, captions, rng)
            pixels = pixels.to(device=device, dtype=dtype)

            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype == torch.float16):
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                    tokens = pipe.tokenizer(
                        texts,
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device)
                    encoder_hidden_states = text_encoder(tokens)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, horizon, (latents.shape[0],), device=device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                predicted = unet(noisy, timesteps, encoder_hidden_states).sample

                # In float32: the squared error of half-precision tensors
                # underflows long before the values themselves do.
                per_sample = torch.nn.functional.mse_loss(
                    predicted.float(), noise.float(), reduction="none"
                ).mean(dim=list(range(1, predicted.ndim)))
                weights = snr_weights(noise_scheduler, timesteps, SNR_GAMMA)
                loss = (per_sample * weights).mean() / accumulation

            check_finite(loss, step)
            scaler.scale(loss).backward()

        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad()

        if (step + 1) % report_every == 0:
            per_step = (time.perf_counter() - started) / (step + 1)
            print(
                f"  {style} r{rank} n{len(paths)} step {step + 1}/{steps}  "
                f"loss {loss.item() * accumulation:.4f}  {per_step:.2f}s/step",
                flush=True,
            )

    out.mkdir(parents=True, exist_ok=True)
    # Saved through the library's own helper rather than torch.save.
    # get_peft_model_state_dict names its keys peft's way; load_lora_weights
    # looks for diffusers' names, finds none, warns, and carries on with the base
    # model -- so an earlier version of this trained correctly, saved correctly,
    # and loaded nothing at all.
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(out),
        unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
        safe_serialization=True,
    )
    return out / WEIGHTS_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    saved = train(
        args.style,
        images=Path("data") / args.style,
        out=args.out or Path("runs") / args.style,
        steps=args.steps,
        seed=args.seed,
        rank=args.rank,
        limit=args.limit,
        batch=args.batch,
    )
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
