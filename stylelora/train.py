"""LoRA training for one style.

Diffusion training is one idea repeated: add a known amount of noise to a real
image, ask the model how much it thinks is there, and nudge it toward the right
answer. Generation is the same skill run backwards.

LoRA leaves the model's own weights frozen and trains a small pair of matrices
beside them. That is what makes the rest of this project possible: an adapter
has a strength dial, and a fully fine-tuned model does not.

Both styles are trained with identical settings. Tuning one and not the other
would put a second difference into a comparison designed to have exactly one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image

from stylelora import budget
from stylelora.data import SIZE

BASE_MODEL = "stabilityai/sd-turbo"

# Rank is the adapter's capacity. Eight is the usual floor for a style: enough
# to move colour, line and texture, too little to memorise a painting.
RANK = 8
LR = 1e-4
STEPS = 500

# A caption with no style word in it. The images carry the style; saying it
# here as well would teach the model to wait for the word before applying it.
CAPTION = "a painting"

# How many images go through the VAE at once. Two fits comfortably; twenty in
# one pass filled a 15 GB card before the first training step.
ENCODE_BATCH = 2

# How many images go through the VAE at once. Two fits comfortably; twenty in
# one pass filled a 15 GB card before the first training step.
ENCODE_BATCH = 2


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _dtype(device: str) -> torch.dtype:
    """Full precision, everywhere.

    Half precision was tried first, to halve the model's footprint. On a T4 it
    produced a loss of nan from the first step: Stable Diffusion's VAE is known
    to overflow float16's range, and once a value reaches inf the gradient that
    follows is nan and every weight after it is ruined.

    The footprint was only ever a problem on a 16 GB laptop, and that path has
    a memory cap for it. A rented GPU has room, so nothing is bought by
    risking the arithmetic.
    """
    del device  # kept in the signature: the choice is device-shaped by nature
    return torch.float32


def _load_pixels(paths: list[Path], device: str, dtype: torch.dtype) -> torch.Tensor:
    """Images as a tensor in [-1, 1], which is the range the VAE expects."""
    arrays = [
        np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
        for p in paths
    ]
    stacked = torch.from_numpy(np.stack(arrays))  # (n, H, W, 3)
    return (stacked.permute(0, 3, 1, 2) / 127.5 - 1.0).to(device=device, dtype=dtype)


def check_finite(loss: torch.Tensor, step: int) -> None:
    """Stop the moment the loss stops being a number.

    A run that produces nan still finishes, still prints its progress and still
    writes a file. Every measurement taken from those weights afterwards would
    be measuring nothing, and nothing in the output would say so.
    """
    if not torch.isfinite(loss):
        raise RuntimeError(f"loss became {loss.item()} at step {step}; weights are not usable")


def train(style: str, images: Path, out: Path, steps: int = STEPS, seed: int = 0) -> Path:
    paths = sorted(images.glob("*.png"))
    if not paths:
        raise ValueError(f"no images in {images}")

    torch.manual_seed(seed)
    device = _device()
    if device == "mps":
        # Unified memory: without a ceiling the allocator will take the machine
        # into swap, and swapping is the freeze. On CUDA the card has its own
        # memory and the driver refuses cleanly, so no cap is needed.
        budget.apply()

    pipe = AutoPipelineForText2Image.from_pretrained(BASE_MODEL, torch_dtype=_dtype(device))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    unet = pipe.unet
    unet.add_adapter(
        LoraConfig(
            r=RANK,
            lora_alpha=RANK,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    # Backprop through the full UNet is what fills memory on a 16 GB machine:
    # every intermediate activation is kept for the backward pass. Checkpointing
    # recomputes them instead -- slower per step, a fraction of the memory.
    unet.enable_gradient_checkpointing()
    unet.train()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=LR)

    # The images and the caption never change, so they are encoded once rather
    # than on every step.
    with torch.no_grad():
        # In chunks: the VAE's intermediate activations at 512x512 are large,
        # and encoding twenty images in one pass asked a 15 GB card for 12.5 GB
        # before training had started. Training itself runs one image at a time.
        chunks = []
        for start in range(0, len(paths), ENCODE_BATCH):
            batch = _load_pixels(paths[start : start + ENCODE_BATCH], device, _dtype(device))
            chunks.append(pipe.vae.encode(batch).latent_dist.sample())
            del batch
        latents = torch.cat(chunks) * pipe.vae.config.scaling_factor
        del chunks

        # One caption, encoded once. Every image trains against the same text,
        # so twenty copies of it would be twenty copies of one answer.
        tokens = pipe.tokenizer(
            [CAPTION],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        embeds = pipe.text_encoder(tokens)[0]

    # The VAE and text encoder are needed once, above, and never again. Keeping
    # them resident through training costs about 2 GB for nothing.
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    horizon = pipe.scheduler.config.num_train_timesteps
    # Ten progress lines per run whatever its length: a long run that prints
    # nothing cannot be told apart from one that has hung.
    report_every = max(steps // 10, 1)
    started = time.perf_counter()

    for step in range(steps):
        i = int(torch.randint(0, len(latents), (1,), generator=generator).item())
        latent = latents[i : i + 1]
        noise = torch.randn(latent.shape, generator=generator).to(
            device=device, dtype=latent.dtype
        )
        timestep = torch.randint(0, horizon, (1,), generator=generator).to(device)

        noisy = pipe.scheduler.add_noise(latent, noise, timestep)
        predicted = unet(noisy, timestep, encoder_hidden_states=embeds).sample
        # The whole of diffusion training: how wrong was the guess at the noise.
        # Computed in float32: the squared error of half-precision tensors
        # underflows long before the values themselves do.
        loss = torch.nn.functional.mse_loss(predicted.float(), noise.float())

        check_finite(loss, step)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (step + 1) % report_every == 0:
            per_step = (time.perf_counter() - started) / (step + 1)
            print(
                f"  {style} step {step + 1}/{steps}  loss {loss.item():.4f}  "
                f"{per_step:.2f}s/step",
                flush=True,
            )

    out.mkdir(parents=True, exist_ok=True)
    weights = out / "lora.pt"
    torch.save(get_peft_model_state_dict(unet), weights)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    saved = train(
        args.style,
        images=Path("data") / args.style,
        out=Path("runs") / args.style,
        steps=args.steps,
        seed=args.seed,
    )
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
