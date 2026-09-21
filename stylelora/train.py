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
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image

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


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_pixels(paths: list[Path], device: str) -> torch.Tensor:
    """Images as a tensor in [-1, 1], which is the range the VAE expects."""
    arrays = [
        np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
        for p in paths
    ]
    stacked = torch.from_numpy(np.stack(arrays))  # (n, H, W, 3)
    return (stacked.permute(0, 3, 1, 2) / 127.5 - 1.0).to(device)


def train(style: str, images: Path, out: Path, steps: int = STEPS, seed: int = 0) -> Path:
    paths = sorted(images.glob("*.png"))
    if not paths:
        raise ValueError(f"no images in {images}")

    torch.manual_seed(seed)
    device = _device()

    pipe = AutoPipelineForText2Image.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
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
        pixels = _load_pixels(paths, device)
        latents = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
        tokens = pipe.tokenizer(
            [CAPTION] * len(paths),
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

    generator = torch.Generator(device="cpu").manual_seed(seed)
    horizon = pipe.scheduler.config.num_train_timesteps

    for step in range(steps):
        i = int(torch.randint(0, len(latents), (1,), generator=generator).item())
        latent = latents[i : i + 1]
        noise = torch.randn(latent.shape, generator=generator).to(device)
        timestep = torch.randint(0, horizon, (1,), generator=generator).to(device)

        noisy = pipe.scheduler.add_noise(latent, noise, timestep)
        predicted = unet(noisy, timestep, encoder_hidden_states=embeds[i : i + 1]).sample
        # The whole of diffusion training: how wrong was the guess at the noise.
        loss = torch.nn.functional.mse_loss(predicted, noise)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step and step % 100 == 0:
            print(f"  {style} step {step}/{steps}  loss {loss.item():.4f}", flush=True)

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
