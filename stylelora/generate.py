"""Generating the same subjects at a given adapter strength.

The prompt set never changes and the seeds never change, so two strengths
differ by strength alone. Fresh prompts or fresh seeds per strength would put
the difference somewhere the measurement cannot separate it from the effect.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from stylelora.train import BASE_MODEL, _device, _dtype

# Subjects only. No style word appears here -- test_generate.py fails if one
# creeps in, because a prompt that names the style would let the base model
# produce it too and the adapter's contribution would vanish into the wording.
PROMPTS: tuple[str, ...] = (
    "a woman holding a lantern",
    "a knight standing in a doorway",
    "a fox in a forest clearing",
    "a city street after rain",
    "a young man reading a letter",
    "a harbour at sunrise",
    "a cat asleep on a windowsill",
    "two people talking at a table",
    "a horse in an open field",
    "a staircase in an empty hall",
    "a bowl of fruit on a cloth",
    "a traveller on a mountain path",
)

# sd-turbo is distilled for very few steps and trained without classifier-free
# guidance, so a guidance scale above zero makes it worse rather than better.
STEPS = 2
GUIDANCE = 0.0


@lru_cache(maxsize=1)
def _base_pipe() -> Any:
    """Loaded once. A sweep builds this seven times over otherwise.

    Typed as Any because from_pretrained picks a concrete pipeline class at
    runtime; the auto class is a factory, not the type it returns.
    """
    device = _device()
    pipe = AutoPipelineForText2Image.from_pretrained(BASE_MODEL, torch_dtype=_dtype(device))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    return pipe


def generate(
    lora: Path | None,
    strength: float,
    prompts: tuple[str, ...] = PROMPTS,
    seed: int = 0,
) -> list[Image.Image]:
    """The prompt set at one adapter strength.

    `lora=None` or `strength=0` is the base model, which is the baseline every
    score is read against rather than an assumed zero.
    """
    pipe = _base_pipe()

    applied = lora is not None and strength > 0
    if applied:
        assert lora is not None
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
        pipe.fuse_lora(lora_scale=strength)

    try:
        images: list[Image.Image] = []
        for i, prompt in enumerate(prompts):
            generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + i)
            images.append(
                pipe(
                    prompt,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    generator=generator,
                ).images[0]
            )
        return images
    finally:
        # Fusing writes the adapter into the weights, so it has to come back
        # out: the cached pipeline is reused at every strength, and a fused
        # adapter left behind would stack onto the next one.
        if applied:
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
