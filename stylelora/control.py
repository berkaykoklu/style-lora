"""A control for the style score.

Style adherence is a cosine against the mean of the training images, and a
mean has no idea what a style is. If those paintings are mostly dark and warm,
then anything dark and warm scores well -- including an image that has learned
nothing at all.

So the base model's own output is tinted, warmed and softened, and scored the
same way. Those transformations know nothing about Baroque or Art Nouveau;
they only move tone. Whatever they earn is what the metric gives away for
free, and the adapter has to beat it before its score means anything.

The first broken adapter this project produced was warm and soft, which is
precisely the shape this control was written to catch.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

Transform = Callable[[Image.Image], Image.Image]


def sepia(image: Image.Image) -> Image.Image:
    """Warm brown tone. No knowledge of any style, only of temperature."""
    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    tint = np.stack([grey * 1.07, grey * 0.94, grey * 0.74], axis=-1)
    return Image.fromarray((np.clip(tint, 0, 1) * 255).astype(np.uint8))


def darken(image: Image.Image) -> Image.Image:
    """Lower the exposure. Baroque paintings are dark; so is this."""
    return ImageEnhance.Brightness(image).enhance(0.65)


def soften(image: Image.Image) -> Image.Image:
    """Blur away fine detail -- what the first working adapter mostly did."""
    return image.filter(ImageFilter.GaussianBlur(radius=3))


def muted(image: Image.Image) -> Image.Image:
    """Drain the colour. Flat pastel palettes read as Art Nouveau."""
    return ImageEnhance.Color(image).enhance(0.45)


CONTROLS: dict[str, Transform] = {
    "sepia": sepia,
    "darken": darken,
    "soften": soften,
    "muted": muted,
}


def apply(name: str, images: list[Image.Image]) -> list[Image.Image]:
    if name not in CONTROLS:
        raise ValueError(f"{name} is not a control; have {sorted(CONTROLS)}")
    transform = CONTROLS[name]
    return [transform(image) for image in images]
