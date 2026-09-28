"""CLIP, and the two numbers built on it.

This module imports nothing that generates images. The measurement has to be
testable without a model run, and a failure here should never look like a
failure in training.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"

Vectors = npt.NDArray[np.float32]


@lru_cache(maxsize=1)
def _clip() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()
    return model, CLIPProcessor.from_pretrained(MODEL_ID)


def _normalise(raw: object) -> Vectors:
    # Some transformers versions hand back the model output rather than the
    # tensor, so the tensor is dug out before anything else touches it.
    tensor = raw if torch.is_tensor(raw) else getattr(raw, "pooler_output", None)
    if tensor is None:
        tensor = raw[0]  # type: ignore[index]
    unit = torch.nn.functional.normalize(tensor, dim=-1)
    return unit.detach().numpy().astype(np.float32)


def embed_images(images: list[Image.Image]) -> Vectors:
    """Unit-length CLIP vectors, so a dot product is already a cosine."""
    model, processor = _clip()
    with torch.no_grad():
        raw = model.get_image_features(**processor(images=images, return_tensors="pt"))
    return _normalise(raw)


def embed_text(prompts: list[str]) -> Vectors:
    model, processor = _clip()
    with torch.no_grad():
        raw = model.get_text_features(
            **processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        )
    return _normalise(raw)


def style_centre(vectors: Vectors) -> Vectors:
    """The direction a style points in.

    The target for style adherence is the training images themselves, not a
    sentence describing them: scoring against a sentence would measure how well
    the model reads that sentence, which is not what the LoRA was taught.
    """
    if len(vectors) == 0:
        raise ValueError("a style needs at least one image")
    mean = vectors.mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def style_direction(here: Vectors, there: Vectors) -> Vectors:
    """The axis from one style to the other, drawn between two sentences.

    A style centre is the mean of real paintings, and CLIP puts a Baroque
    portrait nearer an Impressionist portrait than either is to a landscape of
    its own style. So a centre carries the subjects of the images that built it,
    and an adapter is scored partly on what it drew rather than on how.

    Two sentences do not have that problem. "A japanese woodblock print" names
    no subject at all, so the axis between two such sentences is the part of
    CLIP space that separates the styles and nothing else.

    Measured on the pair this project abandoned: the image centres put the two
    styles 0.021 apart and the words sorted the same images correctly 78% of the
    time. The styles were separable; the centre could not see it.
    """
    axis = here - there
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        raise ValueError("the two descriptions embed identically; there is no axis between them")
    return (axis / norm).astype(np.float32)


def direction_score(images: Vectors, direction: Vectors) -> Vectors:
    """How far along the axis each image sits. Positive is the first style.

    Unlike `style_score` this is signed and centred on nothing: it says which of
    the two an image is nearer, not how close it is to either. That is what
    makes it immune to the drift both adapters share as strength rises.
    """
    return (images @ direction).astype(np.float32)


def style_score(images: Vectors, centre: Vectors) -> Vectors:
    """How close each image sits to the style it was meant to be in."""
    return (images @ centre).astype(np.float32)


def prompt_score(images: Vectors, prompts: Vectors) -> Vectors:
    """How close each image sits to the prompt that asked for it.

    Paired by position. Scoring every image against every prompt and keeping
    the best would let a model that ignored the prompt entirely still look
    obedient, as long as it drew something on the list.
    """
    if len(images) != len(prompts):
        raise ValueError(f"{len(images)} images against {len(prompts)} prompts")
    # Row-wise dot product: multiply element by element, then sum each row.
    # Both sides are unit length, so each sum is already a cosine.
    paired: Vectors = (images * prompts).sum(axis=1).astype(np.float32)
    return paired
