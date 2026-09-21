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
