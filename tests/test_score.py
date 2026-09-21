import numpy as np
import pytest
from PIL import Image

from stylelora.score import (
    embed_images,
    embed_text,
    prompt_score,
    style_centre,
    style_score,
)


def _solid(colour: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (224, 224), colour)


def test_image_embeddings_are_unit_length() -> None:
    """Normalised rows are what makes a dot product a cosine later."""
    vectors = embed_images([_solid((200, 30, 30)), _solid((30, 30, 200))])

    assert vectors.shape == (2, 512)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_text_embeddings_are_unit_length() -> None:
    vectors = embed_text(["a red square", "a blue square"])

    assert vectors.shape == (2, 512)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_an_image_is_identical_to_itself() -> None:
    image = _solid((120, 200, 90))
    a, b = embed_images([image, image.copy()])

    assert float(a @ b) > 0.999


def test_two_different_images_are_not_identical() -> None:
    a, b = embed_images([_solid((240, 20, 20)), _solid((20, 20, 240))])

    assert float(a @ b) < 0.99


def test_an_image_lands_nearer_the_text_that_describes_it() -> None:
    """The whole point of CLIP: pictures and words share one space."""
    images = embed_images([_solid((240, 20, 20)), _solid((20, 20, 240))])
    texts = embed_text(["a solid red square", "a solid blue square"])

    assert float(images[0] @ texts[0]) > float(images[0] @ texts[1])


def test_the_centre_is_unit_length() -> None:
    centre = style_centre(embed_images([_solid((200, 30, 30)), _solid((190, 40, 20))]))

    assert centre.shape == (512,)
    assert abs(float(np.linalg.norm(centre)) - 1.0) < 1e-5


def test_the_centre_of_one_image_is_that_image() -> None:
    vectors = embed_images([_solid((80, 160, 220))])

    assert float(style_centre(vectors) @ vectors[0]) > 0.999


def test_an_empty_style_is_refused_rather_than_averaged_to_nothing() -> None:
    with pytest.raises(ValueError):
        style_centre(np.zeros((0, 512), dtype=np.float32))


def test_an_image_from_the_style_scores_higher_than_one_outside_it() -> None:
    reds = embed_images([_solid((210, 20, 20)), _solid((190, 35, 30)), _solid((230, 10, 40))])
    centre = style_centre(reds)
    probe = embed_images([_solid((200, 25, 25)), _solid((20, 30, 210))])

    scores = style_score(probe, centre)

    assert scores[0] > scores[1]


def test_prompt_score_pairs_each_image_with_its_own_prompt() -> None:
    images = embed_images([_solid((220, 20, 20)), _solid((20, 20, 220))])
    prompts = embed_text(["a red square", "a blue square"])

    scores = prompt_score(images, prompts)

    assert scores.shape == (2,)


def test_prompt_score_pairs_by_position_not_by_best_match() -> None:
    """Scoring every image against every prompt and taking the best would make
    a model that ignored the prompt look obedient."""
    images = embed_images([_solid((220, 20, 20)), _solid((20, 20, 220))])
    right = embed_text(["a solid red square", "a solid blue square"])
    swapped = embed_text(["a solid blue square", "a solid red square"])

    assert prompt_score(images, right).mean() > prompt_score(images, swapped).mean()


def test_prompt_score_refuses_mismatched_lengths() -> None:
    """Silently zipping to the shorter one would score images against the wrong text."""
    images = embed_images([_solid((220, 20, 20)), _solid((20, 20, 220))])
    prompts = embed_text(["a red square"])

    with pytest.raises(ValueError):
        prompt_score(images, prompts)
