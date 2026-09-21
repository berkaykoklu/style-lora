import numpy as np
from PIL import Image

from stylelora.score import embed_images, embed_text


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
