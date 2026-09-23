import numpy as np
import pytest
from PIL import Image

from stylelora.control import CONTROLS, apply, sepia, soften


def _photo() -> Image.Image:
    """A colourful, detailed image -- something every control can visibly change."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8))


def test_every_control_changes_the_image() -> None:
    """A control that changed nothing would score as the base model and prove
    nothing about the metric."""
    original = _photo()
    for name in CONTROLS:
        out = apply(name, [original])[0]
        assert np.asarray(out).shape == np.asarray(original).shape
        assert not np.array_equal(np.asarray(out), np.asarray(original)), name


def test_sepia_warms_the_image() -> None:
    """Red above blue is the whole of it; no style is involved."""
    out = np.asarray(sepia(_photo()), dtype=np.float32)

    assert out[..., 0].mean() > out[..., 2].mean()


def test_soften_removes_detail() -> None:
    """Measured as the spread between neighbouring pixels, which a blur flattens."""
    original = _photo()
    before = np.asarray(original.convert("L"), dtype=np.float32)
    after = np.asarray(soften(original).convert("L"), dtype=np.float32)

    assert np.abs(np.diff(after, axis=1)).mean() < np.abs(np.diff(before, axis=1)).mean()


def test_controls_cannot_have_learned_anything() -> None:
    """The point of a control is that it has nothing to learn from.

    Each takes one image and returns one image, with no other argument to pass
    training data through, and gives the same answer every time. An earlier
    version of this test scanned the source for the word "style" and failed on
    a docstring, which tested the prose rather than the property.
    """
    import inspect

    from stylelora.control import CONTROLS

    original = _photo()
    for name, transform in CONTROLS.items():
        assert len(inspect.signature(transform).parameters) == 1, name
        first = np.asarray(transform(original))
        second = np.asarray(transform(original))
        assert np.array_equal(first, second), name


def test_an_unknown_control_is_refused() -> None:
    with pytest.raises(ValueError):
        apply("baroque", [_photo()])
