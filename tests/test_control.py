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


def test_controls_know_nothing_about_the_training_images() -> None:
    """The point of a control is that it cannot have learned the style: these
    take one image and return one image, with no style anywhere in reach."""
    import inspect

    from stylelora import control

    assert "style" not in inspect.getsource(control.sepia)
    assert "style" not in inspect.getsource(control.darken)


def test_an_unknown_control_is_refused() -> None:
    with pytest.raises(ValueError):
        apply("baroque", [_photo()])
