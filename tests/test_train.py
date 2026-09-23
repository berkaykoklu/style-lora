import os
from pathlib import Path

import pytest
from PIL import Image

from stylelora.train import BASE_MODEL, CAPTION, LR, RANK, STEPS, train


def test_the_base_model_is_the_cached_one() -> None:
    """A different id here means a multi-gigabyte download nobody asked for."""
    assert BASE_MODEL == "stabilityai/sd-turbo"


def test_the_rank_is_small_enough_to_be_a_style_adapter() -> None:
    """Rank is capacity. A large one starts memorising paintings rather than
    learning the style they share."""
    assert RANK <= 16


def test_the_caption_names_no_style() -> None:
    """The images carry the style. Saying it in the caption as well would teach
    the model to wait for the word before applying it."""
    for word in ("baroque", "nouveau", "style"):
        assert word not in CAPTION.lower()


def test_both_styles_would_get_the_same_settings() -> None:
    """One comparison, one difference. Tuning per style would add a second."""
    assert isinstance(STEPS, int) and isinstance(LR, float)


def test_an_empty_folder_is_refused_rather_than_trained_on_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError):
        train("Test", images=empty, out=tmp_path / "lora", steps=1)


# Loading the real pipeline takes several gigabytes and filled a 16 GB machine
# the first time this suite ran it. It stays out of the default run; set
# STYLELORA_HEAVY=1 to include it, with nothing else heavy open.
heavy = pytest.mark.skipif(
    os.environ.get("STYLELORA_HEAVY") != "1",
    reason="loads the full diffusion pipeline; set STYLELORA_HEAVY=1",
)


@heavy
def test_a_tiny_run_writes_weights(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(2):
        Image.new("RGB", (512, 512), (40 * i, 80, 160)).save(images / f"{i}.png")

    out = train("Test", images=images, out=tmp_path / "lora", steps=2)

    assert out.exists()


def test_half_precision_only_where_it_is_well_supported() -> None:
    """MPS half precision is still patchy; that path stays in float32 and
    relies on the memory cap instead."""
    from stylelora.train import _dtype

    assert _dtype("cuda").itemsize == 2
    assert _dtype("mps").itemsize == 4
    assert _dtype("cpu").itemsize == 4
