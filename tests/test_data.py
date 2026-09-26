from pathlib import Path

import pytest
from PIL import Image

from stylelora.data import (
    FALLBACK_CAPTION,
    HOLDOUT,
    LABEL_NAMES,
    PER_STYLE,
    SIZE,
    STYLE_WORDS,
    STYLES,
    TRAIN_POOL,
    Row,
    choose,
    prepare,
    sanitise,
    split,
    style_index,
    write_captions,
)


def _rows(n: int) -> list[Row]:
    return [{"id": i} for i in range(n)]


# --- the pair ---------------------------------------------------------------


def test_both_styles_exist_in_the_dataset() -> None:
    for name in STYLES:
        assert isinstance(style_index(name), int)


def test_the_two_styles_are_different_labels() -> None:
    assert style_index(STYLES[0]) != style_index(STYLES[1])


def test_an_unknown_style_is_refused() -> None:
    with pytest.raises(ValueError):
        style_index("Vaporwave")


def test_both_styles_depict_something() -> None:
    """A pair where one side has no subject would confound style with content:
    an adapter that removed every recognisable object would score well, and
    what this project measures is exactly what style costs in objects."""
    assert set(STYLES) == {"Ukiyo_e", "Baroque"}


# --- the split --------------------------------------------------------------


def test_the_counts_add_up() -> None:
    assert PER_STYLE == TRAIN_POOL + HOLDOUT


def test_the_pair_fits_the_smaller_style() -> None:
    """Ukiyo-e has 66 works in this dataset and the two sets have to match, so
    asking for more would quietly give the styles different sizes."""
    assert PER_STYLE <= 66


def test_the_holdout_is_not_in_the_training_pool(tmp_path: Path) -> None:
    for i in range(PER_STYLE):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{i:03d}.png")

    pool, holdout = split(tmp_path)

    assert len(pool) == TRAIN_POOL
    assert len(holdout) == HOLDOUT
    assert not set(pool) & set(holdout)


# --- choosing ---------------------------------------------------------------


def test_a_smaller_set_is_a_subset_of_a_larger_one() -> None:
    """The data axis compares few images against many. They have to be the same
    few, or the two runs differ by which paintings as well as by how many."""
    rows = _rows(200)

    small = {r["id"] for r in choose(rows, count=20)}
    large = {r["id"] for r in choose(rows, count=60)}

    assert small <= large


def test_choosing_does_not_take_them_in_order() -> None:
    """Order in these collections tracks the artist, and taking the first N
    once produced twenty 'Baroque' images that were twenty Rembrandts."""
    rows = _rows(200)

    assert [r["id"] for r in choose(rows, count=20)] != list(range(20))


def test_the_same_seed_gives_the_same_set() -> None:
    assert choose(_rows(200), count=20) == choose(_rows(200), count=20)


def test_asking_for_more_than_exists_returns_what_exists() -> None:
    assert len(choose(_rows(5), count=50)) == 5


# --- preparing --------------------------------------------------------------


def test_preparing_makes_a_square_of_the_requested_size() -> None:
    assert prepare(Image.new("RGB", (900, 400), (10, 20, 30))).size == (SIZE, SIZE)


def test_preparing_crops_rather_than_squashes() -> None:
    """A squashed painting is a different painting; the aspect ratio is kept."""
    wide = Image.new("RGB", (800, 400), (0, 0, 0))
    wide.paste(Image.new("RGB", (100, 400), (255, 0, 0)), (350, 0))

    centre = prepare(wide).getpixel((SIZE // 2, SIZE // 2))
    assert isinstance(centre, tuple)
    assert centre[0] > 200 and centre[1] < 60


def test_preparing_keeps_the_middle_not_a_corner() -> None:
    tall = Image.new("RGB", (400, 800), (0, 0, 0))
    tall.paste(Image.new("RGB", (400, 100), (0, 255, 0)), (0, 350))

    centre = prepare(tall).getpixel((SIZE // 2, SIZE // 2))
    assert isinstance(centre, tuple)
    assert centre[1] > 200 and centre[0] < 60


def test_preparing_returns_rgb_whatever_went_in() -> None:
    assert prepare(Image.new("L", (600, 600), 128)).mode == "RGB"


def test_the_size_matches_what_the_base_model_was_trained_at() -> None:
    assert SIZE == 512


# --- captions ---------------------------------------------------------------


def test_a_caption_keeps_its_subject() -> None:
    assert sanitise("a woman standing beside a river") == "a woman standing beside a river"


def test_the_style_is_stripped_out_of_a_caption() -> None:
    """The captioner is a model and will write 'an ukiyo-e print of a wave'.
    That hands the style to the text encoder, and the adapter is then measured
    on a job the prompt was already doing."""
    assert "ukiyo" not in sanitise("an ukiyo-e print of a great wave").lower()
    assert "baroque" not in sanitise("a baroque portrait of an old man").lower()


def test_a_caption_that_was_only_a_style_falls_back() -> None:
    """Better to admit the caption knows nothing than to teach 'a of'."""
    assert sanitise("baroque") == FALLBACK_CAPTION
    assert sanitise("japanese woodblock") == FALLBACK_CAPTION


def test_punctuation_does_not_hide_a_style_word() -> None:
    assert "baroque" not in sanitise("a portrait, baroque, of a man").lower()


def test_the_fallback_names_no_style() -> None:
    for word in STYLE_WORDS:
        assert word not in FALLBACK_CAPTION.lower()


def test_every_image_gets_its_own_caption(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"{i:02d}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    def three(images: list[Image.Image]) -> list[str]:
        return [f"a scene with {len(images)} things"] * 3

    written = write_captions(paths, tmp_path, three)

    assert len(written) == 3
    assert set(written) == {p.name for p in paths}


def test_a_captioner_that_returns_the_wrong_count_is_refused(tmp_path: Path) -> None:
    """A silent mismatch would pair every image with someone else's subject."""
    p = tmp_path / "00.png"
    Image.new("RGB", (8, 8)).save(p)

    with pytest.raises(ValueError):
        write_captions([p], tmp_path, lambda images: [])


def test_the_label_order_matches_the_dataset_card() -> None:
    """The loading script is gone, so this order is not checkable at runtime --
    a wrong index here silently trains on another style entirely."""
    assert len(LABEL_NAMES) == 27
    assert LABEL_NAMES[8] == "Ukiyo_e"
    assert LABEL_NAMES[21] == "Baroque"
