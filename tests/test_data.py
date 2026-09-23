import pytest
from PIL import Image

from stylelora.data import PER_STYLE, SIZE, STYLES, prepare, style_index


def test_both_styles_have_a_label() -> None:
    for name in STYLES:
        assert isinstance(style_index(name), int)


def test_baroque_and_art_nouveau_are_different_labels() -> None:
    assert style_index("Baroque") != style_index("Art_Nouveau")


def test_an_unknown_style_is_refused() -> None:
    with pytest.raises(ValueError):
        style_index("Vaporwave")


def test_preparing_makes_a_square_of_the_requested_size() -> None:
    out = prepare(Image.new("RGB", (900, 400), (10, 20, 30)))

    assert out.size == (SIZE, SIZE)


def test_preparing_crops_rather_than_squashes() -> None:
    """A squashed painting is a different painting; the aspect ratio is kept."""
    wide = Image.new("RGB", (800, 400), (0, 0, 0))
    wide.paste(Image.new("RGB", (100, 400), (255, 0, 0)), (350, 0))  # red stripe, centred

    out = prepare(wide)
    centre = out.getpixel((SIZE // 2, SIZE // 2))
    assert isinstance(centre, tuple)

    assert centre[0] > 200 and centre[1] < 60


def test_preparing_keeps_the_middle_not_a_corner() -> None:
    """Cropping from a corner would systematically cut off one side of every
    painting, which is a change to the style rather than to the frame."""
    tall = Image.new("RGB", (400, 800), (0, 0, 0))
    tall.paste(Image.new("RGB", (400, 100), (0, 255, 0)), (0, 350))  # green band, centred

    centre = prepare(tall).getpixel((SIZE // 2, SIZE // 2))
    assert isinstance(centre, tuple)

    assert centre[1] > 200 and centre[0] < 60


def test_preparing_returns_rgb_whatever_went_in() -> None:
    assert prepare(Image.new("L", (600, 600), 128)).mode == "RGB"


def test_the_plan_asks_for_the_same_count_from_both_styles() -> None:
    """Different counts would make the two LoRAs differ by more than style."""
    assert PER_STYLE == 20


def test_the_size_matches_what_the_base_model_was_trained_at() -> None:
    assert SIZE == 512


def test_every_genre_has_a_caption() -> None:
    """A genre with no caption would silently fall back, and that image would
    train against text that says nothing about it."""
    from stylelora.data import GENRE_CAPTIONS, GENRE_NAMES

    for name in GENRE_NAMES:
        assert name in GENRE_CAPTIONS


def test_an_out_of_range_genre_falls_back_rather_than_failing() -> None:
    from stylelora.data import caption_for

    assert caption_for(999) == "a painting"
    assert caption_for(-1) == "a painting"


def test_a_known_genre_names_its_subject() -> None:
    from stylelora.data import GENRE_NAMES, caption_for

    assert caption_for(GENRE_NAMES.index("portrait")) == "a portrait"
    assert caption_for(GENRE_NAMES.index("landscape")) == "a landscape"
