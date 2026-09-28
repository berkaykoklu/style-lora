import json
from pathlib import Path

import pytest
from PIL import Image

from form import QUESTIONS, Item, build


def _items(tmp_path: Path, n: int) -> list[Item]:
    made = []
    for i in range(n):
        path = tmp_path / f"{i:02d}.png"
        Image.new("RGB", (64, 64), (i * 7 % 255, 40, 80)).save(path)
        source = ["base", "Ukiyo_e", "Baroque"][i % 3]
        made.append(Item(path=path, source=source, prompt=f"a thing {i}"))
    return made


def test_the_form_never_says_where_an_image_came_from(tmp_path: Path) -> None:
    """Someone who could see which adapter drew an image would be rating the
    label. The key lives in a separate file the form does not contain."""
    html = build(_items(tmp_path, 6), tmp_path / "f.html", tmp_path / "k.json")

    text = html.read_text()
    assert "Ukiyo_e" not in text
    assert "Baroque" not in text or "Barok" in text  # the question names the style, not the source


def test_the_key_lines_up_with_the_order_shown(tmp_path: Path) -> None:
    """A key in the wrong order would score every answer against the wrong
    image and still look like a result."""
    items = _items(tmp_path, 9)
    build(items, tmp_path / "f.html", tmp_path / "k.json")

    key = json.loads((tmp_path / "k.json").read_text())
    html = (tmp_path / "f.html").read_text()

    assert [row["n"] for row in key] == list(range(1, 10))
    for row in key:
        assert f'<span class="p">{row["prompt"]}</span>' in html
        marker = f'<span class="n">{row["n"]} / 9</span><span class="p">{row["prompt"]}</span>'
        assert marker in html


def test_the_order_is_shuffled(tmp_path: Path) -> None:
    """Left in order, the three sources would alternate and the pattern would
    be visible after a few cards."""
    items = _items(tmp_path, 12)
    build(items, tmp_path / "f.html", tmp_path / "k.json")

    key = json.loads((tmp_path / "k.json").read_text())

    assert [row["prompt"] for row in key] != [it.prompt for it in items]


def test_the_same_seed_gives_the_same_order(tmp_path: Path) -> None:
    items = _items(tmp_path, 9)
    build(items, tmp_path / "a.html", tmp_path / "a.json", seed=3)
    build(items, tmp_path / "b.html", tmp_path / "b.json", seed=3)

    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()


def test_every_item_gets_every_question(tmp_path: Path) -> None:
    build(_items(tmp_path, 4), tmp_path / "f.html", tmp_path / "k.json")

    html = (tmp_path / "f.html").read_text()
    for key, _ in QUESTIONS:
        assert html.count(f'data-q="{key}"') == 4


def test_the_images_are_carried_in_the_file(tmp_path: Path) -> None:
    """One file, nothing behind it: the person rating should not need a server,
    a login, or the folder the images came from."""
    build(_items(tmp_path, 3), tmp_path / "f.html", tmp_path / "k.json")

    assert (tmp_path / "f.html").read_text().count("data:image/jpeg;base64,") == 3


def test_an_empty_form_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build([], tmp_path / "f.html", tmp_path / "k.json")


# --- what the rater reads ---------------------------------------------------


def test_the_card_shows_the_prompt_in_turkish(tmp_path: Path) -> None:
    """The model is prompted in English and always will be. But the content
    question asks whether what the prompt describes is in the picture, and
    someone who cannot read it cannot answer."""
    from form import TURKISH

    path = tmp_path / "00.png"
    Image.new("RGB", (64, 64)).save(path)
    english = "a knight standing in a doorway"
    item = Item(path=path, source="base", prompt=english)
    build([item], tmp_path / "f.html", tmp_path / "k.json")

    html = (tmp_path / "f.html").read_text()
    assert TURKISH[english] in html
    assert english not in html


def test_the_key_keeps_the_english(tmp_path: Path) -> None:
    """Scoring joins on the prompt, and the rest of the project speaks English."""
    path = tmp_path / "00.png"
    Image.new("RGB", (64, 64)).save(path)
    english = "a fox in a forest clearing"
    item = Item(path=path, source="base", prompt=english)
    build([item], tmp_path / "f.html", tmp_path / "k.json")

    assert json.loads((tmp_path / "k.json").read_text())[0]["prompt"] == english


def test_every_measurement_prompt_has_a_translation() -> None:
    """A missing one falls back to English, which is readable but leaves the
    friend guessing on that card."""
    from form import TURKISH
    from stylelora.generate import PROMPTS

    for prompt in PROMPTS:
        assert prompt in TURKISH, prompt


def test_an_unknown_prompt_falls_back_rather_than_blanks(tmp_path: Path) -> None:
    from form import turkish

    assert turkish("a thing nobody translated") == "a thing nobody translated"


# --- the button says what it does -------------------------------------------


def test_a_posting_form_offers_to_send_not_to_download(tmp_path: Path) -> None:
    """The label was hardcoded to the download wording while the page posted.
    A button that promises a file and quietly uploads instead is a page nobody
    trusts twice."""
    path = tmp_path / "00.png"
    Image.new("RGB", (64, 64)).save(path)
    item = Item(path=path, source="base", prompt="a fox in a forest clearing")

    build([item], tmp_path / "f.html", tmp_path / "k.json", submit_url="/api/submit")

    html = (tmp_path / "f.html").read_text()
    assert 'SUBMIT = "/api/submit"' in html
    assert '"Gönder" : "Sonuçları indir"' in html


def test_a_form_with_nowhere_to_post_still_offers_the_file(tmp_path: Path) -> None:
    path = tmp_path / "00.png"
    Image.new("RGB", (64, 64)).save(path)
    item = Item(path=path, source="base", prompt="a fox in a forest clearing")

    build([item], tmp_path / "f.html", tmp_path / "k.json")

    assert 'SUBMIT = ""' in (tmp_path / "f.html").read_text()


def test_the_submit_constant_exists_before_anything_reads_it(tmp_path: Path) -> None:
    """refresh() branches on SUBMIT and is defined above where it used to be
    declared; only the order of the final call kept that working."""
    path = tmp_path / "00.png"
    Image.new("RGB", (64, 64)).save(path)
    item = Item(path=path, source="base", prompt="a fox in a forest clearing")

    build([item], tmp_path / "f.html", tmp_path / "k.json", submit_url="/api/submit")

    html = (tmp_path / "f.html").read_text()
    assert html.index("SUBMIT = ") < html.index("function refresh()")


def test_a_partly_finished_form_can_still_be_sent(tmp_path: Path) -> None:
    """Requiring every answer meant someone who did forty of forty-eight and
    stopped gave us nothing. Forty answers are forty answers."""
    path = tmp_path / "00.png"
    Image.new("RGB", (64, 64)).save(path)
    item = Item(path=path, source="base", prompt="a fox in a forest clearing")

    build([item], tmp_path / "f.html", tmp_path / "k.json", submit_url="/api/submit")

    html = (tmp_path / "f.html").read_text()
    assert "save.disabled = done === 0;" in html
    assert "done === TOTAL" not in html.split("function refresh()")[1].split("}")[0]
