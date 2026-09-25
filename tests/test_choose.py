from stylelora.data import ARTIST_SHARE, GENRE_NAMES, Row, choose

SKETCH = GENRE_NAMES.index("sketch_and_study")
PORTRAIT = GENRE_NAMES.index("portrait")


def _rows(artists: list[int], genre: int = PORTRAIT) -> list[Row]:
    return [{"artist": a, "genre": genre, "image": {"src": f"{i}"}} for i, a in enumerate(artists)]


def test_sketches_are_dropped() -> None:
    """Pen studies on cream paper teach line and paper, not a painted style --
    and half of one style's first sample turned out to be etchings."""
    rows = _rows([0] * 10, genre=SKETCH) + _rows(list(range(10, 20)))

    chosen = choose(rows, count=8)

    assert all(row["genre"] != SKETCH for row in chosen)


def test_no_artist_takes_more_than_their_share() -> None:
    """Shuffling fixes where in the list we look, not who fills it. One painter
    who is most of a style would still be most of the sample."""
    rows = _rows([0] * 100 + list(range(1, 40)))

    chosen = choose(rows, count=32)

    cap = 32 // ARTIST_SHARE
    counts: dict[int, int] = {}
    for row in chosen:
        counts[row["artist"]] = counts.get(row["artist"], 0) + 1
    assert max(counts.values()) <= cap


def test_a_smaller_set_is_a_subset_of_a_larger_one() -> None:
    """The data axis compares 20 images against 100. They have to be the same
    20, or the two runs differ by which paintings as well as by how many."""
    rows = _rows(list(range(200)))

    small = {row["image"]["src"] for row in choose(rows, count=20)}
    large = {row["image"]["src"] for row in choose(rows, count=100)}

    assert small <= large


def test_the_same_seed_gives_the_same_set() -> None:
    rows = _rows(list(range(200)))

    assert choose(rows, count=20) == choose(rows, count=20)


def test_a_different_seed_gives_a_different_set() -> None:
    rows = _rows(list(range(200)))

    first = {row["image"]["src"] for row in choose(rows, count=20, seed=0)}
    second = {row["image"]["src"] for row in choose(rows, count=20, seed=1)}

    assert first != second


def test_a_style_by_one_painter_comes_back_short_rather_than_concentrated() -> None:
    """Returning a full set of one artist would be the original bug wearing a
    cap. Coming back short is what makes fetch refuse and say why."""
    rows = _rows([7] * 100)

    chosen = choose(rows, count=40)

    assert len(chosen) == 40 // ARTIST_SHARE


def test_asking_for_more_than_exists_returns_what_exists() -> None:
    rows = _rows(list(range(5)))

    assert len(choose(rows, count=50)) == 5
