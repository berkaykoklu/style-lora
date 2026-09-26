import pytest

from blind import STRENGTHS, tally
from stylelora.data import STYLES


def test_an_answer_means_the_strength_not_the_side() -> None:
    """The shuffle puts each strength left or right at random, so "L" means
    different things on different rows. Counting sides would measure which
    half of the page the eye prefers."""
    order = [[STYLES[0], 0, 0.4], [STYLES[0], 1, 0.6]]

    both_left = tally(order, "LL")

    assert both_left[STYLES[0]][0.4] == 1
    assert both_left[STYLES[0]][0.6] == 1


def test_answering_the_same_strength_every_time_is_counted_as_such() -> None:
    order = [[STYLES[0], 0, 0.4], [STYLES[0], 1, 0.6], [STYLES[0], 2, 0.4]]

    picked = tally(order, "LRL")  # 0.4, 0.4, 0.4

    assert picked[STYLES[0]][0.4] == 3
    assert picked[STYLES[0]][0.6] == 0


def test_the_two_styles_are_counted_apart() -> None:
    order = [[STYLES[0], 0, 0.4], [STYLES[1], 0, 0.4]]

    picked = tally(order, "LR")

    assert picked[STYLES[0]][0.4] == 1
    assert picked[STYLES[1]][0.6] == 1


def test_every_pair_lands_somewhere() -> None:
    order = [[STYLES[0], i, STRENGTHS[i % 2]] for i in range(10)]

    picked = tally(order, "LRLRLRLRLR")

    assert sum(picked[STYLES[0]].values()) == 10


def test_a_short_answer_string_is_refused() -> None:
    with pytest.raises(ValueError):
        tally([[STYLES[0], 0, 0.4], [STYLES[0], 1, 0.6]], "L")


def test_a_stray_letter_is_refused() -> None:
    with pytest.raises(ValueError):
        tally([[STYLES[0], 0, 0.4]], "X")
