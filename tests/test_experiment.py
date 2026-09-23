from pathlib import Path

import pytest
from PIL import Image

from stylelora.experiment import (
    SIGMAS,
    Cell,
    Lift,
    content_tolerance,
    knee,
    lift,
    measure,
    operating_point,
    separation,
)
from stylelora.score import embed_images, style_centre

Images = list[Image.Image]
Prompts = tuple[str, ...]


def _solid(colour: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (224, 224), colour)


def _cell(strength: float, content: float) -> Cell:
    return Cell(
        strength=strength,
        seeds=(0,),
        own_per_seed=(0.6,),
        other_per_seed=(0.6,),
        content_per_seed=(content,),
    )


def _gapped(strength: float, gaps: tuple[float, ...]) -> Cell:
    """A cell with a chosen gap on each seed and nothing else of interest."""
    return Cell(
        strength=strength,
        seeds=tuple(range(len(gaps))),
        own_per_seed=tuple(0.6 + g for g in gaps),
        other_per_seed=tuple(0.6 for _ in gaps),
        content_per_seed=tuple(0.3 for _ in gaps),
    )


# --- the knee ---------------------------------------------------------------


def test_a_curve_that_never_breaks_has_its_knee_at_the_top() -> None:
    cells = [_cell(s, 0.30) for s in (0.0, 0.2, 0.4, 0.6, 1.0)]

    assert knee(cells, tolerance=0.01) == 1.0


def test_the_knee_is_the_last_strength_before_the_drop() -> None:
    cells = [_cell(0.0, 0.30), _cell(0.2, 0.30), _cell(0.4, 0.20), _cell(0.6, 0.18)]

    assert knee(cells, tolerance=0.01) == 0.2


def test_a_recovery_after_the_break_is_not_believed() -> None:
    """A curve that breaks at 0.4 and reads well again at 0.8 has not come back
    -- it has wandered somewhere else that happens to score alike. Taking the
    maximum would recommend 0.8; walking up stops at the break."""
    cells = [_cell(0.0, 0.30), _cell(0.2, 0.30), _cell(0.4, 0.10), _cell(0.8, 0.30)]

    assert knee(cells, tolerance=0.01) == 0.2


def test_an_adapter_with_no_usable_range_has_no_knee() -> None:
    cells = [_cell(0.0, 0.30), _cell(0.2, 0.10), _cell(0.4, 0.05)]

    assert knee(cells, tolerance=0.01) is None


def test_a_strength_that_improves_the_prompt_score_is_still_usable() -> None:
    """The floor is 'not much worse', not 'no better' -- an adapter is allowed
    to help the prompt score and sometimes does."""
    cells = [_cell(0.0, 0.30), _cell(0.2, 0.34), _cell(0.4, 0.33)]

    assert knee(cells, tolerance=0.01) == 0.4


def test_the_knee_refuses_a_column_with_no_base_model() -> None:
    """Without strength 0 there is nothing to call a drop, and the tolerance
    would be measured against whichever row happened to come first."""
    with pytest.raises(ValueError):
        knee([_cell(0.2, 0.30), _cell(0.4, 0.29)], tolerance=0.01)


def test_the_knee_refuses_a_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        knee([_cell(0.0, 0.30)], tolerance=-0.01)


def test_the_column_does_not_have_to_arrive_in_order() -> None:
    cells = [_cell(0.4, 0.20), _cell(0.0, 0.30), _cell(0.2, 0.30)]

    assert knee(cells, tolerance=0.01) == 0.2


# --- the thresholds ---------------------------------------------------------


def test_the_tolerance_comes_from_the_measured_spread() -> None:
    base = Cell(0.0, (0, 1), (0.6, 0.6), (0.6, 0.6), (0.30, 0.31))

    assert content_tolerance(base) == pytest.approx(SIGMAS * 0.005)


def test_a_flat_base_model_gives_a_tolerance_of_zero() -> None:
    """Not a bug to hide: if every seed scores alike then any drop is real."""
    base = Cell(0.0, (0, 1), (0.6, 0.6), (0.6, 0.6), (0.30, 0.30))

    assert content_tolerance(base) == 0.0


# --- the lift ---------------------------------------------------------------


def test_the_lift_is_measured_against_the_base_model_not_against_zero() -> None:
    """The base model leans towards one style before any adapter is loaded. An
    adapter that left the gap exactly where it found it added nothing, however
    large that gap happens to be."""
    base = _gapped(0.0, (-0.03, -0.03, -0.03, -0.03))
    same = _gapped(0.6, (-0.03, -0.03, -0.03, -0.03))

    assert lift(same, base).mean == pytest.approx(0.0)
    assert not lift(same, base).real


def test_a_consistent_improvement_is_real() -> None:
    """The measured shape: a small gain that repeats on every seed."""
    base = _gapped(0.0, (-0.032, -0.030, -0.035, -0.031))
    adapter = _gapped(0.6, (-0.016, -0.013, -0.018, -0.014))

    result = lift(adapter, base)

    assert result.mean == pytest.approx(0.01675, abs=1e-4)
    assert result.real


def test_a_gain_that_only_happens_on_one_seed_is_not_real() -> None:
    base = _gapped(0.0, (-0.03, -0.03, -0.03, -0.03))
    lucky = _gapped(0.6, (0.05, -0.03, -0.03, -0.03))

    assert lift(lucky, base).mean > 0
    assert not lift(lucky, base).real


def test_pairing_against_different_seeds_is_refused() -> None:
    """Pairing only cancels the seed's noise while the two rows used the same
    seeds. Silently zipping mismatched lists would add noise instead."""
    base = Cell(0.0, (0, 1), (0.6, 0.6), (0.6, 0.6), (0.3, 0.3))
    adapter = Cell(0.6, (0, 2), (0.7, 0.7), (0.6, 0.6), (0.3, 0.3))

    with pytest.raises(ValueError):
        lift(adapter, base)


def test_a_reference_that_is_not_the_base_model_is_refused() -> None:
    with pytest.raises(ValueError):
        lift(_gapped(0.6, (0.01,)), _gapped(0.4, (0.0,)))


# --- the cell ---------------------------------------------------------------


def test_a_cell_lands_nearer_the_centre_its_images_match() -> None:
    red = [_solid((200, 30, 30))] * 3
    red_centre = style_centre(embed_images(red))
    blue_centre = style_centre(embed_images([_solid((30, 30, 200))] * 3))

    def draws_red(lora: Path | None, s: float, prompts: Prompts, seed: int) -> Images:
        return red

    cell = measure(
        None, 1.0, red_centre, blue_centre, ("a red thing",) * 3, draws_red, seeds=(0, 1)
    )

    assert cell.gap > 0
    assert cell.own > cell.other


def test_swapping_the_centres_flips_the_sign() -> None:
    """The gap is a direction, not a magnitude: measured against the wrong
    centre it has to come out negative, or the sign carries no information."""
    red = [_solid((200, 30, 30))] * 2
    red_centre = style_centre(embed_images(red))
    blue_centre = style_centre(embed_images([_solid((30, 30, 200))] * 2))

    def draws_red(lora: Path | None, s: float, prompts: Prompts, seed: int) -> Images:
        return red

    right = measure(None, 1.0, red_centre, blue_centre, ("a", "b"), draws_red, seeds=(0,))
    wrong = measure(None, 1.0, blue_centre, red_centre, ("a", "b"), draws_red, seeds=(0,))

    assert right.gap == pytest.approx(-wrong.gap, abs=1e-6)


def test_every_seed_is_generated_exactly_once() -> None:
    """A cell that quietly reused one seed would report a spread of zero and
    call anything real."""
    seen: list[int] = []

    def record(lora: Path | None, s: float, prompts: Prompts, seed: int) -> Images:
        seen.append(seed)
        return [_solid((90, 90, 90)) for _ in prompts]

    centre = style_centre(embed_images([_solid((90, 90, 90))]))
    cell = measure(None, 0.5, centre, centre, ("a", "b"), record, seeds=(0, 1, 2))

    assert seen == [0, 1, 2]
    assert cell.seeds == (0, 1, 2)


def test_a_cell_with_no_seeds_is_refused() -> None:
    centre = style_centre(embed_images([_solid((90, 90, 90))]))

    with pytest.raises(ValueError):
        measure(None, 0.5, centre, centre, ("a",), lambda *_: [], seeds=())


# --- separation and the operating point -------------------------------------


def test_two_adapters_producing_the_same_images_are_not_separated() -> None:
    """The arithmetic the sum exists for: measured from opposite sides, one
    lift is the exact negative of the other, however large each looks."""
    mirrored = separation(Lift(mean=-0.042, spread=0.01), Lift(mean=+0.043, spread=0.01))

    assert mirrored.mean == pytest.approx(0.001)
    assert not mirrored.real


def test_two_adapters_moving_to_their_own_styles_are_separated() -> None:
    both = separation(Lift(mean=+0.014, spread=0.004), Lift(mean=+0.005, spread=0.004))

    assert both.mean == pytest.approx(0.019)
    assert both.real


def test_the_operating_point_is_the_best_separation_the_content_survives() -> None:
    measured = [
        (0.2, Lift(0.009, 0.003)),
        (0.4, Lift(0.019, 0.003)),
        (0.6, Lift(0.023, 0.003)),
        (1.0, Lift(0.001, 0.003)),
    ]

    assert operating_point(measured, content_knee=0.6) == 0.6


def test_a_peak_the_content_cannot_reach_is_not_offered() -> None:
    """The measured Art Nouveau shape: separation still climbing at 1.0 while
    the prompt score has already broken."""
    measured = [(0.4, Lift(0.019, 0.003)), (0.8, Lift(0.030, 0.003)), (1.0, Lift(0.040, 0.003))]

    assert operating_point(measured, content_knee=0.8) == 0.8


def test_a_separation_that_decays_before_the_knee_moves_the_point_down() -> None:
    """The measured Baroque shape: the knee allows 0.6, but the styles are
    furthest apart at 0.4 and quoting 0.6 would cost quality for no style."""
    measured = [(0.2, Lift(0.009, 0.003)), (0.4, Lift(0.019, 0.003)), (0.6, Lift(0.002, 0.003))]

    assert operating_point(measured, content_knee=0.6) == 0.4


def test_an_adapter_whose_content_never_survives_has_no_operating_point() -> None:
    assert operating_point([(0.2, Lift(0.009, 0.003))], content_knee=None) is None
