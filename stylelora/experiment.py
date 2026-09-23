"""One cell of the grid, and how to read a column of them.

The project's question is a trade-off, so no single number answers it. Every
cell carries three measurements: how close the images sit to the style they were
trained on, how close they sit to the *other* style, and how much of the prompt
survived.

The first two only mean something together. Style scores rise with strength for
reasons that have nothing to do with style -- a warmer, softer image scores
better against any painting -- and the control transforms in `control.py` earn
most of that rise for free. The difference between the two centres is what tone
cannot fake, so that difference is the measurement and the raw scores are
context.

Every number is kept per seed rather than averaged on the way in. Two reasons.
A mean with no spread beside it cannot say whether it is a result. And the base
model is not neutral -- it leans towards one of the two styles before any
adapter is loaded -- so the comparison that matters is against the base model on
the *same seed*, which needs both sets of per-seed numbers to line up.

Nothing here imports a model. The generator is passed in, so the arithmetic can
be tested without a GPU and a failure in measurement never looks like a failure
in training.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from stylelora.score import Vectors, embed_images, embed_text, prompt_score, style_score

Generate = Callable[[Path | None, float, tuple[str, ...], int], list[Image.Image]]

# Four generation seeds per cell. The same adapter at the same strength gives a
# different answer on every seed, and an effect smaller than that wobble is not
# an effect; one seed cannot tell the two apart.
SEEDS: tuple[int, ...] = (0, 1, 2, 3)

# Zero first: the base model is the reference every row is read against, not an
# assumed floor.
STRENGTHS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# How many standard deviations an effect has to clear before it is called real.
# Three, against the spread of single measurements rather than the spread of
# their mean -- the strict version of the test, chosen because the effect being
# measured here is small enough that the generous version would flatter it.
SIGMAS = 3.0


@dataclass(frozen=True)
class Cell:
    """One (adapter, strength) pair, measured on every seed."""

    strength: float
    seeds: tuple[int, ...]
    own_per_seed: tuple[float, ...]
    other_per_seed: tuple[float, ...]
    content_per_seed: tuple[float, ...]

    @property
    def gaps(self) -> npt.NDArray[np.float64]:
        """Own style minus the other one, per seed. Positive is correct."""
        return np.asarray(self.own_per_seed, dtype=np.float64) - np.asarray(
            self.other_per_seed, dtype=np.float64
        )

    @property
    def gap(self) -> float:
        return float(self.gaps.mean())

    @property
    def own(self) -> float:
        return float(np.mean(self.own_per_seed))

    @property
    def other(self) -> float:
        return float(np.mean(self.other_per_seed))

    @property
    def content(self) -> float:
        return float(np.mean(self.content_per_seed))

    @property
    def content_spread(self) -> float:
        return float(np.std(self.content_per_seed))


@dataclass(frozen=True)
class Lift:
    """What the adapter added, measured against the base model seed by seed."""

    mean: float
    spread: float

    @property
    def real(self) -> bool:
        return self.mean > SIGMAS * self.spread


def measure(
    lora: Path | None,
    strength: float,
    own_centre: Vectors,
    other_centre: Vectors,
    prompts: tuple[str, ...],
    generate: Generate,
    seeds: Sequence[int] = SEEDS,
) -> Cell:
    """Generate the prompt set at one strength, on every seed, and score it."""
    if not seeds:
        raise ValueError("a cell needs at least one seed")
    text = embed_text(list(prompts))

    own: list[float] = []
    other: list[float] = []
    content: list[float] = []
    for seed in seeds:
        vectors = embed_images(generate(lora, strength, prompts, seed))
        own.append(float(style_score(vectors, own_centre).mean()))
        other.append(float(style_score(vectors, other_centre).mean()))
        content.append(float(prompt_score(vectors, text).mean()))

    return Cell(
        strength=strength,
        seeds=tuple(seeds),
        own_per_seed=tuple(own),
        other_per_seed=tuple(other),
        content_per_seed=tuple(content),
    )


def lift(cell: Cell, base: Cell) -> Lift:
    """How much nearer its own style the adapter got, paired by seed.

    Paired, not two independent means. The base model does not sit halfway
    between the two styles -- it leans towards one of them -- so the raw gap
    starts off with a bias that belongs to the base model rather than to the
    adapter. Subtracting the base model's gap on the *same* seed removes both
    that bias and the seed's own noise, which is the largest term in the
    measurement.
    """
    if cell.seeds != base.seeds:
        raise ValueError(
            f"paired against different seeds: {cell.seeds} against {base.seeds}"
        )
    if base.strength != 0.0:
        raise ValueError(f"the reference has to be the base model, got strength {base.strength}")
    difference = cell.gaps - base.gaps
    return Lift(mean=float(difference.mean()), spread=float(difference.std()))


def knee(cells: Sequence[Cell], tolerance: float) -> float | None:
    """The strongest setting that still draws what was asked for.

    Walks up from the base model and stops at the first strength whose prompt
    score has fallen further than `tolerance`. Stops rather than takes the
    maximum: a curve that breaks at 0.4 and reads well again at 0.8 has not
    come back, it has wandered somewhere else that happens to score alike, and
    calling 0.8 usable would recommend it.

    Returns None when even the weakest setting breaks, which is a result and
    not an error -- it says this adapter has no usable range.
    """
    ordered = sorted(cells, key=lambda c: c.strength)
    if not ordered or ordered[0].strength != 0.0:
        raise ValueError("knee needs the base model (strength 0) as its reference")
    if tolerance < 0:
        raise ValueError(f"tolerance is a distance, got {tolerance}")

    floor = ordered[0].content - tolerance
    found: float | None = None
    for cell in ordered[1:]:
        if cell.content < floor:
            break
        found = cell.strength
    return found


def content_tolerance(base: Cell, sigmas: float = SIGMAS) -> float:
    """How far the prompt score may drop before the drop means anything.

    Derived from the base model's own spread across seeds rather than picked.
    The same prompt on two seeds is two different pictures and they do not score
    alike; a fixed threshold would either call that noise a failure or hide a
    real failure behind it.
    """
    return sigmas * base.content_spread
