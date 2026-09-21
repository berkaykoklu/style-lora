"""The gate that runs before any training.

If the two styles do not pull apart in CLIP space, then CLIP cannot see the
difference between them -- and every number later in this project is measured
with CLIP. Training two LoRAs first and discovering that afterwards would mean
a curve nobody can trust and hours spent producing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stylelora.score import Vectors

# How far apart is far enough.
#
# Measured on the real pair: Baroque holds together at 0.694 and Art Nouveau at
# 0.656, while the two sit at 0.577 from each other -- a margin of 0.098. Half
# of that is the bar, so a pair has to be about twice as coherent as it is
# confusable before anything gets trained on it. Splitting one style in half
# produces a margin near zero, which is the case this number has to refuse.
MARGIN_FLOOR = 0.05


@dataclass(frozen=True)
class Separation:
    within_a: float
    within_b: float
    between: float
    margin: float
    separated: bool


def _mean_pairwise(vectors: Vectors) -> float:
    """Average similarity inside a set, excluding each vector with itself."""
    n = len(vectors)
    if n < 2:
        return 1.0
    sim = vectors @ vectors.T
    return float((sim.sum() - np.trace(sim)) / (n * n - n))


def measure(a: Vectors, b: Vectors) -> Separation:
    """How far apart two styles sit, against how tight each one is.

    Between-style distance alone says nothing: two styles can look far apart
    simply because neither is coherent. The margin is the comparison that
    matters.
    """
    within_a = _mean_pairwise(a)
    within_b = _mean_pairwise(b)
    between = float((a @ b.T).mean())
    margin = (within_a + within_b) / 2 - between
    return Separation(within_a, within_b, between, margin, margin > MARGIN_FLOOR)
