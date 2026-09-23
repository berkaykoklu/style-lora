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
# First measured on 20 training images per style: within 0.694 and 0.656,
# between 0.577, margin 0.098. Half of that became the bar.
#
# Re-measured on 30 held-out images per style, which is what the centres are
# actually built from: within 0.683 and 0.665, between 0.613, margin **0.061**.
# The styles are half again as close as the first measurement said. Twenty
# images each happened to be more distinct than the styles are; the gap did not
# move, the estimate of it did.
#
# The floor is left at 0.05 because its job is unchanged -- splitting one style
# in half produces a margin near zero and that is the case it has to refuse --
# but 0.061 clears it narrowly, and this pair therefore sits near the limit of
# what CLIP can resolve. Every effect measured downstream is bounded by it: no
# adapter can hold the two styles further apart than the styles themselves are.
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
