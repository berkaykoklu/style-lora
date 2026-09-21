import numpy as np

from stylelora.separation import measure


def _cluster(direction: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = direction + spread * rng.normal(size=(n, len(direction)))
    return (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)


def test_two_tight_far_apart_clusters_separate() -> None:
    a = _cluster(np.eye(512)[0], 20, 0.05, seed=0)
    b = _cluster(np.eye(512)[1], 20, 0.05, seed=1)

    result = measure(a, b)

    assert result.separated
    assert result.margin > 0


def test_one_cluster_split_in_half_does_not_separate() -> None:
    """The honest negative: halves of the same thing are not two styles."""
    whole = _cluster(np.eye(512)[0], 40, 0.05, seed=2)

    result = measure(whole[:20], whole[20:])

    assert not result.separated


def test_the_margin_is_within_minus_between() -> None:
    a = _cluster(np.eye(512)[0], 12, 0.08, seed=3)
    b = _cluster(np.eye(512)[1], 12, 0.08, seed=4)

    result = measure(a, b)
    expected = (result.within_a + result.within_b) / 2 - result.between

    assert abs(result.margin - expected) < 1e-6


def test_a_looser_style_lowers_its_own_within_score() -> None:
    tight = _cluster(np.eye(512)[0], 20, 0.02, seed=5)
    loose = _cluster(np.eye(512)[0], 20, 0.40, seed=6)

    assert measure(tight, tight).within_a > measure(loose, loose).within_a


def test_distance_alone_is_not_enough_to_pass() -> None:
    """Two scattered clouds can sit far apart and still not be two styles.

    Judging on between-style distance alone would pass this; the margin is
    what refuses it.
    """
    a = _cluster(np.eye(512)[0], 20, 1.2, seed=7)
    b = _cluster(np.eye(512)[1], 20, 1.2, seed=8)

    result = measure(a, b)

    assert result.between < result.within_a  # they are further apart than they are tight
    assert not result.separated  # and neither is tight enough for that to mean anything
