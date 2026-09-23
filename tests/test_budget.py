from stylelora.budget import RESERVE_GB, measure


def test_the_cap_leaves_the_reserve_behind() -> None:
    """Whatever is free, the operating system keeps its share."""
    small = measure(reserve_gb=0.5)
    large = measure(reserve_gb=4.0)

    assert large.allowed_gb < small.allowed_gb


def test_the_fraction_is_of_the_whole_machine() -> None:
    budget = measure()

    assert 0 < budget.fraction <= 1.0
    assert abs(budget.allowed_gb / budget.total_gb - budget.fraction) < 1e-6


def test_an_impossible_reserve_still_leaves_something_to_run_with() -> None:
    """Asking for more headroom than the machine has must not produce a
    negative cap, which would read as unlimited."""
    budget = measure(reserve_gb=10_000)

    assert budget.allowed_gb >= 1.0


def test_free_memory_is_not_read_as_free_pages_alone() -> None:
    """macOS parks most of RAM in reclaimable buckets; counting only free
    pages would under-report by gigabytes and cap the run to nothing."""
    budget = measure(reserve_gb=0.0)

    assert budget.available_gb > 0.5
    assert budget.available_gb <= budget.total_gb


def test_the_default_reserve_is_documented_and_modest() -> None:
    assert 1.0 <= RESERVE_GB <= 6.0
