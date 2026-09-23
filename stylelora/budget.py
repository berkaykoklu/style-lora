"""A ceiling on how much memory this process may take.

Watching memory and killing the run when it gets low does not work on a
unified-memory Mac: by the time free memory reads low the machine is already
swapping, and the swapping is the freeze. The fix is to refuse the allocation
in the first place -- PyTorch then raises an error instead of asking the system
for memory it does not have.

Call `apply()` before anything touches the GPU.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

# Left for the operating system and whatever the user already has open.
# Training is a guest on this machine, not its owner.
#
# Measured on the machine this was built for: 16 GB total, 8.2 GB already in
# use, 7.2 GB free. Five gigabytes of headroom left only 2.2 GB, which does not
# hold the model at all; two and a half leaves a working 4.7 GB and still keeps
# the system off swap.
RESERVE_GB = 2.5

GB = 1024**3


@dataclass(frozen=True)
class Budget:
    total_gb: float
    available_gb: float
    allowed_gb: float
    fraction: float


def _total_bytes() -> int:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    return int(out.stdout.strip())


def _available_bytes() -> int:
    """Memory the system could hand out without swapping.

    Free pages alone understate this badly: macOS parks most of RAM in the
    inactive and speculative buckets, which are reclaimable on demand.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 4096
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split()[-2])
        key, _, value = line.partition(":")
        digits = value.strip().rstrip(".")
        if digits.isdigit():
            counts[key.strip()] = int(digits)
    reclaimable = (
        counts.get("Pages free", 0)
        + counts.get("Pages inactive", 0)
        + counts.get("Pages speculative", 0)
    )
    return reclaimable * page


def measure(reserve_gb: float = RESERVE_GB) -> Budget:
    total = _total_bytes()
    available = _available_bytes()
    allowed = max(available - int(reserve_gb * GB), GB)
    return Budget(
        total_gb=total / GB,
        available_gb=available / GB,
        allowed_gb=allowed / GB,
        fraction=min(allowed / total, 1.0),
    )


def apply(reserve_gb: float = RESERVE_GB) -> Budget:
    """Cap this process, and say what the cap is.

    Both levers are set because they fail differently: the environment variable
    is read when the MPS allocator starts, and the API call covers a process
    where that has already happened.
    """
    budget = measure(reserve_gb)

    # PyTorch keeps two thresholds and requires low <= high. The defaults are
    # 1.4 and 1.7, so lowering only the high one is rejected outright -- the
    # first version of this did exactly that and silently kept the default cap.
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{budget.fraction:.3f}"

    try:
        import torch

        setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
        if setter is not None and torch.backends.mps.is_available():
            setter(budget.fraction)
    except Exception as exc:  # noqa: BLE001 -- a missing lever is not fatal
        print(f"  (memory fraction API unavailable: {exc})", flush=True)

    print(
        f"  memory: {budget.total_gb:.1f} GB total, {budget.available_gb:.1f} GB free, "
        f"capped at {budget.allowed_gb:.1f} GB ({budget.fraction:.0%})",
        flush=True,
    )
    return budget


def main() -> None:
    apply()


if __name__ == "__main__":
    main()
