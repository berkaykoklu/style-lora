"""Reading a downloaded results bundle, without a GPU.

Colab trains and measures; this reports. The bundle holds every measured cell
seed by seed, so all of it recomputes on a laptop -- which is the point, since
the session that produced it is usually gone by the time anyone reads it.

    python report.py style-lora-results
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from stylelora.experiment import SIGMAS, Cell, Lift, separation_from_cells

STRENGTHS = (0.2, 0.4, 0.6, 0.8, 1.0)


def load(bundle: Path) -> dict[tuple[str, float], dict[str, Cell]]:
    """Cells keyed by (run, strength), each holding one cell per style."""
    raw = json.loads((bundle / "cells.json").read_text())
    table: dict[tuple[str, float], dict[str, Cell]] = defaultdict(dict)
    for key, values in raw.items():
        config, strength = key.split("@")
        run, style = config.rsplit("/", 1)
        table[(run, float(strength))][style] = Cell(
            strength=values["strength"],
            seeds=tuple(values["seeds"]),
            own_per_seed=tuple(values["own_per_seed"]),
            other_per_seed=tuple(values["other_per_seed"]),
            content_per_seed=tuple(values["content_per_seed"]),
        )
    return table


def separations(table: dict[tuple[str, float], dict[str, Cell]]) -> dict[tuple[str, float], Lift]:
    out: dict[tuple[str, float], Lift] = {}
    for (run, strength), pair in table.items():
        if len(pair) != 2:
            continue
        first, second = (pair[name] for name in sorted(pair))
        out[(run, strength)] = separation_from_cells(first, second)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?", default=Path("style-lora-results"))
    args = parser.parse_args()

    table = load(args.bundle)
    found = separations(table)
    runs = sorted({run for run, _ in found}, key=lambda r: ("t2500" in r, r))

    print("SEPARATION -- how far apart the two adapters are, base model cancelled out")
    print(f"a star marks a value clearing {SIGMAS:.0f} sigma of its own spread across seeds\n")
    print(f"{'run':22}" + "".join(f"{s:>16.1f}" for s in STRENGTHS))
    print("-" * (22 + 16 * len(STRENGTHS)))
    for run in runs:
        row = ""
        for strength in STRENGTHS:
            result = found.get((run, strength))
            if result is None:
                row += f"{'-':>16}"
                continue
            row += f"{result.mean:>+11.3f}±{result.spread:.3f}{'*' if result.real else ' '}"
        print(f"{run:22}{row}")

    print("\nBest separation per run, and the strength it happens at:")
    for run in runs:
        best = max(
            ((found[(run, s)], s) for s in STRENGTHS if (run, s) in found),
            key=lambda pair: pair[0].mean,
        )
        result, strength = best
        print(
            f"  {run:22} @{strength}  {result.mean:+.3f} ± {result.spread:.3f}"
            f"   {'real' if result.real else 'inside the noise'}"
        )

    summary = args.bundle / "results.json"
    if summary.exists():
        gate = json.loads(summary.read_text())["gate"]
        print(
            f"\nCeiling: the styles themselves sit {gate['margin']:.3f} apart on held-out "
            f"images.\nNo adapter can hold them further apart than that."
        )


if __name__ == "__main__":
    main()
