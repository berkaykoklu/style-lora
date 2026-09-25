"""The blind test, on a laptop.

Looking at a grid with the labels showing is reading the labels. This shows the
two strengths side by side in a random order and asks one question per pair, and
only tells you which was which after the answers are in.

Three steps, each resumable, because the first attempt at this died with a
dropped connection and took the answer key with it:

    python blind.py draw          # generate and save every image, plus the key
    python blind.py sheet         # write blind.html, open it, write answers down
    python blind.py score LRRL... # 40 letters, in order

Nothing here needs the training data or the style centres: the metric's answer
is already measured and lives in the results bundle.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from stylelora import budget
from stylelora.data import STYLES
from stylelora.generate import PROMPTS, generate
from stylelora.train import WEIGHTS_NAME

# Viewing only. These never enter a score -- adding them to PROMPTS would
# invalidate every number already measured. Same rule as always: subjects, never
# style words, chosen so the two styles have somewhere different to go.
EXTRA = (
    "a woman with long flowing hair",
    "a vase of lilies on a wooden table",
    "an angel with outstretched wings",
    "a man in a heavy cloak holding a candle",
    "a peacock beside a garden wall",
    "a group of figures around a table at night",
    "a young woman standing among tall flowers",
    "an old man with a beard reading by lamplight",
    "a dancer with draped cloth in motion",
    "a window with climbing vines",
)
VIEW = PROMPTS[:10] + EXTRA

STRENGTHS = (0.4, 0.6)
RUN = "n20_r8_t500_s0"

# What the measurement says, so the reveal compares against something recorded
# rather than something remembered.
METRIC_PREFERS = {"Baroque": 0.4, "Art_Nouveau": 0.6}

# The whole pipeline in float32, plus room to work. Measured against a 16 GB
# machine where the cap refused cleanly rather than swapping -- which is the
# behaviour worth keeping, so this refuses earlier and says what to free.
NEEDED_GB = 6.0


def _folder(out: Path, style: str, strength: float) -> Path:
    return out / f"{style}_{strength}"


def draw(weights_root: Path, out: Path) -> None:
    plan = budget.measure()
    if plan.allowed_gb < NEEDED_GB:
        raise SystemExit(
            f"only {plan.allowed_gb:.1f} GB allowed ({plan.available_gb:.1f} GB free); "
            f"sd-turbo needs about {NEEDED_GB:.0f}. Close some apps until free memory "
            f"reads around {NEEDED_GB + 2.5:.1f} GB and run this again."
        )

    out.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        for strength in STRENGTHS:
            folder = _folder(out, style, strength)
            if len(sorted(folder.glob("*.png"))) == len(VIEW):
                print(f"  skip {folder.name}")
                continue
            folder.mkdir(parents=True, exist_ok=True)
            lora = weights_root / RUN / style / WEIGHTS_NAME
            if not lora.exists():
                raise SystemExit(f"{lora} is missing; unzip the results bundle first")
            for i, image in enumerate(generate(lora, strength, VIEW, seed=0)):
                image.save(folder / f"{i:02d}.png")
            print(f"  wrote {folder.name}")

    # The key goes to disk with the images. Held in memory it dies with the
    # process, and the images alone say nothing.
    key = out / "order.json"
    if not key.exists():
        rng = random.Random(0)
        order = [[s, i, rng.choice(STRENGTHS)] for s in STYLES for i in range(len(VIEW))]
        rng.shuffle(order)
        key.write_text(json.dumps(order))
    print(f"\n{len(json.loads(key.read_text()))} pairs ready in {out}")


def sheet(out: Path) -> Path:
    order = json.loads((out / "order.json").read_text())
    rows = []
    for n, (style, i, left) in enumerate(order, start=1):
        right = STRENGTHS[1] if left == STRENGTHS[0] else STRENGTHS[0]
        rows.append(
            f'<tr><td class="n">#{n}<br><span>{style}</span></td>'
            f'<td><img src="{_folder(out, style, left).name}/{i:02d}.png"><div>L</div></td>'
            f'<td><img src="{_folder(out, style, right).name}/{i:02d}.png"><div>R</div></td></tr>'
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>blind test</title>
<style>
 body {{ font: 15px system-ui; margin: 2rem auto; max-width: 900px; background:#fafafa }}
 td {{ padding: .4rem; text-align: center; vertical-align: top }}
 td.n {{ width: 6rem; text-align: right; color:#666 }}
 td.n span {{ font-size: 12px }}
 img {{ width: 380px; border-radius: 4px }}
 div {{ font-weight: 600; color:#444 }}
</style>
<h1>Which one is more of that style?</h1>
<p>One answer per row, L or R. You are not told which strength is which.
Write all {len(order)} down, then run <code>python blind.py score &lt;letters&gt;</code>.</p>
<table>{"".join(rows)}</table>"""
    path = out / "blind.html"
    path.write_text(html)
    return path


def tally(order: list[list], answers: str) -> dict[str, dict[float, int]]:
    """How often each strength was chosen, per style.

    The shuffle put each strength on the left or the right at random, so an
    answer of "L" means different things on different rows. Counting L and R
    instead of resolving them would measure which side of the page you prefer.
    """
    answers = answers.strip().upper()
    if len(answers) != len(order):
        raise ValueError(f"{len(order)} answers needed, {len(answers)} given")
    if set(answers) - {"L", "R"}:
        raise ValueError("answers are L or R only")

    picked: dict[str, dict[float, int]] = {s: dict.fromkeys(STRENGTHS, 0) for s in STYLES}
    for (style, _, left), answer in zip(order, answers, strict=True):
        right = STRENGTHS[1] if left == STRENGTHS[0] else STRENGTHS[0]
        picked[style][left if answer == "L" else right] += 1
    return picked


def score(out: Path, answers: str) -> None:
    order = json.loads((out / "order.json").read_text())
    try:
        picked = tally(order, answers)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"{'style':16}{'chose 0.4':>11}{'chose 0.6':>11}{'you':>7}{'metric':>9}   verdict")
    print("-" * 64)
    for style in STYLES:
        low, high = picked[style][0.4], picked[style][0.6]
        yours = 0.4 if low > high else 0.6
        # Fourteen of twenty is where a one-sided sign test falls below 0.06.
        # Below that the two strengths were not told apart, which is its own
        # result and not a failed one.
        if max(low, high) < 14:
            verdict = "no preference"
        else:
            verdict = "agrees" if yours == METRIC_PREFERS[style] else "DISAGREES"
        print(f"{style:16}{low:>11}{high:>11}{yours:>7}{METRIC_PREFERS[style]:>9}   {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("draw", "sheet", "score"))
    parser.add_argument("answers", nargs="?", default="")
    parser.add_argument("--weights", type=Path, default=Path("style-lora-results"))
    parser.add_argument("--out", type=Path, default=Path("blind"))
    args = parser.parse_args()

    if args.step == "draw":
        draw(args.weights, args.out)
    elif args.step == "sheet":
        print(f"open {sheet(args.out).resolve()}")
    else:
        score(args.out, args.answers)


if __name__ == "__main__":
    main()
