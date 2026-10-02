"""Scoring the blind rating form.

    python score_form.py results.json --exclude 6d9360f1-...

`results.json` is whatever `/api/results?token=...` returned. The answer key
says which card came from which adapter; the form never did.

Two questions come out of it. Does the adapter for a style get rated as that
style more than the base model does -- which is whether it worked at all. And
does the content rating hold up -- which is the one CLIP may be too blunt for:
its prompt score says the adapter costs nothing in content at any strength, and
someone looking at a knight can say whether that is true.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable
from math import comb
from pathlib import Path
from typing import Any

QUESTIONS = ("ukiyo", "baroque", "content")

Response = dict[str, Any]


def latest_per_rater(responses: Iterable[Response]) -> list[Response]:
    """One rating per person: the fullest thing they sent.

    The form posts on every press of Gönder, so one person part-way through
    leaves several records that are prefixes of each other. Counting them all
    would weight whoever pressed the button most.

    A record with no rater is dropped: those predate the id and are the test
    submissions made while wiring the endpoint up.
    """
    best: dict[str, Response] = {}
    for response in responses:
        rater = response.get("rater")
        if not rater:
            continue
        seen = best.get(rater)
        if seen is None or len(response.get("answers", {})) > len(seen.get("answers", {})):
            best[rater] = response
    return list(best.values())


def after(responses: Iterable[Response], cutoff: str) -> list[Response]:
    """Only ratings made against the key we still hold.

    Rebuilding the form reshuffles it, so a rating given against an older build
    is about a different picture than its card number now names. Nothing in the
    data says so -- the numbers still join, the table still prints -- which is
    why the cutoff is a date rather than a list of ids to remember.
    """
    return [r for r in responses if str(r.get("stored", "")) >= cutoff]


def ratings_by_source(
    responses: list[Response], key: list[dict[str, Any]]
) -> dict[str, dict[str, list[int]]]:
    """Every rating, filed under the adapter that drew the card it was about."""
    source_of = {str(row["n"]): row["source"] for row in key}
    collected: dict[str, dict[str, list[int]]] = {}
    for response in responses:
        for card, answers in response.get("answers", {}).items():
            source = source_of.get(str(card))
            if source is None:
                continue
            per_question = collected.setdefault(source, {q: [] for q in QUESTIONS})
            for question in QUESTIONS:
                value = answers.get(question)
                if isinstance(value, int):
                    per_question[question].append(value)
    return collected


def sign_test(differences: list[int]) -> float:
    """Two-sided probability of seeing a split this lopsided by chance alone.

    A sign test rather than a t-test: these are 1-5 ratings, where the distance
    from 1 to 2 is not the distance from 4 to 5, so only the direction of each
    change is worth trusting. Ties carry no direction and are dropped, which is
    what makes eight drops out of eight non-ties stronger evidence than eight
    drops out of sixteen cards.
    """
    up = sum(1 for d in differences if d > 0)
    down = sum(1 for d in differences if d < 0)
    n = up + down
    if n == 0:
        return 1.0
    extreme = max(up, down)
    tail = sum(comb(n, i) for i in range(extreme, n + 1))
    return min(1.0, 2 * tail / 2**n)


def paired(
    responses: list[Response], key: list[dict[str, Any]], source: str, question: str
) -> list[int]:
    """One difference per prompt: the adapter's rating minus the base model's.

    Paired on the prompt, because a prompt the model draws badly drags both down
    and a prompt it draws well lifts both. Comparing the two averages instead
    would carry that noise into the answer; the difference per prompt cancels it.
    """
    answers_of: dict[str, dict[str, dict[str, Any]]] = {}
    source_of = {str(row["n"]): (row["source"], row["prompt"]) for row in key}
    for response in responses:
        for card, answers in response.get("answers", {}).items():
            found = source_of.get(str(card))
            if found:
                answers_of.setdefault(found[1], {})[found[0]] = answers

    differences = []
    for sides in answers_of.values():
        here, base = sides.get(source), sides.get("base")
        if here is None or base is None:
            continue
        if isinstance(here.get(question), int) and isinstance(base.get(question), int):
            differences.append(here[question] - base[question])
    return differences


def _mean(values: list[int]) -> float:
    return statistics.fmean(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--key", type=Path, default=Path("key.json"))
    parser.add_argument("--exclude", nargs="*", default=[], help="rater ids to drop")
    parser.add_argument(
        "--since",
        default="",
        help="drop ratings stored before this ISO timestamp -- anything older was "
        "given against a previous build of the form and a different card order",
    )
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    responses = payload["responses"] if isinstance(payload, dict) else payload
    if args.since:
        before = len(responses)
        responses = after(responses, args.since)
        print(f"dropped {before - len(responses)} rating(s) given against an older form\n")
    kept = [r for r in latest_per_rater(responses) if r.get("rater") not in args.exclude]
    if not kept:
        raise SystemExit("no ratings left after dropping test submissions")

    key = json.loads(args.key.read_text())
    collected = ratings_by_source(kept, key)

    print(f"{len(kept)} rater(s), {sum(len(r.get('answers', {})) for r in kept)} cards rated\n")
    print(f"{'çizen':14}{'→ ukiyo-e':>12}{'→ barok':>11}{'→ içerik':>11}{'n':>6}")
    print("-" * 54)
    for source in ("base", "Ukiyo_e", "Baroque"):
        scores = collected.get(source)
        if not scores:
            continue
        counts = len(scores["ukiyo"])
        print(
            f"{source:14}{_mean(scores['ukiyo']):>12.2f}{_mean(scores['baroque']):>11.2f}"
            f"{_mean(scores['content']):>11.2f}{counts:>6}"
        )

    print("\nprompt bazında eşleştirilmiş, taban modele karşı:\n")
    print(f"{'karşılaştırma':38}{'ort':>7}{'yukarı':>8}{'eşit':>6}{'aşağı':>7}{'p':>9}")
    print("-" * 75)
    pairs = [
        ("Ukiyo_e", "ukiyo"), ("Baroque", "baroque"),
        ("Ukiyo_e", "baroque"), ("Baroque", "ukiyo"),
        ("Ukiyo_e", "content"), ("Baroque", "content"),
    ]
    for source, question in pairs:
        differences = paired(kept, key, source, question)
        if not differences:
            continue
        up = sum(1 for d in differences if d > 0)
        down = sum(1 for d in differences if d < 0)
        label = f"{source} -> {question}"
        print(
            f"{label:38}{_mean(differences):>+7.2f}{up:>8}{len(differences) - up - down:>6}"
            f"{down:>7}{sign_test(differences):>9.4f}"
        )


if __name__ == "__main__":
    main()
