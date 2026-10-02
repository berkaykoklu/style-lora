"""findings.json is what the site reads. These guard what it may say."""

import json
from pathlib import Path

FINDINGS = json.loads((Path(__file__).parent.parent / "findings.json").read_text())

# Field names that would carry an absolute 1-5 level rather than a difference.
#
# Checked by name rather than by value: the withheld levels are small numbers on
# the same scale as the differences, and +1.31 as a paired gain happens to equal
# one of them exactly. A value scan flags that and is wrong; the schema is what
# actually separates the two.
LEVEL_FIELDS = ("mean", "average", "levels", "absolute", "by_source", "scores", "ratings")


def _field_names(node: object) -> list[str]:
    if isinstance(node, dict):
        return list(node) + [n for v in node.values() for n in _field_names(v)]
    if isinstance(node, list):
        return [n for v in node for n in _field_names(v)]
    return []


def test_the_site_can_find_every_section_it_needs() -> None:
    for section in ("setup", "gates", "clip_separation_by_strength",
                    "clip_content_by_strength", "controls", "human_blind_rating",
                    "headline", "limits"):
        assert section in FINDINGS, section


def test_no_absolute_rating_level_is_published() -> None:
    """The rule Berkay set on 2026-10-02, enforced rather than remembered."""
    names = _field_names(FINDINGS["human_blind_rating"])
    leaked = [n for n in names if n.lower() in LEVEL_FIELDS]

    assert not leaked, f"fields that would carry a 1-5 level: {leaked}"


def test_every_human_number_is_a_difference_not_a_level() -> None:
    """A level survives one harsh rater badly; a difference is that rater's own
    scale against itself, which marking everything down leaves intact."""
    for comparison in FINDINGS["human_blind_rating"]["comparisons"]:
        assert "mean_difference" in comparison
        assert "mean" not in comparison
        assert set(comparison) <= {
            "adapter", "question", "mean_difference", "up", "tied", "down", "p"
        }


def test_each_comparison_counts_every_prompt() -> None:
    """up + tied + down has to be the number of prompts, or a direction went
    missing and the sign test was run on fewer than it reports."""
    prompts = FINDINGS["human_blind_rating"]["prompts"]
    for comparison in FINDINGS["human_blind_rating"]["comparisons"]:
        total = comparison["up"] + comparison["tied"] + comparison["down"]
        assert total == prompts, comparison


def test_the_publishing_rule_travels_with_the_data() -> None:
    """Whoever builds the page should not have to be told separately."""
    rule = FINDINGS["publishing_rule"]
    assert rule["withheld"]
    assert rule["why"]


def test_the_limits_are_not_empty() -> None:
    """One rater and sixteen prompts are the shape of this result, not a
    footnote to it."""
    limits = FINDINGS["limits"]
    assert len(limits) >= 4
    assert any("rater" in limit.lower() for limit in limits)


def test_every_measured_section_says_where_it_came_from() -> None:
    for section in ("gates", "clip_separation_by_strength", "clip_content_by_strength",
                    "controls", "human_blind_rating"):
        assert FINDINGS[section].get("provenance"), section


def test_clip_reports_no_content_cost_which_is_the_whole_point() -> None:
    """If this ever stops being true the headline is wrong and has to change."""
    content = FINDINGS["clip_content_by_strength"]
    assert all(
        row[style] >= content["base"]
        for row in content["rows"]
        for style in ("Ukiyo_e", "Baroque")
    )


def test_a_control_does_not_earn_what_the_adapter_earns() -> None:
    """Tone must not be able to fake the measure, or the curves mean nothing."""
    controls = FINDINGS["controls"]
    biggest = max(abs(controls[name]) for name in ("sepia", "darken", "soften", "muted"))
    adapter = FINDINGS["clip_separation_by_strength"]["rows"][-1]["axis"]

    assert biggest < adapter / 3
