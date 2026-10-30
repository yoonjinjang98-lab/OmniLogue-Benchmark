"""Published per-category baseline numbers for a dataset's methods, so a
`db run` can show how a reproduction compares to what the method's own paper
reported -- not just a bare score with nothing to judge it against.

Looked up from plugins/datasets/<dataset>/paper_baselines.json (optional --
most dataset/method pairs won't have one yet). Schema:

    {
      "<method_name>": {
        "citation": "human-readable paper + table reference",
        "model": "the LLM the paper used (for context; not enforced)",
        "category_labels": {"<raw category value>": "human label", ...},
        "metrics": {"<metric_name>": {"<raw category value>": <0-1 float>, ...}},
        "aggregate": {"<metric_name>": <0-1 float>, ...}
      }
    }

`metrics` (per-category) and `aggregate` (single overall number) are both
optional and independent -- a paper's table might report only one or the
other. `aggregate` exists because not every paper breaks its numbers down
by our category scheme (e.g. a third-party reproduction table that reports
one F1 for the whole dataset); it's diffed against ResultsTable.rows
(the overall score) rather than ResultsTable.by_category.

All scores are stored on the same 0-1 scale our own metrics use (a paper
reporting F1 as a 0-100 percentage should be divided by 100 when transcribed
here) so they can be diffed directly against ResultsTable.by_category/.rows.
"""
import json
from typing import Optional

from dialogue_benchmarks._paths import PLUGINS_ROOT


def load_paper_baseline(dataset: str, method: str) -> Optional[dict]:
    """The `method` entry from plugins/datasets/<dataset>/paper_baselines.json,
    or None if that file doesn't exist or has nothing for this method."""
    path = PLUGINS_ROOT / "datasets" / dataset / "paper_baselines.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(method)


def diff_by_category(ours: dict, baseline: dict, category_labels: Optional[dict] = None) -> list:
    """{category: our_score} + {category: paper_score} -> sorted list of
    (label, ours, paper, delta) tuples, one per category either side has a
    score for (missing values on one side show as None, no delta)."""
    labels = category_labels or {}
    rows = []
    for cat in sorted(set(ours) | set(baseline)):
        o = ours.get(cat)
        p = baseline.get(cat)
        delta = (o - p) if (o is not None and p is not None) else None
        rows.append((labels.get(cat, cat), o, p, delta))
    return rows


def diff_aggregate(ours: dict, baseline: dict) -> list:
    """{metric_name: our_overall_score} + baseline's "aggregate" dict ->
    sorted list of (metric, ours, paper, delta) tuples, one per metric either
    side has a value for. Separate from diff_by_category because a paper's
    table (e.g. a third-party reproduction used as a comparison baseline)
    may report only one overall number per metric, with no breakdown that
    lines up with our category scheme."""
    rows = []
    for metric in sorted(set(ours) | set(baseline)):
        o = ours.get(metric)
        p = baseline.get(metric)
        delta = (o - p) if (o is not None and p is not None) else None
        rows.append((metric, o, p, delta))
    return rows
