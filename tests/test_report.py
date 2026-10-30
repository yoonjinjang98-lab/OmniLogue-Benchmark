"""db report --format csv|md, and ResultsTable rendering of dict-valued
aggregates (recall_at_k, abstention_accuracy, distinct_n, ... return dicts)."""
import json

from dialogue_benchmarks.cli import _score_rows
from dialogue_benchmarks.eval.table import ResultsTable


def test_score_rows_flattens_dict_metrics_and_skips_by_category(tmp_path):
    run_dir = tmp_path / "ds" / "raw" / "m" / "abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "scores.json").write_text(
        json.dumps({"mrr": 0.5, "recall_at_k": {"R@1": 0.25, "R@5": 0.75}, "mrr_by_category": {"x": 1.0}})
    )
    (run_dir / "run.json").write_text(json.dumps({"model": "some-model"}))

    rows = _score_rows(tmp_path)
    got = {(r["metric"], r["value"]) for r in rows}
    assert got == {("mrr", 0.5), ("recall_at_k.R@1", 0.25), ("recall_at_k.R@5", 0.75)}
    assert all(r["dataset"] == "ds" and r["run_id"] == "abc123" and r["model"] == "some-model" for r in rows)


def test_results_table_renders_dict_valued_metrics():
    table = ResultsTable(
        methods=["a", "b"],
        metrics=["mrr", "recall_at_k"],
        rows={
            "a": {"mrr": 0.5, "recall_at_k": {"R@1": 0.25, "R@5": None}},
            "b": {"mrr": 1.0, "recall_at_k": {"R@1": 1.0, "R@5": 1.0}},
        },
    )
    text = repr(table)
    assert "recall_at_k.R@1" in text and "recall_at_k.R@5" in text
    columns, flat = table.flat_rows()
    assert columns == ["mrr", "recall_at_k.R@1", "recall_at_k.R@5"]
    assert flat["a"]["recall_at_k.R@5"] is None
