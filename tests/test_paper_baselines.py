"""eval/paper_baselines.py has no coverage yet even though `db run
--compare-paper` (cli.py) has depended on it since it was added -- this pins
its two lookup/diff helpers directly, independent of any real dataset's
paper_baselines.json content."""
from dialogue_benchmarks.eval.paper_baselines import diff_aggregate, diff_by_category, load_paper_baseline


def test_load_paper_baseline_missing_dataset_returns_none():
    assert load_paper_baseline("no_such_dataset", "no_such_method") is None


def test_load_paper_baseline_missing_method_returns_none():
    # locomo/paper_baselines.json exists but has no entry for this method
    assert load_paper_baseline("locomo", "no_such_method") is None


def test_load_paper_baseline_known_entry():
    entry = load_paper_baseline("locomo", "a_mem")
    assert entry is not None
    assert "citation" in entry
    assert entry["metrics"]["qa_f1"]["1"] == 0.4665


def test_diff_by_category_matches_missing_and_delta():
    ours = {"1": 0.5, "2": 0.3}
    baseline = {"1": 0.4, "3": 0.9}
    rows = diff_by_category(ours, baseline, {"1": "single_hop"})
    by_cat = {label: (o, p, d) for label, o, p, d in rows}
    assert by_cat["single_hop"] == (0.5, 0.4, 0.5 - 0.4)  # both sides -> real delta
    assert by_cat["2"] == (0.3, None, None)  # ours only, no label mapping -> raw key
    assert by_cat["3"] == (None, 0.9, None)  # baseline only


def test_diff_aggregate_matches_missing_and_delta():
    ours = {"qa_f1": 0.34, "qa_em": 0.17}
    baseline = {"qa_f1": 0.39, "qa_llm_judge": 0.74}
    rows = diff_aggregate(ours, baseline)
    by_metric = {m: (o, p, d) for m, o, p, d in rows}
    assert by_metric["qa_f1"] == (0.34, 0.39, 0.34 - 0.39)
    assert by_metric["qa_em"] == (0.17, None, None)
    assert by_metric["qa_llm_judge"] == (None, 0.74, None)
