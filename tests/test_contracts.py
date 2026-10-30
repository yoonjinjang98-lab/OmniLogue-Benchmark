"""Parametrized over whatever's currently registered -- add a new plugin under
plugins/ and it's covered here automatically, no new test to write.
"""
import inspect

import pytest

from dialogue_benchmarks.datasets.registry import list_datasets
from dialogue_benchmarks.methods.registry import list_methods
from dialogue_benchmarks.metrics.registry import list_metrics


def _require_raw(name: str) -> None:
    """Data-dependent tests skip (not fail) on a checkout without that
    dataset's raw files, e.g. a fresh clone of the public repo -- the raw data
    is never part of the repository (meta.json's url says where to get it)."""
    from dialogue_benchmarks.datasets.registry import get_dataset_info

    raw = get_dataset_info(name).raw_path
    if not raw.is_dir() or not any(raw.iterdir()):
        pytest.skip(f"no raw data for {name} under {raw}")


def _load_or_skip(name: str) -> list:
    """Like _require_raw, for adapters that read another dataset's raw dir
    (msc_longterm reads raw/msc): load, and skip only if the files are missing."""
    from dialogue_benchmarks._paths import DATA_ROOT
    from dialogue_benchmarks.datasets.loader import load_dataset

    from dialogue_benchmarks.datasets.registry import get_dataset_info

    try:
        convs = load_dataset(name)
    except FileNotFoundError as e:
        if str(DATA_ROOT / "raw") not in str(e):
            raise
        pytest.skip(f"no raw data for {name}: {e}")
    if not convs and not get_dataset_info(name).raw_path.exists():
        pytest.skip(f"no raw data for {name}")  # adapters that glob return [] instead of raising
    return convs


@pytest.mark.parametrize("info", list_datasets(), ids=lambda i: i.name)
def test_dataset_meta_schema(info):
    required = {"name", "axes", "tier", "license", "url"}
    missing = required - info.meta.keys()
    assert not missing, f"{info.name}: meta.json missing {missing}"


@pytest.mark.parametrize("info", [i for i in list_datasets() if i.has_adapter], ids=lambda i: i.name)
def test_adapter_loads(info):
    convs = _load_or_skip(info.name)
    assert isinstance(convs, list) and len(convs) > 0, f"{info.name}: load() returned no conversations"


@pytest.mark.parametrize("info", list_methods(), ids=lambda i: i.name)
def test_method_meta_schema(info):
    required = {"name", "description"}
    missing = required - info.meta.keys()
    assert not missing, f"{info.name}: meta.json missing {missing}"


@pytest.mark.parametrize("info", [i for i in list_methods() if i.implemented], ids=lambda i: i.name)
def test_method_entry_point_shape(info):
    from dialogue_benchmarks.methods.registry import _import_method_module

    try:
        module = _import_method_module(info.name)
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("dialogue_benchmarks"):
            raise
        pytest.skip(f"{info.name}: optional dependency not installed ({e.name}) -- see pyproject extras")
    except FileNotFoundError as e:
        if "/vendor/" not in str(e):
            raise
        pytest.skip(f"{info.name}: upstream vendor/ code not present -- see NOTICE.md")
    has_class = hasattr(module, "Method") and inspect.isclass(module.Method)
    has_fn = hasattr(module, "answer") and inspect.isfunction(module.answer)
    assert has_class or has_fn, f"{info.name}: exposes neither Method class nor answer() function"
    if has_class:
        for required in ("prepare", "answer", "reset"):
            assert hasattr(module.Method, required), f"{info.name}: Method missing {required}()"


@pytest.mark.parametrize("info", list_metrics(), ids=lambda i: i.name)
def test_metric_meta_schema(info):
    required = {"name", "task_type", "description"}
    missing = required - info.meta.keys()
    assert not missing, f"{info.name}: meta.json missing {missing}"


@pytest.mark.parametrize("info", list_metrics(), ids=lambda i: i.name)
def test_metric_score_fn_shape(info):
    from dialogue_benchmarks.metrics.registry import _import_metric_module

    module = _import_metric_module(info.name)
    assert hasattr(module, "score"), f"{info.name}: no score() function"
    assert len(inspect.signature(module.score).parameters) == 2, f"{info.name}: score() must take 2 args"


@pytest.mark.parametrize(
    "info", [i for i in list_datasets() if i.has_adapter and i.task_type == "qa"], ids=lambda i: i.name
)
def test_qa_build_items_produces_scoreable_items(info):
    """Every task_type='qa' dataset must produce items whose `.reference` a plain
    string metric (qa_em/qa_f1) can actually score -- catches adapters whose QA
    dataclass doesn't use the plain `.answer` convention (e.g. molweni's SQuAD-style
    `.answers: list`, empty when unanswerable) before a full run does."""
    from dialogue_benchmarks.datasets.loader import load_dataset
    from dialogue_benchmarks.datasets.registry import _import_adapter
    from dialogue_benchmarks.tasks.qa import build_items

    convs = _load_or_skip(info.name)
    adapter_module = _import_adapter(info.name)
    groups = build_items(adapter_module, convs)
    assert groups, f"{info.name}: build_items() produced no groups"
    for _, items in groups:
        for item in items:
            assert isinstance(item.question, str) and item.question
            assert not isinstance(item.reference, (list, dict)), (
                f"{info.name}: EvalItem.reference is a {type(item.reference).__name__}, "
                f"not scoreable by qa_em/qa_f1 -- see contracts.py / tasks/_common.py"
            )


def test_smoke_test_locomo_regression():
    """smoke_test's whole purpose is verifying the pipeline wiring end-to-end;
    pin its known-good score so a refactor that silently breaks scoring, item
    building, or caching gets caught immediately."""
    from dialogue_benchmarks.eval.runner import evaluate

    _require_raw("locomo")
    table = evaluate(dataset="locomo", methods=["smoke_test"])
    assert table.rows["smoke_test"]["qa_em"] == pytest.approx(0.0, abs=1e-6)
    assert table.rows["smoke_test"]["qa_f1"] == pytest.approx(0.0286, abs=1e-3)


def test_evaluate_force_recomputes_only_requested_metrics(tmp_path, monkeypatch):
    """force=True must ignore a stale cached score for a requested metric, but
    leave an *unrequested* metric's cached score untouched rather than
    dropping it from scores.json -- see runner.py's docstring on `force`.
    Runs against a scratch results dir (not the shared one) since it
    deliberately corrupts a cached score."""
    from dialogue_benchmarks.eval import results_store
    from dialogue_benchmarks.eval.runner import evaluate

    monkeypatch.setattr(results_store, "RESULTS_DIR", tmp_path)
    _require_raw("locomo")

    table = evaluate(dataset="locomo", methods=["smoke_test"], metrics=["qa_em", "qa_f1"])
    real_em = table.rows["smoke_test"]["qa_em"]
    real_f1 = table.rows["smoke_test"]["qa_f1"]

    scores = results_store.get_scores("locomo", "raw", "smoke_test", "default")
    scores["qa_em"] = -1.0  # simulate a stale/corrupted cached value
    results_store.save_scores("locomo", "raw", "smoke_test", "default", scores)

    # without force: the corrupted cache is served as-is
    stale = evaluate(dataset="locomo", methods=["smoke_test"], metrics=["qa_em"])
    assert stale.rows["smoke_test"]["qa_em"] == -1.0

    # forcing only qa_em: it's recomputed correctly, and qa_f1 -- not part of
    # this call's metrics -- survives untouched in scores.json
    forced = evaluate(dataset="locomo", methods=["smoke_test"], metrics=["qa_em"], force=True)
    assert forced.rows["smoke_test"]["qa_em"] == pytest.approx(real_em, abs=1e-6)
    assert results_store.get_scores("locomo", "raw", "smoke_test", "default")["qa_f1"] == pytest.approx(
        real_f1, abs=1e-6
    )


def test_evaluate_limit_never_persists(tmp_path, monkeypatch):
    """--limit is a disposable smoke test (CLAUDE.md) -- a limited run must
    never overwrite a run_id's real predictions.jsonl/scores.json on disk,
    even under --force. Regression test for an actual incident: a `--limit 3
    --force` smoke test against an existing full run_id silently truncated
    its predictions.jsonl from ~1900 items to 3 and its qa_em score to match."""
    from dialogue_benchmarks.eval import results_store
    from dialogue_benchmarks.eval.runner import evaluate

    monkeypatch.setattr(results_store, "RESULTS_DIR", tmp_path)
    _require_raw("locomo")

    full = evaluate(dataset="locomo", methods=["smoke_test"], metrics=["qa_em"])
    real_em = full.rows["smoke_test"]["qa_em"]
    n_full = len(results_store.load_predictions("locomo", "raw", "smoke_test", "default"))

    limited = evaluate(dataset="locomo", methods=["smoke_test"], metrics=["qa_em"], limit=1, force=True)
    assert limited.rows["smoke_test"]["qa_em"] is not None  # still returns a real (in-memory) result

    on_disk_predictions = results_store.load_predictions("locomo", "raw", "smoke_test", "default")
    assert len(on_disk_predictions) == n_full
    assert results_store.get_scores("locomo", "raw", "smoke_test", "default")["qa_em"] == pytest.approx(
        real_em, abs=1e-6
    )
