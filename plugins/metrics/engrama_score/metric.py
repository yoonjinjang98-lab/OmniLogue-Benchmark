"""engrama_score -- EngramaBench's native scorer (scorer/scorer.py, MIT).

Loaded from the raw-data clone (DATA_ROOT/raw/engramabench/repo/scorer/
scorer.py) at first use rather than copied into this repo, so the numbers are
the benchmark's own and its updates are picked up by re-pulling the clone.
Needs item.metadata["engrama_query"] (the original query record) -- items
from any other dataset score 0.
"""
import importlib.util
import json

from dialogue_benchmarks._paths import DATA_ROOT

_REPO = DATA_ROOT / "raw" / "engramabench" / "repo"
_state = {}


def _scorer():
    if "mod" not in _state:
        spec = importlib.util.spec_from_file_location("_engramabench_scorer", _REPO / "scorer" / "scorer.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with open(_REPO / "datasets" / "full_v1" / "entity_registry.json") as f:
            _state["alias"] = mod.build_alias_table(json.load(f))
        _state["mod"] = mod
    return _state["mod"], _state["alias"]


def _score_one(item, prediction) -> dict:
    query = (item.metadata or {}).get("engrama_query")
    if not query:
        return None
    mod, alias = _scorer()
    return mod.score_query(query, {"system_answer": str(prediction)}, alias)


def score(item, prediction) -> float:
    result = _score_one(item, prediction)
    return float(result["score"]) if result else 0.0


def aggregate(scores: list, items: list, predictions: list) -> dict:
    per_query = [r for r in (_score_one(it, p) for it, p in zip(items, predictions)) if r]
    if not per_query:
        return {"composite_score": 0.0}
    mod, _ = _scorer()
    return mod.aggregate_scores(per_query)["scores"]
