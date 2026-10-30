import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks._paths import PLUGINS_ROOT

PLUGINS_DIR = PLUGINS_ROOT / "metrics"


@dataclass
class MetricInfo:
    name: str
    path: Path
    meta: dict

    @property
    def task_type(self):
        return self.meta.get("task_type")


def list_metrics() -> list[MetricInfo]:
    infos = []
    if not PLUGINS_DIR.exists():
        return infos
    for d in sorted(PLUGINS_DIR.iterdir()):
        meta_path = d / "meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        infos.append(MetricInfo(name=d.name, path=d, meta=meta))
    return infos


def get_metric_info(name: str) -> MetricInfo:
    path = PLUGINS_DIR / name
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise KeyError(f"'{name}' is not a registered metric (no plugins/metrics/{name}/meta.json)")
    with open(meta_path) as f:
        meta = json.load(f)
    return MetricInfo(name=name, path=path, meta=meta)


def _import_metric_module(name: str):
    info = get_metric_info(name)
    metric_py = info.path / "metric.py"
    if not metric_py.exists():
        raise NotImplementedError(f"metric '{name}' has no metric.py (plugins/metrics/{name}/)")
    module_name = f"dialogue_benchmarks._metrics.{name}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, metric_py)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def _takes_item(score_fn) -> bool:
    """Two plugin styles, both two positional args, disambiguated by name:
    - legacy: def score(prediction, reference) -> float
    - item-aware: def score(item, prediction) -> float  (item is an EvalItem --
      needed for LLM-judge metrics that need the question, or anything that
      wants item.metadata for category breakdowns beyond the automatic one)
    Legacy is the default on any ambiguity, since that's what every metric
    written before this contract existed (qa_em, qa_f1) uses.
    """
    params = list(inspect.signature(score_fn).parameters)
    return params[:1] == ["item"]


def load_metric_fn(name: str):
    """Returns a score_fn(item, prediction) -> float, wrapping legacy
    score(prediction, reference) metrics so callers don't need to know which
    style a given metric.py uses."""
    module = _import_metric_module(name)
    raw_score = module.score
    if _takes_item(raw_score):
        return raw_score
    return lambda item, prediction: raw_score(prediction, item.reference)


def load_aggregate_fn(name: str):
    """Returns agg_fn(scores, items, predictions) -> float | dict, or None if
    the metric doesn't define one (runner then falls back to a plain mean).

    Plugins may define either aggregate(scores, items) or
    aggregate(scores, items, predictions) -- the latter for corpus-level
    metrics that can't be recovered from per-item scores (e.g. distinct_n
    counts unique n-grams across *all* predictions at once)."""
    module = _import_metric_module(name)
    agg = getattr(module, "aggregate", None)
    if agg is None:
        return None
    if len(inspect.signature(agg).parameters) >= 3:
        return agg
    return lambda scores, items, predictions: agg(scores, items)
