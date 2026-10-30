"""Batch runner: evaluate a set of methods across every dataset matching one or
more axes (e.g. "multiparty"), driven by a single JSON config instead of one
`db run` invocation per dataset. See templates/bench_config.json for the schema.
"""
from dialogue_benchmarks.datasets.registry import list_datasets
from dialogue_benchmarks.eval.runner import evaluate
from dialogue_benchmarks.methods.registry import list_methods


def _resolve_datasets(config: dict) -> list:
    explicit = config.get("datasets")
    if explicit:
        return explicit
    axes = config.get("axes") or []
    infos = list_datasets()
    if axes:
        infos = [i for i in infos if set(axes) <= set(i.axes)]
    return [i.name for i in infos if i.task_type and i.has_adapter]


def _resolve_methods(config: dict) -> list:
    explicit = config.get("methods")
    if explicit:
        return explicit
    return [m.name for m in list_methods() if m.implemented]


def run_bench(config: dict) -> dict:
    """Runs `evaluate()` once per resolved dataset (same methods/metrics/model/
    limit/method_config for all of them). A dataset that errors -- e.g. a
    "dialogue"-task dataset whose adapter has no eval_items() yet -- is recorded
    as {"error": str} instead of aborting the rest of the batch, since not every
    dataset in an axis is fully wired up yet."""
    datasets = _resolve_datasets(config)
    methods = _resolve_methods(config)
    if not datasets:
        raise ValueError(
            "no datasets matched -- check 'axes'/'datasets' in the config against `db list datasets`"
        )
    if not methods:
        raise ValueError(
            "no methods to run -- check 'methods' in the config against `db list methods`"
        )

    results = {}
    for dataset in datasets:
        try:
            results[dataset] = evaluate(
                dataset=dataset,
                methods=methods,
                metrics=config.get("metrics"),
                variant=config.get("variant"),
                model=config.get("model"),
                limit=config.get("limit"),
                method_config=config.get("method_config"),
            )
        except Exception as exc:
            results[dataset] = {"error": str(exc)}
    return results
