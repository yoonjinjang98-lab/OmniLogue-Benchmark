import time
from typing import Optional

from dialogue_benchmarks import usage
from dialogue_benchmarks.contracts import EvalItem
from dialogue_benchmarks.datasets.loader import load_dataset
from dialogue_benchmarks.datasets.registry import _import_adapter, get_dataset_info
from dialogue_benchmarks.eval import results_store
from dialogue_benchmarks.eval.table import ResultsTable
from dialogue_benchmarks.methods.registry import load_method
from dialogue_benchmarks.metrics.registry import list_metrics, load_aggregate_fn, load_metric_fn
from dialogue_benchmarks.tasks import get_task_module
from dialogue_benchmarks.tasks import run as run_method_over_items


def _default_metrics(task_type: Optional[str], dataset_meta: Optional[dict] = None) -> list:
    """Every registered metric for this task_type, minus ones that opt out via
    meta.json's "default": false (for metrics with a real per-item cost, e.g. an
    LLM-judge call -- those should only run when a caller asks for them by name).

    A dataset's own meta.json may add to that with "extra_metrics": [...] --
    for metrics that only make sense on some datasets of a task_type (e.g.
    mc_accuracy needs item.options, abstention_accuracy needs unanswerable
    items) and so can't be on by default for every qa dataset."""
    names = [m.name for m in list_metrics() if m.task_type == task_type and m.meta.get("default", True)]
    for extra in (dataset_meta or {}).get("extra_metrics", []):
        if extra not in names:
            names.append(extra)
    return names


def _limit_groups(groups: list, limit: Optional[int]) -> list:
    """Caps the total number of items across all conversations, keeping whole
    conversations' worth of items together where possible -- used for cheap
    smoke-testing a method (`db run --limit 5`) before spending real API budget
    on a full run."""
    if limit is None:
        return groups
    limited = []
    remaining = limit
    for conv, items in groups:
        if remaining <= 0:
            break
        take = items[:remaining]
        if take:
            limited.append((conv, take))
        remaining -= len(take)
    return limited


def _run_predictions(method_name, groups, model, method_config):
    method_kwargs = dict(method_config)
    if model is not None:
        method_kwargs.setdefault("model", model)
    method = load_method(method_name, **method_kwargs)

    start = time.monotonic()
    pairs = run_method_over_items(method, groups)
    duration = time.monotonic() - start

    predictions = [
        {
            "conversation_id": item.conversation_id,
            "question": item.question,
            "reference": item.reference,
            "options": item.options,
            "candidates": item.candidates,
            "prediction": out.answer,
            "retrieved": out.retrieved,
            "tokens": out.tokens,
            "latency": out.latency,
            "metadata": item.metadata,
        }
        for item, out in pairs
    ]
    total_cost = sum((out.cost or 0.0) for _, out in pairs)
    return predictions, duration, total_cost


def evaluate(
    dataset: str,
    methods: list,
    metrics: Optional[list] = None,
    variant: Optional[str] = None,
    model: Optional[str] = None,
    limit: Optional[int] = None,
    method_config: Optional[dict] = None,
    force: bool = False,
) -> ResultsTable:
    """model / method_config apply to every method in this call -- that's the
    natural unit of comparison (same model, different baselines). Running the
    same method under a different model is a separate evaluate() call; it
    lands in its own results/.../<run_id>/ folder rather than overwriting the
    first (see results_store.compute_run_id).

    force=True re-runs predictions and recomputes every metric in `metrics`
    for this exact (dataset, variant, model, method_config) run_id even if
    they're already cached -- for when a method/metric's code changed and the
    stale cached numbers would otherwise be served forever. Metrics already
    cached under this run_id but *not* requested this call are left alone
    rather than dropped from scores.json."""
    ds_info = get_dataset_info(dataset)
    task_type = ds_info.task_type
    if not task_type:
        raise ValueError(
            f"dataset '{dataset}' has no task_type set in plugins/datasets/{dataset}/meta.json"
        )

    if metrics is None:
        metrics = _default_metrics(task_type, ds_info.meta)
        if not metrics:
            raise ValueError(
                f"no default metrics registered for task_type='{task_type}'; pass metrics= explicitly"
            )

    method_config = method_config or {}
    variant_id = variant or "raw"
    run_id = results_store.compute_run_id(model, method_config)

    convs = load_dataset(dataset, variant=variant)
    adapter_module = _import_adapter(dataset)
    task_module = get_task_module(task_type)
    groups = _limit_groups(task_module.build_items(adapter_module, convs), limit)
    if not groups:
        raise ValueError(
            f"dataset '{dataset}' (variant={variant_id}) produced no evaluable items for task_type='{task_type}'"
        )

    rows = {}
    by_category_rows = {}
    for method_name in methods:
        cached_scores = {} if force else results_store.get_scores(dataset, variant_id, method_name, run_id)
        missing_metrics = list(metrics) if force else [m for m in metrics if m not in cached_scores]

        # --limit is for cheap, disposable smoke-testing (CLAUDE.md) -- it must
        # never write under a real run_id, or a limited run sharing a run_id
        # with a full one would silently truncate/replace the full result
        # (predictions.jsonl, scores.json) with the smaller one.
        should_persist = limit is None

        # tags every billable LLM call below in the usage ledger (usage.py)
        tags = {"dataset": dataset, "variant": variant_id, "method": method_name, "run_id": run_id}
        if missing_metrics:
            predictions = None if force else results_store.load_predictions(dataset, variant_id, method_name, run_id)
            if predictions is None:
                with usage.context(purpose="method", **tags):
                    predictions, duration, total_cost = _run_predictions(method_name, groups, model, method_config)
                if should_persist:
                    results_store.save_predictions(dataset, variant_id, method_name, run_id, predictions)
                    results_store.save_run_meta(
                        dataset,
                        variant_id,
                        method_name,
                        run_id,
                        model=model,
                        config=method_config,
                        n_items=len(predictions),
                        duration_s=duration,
                        total_cost=total_cost,
                    )

            items = [
                EvalItem(
                    conversation_id=p["conversation_id"],
                    question=p["question"],
                    reference=p["reference"],
                    options=p.get("options"),
                    candidates=p.get("candidates"),
                    metadata=p.get("metadata") or {},
                )
                for p in predictions
            ]

            for metric_name in missing_metrics:
                score_fn = load_metric_fn(metric_name)
                agg_fn = load_aggregate_fn(metric_name)
                with usage.context(purpose="judge", metric=metric_name, **tags):
                    per_item = [score_fn(item, p["prediction"]) for item, p in zip(items, predictions)]
                    cached_scores[metric_name] = (
                        agg_fn(per_item, items, [p["prediction"] for p in predictions])
                        if agg_fn is not None
                        else sum(per_item) / len(per_item)
                    )

                by_category: dict = {}
                for item, s in zip(items, per_item):
                    cat = item.metadata.get("category") if isinstance(item.metadata, dict) else None
                    if cat is not None:
                        by_category.setdefault(str(cat), []).append(s)
                if by_category:
                    cached_scores[f"{metric_name}_by_category"] = {
                        cat: sum(vals) / len(vals) for cat, vals in by_category.items()
                    }

            if should_persist:
                if force:
                    # a forced re-run only touches the metrics actually
                    # requested this call -- fold back in whatever else was
                    # already on disk under this run_id so it isn't silently
                    # dropped.
                    cached_scores = {
                        **results_store.get_scores(dataset, variant_id, method_name, run_id),
                        **cached_scores,
                    }
                results_store.save_scores(dataset, variant_id, method_name, run_id, cached_scores)

        rows[method_name] = {m: cached_scores[m] for m in metrics}
        by_category_rows[method_name] = {
            m: cached_scores.get(f"{m}_by_category", {}) for m in metrics
        }

    return ResultsTable(
        methods=list(methods), metrics=list(metrics), rows=rows, by_category=by_category_rows
    )
