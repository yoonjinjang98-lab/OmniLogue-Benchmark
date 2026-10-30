"""`db` -- the one command every Claude Code session (and human) in this repo
should use to list/validate/run/report, instead of hand-writing python -c
snippets that drift from each other across three people's sessions.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from dialogue_benchmarks._paths import DATA_ROOT, PLUGINS_ROOT, PROJECT_ROOT


def _cmd_list(args):
    kind = args.kind
    if kind in ("datasets", "all"):
        from dialogue_benchmarks.datasets.registry import list_datasets

        print("# datasets")
        for info in list_datasets():
            print(
                f"{info.name:<26} tier={info.tier or '?':<3} task_type={info.task_type or '-':<10} "
                f"axes={info.axes} adapter={'yes' if info.has_adapter else 'NO'}"
            )
    if kind in ("methods", "all"):
        from dialogue_benchmarks.methods.registry import list_methods

        print("\n# methods")
        for info in list_methods():
            print(f"{info.name:<26} implemented={'yes' if info.implemented else 'NO'}")
    if kind in ("metrics", "all"):
        from dialogue_benchmarks.metrics.registry import list_metrics

        print("\n# metrics")
        for info in list_metrics():
            print(f"{info.name:<26} task_type={info.task_type or '-'}")


def _cmd_run(args):
    from dialogue_benchmarks.eval.paper_baselines import diff_aggregate, diff_by_category, load_paper_baseline
    from dialogue_benchmarks.eval.runner import evaluate

    method_config = json.loads(args.config) if args.config else {}
    metrics = args.metrics.split(",") if args.metrics else None
    table = evaluate(
        dataset=args.dataset,
        methods=args.methods.split(","),
        metrics=metrics,
        variant=args.variant,
        model=args.model,
        limit=args.limit,
        method_config=method_config,
        force=args.force,
    )
    print(table)

    if not args.compare_paper:
        return

    for method in table.methods:
        baseline = load_paper_baseline(args.dataset, method)
        for metric, by_cat in table.by_category.get(method, {}).items():
            if not by_cat:
                continue
            baseline_metric = (baseline or {}).get("metrics", {}).get(metric, {})
            rows = diff_by_category(
                by_cat, baseline_metric, (baseline or {}).get("category_labels")
            )
            header = f"\n-- {method}: {metric} by category"
            if baseline_metric:
                header += f" (vs {baseline['citation']}, {baseline.get('model', '?')})"
            print(header + " --")
            has_paper = any(p is not None for _, _, p, _ in rows)
            col = "{:<16}{:>10}" + ("{:>10}{:>10}" if has_paper else "")
            print(col.format("category", "ours", *(["paper", "delta"] if has_paper else [])))
            for label, ours, paper, delta in rows:
                cells = [label, f"{ours:.3f}" if ours is not None else "-"]
                if has_paper:
                    cells += [
                        f"{paper:.3f}" if paper is not None else "-",
                        f"{delta:+.3f}" if delta is not None else "-",
                    ]
                print(col.format(*cells))

        baseline_aggregate = (baseline or {}).get("aggregate")
        if baseline_aggregate:
            agg_rows = diff_aggregate(table.rows.get(method, {}), baseline_aggregate)
            agg_rows = [r for r in agg_rows if r[1] is not None or r[2] is not None]
            if agg_rows:
                print(f"\n-- {method}: vs {baseline['citation']}, {baseline.get('model', '?')} (aggregate) --")
                print("{:<16}{:>10}{:>10}{:>10}".format("metric", "ours", "paper", "delta"))
                for metric, ours, paper, delta in agg_rows:
                    print(
                        "{:<16}{:>10}{:>10}{:>10}".format(
                            metric,
                            f"{ours:.3f}" if ours is not None else "-",
                            f"{paper:.3f}" if paper is not None else "-",
                            f"{delta:+.3f}" if delta is not None else "-",
                        )
                    )


def _cmd_bench(args):
    from dialogue_benchmarks.eval.bench import run_bench

    with open(args.config) as f:
        config = json.load(f)

    results = run_bench(config)

    out = {}
    for dataset, result in results.items():
        print(f"\n=== {dataset} ===")
        if isinstance(result, dict):
            print(f"  skipped: {result['error']}")
            out[dataset] = result
        else:
            print(result)
            out[dataset] = result.to_dict()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


def _score_rows(results_dir) -> list:
    """Every cached score on disk, flattened to one row per (dataset, variant,
    method, run_id, metric[.sub-score]) -- dict-valued aggregates (e.g.
    recall_at_k's {"R@1": ..}) become one row per key; per-category
    breakdowns (*_by_category) are left out of this summary."""
    import json

    rows = []
    for scores_path in sorted(results_dir.glob("*/*/*/*/scores.json")):
        run_dir = scores_path.parent
        dataset, variant, method, run_id = run_dir.parts[-4:]
        with open(scores_path) as f:
            scores = json.load(f)
        model = None
        if (run_dir / "run.json").exists():
            with open(run_dir / "run.json") as f:
                model = json.load(f).get("model")
        for metric, value in scores.items():
            if metric.endswith("_by_category"):
                continue
            values = value.items() if isinstance(value, dict) else [(None, value)]
            for sub, v in values:
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "method": method,
                        "run_id": run_id,
                        "model": model or "",
                        "metric": metric if sub is None else f"{metric}.{sub}",
                        "value": v,
                    }
                )
    return rows


def _cmd_report(args):
    from dialogue_benchmarks.datasets.registry import list_datasets
    from dialogue_benchmarks.methods.registry import list_methods

    results_dir = DATA_ROOT / "results"
    if args.format in ("csv", "md"):
        rows = _score_rows(results_dir)
        cols = ["dataset", "variant", "method", "run_id", "model", "metric", "value"]
        if args.format == "csv":
            import csv

            writer = csv.DictWriter(sys.stdout, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        else:
            print("| " + " | ".join(cols) + " |")
            print("|" + "---|" * len(cols))
            for r in rows:
                v = r["value"]
                cell = f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
                print("| " + " | ".join([*(str(r[c]) for c in cols[:-1]), cell]) + " |")
        return
    datasets = list_datasets()
    methods = list_methods()

    print(f"{'dataset':<26}{'task_type':<14}" + "".join(f"{m.name:<16}" for m in methods))
    for ds in datasets:
        row = f"{ds.name:<26}{ds.task_type or '-':<14}"
        for m in methods:
            variant_dir = results_dir / ds.name / "raw" / m.name
            has_result = variant_dir.exists() and any(variant_dir.iterdir()) if variant_dir.exists() else False
            row += f"{'done' if has_result else '-':<16}"
        print(row)


def _cmd_validate(args):
    script = PROJECT_ROOT / "scripts" / "validate_plugins.py"
    import subprocess

    cmd = [sys.executable, str(script)]
    if args.changed:
        cmd.append("--changed")
    raise SystemExit(subprocess.call(cmd))


def _cmd_new(args):
    kind_map = {"dataset": "datasets", "method": "methods", "metric": "metrics"}
    plural = kind_map[args.kind]
    template_dir = PROJECT_ROOT / "templates" / args.kind
    dest = PLUGINS_ROOT / plural / args.name
    if dest.exists():
        print(f"error: {dest} already exists", file=sys.stderr)
        raise SystemExit(1)
    if not template_dir.exists():
        print(f"error: no template at {template_dir}", file=sys.stderr)
        raise SystemExit(1)
    shutil.copytree(template_dir, dest)
    print(f"created {dest} from templates/{args.kind}/ -- edit meta.json and the entry-point file")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="db", description="dialogue_benchmarks CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list datasets/methods/metrics and their implementation status")
    p_list.add_argument("kind", nargs="?", default="all", choices=["datasets", "methods", "metrics", "all"])
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="evaluate method(s) on a dataset")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--methods", required=True, help="comma-separated method names")
    p_run.add_argument("--metrics", default=None, help="comma-separated metric names (default: task_type default)")
    p_run.add_argument("--variant", default=None)
    p_run.add_argument("--model", default=None, help="passed to each method as config['model']")
    p_run.add_argument("--config", default=None, help="extra method config as a JSON object string")
    p_run.add_argument("--limit", type=int, default=None, help="cap total items evaluated (smoke-test a wiring change cheaply)")
    p_run.add_argument(
        "--compare-paper",
        action="store_true",
        help="also print each metric's per-category breakdown, diffed against "
        "plugins/datasets/<dataset>/paper_baselines.json when that method has an entry",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="ignore cached predictions/scores for this exact (model, config) run_id and "
        "re-run + rescore from scratch (use after changing a method/metric's code)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser(
        "bench", help="run method(s) across every dataset matching an axis, driven by a JSON config"
    )
    p_bench.add_argument("--config", required=True, help="path to a JSON config (see templates/bench_config.json)")
    p_bench.add_argument("--out", default=None, help="also write combined results to this JSON file")
    p_bench.set_defaults(func=_cmd_bench)

    p_report = sub.add_parser("report", help="dataset x method coverage matrix from data/results/")
    p_report.add_argument(
        "--format",
        default="matrix",
        choices=["matrix", "csv", "md"],
        help="matrix: coverage table (default); csv/md: every cached score, one row per metric",
    )
    p_report.set_defaults(func=_cmd_report)

    p_validate = sub.add_parser("validate", help="check plugin contracts (meta.json schema, signatures, sample run)")
    p_validate.add_argument("--changed", action="store_true", help="only validate plugins changed vs HEAD")
    p_validate.set_defaults(func=_cmd_validate)

    p_new = sub.add_parser("new", help="scaffold a new dataset/method/metric plugin from templates/")
    p_new.add_argument("kind", choices=["dataset", "method", "metric"])
    p_new.add_argument("name")
    p_new.set_defaults(func=_cmd_new)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
