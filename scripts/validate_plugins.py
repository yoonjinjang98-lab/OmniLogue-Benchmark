#!/usr/bin/env python3
"""Checks every plugin (or, with --changed, only plugins touched vs HEAD)
against its contract: meta.json schema, entry-point importability, and
signature shape. Deliberately does NOT load raw data or call any LLM -- this
runs on every Edit/Write to plugins/** via the PostToolUse hook (see
.claude/settings.json), so it has to stay fast and free.

Exit code 0 = all good, 1 = at least one plugin failed a check (message on
stderr per failure).
"""
import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dialogue_benchmarks._paths import PLUGINS_ROOT  # noqa: E402

REQUIRED_DATASET_KEYS = {"name", "axes", "tier", "license", "url"}
REQUIRED_METHOD_KEYS = {"name", "description"}
REQUIRED_METRIC_KEYS = {"name", "task_type", "description"}


def _changed_plugin_dirs() -> set:
    """Every plugins/<group>/<name>/ that has an uncommitted or staged change,
    or is untracked -- used by --changed so the hook only re-checks what you
    just touched."""
    repo_root = Path(__file__).resolve().parent.parent
    cmds = [
        ["git", "diff", "--name-only", "HEAD", "--", "plugins"],
        ["git", "diff", "--cached", "--name-only", "--", "plugins"],
        ["git", "ls-files", "--others", "--exclude-standard", "--", "plugins"],
    ]
    dirs = set()
    for cmd in cmds:
        out = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        for line in out.stdout.splitlines():
            parts = Path(line).parts
            if len(parts) >= 3 and parts[0] == "plugins":
                dirs.add(Path(*parts[:3]))
    return dirs


def _load_meta(path: Path):
    meta_path = path / "meta.json"
    if not meta_path.exists():
        return None, [f"{path}: missing meta.json"]
    try:
        with open(meta_path) as f:
            return json.load(f), []
    except json.JSONDecodeError as e:
        return None, [f"{meta_path}: invalid JSON ({e})"]


# plugins that couldn't be imported only because an optional extra or the
# upstream vendor/ code isn't installed (e.g. a fresh clone of the public repo)
SKIPPED = []


def _import(path: Path, entry_file: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path / entry_file)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module, []
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("dialogue_benchmarks"):
            return None, [f"{path / entry_file}: import failed: {e}"]
        SKIPPED.append(f"{path.parent.name}/{path.name}: optional dependency '{e.name}' not installed")
        return None, []
    except FileNotFoundError as e:
        if "/vendor/" not in str(e):
            return None, [f"{path / entry_file}: import failed: {e}"]
        SKIPPED.append(f"{path.parent.name}/{path.name}: upstream vendor/ code not present (see NOTICE.md)")
        return None, []
    except Exception as e:  # noqa: BLE001 -- report any import-time error as a validation failure
        return None, [f"{path / entry_file}: import failed: {e}"]


def check_dataset(path: Path) -> list:
    errors = []
    meta, errs = _load_meta(path)
    errors += errs
    if meta:
        missing = REQUIRED_DATASET_KEYS - meta.keys()
        if missing:
            errors.append(f"{path}/meta.json: missing keys {sorted(missing)}")

    adapter_file = path / "adapter.py"
    if adapter_file.exists():
        module, errs = _import(path, "adapter.py", f"_validate.datasets.{path.name}")
        errors += errs
        if module:
            if not hasattr(module, "load"):
                errors.append(f"{adapter_file}: no load(raw_dir) function")
            elif len(inspect.signature(module.load).parameters) != 1:
                errors.append(f"{adapter_file}: load() must take exactly one argument (raw_dir)")
            if hasattr(module, "eval_items") and len(inspect.signature(module.eval_items).parameters) != 1:
                errors.append(f"{adapter_file}: eval_items() must take exactly one argument (conversation)")
    return errors


def check_method(path: Path) -> list:
    errors = []
    meta, errs = _load_meta(path)
    errors += errs
    if meta:
        missing = REQUIRED_METHOD_KEYS - meta.keys()
        if missing:
            errors.append(f"{path}/meta.json: missing keys {sorted(missing)}")

    method_file = path / "method.py"
    if method_file.exists():
        module, errs = _import(path, "method.py", f"_validate.methods.{path.name}")
        errors += errs
        if module:
            has_class = hasattr(module, "Method") and inspect.isclass(module.Method)
            has_fn = hasattr(module, "answer") and inspect.isfunction(module.answer)
            if not has_class and not has_fn:
                errors.append(
                    f"{method_file}: exposes neither a `Method` class nor a bare `answer(conversation, question)` function"
                )
            if has_class:
                for required in ("prepare", "answer", "reset"):
                    if not hasattr(module.Method, required):
                        errors.append(f"{method_file}: Method class missing {required}()")
            if has_fn and len(inspect.signature(module.answer).parameters) != 2:
                errors.append(f"{method_file}: answer() must take exactly two arguments (conversation, question)")
    return errors


def check_metric(path: Path) -> list:
    errors = []
    meta, errs = _load_meta(path)
    errors += errs
    if meta:
        missing = REQUIRED_METRIC_KEYS - meta.keys()
        if missing:
            errors.append(f"{path}/meta.json: missing keys {sorted(missing)}")

    metric_file = path / "metric.py"
    if metric_file.exists():
        module, errs = _import(path, "metric.py", f"_validate.metrics.{path.name}")
        errors += errs
        if module:
            if not hasattr(module, "score"):
                errors.append(f"{metric_file}: no score() function")
            elif len(inspect.signature(module.score).parameters) != 2:
                errors.append(f"{metric_file}: score() must take exactly two positional arguments")
            if hasattr(module, "aggregate") and len(inspect.signature(module.aggregate).parameters) not in (2, 3):
                errors.append(
                    f"{metric_file}: aggregate() must take (scores, items) or (scores, items, predictions)"
                )
    return errors


CHECKERS = {"datasets": check_dataset, "methods": check_method, "metrics": check_metric}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", action="store_true", help="only validate plugins changed vs HEAD/staged/untracked")
    args = parser.parse_args()

    if args.changed:
        targets = [(d.parts[1], d.name, PLUGINS_ROOT.parent / d) for d in _changed_plugin_dirs()]
    else:
        targets = []
        for group in CHECKERS:
            group_dir = PLUGINS_ROOT / group
            if not group_dir.exists():
                continue
            for d in sorted(group_dir.iterdir()):
                if d.is_dir():
                    targets.append((group, d.name, d))

    all_errors = []
    for group, name, path in targets:
        checker = CHECKERS.get(group)
        if checker is None:
            continue
        errors = checker(path)
        if errors:
            all_errors.append((f"{group}/{name}", errors))

    for s in SKIPPED:
        print(f"SKIP {s}", file=sys.stderr)
    if not all_errors:
        skipped = f" ({len(SKIPPED)} skipped)" if SKIPPED else ""
        print(f"validate: {len(targets)} plugin(s) checked, all OK{skipped}")
        return 0

    for plugin, errors in all_errors:
        print(f"FAIL {plugin}:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
    print(f"\nvalidate: {len(all_errors)}/{len(targets)} plugin(s) failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
