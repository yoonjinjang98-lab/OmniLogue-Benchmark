import getpass
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dialogue_benchmarks._paths import DATA_ROOT, PROJECT_ROOT

RESULTS_DIR = DATA_ROOT / "results"


def compute_run_id(model: Optional[str], config: dict) -> str:
    """Identifies one (model, config) combination for a method so two people
    running the same method with different models/settings land in separate
    result folders instead of one silently overwriting the other.

    "default" (not a hash) for the common case of a method with no model/config
    at all -- keeps results/<dataset>/<variant>/<method>/default/ readable for
    trivial methods like smoke_test, matching the original single-folder layout.
    """
    if not model and not config:
        return "default"
    payload = json.dumps({"model": model, **config}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _dir(dataset: str, variant: str, method: str, run_id: str) -> Path:
    return RESULTS_DIR / dataset / variant / method / run_id


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def get_scores(dataset: str, variant: str, method: str, run_id: str) -> dict:
    path = _dir(dataset, variant, method, run_id) / "scores.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_scores(dataset: str, variant: str, method: str, run_id: str, scores: dict) -> None:
    _write_json_atomic(_dir(dataset, variant, method, run_id) / "scores.json", scores)


def load_predictions(dataset: str, variant: str, method: str, run_id: str) -> Optional[list]:
    path = _dir(dataset, variant, method, run_id) / "predictions.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_predictions(dataset: str, variant: str, method: str, run_id: str, predictions: list) -> None:
    path = _dir(dataset, variant, method, run_id) / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    tmp.rename(path)


def save_run_meta(
    dataset: str,
    variant: str,
    method: str,
    run_id: str,
    *,
    model: Optional[str],
    config: dict,
    n_items: int,
    duration_s: float,
    total_cost: float = 0.0,
) -> None:
    """run.json: who/what/when produced this folder's predictions, so a second
    person looking at shared results/ can tell what they're comparing."""
    meta = {
        "dataset": dataset,
        "variant": variant,
        "method": method,
        "run_id": run_id,
        "model": model,
        "config": config,
        "git_commit": _git_commit(),
        "user": getpass.getuser(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_items": n_items,
        "duration_s": duration_s,
        "total_cost_usd": total_cost,
    }
    _write_json_atomic(_dir(dataset, variant, method, run_id) / "run.json", meta)
