"""Paid-API usage ledger: one JSON line per *billable* call that actually hit a
provider (OpenAI / Anthropic) -- cache hits cost nothing and aren't recorded.

Exists so spend can be reconciled against the provider's own usage dashboard
and invoice (research-grant receipts need "what was this money used for"):
every line carries the experiment it belonged to (dataset / method / metric /
run_id, set by the runner via `context()`), its purpose (`method` = a
baseline answering, `judge` = a metric scoring, `embed` = an embedding call),
and which worktree made it -- the whole team shares one OS account, so the
worktree is what tells the lead / A / B apart.

Ledger files: DATA_ROOT/usage/<YYYY-MM>.jsonl (UTC month). Summarize with
scripts/usage_report.py.
"""
import contextlib
import contextvars
import json
from datetime import datetime, timezone

from dialogue_benchmarks._paths import DATA_ROOT, PROJECT_ROOT

USAGE_DIR = DATA_ROOT / "usage"

_context: contextvars.ContextVar = contextvars.ContextVar("dialogue_benchmarks_usage_context", default={})


@contextlib.contextmanager
def context(**fields):
    """Tag every billable call made inside this block, e.g.
    `with usage.context(purpose="method", dataset="locomo", method="full_context"):`.
    Nested blocks add to / override the outer one's fields."""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def current() -> dict:
    return dict(_context.get())


def record(*, provider: str, model: str, kind: str, tokens_in: int, tokens_out: int,
           cached_tokens: int, cost: float, latency: float) -> None:
    now = datetime.now(timezone.utc)
    ctx = current()
    entry = {
        "ts": now.isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "kind": kind,  # "chat" / "embed"
        "purpose": "embed" if kind == "embed" else ctx.get("purpose", "other"),
        "tokens_in": tokens_in,
        "cached_tokens": cached_tokens,
        "tokens_out": tokens_out,
        "cost_usd": round(cost, 8),
        "latency_s": round(latency, 3),
        "worktree": PROJECT_ROOT.name if PROJECT_ROOT.parent.name == "worktrees" else "main",
        **{k: ctx[k] for k in ("dataset", "variant", "method", "metric", "run_id") if k in ctx},
    }
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    # one short append per call -- O_APPEND writes this small are atomic on
    # Linux, so concurrent runs from different worktrees can share a file
    with open(USAGE_DIR / f"{now:%Y-%m}.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def purpose_now() -> str:
    return current().get("purpose", "other")
