"""Usage ledger (dialogue_benchmarks/usage.py): billable OpenAI calls are
recorded once, tagged with the runner's context, priced with the cached-input
discount, and routed to a per-purpose key when one is set. Uses a fake
openai.OpenAI -- no network. conftest.py points the ledger at tmp_path."""
import json

import pytest


def _fake_openai(monkeypatch, captured):
    import openai

    class Usage:
        prompt_tokens = 1000
        completion_tokens = 100
        prompt_tokens_details = type("D", (), {"cached_tokens": 400})()

    class Response:
        choices = [type("C", (), {"message": type("M", (), {"content": "CORRECT"})()})()]
        usage = Usage()

    class Completions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return Response()

    class FakeClient:
        def __init__(self, api_key=None, **_):
            captured["api_key"] = api_key
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)


def test_openai_call_is_ledgered_with_context_and_cached_price(monkeypatch):
    from dialogue_benchmarks import llm, usage

    captured = {}
    _fake_openai(monkeypatch, captured)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-default")
    monkeypatch.setenv("OPENAI_API_KEY_JUDGE", "sk-judge")

    with usage.context(purpose="judge", dataset="locomo", method="full_context", metric="qa_llm_judge_gpt"):
        resp = llm.complete("q", model="gpt-4o-mini", max_tokens=5, use_cache=False)

    # 600 uncached @0.15 + 400 cached @0.075 + 100 out @0.6, per 1M tokens
    assert resp.cost == pytest.approx((600 * 0.15 + 400 * 0.075 + 100 * 0.6) / 1e6)
    assert captured["api_key"] == "sk-judge"

    lines = [json.loads(l) for f in usage.USAGE_DIR.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["purpose"] == "judge" and entry["metric"] == "qa_llm_judge_gpt" and entry["dataset"] == "locomo"
    assert entry["cached_tokens"] == 400 and entry["cost_usd"] == pytest.approx(resp.cost)


def test_cache_hits_are_not_ledgered(monkeypatch, tmp_path):
    from dialogue_benchmarks import llm, usage

    captured = {}
    _fake_openai(monkeypatch, captured)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-default")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")

    llm.complete("same prompt", model="gpt-4o-mini", max_tokens=5)
    llm.complete("same prompt", model="gpt-4o-mini", max_tokens=5)

    lines = [l for f in usage.USAGE_DIR.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert len(lines) == 1


def test_reasoning_models_get_max_completion_tokens(monkeypatch):
    from dialogue_benchmarks import llm

    captured = {}
    _fake_openai(monkeypatch, captured)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-default")

    llm.complete("q", model="gpt-5-mini", max_tokens=50, use_cache=False)
    assert captured["kwargs"]["max_completion_tokens"] == 50
    assert "max_tokens" not in captured["kwargs"] and "temperature" not in captured["kwargs"]
