"""Keep tests from writing to the real paid-API usage ledger (DATA_ROOT/usage/)
-- it's grant-settlement evidence, and mocked provider calls in test_llm.py
would otherwise show up in it as real spend."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_usage_ledger(tmp_path, monkeypatch):
    from dialogue_benchmarks import usage

    monkeypatch.setattr(usage, "USAGE_DIR", tmp_path / "usage")
