"""qa_llm_judge calls dialogue_benchmarks.llm.complete() for every score() --
these tests monkeypatch that call so no API key/network is needed, mirroring
tests/test_llm.py's monkeypatch style for the temperature regression test.
"""
from dialogue_benchmarks import llm
from dialogue_benchmarks.contracts import EvalItem
from dialogue_benchmarks.metrics.registry import load_metric_fn

score = load_metric_fn("qa_llm_judge")


def _fake_response(text):
    return llm.LLMResponse(text=text, model="google/gemma-4-31B-it")


def test_correct_verdict_scores_one(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _fake_response("CORRECT"))

    item = EvalItem(conversation_id="c1", question="What color is the sky?", reference="blue")
    assert score(item, "the sky is blue") == 1.0


def test_incorrect_verdict_scores_zero(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _fake_response("INCORRECT"))

    item = EvalItem(conversation_id="c1", question="What color is the sky?", reference="blue")
    assert score(item, "the sky is green") == 0.0


def test_prompt_includes_question_reference_and_prediction(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured["prompt"] = prompt
        return _fake_response("CORRECT")

    monkeypatch.setattr(llm, "complete", fake_complete)

    item = EvalItem(conversation_id="c1", question="What color is the sky?", reference="blue")
    score(item, "the sky is blue")

    assert "What color is the sky?" in captured["prompt"]
    assert "blue" in captured["prompt"]
    assert "the sky is blue" in captured["prompt"]
