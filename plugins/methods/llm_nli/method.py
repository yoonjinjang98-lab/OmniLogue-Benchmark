"""llm_nli -- zero-shot 3-way NLI over (metadata["premise"], item.question)."""
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete

_PROMPT = """Decide the relationship between two statements made by the same person in a conversation.

Statement 1: {premise}
Statement 2: {hypothesis}

If Statement 1 is true, is Statement 2 necessarily true (ENTAILMENT), necessarily false or inconsistent (CONTRADICTION), or neither (NEUTRAL)?
Respond with exactly one word: ENTAILMENT, CONTRADICTION, or NEUTRAL."""


def _extra(model: str) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": False}} if model.startswith("google/gemma") else {}


class Method(_MethodBase):
    def __init__(self, model: str = "google/gemma-4-31B-it", **config):
        super().__init__(model=model, **config)
        self.model = model

    def answer(self, item) -> MethodOutput:
        prompt = _PROMPT.format(premise=(item.metadata or {}).get("premise", ""), hypothesis=item.question)
        resp = complete(prompt, model=self.model, max_tokens=5, **_extra(self.model))
        return MethodOutput(
            answer=resp.text.strip(), tokens=resp.tokens_in + resp.tokens_out, latency=resp.latency, cost=resp.cost
        )
