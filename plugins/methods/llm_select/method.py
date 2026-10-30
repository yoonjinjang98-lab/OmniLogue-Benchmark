"""llm_select -- zero-shot LLM pick over item.candidates.

Works for any retrieval item: response selection (candidates = messages) and
addressee recognition (candidates = speaker labels) alike, since the item's
own `question` says what is being chosen. Returns the picked index as a
one-element JSON ranking (metrics/parsing.py format); an unparseable reply
becomes "[]", which the metrics score as a random guess.
"""
import json
import re

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete


def _extra(model: str) -> dict:
    # gemma's thinking mode is off unless asked; keep it explicit so the cache
    # key matches qa_llm_judge's calls to the same model
    return {"chat_template_kwargs": {"enable_thinking": False}} if model.startswith("google/gemma") else {}


class Method(_MethodBase):
    def __init__(self, model: str = "google/gemma-4-31B-it", context_turns: int = 15, max_tokens: int = 8, **config):
        super().__init__(model=model, context_turns=context_turns, max_tokens=max_tokens, **config)
        self.model = model
        self.context_turns = context_turns
        self.max_tokens = max_tokens

    def answer(self, item) -> MethodOutput:
        candidates = list(item.candidates or [])
        if not candidates:
            return MethodOutput(answer="[]")
        meta = item.metadata or {}
        parts = []
        if meta.get("persona"):
            parts.append("### Persona of the next speaker\n" + "\n".join(f"- {p}" for p in meta["persona"]))
        history = meta.get("history", [])[-self.context_turns :]
        parts.append("### Conversation so far\n" + ("\n".join(history) if history else "(empty)"))
        parts.append("### Candidates\n" + "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates)))
        parts.append(f"### Task\n{item.question}\nAnswer with only the number of the chosen candidate, e.g. 3.")
        resp = complete("\n\n".join(parts), model=self.model, max_tokens=self.max_tokens, **_extra(self.model))
        m = re.search(r"\d+", resp.text)
        pick = int(m.group()) if m and int(m.group()) < len(candidates) else None
        return MethodOutput(
            answer=json.dumps([pick] if pick is not None else []),
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )
