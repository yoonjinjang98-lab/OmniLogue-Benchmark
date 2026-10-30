"""llm_dp -- zero-shot LLM dialogue discourse parsing.

Asks for one [x, y, relation] edge per EDU y >= 1 (x < y). Any EDU the reply
doesn't attach (or attaches to a later EDU) falls back to the previous-EDU
chain with "Comment", so output size always matches the gold's one-parent-
per-EDU convention and a malformed reply degrades to the chain baseline
rather than to zero.
"""
import json
import re

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete


def _extra(model: str) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": False}} if model.startswith("google/gemma") else {}


class Method(_MethodBase):
    def __init__(self, model: str = "google/gemma-4-31B-it", max_tokens: int = 1024, **config):
        super().__init__(model=model, max_tokens=max_tokens, **config)
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, item) -> MethodOutput:
        meta = item.metadata or {}
        edus = meta.get("edus", [])
        labels = meta.get("relation_types", [])
        prompt = (
            "Below is a multi-party chat split into numbered EDUs (elementary discourse units).\n\n"
            + "\n".join(edus)
            + "\n\nFor every EDU y from 1 to "
            + str(len(edus) - 1)
            + ", choose the single earlier EDU x (x < y) that it responds to, and the discourse relation, one of: "
            + ", ".join(labels)
            + ".\nReturn only a JSON list of [x, y, \"relation\"] triples, one per EDU y, nothing else."
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens, **_extra(self.model))
        parents = {}
        m = re.search(r"\[.*\]", resp.text, flags=re.S)
        if m:
            try:
                for e in json.loads(m.group()):
                    x, y, rel = int(e[0]), int(e[1]), str(e[2])
                    if 0 <= x < y < len(edus) and y not in parents:
                        parents[y] = [x, y, rel]
            except (ValueError, TypeError, IndexError, KeyError):
                pass
        edges = [parents.get(y, [y - 1, y, "Comment"]) for y in range(1, len(edus))]
        return MethodOutput(
            answer=json.dumps(edges), tokens=resp.tokens_in + resp.tokens_out, latency=resp.latency, cost=resp.cost
        )
