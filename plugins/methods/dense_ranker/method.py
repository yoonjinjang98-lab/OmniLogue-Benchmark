"""dense_ranker -- zero-shot bi-encoder response selection over item.candidates.

Query = the last `context_turns` context turns (+ persona sentences if the
item has them), embedded once; candidates embedded in one batched call; rank
by cosine similarity. Embeddings go through llm.embed(), so they're cached on
disk and shared like every other model call.
"""
import json
import math

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import embed


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _strip_speaker(line: str) -> str:
    return line.split(": ", 1)[1] if ": " in line else line


class Method(_MethodBase):
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2", context_turns: int = 3, use_persona: bool = True, **config):
        super().__init__(embed_model=embed_model, context_turns=context_turns, use_persona=use_persona, **config)
        self.embed_model = embed_model
        self.context_turns = context_turns
        self.use_persona = use_persona

    def answer(self, item) -> MethodOutput:
        candidates = [str(c) for c in (item.candidates or [])]
        if not candidates:
            return MethodOutput(answer="[]")
        meta = item.metadata or {}
        history = [_strip_speaker(h) for h in meta.get("history", [])][-self.context_turns :]
        query = " ".join(history) or item.question
        if self.use_persona and meta.get("persona"):
            query += " " + " ".join(meta["persona"])
        q = embed([query], model=self.embed_model)
        c = embed(candidates, model=self.embed_model)
        scores = [_cos(q.vectors[0], v) for v in c.vectors]
        ranking = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return MethodOutput(answer=json.dumps(ranking), latency=q.latency + c.latency, cost=q.cost + c.cost)
