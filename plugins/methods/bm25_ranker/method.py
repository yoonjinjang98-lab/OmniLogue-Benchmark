"""bm25_ranker -- lexical response selection over item.candidates.

Query = the last `context_turns` context turns (+ persona sentences if the
item has them); documents = the candidates. Returns every candidate index,
best first, as a JSON list -- the ranking format recall_at_k/mrr parse (see
dialogue_benchmarks/metrics/parsing.py).
"""
import json
import re

from rank_bm25 import BM25Okapi

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def _strip_speaker(line: str) -> str:
    return line.split(": ", 1)[1] if ": " in line else line


class Method(_MethodBase):
    def __init__(self, context_turns: int = 3, use_persona: bool = True, **config):
        super().__init__(context_turns=context_turns, use_persona=use_persona, **config)
        self.context_turns = context_turns
        self.use_persona = use_persona

    def answer(self, item) -> MethodOutput:
        candidates = item.candidates or []
        if not candidates:
            return MethodOutput(answer="[]")
        meta = item.metadata or {}
        history = [_strip_speaker(h) for h in meta.get("history", [])][-self.context_turns :]
        query = " ".join(history)
        if self.use_persona and meta.get("persona"):
            query += " " + " ".join(meta["persona"])
        docs = [_tokenize(c) or ["<empty>"] for c in candidates]
        scores = BM25Okapi(docs).get_scores(_tokenize(query) or ["<empty>"])
        ranking = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return MethodOutput(answer=json.dumps(ranking))
