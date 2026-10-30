"""bm25_rag -- family A baseline (BASELINE_METRIC_PLAN.md §4): sparse (lexical)
retrieval over the conversation instead of stuffing the whole thing in context.

Chunks the conversation one turn at a time, indexes the chunks with BM25
(rank_bm25), retrieves the top-k turns most relevant to each question, and
answers from just those. The point of this baseline is to answer "does a real
memory framework beat plain lexical retrieval?" -- if it doesn't, the framework
isn't earning its complexity.

Works with any conversation shape that's either session-grouped (`.sessions`, a
dict of session_index -> list of turns with `.speaker`/`.text`, e.g. locomo) or
flat (`.turns`, a list of turns with `.speaker`/`.text`, e.g. molweni).
"""
from rank_bm25 import BM25Plus

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete


def _extract_chunks(conversation) -> list:
    """Returns [(label, text), ...], one retrievable chunk per turn, in
    chronological order."""
    if hasattr(conversation, "sessions"):
        chunks = []
        for session_idx in sorted(conversation.sessions):
            for turn in conversation.sessions[session_idx]:
                chunks.append((f"session {session_idx} -- {turn.speaker}", turn.text))
        return chunks
    if hasattr(conversation, "turns"):
        return [(turn.speaker, turn.text) for turn in conversation.turns]
    raise ValueError(
        "bm25_rag: conversation has neither `.sessions` (dict of session_idx -> turns) "
        "nor a flat `.turns` list -- can't extract retrievable chunks for this dataset's shape"
    )


def _tokenize(text: str) -> list:
    return text.lower().split()


class Method(_MethodBase):
    def __init__(self, model: str = "claude-sonnet-5", top_k: int = 10, max_tokens: int = 512, **config):
        super().__init__(model=model, top_k=top_k, max_tokens=max_tokens, **config)
        self.model = model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._chunks = None
        self._bm25 = None

    def prepare(self, conversation) -> None:
        self._chunks = _extract_chunks(conversation)
        self._bm25 = BM25Plus([_tokenize(text) for _, text in self._chunks]) if self._chunks else None

    def answer(self, item) -> MethodOutput:
        retrieved = []
        if self._chunks and self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(item.question))
            k = min(self.top_k, len(self._chunks))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            top_indices.sort()  # restore chronological order for the prompt
            retrieved = [f"{self._chunks[i][0]}: {self._chunks[i][1]}" for i in top_indices]

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        prompt = (
            "Answer the question using only the information in the retrieved excerpts below "
            "(they may be incomplete relative to the full conversation). Be concise and "
            "direct -- answer with just the fact asked for, no extra explanation.\n\n"
            f"### Retrieved excerpts\n{chr(10).join(retrieved)}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        return MethodOutput(
            answer=resp.text.strip(),
            retrieved=retrieved,
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )

    def reset(self) -> None:
        self._chunks = None
        self._bm25 = None
