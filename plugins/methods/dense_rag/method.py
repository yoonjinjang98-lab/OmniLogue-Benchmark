"""dense_rag -- family A baseline (BASELINE_METRIC_PLAN.md §4): dense (embedding)
retrieval over the conversation instead of stuffing the whole thing in context
or matching on lexical overlap.

Chunks the conversation one turn at a time (same chunking convention as
bm25_rag/full_context), embeds every chunk once via
dialogue_benchmarks.llm.embed() (text-embedding-3-small by default), and at
answer time embeds the question and retrieves the top-k chunks by cosine
similarity. The comparison point against bm25_rag: does semantic similarity
retrieve anything plain lexical matching misses (paraphrase, coreference,
synonymy)?

Works with any conversation shape that's either session-grouped (`.sessions`, a
dict of session_index -> list of turns with `.speaker`/`.text`, e.g. locomo) or
flat (`.turns`, a list of turns with `.speaker`/`.text`, e.g. molweni).

Embedding the whole conversation once per conversation is real API spend (one
batched embed() call), unlike bm25_rag's free local indexing -- that one-time
cost is charged to the first answer() after prepare() (tokens/cost) and zero
for every later question against the same conversation, same convention as
running_summary. Embedding the question itself is a genuine per-question cost
and is added to every answer() call directly.
"""
import math

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed


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
        "dense_rag: conversation has neither `.sessions` (dict of session_idx -> turns) "
        "nor a flat `.turns` list -- can't extract retrievable chunks for this dataset's shape"
    )


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        top_k: int = 10,
        max_tokens: int = 512,
        **config,
    ):
        super().__init__(model=model, embed_model=embed_model, top_k=top_k, max_tokens=max_tokens, **config)
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._chunks = None
        self._vectors = None
        self._ingest_tokens = 0
        self._ingest_cost = 0.0
        self._ingest_charged = True

    def prepare(self, conversation) -> None:
        self._chunks = _extract_chunks(conversation)
        self._vectors = None
        self._ingest_tokens = 0
        self._ingest_cost = 0.0
        if self._chunks:
            resp = embed([text for _, text in self._chunks], model=self.embed_model)
            self._vectors = resp.vectors
            self._ingest_tokens = resp.tokens_in
            self._ingest_cost = resp.cost or 0.0
        self._ingest_charged = False

    def answer(self, item) -> MethodOutput:
        retrieved = []
        q_resp = None
        if self._chunks and self._vectors:
            q_resp = embed(item.question, model=self.embed_model)
            q_vector = q_resp.vectors[0]
            scores = [_cosine(q_vector, v) for v in self._vectors]
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

        ingest_tokens, ingest_cost = (0, 0.0) if self._ingest_charged else (self._ingest_tokens, self._ingest_cost)
        self._ingest_charged = True
        q_tokens = q_resp.tokens_in if q_resp is not None else 0
        q_cost = (q_resp.cost or 0.0) if q_resp is not None else 0.0
        q_latency = q_resp.latency if q_resp is not None else 0.0

        return MethodOutput(
            answer=resp.text.strip(),
            retrieved=retrieved,
            tokens=ingest_tokens + q_tokens + resp.tokens_in + resp.tokens_out,
            latency=q_latency + resp.latency,
            cost=ingest_cost + q_cost + (resp.cost or 0.0),
        )

    def reset(self) -> None:
        self._chunks = None
        self._vectors = None
        self._ingest_tokens = 0
        self._ingest_cost = 0.0
        self._ingest_charged = True
