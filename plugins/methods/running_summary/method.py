"""running_summary -- family B baseline (BASELINE_METRIC_PLAN.md §4): MemoryBank-
style rolling summary instead of retrieval or a full transcript dump.

Walks the conversation one chunk at a time -- one chunk per session where the
dataset is session-grouped (`.sessions`, e.g. locomo), or fixed-size windows of
turns otherwise (`.turns`, e.g. molweni) -- updating a single running summary
after each chunk via one LLM call per chunk: "here's what's remembered so far,
here's the new chunk, produce the updated summary." Answering a question only
ever sees the final summary, never the raw transcript, so unlike full_context
this stays usable on conversations too long for any context window (e.g.
beam's ~10M-token ones) -- at the cost of whatever the summarization step
drops or gets wrong along the way. That's the comparison this baseline is for:
does a fancier memory framework actually retain more than a plain rolling
summary?

The one-time cost of building the summary is real API spend (one LLM call per
chunk), unlike full_context/bm25_rag's free-until-questioned prepare(). It's
charged to the first answer() after prepare() (tokens/cost), and to zero for
every subsequent question against the same conversation, so summing per-item
`cost`/`tokens` across a conversation's items gives the true total instead of
multiply-counting ingestion once per question.
"""
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete


def _extract_chunks(conversation, window: int) -> list:
    """Returns [(label, text), ...] chunks in chronological order -- one per
    session if the conversation is session-grouped, else fixed-size windows
    over flat turns."""
    if hasattr(conversation, "sessions"):
        chunks = []
        for session_idx in sorted(conversation.sessions):
            text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in conversation.sessions[session_idx])
            chunks.append((f"session {session_idx}", text))
        return chunks
    if hasattr(conversation, "turns"):
        turns = conversation.turns
        chunks = []
        for start in range(0, len(turns), window):
            block = turns[start : start + window]
            text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in block)
            chunks.append((f"turns {start}-{start + len(block) - 1}", text))
        return chunks
    raise ValueError(
        "running_summary: conversation has neither `.sessions` (dict of session_idx -> turns) "
        "nor a flat `.turns` list -- can't extract chunks for this dataset's shape"
    )


_UPDATE_PROMPT = (
    "You are maintaining a running summary of an ongoing conversation so it can be used "
    "later to answer questions without access to the raw transcript. Update the summary "
    "below to incorporate the new chunk: keep every fact, event, date, and stated "
    "preference from the existing summary that the new chunk doesn't contradict, and add "
    "what's new. Write dense, terse bullet points, not prose.\n\n"
    "### Existing summary\n{summary}\n\n"
    "### New chunk ({label})\n{text}\n\n"
    "### Updated summary"
)


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 512,
        summary_max_tokens: int = 1024,
        window: int = 20,
        **config,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            summary_max_tokens=summary_max_tokens,
            window=window,
            **config,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.summary_max_tokens = summary_max_tokens
        self.window = window
        self._summary = None
        self._ingest_tokens = 0
        self._ingest_cost = 0.0
        self._ingest_charged = True

    def prepare(self, conversation) -> None:
        summary = "(nothing yet)"
        tokens = 0
        cost = 0.0
        for label, text in _extract_chunks(conversation, self.window):
            prompt = _UPDATE_PROMPT.format(summary=summary, label=label, text=text)
            resp = complete(prompt, model=self.model, max_tokens=self.summary_max_tokens)
            summary = resp.text.strip()
            tokens += resp.tokens_in + resp.tokens_out
            cost += resp.cost or 0.0
        self._summary = summary
        self._ingest_tokens = tokens
        self._ingest_cost = cost
        self._ingest_charged = False

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        prompt = (
            "Answer the question using only the information in the summary below (it's a "
            "compressed summary of a longer conversation, so it may be incomplete). Be "
            "concise and direct -- answer with just the fact asked for, no extra "
            "explanation.\n\n"
            f"### Summary\n{self._summary}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)

        ingest_tokens, ingest_cost = (0, 0.0) if self._ingest_charged else (self._ingest_tokens, self._ingest_cost)
        self._ingest_charged = True

        return MethodOutput(
            answer=resp.text.strip(),
            retrieved=[self._summary],
            tokens=ingest_tokens + resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=ingest_cost + (resp.cost or 0.0),
        )

    def reset(self) -> None:
        self._summary = None
        self._ingest_tokens = 0
        self._ingest_cost = 0.0
        self._ingest_charged = True
