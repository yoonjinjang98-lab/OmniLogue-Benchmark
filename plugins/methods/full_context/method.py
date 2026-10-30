"""full_context -- family A baseline (BASELINE_METRIC_PLAN.md §4): dump the whole
conversation transcript into the prompt and ask the LLM the question directly.

No retrieval, no summarization, no memory mechanism -- this is the cheapest
possible baseline and the comparison floor every memory framework should beat.
It degrades (or blows the context window entirely) on very long conversations
(e.g. beam's ~10M-token ones); that's expected, not a bug to work around here --
that's exactly the gap running_summary/raptor/RAG baselines are meant to fill.

Works with any conversation shape that's either session-grouped (`.sessions`, a
dict of session_index -> list of turns with `.speaker`/`.text`, e.g. locomo) or
flat (`.turns`, a list of turns with `.speaker`/`.text`, e.g. molweni).
"""
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete


def _build_transcript(conversation) -> str:
    if hasattr(conversation, "sessions"):
        datetimes = getattr(conversation, "session_datetimes", {})
        lines = []
        for session_idx in sorted(conversation.sessions):
            dt = datetimes.get(session_idx)
            lines.append(f"[Session {session_idx}" + (f" -- {dt}]" if dt else "]"))
            lines.extend(f"{turn.speaker}: {turn.text}" for turn in conversation.sessions[session_idx])
        return "\n".join(lines)
    if hasattr(conversation, "turns"):
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in conversation.turns)
    raise ValueError(
        "full_context: conversation has neither `.sessions` (dict of session_idx -> turns) "
        "nor a flat `.turns` list -- can't build a transcript for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 512, **config):
        super().__init__(model=model, max_tokens=max_tokens, **config)
        self.model = model
        self.max_tokens = max_tokens
        self._transcript = None

    def prepare(self, conversation) -> None:
        self._transcript = _build_transcript(conversation)

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        prompt = (
            "Answer the question using only the information in the conversation below. "
            "Be concise and direct -- answer with just the fact asked for, no extra explanation.\n\n"
            f"### Conversation\n{self._transcript}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        return MethodOutput(
            answer=resp.text.strip(),
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )

    def reset(self) -> None:
        self._transcript = None
