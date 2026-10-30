"""llm_mc -- full-context lettered multiple choice.

Transcript building mirrors full_context (`.sessions` or flat `.turns`);
the answer is a single option letter, which mc_accuracy parses directly.
Items without options fall back to a free-form answer.
"""
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete
from dialogue_benchmarks.metrics.parsing import option_letter


def _build_transcript(conversation) -> str:
    if hasattr(conversation, "sessions"):
        datetimes = getattr(conversation, "session_datetimes", {}) or {}
        lines = []
        for idx in sorted(conversation.sessions):
            dt = datetimes.get(idx)
            lines.append(f"[Session {idx}" + (f" -- {dt}]" if dt else "]"))
            lines.extend(f"{t.speaker}: {t.text}" for t in conversation.sessions[idx])
        return "\n".join(lines)
    if hasattr(conversation, "turns"):
        return "\n".join(f"{t.speaker}: {t.text}" for t in conversation.turns)
    raise ValueError("llm_mc: conversation has neither `.sessions` nor `.turns`")


def _extra(model: str) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": False}} if model.startswith("google/gemma") else {}


class Method(_MethodBase):
    def __init__(self, model: str = "google/gemma-4-31B-it", **config):
        super().__init__(model=model, **config)
        self.model = model
        self._transcript = None

    def prepare(self, conversation) -> None:
        self._transcript = _build_transcript(conversation)

    def answer(self, item) -> MethodOutput:
        if item.options:
            choices = "\n".join(f"({option_letter(i)}) {o}" for i, o in enumerate(item.options))
            task = (
                f"The user now asks: {item.question}\n\nWhich response best fits this particular user, given "
                f"everything the conversation reveals about them?\n{choices}\n\nAnswer with only the letter."
            )
            max_tokens = 3
        else:
            task = f"Question: {item.question}\nAnswer concisely."
            max_tokens = 128
        prompt = f"### Conversation history\n{self._transcript}\n\n### Task\n{task}"
        resp = complete(prompt, model=self.model, max_tokens=max_tokens, **_extra(self.model))
        return MethodOutput(
            answer=resp.text.strip(), tokens=resp.tokens_in + resp.tokens_out, latency=resp.latency, cost=resp.cost
        )

    def reset(self) -> None:
        self._transcript = None
