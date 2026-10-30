"""persona_prompt -- prompted next-turn generation with the speaker's persona.

Prompt = [earlier sessions, if the conversation has `.sessions`] + [persona
sentences, if item.metadata has them and use_persona] + item.metadata
["history"] -> one next turn, written as metadata["speaker"] (and to
metadata["addressee"], when a multi-party dataset gives one). `no_persona` is this same method with
use_persona=False, so the ablation differs in exactly one thing.
"""
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete


def _extra(model: str) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": False}} if model.startswith("google/gemma") else {}


def _memory(conversation, max_chars: int) -> str:
    sessions = getattr(conversation, "sessions", None)
    if not sessions:
        return ""
    datetimes = getattr(conversation, "session_datetimes", {}) or {}
    lines = []
    for idx in sorted(sessions):
        dt = datetimes.get(idx)
        lines.append(f"[Session {idx}" + (f" -- {dt}]" if dt else "]"))
        lines.extend(f"{t.speaker}: {t.text}" for t in sessions[idx])
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "google/gemma-4-31B-it",
        use_persona: bool = True,
        use_memory: bool = True,
        max_tokens: int = 64,
        max_memory_chars: int = 40000,
        **config,
    ):
        super().__init__(
            model=model,
            use_persona=use_persona,
            use_memory=use_memory,
            max_tokens=max_tokens,
            max_memory_chars=max_memory_chars,
            **config,
        )
        self.model = model
        self.use_persona = use_persona
        self.use_memory = use_memory
        self.max_tokens = max_tokens
        self.max_memory_chars = max_memory_chars
        self._memory = ""

    def prepare(self, conversation) -> None:
        self._memory = _memory(conversation, self.max_memory_chars) if self.use_memory else ""

    def answer(self, item) -> MethodOutput:
        meta = item.metadata or {}
        parts = []
        if self._memory:
            parts.append("### Earlier sessions between these speakers\n" + self._memory)
        persona = meta.get("persona") or []
        if self.use_persona and persona:
            parts.append("### Your persona\n" + "\n".join(f"- {p}" for p in persona))
        history = meta.get("history") or []
        parts.append("### Current conversation\n" + ("\n".join(history) if history else "(start of conversation)"))
        who = f" as {meta['speaker']}" if meta.get("speaker") else ""
        # multi-party generation (ubuntu_irc_gen): the addressee is a given
        # input in that setting, not something the model has to infer
        if meta.get("addressee"):
            who += f", replying to {meta['addressee']}"
        parts.append(
            f"### Task\nWrite the next turn of the current conversation{who}. Reply with only that one turn -- "
            "a single natural, conversational message, no speaker label, no explanation."
        )
        resp = complete("\n\n".join(parts), model=self.model, max_tokens=self.max_tokens, **_extra(self.model))
        text = resp.text.strip().split("\n")[0].strip()
        if meta.get("speaker") and text.startswith(f"{meta['speaker']}:"):
            text = text[len(meta["speaker"]) + 1 :].strip()
        return MethodOutput(answer=text, tokens=resp.tokens_in + resp.tokens_out, latency=resp.latency, cost=resp.cost)

    def reset(self) -> None:
        self._memory = ""
