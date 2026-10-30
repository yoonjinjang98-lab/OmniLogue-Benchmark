"""ld_agent -- LD-Agent (Li et al., "Hello Again! LLM-powered Personalized Agent
for Long-term Dialogue", NAACL 2025, github.com/leolee99/LD-Agent, MIT): the
model-agnostic long-term dialogue framework, with its three modules driven by
whatever LLM `--model` names (llm.complete()):

- event memory: each earlier session is summarized (EventSummary prompt) and
  stored with its spaCy noun-lemma topics; per turn, the single stored
  memory with the best noun-overlap x time-decay score is retrieved
  ("<gap> ago, <summary>.", or "No relevant Memories.")
- personas: user / agent traits extracted turn by turn (PersonaExtraction
  prompt), accumulated across the agent's sessions
- response generator: LD-Agent's own role-play prompt over the current
  session's context, the retrieved memory and both trait lists

Prompts and the scoring rule are vendored verbatim (vendor/ld_agent_prompts.py).
Deviations, all forced by the harness shape rather than chosen:
- chromadb is dropped: with <= 4 stored session memories its top-n query
  returns every memory anyway, and the pick is decided by the noun-overlap
  score, which is reproduced exactly.
- one conversation object per target session (msc_longterm /
  conversation_chronicles), so prepare() rebuilds the memory of the earlier
  sessions every time -- identical prompts, so llm.complete()'s cache makes
  the repeats free.
- traits follow LD-Agent's own timing: extracted from sessions 2+ (session 1
  only initializes memory, as in its memory_bank_init), user traits up to
  the current inquiry, agent traits from the agent's *previous* turns.
- session time gaps: Conversation Chronicles' labels ("A few months after")
  are converted to seconds; MSC's HF release has none, so a 1-week gap is
  assumed.
"""
import importlib.util
import re
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete

_spec = importlib.util.spec_from_file_location(
    "_ld_agent_prompts", Path(__file__).resolve().parent / "vendor" / "ld_agent_prompts.py"
)
_P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_P)

_NLP = {}
_UNIT_SECONDS = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2592000, "year": 31536000}
_QUANTITY = {"a": 1, "an": 1, "one": 1, "couple": 2, "two": 2, "few": 3, "three": 3, "several": 4, "many": 5}
DEFAULT_GAP_S = 7 * 86400


def _nouns(text: str) -> list:
    if "nlp" not in _NLP:
        import spacy

        _NLP["nlp"] = spacy.load("en_core_web_sm")
    return sorted({t.lemma_ for t in _NLP["nlp"](text) if t.pos_ == "NOUN"})


def gap_seconds(label) -> float:
    """'A few months after' -> ~3 months in seconds; unknown -> 1 week."""
    if not label or str(label).strip().lower() == "start":
        return 0.0 if label else DEFAULT_GAP_S
    words = re.findall(r"[a-z]+", str(label).lower())
    for i, w in enumerate(words):
        unit = w.rstrip("s")
        if unit in _UNIT_SECONDS:
            qty = next((_QUANTITY[x] for x in reversed(words[:i]) if x in _QUANTITY), 1)
            return float(qty * _UNIT_SECONDS[unit])
    return DEFAULT_GAP_S


def _extra(model: str) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": False}} if model.startswith("google/gemma") else {}


class Method(_MethodBase):
    def __init__(self, model: str = "google/gemma-4-31B-it", max_tokens: int = 128, **config):
        super().__init__(model=model, max_tokens=max_tokens, **config)
        self.model = model
        self.max_tokens = max_tokens
        self._reset_state()

    def _reset_state(self):
        self._memories = []  # [{"time", "summary", "topics"}]
        self._now = 0.0
        self._agent = None
        self._user_traits, self._agent_traits = [], []
        self._traits_done = 0  # history lines already mined for traits

    def _llm(self, system: str, user: str, max_tokens: int = None) -> str:
        resp = complete(user, model=self.model, system=system, max_tokens=max_tokens or self.max_tokens, **_extra(self.model))
        return (resp.text or "").strip()

    def _extract_trait(self, sentence: str, into: list) -> None:
        trait = self._llm(*_P.persona_extraction_prompts(sentence), max_tokens=64)
        if "NO_TRAIT" not in trait and len(trait) > 3:
            into.append(trait)

    def _mine_traits(self, lines: list) -> None:
        """LD-Agent's traits_update over (inquiry, response) turns: user
        lines feed user traits, agent lines agent traits."""
        for line in lines:
            speaker, _, text = line.partition(": ")
            if not text:
                continue
            self._extract_trait(text, self._agent_traits if speaker == self._agent else self._user_traits)

    def prepare(self, conversation) -> None:
        self._reset_state()
        sessions = getattr(conversation, "sessions", None) or {}
        gaps = getattr(conversation, "session_datetimes", {}) or {}
        self._agent = getattr(conversation, "agent", None)
        for idx in sorted(sessions):
            self._now += gap_seconds(gaps.get(idx)) if idx > 1 else 0.0
            turns = sessions[idx]
            if self._agent is None:
                speakers = list(dict.fromkeys(t.speaker for t in turns))
                self._agent = speakers[1] if len(speakers) > 1 else None
            context = "\n".join(f"(line {i + 1}) {t.speaker}: {t.text}." for i, t in enumerate(turns))
            usr = next((t.speaker for t in turns if t.speaker != self._agent), "User")
            summary = self._llm(*_P.event_summary_prompts(usr, self._agent or "Agent", context))
            self._memories.append({"time": self._now, "summary": summary, "topics": _nouns(" ".join(t.text for t in turns))})
            if idx > 1:
                self._mine_traits([f"{t.speaker}: {t.text}" for t in turns])
        # the target session itself starts after its own gap
        self._now += gap_seconds(getattr(conversation, "target_interval", None))

    def _retrieve(self, inquiry: str) -> str:
        query = _nouns(inquiry)
        best, best_score = None, 0.0
        for m in self._memories:
            score, overlap = _P.relevance_score(query, m["topics"], self._now - m["time"])
            if overlap > 0 and score >= best_score:
                best, best_score = m, score
        if best is None:
            return "No relevant Memories."
        return f"{_P.convert_seconds_to_full_time(self._now - best['time'])} ago, {best['summary']}."

    def answer(self, item) -> MethodOutput:
        meta = item.metadata or {}
        if meta.get("speaker"):
            self._agent = meta["speaker"]
        history = meta.get("history") or []
        # traits from this session's turns not yet mined, except the current
        # inquiry, which LD-Agent mines before answering it (user side only)
        self._mine_traits(history[self._traits_done :])
        self._traits_done = len(history)

        inquiry_line = history[-1] if history else ""
        usr_name, _, inquiry = inquiry_line.partition(": ")
        usr_name = usr_name if inquiry else "User"
        inquiry = inquiry or inquiry_line
        context_lines = [f"[TURN {i}] : {line}." for i, line in enumerate(history[-31:-1])]
        context_lines.append(f"In this turn, {usr_name} said: {inquiry}.")
        sys_prompt, user_prompt = _P.response_prompts(
            self._agent or "Agent",
            usr_name,
            inquiry,
            "\n".join(context_lines),
            self._retrieve(inquiry),
            "\n".join(self._user_traits),
            "\n".join(self._agent_traits),
        )
        text = self._llm(sys_prompt, user_prompt)
        text = re.sub(r"^\s*RESPONSE:\s*", "", text, flags=re.I).strip().split("\n")[0].strip()
        return MethodOutput(answer=text)

    def reset(self) -> None:
        self._reset_state()
