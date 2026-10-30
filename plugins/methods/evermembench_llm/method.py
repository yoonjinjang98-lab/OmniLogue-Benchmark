"""evermembench_llm -- family D baseline (BASELINE_METRIC_PLAN.md §4): wraps
evermembench's own "LLM system" baseline (full dialogue as context, no
memory system) from https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic
/ https://github.com's EverMemBench eval code (Apache-2.0 -- see the dataset's
HF card front matter) instead of reimplementing generic full_context against
this dataset.

Ported verbatim from eval/src/adapters/llm_adapter.py:_format_dialogue_as_context
and eval/config/prompts.yaml's `llm_answer` section (used only when the paper's
own harness is run with `--system llm`, i.e. exactly the comparison point this
baseline is for):
- dialogue formatting: `=== Date: {date} ===` / `[{group_name}]` /
  `[HH:MM:SS] {speaker}: {content}` -- so retrieval-free "just read the whole
  thing" numbers are comparable to upstream's own reported LLM baseline
- the exact multiple_choice/open_ended prompt templates for the "LLM system"
  path (distinct from their memory-system prompts, which expect retrieved
  memories rather than a full transcript)
- `_parse_mc_answer()` from eval/src/core/answerer.py: their exact
  letter-extraction regex, since a mismatch there would look like a model
  accuracy difference instead of a parsing difference

Deliberately NOT ported: the memory-system adapters (mem0/zep/memos/
memobase/evermemos -- family C, out of scope for now) and their own
`llm_judge` scoring prompt/loop (eval/src/core/evaluator.py) -- this repo's
Method/Metric contracts keep answering and scoring separate; score this
baseline's answers with qa_em/qa_f1/qa_llm_judge (or a future mc_accuracy
metric for the multiple_choice items) instead.

No dataset adapter for evermembench exists yet (plugins/datasets/**, A's
lane), so this hasn't been smoke-tested end-to-end. Expects a conversation
shaped like evermembench's own `Dataset` (`.days`: list of day objects with
`.date` and `.groups`, a dict of group_name -> list of message objects with
`.timestamp`/`.speaker`/`.content`) if a future adapter preserves that shape;
falls back to this repo's generic `.sessions`/`.turns` convention otherwise
(losing the per-group/timestamp structure, but at least stays usable).
"""
import re

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete

# Verbatim from eval/config/prompts.yaml's `llm_answer` section.
_MC_PROMPT = """You are a rigorous question-answering assistant. You will be given a very long dialogue transcript and one multiple-choice question with four options. Your task is to choose the single best answer (A/B/C/D) based ONLY on the dialogue.

Rules:
1. Use only information explicitly stated in the dialogue or directly entailed by it. Do NOT use outside knowledge, assumptions, or guesses beyond the dialogue.
2. If multiple options seem plausible, choose the one most strongly and directly supported by the dialogue.
3. If the dialogue does not provide enough information to be certain, you MUST still pick one option (A/B/C/D). Choose the option that is least inconsistent with the dialogue.
4. Do NOT output any reasoning, explanation, punctuation, or extra text.

Output format (must be followed exactly):
Output ONLY a single uppercase letter: A or B or C or D

[DIALOGUE]
{context}

[QUESTION]
{question}

[OPTIONS]
{options}
"""

_OE_PROMPT = """You are a rigorous question-answering assistant. You will be given a very long dialogue transcript and one open-ended question. Your task is to answer the question using ONLY the information in the dialogue.

Rules:
1. Use only information explicitly stated in the dialogue or directly entailed by it. Do NOT use outside knowledge, assumptions, or speculation.
2. If the dialogue does not contain enough information to answer, output exactly: NOT ENOUGH INFORMATION
3. Keep the answer concise, specific, and aligned with the dialogue. Do not add commentary.
4. Do NOT output any reasoning steps, explanations, or extra text beyond the final answer.

Output format (must be followed exactly):
Output ONLY the answer text (or exactly: NOT ENOUGH INFORMATION)

[DIALOGUE]
{context}

[QUESTION]
{question}
"""


def _format_time(ts) -> str:
    try:
        return ts.strftime("%H:%M:%S")
    except AttributeError:
        return str(ts)


def _build_dialogue_context(conversation) -> str:
    """Verbatim port of llm_adapter.py:_format_dialogue_as_context when the
    conversation carries evermembench's own `.days` shape; falls back to this
    repo's generic `.sessions`/`.turns` convention otherwise."""
    if hasattr(conversation, "days"):
        lines = []
        for day in conversation.days:
            lines.append(f"=== Date: {day.date} ===")
            lines.append("")
            for group_name in sorted(day.groups.keys()):
                lines.append(f"[{group_name}]")
                for msg in day.groups[group_name]:
                    lines.append(f"[{_format_time(msg.timestamp)}] {msg.speaker}: {msg.content}")
                lines.append("")
            lines.append("")
        return "\n".join(lines)
    if hasattr(conversation, "sessions"):
        lines = []
        for session_idx in sorted(conversation.sessions):
            lines.append(f"[Session {session_idx}]")
            lines.extend(f"{turn.speaker}: {turn.text}" for turn in conversation.sessions[session_idx])
        return "\n".join(lines)
    if hasattr(conversation, "turns"):
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in conversation.turns)
    raise ValueError(
        "evermembench_llm: conversation has neither `.days` (evermembench's own raw shape), "
        "`.sessions`, nor `.turns` -- can't build a dialogue context for this dataset's shape"
    )


def _format_options(options: list) -> str:
    """Evermembench's own QAItem.options are already pre-lettered ('A. text',
    'B. text', ...); letter anything that isn't, so this also works against a
    plain list of option strings."""
    formatted = []
    for i, opt in enumerate(options):
        if re.match(r"^[A-Da-d][.):]", opt.strip()):
            formatted.append(opt)
        else:
            formatted.append(f"{chr(ord('A') + i)}. {opt}")
    return "\n".join(formatted)


def _parse_mc_answer(response: str) -> str:
    """Verbatim from eval/src/core/answerer.py:_parse_mc_answer."""
    response = (response or "").strip().upper()
    if not response:
        return "[EMPTY]"
    if len(response) == 1 and response in "ABCD":
        return response
    match = re.search(r"\b([ABCD])[.):,\s]", response)
    if match:
        return match.group(1)
    match = re.search(r"(?:answer|choice|option|select)[:\s]+([ABCD])\b", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if response[0] in "ABCD" and (len(response) == 1 or not response[1].isalpha()):
        return response[0]
    if response[-1] in "ABCD" and (len(response) == 1 or not response[-2].isalpha()):
        return response[-1]
    return "[INVALID]"


class Method(_MethodBase):
    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 512, **config):
        super().__init__(model=model, max_tokens=max_tokens, **config)
        self.model = model
        self.max_tokens = max_tokens
        self._context = None

    def prepare(self, conversation) -> None:
        self._context = _build_dialogue_context(conversation)

    def answer(self, item) -> MethodOutput:
        if item.options:
            prompt = _MC_PROMPT.format(context=self._context, question=item.question, options=_format_options(item.options))
            resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
            answer = _parse_mc_answer(resp.text)
        else:
            prompt = _OE_PROMPT.format(context=self._context, question=item.question)
            resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
            answer = resp.text.strip() if resp.text else "[EMPTY]"

        return MethodOutput(
            answer=answer,
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )

    def reset(self) -> None:
        self._context = None
