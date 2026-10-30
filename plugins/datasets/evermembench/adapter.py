"""Adapter for EverMemBench-Dynamic --
data/raw/evermembench/0[1-5]/{dialogue.json,qa_0N.json}
(https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic -- its own
dataset card's `dialogues`/`qars` HF configs load exactly these paths; the
top-level `EverMemBench_*.json` / `dataset/` files in this raw dir are an
older/alternate dump and not what this adapter reads).

Source shape:
- `0N/dialogue.json`: list of day entries `{topic_id, date, dialogues:
  {"Group 1": [{speaker, time, dialogue, message_index}], "Group 2": [...] or
  null, ...}}`, spanning ~250 days per topic.
- `0N/qa_0N.json`: list of QA items `{topic_id, id, Q, A, R, options}`.
  `options` is a dict `{"A": text, ...}` or null (open-ended); `R` is reference
  evidence (date/group/message_index pointers into `dialogue.json`), kept
  as-is in metadata rather than resolved into text.

One `Conversation` = one topic (01-05), matching
plugins/methods/evermembench_llm's expected `.days` shape (list of day objects
with `.date` and `.groups`, a dict of group_name -> messages with
`.timestamp`/`.speaker`/`.content`).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dialogue_benchmarks.contracts import EvalItem

TOPIC_IDS = ("01", "02", "03", "04", "05")


@dataclass
class EvermemMessage:
    speaker: str
    timestamp: str
    content: str
    message_index: int


@dataclass
class EvermemDay:
    date: str
    groups: dict  # group_name -> [EvermemMessage, ...]


@dataclass
class EvermemQA:
    id: str
    question: str
    answer: str
    options: Optional[dict]  # {"A": text, ...} or None (open-ended)
    evidence: list  # raw `R` entries, unresolved (see module docstring)


@dataclass
class EvermemTopic:
    conv_id: str
    topic_id: str
    days: list  # [EvermemDay, ...]
    qa: list = field(default_factory=list)  # [EvermemQA, ...]


def _load_days(raw_dir: Path, topic_id: str) -> list:
    with open(raw_dir / topic_id / "dialogue.json") as f:
        raw = json.load(f)
    days = []
    for entry in raw:
        groups = {}
        for group_name, messages in (entry.get("dialogues") or {}).items():
            if not messages:
                continue
            groups[group_name] = [
                EvermemMessage(
                    speaker=m["speaker"],
                    timestamp=m["time"],
                    content=m["dialogue"],
                    message_index=m["message_index"],
                )
                for m in messages
            ]
        days.append(EvermemDay(date=entry["date"], groups=groups))
    return days


def _load_qa(raw_dir: Path, topic_id: str) -> list:
    with open(raw_dir / topic_id / f"qa_{topic_id}.json") as f:
        raw = json.load(f)
    return [
        EvermemQA(id=q["id"], question=q["Q"], answer=q["A"], options=q.get("options"), evidence=q.get("R", []))
        for q in raw
    ]


def load(raw_dir: Path) -> list:
    return [
        EvermemTopic(
            conv_id=f"evermembench_{topic_id}",
            topic_id=topic_id,
            days=_load_days(raw_dir, topic_id),
            qa=_load_qa(raw_dir, topic_id),
        )
        for topic_id in TOPIC_IDS
    ]


def _category(question_id: str) -> str:
    """Verbatim from code/tools/analyze_results.py:parse_question_id's "combined"
    category -- id prefixes are the dataset's own 9 subtasks (e.g. "MA_U",
    "P_Skill", "F_TP"), not something this adapter invents."""
    parts = question_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else "Unknown"


def eval_items(conversation: EvermemTopic) -> list:
    """Not the generic `.qa` fallback: multiple-choice items need `options` as
    a plain lettered list on EvalItem, converted from the source's
    `{"A": text, ...}` dict (the fallback has no such conversion, and would
    otherwise pass the raw dict through where a list is expected), and
    `category` needs deriving from the id prefix (see `_category`) rather
    than the fallback's fixed attr allowlist, which has no such derivation."""
    items = []
    for q in conversation.qa:
        options = [f"{letter}. {text}" for letter, text in sorted(q.options.items())] if q.options else None
        items.append(
            EvalItem(
                conversation_id=conversation.conv_id,
                question=q.question,
                reference=q.answer,
                options=options,
                metadata={"id": q.id, "category": _category(q.id), "evidence": q.evidence},
            )
        )
    return items
