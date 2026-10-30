"""Adapter for BEAM -- data/raw/beam/data/{100K,500K,1M}-00000-of-00001.parquet

Source shape per row: {conversation_id, conversation_seed, user_profile,
chat: [[{role, content, time_anchor, ...}, ...] per session/batch],
probing_questions: a Python-literal *string* of
{ability: [{question, <answer field>, difficulty, rubric, ...}, ...]}} for 10
memory abilities (abstention, contradiction_resolution, event_ordering,
information_extraction, instruction_following, knowledge_update,
multi_session_reasoning, preference_following, summarization,
temporal_reasoning).

The gold-answer field name differs per ability (answer / ideal_answer /
ideal_response / ideal_summary / expected_compliance) -- normalized to
`.answer` here, with the ability as `category` and the rubric kept in
metadata for an LLM judge. Abstention questions are marked answerable=False.
User turns carry a trailing " ->-> <batch>,<turn>" generation marker that
isn't part of the conversation; it's stripped.

Loads the 100K-token tier by default (20 conversations, 400 questions);
BEAM_TIER=500K|1M selects the larger ones (35 conversations each).
"""
import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dialogue_benchmarks.contracts import EvalItem

_ANSWER_FIELDS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance")
_MARKER = re.compile(r"\s*->->\s*\d+\s*,\s*\d+\s*$")


@dataclass
class BeamTurn:
    speaker: str  # "user" / "assistant"
    text: str


@dataclass
class BeamQA:
    question: str
    answer: str
    category: str
    answerable: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class BeamConversation:
    conv_id: str
    title: str
    sessions: dict  # {session_index: [BeamTurn, ...]}
    session_datetimes: dict  # {session_index: time anchor str}
    qa: list  # [BeamQA, ...]


def _answer_of(q: dict) -> str:
    for key in _ANSWER_FIELDS:
        if q.get(key):
            return str(q[key])
    return ""


def load(raw_dir: Path) -> list:
    tier = os.environ.get("BEAM_TIER", "100K")
    df = pd.read_parquet(raw_dir / "data" / f"{tier}-00000-of-00001.parquet")

    out = []
    for _, row in df.iterrows():
        sessions, datetimes = {}, {}
        for i, batch in enumerate(row["chat"], start=1):
            sessions[i] = [
                BeamTurn(speaker=m["role"], text=_MARKER.sub("", m["content"] or "")) for m in batch
            ]
            anchors = [m.get("time_anchor") for m in batch if m.get("time_anchor")]
            datetimes[i] = anchors[0] if anchors else None

        probing = ast.literal_eval(row["probing_questions"])
        qa = []
        for ability, questions in probing.items():
            for q in questions:
                qa.append(
                    BeamQA(
                        question=q["question"],
                        answer=_answer_of(q),
                        category=ability,
                        answerable=ability != "abstention",
                        metadata={"difficulty": q.get("difficulty"), "rubric": list(q.get("rubric") or [])},
                    )
                )
        out.append(
            BeamConversation(
                conv_id=f"beam_{tier}_{row['conversation_id']}",
                title=row["conversation_seed"].get("title", ""),
                sessions=sessions,
                session_datetimes=datetimes,
                qa=qa,
            )
        )
    return out


def eval_items(conversation: BeamConversation) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=qa.question,
            reference=qa.answer,
            metadata={"category": qa.category, "answerable": qa.answerable, **qa.metadata},
        )
        for qa in conversation.qa
    ]
