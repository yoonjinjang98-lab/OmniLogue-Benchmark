"""Adapter for LongMemEval -- data/raw/longmemeval/longmemeval_{s_cleaned,m_cleaned,oracle}.json

Source shape: a list of 500 questions, each carrying its *own* haystack --
{question_id, question_type, question, answer, question_date, haystack_dates,
haystack_session_ids, haystack_sessions: [[{role, content, has_answer?}, ...]],
answer_session_ids}. So one source record becomes one conversation (its
haystack, as dated sessions) with exactly one question.

Loads the S split (~115K tokens / ~40 sessions per question) by default -- the
one the paper's headline numbers use. M (~500 sessions, 2.7GB) and oracle
(evidence sessions only) are selected with LONGMEMEVAL_SPLIT=m|oracle.

Abstention questions (question_id ending in "_abs", 30 of 500) are marked
metadata["answerable"] = False for abstention_accuracy. The question is
prefixed with the question date, as in the official evaluation prompt --
temporal-reasoning questions ("how many days ago ...") are unanswerable
without it.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dialogue_benchmarks.contracts import EvalItem

SPLIT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


@dataclass
class LongMemEvalTurn:
    speaker: str  # "user" / "assistant"
    text: str
    has_answer: bool = False


@dataclass
class LongMemEvalQA:
    question: str
    answer: Any
    category: str  # question_type
    question_date: str
    answerable: bool
    answer_session_ids: list = field(default_factory=list)


@dataclass
class LongMemEvalConversation:
    conv_id: str
    sessions: dict  # {session_index: [LongMemEvalTurn, ...]}, 1-based, chronological as given
    session_datetimes: dict  # {session_index: str}
    session_ids: dict  # {session_index: haystack_session_id}
    qa: list  # [LongMemEvalQA] -- exactly one


def load(raw_dir: Path) -> list:
    split = os.environ.get("LONGMEMEVAL_SPLIT", "s")
    with open(raw_dir / SPLIT_FILES[split]) as f:
        raw = json.load(f)

    out = []
    for rec in raw:
        sessions, datetimes, ids = {}, {}, {}
        for i, (sess, date, sid) in enumerate(
            zip(rec["haystack_sessions"], rec["haystack_dates"], rec["haystack_session_ids"]), start=1
        ):
            sessions[i] = [
                LongMemEvalTurn(speaker=t["role"], text=t["content"], has_answer=bool(t.get("has_answer")))
                for t in sess
            ]
            datetimes[i] = date
            ids[i] = sid
        out.append(
            LongMemEvalConversation(
                conv_id=f"longmemeval_{split}_{rec['question_id']}",
                sessions=sessions,
                session_datetimes=datetimes,
                session_ids=ids,
                qa=[
                    LongMemEvalQA(
                        question=rec["question"],
                        answer=str(rec["answer"]),
                        category=rec["question_type"],
                        question_date=rec["question_date"],
                        answerable=not rec["question_id"].endswith("_abs"),
                        answer_session_ids=list(rec.get("answer_session_ids", [])),
                    )
                ],
            )
        )
    return out


def eval_items(conversation: LongMemEvalConversation) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=f"(Current date: {qa.question_date}) {qa.question}",
            reference=qa.answer,
            metadata={
                "category": qa.category if qa.answerable else "abstention",
                "question_type": qa.category,
                "answerable": qa.answerable,
                "evidence": qa.answer_session_ids,
            },
        )
        for qa in conversation.qa
    ]
