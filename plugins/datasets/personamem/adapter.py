"""Adapter for PersonaMem-v2 (text track) --
data/raw/personamem/benchmark/text/benchmark.csv + data/chat_history_32k/*.json

Source shape: benchmark.csv has one row per question (5000 rows over 200
personas); each row points at its persona's chat history JSON
(`chat_history_32k_link`, {metadata, chat_history: [{role, content}, ...]})
and carries `user_query` (a {role, content} dict serialized as a Python
literal), `correct_answer`, and three `incorrect_answers`.

One conversation per chat-history file (all rows sharing it become its
questions), so a memory method ingests each 32K history once. The leading
system message is dropped: it spells out the full persona profile verbatim,
which would turn an *implicit*-persona benchmark into a lookup. Options are
the 4 answers shuffled with a per-question seed (stable across runs and
machines); reference is the correct answer's text and
metadata["answer_index"] its position, for mc_accuracy.

The 128K histories are selected with PERSONAMEM_CONTEXT=128k.
"""
import ast
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dialogue_benchmarks.contracts import EvalItem


@dataclass
class PersonaMemTurn:
    speaker: str  # "user" / "assistant"
    text: str


@dataclass
class PersonaMemQA:
    question: str
    answer: str
    options: list
    answer_index: int
    category: str  # pref_type
    metadata: dict = field(default_factory=dict)


@dataclass
class PersonaMemConversation:
    conv_id: str
    persona_id: int
    turns: list  # [PersonaMemTurn, ...]
    qa: list  # [PersonaMemQA, ...]

    @property
    def sessions(self) -> dict:
        """The history as one undated session, for methods that only read the
        session-grouped shape (`.sessions`), e.g. smoke_test."""
        return {1: self.turns}


def _parse_literal(value):
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (ValueError, SyntaxError):
            continue
    return value


def _shuffled_options(qid: str, correct: str, incorrect: list) -> tuple:
    options = [correct] + list(incorrect)
    seed = int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    shuffled = [options[i] for i in order]
    return shuffled, order.index(0)


def load(raw_dir: Path) -> list:
    context = os.environ.get("PERSONAMEM_CONTEXT", "32k")
    link_col = f"chat_history_{context}_link"
    df = pd.read_csv(raw_dir / "benchmark" / "text" / "benchmark.csv")

    out = []
    for link, rows in df.groupby(link_col, sort=False):
        with open(raw_dir / link) as f:
            history = json.load(f)["chat_history"]
        turns = [PersonaMemTurn(speaker=m["role"], text=m["content"]) for m in history if m["role"] != "system"]
        qa = []
        for row_idx, row in rows.iterrows():
            query = _parse_literal(row["user_query"])
            question = query.get("content", str(query)) if isinstance(query, dict) else str(query)
            incorrect = _parse_literal(row["incorrect_answers"])
            options, answer_index = _shuffled_options(f"personamem_{row_idx}", row["correct_answer"], incorrect)
            qa.append(
                PersonaMemQA(
                    question=question,
                    answer=row["correct_answer"],
                    options=options,
                    answer_index=answer_index,
                    category=row["pref_type"],
                    metadata={
                        "topic_query": row["topic_query"],
                        "conversation_scenario": row["conversation_scenario"],
                        "updated": bool(row["updated"]),
                        "who": row["who"],
                        "row": int(row_idx),
                    },
                )
            )
        persona_id = int(rows["persona_id"].iloc[0])
        out.append(
            PersonaMemConversation(
                conv_id=f"personamem_{context}_{Path(link).stem}", persona_id=persona_id, turns=turns, qa=qa
            )
        )
    return out


def eval_items(conversation: PersonaMemConversation) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=qa.question,
            reference=qa.answer,
            options=qa.options,
            metadata={"category": qa.category, "answer_index": qa.answer_index, **qa.metadata},
        )
        for qa in conversation.qa
    ]
