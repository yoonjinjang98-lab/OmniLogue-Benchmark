"""Adapter for LoCoMo -- data/raw/locomo/locomo10.json

Source shape: a list of 10 conversations, each with speaker_a/speaker_b, per-session
turns (session_N: [{speaker, dia_id, text}]) and a `qa` list ({question, answer,
evidence, category}) used for the memory-QA evaluation task.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LocomoTurn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class LocomoQA:
    question: str
    answer: Any
    evidence: list
    category: int


@dataclass
class LocomoConversation:
    conv_id: str
    speaker_a: str
    speaker_b: str
    sessions: dict  # {session_index: [LocomoTurn, ...]}
    session_datetimes: dict  # {session_index: str}
    qa: list  # [LocomoQA, ...]


def load(raw_dir: Path) -> list:
    with open(raw_dir / "locomo10.json") as f:
        raw = json.load(f)

    conversations = []
    for item in raw:
        conv = item["conversation"]
        sessions = {}
        session_datetimes = {}
        i = 1
        while f"session_{i}" in conv:
            sessions[i] = [
                LocomoTurn(speaker=t["speaker"], dia_id=t["dia_id"], text=t["text"])
                for t in conv[f"session_{i}"]
            ]
            session_datetimes[i] = conv.get(f"session_{i}_date_time")
            i += 1

        qa = [
            LocomoQA(
                question=q["question"],
                answer=q.get("answer"),
                evidence=q.get("evidence", []),
                category=q.get("category"),
            )
            for q in item.get("qa", [])
        ]

        conversations.append(
            LocomoConversation(
                conv_id=item["sample_id"],
                speaker_a=conv["speaker_a"],
                speaker_b=conv["speaker_b"],
                sessions=sessions,
                session_datetimes=session_datetimes,
                qa=qa,
            )
        )
    return conversations
