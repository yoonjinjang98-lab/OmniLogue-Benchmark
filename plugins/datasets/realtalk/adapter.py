"""Adapter for REALTALK -- data/raw/realtalk/repo/data/Chat_<n>_<A>_<B>.json
(a git clone of github.com/danny911kr/REALTALK)

Source shape per file: LoCoMo-style -- {name: {speaker_1, speaker_2},
session_<i>: [{clean_text, speaker, date_time, dia_id, img_url, ...}],
session_<i>_date_time, events_session_<i>, qa: [{question, answer, evidence,
category}]}. Unlike LoCoMo, the conversations are real 21-day messaging
between two people, not LLM-generated. Images (img_url) are dropped -- the
harness is text-only -- but their presence is kept as a "[shares an image]"
marker when a turn has no text, so turn structure stays intact.

Known source quirk, deliberately left as-is: in Chat_1 the speaker labels are
chat handles ("Emi", "elise") while the two participants address each other --
and the QA refers to them -- by real names (Kate, Emily), so a method has to
infer that "Emi" is Kate from the dialogue itself (a retrieval baseline
mostly can't). Other files only have spelling variants between labels and
questions (Akib/Aqib, Muhhamed/Muhammad).

Only the top-level Chat_*.json files are the REALTALK data; data/locomo/ in
the same repo is the authors' reformatted LoCoMo copy and is ignored.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RealtalkTurn:
    speaker: str
    dia_id: str
    text: str
    date_time: str


@dataclass
class RealtalkQA:
    question: str
    answer: Any
    evidence: list
    category: int


@dataclass
class RealtalkConversation:
    conv_id: str
    speaker_a: str
    speaker_b: str
    sessions: dict  # {session_index: [RealtalkTurn, ...]}
    session_datetimes: dict  # {session_index: str}
    events: dict  # {session_index: {agent_a: [...], agent_b: [...]}}
    qa: list  # [RealtalkQA, ...]


def load(raw_dir: Path) -> list:
    out = []
    for path in sorted((raw_dir / "repo" / "data").glob("Chat_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(path) as f:
            d = json.load(f)
        sessions, datetimes, events = {}, {}, {}
        i = 1
        while f"session_{i}" in d:
            sessions[i] = [
                RealtalkTurn(
                    speaker=t["speaker"],
                    dia_id=t.get("dia_id", ""),
                    text=t.get("clean_text") or ("[shares an image]" if t.get("img_url") else ""),
                    date_time=t.get("date_time", ""),
                )
                for t in d[f"session_{i}"]
            ]
            datetimes[i] = d.get(f"session_{i}_date_time")
            events[i] = d.get(f"events_session_{i}", {})
            i += 1
        out.append(
            RealtalkConversation(
                conv_id=f"realtalk_{path.stem}",
                speaker_a=d["name"]["speaker_1"],
                speaker_b=d["name"]["speaker_2"],
                sessions=sessions,
                session_datetimes=datetimes,
                events=events,
                qa=[
                    RealtalkQA(
                        question=q["question"],
                        answer=str(q["answer"]),
                        evidence=q.get("evidence", []),
                        category=q.get("category"),
                    )
                    for q in d.get("qa", [])
                ],
            )
        )
    return out
