"""Adapter for EngramaBench (full_v1) --
data/raw/engramabench/repo/datasets/full_v1/{conversations,queries}_<persona>.json
(a git clone of github.com/julianacunadc/engramabench)

Source shape: per persona, conversations_<p>.json is a list of
{conversation_id, persona_id, space_id, timestamp, depth_index, messages:
[{role, content, message_id}]}, and queries_<p>.json a list of {query_id,
question, expected_answer (str or list), answer_type (entity / date / number
/ set / short_span / abstain / insight), task_family, adversarial, ...}.

One conversation per persona -- its 20 conversations become chronologically
ordered sessions (dated, labeled with their space) -- with that persona's
queries as its questions. A list-valued expected_answer is joined with ", "
so string metrics can score it; the original query record is kept whole in
metadata["engrama_query"] for the engrama_score metric, which re-runs the
benchmark's own typed scorer. Adversarial queries are answerable=False.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

PERSONAS = ("priya", "kai", "carlos", "sam", "diane")


@dataclass
class EngramaTurn:
    speaker: str  # "user" / "assistant"
    text: str
    message_id: str


@dataclass
class EngramaQuery:
    question: str
    answer: str
    category: str  # task_family
    answerable: bool
    raw: dict = field(default_factory=dict)


@dataclass
class EngramaPersona:
    conv_id: str
    persona_id: str
    sessions: dict  # {session_index: [EngramaTurn, ...]}
    session_datetimes: dict  # {session_index: "<timestamp> [space]"}
    qa: list  # [EngramaQuery, ...]


def load(raw_dir: Path) -> list:
    base = raw_dir / "repo" / "datasets" / "full_v1"
    out = []
    for persona in PERSONAS:
        with open(base / f"conversations_{persona}.json") as f:
            convs = sorted(json.load(f), key=lambda c: (c["timestamp"], c.get("depth_index", 0)))
        with open(base / f"queries_{persona}.json") as f:
            queries = json.load(f)
        sessions, datetimes = {}, {}
        for i, c in enumerate(convs, start=1):
            sessions[i] = [
                EngramaTurn(speaker=m["role"], text=m["content"], message_id=m.get("message_id", "")) for m in c["messages"]
            ]
            datetimes[i] = f"{c['timestamp']} [{c.get('space_id', '')}]"
        qa = []
        for q in queries:
            ans = q["expected_answer"]
            qa.append(
                EngramaQuery(
                    question=q["question"],
                    answer=", ".join(map(str, ans)) if isinstance(ans, list) else str(ans),
                    category=q["task_family"],
                    answerable=not q.get("adversarial", False),
                    raw=q,
                )
            )
        out.append(
            EngramaPersona(
                conv_id=f"engramabench_{persona}", persona_id=persona, sessions=sessions, session_datetimes=datetimes, qa=qa
            )
        )
    return out


def eval_items(conversation: EngramaPersona) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=q.question,
            reference=q.answer,
            metadata={
                "category": q.category,
                "answerable": q.answerable,
                "answer_type": q.raw.get("answer_type"),
                "engrama_query": q.raw,
            },
        )
        for q in conversation.qa
    ]
