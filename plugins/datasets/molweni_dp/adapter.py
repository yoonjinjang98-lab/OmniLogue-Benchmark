"""Molweni discourse parsing -- data/raw/molweni/DP/test.json, the
discourse-parsing half of molweni (the `molweni` dataset covers the MRC/QA
half). Raw files are reused from molweni's raw dir rather than duplicated.

Source shape: a list of dialogues, each {edus: [{text, speaker}], relations:
[{x, y, type}]} -- EDU y responds to EDU x, type one of 15 STAC relations
(Comment, Clarification_question, QAP, Continuation, Result, ...).

task_type="discourse_parsing": one item per dialogue, predict its full
discourse graph. item.reference is the gold edge list [[x, y, type], ...];
metadata carries the numbered EDUs and the relation label set for a prompted
method. Scored by discourse_relation_f1 (link F1 / link+relation F1).

Only the standard test split (500 dialogues) is loaded.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

RELATION_TYPES = [
    "Comment",
    "Clarification_question",
    "QAP",
    "Continuation",
    "Result",
    "Acknowledgement",
    "Explanation",
    "Q-Elab",
    "Narration",
    "Elaboration",
    "Correction",
    "Conditional",
    "Contrast",
    "Background",
    "Parallel",
    "Alternation",
]


@dataclass
class MolweniDpTurn:
    idx: int
    speaker: str
    text: str


@dataclass
class MolweniDpDialogue:
    conv_id: str
    turns: list  # [MolweniDpTurn, ...] -- EDUs
    relations: list  # [[x, y, type], ...]


def load(raw_dir: Path) -> list:
    with open(raw_dir.parent / "molweni" / "DP" / "test.json") as f:
        raw = json.load(f)
    return [
        MolweniDpDialogue(
            conv_id=f"molweni_dp_test_{i}",
            turns=[MolweniDpTurn(idx=j, speaker=e["speaker"], text=e["text"]) for j, e in enumerate(d["edus"])],
            relations=[[r["x"], r["y"], r["type"]] for r in d.get("relations", [])],
        )
        for i, d in enumerate(raw)
    ]


def eval_items(conversation: MolweniDpDialogue) -> list:
    if not conversation.relations:
        return []
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=(
                "Parse the discourse structure of this conversation: for every EDU after the first, give the "
                "earlier EDU it responds to and the relation type, as a JSON list of [x, y, type] triples "
                "(EDU y responds to EDU x, 0-based)."
            ),
            reference=conversation.relations,
            metadata={
                "edus": [f"{t.idx}. {t.speaker}: {t.text}" for t in conversation.turns],
                "relation_types": RELATION_TYPES,
                "n_edus": len(conversation.turns),
            },
        )
    ]
