"""Adapter for Molweni -- data/raw/molweni/MRC(withDiscourse)/{train,dev,test}.json

Molweni also ships a separate Discourse-Parsing-only split (DP/) that has discourse
relations but no QA pairs; this adapter only covers the MRC(withDiscourse) format,
which is a superset (edus + discourse relations + QA) and is what the QA task_type
in config/meta.json refers to.

Source shape per dialogue: `edus` ([{text, speaker}]), `qas`
([{id, question, answers: [{text, answer_start}], is_impossible}]), `relations`
([{x, y, type}] -- y is the discourse-relation response to x; this is a real gold
label, unlike the addressee fallback used in some other adapters historically).
"""
import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "dev", "test")


@dataclass
class MolweniTurn:
    idx: int
    speaker: str
    text: str


@dataclass
class MolweniRelation:
    x: int
    y: int
    type: str


@dataclass
class MolweniQA:
    id: str
    question: str
    answers: list  # answer texts; empty if is_impossible
    is_impossible: bool


@dataclass
class MolweniDialogue:
    conv_id: str
    split: str
    turns: list  # [MolweniTurn, ...]
    relations: list  # [MolweniRelation, ...]
    qa: list  # [MolweniQA, ...]


def _load_split(raw_dir: Path, split: str) -> list:
    path = raw_dir / "MRC(withDiscourse)" / f"{split}.json"
    with open(path) as f:
        raw = json.load(f)
    dialogues = raw["data"]["dialogues"]

    out = []
    for i, d in enumerate(dialogues):
        turns = [
            MolweniTurn(idx=idx, speaker=e["speaker"], text=e["text"])
            for idx, e in enumerate(d["edus"])
        ]
        relations = [MolweniRelation(x=r["x"], y=r["y"], type=r["type"]) for r in d.get("relations", [])]
        qa = [
            MolweniQA(
                id=q["id"],
                question=q["question"],
                answers=[a["text"] for a in q.get("answers", [])],
                is_impossible=q.get("is_impossible", False),
            )
            for q in d.get("qas", [])
        ]
        out.append(
            MolweniDialogue(conv_id=f"molweni_{split}_{i}", split=split, turns=turns, relations=relations, qa=qa)
        )
    return out


def load(raw_dir: Path) -> list:
    result = []
    for split in SPLITS:
        result.extend(_load_split(raw_dir, split))
    return result
