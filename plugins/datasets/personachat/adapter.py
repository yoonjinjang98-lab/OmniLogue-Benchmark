"""Adapter for PersonaChat (ConvAI2 format) --
data/raw/personachat/data/{train,validation}-00000-of-00001.parquet

Source shape per row: {personality: [persona sentence, ...], utterances: [{history:
[text, ...], candidates: [distractor, ..., gold_response]}]}. The last entry in
`candidates` is the real next utterance; the rest are ConvAI2's distractor
candidates (kept as-is, not discarded, since some methods may want them).
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dialogue_benchmarks.contracts import EvalItem

SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
}


@dataclass
class PersonaChatTurn:
    history: list
    candidates: list  # candidates[-1] is the gold next utterance


@dataclass
class PersonaChatDialogue:
    conv_id: str
    split: str
    personality: list
    turns: list  # [PersonaChatTurn, ...]


def _load_split(raw_dir: Path, split: str) -> list:
    df = pd.read_parquet(raw_dir / "data" / SPLIT_FILES[split])
    out = []
    for i, row in df.iterrows():
        turns = [
            PersonaChatTurn(history=list(u["history"]), candidates=list(u["candidates"]))
            for u in row["utterances"]
        ]
        out.append(
            PersonaChatDialogue(
                conv_id=f"personachat_{split}_{i}",
                split=split,
                personality=list(row["personality"]),
                turns=turns,
            )
        )
    return out


def load_split(raw_dir: Path, split: str) -> list:
    return _load_split(raw_dir, split)


def load(raw_dir: Path) -> list:
    result = []
    for split in SPLIT_FILES:
        result.extend(_load_split(raw_dir, split))
    return result


def eval_items(conversation: PersonaChatDialogue) -> list:
    """task_type="dialogue": one item per source `turns` entry -- each already
    pins its own history/gold-response boundary, unlike msc's flat stream.
    `candidates` carries ConvAI2's distractor set through for methods that want
    it (e.g. a retrieval-style next-response-selection baseline)."""
    items = []
    for i, turn in enumerate(conversation.turns):
        items.append(
            EvalItem(
                conversation_id=conversation.conv_id,
                question="produce the next turn",
                reference=turn.candidates[-1] if turn.candidates else "",
                candidates=turn.candidates,
                metadata={
                    "persona": conversation.personality,
                    "history": list(turn.history),
                    "turn_index": i,
                },
            )
        )
    return items
