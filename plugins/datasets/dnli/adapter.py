"""Adapter for Dialogue NLI (dnli) --
data/raw/dnli/xksteven_dialogue_nli_{train,validation,test}.jsonl

NOT a conversational dataset: dnli is premise/hypothesis/label triples derived
from PersonaChat, used as a persona-consistency NLI resource (see config/meta.json
notes). This adapter returns DnliPair records, not Conversation-shaped objects --
forcing it into a dialogue schema would misrepresent what it actually is.

As its own task_type="nli" benchmark, eval_items() scores only the test
split: each pair is its own "conversation" with one item -- question is the
hypothesis, metadata["premise"] the premise, reference the DNLI label
(positive/negative/neutral, which nli_accuracy maps onto
entailment/contradiction/neutral).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dialogue_benchmarks.contracts import EvalItem

SPLIT_FILES = {
    "train": "xksteven_dialogue_nli_train.jsonl",
    "validation": "xksteven_dialogue_nli_validation.jsonl",
    "test": "xksteven_dialogue_nli_test.jsonl",
}


@dataclass
class DnliPair:
    id: str
    split: str
    premise: str
    hypothesis: str
    label: str  # "positive" / "negative" / "neutral"
    dtype: Optional[str]


def _load_split(raw_dir: Path, split: str) -> list:
    pairs = []
    with open(raw_dir / SPLIT_FILES[split]) as f:
        for line in f:
            d = json.loads(line)
            pairs.append(
                DnliPair(
                    id=d["id"],
                    split=split,
                    premise=d["premise"],
                    hypothesis=d["hypothesis"],
                    label=d["label"],
                    dtype=d.get("dtype"),
                )
            )
    return pairs


def load(raw_dir: Path) -> list:
    result = []
    for split in SPLIT_FILES:
        result.extend(_load_split(raw_dir, split))
    return result


def eval_items(conversation: DnliPair) -> list:
    if conversation.split != "test":
        return []
    return [
        EvalItem(
            conversation_id=conversation.id,
            question=conversation.hypothesis,
            reference=conversation.label,
            metadata={"premise": conversation.premise, "category": conversation.dtype},
        )
    ]
