"""Adapter for Multi-Session Chat (MSC) --
data/raw/msc/nayohan_multi_session_chat_{train,validation,test}.jsonl

Source shape per line: {dataset, dialoug_id, session_id, persona1, persona2,
dialogue: [utterance text, ...], speaker: ["Speaker 1"/"Speaker 2", ...] aligned
1:1 with dialogue}. No addressee/reply_to labels exist in the source -- earlier
work in this project fabricated those for MSC as a fallback and later had to
correct it (see config/meta.json notes), so this adapter does not include them.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

SPLIT_FILES = {
    "train": "nayohan_multi_session_chat_train.jsonl",
    "validation": "nayohan_multi_session_chat_validation.jsonl",
    "test": "nayohan_multi_session_chat_test.jsonl",
}


@dataclass
class MscSession:
    conv_id: str
    split: str
    dialogue_id: int
    session_id: int
    persona1: list
    persona2: list
    speakers: list  # aligned 1:1 with utterances
    utterances: list


def _load_split(raw_dir: Path, split: str) -> list:
    sessions = []
    with open(raw_dir / SPLIT_FILES[split]) as f:
        for line in f:
            d = json.loads(line)
            sessions.append(
                MscSession(
                    conv_id=f"msc_{split}_{d['dialoug_id']}_{d['session_id']}",
                    split=split,
                    dialogue_id=d["dialoug_id"],
                    session_id=d["session_id"],
                    persona1=d.get("persona1", []),
                    persona2=d.get("persona2", []),
                    speakers=d.get("speaker", []),
                    utterances=d.get("dialogue", []),
                )
            )
    return sessions


def load(raw_dir: Path) -> list:
    result = []
    for split in SPLIT_FILES:
        result.extend(_load_split(raw_dir, split))
    return result


def eval_items(conversation: MscSession) -> list:
    """task_type="dialogue": one item per utterance the flat `.dialogue` stream
    can predict from prior context (index 0 has no history, so it's skipped).
    Persona is picked by whichever speaker is due to talk next -- persona1 for
    "Speaker 1", persona2 for anyone else -- matching plugins/methods/paa's
    fallback inference for this same shape.
    """
    speakers = conversation.speakers or []
    utterances = conversation.utterances or []
    items = []
    for i in range(1, len(utterances)):
        speaker = speakers[i] if i < len(speakers) else None
        persona = conversation.persona1 if speaker == "Speaker 1" else conversation.persona2
        items.append(
            EvalItem(
                conversation_id=conversation.conv_id,
                question="produce the next turn",
                reference=utterances[i],
                metadata={
                    "persona": persona,
                    "history": list(utterances[:i]),
                    "turn_index": i,
                    "speaker": speaker,
                },
            )
        )
    return items
