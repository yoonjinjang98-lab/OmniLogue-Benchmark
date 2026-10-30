"""MPCHAT multi-party, persona-grounded response *generation* -- same
instances as `mpchat` (NRP test split, one per (dialogue, main-author turn),
data/raw/mpchat/ reused via raw_dir's sibling), but the method writes the
main author's next message instead of picking it from 100 candidates.

task_type="dialogue": metadata carries "history", "speaker" (the main
author) and "persona" (titles of their persona posts -- text-only), so
persona_consistency applies; the reference is the gold message (candidate 0
in the source).
"""
import importlib.util
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

_SIBLING = Path(__file__).resolve().parent.parent / "mpchat" / "adapter.py"
_spec = importlib.util.spec_from_file_location("_mpchat_adapter_for_gen", _SIBLING)
_mpchat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mpchat)


def load(raw_dir: Path) -> list:
    return _mpchat.load(raw_dir.parent / "mpchat")


def load_split(raw_dir: Path, split: str) -> list:
    """train / val / test. test is exactly `load()`; train and val carry no NRP
    candidates (MPCHAT ships those for val/test only), so instances there are
    every main-author turn after the first message, with the gold message as
    the single "candidate" -- all a generation-training baseline needs."""
    if split == "test":
        return load(raw_dir)
    import json

    with open(raw_dir.parent / "mpchat" / "extracted" / "mpchat_nrp.json") as f:
        dialogues = json.load(f)[split]
    out = []
    for d_idx, d in enumerate(dialogues):
        persona = [p.get("title", "") for p in d.get("candidate_personas", [])]
        for t_idx in range(1, len(d["messages"])):
            if d["authors"][t_idx] != d["main_author"]:
                continue
            out.append(
                _mpchat.MpchatInstance(
                    conv_id=f"mpchat_{split}_{d_idx}_{t_idx}",
                    subreddit=d.get("subreddit", ""),
                    main_author=d["main_author"],
                    persona=persona,
                    turns=[_mpchat.MpchatTurn(speaker=a, text=m) for a, m in zip(d["authors"][:t_idx], d["messages"][:t_idx])],
                    candidates=[d["messages"][t_idx]],
                    gold_index=0,
                )
            )
    return out


def eval_items(conversation) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question="produce the next turn",
            reference=conversation.candidates[conversation.gold_index],
            metadata={
                "history": [f"{t.speaker}: {t.text}" for t in conversation.turns],
                "speaker": conversation.main_author,
                "persona": conversation.persona,
                "category": conversation.subreddit,
            },
        )
    ]
