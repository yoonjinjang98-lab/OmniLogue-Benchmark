"""Adapter for MPCHAT next-response prediction (NRP) --
data/raw/mpchat/extracted/mpchat_nrp.json ({train, val, test} lists)

Source shape per dialogue (see data/raw/mpchat/code/DATASET.md): {subreddit,
messages: [text, ...], authors: [author per message], main_author,
candidate_personas: [{title, file_name, ...}] (<= 5 of the main author's
persona posts), nrp_candidate_responses: [[] or [gold, 99 distractors] per
turn] -- non-empty only on the main author's turns, val/test only}.

task_type="retrieval": one instance per (dialogue, main-author turn), so the
conversation a method sees is only the messages *before* that turn -- a
dialogue with two scoreable turns would otherwise leak the later one's gold
response into the earlier one's context. The source keeps the gold response
at candidate index 0; candidates are shuffled with a per-instance seed so a
method can't win by always picking 0. Persona is text-only: the persona
posts' titles (MPCHAT's images are distributed as URLs only, and this
harness is text-only).

Only the test split (1499 dialogues, 2301 instances) is loaded.
"""
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem


@dataclass
class MpchatTurn:
    speaker: str
    text: str


@dataclass
class MpchatInstance:
    conv_id: str
    subreddit: str
    main_author: str
    persona: list  # titles of the main author's candidate persona posts
    turns: list  # [MpchatTurn, ...] -- context before the target turn only
    candidates: list
    gold_index: int


def load(raw_dir: Path) -> list:
    with open(raw_dir / "extracted" / "mpchat_nrp.json") as f:
        dialogues = json.load(f)["test"]

    out = []
    for d_idx, d in enumerate(dialogues):
        persona = [p.get("title", "") for p in d.get("candidate_personas", [])]
        for t_idx, cands in enumerate(d.get("nrp_candidate_responses", [])):
            if not cands:
                continue
            conv_id = f"mpchat_test_{d_idx}_{t_idx}"
            order = list(range(len(cands)))
            random.Random(int(hashlib.md5(conv_id.encode()).hexdigest()[:8], 16)).shuffle(order)
            out.append(
                MpchatInstance(
                    conv_id=conv_id,
                    subreddit=d.get("subreddit", ""),
                    main_author=d["main_author"],
                    persona=persona,
                    turns=[MpchatTurn(speaker=a, text=m) for a, m in zip(d["authors"][:t_idx], d["messages"][:t_idx])],
                    candidates=[cands[i] for i in order],
                    gold_index=order.index(0),
                )
            )
    return out


def eval_items(conversation: MpchatInstance) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=f"Which candidate is {conversation.main_author}'s next message?",
            reference=conversation.gold_index,
            candidates=conversation.candidates,
            metadata={
                "history": [f"{t.speaker}: {t.text}" for t in conversation.turns],
                "persona": conversation.persona,
                "speaker": conversation.main_author,
                "category": conversation.subreddit,
            },
        )
    ]
