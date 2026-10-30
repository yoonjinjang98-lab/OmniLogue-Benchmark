"""MSC as a *long-term* generation benchmark -- the protocol LD-Agent (NAACL
2025) and HAHT evaluate on, over the same raw files as `msc`
(data/raw/msc/, reused via raw_dir's sibling).

`msc` treats every session as an independent dialogue, so nothing from
earlier sessions is ever needed -- no long-term memory is exercised. Here the
test split's 501 five-session dialogues (exactly LD-Agent's 501 test samples)
become one conversation per (dialogue, target session 2..5):

- `.sessions` holds only the *earlier* sessions (1..s-1) -- the long-term
  memory a method may ingest in prepare(). The target session is never in
  it, and later sessions don't exist on the object at all, so nothing leaks.
- items: every "Speaker 2" turn of the target session (LD-Agent's agent
  role), with that session's gold turns so far as metadata["history"],
  Speaker 2's persona sentences for persona metrics, and
  category = "session_<s>" so the runner reports the per-session breakdown
  the papers tabulate.

The HF release has no inter-session time gaps, so session_datetimes are None.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

AGENT = "Speaker 2"


@dataclass
class MscTurn:
    speaker: str
    text: str


@dataclass
class MscLongTermConversation:
    conv_id: str
    dialogue_id: int
    target_session: int  # 2..5 (1-based)
    sessions: dict  # {1..s-1: [MscTurn, ...]} -- memory only
    session_datetimes: dict
    target_turns: list  # [MscTurn, ...]
    agent_persona: list = field(default_factory=list)


def load(raw_dir: Path) -> list:
    return load_split(raw_dir, "test")


def load_split(raw_dir: Path, split: str) -> list:
    """Any split (train/validation have dialogues with 3-5 sessions); used by
    training-based baselines (plugins/methods/llm_sft/train.py) for their
    train split."""
    path = raw_dir.parent / "msc" / f"nayohan_multi_session_chat_{split}.jsonl"
    by_dialogue = defaultdict(dict)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            turns = [MscTurn(speaker=s, text=t) for s, t in zip(d.get("speaker", []), d.get("dialogue", []))]
            by_dialogue[d["dialoug_id"]][d["session_id"] + 1] = (turns, d.get("persona2", []))

    out = []
    for dlg in sorted(by_dialogue):
        sessions = by_dialogue[dlg]
        for s in sorted(sessions):
            if s < 2:
                continue
            target_turns, persona2 = sessions[s]
            out.append(
                MscLongTermConversation(
                    conv_id=f"msc_lt_{split}_{dlg}_s{s}",
                    dialogue_id=dlg,
                    target_session=s,
                    sessions={i: sessions[i][0] for i in sorted(sessions) if i < s},
                    session_datetimes={i: None for i in sorted(sessions) if i < s},
                    target_turns=target_turns,
                    agent_persona=list(persona2),
                )
            )
    return out


def eval_items(conversation: MscLongTermConversation) -> list:
    turns = conversation.target_turns
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question="produce the next turn",
            reference=turns[i].text,
            metadata={
                "category": f"session_{conversation.target_session}",
                "history": [f"{t.speaker}: {t.text}" for t in turns[:i]],
                "speaker": AGENT,
                "persona": conversation.agent_persona,
                "turn_index": i,
                "session": conversation.target_session,
            },
        )
        for i in range(1, len(turns))
        if turns[i].speaker == AGENT
    ]
