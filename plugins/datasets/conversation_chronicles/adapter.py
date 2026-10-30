"""Adapter for Conversation Chronicles -- data/raw/conversation_chronicles/{train,valid,test}.jsonl

Source shape per line (one 5-session episode): {dataID, relationship (e.g.
"Neighbors", "Employee and Boss"), time_interval: [5 strings, "Start" first],
summary: [5 strings], {first..fifth}_session_dialogue: [utterance, ...],
{first..fifth}_session_speakers: [speaker label, ...]}.

task_type="dialogue", following LD-Agent's (NAACL 2025) multi-session
protocol, the same as `msc_longterm`: one conversation per (episode, target
session 2..5) whose `.sessions` holds only the earlier sessions (long-term
memory for prepare() -- nothing from the target or later sessions leaks),
with session_datetimes = the source's time gap before each session ("A few
months after", ...). Items are the agent's turns in the target session --
the episode's second speaker (e.g. "Neighbors B"), LD-Agent's SPEAKER_2 --
with the session's gold turns so far as metadata["history"]; category =
"session_<s>" for the per-session breakdown, relationship kept in metadata.

Test split only, and by default only its first 501 episodes -- the size of
LD-Agent's CC test set (it mirrors MSC's 501); CC_MAX_EPISODES=0 loads all
20K (~960K items).
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

_ORDINALS = ("first", "second", "third", "fourth", "fifth")


@dataclass
class ChroniclesTurn:
    speaker: str
    text: str


@dataclass
class ChroniclesConversation:
    conv_id: str
    episode_id: str
    relationship: str
    target_session: int  # 2..5
    agent: str  # the episode's second speaker
    sessions: dict  # {1..s-1: [ChroniclesTurn, ...]} -- memory only
    session_datetimes: dict  # {1..s-1: time gap before that session}
    target_turns: list
    target_interval: str


def _turns(d: dict, ordinal: str) -> list:
    texts = d.get(f"{ordinal}_session_dialogue", [])
    speakers = d.get(f"{ordinal}_session_speakers", [])
    return [ChroniclesTurn(speaker=s, text=t) for s, t in zip(speakers, texts)]


def load(raw_dir: Path) -> list:
    return load_split(raw_dir, "test", int(os.environ.get("CC_MAX_EPISODES", "501")))


def load_split(raw_dir: Path, split: str, max_episodes: int = 0) -> list:
    """`split` in train/valid/test; max_episodes=0 means all. Used by
    training-based baselines for the train split."""
    out = []
    with open(raw_dir / f"{split}.jsonl") as f:
        for n, line in enumerate(f):
            if max_episodes and n >= max_episodes:
                break
            d = json.loads(line)
            intervals = d.get("time_interval", [])
            sessions = {i + 1: _turns(d, o) for i, o in enumerate(_ORDINALS)}
            first_speakers = list(dict.fromkeys(t.speaker for t in sessions[1]))
            agent = first_speakers[1] if len(first_speakers) > 1 else (first_speakers[0] if first_speakers else "")
            for s in range(2, 6):
                out.append(
                    ChroniclesConversation(
                        conv_id=f"cc_{split}_{d['dataID']}_s{s}",
                        episode_id=d["dataID"],
                        relationship=d.get("relationship", ""),
                        target_session=s,
                        agent=agent,
                        sessions={i: sessions[i] for i in range(1, s)},
                        session_datetimes={i: intervals[i - 1] if i - 1 < len(intervals) else None for i in range(1, s)},
                        target_turns=sessions[s],
                        target_interval=intervals[s - 1] if s - 1 < len(intervals) else None,
                    )
                )
    return out


def eval_items(conversation: ChroniclesConversation) -> list:
    turns = conversation.target_turns
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question="produce the next turn",
            reference=turns[i].text,
            metadata={
                "category": f"session_{conversation.target_session}",
                "history": [f"{t.speaker}: {t.text}" for t in turns[:i]],
                "speaker": conversation.agent,
                "relationship": conversation.relationship,
                "time_interval": conversation.target_interval,
                "turn_index": i,
                "session": conversation.target_session,
            },
        )
        for i in range(1, len(turns))
        if turns[i].speaker == conversation.agent
    ]
