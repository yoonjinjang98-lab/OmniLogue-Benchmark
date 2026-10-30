"""Adapter for Ubuntu IRC multi-party response selection --
data/raw/ubuntu_irc/ubuntu_data_{train,valid,test}.json (JSON lines)

Source shape per line (Hu et al. 2019 GSN / Ouchi & Tsuboi 2016 Ubuntu IRC
format): {context: [utterance, ...], ctx_spk: [speaker id per utterance],
ctx_adr: [addressee id per utterance, -1 = none], relation_at: [[reply, to],
...] (gold reply-to edges), answer: the next utterance, ans_spk, ans_adr,
ans_idx}.

task_type="retrieval", response selection: the source ships no negative
candidates, so each item gets 9 distractors drawn from *other* test responses
with a per-instance seed (the standard R_10@k setup, deterministic across
runs). item.candidates is the shuffled 10-way pool and item.reference the
gold index; metadata keeps the speaker/addressee ids for methods that use
them. Addressee recognition over the same instances is the separate
`ubuntu_irc_addressee` dataset (its own task, its own metrics).

Only the test split (5000 instances) is loaded.
"""
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

N_CANDIDATES = 10


@dataclass
class UbuntuTurn:
    speaker: str  # "Speaker <id>"
    text: str
    addressee: str  # "Speaker <id>" or "" if none


@dataclass
class UbuntuInstance:
    conv_id: str
    turns: list  # [UbuntuTurn, ...] -- the context only, never the answer
    reply_to: list  # [[reply_idx, to_idx], ...] gold edges among `turns`
    answer: str
    answer_speaker: str
    answer_addressee: str
    candidates: list = None  # set by load(): gold + 9 sampled distractors, shuffled
    gold_index: int = -1
    # the source's raw structure fields, for structure-aware generators
    # (SS-MPC / EMMDG / HeterMPC-style inputs build on these directly)
    raw: dict = None


def _spk(i) -> str:
    return f"Speaker {i}" if i is not None and i >= 0 else ""


def load_split(raw_dir: Path, split: str = "test") -> list:
    out = []
    with open(raw_dir / f"ubuntu_data_{split}.json") as f:
        for n, line in enumerate(f):
            d = json.loads(line)
            spk, adr = d["ctx_spk"], d["ctx_adr"]
            out.append(
                UbuntuInstance(
                    conv_id=f"ubuntu_{split}_{n}",
                    turns=[UbuntuTurn(speaker=_spk(s), text=t, addressee=_spk(a)) for t, s, a in zip(d["context"], spk, adr)],
                    reply_to=[list(e) for e in d.get("relation_at", [])],
                    answer=d["answer"],
                    answer_speaker=_spk(d["ans_spk"]),
                    answer_addressee=_spk(d["ans_adr"]),
                    raw={k: d[k] for k in ("relation_at", "ctx_spk", "ctx_adr", "ans_idx", "ans_spk", "ans_adr") if k in d},
                )
            )
    return out


def load(raw_dir: Path) -> list:
    instances = load_split(raw_dir, "test")
    pool = [inst.answer for inst in instances]
    for idx, inst in enumerate(instances):
        rng = random.Random(int(hashlib.md5(inst.conv_id.encode()).hexdigest()[:8], 16))
        negatives = []
        while len(negatives) < N_CANDIDATES - 1:
            j = rng.randrange(len(pool))
            if j != idx and pool[j] != inst.answer and pool[j] not in negatives:
                negatives.append(pool[j])
        candidates = [inst.answer] + negatives
        rng.shuffle(candidates)
        inst.candidates = candidates
        inst.gold_index = candidates.index(inst.answer)
    return instances


def eval_items(conversation: UbuntuInstance) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=(
                f"Which candidate is the next message, sent by {conversation.answer_speaker}"
                + (f" to {conversation.answer_addressee}" if conversation.answer_addressee else "")
                + "?"
            ),
            reference=conversation.gold_index,
            candidates=conversation.candidates,
            metadata={
                "history": [f"{t.speaker}: {t.text}" for t in conversation.turns],
                "speaker": conversation.answer_speaker,
                "addressee": conversation.answer_addressee,
                "n_speakers": len({t.speaker for t in conversation.turns}),
            },
        )
    ]
