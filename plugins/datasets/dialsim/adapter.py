"""Adapter for DialSim -- data/raw/dialsim/v1.1/{show}_dialsim.pickle
(github.com/jiho283/DialSim, no LICENSE file upstream -- vendored for
internal reproduction only, same call already made for groupmembench/
evermembench, see BASELINE_METRIC_PLAN.md §4).

Source shape per show: `{episode_filename: {scene_num: {date, script,
easy_q, hard_q}}}`. `script` is dialogue lines `"Speaker: text"`; `easy_q`
(fan-quiz-style) and `hard_q` (temporal-knowledge-graph reasoning) are
pre-generated question pools, each `{subtype: {id: entry}}` for `easy_q` or
`{subtype: [entry, ...]}` for `hard_q`, where `entry` is `{questions:
{asker_name: text, ...}, options, answer}` -- this pickle already holds a
FIXED, enumerable question set, contra earlier notes in
BASELINE_METRIC_PLAN.md §4 ("not a fixed, enumerable QA set"): what's
actually random at runtime is the *simulator*'s question SELECTION each
session, not the pool it selects from.

The pool itself is enormous though (millions of hard_q combinations per
show -- TKG relation-type combinatorics, never meant to all be asked in one
run, same as the real simulator only ever asking ~1 question per scene): this
adapter takes up to one `easy_q` and one `hard_q` per scene (first non-empty
subtype in a fixed priority order below, first entry within it) --
deterministic, not a random sample, so the resulting eval set is
reproducible across runs and roughly matches the real simulator's per-scene
question density.

One `Conversation` = one show (Friends/BigBang/TheOffice); one session = one
scene, in the show's original scene order -- this repo's generic
`.sessions`/`.qa` conventions, so full_context/bm25_rag/dense_rag/
running_summary already work against this dataset with no dialsim-specific
method needed.
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SHOWS = ("friends", "bigbang", "theoffice")

_EASY_Q_PRIORITY = ("ans_w_time", "ans_wo_time", "before_event_unans", "dont_know_unans", "dont_know_unans_time")
_HARD_Q_PRIORITY = (
    "past",
    "cur",
    "fu",
    "past_past",
    "past_cur",
    "cur_past",
    "cur_cur",
    "past_fu",
    "fu_past",
    "fu_cur",
    "cur_fu",
    "fu_fu",
)


@dataclass
class DialsimTurn:
    speaker: str
    text: str


@dataclass
class DialsimQA:
    question: str
    answer: str
    options: Optional[list]
    category: str  # "easy_q:<subtype>" or "hard_q:<subtype>"
    evidence: list = field(default_factory=list)  # [session_idx]


@dataclass
class DialsimShow:
    conv_id: str
    show: str
    sessions: dict = field(default_factory=dict)  # session_idx -> [DialsimTurn, ...]
    session_datetimes: dict = field(default_factory=dict)
    qa: list = field(default_factory=list)  # [DialsimQA, ...]


def _parse_script(script: str) -> list:
    turns = []
    for line in (script or "").split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        turns.append(DialsimTurn(speaker=speaker.strip(), text=text.strip()))
    return turns


def _first_easy_q(easy_q: dict):
    for subtype in _EASY_Q_PRIORITY:
        for entry in (easy_q.get(subtype) or {}).values():
            return subtype, entry
    return None, None


def _first_hard_q(hard_q: dict):
    for subtype in _HARD_Q_PRIORITY:
        bucket = hard_q.get(subtype) or []
        if bucket:
            return subtype, bucket[0]
    return None, None


def _to_qa(category: str, entry: dict, session_idx: int) -> DialsimQA:
    questions = entry.get("questions", {})
    question = questions.get("default") or next(iter(questions.values()), "")
    return DialsimQA(
        question=question,
        answer=entry.get("answer", ""),
        options=entry.get("options"),
        category=category,
        evidence=[session_idx],
    )


def _load_show(raw_dir: Path, show: str) -> "DialsimShow":
    with open(raw_dir / "v1.1" / f"{show}_dialsim.pickle", "rb") as f:
        data = pickle.load(f)

    sessions, session_datetimes, qa = {}, {}, []
    session_idx = 0
    for episode_name in data:
        for scene_num in data[episode_name]:
            scene = data[episode_name][scene_num]
            sessions[session_idx] = _parse_script(scene.get("script"))
            session_datetimes[session_idx] = scene.get("date")

            easy_subtype, easy_entry = _first_easy_q(scene.get("easy_q") or {})
            if easy_entry:
                qa.append(_to_qa(f"easy_q:{easy_subtype}", easy_entry, session_idx))
            hard_subtype, hard_entry = _first_hard_q(scene.get("hard_q") or {})
            if hard_entry:
                qa.append(_to_qa(f"hard_q:{hard_subtype}", hard_entry, session_idx))

            session_idx += 1

    return DialsimShow(
        conv_id=f"dialsim_{show}", show=show, sessions=sessions, session_datetimes=session_datetimes, qa=qa
    )


def load(raw_dir: Path) -> list:
    return [_load_show(raw_dir, show) for show in SHOWS]
