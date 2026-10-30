"""Adapter for ComperDial -- data/raw/comperdial/ComperDial_{data,turn_level_evaluation}.json
(downloaded from huggingface.co/datasets/Sony/ComperDial)

Source shape: ComperDial_data.json is 100 dialogues {dialogue_id, persona:
{A: [...], B: [...]}, dialogue: [{persona_id: "A"/"B", text}]};
ComperDial_turn_level_evaluation.json scores, for 15 of those dialogues,
~100 system responses per context turn with 3 annotators' 1-5 "overall"
ratings ({dialogue_id, turn_number, context, evaluation_unit: [{system_id,
response, overall: [...]}]}).

task_type="dialogue": like personachat, one item per speaker-B turn with B's
persona and the history so far. The human ratings are attached to each
dialogue as `turn_judgments` (turn_number -> [{system_id, response,
mean_overall}]) -- they aren't something a method answers, but they're what a
metric-vs-human correlation check needs (BASELINE_METRIC_PLAN.md §5).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem


@dataclass
class ComperDialTurn:
    speaker: str  # "A" / "B"
    text: str


@dataclass
class ComperDialDialogue:
    conv_id: str
    dialogue_id: int
    persona: dict  # {"A": [...], "B": [...]}
    turns: list  # [ComperDialTurn, ...]
    turn_judgments: dict = field(default_factory=dict)


def load(raw_dir: Path) -> list:
    with open(raw_dir / "ComperDial_data.json") as f:
        dialogues = json.load(f)
    with open(raw_dir / "ComperDial_turn_level_evaluation.json") as f:
        judgments = json.load(f)

    by_dialogue: dict = {}
    for j in judgments:
        units = []
        for u in j["evaluation_unit"]:
            scores = [int(s) for s in u.get("overall", []) if str(s).strip().isdigit()]
            units.append(
                {
                    "system_id": u["system_id"],
                    "response": u["response"].strip(),
                    "mean_overall": sum(scores) / len(scores) if scores else None,
                }
            )
        by_dialogue.setdefault(j["dialogue_id"], {})[j["turn_number"]] = units

    return [
        ComperDialDialogue(
            conv_id=f"comperdial_{d['dialogue_id']}",
            dialogue_id=d["dialogue_id"],
            persona=d["persona"],
            turns=[ComperDialTurn(speaker=t["persona_id"], text=t["text"]) for t in d["dialogue"]],
            turn_judgments=by_dialogue.get(d["dialogue_id"], {}),
        )
        for d in dialogues
    ]


def eval_items(conversation: ComperDialDialogue) -> list:
    items = []
    for i, turn in enumerate(conversation.turns):
        if i == 0 or turn.speaker != "B":
            continue
        items.append(
            EvalItem(
                conversation_id=conversation.conv_id,
                question="produce the next turn",
                reference=turn.text,
                metadata={
                    "persona": conversation.persona.get("B", []),
                    "history": [f"{t.speaker}: {t.text}" for t in conversation.turns[:i]],
                    "speaker": "B",
                    "turn_index": i,
                },
            )
        )
    return items
