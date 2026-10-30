"""Adapter for REPLACE_ME -- data/raw/REPLACE_ME/<original files>

Describe the source shape here: what's a conversation, what's a turn, what
QA-like structure (if any) exists.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Qa:
    question: str
    answer: Any
    # Optional: put anything you want automatic per-category score breakdown
    # for into metadata via the fields below -- the runner groups by `category`
    # if present.
    category: Any = None


@dataclass
class Conversation:
    conv_id: str
    turns: list = field(default_factory=list)
    qa: list = field(default_factory=list)  # task_type="qa": .question/.answer per item


def load(raw_dir: Path) -> list:
    """Required. Parse the original files under raw_dir and return a list of
    Conversation (or whatever dataclass makes sense for this dataset -- don't
    force fields that don't exist in the source, see README.md)."""
    raise NotImplementedError(f"implement load() for {raw_dir}")


# Optional. Only needed if this dataset's task_type isn't "qa", or if "qa"
# doesn't fit the plain `.qa` convention (e.g. multiple-choice, needs
# per-item `options`). See src/dialogue_benchmarks/contracts.py:EvalItem and
# src/dialogue_benchmarks/tasks/ for which task_types need this.
#
# def eval_items(conversation) -> list[EvalItem]:
#     ...
