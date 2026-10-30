"""persona_consistency_gpt -- persona_consistency's prompt, judged by an OpenAI model.

Loads persona_consistency's judge() from its sibling plugin directory, so the
two differ only in the model.
"""
import importlib.util
from pathlib import Path

JUDGE_MODEL = "gpt-4o-mini"

_SIBLING = Path(__file__).resolve().parent.parent / "persona_consistency" / "metric.py"
_spec = importlib.util.spec_from_file_location("_persona_consistency_for_gpt", _SIBLING)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def score(item, prediction) -> float:
    return _base.judge(item, prediction, JUDGE_MODEL)
