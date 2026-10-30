"""qa_llm_judge_gpt -- qa_llm_judge's prompt, graded by an OpenAI model.

Loads qa_llm_judge's judge() from its sibling plugin directory instead of
copying the prompt, so the two judges can only differ in the model.
"""
import importlib.util
from pathlib import Path

JUDGE_MODEL = "gpt-4o-mini"

_SIBLING = Path(__file__).resolve().parent.parent / "qa_llm_judge" / "metric.py"
_spec = importlib.util.spec_from_file_location("_qa_llm_judge_for_gpt", _SIBLING)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def score(item, prediction) -> float:
    return _base.judge(item, prediction, JUDGE_MODEL)
