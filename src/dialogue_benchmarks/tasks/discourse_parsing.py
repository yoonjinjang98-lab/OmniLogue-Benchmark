"""task_type="discourse_parsing" -- molweni's DP split: predict EDU/relation
graph edges, not a QA answer. Placeholder wiring only -- no dataset exposes
this task_type yet (molweni's adapter currently only exposes the MRC/`qa`
side). Add eval_items() to an adapter and set task_type="discourse_parsing"
in its meta.json to light this up; no runner changes needed.
"""
from dialogue_benchmarks.tasks._common import require_eval_items


def build_items(adapter_module, convs) -> list:
    return require_eval_items(adapter_module, convs, task_type="discourse_parsing")
