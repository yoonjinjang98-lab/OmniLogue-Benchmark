"""task_type="qa" -- locomo, longmemeval, molweni(MRC), personamem, beam, ...

Uses the adapter's eval_items(conversation) when it defines one (needed for
anything that isn't a plain free-form question, e.g. personamem's multiple
choice), otherwise falls back to the `.qa` convention.
"""
from dialogue_benchmarks.tasks._common import qa_fallback_items


def build_items(adapter_module, convs) -> list:
    if hasattr(adapter_module, "eval_items"):
        groups = []
        for conv in convs:
            items = adapter_module.eval_items(conv)
            if items:
                groups.append((conv, items))
        return groups
    return qa_fallback_items(convs)
