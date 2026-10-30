"""task_type="nli" -- dnli: classify premise/hypothesis pairs (entailment /
neutral / contradiction), not a conversation to run memory methods against.
No generic convention -- the adapter must define eval_items(), where each
"conversation" a group is keyed on can just be the pair itself (methods that
don't need per-conversation state can leave prepare()/reset() as no-ops).
"""
from dialogue_benchmarks.tasks._common import require_eval_items


def build_items(adapter_module, convs) -> list:
    return require_eval_items(adapter_module, convs, task_type="nli")
