"""task_type="dialogue" -- msc, personachat, conversation_chronicles: next-turn
generation, not QA. There's no `.qa`-style generic convention for this shape
(the reference is the gold next utterance, not an answer to a question), so
the adapter must define eval_items(conversation) -> list[EvalItem] where
`question` is the generation prompt/context and `reference` is the gold turn.
"""
from dialogue_benchmarks.tasks._common import require_eval_items


def build_items(adapter_module, convs) -> list:
    return require_eval_items(adapter_module, convs, task_type="dialogue")
