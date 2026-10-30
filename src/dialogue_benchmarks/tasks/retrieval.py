"""task_type="retrieval" -- ubuntu_irc (response selection), mpchat (NRP):
ranking a fixed candidate set rather than generating free text. Reference is
the correct candidate (or its index); `item.candidates` carries the pool.
No generic convention -- the adapter must define eval_items().
"""
from dialogue_benchmarks.tasks._common import require_eval_items


def build_items(adapter_module, convs) -> list:
    return require_eval_items(adapter_module, convs, task_type="retrieval")
