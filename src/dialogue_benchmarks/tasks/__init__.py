from dialogue_benchmarks.tasks import dialogue, discourse_parsing, nli, qa, retrieval
from dialogue_benchmarks.tasks._common import run

_TASK_MODULES = {
    "qa": qa,
    "dialogue": dialogue,
    "retrieval": retrieval,
    "nli": nli,
    "discourse_parsing": discourse_parsing,
}


def get_task_module(task_type: str):
    if task_type not in _TASK_MODULES:
        raise NotImplementedError(
            f"task_type='{task_type}' has no evaluation logic yet "
            f"(known: {sorted(_TASK_MODULES)}). Add a module to "
            f"src/dialogue_benchmarks/tasks/ exposing build_items(adapter_module, convs)."
        )
    return _TASK_MODULES[task_type]


__all__ = ["get_task_module", "run"]
