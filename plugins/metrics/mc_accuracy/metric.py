"""mc_accuracy -- multiple-choice accuracy. score(item, prediction) -> 1.0/0.0

Needs item.options; the gold option is whichever option equals item.reference
(or item.metadata["answer_index"] when an adapter sets it explicitly).
"""
from dialogue_benchmarks.metrics.parsing import normalize, parse_choice


def _gold_index(item):
    if isinstance(item.metadata, dict) and "answer_index" in item.metadata:
        return item.metadata["answer_index"]
    ref = normalize(item.reference)
    for i, opt in enumerate(item.options or []):
        if normalize(opt) == ref:
            return i
    return None


def score(item, prediction) -> float:
    if not item.options:
        return 0.0
    gold = _gold_index(item)
    return 1.0 if gold is not None and parse_choice(prediction, item.options) == gold else 0.0
