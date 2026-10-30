"""mrr -- mean reciprocal rank. score(item, prediction) -> 1 / rank(gold)"""
from dialogue_benchmarks.metrics.parsing import gold_rank


def score(item, prediction) -> float:
    return 1.0 / gold_rank(prediction, item.candidates, int(item.reference))
