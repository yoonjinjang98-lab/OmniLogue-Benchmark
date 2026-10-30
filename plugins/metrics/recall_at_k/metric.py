"""recall_at_k -- R_n@k for candidate ranking (ubuntu_irc, mpchat NRP).

score(item, prediction) -> 1.0 if the gold candidate is ranked first (R@1);
aggregate() re-derives the full {R@1, R@2, R@5, R@10} from the predictions.
"""
from dialogue_benchmarks.metrics.parsing import gold_rank

_KS = (1, 2, 5, 10)


def score(item, prediction) -> float:
    return 1.0 if gold_rank(prediction, item.candidates, int(item.reference)) <= 1 else 0.0


def aggregate(scores: list, items: list, predictions: list) -> dict:
    ranks = [gold_rank(p, it.candidates, int(it.reference)) for it, p in zip(items, predictions)]
    if not ranks:
        return {f"R@{k}": 0.0 for k in _KS}
    return {f"R@{k}": sum(r <= k for r in ranks) / len(ranks) for k in _KS}
