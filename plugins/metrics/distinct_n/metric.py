"""distinct_n -- corpus-level distinct-1/distinct-2 (Li et al. 2016).

Corpus-level distinct-n counts unique n-grams across *all* responses at once,
which can't be recovered from per-item scores -- so aggregate() takes the
predictions directly (see metrics/registry.py:load_aggregate_fn).
"""


def _ngrams(text, n: int) -> list:
    toks = str(text).lower().split()
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def _distinct(texts, n: int) -> float:
    total, unique = 0, set()
    for t in texts:
        grams = _ngrams(t, n)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def score(prediction, reference) -> float:
    return _distinct([prediction], 2)


def aggregate(scores: list, items: list, predictions: list) -> dict:
    return {"distinct_1": _distinct(predictions, 1), "distinct_2": _distinct(predictions, 2)}
