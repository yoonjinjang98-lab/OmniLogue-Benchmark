"""REPLACE_ME -- score(...) -> float in [0, 1] (or whatever range you document
in meta.json's description).

Two supported shapes, picked by parameter name (see
dialogue_benchmarks.metrics.registry._takes_item):

- legacy: def score(prediction, reference) -> float
  Use this if all you need is the prediction string and the gold reference
  (this is what qa_em/qa_f1 use).

- item-aware: def score(item, prediction) -> float
  Use this if you need more than prediction+reference -- e.g. an LLM-judge
  metric needs `item.question`, and anything wanting extra per-item metadata
  beyond `item.metadata["category"]` (which the runner already breaks scores
  out by automatically) needs `item`.
"""


def score(item, prediction) -> float:
    reference = item.reference
    raise NotImplementedError


# Optional. If your metric isn't a plain per-item mean (e.g. corpus-level BLEU,
# or a dict of sub-scores), define this -- the runner calls it instead of
# averaging per-item scores itself.
#
# def aggregate(scores: list, items: list) -> float | dict:
#     ...
#
# Add a third `predictions` parameter if the corpus-level number can't be
# recovered from per-item scores (e.g. distinct_n's unique n-grams across all
# predictions):
#
# def aggregate(scores: list, items: list, predictions: list) -> float | dict:
#     ...
