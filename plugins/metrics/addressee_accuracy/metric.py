"""addressee_accuracy -- top-1 accuracy over candidate addressees.

Same prediction format as the other retrieval metrics (index / ranking /
candidate text), so a ranker baseline and a "name the speaker" LLM baseline
are scored identically.
"""
from dialogue_benchmarks.metrics.parsing import parse_ranking


def score(item, prediction) -> float:
    ranking = parse_ranking(prediction, item.candidates)
    return 1.0 if ranking and ranking[0] == int(item.reference) else 0.0
