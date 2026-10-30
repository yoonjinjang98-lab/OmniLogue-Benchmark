"""rouge_l -- ROUGE-L F-measure via google/rouge_score. score(prediction, reference) -> float in [0, 1]"""
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def score(prediction, reference) -> float:
    prediction = str(prediction)
    reference = str(reference)

    if not prediction.strip() and not reference.strip():
        return 1.0
    if not prediction.strip() or not reference.strip():
        return 0.0

    return _scorer.score(reference, prediction)["rougeL"].fmeasure
