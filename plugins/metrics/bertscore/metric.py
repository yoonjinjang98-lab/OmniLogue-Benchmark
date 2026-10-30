"""bertscore -- BERTScore F1 via the bert-score package, rescaled against its
bundled baseline (see meta.json for why). score(prediction, reference) -> float,
usually in [0, 1] but not hard-bounded.

Loads the model once at import time (module-level _scorer) rather than per call --
bert_score.score() reloads the model on every invocation, which would be
prohibitively slow when the runner calls score() once per item.
"""
from bert_score import BERTScorer

_scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def score(prediction, reference) -> float:
    prediction = str(prediction)
    reference = str(reference)

    if not prediction.strip() and not reference.strip():
        return 1.0
    if not prediction.strip() or not reference.strip():
        return 0.0

    _, _, f1 = _scorer.score([prediction], [reference])
    return f1.item()
