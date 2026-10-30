"""bleu -- sentence-level BLEU via sacrebleu. score(prediction, reference) -> float in [0, 1]"""
import sacrebleu


def score(prediction, reference) -> float:
    prediction = str(prediction)
    reference = str(reference)

    if not prediction.strip() and not reference.strip():
        return 1.0

    result = sacrebleu.sentence_bleu(prediction, [reference], smooth_method="exp", use_effective_order=True)
    return result.score / 100.0
