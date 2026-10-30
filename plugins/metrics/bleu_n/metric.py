"""bleu_n -- corpus BLEU-1~4, the multi-party response generation convention.

Both sides are lowercased and nltk-word-tokenized, so a model's natural
output ("doesn't") matches a pre-tokenized reference ("does n't") the way
those papers' own tokenized outputs did. The headline numbers are computed
over the whole corpus in aggregate() (BLEU is not an average of sentence
scores); score() exists for the per-item / per-category view.
"""
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

_WEIGHTS = {n: tuple([1.0 / n] * n) for n in (1, 2, 3, 4)}
_SMOOTH = SmoothingFunction().method1


def _tok(text) -> list:
    return word_tokenize(str(text).lower())


def score(prediction, reference) -> float:
    hyp = _tok(prediction)
    if not hyp:
        return 0.0
    return sentence_bleu([_tok(reference)], hyp, weights=_WEIGHTS[4], smoothing_function=_SMOOTH)


def aggregate(scores: list, items: list, predictions: list) -> dict:
    refs = [[_tok(item.reference)] for item in items]
    hyps = [_tok(p) for p in predictions]
    return {f"BLEU-{n}": corpus_bleu(refs, hyps, weights=w) for n, w in _WEIGHTS.items()}
