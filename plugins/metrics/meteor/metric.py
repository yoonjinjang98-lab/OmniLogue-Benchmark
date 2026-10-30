"""meteor -- METEOR via nltk. score(prediction, reference) -> float in [0, 1]

Needs the 'wordnet', 'omw-1.4', 'punkt'/'punkt_tab' nltk corpora on disk --
`python -c "import nltk; [nltk.download(p) for p in ('wordnet','omw-1.4','punkt','punkt_tab')]"`
"""
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score


def score(prediction, reference) -> float:
    prediction = str(prediction)
    reference = str(reference)

    if not prediction.strip() and not reference.strip():
        return 1.0
    if not prediction.strip() or not reference.strip():
        return 0.0

    pred_tokens = word_tokenize(prediction.lower())
    ref_tokens = word_tokenize(reference.lower())
    return meteor_score([ref_tokens], pred_tokens)
