"""qa_em -- SQuAD-style exact match. score(prediction, reference) -> 0.0 or 1.0"""
import re
import string


def _normalize(text) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score(prediction, reference) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0
