"""nli_accuracy -- 3-way NLI label accuracy. score(prediction, reference) -> 1.0/0.0

DNLI's own labels are positive/negative/neutral; prompted LLM baselines tend to
answer entailment/contradiction/neutral. Both map onto the same three classes.
"""
import re

_SYNONYMS = {
    "entailment": "entailment",
    "entail": "entailment",
    "entails": "entailment",
    "entailed": "entailment",
    "positive": "entailment",
    "contradiction": "contradiction",
    "contradict": "contradiction",
    "contradicts": "contradiction",
    "contradictory": "contradiction",
    "negative": "contradiction",
    "neutral": "neutral",
}


def normalize_label(text):
    """First recognized label word in `text`, or None."""
    for word in re.findall(r"[a-z]+", str(text).lower()):
        if word in _SYNONYMS:
            return _SYNONYMS[word]
    return None


def score(prediction, reference) -> float:
    gold = normalize_label(reference)
    return 1.0 if gold is not None and normalize_label(prediction) == gold else 0.0
