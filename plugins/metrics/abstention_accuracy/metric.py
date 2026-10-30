"""abstention_accuracy -- unanswerable-question handling. score(item, prediction) -> 1.0/0.0

An item is unanswerable when an adapter marks it with
item.metadata["answerable"] = False, when its category is "abstention"
(groupmembench), or (SQuAD-style, e.g. molweni) when its reference is empty. Abstention is detected with a keyword heuristic rather than
an LLM call so the metric is free and deterministic; pair it with qa_llm_judge
when the phrasing of a refusal matters.
"""
import re

_ABSTAIN_PATTERNS = [
    r"\bnot (?:been )?(?:mentioned|stated|provided|specified|discussed|available|known|clear|sure)\b",
    r"\bno (?:information|mention|record|details?|evidence|data)\b",
    r"\b(?:can ?not|can't|unable to|could not|couldn't) (?:answer|determine|find|tell|say|know|be determined)\b",
    r"\b(?:don't|do not|doesn't|does not) (?:know|have (?:that|this|any|enough)|mention|say|specify)\b",
    r"\bunknown\b",
    r"\bunanswerable\b",
    r"\binsufficient information\b",
    r"\bnot enough information\b",
    r"\bthere is no\b.*\b(?:information|mention|indication)\b",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), flags=re.I)


def abstains(prediction) -> bool:
    text = str(prediction).strip()
    return not text or bool(_ABSTAIN_RE.search(text))


def is_unanswerable(item) -> bool:
    meta = item.metadata if isinstance(item.metadata, dict) else {}
    if "answerable" in meta:
        return not meta["answerable"]
    # groupmembench-style: a dedicated "abstention" question category
    if str(meta.get("category", "")).lower() == "abstention":
        return True
    return str(item.reference).strip() == ""


def score(item, prediction) -> float:
    return 1.0 if abstains(prediction) == is_unanswerable(item) else 0.0


def aggregate(scores: list, items: list, predictions: list) -> dict:
    unans = [s for s, it in zip(scores, items) if is_unanswerable(it)]
    ans = [s for s, it in zip(scores, items) if not is_unanswerable(it)]
    out = {"accuracy": sum(scores) / len(scores) if scores else 0.0}
    # recall on unanswerable items: fraction where the method correctly abstained
    out["unanswerable_recall"] = sum(unans) / len(unans) if unans else None
    # fraction of answerable items where the method (wrongly) abstained
    out["false_abstention_rate"] = 1 - sum(ans) / len(ans) if ans else None
    out["n_unanswerable"] = len(unans)
    return out
