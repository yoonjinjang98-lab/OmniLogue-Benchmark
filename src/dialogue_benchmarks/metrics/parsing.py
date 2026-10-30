"""Shared prediction-parsing conventions for metrics that score a *choice*
rather than free text -- multiple-choice (`item.options`) and ranking
(`item.candidates`). Lives in the engine rather than in each metric plugin so
the methods that produce these predictions and the metrics that grade them
agree on exactly one format.

The format a method should emit (as its plain-string answer):

- ranking over `item.candidates`: a JSON list of 0-based candidate indices,
  best first -- e.g. "[3, 0, 7]". A partial list is fine; unranked candidates
  are treated as tied after the ranked ones. A single index ("3") or the
  verbatim text of one candidate also parse, as a one-element ranking.
- choice among `item.options`: an option letter ("B", "(b)", "B.") or the
  option's text. Free text is matched to the closest option by token F1.
"""
import json
import re
import string
from collections import Counter
from typing import Optional

_LETTERS = string.ascii_uppercase


def normalize(text) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(a, b) -> float:
    a_toks, b_toks = normalize(a).split(), normalize(b).split()
    if not a_toks or not b_toks:
        return float(a_toks == b_toks)
    same = sum((Counter(a_toks) & Counter(b_toks)).values())
    if same == 0:
        return 0.0
    p, r = same / len(a_toks), same / len(b_toks)
    return 2 * p * r / (p + r)


def _match_text(prediction: str, texts: list, min_f1: float = 0.5) -> Optional[int]:
    norm = normalize(prediction)
    if not norm:
        return None
    for i, t in enumerate(texts):
        if normalize(t) == norm:
            return i
    best_i, best_f1 = None, 0.0
    for i, t in enumerate(texts):
        f1 = token_f1(prediction, t)
        if f1 > best_f1:
            best_i, best_f1 = i, f1
    return best_i if best_f1 >= min_f1 else None


def parse_ranking(prediction, candidates: Optional[list]) -> list:
    """Returns ranked 0-based candidate indices, best first (possibly partial,
    possibly empty if nothing parses)."""
    n = len(candidates) if candidates else None

    def _valid(idxs):
        seen, out = set(), []
        for i in idxs:
            if isinstance(i, bool) or not isinstance(i, int):
                continue
            if (n is None or 0 <= i < n) and i not in seen:
                seen.add(i)
                out.append(i)
        return out

    if isinstance(prediction, list):
        return _valid(prediction)
    if isinstance(prediction, int):
        return _valid([prediction])
    text = str(prediction).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _valid(parsed)
        if isinstance(parsed, int):
            return _valid([parsed])
    except (ValueError, TypeError):
        pass
    if re.fullmatch(r"[\[\(]?\s*\d+(\s*[,\s]\s*\d+)*\s*[\]\)]?", text):
        return _valid([int(x) for x in re.findall(r"\d+", text)])
    if candidates:
        hit = _match_text(text, [str(c) for c in candidates])
        if hit is not None:
            return [hit]
    return []


def gold_rank(prediction, candidates: Optional[list], gold_index: int) -> float:
    """1-based rank of the gold candidate. If the gold candidate isn't in the
    (possibly partial) ranking, it gets the expected rank of a random draw
    among the unranked remainder -- so a method that only names its top pick
    isn't scored as if it ranked the gold answer dead last."""
    ranking = parse_ranking(prediction, candidates)
    if gold_index in ranking:
        return float(ranking.index(gold_index) + 1)
    n = len(candidates) if candidates else len(ranking) + 1
    remaining = max(n - len(ranking), 1)
    return len(ranking) + (remaining + 1) / 2


def parse_choice(prediction, options: Optional[list]) -> Optional[int]:
    """0-based index of the option a prediction picks, or None."""
    if not options:
        return None
    text = str(prediction).strip()
    # a bare letter, or a letter followed by a delimiter -- "A", "(b)", "B." --
    # but not prose that merely starts with one ("A good option is ...")
    m = re.match(r"[\(\[]?\s*([A-Za-z])\s*(?:[\)\]\.:]|$)", text)
    if m and _LETTERS.index(m.group(1).upper()) < len(options):
        return _LETTERS.index(m.group(1).upper())
    m = re.match(r"^(?:answer|option)\s*[:\-]?\s*\(?([A-Za-z])\)?\b", text, flags=re.I)
    if m and _LETTERS.index(m.group(1).upper()) < len(options):
        return _LETTERS.index(m.group(1).upper())
    return _match_text(text, [str(o) for o in options], min_f1=0.0)


def option_letter(i: int) -> str:
    return _LETTERS[i]
