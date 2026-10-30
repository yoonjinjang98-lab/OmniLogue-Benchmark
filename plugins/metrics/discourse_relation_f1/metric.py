"""discourse_relation_f1 -- link / link+relation F1 for dialogue discourse parsing.

item.reference is the gold edge list [[x, y, type], ...] (x = the EDU being
responded to, y = the responding EDU, as in molweni's `relations`).
score(item, prediction) -> per-dialogue link+relation F1; aggregate() reports
micro-averaged link_f1 and link_rel_f1 over all edges, which is how Molweni /
STAC parsers are usually compared.
"""
import json
import re


def _norm_type(t) -> str:
    return re.sub(r"[^a-z]", "", str(t).lower())


def parse_edges(prediction) -> set:
    if isinstance(prediction, list):
        raw = prediction
    else:
        text = str(prediction).strip()
        try:
            raw = json.loads(text)
        except (ValueError, TypeError):
            raw = []
            for line in text.splitlines():
                m = re.match(r"\s*\[?\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*[,\s]\s*['\"]?([A-Za-z\-_ ]+)", line)
                if m:
                    raw.append([int(m.group(1)), int(m.group(2)), m.group(3)])
    edges = set()
    for e in raw if isinstance(raw, list) else []:
        if isinstance(e, dict):
            e = [e.get("x"), e.get("y"), e.get("type")]
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            try:
                edges.add((int(e[0]), int(e[1]), _norm_type(e[2])))
            except (TypeError, ValueError):
                continue
    return edges


def _f1(n_pred, n_gold, n_hit) -> float:
    if n_pred == 0 and n_gold == 0:
        return 1.0
    if n_pred == 0 or n_gold == 0 or n_hit == 0:
        return 0.0
    p, r = n_hit / n_pred, n_hit / n_gold
    return 2 * p * r / (p + r)


def score(item, prediction) -> float:
    pred, gold = parse_edges(prediction), parse_edges(item.reference)
    return _f1(len(pred), len(gold), len(pred & gold))


def aggregate(scores: list, items: list, predictions: list) -> dict:
    n_pred = n_gold = hit_link = hit_rel = 0
    for item, p in zip(items, predictions):
        pred, gold = parse_edges(p), parse_edges(item.reference)
        n_pred += len(pred)
        n_gold += len(gold)
        hit_rel += len(pred & gold)
        hit_link += len({(x, y) for x, y, _ in pred} & {(x, y) for x, y, _ in gold})
    return {"link_f1": _f1(n_pred, n_gold, hit_link), "link_rel_f1": _f1(n_pred, n_gold, hit_rel)}
