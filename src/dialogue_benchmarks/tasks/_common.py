"""Shared machinery every task_type module in this package is built from.

A task module's whole job is `build_items(adapter_module, convs) ->
list[(conversation, list[EvalItem])]` -- how to turn loaded conversations into
gradeable items for that task shape. Running a method against those items
(prepare once per conversation, answer per item, reset) is identical across
task types, so it lives here instead of being copy-pasted into every module.
"""
from dialogue_benchmarks.contracts import EvalItem, MethodOutput


def _conv_id(conv) -> str:
    for attr in ("conv_id", "conversation_id", "id"):
        if hasattr(conv, attr):
            return str(getattr(conv, attr))
    return str(id(conv))


def _qa_extra(qa) -> dict:
    extra = {}
    for attr in ("category", "evidence", "question_type", "qtype"):
        if hasattr(qa, attr):
            extra[attr] = getattr(qa, attr)
    return extra


def _qa_reference(qa):
    """Most adapters expose a single `.answer`. SQuAD-style ones (molweni) expose
    `.answers`: a list of acceptable answer strings, empty when the question is
    unanswerable -- take the first (or "" for unanswerable) so string-based metrics
    like qa_em/qa_f1 keep working. A metric that needs the full answer set for
    proper multi-reference scoring should use `item.metadata["answers"]` instead
    (see below) rather than assuming `.reference` is ever a list.
    """
    if hasattr(qa, "answer"):
        return qa.answer
    answers = getattr(qa, "answers", None)
    if answers is not None:
        return answers[0] if answers else ""
    raise AttributeError(f"{qa!r} has neither `.answer` nor `.answers` -- qa_fallback_items() can't score it")


def qa_fallback_items(convs) -> list:
    """The one light convention task_type="qa" adapters follow so build_items()
    works without an explicit eval_items(): each conversation exposes a `.qa`
    list of objects with `.question`/`.answer`. Mirrors the convention the
    original runner._collect_qa_items() used."""
    groups = []
    for conv in convs:
        qa_list = getattr(conv, "qa", None)
        if not qa_list:
            continue
        conv_id = _conv_id(conv)
        items = [
            EvalItem(
                conversation_id=conv_id,
                question=qa.question,
                reference=_qa_reference(qa),
                options=getattr(qa, "options", None),
                metadata={**_qa_extra(qa), **({"answers": qa.answers} if hasattr(qa, "answers") else {})},
            )
            for qa in qa_list
        ]
        groups.append((conv, items))
    return groups


def require_eval_items(adapter_module, convs, task_type: str) -> list:
    """For task types with no generic convention (dialogue/retrieval/nli/...):
    the adapter must define eval_items(conversation) -> list[EvalItem]."""
    if not hasattr(adapter_module, "eval_items"):
        raise NotImplementedError(
            f"task_type='{task_type}' has no generic item convention (unlike 'qa', "
            f"which falls back to `.qa`) -- this dataset's adapter.py must define "
            f"eval_items(conversation) -> list[EvalItem]. See contracts.py / CLAUDE.md."
        )
    groups = []
    for conv in convs:
        items = adapter_module.eval_items(conv)
        if items:
            groups.append((conv, items))
    return groups


def run(method, groups: list) -> list:
    """groups: list[(conversation, list[EvalItem])]. Returns list[(EvalItem,
    MethodOutput)] in the same order, having called prepare() once and reset()
    once per conversation -- the one place ingestion-cost handling lives, so
    every task_type gets it for free."""
    results = []
    for conv, items in groups:
        if not items:
            continue
        method.prepare(conv)
        for item in items:
            raw = method.answer(item)
            out = raw if isinstance(raw, MethodOutput) else MethodOutput(answer=raw)
            results.append((item, out))
        method.reset()
    return results
