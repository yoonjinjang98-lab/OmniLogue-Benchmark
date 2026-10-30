"""Shared types that datasets / methods / metrics plugins are written against.

Adding a task_type or a plugin should never require touching this file -- if you
find yourself doing that, the type is probably missing a field that belongs in
`metadata` instead.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalItem:
    """One thing a method must answer and a metric must score.

    `question` is deliberately generic: for task_type="qa" it's a question, for
    "dialogue" it's "produce the next turn", for "nli" it's the hypothesis (with
    the premise reachable via metadata or the conversation object itself). Adapters
    that need a different shape implement `eval_items(conversation) -> list[EvalItem]`
    instead of relying on the `.qa` convention.
    """

    conversation_id: str
    question: str
    reference: Any
    options: Optional[list] = None
    candidates: Optional[list] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MethodOutput:
    """What a method's answer() may return. A plain string is also accepted --
    it gets wrapped as MethodOutput(answer=that_string) by the runner."""

    answer: str
    retrieved: Optional[list] = None
    tokens: Optional[int] = None
    latency: Optional[float] = None
    cost: Optional[float] = None


class Method:
    """Base class for methods that need to ingest a conversation once and then
    answer many questions against it (RAG index, memory-agent state, ...).

    Subclassing this is optional -- a plugin's method.py can instead just expose
    a bare `answer(conversation, question) -> str` function, which the registry
    auto-wraps into an instance of this shape with no-op prepare/reset. Use the
    class form whenever ingestion cost should be paid once per conversation
    rather than once per question.
    """

    def __init__(self, **config):
        self.config = config

    def prepare(self, conversation) -> None:
        """Called once per conversation before any answer() calls. Build an
        index, run memory ingestion, etc. Default: no-op."""

    def answer(self, item: EvalItem):
        raise NotImplementedError

    def reset(self) -> None:
        """Called after the last answer() for a conversation, before prepare()
        is called again for the next one. Default: no-op."""
