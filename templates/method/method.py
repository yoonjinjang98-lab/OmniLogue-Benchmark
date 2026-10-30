"""REPLACE_ME baseline.

Preferred shape: a `Method` subclass, so prepare() pays any ingestion cost
(building a RAG index, running memory-framework ingest, ...) exactly once per
conversation instead of once per question. For a truly stateless baseline
that needs no per-conversation setup, a bare
`def answer(conversation, question) -> str` function also works and is
auto-wrapped (see plugins/methods/smoke_test/method.py) -- but ingestion then
runs redundantly once per question, which is fine only for something with no
real ingestion cost.

Never call an LLM SDK directly -- use dialogue_benchmarks.llm.complete(), which
gives every method in this repo a shared disk cache and cost tracking for free.
"""
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete


class Method(_MethodBase):
    def __init__(self, model: str = "claude-sonnet-5", **config):
        super().__init__(model=model, **config)
        self.model = model

    def prepare(self, conversation) -> None:
        """Called once per conversation. Build whatever state answer() needs
        (index, memory store, ...) here -- not in answer()."""
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        prompt = f"...\n\nQuestion: {item.question}"
        resp = complete(prompt, model=self.model)
        return MethodOutput(answer=resp.text, tokens=resp.tokens_in + resp.tokens_out, cost=resp.cost)

    def reset(self) -> None:
        self._conversation = None
