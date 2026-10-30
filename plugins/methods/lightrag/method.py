"""lightrag -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps HKUDS's
LightRAG (HKUDS/LightRAG), MIT licensed. LLM-driven entity/relationship
extraction builds a knowledge graph from the conversation (same overall idea
as plugins/methods/graphrag and plugins/methods/hipporag -- extract, then
graph-search), answered via one of five retrieval modes (`mode=`): "local"
(entity-anchored), "global" (relationship-anchored), "hybrid" (both), "naive"
(plain vector search over text chunks, no graph), or "mix" (graph + vector
together -- LightRAG's own default and this Method's default too).

Compared to `graphrag`, this is the simplest family-C integration in this
repo: LightRAG's provider seam is two **plain functions**, not ABC
subclasses or a factory-registration API --
`llm_model_func(prompt, system_prompt=None, history_messages=None,
response_format=None, **kwargs) -> str` and `EmbeddingFunc(embedding_dim=,
func=async (texts: list[str]) -> np.ndarray)`. No monkeypatching, no
`sys.modules` stub, no ABC to subclass -- `functools.partial` binds this
Method instance's `model`/`embed_model`/`db_stats` straight onto
`dialogue_benchmarks.llm.complete()`/`embed()` and that's the whole
integration. `response_format` here is OpenAI's plain
`{"type": "json_object"}` dict (used for keyword extraction and,
optionally, structured entity extraction), not a pydantic class like
GraphRAG's -- so no schema parsing is needed on this end, just an
instruction appended to the prompt telling the model to reply with JSON;
LightRAG parses its own JSON downstream.

Storage is all local and file-based by default (`JsonKVStorage`,
`NanoVectorDBStorage`, `NetworkXStorage`, `JsonDocStatusStorage` -- pure
Python/JSON/networkx, no embedded database process at all, not even
LightRAG's own graph store), the simplest "no external service" setup of
any family-C baseline here (graphrag needs LanceDB, zep/hipporag need
Kuzu). `rerank_model_func` is left unset and `QueryParam(enable_rerank=
False)` is passed explicitly to silence LightRAG's own warning about
reranking being requested with no rerank model configured, rather than
adding a reranker this repo has no other baseline using.

Token/cost/latency accounting: `db_stats`, a plain dict, is bound via
`functools.partial` into both provider functions and cleared at the start
of each `answer()`, mirroring `graphrag`'s approach -- `prepare()`'s
ingestion-time LLM calls are deliberately not reported, matching every
other family-C baseline here.

Event-loop bridging: `lightrag.kg.shared_storage` is a process-wide
singleton -- its locks/queues/worker pools are set up once (the first time
`initialize_storages()`/`initialize_pipeline_status()` runs) and a repeat
call is a silent no-op ("already initialized"), and the worker pools
`initialize_storages()`/the first `ainsert()` spin up expect the event loop
to keep running continuously for the whole object's lifetime. Confirmed
empirically that this breaks both the obvious approaches used by every
other async family-C baseline here: a fresh `asyncio.run()` per call, and
even one `asyncio.new_event_loop()` reused via repeated
`run_until_complete()` (which stops the loop between calls) -- both raise
"bound to a different event loop" / "no running event loop" from
LightRAG's own worker threads and silently drop every embedding, so every
conversation after the first returns `[no-context]`. The fix used here: a
dedicated background thread runs the loop via `run_forever()` for the whole
prepare()..reset() span (`_run()` submits coroutines to it with
`run_coroutine_threadsafe()`), and `reset()` explicitly calls
`finalize_share_data()` -- the library's own documented reset "for when the
application is shutting down" -- before stopping that loop, so the next
conversation's `prepare()` gets a genuinely clean re-init rather than
silently inheriting dead locks bound to a closed loop.

Known cosmetic issue: even with the above fix, tearing down one
conversation's `LightRAG` while a new one starts logs a burst of "Critical
error in worker: Event loop is closed" from the previous instance's daemon
worker threads finishing their last iteration against an already-stopped
loop. Confirmed harmless in testing -- every conversation still returns a
correctly-grounded answer, timed to *before* any of this noise appears in
the log -- but it's real log spam, not something this code found a way to
fully silence.

Verification: mock-tested against locomo conversations with `complete()`/
`embed()` monkeypatched, covering the full pipeline --
`LightRAG.ainsert()` (chunking, entity/relationship extraction via
LightRAG's own tuple-delimited default prompt format, graph + vector
upsert) through `LightRAG.aquery()` in all five modes (local/global/hybrid/
naive/mix) -- and back into MethodOutput, including reusing one `Method`
instance across two conversations in sequence (`prepare()`/`answer()`/
`reset()` repeated), the same pattern the real eval runner uses. Not yet
run against a real LLM (no API key configured in this worktree, see
plugins/methods/README.md).
"""
import asyncio
import functools
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import finalize_share_data, initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed


def _record(stats: dict, resp) -> None:
    stats["tokens"] = stats.get("tokens", 0) + resp.tokens_in + resp.tokens_out
    stats["cost"] = stats.get("cost", 0.0) + resp.cost
    stats["latency"] = stats.get("latency", 0.0) + resp.latency


async def _db_llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    response_format=None,
    keyword_extraction=False,
    model: str = "claude-sonnet-5",
    max_tokens: int = 2048,
    stats: dict = None,
    **_ignored,
) -> str:
    full_prompt = prompt
    if history_messages:
        history_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history_messages)
        full_prompt = f"{history_text}\n\n{prompt}"
    if response_format or keyword_extraction:
        full_prompt += "\n\nRespond with ONLY a valid JSON object -- no markdown fences, no extra commentary."
    resp = complete(full_prompt, model=model, system=system_prompt, max_tokens=max_tokens)
    if stats is not None:
        _record(stats, resp)
    return resp.text


async def _db_embedding_func(texts, model: str = "text-embedding-3-small", stats: dict = None, **_ignored) -> np.ndarray:
    resp = embed(list(texts), model=model)
    if stats is not None:
        stats["tokens"] = stats.get("tokens", 0) + resp.tokens_in
        stats["cost"] = stats.get("cost", 0.0) + resp.cost
        stats["latency"] = stats.get("latency", 0.0) + resp.latency
    return np.array(resp.vectors)


def _conversation_text(conversation) -> str:
    if hasattr(conversation, "sessions"):
        parts = []
        for idx in sorted(conversation.sessions):
            parts.append(f"[session {idx}]")
            parts.extend(f"{t.speaker}: {t.text}" for t in conversation.sessions[idx])
        return "\n".join(parts)
    if hasattr(conversation, "turns"):
        return "\n".join(f"{t.speaker}: {t.text}" for t in conversation.turns)
    raise ValueError("lightrag: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat `.turns` list")


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        max_gleanings: int = 1,
        mode: str = "mix",
        response_type: str = "Multiple Paragraphs",
        top_k: int = 40,
        **config,
    ):
        super().__init__(
            model=model,
            embed_model=embed_model,
            max_gleanings=max_gleanings,
            mode=mode,
            response_type=response_type,
            top_k=top_k,
            **config,
        )
        self.model = model
        self.embed_model = embed_model
        self.max_gleanings = max_gleanings
        self.mode = mode
        self.response_type = response_type
        self.top_k = top_k
        self._stats = {}
        self._work_dir = None
        self._rag = None
        self._loop = None
        self._loop_thread = None

    def _run(self, coro):
        """Submit a coroutine to the persistent background-thread loop --
        see the module docstring's "Event-loop bridging" section for why
        this can't just be `asyncio.run()` per call."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def prepare(self, conversation) -> None:
        self._stats = {}
        self._work_dir = Path(tempfile.mkdtemp(prefix="lightrag_"))
        vector_size = len(embed("dimension probe", model=self.embed_model).vectors[0])

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self._rag = LightRAG(
            working_dir=str(self._work_dir),
            llm_model_func=functools.partial(_db_llm_model_func, model=self.model, stats=self._stats),
            embedding_func=EmbeddingFunc(
                embedding_dim=vector_size,
                func=functools.partial(_db_embedding_func, model=self.embed_model, stats=self._stats),
            ),
            entity_extract_max_gleaning=self.max_gleanings,
        )
        self._run(self._rag.initialize_storages())
        self._run(initialize_pipeline_status())
        self._run(self._rag.ainsert(_conversation_text(conversation)))

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        self._stats.clear()
        response = self._run(self._rag.aquery(
            question,
            param=QueryParam(mode=self.mode, response_type=self.response_type, top_k=self.top_k, enable_rerank=False),
        ))

        return MethodOutput(
            answer=(response or "").strip() if isinstance(response, str) else str(response),
            tokens=self._stats.get("tokens"),
            cost=self._stats.get("cost"),
            latency=self._stats.get("latency"),
        )

    def reset(self) -> None:
        if self._rag is not None:
            try:
                self._run(self._rag.finalize_storages())
            except Exception:
                pass
        self._rag = None
        if self._loop is not None:
            # Reset the process-wide shared_storage singleton so the next
            # conversation's prepare() re-initializes cleanly -- see the
            # module docstring's "Event-loop bridging" section.
            finalize_share_data()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=10)
            self._loop.close()
        self._loop = None
        self._loop_thread = None
        if self._work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
