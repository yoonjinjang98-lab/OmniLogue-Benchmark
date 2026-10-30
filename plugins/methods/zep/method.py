"""zep -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps graphiti-core
(getzep/graphiti, the open-source temporal knowledge graph engine behind
"Zep: A Temporal Knowledge Graph Architecture for Agent Memory",
arXiv:2501.13956), Apache-2.0. Named `zep` here (not `graphiti`) since that's
the name the paper and benchmark comparisons use -- Zep the hosted product
and graphiti-core the open engine are the same underlying codebase, so
there's no PyPI-name mismatch to flag the way plugins/methods/memoryos
needed one.

Ingests each conversation chunk as an "episode" via `Graphiti.add_episode()`
(LLM-driven entity/relation extraction into a temporal knowledge graph, per
the paper), answers via `Graphiti.search()` (hybrid vector+text search over
graph edges) feeding the retrieved facts into a QA prompt.

All three of graphiti-core's external-call seams are constructor-injectable
on `Graphiti(...)` -- no monkeypatching needed, unlike every other family-C
baseline here:
- `llm_client: LLMClient` -- `_DBLLMClient` implements the library's one
  abstract method, `_generate_response()`, routing through
  `dialogue_benchmarks.llm.complete()`. The base class already appends the
  JSON schema instruction to the message text before calling this, so (unlike
  langmem) no tool-calling/structured-output emulation is needed here.
- `embedder: EmbedderClient` -- `_DBEmbedder` routes `create()`/
  `create_batch()` through `dialogue_benchmarks.llm.embed()`.
- `cross_encoder: CrossEncoderClient` -- graphiti-core's default
  (`OpenAIRerankerClient`) eagerly builds a real `AsyncOpenAI()` client at
  `Graphiti.__init__` time regardless of whether reranking ever actually
  runs (the same "even a dummy key isn't enough, must supply your own"
  problem already hit in plugins/methods/mem0 and a_mem), so this always
  supplies `_DBCrossEncoder`: a plain cosine-similarity ranker over
  `dialogue_benchmarks.llm.embed()` vectors, since graphiti-core's own
  contract only needs `[(passage, score), ...]` back.
- `graph_driver` -- `KuzuDriver`, graphiti-core's embedded, no-server graph
  database backend (same "keep it local, no external service" pattern as
  every other backend used across mem0/a_mem: Qdrant, Chroma both run
  local-file/in-process too). Upstream marks this driver deprecated ("the
  upstream Kuzu project is no longer maintained... migrate to Neo4j or
  FalkorDB"), but both alternatives need a real running server process,
  which would be a new kind of infrastructure dependency for this repo -- so
  this uses Kuzu anyway, despite the warning.

graphiti-core's whole API is async (`add_episode`/`search`/`close`/...);
this Method bridges it with `asyncio.run()` since the rest of this repo's
Method contract is synchronous.
"""
import asyncio
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graph_queries import get_fulltext_indices
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.nodes import EpisodeType

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_CHUNK_WINDOW = 20  # turns per add_episode() call for flat `.turns` datasets


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


class _DBLLMClient(LLMClient):
    def __init__(self, model: str):
        super().__init__(LLMConfig(model=model), cache=False)
        self.model = model

    async def _generate_response(
        self, messages, response_model=None, max_tokens: int = DEFAULT_MAX_TOKENS, model_size: ModelSize = ModelSize.medium
    ) -> dict:
        system = None
        turns = []
        for m in messages:
            if m.role == "system" and system is None:
                system = m.content
            else:
                turns.append(f"{m.role}: {m.content}")
        resp = complete("\n\n".join(turns), model=self.model, system=system, max_tokens=max_tokens)
        try:
            return json.loads(_strip_fences(resp.text))
        except (json.JSONDecodeError, TypeError):
            return {}


class _DBEmbedder(EmbedderClient):
    def __init__(self, model: str):
        self.model = model

    async def create(self, input_data) -> list:
        text = input_data if isinstance(input_data, str) else " ".join(str(x) for x in input_data)
        return embed(text, model=self.model).vectors[0]

    async def create_batch(self, input_data_list) -> list:
        return embed(list(input_data_list), model=self.model).vectors


class _DBCrossEncoder(CrossEncoderClient):
    """graphiti-core's default cross encoder eagerly builds a real
    AsyncOpenAI() client at Graphiti.__init__ time -- always supply this
    instead. A plain cosine-similarity ranker over embed() vectors satisfies
    the `[(passage, score), ...]` contract without adding a new call shape."""

    def __init__(self, model: str):
        self.model = model

    async def rank(self, query: str, passages: list) -> list:
        if not passages:
            return []
        vectors = embed([query] + list(passages), model=self.model).vectors
        qv = vectors[0]
        q_norm = sum(x * x for x in qv) ** 0.5 or 1.0
        scored = []
        for passage, vec in zip(passages, vectors[1:]):
            v_norm = sum(x * x for x in vec) ** 0.5 or 1.0
            score = sum(a * b for a, b in zip(qv, vec)) / (q_norm * v_norm)
            scored.append((passage, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


def _extract_chunks(conversation) -> list:
    """Same chunking convention as running_summary/mem0/langmem/a_mem: one
    chunk per session if session-grouped, else fixed-size windows over flat
    `.turns`."""
    if hasattr(conversation, "sessions"):
        return [
            "\n".join(f"{turn.speaker}: {turn.text}" for turn in conversation.sessions[idx])
            for idx in sorted(conversation.sessions)
        ]
    if hasattr(conversation, "turns"):
        turns = conversation.turns
        return [
            "\n".join(f"{t.speaker}: {t.text}" for t in turns[start : start + _CHUNK_WINDOW])
            for start in range(0, len(turns), _CHUNK_WINDOW)
        ]
    raise ValueError(
        "zep: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract chunks for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        top_k: int = 10,
        max_tokens: int = 512,
        **config,
    ):
        super().__init__(model=model, embed_model=embed_model, top_k=top_k, max_tokens=max_tokens, **config)
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._graphiti = None
        self._path = None

    def prepare(self, conversation) -> None:
        self._path = tempfile.mkdtemp(prefix="zep_")
        driver = KuzuDriver(db=str(Path(self._path) / "graph.kuzu"))
        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=_DBLLMClient(self.model),
            embedder=_DBEmbedder(self.embed_model),
            cross_encoder=_DBCrossEncoder(self.embed_model),
        )
        asyncio.run(self._graphiti.build_indices_and_constraints())
        # KuzuDriver.build_indices_and_constraints() is a documented no-op ("indices
        # are created during setup_schema()"), but setup_schema()'s SCHEMA_QUERIES only
        # creates tables, never the full-text-search indices search() later requires --
        # graphiti-core's own KuzuGraphMaintenanceOperations.build_indices_and_constraints
        # (driver/kuzu/operations/graph_ops.py) does create them via this exact same
        # get_fulltext_indices() helper, it's just never reached through
        # Graphiti.build_indices_and_constraints() -> driver.build_indices_and_constraints
        # for the Kuzu driver. Create them directly to work around the gap.
        for query in get_fulltext_indices(GraphProvider.KUZU):
            asyncio.run(driver.execute_query(query))
        for i, chunk in enumerate(_extract_chunks(conversation)):
            asyncio.run(
                self._graphiti.add_episode(
                    name=f"chunk_{i}",
                    episode_body=chunk,
                    source_description="conversation",
                    reference_time=datetime.now(timezone.utc),
                    source=EpisodeType.message,
                )
            )

    def answer(self, item) -> MethodOutput:
        edges = asyncio.run(self._graphiti.search(item.question, num_results=self.top_k))
        retrieved = [e.fact for e in edges]

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        passages = "\n".join(f"- {m}" for m in retrieved) or "(no facts retrieved)"
        prompt = (
            "Answer the question using only the information in the facts below. "
            "Be concise and direct -- answer with just the fact asked for, no extra explanation.\n\n"
            f"### Facts\n{passages}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        return MethodOutput(
            answer=resp.text.strip(),
            retrieved=retrieved,
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )

    def reset(self) -> None:
        if self._graphiti is not None:
            try:
                asyncio.run(self._graphiti.close())
            except Exception:
                pass
        self._graphiti = None
        if self._path:
            shutil.rmtree(self._path, ignore_errors=True)
        self._path = None
