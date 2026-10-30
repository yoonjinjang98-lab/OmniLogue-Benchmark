"""graphrag -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps
Microsoft's GraphRAG (microsoft/graphrag), MIT licensed. LLM-driven entity/
relationship extraction builds a knowledge graph from the conversation,
Leiden community detection (graspologic, no LLM) groups it hierarchically,
an LLM writes a report per community, and querying is either `local_search`
(entity-anchored retrieval over the graph + reports, the default here -- the
closest analogue to hipporag's/zep's targeted retrieval) or `global_search`
(map-reduce over every community report, better for "summarize everything"
questions -- set `search_mode="global"`).

Unlike hipporag/hypermem, this needed no vendoring: GraphRAG 3.x ships as a
regular pip package (`graphrag`, pulling in `graphrag_llm`/`graphrag_storage`/
`graphrag_vectors`/`graphrag_input`/`graphrag_chunking`/`graphrag_common`) and
its LLM/embedding seams are an *officially documented* plugin API --
`graphrag_llm.completion.register_completion()` / `graphrag_llm.embedding.
register_embedding()` -- rather than something to monkeypatch or vendor
around, the cleanest integration point found across every family-C baseline
this session. `_DBLLMCompletion`/`_DBLLMEmbedding` implement the library's
`LLMCompletion`/`LLMEmbedding` ABCs and are registered once at import time
under the custom provider type "dialogue_benchmarks"; `ModelConfig.model`
carries this repo's real model name straight through (`extra="allow"` on
that pydantic model is itself documented as existing "to support custom LLM
provider implementations"), so no monkeypatching or per-instance closures are
needed either.

Two integration wrinkles worth flagging:
- `local_search`/`global_search` always call `completion_async(stream=True)`
  for the final answer generation (even the "non-streaming" `local_search()`
  API is implemented as `local_search_streaming()` accumulated into a
  string) -- `_DBLLMCompletion` handles this by calling
  `dialogue_benchmarks.llm.complete()` once (it isn't natively streaming
  either) and yielding the whole answer as a single chunk, same "one big
  chunk" shape other non-streaming backends get wrapped in.
- Community-report generation calls `completion_async(response_format=
  CommunityReportResponse)` (structured JSON mode). There's no tool-calling
  emulation needed here (unlike plugins/methods/langmem) -- the JSON schema
  is just appended to the prompt text and the reply is parsed with
  `response_format.model_validate(json.loads(...))`; a parse failure is
  swallowed by GraphRAG's own `community_reports_extractor.py` (already
  wrapped in try/except, degrades to an empty report) rather than by this
  code, so a schema-parsing miss here fails soft exactly like an upstream
  LLM including markdown fences a stricter parser would reject.

Token/cost/latency accounting: `ModelConfig` gets one extra field, `db_stats`
-- a plain dict shared by reference between every completion/embedding
instance the library transiently constructs during one `prepare()`/
`answer()` call (`register()`'s default scope is "transient": a fresh
instance per internal call, but `model_config.model_extra["db_stats"]` is
the same dict object every time). `answer()` clears it before searching and
reports the accumulated totals in `MethodOutput` -- ingestion-time
(`prepare()`) LLM calls are deliberately not reported, matching every other
family-C baseline here (mem0/zep/hipporag/hypermem also only cost-track the
final per-question completion, not indexing).

Everything else uses GraphRAG's own defaults: LanceDB (embedded, file-based,
no external server -- the same "local, no separate service" choice as
mem0/Qdrant, a_mem/Chroma, zep/hipporag's Kuzu) for the vector store, and
the standard indexing pipeline's own chunking/extraction/community-detection
parameters. `vector_store.vector_size` is set dynamically from one throwaway
`embed()` call rather than hardcoded, since GraphRAG's own default (3072,
matching its own default `text-embedding-3-large`) would silently mismatch
this repo's default `text-embedding-3-small` (1536) and reject every write.

Verification: mock-tested against locomo conversations with `complete()`/
`embed()` monkeypatched to canned responses (entity/relationship extraction
replies with the exact tuple-delimited format GraphRAG's own default prompt
expects, community reports with valid `CommunityReportResponse` JSON) --
covers the full pipeline: `build_index()` (chunking, extraction, Leiden
community detection, report generation, text/graph embedding) through
`local_search()`/`global_search()` and back into `MethodOutput`. Not yet run
against a real LLM (no API key configured in this worktree, see
plugins/methods/README.md) -- the real model's raw-text replies to
GraphRAG's own upstream extraction prompts are the one thing a mock can't
fully stand in for, same caveat noted for every other baseline here.
"""
import asyncio
import json
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import graphrag.api as graphrag_api
from graphrag.config.models.extract_graph_config import ExtractGraphConfig
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag_llm.completion import LLMCompletion, register_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.embedding import LLMEmbedding, register_embedding
from graphrag_llm.metrics.noop_metrics_store import NoopMetricsStore
from graphrag_llm.types import (
    LLMChoice,
    LLMChoiceChunk,
    LLMChoiceDelta,
    LLMCompletionChunk,
    LLMCompletionMessage,
    LLMCompletionResponse,
    LLMEmbeddingResponse,
    LLMEmbeddingUsage,
)
from graphrag_llm.types import LLMEmbedding as LLMEmbeddingData
from graphrag_storage import StorageConfig, create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider
from graphrag_vectors import VectorStoreConfig

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_PROVIDER_TYPE = "dialogue_benchmarks"
_OUTPUT_TABLES = ["entities", "communities", "community_reports", "text_units", "relationships"]


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _flatten_messages(messages):
    """GraphRAG passes either a plain string or a list of OpenAI-format
    message dicts -- `dialogue_benchmarks.llm.complete()` wants one prompt
    string plus an optional system string, same flattening as
    plugins/methods/zep's `_DBLLMClient._generate_response`."""
    if isinstance(messages, str):
        return None, messages
    system = None
    parts = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role == "system" and system is None:
            system = content
        else:
            parts.append(f"{role}: {content}")
    return system, "\n\n".join(parts)


def _record(stats: dict, resp) -> None:
    stats["tokens"] = stats.get("tokens", 0) + resp.tokens_in + resp.tokens_out
    stats["cost"] = stats.get("cost", 0.0) + resp.cost
    stats["latency"] = stats.get("latency", 0.0) + resp.latency


class _DBLLMCompletion(LLMCompletion):
    def __init__(
        self,
        *,
        model_id,
        model_config: ModelConfig,
        tokenizer,
        metrics_store=None,
        metrics_processor=None,
        rate_limiter=None,
        retrier=None,
        cache=None,
        cache_key_creator=None,
        db_stats: dict = None,
        **kwargs,
    ):
        self._model = model_config.model
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store or NoopMetricsStore()
        self._stats = db_stats if db_stats is not None else {}

    @property
    def metrics_store(self):
        return self._metrics_store

    @property
    def tokenizer(self):
        return self._tokenizer

    def _complete_once(self, *, messages, response_format=None, max_completion_tokens=None, max_tokens=None, temperature=None, **_ignored):
        system, prompt = _flatten_messages(messages)
        if response_format is not None:
            prompt = (
                f"{prompt}\n\nRespond with ONLY a JSON object matching this schema "
                f"(no markdown fences, no extra commentary):\n{json.dumps(response_format.model_json_schema())}"
            )
        resp = complete(
            prompt,
            model=self._model,
            system=system,
            max_tokens=max_completion_tokens or max_tokens or 1024,
            temperature=temperature,
        )
        _record(self._stats, resp)
        formatted = None
        if response_format is not None:
            try:
                formatted = response_format.model_validate(json.loads(_strip_fences(resp.text)))
            except Exception:
                formatted = None
        return LLMCompletionResponse(
            id=f"db-{uuid.uuid4()}",
            choices=[LLMChoice(finish_reason="stop", index=0, message=LLMCompletionMessage(role="assistant", content=resp.text))],
            created=int(time.time()),
            model=self._model,
            object="chat.completion",
            formatted_response=formatted,
        )

    async def _stream_chunks(self, **kwargs):
        kwargs.pop("stream", None)
        response = self._complete_once(**kwargs)
        yield LLMCompletionChunk(
            id=response.id,
            choices=[LLMChoiceChunk(index=0, delta=LLMChoiceDelta(content=response.content, role="assistant"), finish_reason="stop")],
            created=response.created,
            model=self._model,
            object="chat.completion.chunk",
        )

    def completion(self, /, **kwargs):
        if kwargs.get("stream"):
            raise NotImplementedError("graphrag baseline: sync streaming unused by local_search/global_search (they call completion_async)")
        return self._complete_once(**kwargs)

    async def completion_async(self, /, **kwargs):
        if kwargs.get("stream"):
            return self._stream_chunks(**kwargs)
        return self._complete_once(**kwargs)


class _DBLLMEmbedding(LLMEmbedding):
    def __init__(
        self,
        *,
        model_id,
        model_config: ModelConfig,
        tokenizer,
        metrics_store=None,
        metrics_processor=None,
        rate_limiter=None,
        retrier=None,
        cache=None,
        cache_key_creator=None,
        db_stats: dict = None,
        **kwargs,
    ):
        self._model = model_config.model
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store or NoopMetricsStore()
        self._stats = db_stats if db_stats is not None else {}

    @property
    def metrics_store(self):
        return self._metrics_store

    @property
    def tokenizer(self):
        return self._tokenizer

    def _embed_once(self, *, input, **_ignored):
        resp = embed(list(input), model=self._model)
        self._stats["tokens"] = self._stats.get("tokens", 0) + resp.tokens_in
        self._stats["cost"] = self._stats.get("cost", 0.0) + resp.cost
        self._stats["latency"] = self._stats.get("latency", 0.0) + resp.latency
        data = [LLMEmbeddingData(index=i, embedding=vec, object="embedding") for i, vec in enumerate(resp.vectors)]
        return LLMEmbeddingResponse(data=data, model=self._model, object="list", usage=LLMEmbeddingUsage(prompt_tokens=resp.tokens_in, total_tokens=resp.tokens_in))

    def embedding(self, /, **kwargs):
        return self._embed_once(**kwargs)

    async def embedding_async(self, /, **kwargs):
        return self._embed_once(**kwargs)


register_completion(_PROVIDER_TYPE, _DBLLMCompletion)
register_embedding(_PROVIDER_TYPE, _DBLLMEmbedding)


def _conversation_text(conversation) -> str:
    if hasattr(conversation, "sessions"):
        parts = []
        for idx in sorted(conversation.sessions):
            parts.append(f"[session {idx}]")
            parts.extend(f"{t.speaker}: {t.text}" for t in conversation.sessions[idx])
        return "\n".join(parts)
    if hasattr(conversation, "turns"):
        return "\n".join(f"{t.speaker}: {t.text}" for t in conversation.turns)
    raise ValueError("graphrag: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat `.turns` list")


def _summarize_context(context) -> list:
    """Best-effort human-readable trace for MethodOutput.retrieved -- context
    is a dict of DataFrames (entities/reports/relationships/sources/...),
    shape depends on search_mode. Not used for scoring, diagnostic only."""
    if not isinstance(context, dict):
        return []
    retrieved = []
    for key, df in context.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        text_col = next((c for c in ("description", "content", "text", "title", "summary") if c in df.columns), None)
        if text_col:
            retrieved.extend(f"[{key}] {v}" for v in df[text_col].head(10).tolist())
    return retrieved


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        max_gleanings: int = 1,
        search_mode: str = "local",
        community_level: int = 1,
        response_type: str = "multiple paragraphs",
        **config,
    ):
        super().__init__(
            model=model,
            embed_model=embed_model,
            max_gleanings=max_gleanings,
            search_mode=search_mode,
            community_level=community_level,
            response_type=response_type,
            **config,
        )
        self.model = model
        self.embed_model = embed_model
        self.max_gleanings = max_gleanings
        self.search_mode = search_mode
        self.community_level = community_level
        self.response_type = response_type
        self._stats = {}
        self._work_dir = None
        self._config = None
        self._dataframes = None

    def _model_config(self, model_name: str) -> ModelConfig:
        return ModelConfig(
            type=_PROVIDER_TYPE,
            model_provider=_PROVIDER_TYPE,
            model=model_name,
            metrics=None,
            db_stats=self._stats,
        )

    def _read_outputs(self) -> dict:
        storage_obj = create_storage(self._config.output_storage)
        table_provider = create_table_provider(self._config.table_provider, storage=storage_obj)
        reader = DataReader(table_provider)
        return {name: asyncio.run(getattr(reader, name)()) for name in _OUTPUT_TABLES}

    def prepare(self, conversation) -> None:
        self._stats = {}
        self._work_dir = Path(tempfile.mkdtemp(prefix="graphrag_"))
        vector_size = len(embed("dimension probe", model=self.embed_model).vectors[0])

        self._config = GraphRagConfig(
            completion_models={"default_completion_model": self._model_config(self.model)},
            embedding_models={"default_embedding_model": self._model_config(self.embed_model)},
            input_storage=StorageConfig(base_dir=str(self._work_dir / "input")),
            output_storage=StorageConfig(base_dir=str(self._work_dir / "output")),
            vector_store=VectorStoreConfig(db_uri=str(self._work_dir / "lancedb"), vector_size=vector_size),
            extract_graph=ExtractGraphConfig(max_gleanings=self.max_gleanings),
        )

        input_documents = pd.DataFrame([{
            "id": "doc-0",
            "text": _conversation_text(conversation),
            "title": "conversation",
            "creation_date": datetime.now(timezone.utc).isoformat(),
        }])
        asyncio.run(graphrag_api.build_index(self._config, input_documents=input_documents))
        self._dataframes = self._read_outputs()

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        self._stats.clear()
        if self.search_mode == "global":
            response, context = asyncio.run(graphrag_api.global_search(
                config=self._config,
                entities=self._dataframes["entities"],
                communities=self._dataframes["communities"],
                community_reports=self._dataframes["community_reports"],
                community_level=self.community_level,
                dynamic_community_selection=False,
                response_type=self.response_type,
                query=question,
            ))
        else:
            response, context = asyncio.run(graphrag_api.local_search(
                config=self._config,
                entities=self._dataframes["entities"],
                communities=self._dataframes["communities"],
                community_reports=self._dataframes["community_reports"],
                text_units=self._dataframes["text_units"],
                relationships=self._dataframes["relationships"],
                covariates=None,
                community_level=self.community_level,
                response_type=self.response_type,
                query=question,
            ))

        return MethodOutput(
            answer=(response or "").strip() if isinstance(response, str) else str(response),
            retrieved=_summarize_context(context),
            tokens=self._stats.get("tokens"),
            cost=self._stats.get("cost"),
            latency=self._stats.get("latency"),
        )

    def reset(self) -> None:
        self._config = None
        self._dataframes = None
        if self._work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
