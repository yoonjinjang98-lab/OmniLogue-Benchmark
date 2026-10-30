"""hypergraphrag -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps
HyperGraphRAG (LHRLAB/HyperGraphRAG, MIT, NeurIPS 2025 --
"HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured
Knowledge Representation"). Same overall idea as plugins/methods/graphrag,
plugins/methods/hipporag, and plugins/methods/lightrag -- LLM extracts a
graph from the conversation, then answer via graph-aware retrieval -- but
this is the one baseline in that group whose actual research contribution
matters for the extraction step: instead of a binary (source_entity,
target_entity, description) relation, each extracted "hyper-relation" is a
knowledge-segment sentence (with a completeness score) that can link an
arbitrary number of entities at once -- a true hyperedge, not a graph edge.
Querying (`QueryParam`'s only implemented mode, "hybrid", the library's own
default) does vector search over entities and hyperedges together, using
the graph structure both attach to for context expansion.

**Vendored** (`vendor/hypergraphrag/`, MIT -- see `vendor/LICENSE.txt`), no
PyPI package exists. Unlike plugins/methods/hipporag's vendoring (which had
to drop files because the pip pre-release hard-required `vllm` just to
import), nothing needed trimming here: the upstream repo's non-default
storage backends (Neo4j/Oracle/Milvus/MongoDB/TiDB/Chroma) are all behind
`lazy_external_import()`, a small helper the library itself already ships
that defers the actual `importlib.import_module()` call until one of those
backends is *instantiated* -- since this Method always uses the (also
upstream) defaults (`JsonKVStorage`/`NanoVectorDBStorage`/`NetworkXStorage`,
the same "local, no external service" choice as every other family-C
baseline here), none of those heavy driver imports (`neo4j`, `oracledb`,
`pymilvus`, `pymongo`, `pymysql`, `chromadb`) ever fire. The one real gap:
`hypergraphrag/llm.py` unconditionally imports `torch`/`transformers`
(already present in this venv for hipporag/paa) plus `ollama`/`aioboto3`
(added as this baseline's own pip extra, both lightweight clients with no
version-pinning conflicts to worry about, unlike hipporag's vllm problem)
purely for reference LLM-provider implementations this Method never calls.

Provider seam: two plain functions, structurally identical to
plugins/methods/lightrag's older pre-rewrite shape --
`llm_model_func(prompt, system_prompt=None, history_messages=[],
keyword_extraction=False, **kwargs) -> str` and `EmbeddingFunc(
embedding_dim=, max_token_size=, func=async (texts) -> np.ndarray)`,
bound via `functools.partial`. `keyword_extraction=True` is passed for the
query-time keyword-extraction call, but -- confirmed by reading
`operate.py` -- HyperGraphRAG actually re-runs the *same* hyper-relation/
entity extraction prompt against the query text for that step rather than
a separate JSON keyword schema (`PROMPTS["keywords_extraction"]` exists in
the vendored code but is dead: never referenced from `operate.py`), so no
schema/response_format handling is needed here at all, just the same
prompt-detection branch used for indexing-time extraction.

Unlike plugins/methods/lightrag, no event-loop bridging trick is needed:
this is an older, simpler codebase (predating LightRAG's async worker-pool/
shared-storage-singleton rewrite) -- `insert()`/`query()` are already
synchronous wrappers that grab-or-create one event loop and
`run_until_complete()` per call, with no persistent background workers
tying storage to a specific loop. Confirmed empirically: two separate
`HyperGraphRAG` instances constructed and used sequentially in the same
process (matching this Method's real prepare/answer/reset per conversation
usage) both work correctly with no cross-instance interference, unlike
lightrag's process-wide singleton problem.

Known upstream quirk: `HyperGraphRAG.__post_init__` builds its log file
path as `os.path.join("hypergraphrag.log")` -- a no-op join, so it always
writes to `hypergraphrag.log` in whatever the process's current working
directory happens to be, not under `working_dir` like every other artifact
this Method already puts in its own per-conversation temp dir. Harmless
(just a log file, and it gets overwritten/appended across runs), but don't
be surprised by a stray `hypergraphrag.log` showing up wherever `db run` is
invoked from -- it isn't this Method's own doing, and isn't tracked by git.

Verification: mock-tested against locomo conversations with `complete()`/
`embed()` monkeypatched (mock LLM emits the hyper-relation/entity
tuple-delimited format HyperGraphRAG's own default prompt expects) --
covers the full pipeline: `insert()` (chunking, hyperedge/entity
extraction, graph + vector upsert) through `query()` and back into
MethodOutput, including two conversations in sequence via separate `Method`
instances. Not yet run against a real LLM (no API key configured in this
worktree, see plugins/methods/README.md).
"""
import functools
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from hypergraphrag import HyperGraphRAG, QueryParam  # noqa: E402
from hypergraphrag.utils import EmbeddingFunc  # noqa: E402


def _record(stats: dict, resp) -> None:
    stats["tokens"] = stats.get("tokens", 0) + resp.tokens_in + resp.tokens_out
    stats["cost"] = stats.get("cost", 0.0) + resp.cost
    stats["latency"] = stats.get("latency", 0.0) + resp.latency


async def _db_llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    model: str = "claude-sonnet-5",
    max_tokens: int = 2048,
    stats: dict = None,
    **_ignored,
) -> str:
    full_prompt = prompt
    if history_messages:
        history_text = "\n".join(
            (m.get("role", "") + ": " + m.get("content", "")) if isinstance(m, dict) else str(m)
            for m in history_messages
        )
        full_prompt = f"{history_text}\n\n{prompt}"
    resp = complete(full_prompt, model=model, system=system_prompt, max_tokens=max_tokens)
    if stats is not None:
        _record(stats, resp)
    return resp.text


async def _db_embedding_func(texts, model: str = "text-embedding-3-small", stats: dict = None) -> np.ndarray:
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
    raise ValueError("hypergraphrag: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat `.turns` list")


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        max_gleanings: int = 2,
        top_k: int = 60,
        response_type: str = "Multiple Paragraphs",
        **config,
    ):
        super().__init__(
            model=model,
            embed_model=embed_model,
            max_gleanings=max_gleanings,
            top_k=top_k,
            response_type=response_type,
            **config,
        )
        self.model = model
        self.embed_model = embed_model
        self.max_gleanings = max_gleanings
        self.top_k = top_k
        self.response_type = response_type
        self._stats = {}
        self._work_dir = None
        self._rag = None

    def prepare(self, conversation) -> None:
        self._stats = {}
        self._work_dir = Path(tempfile.mkdtemp(prefix="hypergraphrag_"))
        vector_size = len(embed("dimension probe", model=self.embed_model).vectors[0])

        self._rag = HyperGraphRAG(
            working_dir=str(self._work_dir),
            llm_model_func=functools.partial(_db_llm_model_func, model=self.model, stats=self._stats),
            embedding_func=EmbeddingFunc(
                embedding_dim=vector_size,
                max_token_size=8192,
                func=functools.partial(_db_embedding_func, model=self.embed_model, stats=self._stats),
            ),
            entity_extract_max_gleaning=self.max_gleanings,
        )
        self._rag.insert(_conversation_text(conversation))

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        self._stats.clear()
        response = self._rag.query(
            question,
            param=QueryParam(top_k=self.top_k, response_type=self.response_type),
        )

        return MethodOutput(
            answer=(response or "").strip() if isinstance(response, str) else str(response),
            tokens=self._stats.get("tokens"),
            cost=self._stats.get("cost"),
            latency=self._stats.get("latency"),
        )

    def reset(self) -> None:
        self._rag = None
        if self._work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
