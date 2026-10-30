"""pathrag -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps PathRAG
(BUPT-GAMMA/PathRAG). Same overall idea as plugins/methods/graphrag,
plugins/methods/hipporag, plugins/methods/lightrag, and
plugins/methods/hypergraphrag -- LLM extracts a graph from the conversation,
then answer via graph-aware retrieval -- but PathRAG's actual contribution
is on the retrieval side: instead of pulling in an entity's 1-hop
neighborhood (or a pre-summarized community, or a hyperedge), it finds and
prunes relational *paths* through the graph between query-relevant
entities, so a multi-hop question can be answered by following an explicit
chain of relationships rather than relying on everything relevant already
sitting within one hop.

**Licensing note, read before reproducing this baseline elsewhere**:
BUPT-GAMMA/PathRAG's README claims "This project is licensed under the MIT
License -- see the LICENSE file for details", but no LICENSE file actually
exists in the repository, and GitHub's own license detector reports none
either. This is the same situation already hit and explicitly decided for
plugins/methods/groupmembench_bm25 and plugins/methods/dialsim_bm25's
source datasets (see BASELINE_METRIC_PLAN.md) -- confirmed again with the
user this session and vendored on that same basis: internal reproduction
only, not redistribution.

**Vendored** (`vendor/pathrag/`), no PyPI package exists. Same lineage as
plugins/methods/hypergraphrag (both are old LightRAG-derived codebases,
sharing the exact same `lazy_external_import()` pattern for their
non-default storage backends), so the same low-effort vendoring applies:
Neo4j/Oracle/Milvus/MongoDB/TiDB/Chroma/AGE backends are never actually
imported since this Method always uses the (also upstream-default) local
`JsonKVStorage`/`NanoVectorDBStorage`/`NetworkXStorage`. Three real
problems were found and fixed at vendor time, though, beyond what
`hypergraphrag` needed:
- `pathrag/llm.py` unconditionally imports `modelscope` *and* `from vllm
  import LLM` (reference LLM providers this Method never calls) --
  `hipporag`'s exact problem (vllm pins old torch/transformers that would
  conflict with this shared venv's other baselines), so both import lines
  were deleted, same fix.
- `pathrag/utils.py` did `from PathRAG.prompt import PROMPTS` -- an
  *absolute* self-import that only worked because upstream's own checkout
  happens to be named literally "PathRAG" on `sys.path`. This vendored copy
  is imported as `pathrag` (lowercase, matching every other vendored
  baseline here), so that raised `ModuleNotFoundError`; fixed to the
  relative `from .prompt import PROMPTS` the author clearly meant.
- `pathrag/operate.py`'s `extract_entities()` opened with an unexplained,
  uncommented `time.sleep(20)` -- once per `insert()` call, not per chunk,
  but still a pointless fixed delay with no apparent rate-limiting purpose
  (this repo's own `complete()`/`embed()` already handle retries). Deleted.

Two more real bugs found, **not fixed in the vendored code** (worked around
in this Method instead, since they're both in the class's own public
entry points rather than internals we'd have to reach past regardless):
- `PathRAG.insert()`/`.query()` (the sync convenience wrappers) are
  themselves declared `async def`, and internally do
  `loop.run_until_complete(await self.ainsert(...))` -- awaiting the
  coroutine *before* handing it to `run_until_complete()`, which needs an
  awaitable, not the already-resolved result. Calling either raises a
  `TypeError` immediately. This Method calls `ainsert()`/`aquery()`
  directly instead, driven by its own `asyncio.run()` -- exactly what those
  wrappers were trying (and failing) to do.
- `QueryParam.mode` is typed `Literal["hybrid"]` but its own dataclass
  default value is the string `"global"` -- a mode `aquery()` doesn't
  handle at all (`raise ValueError(f"Unknown mode {param.mode}")`).
  Confirmed empirically: an unmodified `QueryParam()` crashes every query.
  This Method always passes `QueryParam(mode="hybrid")` explicitly.

No event-loop bridging trick was needed (unlike plugins/methods/lightrag):
this is an old, simple codebase whose only per-call state is the storage
backends, with no persistent background worker pool tying anything to a
specific event loop -- confirmed empirically that two separate `PathRAG`
instances constructed and used sequentially in one process (this Method's
real per-conversation usage) both work correctly.

Provider seam: two plain functions, structurally identical to
plugins/methods/lightrag's older pre-rewrite shape and
plugins/methods/hypergraphrag's --
`llm_model_func(prompt, system_prompt=None, history_messages=[],
keyword_extraction=False, **kwargs) -> str` and `EmbeddingFunc(
embedding_dim=, max_token_size=, func=async (texts) -> np.ndarray)`, bound
via `functools.partial`. Unlike `hypergraphrag`, PathRAG's query-time
keyword extraction *does* use a separate structured-JSON prompt (the
standard `{"high_level_keywords": [...], "low_level_keywords": [...]}`
shape also seen in plugins/methods/lightrag) rather than reusing the entity
extraction prompt, signalled via the `keyword_extraction=True` kwarg --
handled the same way as lightrag: an instruction appended to the prompt
telling the model to reply with JSON, no schema validation needed since
PathRAG parses its own JSON downstream. A prompt-substring sniffing
approach (checking for the entity-extraction prompt's own text) would have
been ambiguous here -- that prompt's own step 3 instructions literally
contain the substring "high_level_keywords" as a template placeholder name
(`("content_keywords"<|><high_level_keywords>)`), confirmed by watching a
mock initially misfire on it during testing -- so this Method branches on
the explicit `keyword_extraction` kwarg instead, never on prompt content.
(A second trap for anyone hand-writing a mock LLM for this baseline by
sniffing prompt text instead: the keyword-extraction prompt's own header is
`---Goal---`, which contains entity-extraction's `-Goal-` header as a
literal substring -- a naive `"-Goal-" in prompt` check matches both.)

Known upstream quirk, same class as plugins/methods/hypergraphrag's:
`PathRAG.__post_init__` writes its log file to a bare `"PathRAG.log"`,
landing in whatever the current working directory happens to be rather
than under this Method's per-conversation `working_dir`. Harmless, not
tracked by git, but don't be surprised by a stray `PathRAG.log` wherever
`db run` gets invoked from.

Verification: mock-tested against locomo conversations with `complete()`/
`embed()` monkeypatched (mock LLM emits the entity/relationship
tuple-delimited format PathRAG's own default prompt expects, plus the
keyword-extraction JSON shape when `keyword_extraction=True`) -- covers the
full pipeline: `ainsert()` (chunking, entity/relationship extraction, graph
+ vector upsert) through `aquery()` and back into MethodOutput, including
two conversations in sequence via separate `Method` instances. Not yet run
against a real LLM (no API key configured in this worktree, see
plugins/methods/README.md).
"""
import asyncio
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

from pathrag import PathRAG, QueryParam  # noqa: E402
from pathrag.utils import EmbeddingFunc  # noqa: E402


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
    if keyword_extraction:
        full_prompt += "\n\nRespond with ONLY a valid JSON object -- no markdown fences, no extra commentary."
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
    raise ValueError("pathrag: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat `.turns` list")


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        max_gleanings: int = 1,
        top_k: int = 40,
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
        self._work_dir = Path(tempfile.mkdtemp(prefix="pathrag_"))
        vector_size = len(embed("dimension probe", model=self.embed_model).vectors[0])

        self._rag = PathRAG(
            working_dir=str(self._work_dir),
            llm_model_func=functools.partial(_db_llm_model_func, model=self.model, stats=self._stats),
            embedding_func=EmbeddingFunc(
                embedding_dim=vector_size,
                max_token_size=8192,
                func=functools.partial(_db_embedding_func, model=self.embed_model, stats=self._stats),
            ),
            entity_extract_max_gleaning=self.max_gleanings,
        )
        asyncio.run(self._rag.ainsert(_conversation_text(conversation)))

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        self._stats.clear()
        response = asyncio.run(self._rag.aquery(
            question,
            # QueryParam()'s own default (mode="global") isn't a mode aquery()
            # handles at all -- see module docstring. "hybrid" is the only
            # mode PathRAG's aquery() implements.
            param=QueryParam(mode="hybrid", top_k=self.top_k, response_type=self.response_type),
        ))

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
