"""hipporag -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps HippoRAG
(OSU-NLP-Group/HippoRAG, MIT), a knowledge-graph-based RAG framework -- LLM-driven
open information extraction (NER + triple extraction) builds a passage/entity/
fact graph, indexed and queried via a mix of embedding similarity and
Personalized PageRank over the graph.

Vendored under vendor/hipporag/ (not pip-installed) for two reasons:
- The pip package (`HippoRAG`, pre-release 2.0.0a4) fails to import at all
  without the real `vllm` package present: `hipporag/__init__.py` ->
  `HippoRAG.py` unconditionally imports `openie_vllm_offline.py`, which
  imports `llm/vllm_offline.py`, which does `from vllm import ...` --- even
  though this method always uses `openie_mode="online"` and never touches
  vLLM. `vllm` itself pins exact old `torch==2.5.1`/`transformers==4.45.2`,
  which would downgrade the newer versions plugins/methods/a_mem,
  plugins/methods/memoryos, and plugins/methods/paa already depend on in
  this shared venv.
- Vendoring lets those two files (plus `embedding_model/GritLM.py`, needing
  the separate `gritlm` package for a backend this method never selects) be
  dropped outright, same trimming precedent as plugins/methods/raptor.

Two seams route through this repo's shared LLM/embedding client:
- `hipporag.llm._get_llm_class(config) -> BaseLLM instance` -- `_DBLLM`
  implements the real (not the possibly-stale ABC-documented) contract used
  throughout the codebase: `infer(messages) -> (text: str, {"finish_reason":
  ..., ...}, cache_hit: bool)`, routed through `dialogue_benchmarks.llm.complete()`.
- `hipporag.embedding_model._get_embedding_model_class(name) -> BaseEmbeddingModel
  class` -- `_DBEmbeddingModel.batch_encode(texts) -> np.ndarray`, routed
  through `dialogue_benchmarks.llm.embed()`.

Both factories have to be patched *before* `hipporag.HippoRAG` is imported
for the first time (`HippoRAG.py` does `from .llm import _get_llm_class`,
`from .embedding_model import _get_embedding_model_class` at its own module
level, binding a snapshot -- patching after the fact wouldn't reach it), and
`hipporag/__init__.py` itself does `from .HippoRAG import HippoRAG`
immediately, so even `import hipporag.llm` (to reach in and patch it) would
run that first and bind the real, unpatched factories -- the same problem
plugins/methods/memoryos hit, solved the same way: insert a bare
`sys.modules["hipporag"]` package stub first (just `__path__`, `__init__.py`
body never runs) so `hipporag.llm`/`hipporag.embedding_model` load and get
patched in isolation, then drop the stub so the real package import
proceeds and picks up the already-patched submodules from sys.modules.

Unlike mem0/a_mem (whose factory-produced objects get captured by *other*
internal objects at construction time, e.g. mem0's Memory holding onto
`.retriever`), patching the factory here is enough on its own: HippoRAG.py
constructs `self.llm_model`/`self.embedding_model` via these exact
factories inside `__init__`, and everything downstream (`OpenIE`,
`EmbeddingStore` x3) receives those already-correct objects at construction
time -- there's no later swap-after-the-fact step needed.
"""
import sys
import types
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_CHUNK_WINDOW = 20  # turns per index() document for flat `.turns` datasets
_EMBED_DIM = 1536


def _load_hipporag_class():
    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))

    if "hipporag.llm" not in sys.modules:
        import importlib.util

        pkg_spec = importlib.util.find_spec("hipporag")  # locates only, doesn't execute __init__.py
        pkg_dir = Path(pkg_spec.submodule_search_locations[0])

        stub = types.ModuleType("hipporag")
        stub.__path__ = [str(pkg_dir)]
        sys.modules["hipporag"] = stub

        for sub in ("llm", "embedding_model"):
            sub_dir = pkg_dir / sub
            spec = importlib.util.spec_from_file_location(
                f"hipporag.{sub}", sub_dir / "__init__.py", submodule_search_locations=[str(sub_dir)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"hipporag.{sub}"] = module
            spec.loader.exec_module(module)

        sys.modules["hipporag.llm"]._get_llm_class = lambda config: _DBLLM(config)
        sys.modules["hipporag.embedding_model"]._get_embedding_model_class = lambda embedding_model_name=None: _DBEmbeddingModel

        del sys.modules["hipporag"]

    from hipporag import HippoRAG

    return HippoRAG


class _DBLLM:
    """Duck-types hipporag.llm.base.BaseLLM's real contract (see module
    docstring for why the ABC's own type hints don't match it)."""

    def __init__(self, global_config):
        self.global_config = global_config
        self.model = global_config.llm_name
        self.llm_config = None

    def _init_llm_config(self) -> None:
        self.llm_config = None

    def infer(self, messages, **kwargs):
        system = None
        turns = []
        for m in messages:
            role, content = m.get("role"), m.get("content", "")
            if role == "system" and system is None:
                system = content
            else:
                turns.append(f"{role}: {content}")
        resp = complete("\n\n".join(turns), model=self.model, system=system, max_tokens=1024)
        metadata = {"prompt_tokens": resp.tokens_in, "completion_tokens": resp.tokens_out, "finish_reason": "stop"}
        return resp.text, metadata, resp.cached

    def batch_infer(self, batch_chat, **kwargs):
        texts, metadatas = [], []
        for chat in batch_chat:
            text, metadata, _ = self.infer(chat)
            texts.append(text)
            metadatas.append(metadata)
        return texts, metadatas


class _DBEmbeddingModel:
    """Duck-types hipporag.embedding_model.base.BaseEmbeddingModel."""

    def __init__(self, global_config=None, embedding_model_name=None):
        self.global_config = global_config
        self.embedding_model_name = embedding_model_name or "text-embedding-3-small"
        self.embedding_dim = _EMBED_DIM

    def batch_encode(self, texts, **kwargs):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        return np.array(embed(list(texts), model=self.embedding_model_name).vectors)


def _extract_docs(conversation) -> list:
    """One document per session if session-grouped (e.g. locomo), else
    fixed-size windows over flat `.turns` -- same convention as
    running_summary/mem0/langmem/a_mem/zep, but HippoRAG.index() wants plain
    document strings rather than turn objects."""
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
        "hipporag: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract documents for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        # HippoRAG's own paper default. Needs transformers==4.46.3 pinned in
        # this worktree's venv (see pyproject.toml's hipporag extra) -- the
        # newer transformers this repo otherwise runs has three internal API
        # changes nvidia/NV-Embed-v2's trust_remote_code modeling code
        # predates (dialogue_benchmarks.llm's _patch_* functions cover those
        # three, version-gated to a no-op on <5.0 since 4.46.3 doesn't need
        # them at all), and it's loaded via plain transformers.AutoModel +
        # its own .encode() rather than SentenceTransformer(model_name) --
        # see llm.py's _call_nvembed_direct docstring for why. Live-verified
        # end to end: real 4096-dim embeddings, sane values.
        embed_model: str = "nvidia/NV-Embed-v2",
        top_k: int = 10,
        **config,
    ):
        super().__init__(model=model, embed_model=embed_model, top_k=top_k, **config)
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self._HippoRAG = _load_hipporag_class()
        self._hippo = None
        self._path = None

    def prepare(self, conversation) -> None:
        import tempfile

        self._path = tempfile.mkdtemp(prefix="hipporag_")
        self._hippo = self._HippoRAG(
            save_dir=self._path,
            llm_model_name=self.model,
            embedding_model_name=self.embed_model,
        )
        self._hippo.global_config.retrieval_top_k = self.top_k
        self._hippo.global_config.linking_top_k = self.top_k
        self._hippo.index(_extract_docs(conversation))

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        solutions, _, _ = self._hippo.rag_qa([question])
        solution = solutions[0]
        return MethodOutput(answer=(solution.answer or "").strip(), retrieved=list(solution.docs or []))

    def reset(self) -> None:
        self._hippo = None
        if self._path:
            import shutil

            shutil.rmtree(self._path, ignore_errors=True)
        self._path = None
