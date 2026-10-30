"""AnchorMem -- Anchored Facts with Associative Contexts for Building Memory
in Large Language Models (RayNeo-AI-2025/AnchorMem, ACL 2026 Findings,
arXiv:2604.17377). Family C: an LLM-driven fact-extraction memory system,
same family as mem0/a_mem/lightmem, but with a genuinely different
mechanism -- a two-layer hierarchy of "Facts" (atomic, LLM-extracted per
chunk) and "Events" (LLM-summarized clusters of semantically-related facts
across chunks, found via embedding similarity), retrieved together at
query time ("Dual-path retrieval that matches both facts and events").

**Investigated earlier this session and initially skipped, then revisited
and actually built**: the first pass found `AnchorMem.py` importing a
`src/` package that did not exist anywhere in the public repo at the time
-- and the import paths (`src.llm`, `src.embedding_model`, matching
`_get_llm_class`/`_get_embedding_model_class` factory functions with
identical signatures) were an exact match for `plugins/methods/hipporag`'s
own `hipporag/{llm,embedding_model}` layout, strongly suggesting AnchorMem
was built as a fork/extension of HippoRAG's codebase whose release only
shipped the new files, not the underlying `src/` it depends on -- at that
point, reproducing it would have meant vendoring HippoRAG a second time
inside it. Re-checked at the user's prompt: the upstream repo now DOES
ship a complete `src/` (confirmed populated, not empty placeholder dirs,
`updated_at: 2026-06-28` -- after the original check), so the original
blocker no longer applies.

**Vendored** (`vendor/anchormem_src/` -- renamed from upstream's bare
`src/` specifically to avoid a `sys.modules["src"]` collision with any
other plugin that might also vendor a generically-named `src/` package
onto `sys.path`; `vendor/AnchorMem.py`/`vendor/embedding_store.py` --
upstream's own top-level files, with their `from src.X import Y` imports
rewritten to `from anchormem_src.X import Y` to match). Confirmed the
`src/` layout really is HippoRAG's own lineage by direct comparison:
identical `BaseLLM`/`BaseEmbeddingModel` ABCs (`infer`/`batch_infer`/
`batch_encode`), identical SQLite response-caching decorator in
`llm/openai_gpt.py`, identical `_get_llm_class(config)`/
`_get_embedding_model_class(name)` factory signatures -- so this Method
reuses `plugins/methods/hipporag/method.py`'s own injection technique
near-verbatim: create stub modules for `anchormem_src.llm`/
`anchormem_src.embedding_model` via `importlib.util.spec_from_file_location`
(so their real bodies execute, but under controlled module objects we
still hold a reference to), execute them, THEN monkeypatch
`_get_llm_class`/`_get_embedding_model_class` onto those module objects
before `AnchorMem.py` itself gets imported (its own `from anchormem_src.llm
import _get_llm_class` binds the name at import time, so the patch must
land first). Only one dead import needed trimming at vendor time (same
"grep for the real call, not just the import" technique as this session's
other vendoring): `embedding_model/__init__.py`'s eager `from .GritLM
import GritLMEmbeddingModel` needs the separate `gritlm` package (a
vllm-adjacent dependency chain) -- confirmed unreachable regardless, since
this Method's monkeypatched `_get_embedding_model_class` never returns
that class; dropped the import, left the dispatch branch raising instead
of silently returning something wrong if `embedding_model_name` ever
contained "GritLM" (it never will here).

**Mechanism** (`AnchorMem.index()`/`.rag_qa()`, read directly, not
assumed from the README): `index(docs)` chunks each doc, extracts atomic
facts per chunk via an LLM call (`FactExtractor`/`fact_extraction_openai.
extract_event_list`), embeds every fact, clusters facts with high
embedding similarity across chunks, and asks the LLM to summarize each
cluster into a higher-level "Event" -- both facts and events get their
own `EmbeddingStore`. `.rag_qa(questions)` retrieves top-k facts by
embedding similarity, expands to their source chunks plus any linked
event summaries, and generates the final answer via one more LLM call --
returns `(solutions, responses, metadata)` where `responses[i]` is the
plain answer string and `solutions[i].docs` is the retrieved context.
This is the only LLM-driven fact-extraction Family C baseline this
session where facts get a genuine second-pass LLM summarization step
(the "Event" layer) rather than being used flat -- mem0/a_mem/lightmem
all retrieve facts directly with no analogous hierarchy.

**Licensing**: real MIT LICENSE file in the repo (`vendor/LICENSE.txt`)
-- no gap to resolve, unlike pathrag/orig/sbys/lmedr/memalpha/rfmem this
session.

**Verification**: mock-verified end-to-end (`complete()`/`embed()` both
mocked, no API key configured in this worktree) -- confirmed `index()`
runs through fact extraction and event clustering without error on a
real multi-session locomo-shaped conversation, and `rag_qa()` returns a
correctly-shaped `(solutions, responses, metadata)` tuple. Not yet run
against a real LLM.
"""
import sys
import time
import types
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_CHUNK_WINDOW = 20
_EMBED_DIM = 1536


def _load_anchormem_class():
    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))

    if "anchormem_src.llm" not in sys.modules:
        import importlib.util

        pkg_spec = importlib.util.find_spec("anchormem_src")
        pkg_dir = Path(pkg_spec.submodule_search_locations[0])

        stub = types.ModuleType("anchormem_src")
        stub.__path__ = [str(pkg_dir)]
        sys.modules["anchormem_src"] = stub

        for sub in ("llm", "embedding_model"):
            sub_dir = pkg_dir / sub
            spec = importlib.util.spec_from_file_location(
                f"anchormem_src.{sub}", sub_dir / "__init__.py", submodule_search_locations=[str(sub_dir)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"anchormem_src.{sub}"] = module
            spec.loader.exec_module(module)

        sys.modules["anchormem_src.llm"]._get_llm_class = lambda config: _DBLLM(config)
        sys.modules["anchormem_src.embedding_model"]._get_embedding_model_class = lambda embedding_model_name=None: _DBEmbeddingModel

        del sys.modules["anchormem_src"]

    from AnchorMem import AnchorMem

    return AnchorMem


class _DBLLM:
    """Duck-types anchormem_src.llm.base.BaseLLM's real contract."""

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
    """Duck-types anchormem_src.embedding_model.base.BaseEmbeddingModel."""

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
    """Same convention as hipporag/mem0/lightmem/memalpha's Methods: one
    document per session if session-grouped (e.g. locomo), else fixed-size
    windows over a flat `.turns` list."""
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
        "anchormem: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract documents for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        top_k: int = 10,
        **config,
    ):
        super().__init__(model=model, embed_model=embed_model, top_k=top_k, **config)
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self._AnchorMem = _load_anchormem_class()
        self._memory = None
        self._path = None

    def prepare(self, conversation) -> None:
        import tempfile

        self._path = tempfile.mkdtemp(prefix="anchormem_")
        self._memory = self._AnchorMem(
            save_dir=self._path,
            llm_model_name=self.model,
            embedding_model_name=self.embed_model,
        )
        self._memory.global_config.retrieval_top_k = self.top_k
        self._memory.global_config.linking_top_k = self.top_k
        # Upstream's own answer-generation prompt selection falls back to a
        # MuSiQue-specific template unless `global_config.dataset` names one
        # of its own `prompts/templates/rag_qa_*.py` files -- `rag_qa_locomo`
        # exists and matches this repo's primary target dataset, confirmed
        # by reading `AnchorMem.qa()`'s actual template-name lookup, not
        # assumed from the README.
        self._memory.global_config.dataset = "locomo"
        self._memory.index(_extract_docs(conversation))

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        solutions, responses, _ = self._memory.rag_qa([question])
        solution = solutions[0]
        return MethodOutput(answer=(responses[0] or "").strip(), retrieved=list(solution.docs or []))

    def reset(self) -> None:
        self._memory = None
        if self._path:
            import shutil

            shutil.rmtree(self._path, ignore_errors=True)
        self._path = None
