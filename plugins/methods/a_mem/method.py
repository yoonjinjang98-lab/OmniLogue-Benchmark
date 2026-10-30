"""a_mem -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps the `a-mem`
PyPI package (WujiangXu/A-mem, "A-MEM: Agentic Memory for LLM Agents", NeurIPS
2025) -- a Zettelkasten-style memory system: each ingested note gets
LLM-generated keywords/context/tags, is embedded and stored in ChromaDB, and
periodically triggers an "evolution" pass that lets the LLM link/update
existing notes based on new ones. Retrieval at answer time is plain ChromaDB
vector search over accumulated notes.

Both of a-mem's external-call seams are small, explicit abstract interfaces
(unlike mem0's closed provider registry or langmem's tool-calling extraction),
so no vendoring or tool-calling emulation was needed here:
- `agentic_memory.llm_controller.BaseLLMController.get_completion(prompt,
  response_format, temperature) -> str` -- `_DBLLMController` routes this
  through `dialogue_benchmarks.llm.complete()`. a-mem's own prompts already
  spell out the desired JSON shape in plain text (the `response_format` arg is
  only used by its real OpenAIController to request structured-output mode),
  so a plain completion is enough; this wrapper only adds markdown-fence
  stripping since a-mem's own `json.loads(response)` calls have none.
- `agentic_memory.retrievers.ChromaRetriever` hardcodes a local
  SentenceTransformer as its Chroma embedding function. `_DBChromaRetriever`
  subclasses it and overrides just `__init__` to plug in `_DBEmbeddingFunction`
  (routing through `dialogue_benchmarks.llm.embed()`) instead, keeping every
  other inherited method (add_document/search/...) unchanged.

`AgenticMemorySystem.__init__` builds its own `.llm_controller`/`.retriever`
internally (not constructor-injectable) -- `.llm_controller.llm` is safe to
swap in right after construction (same pattern as plugins/methods/mem0), but
`.retriever` is not: by the time `__init__` returns it has already created
(and persisted to disk) a ChromaDB collection bound to the *real*
SentenceTransformer embedding function, and Chroma refuses to reopen that
same collection with a different embedding function afterward ("embedding
function conflict"). So `_DBChromaRetriever` is swapped in *before*
construction instead, via a temporary monkeypatch of
`agentic_memory.memory_system.ChromaRetriever` -- the module-level name
`AgenticMemorySystem.__init__` actually calls -- restored immediately after.
"""
import re
import shutil
import tempfile

import chromadb.api.types

import agentic_memory.memory_system as _amem_module
from agentic_memory.llm_controller import BaseLLMController
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import ChromaRetriever

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_CHUNK_WINDOW = 20  # turns per add_note() call for flat `.turns` datasets


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


class _DBLLMController(BaseLLMController):
    def __init__(self, model: str):
        self.model = model

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 1.0, max_tokens: int = None) -> str:
        resp = complete(prompt, model=self.model, max_tokens=max_tokens or 1024, temperature=temperature)
        return _strip_fences(resp.text)


class _DBEmbeddingFunction(chromadb.api.types.EmbeddingFunction):
    """Chroma's EmbeddingFunction protocol: __call__(Documents) -> Embeddings.
    Subclassing (not just duck-typing the Protocol) matters here -- it's what
    supplies the default `embed_query()` (falls back to `__call__`) that
    Collection.query() requires and a bare object won't have."""

    def __init__(self, model: str):
        self.model = model

    def __call__(self, input):
        return embed(list(input), model=self.model).vectors

    def name(self) -> str:
        return f"dialogue_benchmarks:{self.model}"


class _DBChromaRetriever(ChromaRetriever):
    """`model_name` is accepted only because AgenticMemorySystem.__init__ passes
    it positionally-as-kwarg to whatever class `ChromaRetriever` currently
    names (see the monkeypatch in Method.prepare()) -- it's the sentence-
    transformer model name and is ignored here in favor of `embed_model`."""

    def __init__(self, collection_name: str, model_name: str = None, persist_directory: str = None, embed_model: str = "all-MiniLM-L6-v2"):
        import chromadb

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = _DBEmbeddingFunction(embed_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )


def _extract_chunks(conversation) -> list:
    """Same chunking convention as running_summary/mem0/langmem: one chunk per
    session if session-grouped, else fixed-size windows over flat `.turns`."""
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
        "a_mem: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract chunks for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
        max_tokens: int = 512,
        evo_threshold: int = 100,
        **config,
    ):
        super().__init__(
            model=model, embed_model=embed_model, top_k=top_k, max_tokens=max_tokens, evo_threshold=evo_threshold, **config
        )
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.evo_threshold = evo_threshold
        self._system = None
        self._path = None

    def prepare(self, conversation) -> None:
        import functools

        self._path = tempfile.mkdtemp(prefix="a_mem_")
        bound_retriever = functools.partial(_DBChromaRetriever, embed_model=self.embed_model)
        original_retriever_cls = _amem_module.ChromaRetriever
        _amem_module.ChromaRetriever = bound_retriever
        try:
            # api_key is a throwaway: LLMController eagerly builds a real
            # openai.OpenAI() client, which raises immediately with no key set
            # anywhere -- doesn't matter what it is, since .llm_controller.llm
            # gets swapped right below and this client is never actually called
            # (same as plugins/methods/mem0).
            system = AgenticMemorySystem(
                llm_backend="openai",
                llm_model=self.model,
                api_key="unused",
                storage_path=self._path,
                evo_threshold=self.evo_threshold,
            )
        finally:
            _amem_module.ChromaRetriever = original_retriever_cls
        system.llm_controller.llm = _DBLLMController(self.model)
        self._system = system
        for chunk in _extract_chunks(conversation):
            self._system.add_note(chunk)

    def answer(self, item) -> MethodOutput:
        raw = self._system.retriever.search(item.question, k=self.top_k)
        retrieved = list(raw.get("documents", [[]])[0]) if raw.get("ids", [[]])[0] else []

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        passages = "\n".join(f"- {m}" for m in retrieved) or "(no memories retrieved)"
        prompt = (
            "Answer the question using only the information in the memories below. "
            "Be concise and direct -- answer with just the fact asked for, no extra explanation.\n\n"
            f"### Memories\n{passages}\n\n"
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
        self._system = None
        if self._path:
            shutil.rmtree(self._path, ignore_errors=True)
        self._path = None
