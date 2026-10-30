"""memoryos -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps
memoryos-pro (BAI-LAB/MemoryOS, "Memory OS of AI Agent", EMNLP 2025 Oral,
Apache-2.0) -- a hierarchical short/mid/long-term memory system built for
personal AI assistants, evaluated in its own paper on LoCoMo (this repo's
own flagship dataset).

NOT the `memoryos` PyPI package -- that name is squatted by an unrelated
project (MemTensor's "MemOS", installs as `import memos`, a much larger
system with its own API server/scheduler). The actual pip-installable
release of this repo is `memoryos-pro` (see its README's "MemoryOS_PyPi
Getting Started" section), which installs as `import memoryos`.

Two seams to route through this repo's shared LLM/embedding client:
- `memoryos.utils.OpenAIClient.chat_completion(model, messages, temperature,
  max_tokens) -> str` -- not an abstract interface, a concrete class every
  other module constructs directly (`Memoryos.__init__` does
  `self.client = OpenAIClient(...)`, `MidTermMemory`/`Updater` take a
  `client` instance and just call `.chat_completion()` on it). `_DBOpenAIClient`
  duck-types the same public method and routes it through
  `dialogue_benchmarks.llm.complete()`.
- `memoryos.utils.get_embedding(text, model_name=..., use_cache=True)` -- a
  free function using a local SentenceTransformer, called directly (not via
  an instance) from long_term.py/mid_term.py/updater.py.

Both are swapped in by monkeypatching `memoryos.utils.OpenAIClient`/
`get_embedding` *before* `memoryos.memoryos` (and the long_term/mid_term/
updater/retriever submodules it imports) are imported for the first time --
those submodules do `from .utils import OpenAIClient, get_embedding` at
their own module level, binding a local snapshot of whatever `utils.X` is at
that moment, so patching *after* import (a_mem's ChromaRetriever-swap
pattern) wouldn't reach already-bound references here. Getting the patch to
land *before* that first import needs one extra step beyond a_mem's version
of this same problem: `memoryos/__init__.py` itself does `from .memoryos
import Memoryos`, so even the seemingly-safe `import memoryos.utils` (to
reach in and patch it) runs `__init__.py` first and imports the real,
unpatched `utils.OpenAIClient`/`get_embedding` before control ever returns
to patch anything. `_load_memoryos_class()` works around this with a bare
`sys.modules["memoryos"]` package stub (see its own docstring) so
`memoryos/utils.py` can be loaded and patched in isolation first, before the
real package (and its `__init__.py`) ever gets a chance to import it
unpatched. Done once per process, in Method.__init__ (not prepare()).

`Memoryos.add_memory(user_input, agent_response, ...)` is turn-pair shaped
(built for a chat agent remembering its own responses), not "ingest an
arbitrary multi-party transcript" -- so ingestion here treats consecutive
turn pairs as (user_input, agent_response), same adaptation full_context/
running_summary make for this dataset shape via speaker-alternation.
Answering uses `Retriever.retrieve_context()` directly (retrieval only, no
generation) rather than `Memoryos.get_response()`, which also has the side
effect of writing the Q/A back into memory -- undesirable when scoring a
fixed set of benchmark questions against a fixed context.
"""
import shutil
import tempfile

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed


class _DBOpenAIClient:
    """Duck-types memoryos.utils.OpenAIClient's public interface."""

    def __init__(self, api_key=None, base_url=None, max_workers=5):
        pass

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        system = None
        turns = []
        for m in messages:
            role, content = m.get("role"), m.get("content", "")
            if role == "system" and system is None:
                system = content
            else:
                turns.append(f"{role}: {content}")
        resp = complete("\n\n".join(turns), model=model, system=system, max_tokens=max_tokens, temperature=temperature)
        return resp.text


def _make_get_embedding(embed_model: str):
    import numpy as np

    def _get_embedding(text, model_name="all-MiniLM-L6-v2", use_cache=True):
        return np.array(embed(text, model=embed_model).vectors[0], dtype=np.float32)

    return _get_embedding


def _load_memoryos_class(embed_model: str):
    """`memoryos/__init__.py` does `from .memoryos import Memoryos`, so ANY
    import that touches the `memoryos` package at all -- even a plain
    `import memoryos.utils` -- runs that first and transitively imports the
    real (unpatched) OpenAIClient/get_embedding before we'd ever get a chance
    to patch them: Python always executes a parent package's `__init__.py`
    before its submodules.

    Worked around by inserting a bare stub for the "memoryos" package into
    sys.modules first (just `__path__`, `__init__.py` body never runs), so
    `memoryos/utils.py` -- whose own `from . import prompts` needs `memoryos`
    to exist as *a* package, just not yet-initialized -- loads and gets
    patched safely. The stub is then dropped so the *real* package import
    proceeds normally (running the real `__init__.py`); its `from .utils
    import ...` (and every other submodule's) resolves the already-patched
    module straight from sys.modules instead of re-reading the real file.
    """
    import sys
    import types
    from pathlib import Path

    if "memoryos.utils" not in sys.modules:
        import importlib.util

        pkg_spec = importlib.util.find_spec("memoryos")  # locates only, doesn't execute __init__.py
        pkg_dir = Path(pkg_spec.submodule_search_locations[0])

        stub = types.ModuleType("memoryos")
        stub.__path__ = [str(pkg_dir)]
        sys.modules["memoryos"] = stub

        utils_spec = importlib.util.spec_from_file_location("memoryos.utils", pkg_dir / "utils.py")
        utils_module = importlib.util.module_from_spec(utils_spec)
        sys.modules["memoryos.utils"] = utils_module
        utils_spec.loader.exec_module(utils_module)
        utils_module.OpenAIClient = _DBOpenAIClient
        utils_module.get_embedding = _make_get_embedding(embed_model)

        del sys.modules["memoryos"]

    from memoryos.memoryos import Memoryos

    return Memoryos


def _turn_pairs(conversation) -> list:
    """[(user_input, agent_response), ...] from consecutive turns -- one
    chunk per session if session-grouped (e.g. locomo), else the flat
    `.turns` list. An odd trailing turn with no partner is dropped."""
    if hasattr(conversation, "sessions"):
        turns = [turn for idx in sorted(conversation.sessions) for turn in conversation.sessions[idx]]
    elif hasattr(conversation, "turns"):
        turns = conversation.turns
    else:
        raise ValueError(
            "memoryos: conversation has neither `.sessions` (dict of session_idx -> turns) nor a "
            "flat `.turns` list -- can't extract turn pairs for this dataset's shape"
        )
    return [(turns[i].text, turns[i + 1].text) for i in range(0, len(turns) - 1, 2)]


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "all-MiniLM-L6-v2",  # memoryos-pro's own paper default
        top_k: int = 10,
        max_tokens: int = 512,
        **config,
    ):
        super().__init__(model=model, embed_model=embed_model, top_k=top_k, max_tokens=max_tokens, **config)
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._Memoryos = _load_memoryos_class(embed_model)
        self._system = None
        self._path = None

    def prepare(self, conversation) -> None:
        self._path = tempfile.mkdtemp(prefix="memoryos_")
        self._system = self._Memoryos(
            user_id="conv",
            openai_api_key="unused",  # never used -- _DBOpenAIClient ignores it
            data_storage_path=self._path,
            llm_model=self.model,
        )
        for user_input, agent_response in _turn_pairs(conversation):
            self._system.add_memory(user_input=user_input, agent_response=agent_response)

    def answer(self, item) -> MethodOutput:
        result = self._system.retriever.retrieve_context(
            user_query=item.question, user_id="conv", top_k_sessions=self.top_k
        )
        retrieved = [
            f"User: {p.get('user_input', '')}\nAssistant: {p.get('agent_response', '')}"
            for p in result.get("retrieved_pages", [])
        ]

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
