"""mem0 -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps mem0ai's Memory
API (https://github.com/mem0ai/mem0) as a memory-agent baseline: ingest the
conversation chunk by chunk via `Memory.add()` (mem0's own LLM-based fact
extraction and add/update/delete consolidation, not raw storage -- the actual
point of comparison against bm25_rag/dense_rag/running_summary), then
`Memory.search()` retrieves relevant memories per question and an LLM answers
from just those.

mem0 has no "bring your own callable" LLM/embedder provider -- only named ones
(openai, anthropic, litellm, ollama, ...; see the installed package's
mem0/llms/, mem0/embeddings/), so this constructs a real `Memory` via
`Memory.from_config()` using the "openai" provider purely to satisfy config
validation (never actually called -- no network access, no key needed at
construction) and immediately swaps in `_DBLLM`/`_DBEmbedder`, small
`LLMBase`/`EmbeddingBase` subclasses that route every call through
`dialogue_benchmarks.llm.complete()`/`embed()` like every other baseline in
this repo (CLAUDE.md's LLM access section).

Vector store: mem0's local on-disk Qdrant backend (`qdrant-client`, installed
as mem0ai's own dependency), `embedding_model_dims=1536` to match
text-embedding-3-small (also mem0's own default dim) -- a fresh temp directory
per conversation (prepare()/reset()) so memories never leak across
conversations, since mem0's own `user_id`-scoping is meant for persistent
cross-session memory, not per-conversation isolation.
"""
import shutil
import tempfile
import uuid

from mem0 import Memory
from mem0.embeddings.base import EmbeddingBase
from mem0.llms.base import LLMBase

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_CHUNK_WINDOW = 20  # turns per add() call for flat `.turns` datasets


class _DBLLM(LLMBase):
    """Routes mem0's fact-extraction/consolidation calls through complete()."""

    def __init__(self, model: str, config=None):
        super().__init__(config)
        self.model = model

    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        system = None
        turns = list(messages)
        if turns and turns[0].get("role") == "system":
            system = turns[0]["content"]
            turns = turns[1:]
        prompt = "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in turns)
        return complete(prompt, model=self.model, system=system, max_tokens=1024).text


class _DBEmbedder(EmbeddingBase):
    """Routes mem0's embedding calls through embed()."""

    def __init__(self, model: str, config=None):
        super().__init__(config)
        self.model = model

    def embed(self, text, memory_action=None):
        return embed(text, model=self.model).vectors[0]

    def embed_batch(self, texts, memory_action="add"):
        return embed(list(texts), model=self.model).vectors


def _extract_chunks(conversation) -> list:
    """One chunk per session if session-grouped (`.sessions`, e.g. locomo), else
    fixed-size windows over flat `.turns` (e.g. molweni) -- same convention as
    running_summary's chunking, so it stays usable on very long conversations
    instead of issuing one add() call per turn."""
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
        "mem0: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract chunks for this dataset's shape"
    )


def _build_memory(model: str, embed_model: str, path: str) -> Memory:
    # api_key here is a throwaway: constructing the real openai-backed LLM/embedder
    # eagerly builds an `openai.OpenAI()` client, which raises immediately if no key
    # is set anywhere (env var or here) -- doesn't matter what it is, since `.llm`/
    # `.embedding_model` get swapped out for _DBLLM/_DBEmbedder right below and this
    # client is never actually called.
    mem = Memory.from_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": path, "embedding_model_dims": 1536},
            },
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "api_key": "unused"}},
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small", "api_key": "unused"},
            },
        }
    )
    mem.llm = _DBLLM(model=model)
    mem.embedding_model = _DBEmbedder(model=embed_model)
    return mem


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
        self._memory = None
        self._path = None
        self._user_id = None

    def prepare(self, conversation) -> None:
        self._path = tempfile.mkdtemp(prefix="mem0_")
        self._memory = _build_memory(self.model, self.embed_model, self._path)
        self._user_id = f"conv_{uuid.uuid4().hex}"
        for chunk in _extract_chunks(conversation):
            self._memory.add(chunk, user_id=self._user_id)

    def answer(self, item) -> MethodOutput:
        result = self._memory.search(item.question, filters={"user_id": self._user_id}, top_k=self.top_k)
        memories = result.get("results", []) if isinstance(result, dict) else result
        retrieved = [m.get("memory", str(m)) for m in memories]

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
        self._memory = None
        if self._path:
            shutil.rmtree(self._path, ignore_errors=True)
        self._path = None
        self._user_id = None
