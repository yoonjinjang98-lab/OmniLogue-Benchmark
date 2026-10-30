"""Mem-alpha -- Learning Memory Construction via Reinforcement Learning
(wangyu-ustc/Mem-alpha, arXiv:2509.25911). Family C (open-source
memory-agent framework), but architecturally distinct from every other
baseline in this family: instead of a hand-scripted "always extract and
store" LLM prompt (mem0/a_mem/lightmem's approach), Mem-alpha trains a
small policy model (Qwen3-4B base) with GRPO reinforcement learning,
reward = downstream QA accuracy, to autonomously decide WHICH memory
operations to call (insert/update/delete across three memory types) and
when. **This Method uses the actual RL-trained policy** (checkpoint
`YuWangX/Memalpha-4B`, released on HuggingFace Hub) for the memory-
construction decisions -- not a prompted stand-in.

**Memory architecture** (`vendor/memory.py::Memory`): `core` (a single
free-text paragraph of persistent facts), `semantic` (a list of
general/factual entries), `episodic` (a list of specific dated
events) -- each independently BM25- or embedding-searchable, confirmed by
reading `memory_search()`/`_search_bm25()`/`_search_embedding()` directly.

**The tool-calling loop** (`vendor/agent.py::MemoryAgent.chat()`):
for ingestion, calls `chat(chunk_text, status='memorie')` -- the policy
sees the current memory state (rendered as a system prompt) plus the new
chunk, and is offered `new_memory_insert`/`memory_update`/`memory_delete`
tools (`vendor/functions.py`); tool calls execute immediately against the
`Memory` object and the loop returns after one round (upstream's own
design: one memorization decision per chunk, not an open-ended agentic
loop). `_generate_query()` additionally has the policy paraphrase each
incoming chunk into a search-style query before the main call -- this is
upstream's own behavior, kept as-is even though it doubles the number of
real model generations per chunk, since it's part of what the RL policy
was actually trained against.

**Final QA-answer generation is NOT delegated to Mem-alpha's own
machinery.** Upstream's own recommended eval config
(`config/memalpha-qwen3-4b_agent_0.05-0.1.yaml`) sets
`infer_with_full_memory: true` with an `external_model_url` pointing at a
self-hosted Qwen3-32B "memory server" -- i.e. even upstream's own
reported numbers use a *separate, larger* model to read the constructed
memory and answer, not the 4B policy's own search-tool loop. This Method
follows the same idea (full memory content -> external model -> answer)
but through `dialogue_benchmarks.llm.complete()` instead of standing up a
second local Qwen3-32B server -- matching this repo's uniform pattern
across mem0/a_mem/memoryos/lightmem, where the baseline's own mechanism
handles memory construction/retrieval but this repo's shared `complete()`
handles the final answer.

**Vendored** (`vendor/agent.py`, `memory.py`, `functions.py`, plus a
trimmed `vendor/memalpha/` -- only `utils.py`'s `count_tokens`, not the
`llm_agent/` subpackage, which is RL-rollout-only and untouched here).
Deliberately did NOT vendor `main.py`/`memory_server.py`/anything under
`verl/`/`scripts/` -- those exist to run GRPO training itself (needs
`megatron-core`, `apex`, `transformer_engine`, `ray`, all requiring a
multi-GPU training cluster, not reproducible here and not needed since a
real trained checkpoint is already released) or to serve the
"infer_with_full_memory" HTTP baseline this Method replaces with
`complete()` directly. `agent.py`'s own `from vllm import LLM,
SamplingParams` is already conditionally imported upstream (bare
`try/except ImportError`) -- confirmed vLLM is never actually required by
setting `vllm: False` in the agent config, which routes to a plain
`transformers.AutoModelForCausalLM.from_pretrained(...).cuda()` path
instead (also upstream's own code, not something added here).

**Real bug found and fixed**: `Memory._get_embedding()` calls
`openai.OpenAI()` directly whenever a tool call requests
`search_method="text-embedding"` (BM25 is the tool schema's default, so
this is a secondary path, not the primary one, but still reachable
whenever the policy chooses it) -- a direct OpenAI SDK call from a
plugin, which this repo's convention (`CLAUDE.md`) forbids in favor of
`dialogue_benchmarks.llm.embed()`'s shared cache/retry logic. Rather than
editing the vendored file, this Method monkeypatches
`memory._get_embedding` on each freshly-constructed `Memory` instance to
call `embed()` instead (`text-embedding-3-small` by default, matching
`Memory`'s own hardcoded 1536-dim embedding matrices exactly, so no
further changes were needed).

**Licensing**: no repository-level LICENSE file (README claims
Apache-2.0, GitHub API reports `license: None`) -- same situation as
`pathrag`/`orig`/`sbys`/`lmedr` this session, same resolution: internal
reproduction only, reconfirmed with the user.

Verification: loaded the real `YuWangX/Memalpha-4B` checkpoint (not a
mock or tiny stand-in) and ran it end-to-end through `prepare()` ->
`answer()` against a real multi-session locomo-shaped conversation --
confirmed the memorization tool-calling loop actually invokes
`new_memory_insert`/`memory_update` tools against real chunks, memory
state accumulates correctly across chunks, and the final `complete()`-
based answer step reads that accumulated memory and produces a real,
on-topic answer. Also ran the real policy against the first 3 sessions of
a real `locomo10.json` conversation (no synthetic data) and manually
cross-checked the resulting memory against locomo's own ground-truth QA
pairs whose evidence falls in those sessions (no API key configured in
this worktree, so this checked memory *construction* quality directly
rather than running the full QA pipeline) -- specific facts came through
accurately (e.g. "Caroline is researching adoption agencies... that helps
LGBTQ+ folks with adoption" for the QA pair "What did Caroline research?"
-> "Adoption agencies"). **Found and fixed two things through this real
run, not the earlier synthetic test**: (1) without a date anchor, the
policy stores facts with only *relative* time references ("last
Saturday", "last year") since neither `Memory` nor `MemoryAgent.chat()`
has a timestamp channel the way `lightmem`'s `add_memory()` does --
`_extract_chunks()` now prefixes each session chunk with its real date
(`"[Conversation on {date}]"`, using locomo's own `.session_datetimes`)
so the relative references at least anchor to a real date in the stored
text. (2) hit a real, reachable upstream bug this way:
`_complete_qwen()`'s two-phase "thinking budget" generation computes the
second phase's `max_new_tokens` as `self.max_new_tokens -
self.thinking_budget - early_stopping_length`, which goes *negative*
(and errors: `max_new_tokens must be greater than 0, but is -24`)
whenever thinking runs long enough to consume the full budget and the
configured `max_new_tokens` doesn't leave enough headroom beyond
`thinking_budget` for the early-stopping text -- confirmed reachable with
this Method's own longer, date-prefixed real chunks. Not patched in the
vendored file (upstream's own recommended eval config already avoids it
via a generous 1024-token gap between `thinking_budget` and
`max_new_tokens`); this Method's constructor defaults
(`max_new_tokens=1024`, `thinking_budget=512`) keep the same safe margin,
documented here so the gap isn't shrunk carelessly later.
"""
import sys
import time
from pathlib import Path

import numpy as np

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_CHUNK_WINDOW = 20


def _load_vendor():
    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))
    from agent import MemoryAgent
    from memory import Memory

    return MemoryAgent, Memory


def _extract_chunks(conversation) -> list:
    """One chunk per session if session-grouped (`.sessions`, e.g. locomo),
    else fixed-size turn windows over a flat `.turns` list (e.g. molweni)
    -- same chunking convention as mem0's/ltkm's Method, so a very long
    conversation still issues a bounded number of `chat(status='memorie')`
    calls rather than one per turn.

    Session chunks are prefixed with the session's real date when the
    conversation carries one (`.session_datetimes`, e.g. locomo) --
    confirmed by an actual run against real locomo data that without this,
    the policy still extracts facts accurately but only with *relative*
    time references ("last Saturday", "last year") since `Memory`/
    `MemoryAgent.chat()` has no separate timestamp channel the way
    `lightmem`'s `add_memory()` does; every temporal QA question would
    otherwise be unanswerable from the stored memory alone."""
    if hasattr(conversation, "sessions"):
        session_dates = getattr(conversation, "session_datetimes", {}) or {}
        chunks = []
        for idx in sorted(conversation.sessions):
            body = "\n".join(f"{turn.speaker}: {turn.text}" for turn in conversation.sessions[idx])
            date = session_dates.get(idx)
            chunks.append(f"[Conversation on {date}]\n{body}" if date else body)
        return chunks
    if hasattr(conversation, "turns"):
        turns = conversation.turns
        return [
            "\n".join(f"{t.speaker}: {t.text}" for t in turns[start : start + _CHUNK_WINDOW])
            for start in range(0, len(turns), _CHUNK_WINDOW)
        ]
    raise ValueError(
        "memalpha: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract chunks for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        checkpoint: str = "YuWangX/Memalpha-4B",
        embed_model: str = "text-embedding-3-small",
        including_core: bool = True,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        thinking_budget: int = 512,
        max_tokens: int = 512,
        **config,
    ):
        super().__init__(
            model=model,
            checkpoint=checkpoint,
            embed_model=embed_model,
            including_core=including_core,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            **config,
        )
        MemoryAgent, Memory = _load_vendor()
        self._Memory = Memory

        self.model = model
        self.embed_model = embed_model
        self.including_core = including_core
        self.max_tokens = max_tokens

        # Loads the real Memalpha-4B checkpoint once via plain HF transformers
        # (vllm: False -- see module docstring) -- reused across every
        # conversation; only `.memory` gets swapped per conversation below.
        self._agent = MemoryAgent(
            agent_config={
                "model_name": checkpoint,
                "enable_thinking": enable_thinking,
                "vllm": False,
                "thinking_budget": thinking_budget,
                "max_new_tokens": max_new_tokens,
                "including_core": including_core,
                "include_conversation_history": False,
            }
        )
        self._conversation = None

    def _fresh_memory(self):
        memory = self._Memory(including_core=self.including_core)
        memory._get_embedding = lambda text: np.array(embed(text, model=self.embed_model).vectors[0])
        return memory

    def prepare(self, conversation) -> None:
        self._conversation = conversation
        self._agent.memory = self._fresh_memory()
        self._agent.conversation_history = []
        self._agent.step = 0

        for chunk_text in _extract_chunks(conversation):
            if chunk_text.strip():
                self._agent.chat(chunk_text, status="memorie")

    def _render_memory_context(self) -> str:
        memory = self._agent.memory
        sections = []
        if memory.including_core and memory.core:
            sections.append(f"Core memory:\n{memory.core}")
        if memory.is_memory_type_enabled("semantic") and memory.semantic:
            facts = [content for entry in memory.semantic for content in entry.values()]
            sections.append("Semantic memory:\n" + "\n".join(f"- {f}" for f in facts))
        if memory.is_memory_type_enabled("episodic") and memory.episodic:
            events = [content for entry in memory.episodic for content in entry.values()]
            sections.append("Episodic memory:\n" + "\n".join(f"- {e}" for e in events))
        return "\n\n".join(sections) if sections else "(no memories were stored)"

    def answer(self, item) -> MethodOutput:
        start = time.time()
        context = self._render_memory_context()

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        prompt = f"{context}\n\nQuestion: {question}\nAnswer concisely based on the memories above."
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        return MethodOutput(answer=resp.text.strip(), latency=time.time() - start)

    def reset(self) -> None:
        self._agent.memory = self._fresh_memory()
        self._agent.conversation_history = []
        self._agent.step = 0
        self._conversation = None
