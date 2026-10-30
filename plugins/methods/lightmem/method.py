"""LightMem -- Lightweight and Efficient Memory-Augmented Generation
(zjunlp/LightMem, MIT, ICLR 2026, arXiv:2510.18866). Family C (open-source
memory-agent framework), same family as mem0/langmem/a_mem/memoryos/zep/
hipporag/hypermem/graphrag/lightrag/hypergraphrag/pathrag.

**The architecture**: `LightMemory.add_memory()` runs a pipeline of
(1) optional pre-compression (llmlingua-2, disabled here -- see below),
(2) topic segmentation -- an attention-based encoder (`llmlingua-2`'s own
base model, cross-sentence attention over consecutive turns) that groups
turns into coherent topics *before* any LLM call is spent on them, so
extraction only fires once a topic segment closes rather than once per
turn, (3) LLM-driven metadata/fact extraction into `MemoryEntry` objects,
embedded and stored in a local Qdrant collection, and (4) an "offline"
update pass (`offline_update`) that periodically merges/prunes
near-duplicate entries in the background instead of on every write. This
segmentation-before-extraction + batched-consolidation design is the
paper's actual "lightweight/efficient" contribution versus mem0/A-MEM
(which call an LLM on every turn) -- confirmed by reading `add_memory`'s
own pipeline directly, not just the README's framing. `pre_compress`
(llmlingua-2 prompt compression, a separate optional stage cutting token
*volume* rather than call *count*) is left disabled: it needs the extra
`llmlingua` package and is orthogonal to the segmentation/consolidation
mechanism this Method exercises, matching how other family-C baselines
here (e.g. hypergraphrag's non-default storage backends) skip optional
knobs that don't change what's actually being compared.

**LLM/embedding injection**: LightMem's own `MemoryManagerFactory`/
`TextEmbedderFactory` are fixed factories keyed by provider name (openai/
deepseek/ollama/vllm/huggingface, no custom-callable option) -- the same
shape as memoryos/mem0, but with a much smaller surface here: every
higher-level method in `OpenaiManager` (`meta_text_extract`,
`_call_update_llm`) routes through exactly one method,
`generate_response()`, and `TextEmbedderOpenAI` has exactly one method,
`embed()`. Rather than monkeypatching the factories' class-level model
lists, this Method constructs `LightMemory` with `model_name="openai"`
(a real `OpenaiManager`/`TextEmbedderOpenAI` gets built, with a throwaway
`api_key` so the `openai.OpenAI()` client construction doesn't raise) and
then swaps `instance.manager`/`instance.text_embedder` for `_DBManager`/
`_DBEmbedder` subclasses overriding just those two methods to call
`dialogue_benchmarks.llm.complete()`/`embed()` -- no shared mutable class
state touched, unlike the sys.modules-stub tricks memoryos/hipporag
needed. `complete()` takes a single `prompt` (+ optional `system`), not a
message list, so `_DBManager.generate_response()` splits LightMem's
`[{"role": "system", ...}, {"role": "user", ...}]` convention (confirmed
by reading every real call site -- `_extract_with_prompt`/
`_call_update_llm` never send more than these two) into `system=`/
`prompt=`; `response_format={"type": "json_object"}` requests are handled
by appending a "respond with a single JSON object" instruction to the
prompt and parsing the reply as JSON, the same convention `graphrag`'s
Method here already uses for the same reason (`complete()` has no native
JSON-mode passthrough).

**Vendored** (`vendor/lightmem/`, MIT -- `vendor/LICENSE.txt`): no usable
PyPI release exists -- the `lightmem` name on PyPI is an unrelated
placeholder stub (description "# lightmem", version 0.0.0), same
name-squatting situation as `memoryos`'s PyPI package. Vendored the whole
`src/lightmem/` tree as-is (2.3MB, 219 files) -- unlike bob/hipporag,
nothing needed trimming: `lightmem/__init__.py` is completely empty, so
none of the six memory-manager backends or two text-embedder backends
import eagerly regardless of which one is configured.

**Real bug found and fixed, confirmed by actually constructing
`LightMemory`** (not just read from source): `LightMemory.__init__` only
ever sets `self.compressor` inside the `if self.config.pre_compress:`
branch, but the very next `if self.config.topic_segment:` branch
unconditionally reads `self.compressor` as a plain attribute access when
building `TopicSegmenterFactory.from_config(...)`'s third argument --
`AttributeError: 'LightMemory' object has no attribute 'compressor'`
whenever `topic_segment=True` and `pre_compress=False` (exactly this
Method's configuration, confirmed by constructing `LightMemory` with that
combination before applying the fix). Fixed by initializing
`self.compressor = None` unconditionally before the `pre_compress`
branch -- `LlmLingua2Segmenter.__init__`'s own `shared=False` default path
(used here) never actually touches the `compressor` argument's value, so
this doesn't change behavior beyond avoiding the crash.

**Licensing**: real MIT LICENSE file at the repo root -- no licensing gap
to decide on (`pyproject.toml`'s own `license = "Apache-2.0"` metadata
field is inconsistent with the actual LICENSE file, an upstream
inconsistency, not something this repo needs to resolve; the LICENSE file
governs).

**Modeling note**: `add_memory()` requires every message dict to carry a
`role` in `{"user", "assistant"}` (used both for `messages_use` filtering,
default `"user_only"`, and for the extraction prompt's turn numbering) and
a session-level `time_stamp` string (identical for every message added in
one call -- `MessageNormalizer` disambiguates order within that call by
its own internal offset, confirmed by reading it directly, not by
incrementing timestamps ourselves). This repo's conversation shapes don't
have a clean user/assistant split for multi-party data (locomo, msc), so
every turn is tagged `role="user"` with the real speaker name carried
separately in `speaker_name` -- consistent with `messages_use="user_only"`
(the default) still processing every turn, and matching how this repo's
other family-C baselines (mem0, a_mem) don't force a user/assistant split
onto non-chatbot conversation data either. One `add_memory()` call is
issued per session (locomo's real `session_datetimes`, parsed via
`dateutil`) or, for flat-`.turns` datasets with no native per-turn dates,
per fixed-size window with a synthetic incrementing timestamp;
`force_segment=force_extract=True` only on the very last call, matching
upstream's own tutorial notebook's flush-on-last-turn pattern, so the
topic-segmentation buffer doesn't silently strand a final partial segment
unextracted.

Verification: mocked `complete()`/`embed()` (deterministic hash-based fake
embeddings, canned JSON extraction replies) and ran a real multi-session
locomo-shaped conversation end-to-end through `prepare()` -> `answer()` --
confirmed segmentation, extraction (through `_DBManager`), embedding
(through `_DBEmbedder`), local Qdrant storage/retrieval, and the final
completion call all execute without error and retrieval surfaces the
mock-extracted facts relevant to the query. Not yet run against a real
LLM.
"""
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _load_vendor():
    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))
    from lightmem.factory.memory_manager.openai import OpenaiManager
    from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
    from lightmem.memory.lightmem import LightMemory

    return LightMemory, OpenaiManager, TextEmbedderOpenAI


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        segmenter_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        top_k: int = 10,
        max_tokens: int = 512,
        window_size: int = 20,
        **config,
    ):
        super().__init__(
            model=model,
            embed_model=embed_model,
            segmenter_model=segmenter_model,
            top_k=top_k,
            max_tokens=max_tokens,
            window_size=window_size,
            **config,
        )
        LightMemory, OpenaiManager, TextEmbedderOpenAI = _load_vendor()

        class _DBManager(OpenaiManager):
            def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto"):
                if tools:
                    raise NotImplementedError("lightmem: tool-calling extraction is never used in this configuration")
                system = None
                user_parts = []
                for msg in messages:
                    if msg["role"] == "system":
                        system = msg["content"]
                    else:
                        user_parts.append(msg["content"])
                prompt = "\n\n".join(user_parts)
                if response_format and response_format.get("type") == "json_object":
                    prompt += "\n\nRespond with a single valid JSON object and nothing else."
                resp = complete(prompt, model=model, system=system, max_tokens=max_tokens)
                usage = {
                    "prompt_tokens": resp.tokens_in,
                    "completion_tokens": resp.tokens_out,
                    "total_tokens": resp.tokens_in + resp.tokens_out,
                }
                return resp.text, usage

        class _DBEmbedder(TextEmbedderOpenAI):
            def embed(self, text):
                if isinstance(text, list):
                    return embed(text, model=embed_model).vectors
                return embed(text, model=embed_model).vectors[0]

        self._DBManager = _DBManager
        self._DBEmbedder = _DBEmbedder
        self._LightMemory = LightMemory

        self.model = model
        self.embed_model = embed_model
        self.segmenter_model = segmenter_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.window_size = window_size
        self._embedding_dims = len(embed("probe", model=embed_model).vectors[0])

        self._memory = None
        self._db_path = None
        self._conversation = None

    def _build_memory(self):
        db_path = tempfile.mkdtemp(prefix="lightmem_")
        config = {
            "pre_compress": False,
            "topic_segment": True,
            "topic_segmenter": {"model_name": "llmlingua-2", "configs": {"model_name": self.segmenter_model}},
            "messages_use": "user_only",
            "memory_manager": {"model_name": "openai", "configs": {"model": self.model, "api_key": "unused"}},
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "openai",
                "configs": {"model": self.embed_model, "api_key": "unused"},
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": "lightmem",
                    "embedding_model_dims": self._embedding_dims,
                    "path": db_path,
                },
            },
            "update": "offline",
        }
        memory = self._LightMemory.from_config(config)
        memory.manager = self._DBManager(config=memory.manager.config)
        memory.text_embedder = self._DBEmbedder(config=memory.text_embedder.config)
        return memory, db_path

    def prepare(self, conversation) -> None:
        self._conversation = conversation
        self._memory, self._db_path = self._build_memory()

        units = self._extract_units(conversation)
        for i, (turns, time_stamp) in enumerate(units):
            is_last = i == len(units) - 1
            messages = []
            for turn in turns:
                speaker = getattr(turn, "speaker", None) or getattr(turn, "speaker_id", None) or "speaker"
                text = getattr(turn, "text", None) or getattr(turn, "content", None) or ""
                messages.append(
                    {
                        "role": "user",
                        "content": text,
                        "speaker_name": speaker,
                        "time_stamp": time_stamp,
                    }
                )
            if not messages:
                continue
            self._memory.add_memory(messages=messages, force_segment=is_last, force_extract=is_last)

    def _extract_units(self, conversation):
        """One (turns, iso_timestamp) unit per session for session-grouped
        conversations (locomo -- real dates via `session_datetimes`,
        parsed with dateutil since they're not in a form
        `datetime.fromisoformat` accepts directly), else fixed-size
        windows over a flat `.turns` list with a synthetic incrementing
        timestamp (msc/molweni/personachat don't carry per-turn dates)."""
        if hasattr(conversation, "sessions"):
            from dateutil import parser as date_parser

            units = []
            base = datetime(2024, 1, 1)
            for idx in sorted(conversation.sessions):
                raw = None
                if hasattr(conversation, "session_datetimes"):
                    raw = conversation.session_datetimes.get(idx)
                try:
                    ts = date_parser.parse(raw).isoformat() if raw else (base + timedelta(hours=idx)).isoformat()
                except (ValueError, TypeError):
                    ts = (base + timedelta(hours=idx)).isoformat()
                units.append((conversation.sessions[idx], ts))
            return units
        if hasattr(conversation, "turns"):
            turns = conversation.turns
            base = datetime(2024, 1, 1)
            units = []
            for i, start in enumerate(range(0, len(turns), self.window_size)):
                ts = (base + timedelta(hours=i)).isoformat()
                units.append((turns[start : start + self.window_size], ts))
            return units or [([], base.isoformat())]
        raise ValueError(
            "lightmem: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
            "`.turns` list -- can't extract memory-ingestion units for this dataset's shape"
        )

    def answer(self, item) -> MethodOutput:
        start = time.time()
        retrieved = self._memory.retrieve(item.question, limit=self.top_k)
        context = "\n".join(retrieved) if retrieved else "(no relevant memories found)"

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        prompt = f"Relevant memories:\n{context}\n\nQuestion: {question}\nAnswer concisely based on the memories above."
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        return MethodOutput(answer=resp.text.strip(), latency=time.time() - start)

    def reset(self) -> None:
        self._memory = None
        if self._db_path:
            rmtree(self._db_path, ignore_errors=True)
        self._db_path = None
        self._conversation = None
