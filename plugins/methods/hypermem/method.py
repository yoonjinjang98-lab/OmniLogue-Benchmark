"""hypermem -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps HyperMem
(EverMind-AI/HyperMem, Apache-2.0, ACL 2026) -- a three-level hypergraph
(topic -> episode -> fact) memory architecture with coarse-to-fine top-down
retrieval (BM25+dense RRF fusion per layer).

Vendored under vendor/hypermem/ (not pip-installed -- no PyPI package exists).
Unlike every other family-C wrapper here, HyperMem ships as a **5-stage
file-based pipeline** (`hypermem/main/stage1..5_*.py`, ~5000 lines, each
stage a standalone script reading the previous stage's JSON/pickle output
from disk and writing its own), not a single importable class -- there is no
`HyperMem` class with `.index()`/`.answer()` to call. What makes this
tractable rather than a full reimplementation: every stage's *actual logic*
already lives in real functions/classes (`episode_extraction_from_conversation`,
`process_single_conversation`, `build_bm25_index_for_single_conv`,
`build_embedding_index_for_single_conv`, `hierarchical_retrieval`) that are
already scoped to one conversation at a time -- only `stage1`'s own
`load_locomo_raw_data()` is LoCoMo-specific (a JSON parser, not the
extraction logic itself). This method calls those functions directly,
chained through a temp directory using the exact same JSON/pickle file names
upstream's own stages read and write, and replaces `load_locomo_raw_data()`
with `_turns_to_messages()` below to feed in this repo's `.sessions`/`.turns`
conversations instead.

Two provider seams route through complete()/embed(), both plain duck-typed
classes (no ABC/Protocol to subclass -- `hypermem.llm.protocol.LLMProvider`
is a `typing.Protocol`, and the concrete `hypermem.llm.llm_provider.LLMProvider`
facade only special-cases `provider_type="openai"`, so a lookalike class is
simplest, same pattern as every other family-C baseline here):
- `_DBLLMProvider.generate(prompt, ...) -> str`, plus `get_accumulated_stats()`/
  `reset_accumulated_stats()` (called directly by stage2's
  `process_single_conversation`, not part of the Protocol but required by
  the real call sites).
- `_DBEmbeddingProvider.embed(texts) -> list[list[float]]` (sync, unlike the
  LLM side).

The final answer-generation step (`stage5_response.py:locomo_response`) is
NOT reused -- it calls a raw OpenAI SDK client (`llm_client.chat.completions.
create(...)`) directly rather than going through `LLMProvider`, so this
method instead reuses its exact prompt template
(`hypermem.prompts.answer_prompts.ANSWER_PROMPT_NEMORI_COT`, the config
default `answer_type="cot"`) and calls `complete()` itself, replicating the
"FINAL ANSWER:" extraction `locomo_response()` does.

Reranking (`use_reranker`) defaults to `False` upstream and is left that way
here -- it would need Qwen3-Reranker-4B served locally via vLLM, same
GPU-serving assumption already avoided for the embedding side.

**Verification gap, stated plainly**: `prepare()`'s call to `_stage2.process_single_conversation()`
(topic extraction -> fact extraction -> hypergraph construction) has **not**
been exercised end-to-end against a mocked LLM, and is the one piece of this
method's actual code path that isn't. Unlike every other family-C baseline
here, that call could not be smoke-tested with a static mock:
`hypermem/extractors/topic_extractor.py` and `fact_extractor.py` retry
topic/fact/role-assignment extraction in unbounded `while True:` loops with
no max-attempt cap whenever the LLM response fails validation (missing
fields, or an id that doesn't match the *exact* current valid set, e.g.
"episode_1" but not also "episode_2" once a topic only actually contains one
episode) -- a real robustness gap (every other framework wrapped today has
some retry cap or graceful-degradation path instead). A real LLM reliably
echoes back the ids it's given in context, so this isn't expected to bite in
practice, but it makes a synthetic mock impractical without re-deriving
HyperMem's own id-generation to keep every nested schema self-consistent
across stage2's several extraction sub-steps.

What *is* actually verified:
- `stage1` (episode boundary detection + episode memory generation, called
  from `prepare()` just before stage2) runs cleanly against a mocked LLM
  (its own validation has safe `.get(key, default)` fallbacks, no retry
  loop) -- confirms `_DBLLMProvider` and `_turns_to_messages()` work.
- `stage3`/`stage4`/answer-generation (the rest of `prepare()` plus all of
  `answer()`) verified end-to-end against a *hand-built* valid `Hypergraph`
  standing in for stage2's real output (bypassing stage2 entirely, not
  exercising it): BM25 + embedding index building, `hierarchical_retrieval()`,
  `format_hierarchical_results()`, and the final `complete()` call with
  "FINAL ANSWER:" extraction all confirmed to correctly surface a
  fact-grounded answer from real hypergraph content.

Net effect: this method has not been run start-to-finish on real data even
with a mocked LLM. It should work against a real LLM (which won't trip the
validation loops), but that's an expectation, not something confirmed here --
worth an actual `db run` with a real API key before trusting results from it.
"""
import asyncio
import importlib.util
import json
import pickle
import re
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from hypermem.utils.datetime_utils import get_now_with_timezone, to_iso_format  # noqa: E402


def _load_stage_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _VENDOR_DIR / "hypermem" / "main" / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stage1 = _load_stage_module("hypermem_stage1", "stage1_memory_extraction.py")
_stage2 = _load_stage_module("hypermem_stage2", "stage2_hypergraph_extraction.py")
_stage3 = _load_stage_module("hypermem_stage3", "stage3_hypergraph_index.py")
_stage4 = _load_stage_module("hypermem_stage4", "stage4_hypergraph_retrieval.py")

from hypermem.config import ExperimentConfig  # noqa: E402
from hypermem.extractors.conv_episode_extractor import ConvEpisodeExtractor  # noqa: E402
from hypermem.prompts.answer_prompts import ANSWER_PROMPT_NEMORI_COT  # noqa: E402
from hypermem.structure import Hypergraph  # noqa: E402

_CHUNK_TURNS = 1  # one HyperMem "message" per turn (episode boundary detection needs turn granularity)


class _DBLLMProvider:
    """Duck-types hypermem.llm.llm_provider.LLMProvider's real call surface."""

    def __init__(self, model: str):
        self.model = model
        self._prompt_tokens = 0
        self._completion_tokens = 0

    async def generate(self, prompt: str, temperature=None, extra_body=None, response_format=None) -> str:
        resp = complete(prompt, model=self.model, max_tokens=2048)
        self._prompt_tokens += resp.tokens_in
        self._completion_tokens += resp.tokens_out
        return resp.text

    def get_accumulated_stats(self):
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }

    def reset_accumulated_stats(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0


class _DBEmbeddingProvider:
    """Duck-types hypermem.llm.embedding_provider.EmbeddingProvider."""

    def __init__(self, model: str):
        self.model = model

    def embed(self, texts) -> list:
        return embed(list(texts), model=self.model).vectors

    def cosine_similarity(self, query_vec, doc_vecs):
        import numpy as np

        dot = np.dot(doc_vecs, query_vec)
        denom = np.linalg.norm(query_vec) * np.linalg.norm(doc_vecs, axis=1)
        denom[denom == 0] = 1e-9
        return dot / denom


def _turns_to_messages(conversation) -> list:
    """HyperMem message-dict shape, matching stage1's own
    load_locomo_raw_data() output (the only LoCoMo-specific piece of the
    pipeline -- everything downstream just consumes this shape)."""
    if hasattr(conversation, "sessions"):
        flat = [
            (f"session_{idx}", turn) for idx in sorted(conversation.sessions) for turn in conversation.sessions[idx]
        ]
    elif hasattr(conversation, "turns"):
        flat = [("session_0", turn) for turn in conversation.turns]
    else:
        raise ValueError(
            "hypermem: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
            "`.turns` list -- can't build messages for this dataset's shape"
        )

    base_time = get_now_with_timezone()
    messages = []
    for i, (session_key, turn) in enumerate(flat):
        speaker_id = turn.speaker.lower().replace(" ", "_")
        messages.append(
            {
                "speaker_id": speaker_id,
                "user_name": turn.speaker,
                "speaker_name": turn.speaker,
                "content": turn.text,
                "timestamp": to_iso_format(base_time + timedelta(seconds=i * 30)),
                "dia_id": f"turn_{i}",
                "session": session_key,
            }
        )
    return messages


_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:", re.IGNORECASE)


def _extract_final_answer(text: str) -> str:
    parts = _FINAL_ANSWER_RE.split(text, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else text.strip()


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "Qwen/Qwen3-Embedding-4B",  # HyperMem's own paper default
        **config,
    ):
        super().__init__(model=model, embed_model=embed_model, **config)
        self.model = model
        self.embed_model = embed_model
        self._path = None
        self._hypergraph = None
        self._bm25 = None
        self._docs = None
        self._emb_index = None
        self._exp_config = None

    def prepare(self, conversation) -> None:
        self._path = Path(tempfile.mkdtemp(prefix="hypermem_"))
        hg_dir = self._path / "hypergraphs"
        bm25_dir = self._path / "bm25_index"
        emb_dir = self._path / "vectors"
        for d in (hg_dir, bm25_dir, emb_dir):
            d.mkdir(parents=True, exist_ok=True)

        llm_provider = _DBLLMProvider(self.model)
        embedding_provider = _DBEmbeddingProvider(self.embed_model)

        conv_id = "0"
        messages = _turns_to_messages(conversation)
        raw_data_list = _stage1.convert_conversation_to_raw_data_list(messages)

        episode_extractor = ConvEpisodeExtractor(llm_provider=llm_provider)
        episode_list = asyncio.run(
            _stage1.episode_extraction_from_conversation(
                raw_data_list, llm_provider=llm_provider, episode_extractor=episode_extractor, conv_id=conv_id
            )
        )
        episodes_file = self._path / f"episode_list_conv_{conv_id}.json"
        with open(episodes_file, "w", encoding="utf-8") as f:
            json.dump([ep.to_dict() for ep in episode_list], f, ensure_ascii=False)

        hypergraph = asyncio.run(
            _stage2.process_single_conversation(
                conv_id=conv_id,
                episodes_file=episodes_file,
                save_dir=hg_dir,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                skip_existing=False,
            )
        )
        if hypergraph is None:
            hypergraph = Hypergraph()
        hypergraph_file = hg_dir / f"hypergraph_conv_{conv_id}.json"
        with open(hypergraph_file, "w", encoding="utf-8") as f:
            json.dump(hypergraph.to_dict(), f, ensure_ascii=False)

        stemmer = _stage3.PorterStemmer()
        _stage3.ensure_nltk_data()
        stop_words = set(_stage3.stopwords.words("english"))
        _stage3.build_bm25_index_for_single_conv(conv_id, hg_dir, bm25_dir, stemmer, stop_words, skip_existing=False)
        _stage3.build_embedding_index_for_single_conv(
            conv_id, hg_dir, emb_dir, embedding_provider, skip_existing=False
        )

        with open(hypergraph_file, "r", encoding="utf-8") as f:
            self._hypergraph = json.load(f)
        bm25_file = bm25_dir / f"hypergraph_bm25_index_conv_{conv_id}.pkl"
        if bm25_file.exists():
            with open(bm25_file, "rb") as f:
                index_data = pickle.load(f)
            self._bm25 = index_data["bm25"]
            self._docs = index_data["docs"]
        emb_file = emb_dir / f"hypergraph_embedding_index_conv_{conv_id}.pkl"
        if emb_file.exists():
            with open(emb_file, "rb") as f:
                self._emb_index = pickle.load(f)

        self._llm_provider = llm_provider
        self._embedding_provider = embedding_provider
        self._exp_config = ExperimentConfig()

    def answer(self, item) -> MethodOutput:
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        results = _stage4.hierarchical_retrieval(
            question,
            self._hypergraph,
            bm25=self._bm25,
            docs=self._docs,
            emb_index=self._emb_index,
            embedding_provider=self._embedding_provider,
            reranker_provider=None,
            config=self._exp_config,
        )
        speakers = self._hypergraph.get("metadata", {}) or {}
        context = _stage4.format_hierarchical_results(
            results, speakers.get("speaker_a", "Speaker A"), speakers.get("speaker_b", "Speaker B")
        )
        # `results` also carries a "retrieval_log" key (a debug dict, not a
        # list of (data, score) pairs like the three real layers) -- only
        # those three are safe to iterate here.
        retrieved = [
            f"[{layer}] {data.get('content') or data.get('summary') or data}"
            for layer in ("topics", "episodes", "facts")
            for data, _score in results.get(layer, [])
        ]

        prompt = ANSWER_PROMPT_NEMORI_COT.format(context=context, question=question)
        resp = complete(prompt, model=self.model, max_tokens=2048)
        answer = _extract_final_answer(resp.text)
        return MethodOutput(
            answer=answer, retrieved=retrieved, tokens=resp.tokens_in + resp.tokens_out, latency=resp.latency, cost=resp.cost
        )

    def reset(self) -> None:
        self._hypergraph = None
        self._bm25 = None
        self._docs = None
        self._emb_index = None
        self._exp_config = None
        if self._path:
            shutil.rmtree(self._path, ignore_errors=True)
        self._path = None
