"""RF-Mem -- Evoking User Memory: Personalizing LLM via
Recollection-Familiarity Adaptive Retrieval (Applied-Machine-Learning-Lab/
ICLR2026_RF-Mem, ICLR 2026). Sits at the boundary between family A
(full-context/RAG comparison baselines) and family C (memory-agent
frameworks): it's a retrieval-only technique with no LLM-driven fact
extraction at ingestion time, but the retrieval *strategy itself* is the
actual research contribution (unlike bm25_rag/dense_rag's fixed top-k),
so it's grouped with the other family-C entries in
BASELINE_METRIC_PLAN.md/README.md rather than treated as a fourth
full-context-style baseline.

**The mechanism** (`vendor/rf_mem.py`, adapted from upstream's own
`RF_mem/LongMemEval/src/retrieval/RF_mem.py`, confirmed by reading the
actual pipeline dispatch code, not just the README's framing): grounded
in the dual-process theory of human memory (Familiarity = fast
recognition, Recollection = slow deliberate reconstruction). At query
time, a cheap probe retrieval (`decide_strategy`) checks mean similarity
score and score entropy over the top candidates:
- High confidence (high mean score, or low entropy) -> "fast" path:
  plain top-k cosine/inner-product similarity search.
- Low confidence -> "slow" path (`retrieve_pyramid_v2`): iterative beam
  search over query-embedding branches. At each level, newly-retrieved
  candidates are MMR-reranked for diversity, then KMeans-clustered into
  `beam_width` groups; each group's centroid nudges the query embedding
  for the next level (`alpha` interpolates old query vs. new centroid,
  plus a pull back toward the *original* query so branches don't drift
  arbitrarily far). This lets a genuinely uncertain query iteratively
  "reconstruct" what it's looking for across multiple rounds instead of
  committing to one static top-k, while a confident query stays cheap.

**Zero LLM calls in the retrieval path itself** -- confirmed by reading
upstream's actual `if args.retrieval_mode == 'RF-Mem':` dispatch branch
in the original file: it only ever calls `decide_strategy_with_probe`
(pure numpy over embedding-search scores) then either `retriever.query()`
or `retriever.retrieve_pyramid_v2()` (both pure `faiss`/`sklearn`/numpy).
Upstream's file does define a `gen_retrieval_prompt_HyDE()` helper that
would call an LLM to expand the query before embedding it, but it is
never actually invoked anywhere in the file (dead code) -- confirmed via
grep, not assumed. This means the entire retrieval mechanism (both
paths, and the fast/slow decision) is fully verifiable without any LLM
API access, only `dialogue_benchmarks.llm.embed()`'s local provider --
the final answer-generation step (given the retrieved chunks) is the only
part of this Method that calls `complete()`, same division as
bm25_rag/dense_rag.

**Adapted from upstream, not a bug fix**: upstream's `RAGRetriever` calls
a raw `SentenceTransformer.encode()` in-process; this Method routes every
embedding call through `dialogue_benchmarks.llm.embed()` instead (shared
cache, consistent provider across baselines) -- `vendor/rf_mem.py`'s
`RAGRetriever` methods take pre-computed embedding vectors as arguments
rather than raw text. `embed()`'s local provider only has one model
registered (`all-MiniLM-L6-v2`, not upstream's own `multi-qa-MiniLM-L6-
cos-v1`/`BAAI/bge-base-en-v1.5` defaults) -- adding a new local-embedding
model means editing `src/dialogue_benchmarks/llm.py`, out of scope for a
baseline-worktree change (see CLAUDE.md's ownership table), so this
Method uses the one already-registered local model instead; both are
small sentence-transformers models from the same MiniLM-L6 lineage, not a
fundamentally different embedding space. Also dropped several confirmed-
dead imports from the vendored file (`cvxpy`, `ot`, `pandas`,
`scipy.sparse.*`, `sklearn.decomposition.TruncatedSVD`,
`sklearn.metrics.pairwise.cosine_similarity`) -- see vendor/rf_mem.py's
own module docstring.

**No dataset adapter exists yet in this repo for RF-Mem's own target
benchmarks** (LongMemEval, PersonaBench, PersonaMem -- all A's lane, none
implemented) -- this Method instead targets any conversation already
supported here via the same `.sessions`/`.turns` chunking convention as
mem0/lightmem/memalpha's Methods (one chunk per session for
session-grouped conversations like locomo, fixed-size windows over a flat
`.turns` list otherwise).

**Licensing**: no repository-level LICENSE file (README claims MIT,
GitHub API reports `license: None`) -- same situation and resolution as
pathrag/orig/sbys/lmedr/memalpha this session: internal reproduction
only, reconfirmed with the user.

Verification: since the entire retrieval mechanism needs no LLM access,
this was verified for real (not mocked) against a real multi-session
locomo conversation -- confirmed both `query()`/`retrieve_pyramid_v2()`
return real, on-topic retrieved sessions for real locomo questions, and
the KMeans/MMR expansion in the slow path runs without error across
multiple beam levels. **Threshold recalibration found by this real run**:
upstream's own `run.sh` uses `score_th=0.6` (tuned for their own
`multi-qa-MiniLM-L6-cos-v1`/`BAAI/bge-base-en-v1.5` embeddings); with
`all-MiniLM-L6-v2` (the embedding model substitution above) real locomo
query/session similarity scores never reached 0.6 even for a
near-verbatim-match probe query, so `decide_strategy` always picked
"slow" regardless of query specificity -- confirmed the "fast" path
still works correctly by lowering `score_th` and re-running the same
probe (mean_score 0.487 with `score_th=0.3` correctly triggered "fast"
and returned valid results). Lowered this Method's own default
`score_th` from upstream's `0.6` to `0.3` to actually exercise both
paths with this embedding model's real score distribution, rather than
silently defaulting to "always slow."

**`top_k` vs `search_pool_k` -- found via real LongMemEval-M testing, not
a hypothetical**: `retrieve_pyramid_v2`'s beam search only explores as
many candidates as `out_k` calls for (`K = (beam_width+lvl)*fanout` per
level; the search stops expanding once that's covered), so passing a
small final answer-count directly as `out_k` makes the "slow" path
under-explore and collapse toward "fast"'s behavior -- confirmed:
running both paths on a 300-sample LongMemEval-M subset with
`out_k=top_k=10` gave byte-identical recall (0.7367) for "fast",
"slow", and adaptive; re-running a 60-sample spot check with the
search pool widened to 50 broke the byte-identical tie (0.6833 vs
0.7000 recall_any@5), which is what motivated decoupling the two
parameters below. Checked whether upstream itself separates these: in
`RF_mem/LongMemEval/src/retrieval/RF_mem.py` (the actual LongMemEval
driver, i.e. the benchmark the paper reports numbers on) it does --
`topk_uni=50` is fixed and passed as both `query()`'s `top_k` and
`retrieve_pyramid_v2()`'s `out_k`, with the final k=5/10/20/50 cutoff
applied only at scoring time (`evaluate_retrieval(..., k=k)`). Two
*other* experiment scripts in the same upstream repo
(`personamem_data/main_batch.py`, `personabench_data/main_batch.py`,
neither targeting the benchmark this paper reports) do NOT separate
them -- they pass the CLI's small `top_k`/`num_chunks` straight through
as `out_k`. Since LongMemEval is the paper's actual reported benchmark,
this Method follows that convention: `search_pool_k` (default `50`,
matching upstream's `topk_uni`) sizes the internal search/beam pool for
both `query()` and `retrieve_pyramid_v2()`; `top_k` (default `5`) only
truncates the result list afterward, right before it's joined into the
final answer-generation prompt. `search_pool_k` should stay `>= top_k`.

**Re-verified after the fix -- honest result, not the hoped-for one**:
with `search_pool_k=50`/`top_k=5` wired through `Method.answer()`
itself (not a bespoke script) and scored with upstream's own
`recall_any@5`/`recall_all@5`/`ndcg_any@5` (`eval_utils.py`, ported
inline rather than assumed), a proper 150-sample LongMemEval-M run
found `recall_any`/`recall_all` **identical** between "fast"-forced and
"slow"-forced (0.7067/0.6667), with `ndcg` differing only at the third
decimal (0.6456 vs 0.6432); `score_th=0.3` routes 135/150 questions to
"fast" on this dataset's real score distribution, and even the 15 that
go "slow" score close enough to fast that the adaptive aggregate is
indistinguishable from fast-only. The earlier 60-sample spot check's
apparent divergence did not hold up at n=150 with the real metric.
**Conclusion**: the `search_pool_k` decoupling itself is a correctness
fix (matches upstream's actual LongMemEval convention, confirmed by
reading their driver script directly) and should stay, but it does not
by itself demonstrate that RF-Mem's adaptive strategy beats plain
top-k retrieval in this repo's setup -- that remains unconfirmed both
on locomo and on LongMemEval-M. Absolute recall (~0.71) also still
runs above the paper's own reported LongMemEval-M range (~0.42-0.52);
the leading hypothesis is the embedding-model substitution
(`all-MiniLM-L6-v2` vs. the paper's `multi-qa-MiniLM-L6-cos-v1`/
`BAAI/bge-base-en-v1.5`), but that's unconfirmed -- `embed()`'s local
provider whitelist (`_LOCAL_EMBED_MODELS` in `src/dialogue_benchmarks/
llm.py`) has no other local embedding model registered, and adding one
is outside this worktree's lane (CLAUDE.md ownership table).
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
    from rf_mem import RAGRetriever, decide_strategy

    return RAGRetriever, decide_strategy


def _extract_chunks(conversation) -> list:
    """Same chunking convention as mem0/lightmem/memalpha's Methods: one
    chunk per session if session-grouped (`.sessions`, e.g. locomo), else
    fixed-size turn windows over a flat `.turns` list."""
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
        "rfmem: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract chunks for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        search_pool_k: int = 50,
        probe_k: int = 10,
        ent_th: float = 0.2,
        score_th: float = 0.3,
        score_bt: float = 0.0,
        alpha: float = 0.8,
        beam_width: int = 4,
        fanout: int = 1,
        depth: int = 10,
        max_tokens: int = 512,
        **config,
    ):
        super().__init__(
            model=model,
            embed_model=embed_model,
            top_k=top_k,
            search_pool_k=search_pool_k,
            probe_k=probe_k,
            ent_th=ent_th,
            score_th=score_th,
            score_bt=score_bt,
            alpha=alpha,
            beam_width=beam_width,
            fanout=fanout,
            depth=depth,
            max_tokens=max_tokens,
            **config,
        )
        RAGRetriever, decide_strategy = _load_vendor()
        self._RAGRetriever = RAGRetriever
        self._decide_strategy = decide_strategy

        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.search_pool_k = search_pool_k
        self.probe_k = probe_k
        self.ent_th = ent_th
        self.score_th = score_th
        self.score_bt = score_bt
        self.alpha = alpha
        self.beam_width = beam_width
        self.fanout = fanout
        self.depth = depth
        self.max_tokens = max_tokens

        self._retriever = None
        self._conversation = None

    def prepare(self, conversation) -> None:
        self._conversation = conversation
        chunks = _extract_chunks(conversation)
        vectors = np.array(embed(chunks, model=self.embed_model).vectors, dtype=np.float32)
        self._retriever = self._RAGRetriever()
        self._retriever.build_index(chunks, list(range(len(chunks))), vectors)

    def answer(self, item) -> MethodOutput:
        start = time.time()
        q_emb = np.array(embed(item.question, model=self.embed_model).vectors[0], dtype=np.float32)

        mode, diag = self._decide_strategy(
            self._retriever,
            q_emb,
            tau=0.0,
            probe_k=self.probe_k,
            ent_th=self.ent_th,
            score_th=self.score_th,
            score_bt=self.score_bt,
        )
        if mode == "fast":
            _, _, retrieved_chunks, _ = self._retriever.query(q_emb, top_k=self.search_pool_k, tau=0.0)
        else:
            _, _, retrieved_chunks, _ = self._retriever.retrieve_pyramid_v2(
                q_emb,
                out_k=self.search_pool_k,
                tau=0.0,
                depth=self.depth,
                beam_width=self.beam_width,
                fanout=self.fanout,
                expansion="mean_group",
                alpha=self.alpha,
            )
        retrieved_chunks = retrieved_chunks[: self.top_k]

        context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "(no relevant memories found)"
        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)

        prompt = f"{context}\n\nQuestion: {question}\nAnswer concisely based on the context above."
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        return MethodOutput(answer=resp.text.strip(), latency=time.time() - start)

    def reset(self) -> None:
        self._retriever = None
        self._conversation = None
