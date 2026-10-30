"""raptor -- family B baseline (BASELINE_METRIC_PLAN.md §4): RAPTOR-style
hierarchical summary tree + retrieval (Sarthi et al. 2024,
https://github.com/parthsarthi03/raptor), for conversations too long for
full_context/running_summary to handle well.

Vendored (MIT-licensed) under vendor/raptor_lib/, trimmed to drop its
OpenAI/SBert/T5-specific concrete Embedding/Summarization/QA model classes --
those called `openai`/`sentence-transformers`/`transformers` directly, which
plugins here must never do (CLAUDE.md: always go through
dialogue_benchmarks.llm). Only the library's clustering/tree/retrieval logic
and its three abstract base classes (BaseEmbeddingModel,
BaseSummarizationModel, BaseQAModel) are kept; this module supplies its own
thin subclasses backed by dialogue_benchmarks.llm.embed()/complete(). See
vendor/raptor_lib/*.py module docstrings for the exact diffs from upstream.

Algorithm (prepare()): flattens the conversation into one transcript (same
convention as full_context), splits it into ~100-token leaf chunks, embeds
each leaf, then recursively clusters leaves (UMAP dimensionality reduction +
Gaussian-mixture soft clustering) and summarizes each cluster into a parent
node, layer by layer, until a layer is too small to cluster further -- this
is the "tree." At answer time (answer()), retrieves the top-k nodes by
embedding cosine distance across the *collapsed* tree (all layers considered
together, so a leaf-level detail and a high-layer summary can both surface),
then answers from just that retrieved context via
dialogue_benchmarks.llm.complete(). Comparison point against running_summary:
does preserving a queryable multi-resolution tree (vs. collapsing everything
into one linear summary) retain more on very long conversations?

Building the tree is real API spend (one embed() call per leaf/node, one
complete() call per cluster-summary) -- charged to the first answer() after
prepare() (tokens/cost), zero for every later question against the same
conversation, same convention as running_summary/dense_rag. Each answer()'s
own query-embedding + final-QA calls are added directly to that item's cost.

Works with any conversation shape that's either session-grouped (`.sessions`)
or flat (`.turns`) -- see full_context for the same convention.
"""
import sys
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _load_vendor():
    """Imports tiktoken/umap-learn/scikit-learn/scipy and the vendored
    raptor_lib code lazily, so merely importing this module (e.g.
    `db validate`, `db list`) doesn't require the heavy `raptor` extra --
    only actually constructing Method() does."""
    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))
    try:
        from raptor_lib import (
            BaseEmbeddingModel,
            BaseQAModel,
            BaseSummarizationModel,
            RetrievalAugmentation,
            RetrievalAugmentationConfig,
        )
    except ImportError as e:
        raise ImportError(
            "raptor: missing a dependency for this baseline -- install it with "
            "`.venv/bin/pip install -e '.[raptor]'` (tiktoken/umap-learn/scikit-learn/scipy, "
            "declared as an optional extra since they're heavy and raptor-specific)"
        ) from e
    return BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel, RetrievalAugmentation, RetrievalAugmentationConfig


def _build_transcript(conversation) -> str:
    if hasattr(conversation, "sessions"):
        datetimes = getattr(conversation, "session_datetimes", {})
        lines = []
        for session_idx in sorted(conversation.sessions):
            dt = datetimes.get(session_idx)
            lines.append(f"[Session {session_idx}" + (f" -- {dt}]" if dt else "]"))
            lines.extend(f"{turn.speaker}: {turn.text}" for turn in conversation.sessions[session_idx])
        return "\n".join(lines)
    if hasattr(conversation, "turns"):
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in conversation.turns)
    raise ValueError(
        "raptor: conversation has neither `.sessions` (dict of session_idx -> turns) "
        "nor a flat `.turns` list -- can't build a transcript for this dataset's shape"
    )


_SUMMARIZE_PROMPT = (
    "Write a dense, factual summary of the following conversation excerpt, preserving as "
    "many key details (facts, dates, names, events, stated preferences) as possible:\n\n{context}"
)


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        max_tokens: int = 512,
        leaf_max_tokens: int = 100,
        num_layers: int = 5,
        summarization_length: int = 150,
        top_k: int = 10,
        retrieval_max_tokens: int = 2000,
        **config,
    ):
        super().__init__(
            model=model,
            embed_model=embed_model,
            max_tokens=max_tokens,
            leaf_max_tokens=leaf_max_tokens,
            num_layers=num_layers,
            summarization_length=summarization_length,
            top_k=top_k,
            retrieval_max_tokens=retrieval_max_tokens,
            **config,
        )
        (
            BaseEmbeddingModel,
            BaseQAModel,
            BaseSummarizationModel,
            self._RetrievalAugmentation,
            self._RetrievalAugmentationConfig,
        ) = _load_vendor()

        self.model = model
        self.embed_model = embed_model
        self.max_tokens = max_tokens
        self.leaf_max_tokens = leaf_max_tokens
        self.num_layers = num_layers
        self.summarization_length = summarization_length
        self.top_k = top_k
        self.retrieval_max_tokens = retrieval_max_tokens

        self._running_tokens = 0
        self._running_cost = 0.0

        method = self

        class _Embedding(BaseEmbeddingModel):
            def create_embedding(self_, text):
                resp = embed(text, model=method.embed_model)
                method._running_tokens += resp.tokens_in
                method._running_cost += resp.cost or 0.0
                return resp.vectors[0]

        class _Summarizer(BaseSummarizationModel):
            def summarize(self_, context, max_tokens=150):
                resp = complete(_SUMMARIZE_PROMPT.format(context=context), model=method.model, max_tokens=max_tokens)
                method._running_tokens += resp.tokens_in + resp.tokens_out
                method._running_cost += resp.cost or 0.0
                return resp.text.strip()

        class _QA(BaseQAModel):
            # Never actually called -- answer() below does its own complete()
            # call so it can honor item.options and match this repo's other
            # baselines' MethodOutput/prompt conventions. RetrievalAugmentation
            # still requires a valid BaseQAModel instance to construct, though.
            def answer_question(self_, context, question):
                resp = complete(f"Context:\n{context}\n\nQuestion: {question}", model=method.model)
                return resp.text.strip()

        self._embedding_model = _Embedding()
        self._summarization_model = _Summarizer()
        self._qa_model = _QA()

        self._ra = None
        self._ingest_tokens = 0
        self._ingest_cost = 0.0
        self._ingest_charged = True

    def prepare(self, conversation) -> None:
        transcript = _build_transcript(conversation)
        self._running_tokens = 0
        self._running_cost = 0.0

        config = self._RetrievalAugmentationConfig(
            embedding_model=self._embedding_model,
            summarization_model=self._summarization_model,
            qa_model=self._qa_model,
            tb_max_tokens=self.leaf_max_tokens,
            tb_num_layers=self.num_layers,
            tb_summarization_length=self.summarization_length,
            tr_top_k=self.top_k,
        )
        self._ra = self._RetrievalAugmentation(config=config)
        self._ra.add_documents(transcript)

        self._ingest_tokens = self._running_tokens
        self._ingest_cost = self._running_cost
        self._ingest_charged = False

    def answer(self, item) -> MethodOutput:
        before_tokens, before_cost = self._running_tokens, self._running_cost
        context, layer_info = self._ra.retrieve(
            item.question,
            top_k=self.top_k,
            max_tokens=self.retrieval_max_tokens,
            collapse_tree=True,
            return_layer_information=True,
        )
        retrieval_tokens = self._running_tokens - before_tokens
        retrieval_cost = self._running_cost - before_cost

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        prompt = (
            "Answer the question using only the information in the retrieved excerpts below "
            "(they may be incomplete relative to the full conversation, and range from raw "
            "details to higher-level summaries). Be concise and direct -- answer with just the "
            "fact asked for, no extra explanation.\n\n"
            f"### Retrieved excerpts\n{context}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)

        ingest_tokens, ingest_cost = (0, 0.0) if self._ingest_charged else (self._ingest_tokens, self._ingest_cost)
        self._ingest_charged = True

        return MethodOutput(
            answer=resp.text.strip(),
            retrieved=[f"layer {n['layer_number']} node {n['node_index']}" for n in layer_info] or [context],
            tokens=ingest_tokens + retrieval_tokens + resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=ingest_cost + retrieval_cost + (resp.cost or 0.0),
        )

    def reset(self) -> None:
        self._ra = None
        self._ingest_tokens = 0
        self._ingest_cost = 0.0
        self._ingest_charged = True
        self._running_tokens = 0
        self._running_cost = 0.0
