"""groupmembench_bm25 -- family D baseline (BASELINE_METRIC_PLAN.md §4): wraps
groupmembench's own BM25 baseline (https://github.com/UCSB-NLP-Chang/GroupMemBench,
no LICENSE file -- vendored here for internal benchmark-reproduction use, not
redistribution; see conversation history for the license check) instead of
reimplementing generic bm25_rag against this dataset. Family D's point is
reproducing exactly what a dataset's own paper reports, not a from-scratch
approximation -- groupmembench's headline finding is that this baseline
"matches memory systems" on their question set, which is only a meaningful
comparison if the tokenizer/passage format matches theirs byte-for-byte.

Ported from baselines/rag_common/eval_lib.py + baselines/bm25/eval_benchmark.py:
- `_tokenize()`: their exact word-level regex tokenizer feeding BM25Okapi
  (rank_bm25 is already this repo's own core dependency)
- message loading/sort order: content-only text goes into the BM25 index;
  the rest of a message's metadata (author/role/channel/phase/topic/
  timestamp/reply_to/msg_node) is attached to the passage only when it's
  shown to the QA model, never at retrieval time
- `_format_retrieved_message()`: their exact passage header format
- the agent system prompt (`prompts/hipporag_agent_system.txt`), reused
  verbatim for the QA call so answer generation stays faithful to upstream
- `_extract_final()`: their answer-parsing logic (the agent prompt asks for
  reasoning + a trailing `Final: <answer>` line)

Deliberately NOT ported: their own gpt-5 agent+judge loop
(eval_lib.py:run_qa/call_chat). This repo's Method/Metric contracts keep
answering and scoring separate -- score this baseline's answers with
qa_em/qa_f1/qa_llm_judge instead of baking a second judge call into the
method itself.

No dataset adapter for groupmembench exists yet (plugins/datasets/**, A's
lane -- see BASELINE_METRIC_PLAN.md), so this hasn't been smoke-tested
end-to-end. Expects a conversation shaped like groupmembench's own raw JSON
(`.channels`: dict of channel_name -> list of message dicts with `content`/
`author`/`role`/`timestamp`/`msg_node`/...) if a future adapter preserves
that shape; falls back to this repo's generic `.sessions`/`.turns`
convention otherwise so it still runs against any dataset like this repo's
other RAG baselines, at the cost of the extra metadata tags
`_format_retrieved_message()` would otherwise show the QA model.
"""
import html
import re

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Verbatim from groupmembench's prompts/hipporag_agent_system.txt.
_AGENT_SYSTEM_PROMPT = (
    "You are a careful QA agent. Use only the retrieved conversation passages to answer the question.\n"
    "First provide a chain-of-thought reasoning paragraph. Then provide the final answer on a new line "
    "using the format:\n"
    "Final: <answer>\n"
    "If the answer is not supported by the retrieved passages, say so in the Final line."
)


def _tokenize(text: str) -> list:
    """Verbatim from baselines/bm25/eval_benchmark.py -- word-level, no
    stopword removal, fully deterministic."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _message_index_text(message: dict) -> str:
    content = message.get("content", "")
    if not isinstance(content, str):
        return ""
    return html.unescape(content).strip()


def _format_retrieved_message(message: dict) -> str:
    author = message.get("author") or "?"
    tags = [f"user={author}"]
    for src_key, label in (
        ("role", "speaker_role"),
        ("_channel", "channel"),
        ("phase_name", "phase_name"),
        ("topic", "topic"),
        ("timestamp", "timestamp"),
        ("reply_to", "reply_to"),
        ("msg_node", "msg_node"),
    ):
        value = message.get(src_key)
        if value not in (None, ""):
            tags.append(f"{label}={value}")
    body = message.get("content", "")
    body = html.unescape(body).strip() if isinstance(body, str) else ""
    return f"[{' / '.join(tags)}]\n{body}".strip()


def _extract_final(text: str) -> str:
    """Verbatim logic from eval_lib.py:extract_final. The agent prompt asks
    for reasoning followed by a `Final: <answer>` line; this pulls out just
    the final answer so it can be scored by qa_em/qa_f1 like any other
    baseline's answer."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if low.startswith("final:") or low.startswith("final answer:"):
            return ln.split(":", 1)[1].strip()
    return lines[-1] if lines else text.strip()


def _extract_messages(conversation) -> list:
    """Returns message dicts shaped like groupmembench's own loader
    (eval_lib.py:load_conversation_messages) when the conversation carries
    the raw `.channels` shape; otherwise degrades to a plain content-only
    dict per turn."""
    if hasattr(conversation, "channels"):
        messages = []
        for channel, msgs in conversation.channels.items():
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                m2 = dict(m)
                m2["_channel"] = channel
                messages.append(m2)
        messages.sort(key=lambda m: (m.get("_channel", ""), m.get("timestamp", ""), m.get("msg_node", "")))
        return messages
    if hasattr(conversation, "sessions"):
        return [
            {"content": turn.text, "author": turn.speaker, "_channel": f"session {idx}"}
            for idx in sorted(conversation.sessions)
            for turn in conversation.sessions[idx]
        ]
    if hasattr(conversation, "turns"):
        return [{"content": turn.text, "author": turn.speaker} for turn in conversation.turns]
    raise ValueError(
        "groupmembench_bm25: conversation has neither `.channels` (groupmembench's own raw "
        "shape), `.sessions`, nor `.turns` -- can't extract retrievable messages for this "
        "dataset's shape"
    )


class Method(_MethodBase):
    def __init__(self, model: str = "claude-sonnet-5", top_k: int = 10, max_tokens: int = 512, **config):
        super().__init__(model=model, top_k=top_k, max_tokens=max_tokens, **config)
        self.model = model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._messages = None
        self._bm25 = None

    def prepare(self, conversation) -> None:
        self._messages = _extract_messages(conversation)
        corpus_tokens = [_tokenize(_message_index_text(m)) for m in self._messages]
        self._bm25 = BM25Okapi(corpus_tokens) if self._messages else None

    def answer(self, item) -> MethodOutput:
        asking_user_id = (item.metadata or {}).get("asking_user_id")

        retrieved_docs = []
        if self._messages and self._bm25 is not None:
            search_query = f"{asking_user_id} {item.question}" if asking_user_id else item.question
            scores = self._bm25.get_scores(_tokenize(search_query))
            k = min(self.top_k, len(self._messages))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            retrieved_docs = [_format_retrieved_message(self._messages[i]) for i in top_indices]

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        asker_line = f"Asking user: {asking_user_id}\n\n" if asking_user_id else ""
        passages_text = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved_docs, start=1))
        prompt = (
            f"{asker_line}"
            f"Question:\n{question}\n\n"
            f"Retrieved passages:\n{passages_text}\n\n"
            "Answer the question using the retrieved passages."
        )
        resp = complete(prompt, model=self.model, system=_AGENT_SYSTEM_PROMPT, max_tokens=self.max_tokens)
        return MethodOutput(
            answer=_extract_final(resp.text),
            retrieved=retrieved_docs,
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )

    def reset(self) -> None:
        self._messages = None
        self._bm25 = None
