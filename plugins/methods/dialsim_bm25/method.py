"""dialsim_bm25 -- family D baseline (BASELINE_METRIC_PLAN.md §4): wraps
DialSim's own BM25-RAG baseline (github.com/jiho283/DialSim, no LICENSE file
-- vendored here for internal benchmark-reproduction use only, same call
already made for groupmembench/evermembench) instead of reimplementing
generic bm25_rag against this dataset. Family D's point is reproducing
exactly what a dataset's own paper reports, which is only meaningful if the
tokenizer/document-granularity/prompt/answer-format match theirs.

Ported from `code/simulator.py`'s default CLI config (`--ret_method bm25`,
`--history_type session-entire`, `--answer_format multi_choice_structured`,
`--num_ret_history 10`) + `code/prompt/RAG_qa_prompt_multi_choice_structured.txt`:
- `_tokenize()`: their exact `nltk.word_tokenize` (lowercased) tokenizer
  feeding `BM25Okapi` (`rank_bm25` is already this repo's own core
  dependency)
- `_session_document()`: their `history_type="session-entire"` chunking --
  one whole scene/session per retrievable document (`save_history()`'s
  `f"[Date: {date}, Session #{n}]\n{history}"` format), not per-utterance
- the `RAG_qa_prompt_multi_choice_structured.txt` template verbatim,
  including the chatbot-persona framing (Ross/Sheldon/Michael per show) and
  the mandatory "(E)" don't-know option
- `_parse_choice()`: which of (A)-(E) the response names, mapped back to
  that option's actual answer text -- not their own `judge_eq()`
  correctness/ambiguity logic. This repo's Method/Metric contracts keep
  answering and scoring separate; score this baseline's answers with
  qa_em/qa_f1/qa_llm_judge instead (same choice already made for
  groupmembench_bm25/evermembench_llm).

One documented deviation: upstream's query and prompt use one of six
per-asker-character phrasings of each question (`char_ask: real_question`,
`char_ask` picked at random each run); `plugins/datasets/dialsim/adapter.py`
only keeps each question's neutral "default" phrasing (see its own
docstring), so this method's retrieval query and prompt use that instead of
a character-attributed one. Retrieval granularity, tokenizer, prompt
wording, and answer format are otherwise unchanged from upstream.
"""
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete

_CHATBOT_BY_SHOW = {"friends": "Ross", "bigbang": "Sheldon", "theoffice": "Michael"}

_OPTION_LETTERS = ("(A)", "(B)", "(C)", "(D)", "(E)")

# Verbatim from code/prompt/RAG_qa_prompt_multi_choice_structured.txt.
_RAG_PROMPT = (
    "You are {chatbot}, a long-term conversation agent capable of interacting with multiple users. \n"
    "Based on the [Retrieved Dialog History] provided, please answer the given [Question]. \n"
    "Note the following points: \n"
    "1. Your answer must exclusively be one of the options: (A), (B), (C), (D), (E). \n"
    "2. Your responses should solely rely on the retrieved dialog history. If the information in the "
    "dialog history is insufficient to answer the question, you must choose (E).\n"
    "3. This question is being asked in the context of {date}.\n\n"
    "[Retrieved Dialog History]\n{history}\n"
    "[Question] {question}\n"
    "[Answer]"
)


def _tokenize(text: str) -> list:
    """Verbatim from simulator.py: nltk word_tokenize on lowercased text."""
    return word_tokenize((text or "").lower())


def _session_document(session_idx, turns, date) -> str:
    """Verbatim shape from simulator.py:save_history's history_type="session-entire"."""
    script = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
    return f"[Date: {date}, Session #{session_idx}]\n{script}"


def _format_options(options: list) -> str:
    return "\n".join(f"\t{letter} {opt}" for letter, opt in zip(_OPTION_LETTERS, options))


def _parse_choice(response: str, options: list) -> str:
    for letter, opt in zip(_OPTION_LETTERS, options):
        if letter in (response or ""):
            return opt
    return (response or "").strip()


class Method(_MethodBase):
    def __init__(self, model: str = "claude-sonnet-5", top_k: int = 10, max_tokens: int = 256, **config):
        super().__init__(model=model, top_k=top_k, max_tokens=max_tokens, **config)
        self.model = model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._docs = None
        self._bm25 = None
        self._chatbot = None
        self._session_dates = None

    def prepare(self, conversation) -> None:
        self._chatbot = _CHATBOT_BY_SHOW.get(getattr(conversation, "show", None), "Ross")
        self._session_dates = getattr(conversation, "session_datetimes", {}) or {}
        session_ids = sorted(conversation.sessions)
        self._docs = [
            _session_document(idx, conversation.sessions[idx], self._session_dates.get(idx, "")) for idx in session_ids
        ]
        corpus_tokens = [_tokenize(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(corpus_tokens) if self._docs else None

    def answer(self, item) -> MethodOutput:
        options = item.options or []
        evidence = (item.metadata or {}).get("evidence") or []
        session_idx = evidence[0] if evidence else None
        date = self._session_dates.get(session_idx, "") if session_idx is not None else ""

        retrieved = []
        if self._docs and self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(item.question))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
            retrieved = [self._docs[i] for i in top_indices]

        question_block = item.question
        if options:
            question_block += "\n" + _format_options(options)

        prompt = _RAG_PROMPT.format(
            chatbot=self._chatbot, date=date, history="\n".join(retrieved), question=question_block
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        answer = _parse_choice(resp.text, options) if options else (resp.text or "").strip()
        return MethodOutput(
            answer=answer,
            retrieved=retrieved,
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )

    def reset(self) -> None:
        self._docs = None
        self._bm25 = None
        self._chatbot = None
        self._session_dates = None
