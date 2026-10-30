"""qa_llm_judge -- LLM-as-judge correctness for free-form QA.

score(item, prediction) -> 1.0 if a cheap judge model says `prediction` is
semantically equivalent to `item.reference`, else 0.0. Item-aware because the
judge prompt needs item.question -- token-overlap metrics (qa_em/qa_f1) alone
underscore correct free-form answers that are phrased differently from the
gold reference (see BASELINE_METRIC_PLAN.md §2 for locomo/dialsim/
groupmembench/evermembench).
"""
from dialogue_benchmarks import llm

# Local vLLM gateway, free and no API key needed (this worktree has none) --
# and unlike openai/gpt-oss-120b, its reasoning can be cleanly turned off
# (enable_thinking=False, the default), so the response is just the bare
# verdict with no stray reasoning trace to strip out of a 5-token budget.
_JUDGE_MODEL = "google/gemma-4-31B-it"

_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Is the predicted answer correct? Judge by meaning, not exact wording -- a
prediction that's phrased differently from the reference but conveys the same
answer is CORRECT. Respond with exactly one word: CORRECT or INCORRECT."""


def judge(item, prediction, model: str, **params) -> float:
    """The judge itself, model-agnostic -- qa_llm_judge_gpt reuses it with an
    OpenAI model so both metrics grade with the exact same prompt."""
    prompt = _PROMPT.format(question=item.question, reference=item.reference, prediction=prediction)
    resp = llm.complete(prompt, model=model, temperature=0.0, max_tokens=5, **params)
    verdict = (resp.text or "").strip().upper()
    return 1.0 if verdict.startswith("CORRECT") else 0.0


def score(item, prediction) -> float:
    return judge(item, prediction, _JUDGE_MODEL, chat_template_kwargs={"enable_thinking": False})
