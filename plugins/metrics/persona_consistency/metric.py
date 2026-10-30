"""persona_consistency -- LLM-NLI C-score against the speaker's persona.

score(item, prediction) -> +1 (entailed by some persona sentence), 0 (neutral),
-1 (contradicts one). The original C-score runs a DNLI-trained classifier per
(persona sentence, response) pair; this asks the same question of the local
judge model once per item over the whole persona, which keeps the cost to one
call per item. Items with no persona score 0.
"""
from dialogue_benchmarks import llm

# same local, free judge model as qa_llm_judge
_JUDGE_MODEL = "google/gemma-4-31B-it"

_PROMPT = """You are checking a dialogue response for consistency with the speaker's persona.

Persona of the speaker:
{persona}

Response the speaker produced:
{response}

Does the response CONTRADICT any persona statement, is it ENTAILED by (consistent with and expressing) one of them, or is it NEUTRAL (unrelated to the persona)?
Respond with exactly one word: ENTAIL, NEUTRAL, or CONTRADICT."""


def judge(item, prediction, model: str, **params) -> float:
    """Model-agnostic C-score judgment -- persona_consistency_gpt reuses it."""
    persona = (item.metadata or {}).get("persona") or []
    if isinstance(persona, str):
        persona = [persona]
    if not persona or not str(prediction).strip():
        return 0.0
    prompt = _PROMPT.format(persona="\n".join(f"- {p}" for p in persona), response=prediction)
    resp = llm.complete(prompt, model=model, temperature=0.0, max_tokens=5, **params)
    verdict = (resp.text or "").strip().upper()
    if verdict.startswith("CONTRADICT"):
        return -1.0
    if verdict.startswith("ENTAIL"):
        return 1.0
    return 0.0


def score(item, prediction) -> float:
    return judge(item, prediction, _JUDGE_MODEL, chat_template_kwargs={"enable_thinking": False})
