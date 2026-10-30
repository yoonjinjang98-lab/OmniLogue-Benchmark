"""smoke_test -- trivial baseline for verifying the evaluate() pipeline end-to-end.

Picks the turn with the most word overlap with the question and returns its text
verbatim. Has no real memory mechanism; do not use it as an actual comparison
point against real methods (A-mem, MemoryOS, etc).
"""


def answer(conversation, question: str) -> str:
    all_turns = [turn for session in conversation.sessions.values() for turn in session]
    if not all_turns:
        return ""
    question_words = set(question.lower().split())
    best = max(all_turns, key=lambda t: len(set(t.text.lower().split()) & question_words))
    return best.text
