"""Persona-aware Question Construction Module (paper Sec. "Learning to Know
Myself"). Builds the persona-dense (Q_{P_i}, P_i) pairs used to augment
stage-1 (coarse) training: for each persona sentence P_i, predict its DNLI
relation with `relation_extractor.RelationExtractor`, extract the subject
entity e1 by dependency parsing + pronoun-flip (the paper's "keyword
matching", made a bit more robust here via spaCy since plain keyword
matching has no real definition to follow), and render a question via
`templates.build_question`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .relation_extractor import RelationExtractor
from .templates import build_question

_PRONOUN_FLIP = {
    "i": "you",
    "me": "you",
    "my": "your",
    "mine": "yours",
    "myself": "yourself",
    "we": "you",
    "us": "you",
    "our": "your",
    "ours": "yours",
    "ourselves": "yourselves",
}


@dataclass
class PersonaAwareQuestionPair:
    persona: str
    question: str
    relation: str
    subject_phrase: str


def _flip_pronouns(tokens: list[str]) -> list[str]:
    return [_PRONOUN_FLIP.get(tok.lower(), tok) for tok in tokens]


def extract_subject_phrase(persona_sentence: str, nlp) -> str:
    """Extract the sentence's subject (nsubj/nsubjpass), including its
    determiner/possessive children (e.g. "my parents", not just "parents"),
    then flip first-person pronouns to second-person so the phrase reads
    naturally as a question directed *at* the persona-bearing agent (the
    paper's own examples: "I like playing basketball." -> "you"; "My
    parents live in Australia." -> "your parents"). Falls back to "you"
    (the overwhelmingly common case in Persona-Chat-style personas, which
    are almost all first-person "I ..." sentences) if no subject is found.

    `nlp` is a loaded spaCy pipeline (e.g. `spacy.load("en_core_web_sm")`),
    passed in rather than loaded here so callers construct it once.
    """
    doc = nlp(persona_sentence)
    subject = next((tok for tok in doc if tok.dep_ in ("nsubj", "nsubjpass")), None)
    if subject is None:
        return "you"

    subtree = sorted(subject.subtree, key=lambda t: t.i)
    words = _flip_pronouns([t.text for t in subtree])
    return " ".join(words) if words else "you"


class PersonaAwareQuestionConstructor:
    def __init__(self, relation_extractor: RelationExtractor, nlp=None):
        self.relation_extractor = relation_extractor
        if nlp is None:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp

    def construct(self, persona_sentence: str) -> PersonaAwareQuestionPair:
        relation = self.relation_extractor.predict(persona_sentence)
        subject_phrase = extract_subject_phrase(persona_sentence, self.nlp)
        question = build_question(subject_phrase, relation)
        return PersonaAwareQuestionPair(
            persona=persona_sentence,
            question=question,
            relation=relation,
            subject_phrase=subject_phrase,
        )

    def construct_all(self, persona_sentences: list[str]) -> list[PersonaAwareQuestionPair]:
        return [self.construct(p) for p in persona_sentences]
