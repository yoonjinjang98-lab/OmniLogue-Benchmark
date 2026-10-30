"""Stage-1 (coarse, "Learning to Know Myself") and stage-2 (fine, "Learning
to Avoid Misidentification") training loops (paper Eq. 1 and Eq. 4-6).
Both stages fine-tune the *same* `generator.PersonaDialogueGenerator` in
place -- stage 2 is meant to run after stage 1 (the paper: "the dialogue
agent is trained in our coarse-grained persona-aware training stage" before
stage 2 begins, and "we do not utilize the target model during training"
for the *scoring* functions in stage 2, but the generator whose log-prob
Eq. 5 evaluates is always the one currently being trained).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .entailment_scorer import EntailmentScorer
from .generator import PersonaDialogueGenerator
from .negative_sampler import NegativeSampleGenerator
from .question_construction import PersonaAwareQuestionConstructor


@dataclass
class DialogueExample:
    """One (persona, history, query, response) training example -- a
    single query-response pair plus the persona it should be consistent
    with, matching the paper's task definition (P, Q) -> Y."""

    persona: list[str]
    history: list[str]
    query: str
    response: str


@dataclass
class Stage2Config:
    num_negatives: int = 3  # paper's K
    max_grad_norm: float = 1.0


def load_dialogue_examples(path: str | Path) -> list[DialogueExample]:
    """Load training examples from a JSON list of
    `{"persona": [...], "history": [...], "query": ..., "response": ...}`
    records -- this repo's own format (the paper trains on ConvAI2/
    Persona-Chat/DuLeMon directly; converting one of those into this shape
    is a preprocessing step this scaffolding doesn't include, see
    method.py's module docstring)."""
    records = json.loads(Path(path).read_text())
    return [
        DialogueExample(
            persona=list(r["persona"]),
            history=list(r.get("history", [])),
            query=r["query"],
            response=r["response"],
        )
        for r in records
    ]


def build_stage1_examples(
    examples: list[DialogueExample],
    question_constructor: PersonaAwareQuestionConstructor,
) -> list[DialogueExample]:
    """Eq. 1's training set augmentation: the original (persona, history,
    query, response) pairs, plus one new persona-dense (Q_{P_i}, P_i) pair
    per persona sentence per example (paper: "the constructed persona-dense
    query-response pairs... will be used together with original
    query-response pairs in the training corpus to augment it")."""
    augmented = list(examples)
    seen_personas: set[tuple[str, ...]] = set()

    for ex in examples:
        key = tuple(ex.persona)
        if key in seen_personas:
            continue
        seen_personas.add(key)

        for pair in question_constructor.construct_all(ex.persona):
            augmented.append(
                DialogueExample(persona=ex.persona, history=[], query=pair.question, response=pair.persona)
            )
    return augmented


def train_stage1_coarse(
    generator: PersonaDialogueGenerator,
    examples: list[DialogueExample],
    epochs: int = 1,
    lr: float = 5e-5,
) -> None:
    """MLE fine-tuning (Eq. 1) over the augmented example set from
    `build_stage1_examples`. Plain per-example SGD, no batching -- this
    repo's other family-C baselines batch API calls for throughput, but
    there's no equivalent concern training a single local model, and
    variable-length label-masked sequences make naive batching need padding
    logic the paper doesn't specify either."""
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=lr)
    generator.model.train()

    for _epoch in range(epochs):
        for ex in examples:
            inputs = generator.training_example(ex.persona, ex.history, ex.query, ex.response)
            optimizer.zero_grad()
            loss = generator.model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 1.0)
            optimizer.step()


def train_stage2_fine(
    generator: PersonaDialogueGenerator,
    examples: list[DialogueExample],
    entailment_scorer: EntailmentScorer,
    negative_sampler: NegativeSampleGenerator,
    config: Stage2Config = field(default_factory=Stage2Config),
    epochs: int = 1,
    lr: float = 5e-6,
) -> None:
    """Contrastive fine-tuning (Eq. 4-6). For each example: build up to K
    negative responses (`negative_sampler`), compute the length-normalized
    sigmoid score D(Q,Y;theta) (Eq. 5) for the positive and each negative
    against the *live* model, combine into the contrastive loss (Eq. 4),
    and add the plain MLE loss (Eq. 1) gated by gamma -- 1 if the response
    is persona-consistent, 0 otherwise (Eq. 6)."""
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=lr)
    generator.model.train()

    for _epoch in range(epochs):
        for ex in examples:
            if not ex.persona:
                continue
            primary_persona = ex.persona[0]

            negatives = negative_sampler.generate_negatives(
                ex.response, primary_persona, entailment_scorer, num_negatives=config.num_negatives
            )
            if not negatives:
                continue

            optimizer.zero_grad()

            pos_log_prob = generator.sequence_log_prob(ex.persona, ex.history, ex.query, ex.response)
            pos_len = generator.response_length(ex.response)
            d_pos = torch.sigmoid(pos_log_prob / pos_len)
            contrastive_loss = -torch.log(d_pos.clamp_min(1e-8))

            for negative in negatives:
                neg_log_prob = generator.sequence_log_prob(ex.persona, ex.history, ex.query, negative)
                neg_len = generator.response_length(negative)
                d_neg = torch.sigmoid(neg_log_prob / neg_len)
                contrastive_loss = contrastive_loss - torch.log((1 - d_neg).clamp_min(1e-8)) / len(negatives)

            gamma = 1.0 if entailment_scorer.is_persona_consistent(ex.response, primary_persona) else 0.0
            mle_inputs = generator.training_example(ex.persona, ex.history, ex.query, ex.response)
            mle_loss = generator.model(**mle_inputs).loss

            total_loss = mle_loss + gamma * contrastive_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), config.max_grad_norm)
            optimizer.step()
