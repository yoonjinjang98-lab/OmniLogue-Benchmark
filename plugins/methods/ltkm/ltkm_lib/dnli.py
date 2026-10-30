"""Loaders for the two DNLI (Dialogue NLI, Welleck et al. 2019) data shapes
this framework needs. DNLI itself (https://wellecks.com/dialogue_nli/) is
distributed as Google Drive JSON files, not a pip/HF-hub package -- there is
no way to verify the exact on-disk schema without downloading it (not done
as part of this scaffolding, see method.py's module docstring), so the
schemas below are written against the fields confirmed from a third-party
loader that consumes the same released files
(github.com/soroushjavdan/NLIbert, `sentence1`/`sentence2`/`label`/`id`)
plus the paper's own description of the "personal attribute triple"
annotation (Sec. "Persona-aware Question Construction Module": each persona
sentence is paired with a human-labeled (e1, r, e2) triple). If the actual
downloaded files use different field names, adjust `_NLI_FIELDS`/
`_TRIPLE_FIELDS` below rather than the parsing logic itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DNLIPair:
    """One (premise, hypothesis, label) NLI example. `label` is normalized
    to one of "positive" (entailment), "neutral", "negative" (contradiction)
    -- DNLI's own label strings, per the confirmed third-party loader."""

    id: str
    sentence1: str
    sentence2: str
    label: str


@dataclass
class PersonaTriple:
    """One persona sentence with its annotated (e1, relation, e2) triple,
    e.g. persona="I like playing basketball." -> (e1="I", relation=
    "like_sports", e2="basketball") (the paper's own worked example)."""

    persona: str
    e1: str
    relation: str
    e2: str


_NLI_FIELDS = {"id": "id", "sentence1": "sentence1", "sentence2": "sentence2", "label": "label"}
_TRIPLE_FIELDS = {"persona": "persona", "e1": "e1", "relation": "relation", "e2": "e2"}


def load_dnli_pairs(path: str | Path) -> list[DNLIPair]:
    """Load a DNLI sentence-pair file (train/dev/test split) -- a JSON list
    of records with (at minimum) `sentence1`/`sentence2`/`label`/`id` keys.
    Used to train `entailment_scorer.EntailmentScorer` (stage 2)."""
    records = json.loads(Path(path).read_text())
    pairs = []
    for r in records:
        pairs.append(
            DNLIPair(
                id=str(r[_NLI_FIELDS["id"]]),
                sentence1=r[_NLI_FIELDS["sentence1"]],
                sentence2=r[_NLI_FIELDS["sentence2"]],
                label=r[_NLI_FIELDS["label"]],
            )
        )
    return pairs


def load_persona_triples(path: str | Path) -> list[PersonaTriple]:
    """Load persona-sentence -> (e1, relation, e2) triple annotations. Used
    to train `relation_extractor.RelationExtractor` (stage 1): the model
    learns persona sentence -> relation label; e1 is recovered at inference
    time by `question_construction._extract_subject_entity`, not predicted
    by this model (matching the paper: "the entity e1 will be extracted by
    keyword matching to form the query")."""
    records = json.loads(Path(path).read_text())
    triples = []
    for r in records:
        triples.append(
            PersonaTriple(
                persona=r[_TRIPLE_FIELDS["persona"]],
                e1=r[_TRIPLE_FIELDS["e1"]],
                relation=r[_TRIPLE_FIELDS["relation"]],
                e2=r[_TRIPLE_FIELDS["e2"]],
            )
        )
    return triples


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
