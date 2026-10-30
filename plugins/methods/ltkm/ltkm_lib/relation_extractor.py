"""Relation-extraction model (paper Sec. "Persona-aware Question
Construction Module"): a RoBERTa-large classifier fine-tuned on DNLI's
persona-sentence -> relation-triple annotations (paper reports 82.3%
micro-F1 on "41 filtered relations"). Used to map a persona sentence to a
DNLI relation label at stage-1 training/inference time; the two other
triple elements (e1, e2) are *not* predicted by this model -- e1 comes from
`question_construction.extract_subject_phrase`, e2 is unused (the question
only needs e1 + relation, per the paper's own template).

This repo doesn't ship a trained checkpoint (see method.py's module
docstring) -- `RelationExtractor(model_path=...)` needs one produced by
`train_relation_extractor()` below (or the `train_relation_extractor.py`
CLI script) run against real DNLI triple data.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .dnli import PersonaTriple

_DEFAULT_BASE_MODEL = "roberta-large"


@dataclass
class RelationExtractorConfig:
    base_model: str = _DEFAULT_BASE_MODEL
    max_length: int = 64


class _TripleDataset(Dataset):
    def __init__(self, triples: list[PersonaTriple], tokenizer, label_to_id: dict[str, int], max_length: int):
        self.triples = triples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        triple = self.triples[idx]
        enc = self.tokenizer(
            triple.persona,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(self.label_to_id[triple.relation], dtype=torch.long),
        }


class RelationExtractor:
    """Inference wrapper around a fine-tuned relation-classification model."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.id_to_label: dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, persona_sentence: str) -> str:
        enc = self.tokenizer(persona_sentence, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits
        predicted_id = int(logits.argmax(dim=-1).item())
        return self.id_to_label[predicted_id]


def train_relation_extractor(
    triples: list[PersonaTriple],
    output_dir: str,
    base_model: str = _DEFAULT_BASE_MODEL,
    max_length: int = 64,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-5,
    device: str | None = None,
) -> None:
    """Fine-tune a RoBERTa-large sequence classifier on persona-sentence ->
    relation-label pairs, saving the result to `output_dir` for
    `RelationExtractor(output_dir)` to load."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    relations = sorted({t.relation for t in triples})
    label_to_id = {r: i for i, r in enumerate(relations)}
    id_to_label = {i: r for r, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(relations),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    ).to(device)

    dataset = _TripleDataset(triples, tokenizer, label_to_id, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
