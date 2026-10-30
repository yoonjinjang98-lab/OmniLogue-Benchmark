"""DNLI entailment scorer (paper Eq. 2, "Scoring" under "Negative Samples
Generation Module"): a RoBERTa-large 3-way NLI classifier (entailment/
neutral/contradiction) fine-tuned on DNLI sentence pairs. Two things use it:

1. `key_persona_scores()` -- for a golden consistent response Y and its
   persona sentence P_j, scores each word y_i in Y by how much removing it
   drops the entailment probability of (P_j -> Y). High-scoring words are
   the "key persona words" stage-2 training perturbs to build negatives.
2. `average_consistency_score()` -- the paper's evaluation metric "C"
   (Madotto et al. 2019 convention, Table 2-5's `C` column) and also what
   decides `gamma` in the stage-2 total loss (Eq. 6: gamma=1 if Y is
   persona-consistent, else 0) -- both need the same entailment judgment,
   just read off differently (argmax label vs. raw entailment probability).

This repo doesn't ship a trained checkpoint (see method.py's module
docstring) -- needs one produced by `train_entailment_scorer()` below run
against real DNLI sentence-pair data.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .dnli import ID_TO_LABEL, LABEL_TO_ID, DNLIPair

_DEFAULT_BASE_MODEL = "roberta-large"
_ENTAILMENT_LABEL_ID = LABEL_TO_ID["positive"]


class _PairDataset(Dataset):
    def __init__(self, pairs: list[DNLIPair], tokenizer, max_length: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        enc = self.tokenizer(
            pair.sentence1,
            pair.sentence2,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(LABEL_TO_ID[pair.label], dtype=torch.long),
        }


class EntailmentScorer:
    def __init__(self, model_path: str, device: str | None = None, max_length: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """p(entailment | premise, hypothesis) -- Eq. 2's p(E|[P_j; Y]; phi)."""
        enc = self.tokenizer(
            premise, hypothesis, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]
        return float(probs[_ENTAILMENT_LABEL_ID].item())

    @torch.no_grad()
    def predict_label(self, premise: str, hypothesis: str) -> str:
        enc = self.tokenizer(
            premise, hypothesis, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        predicted_id = int(self.model(**enc).logits.argmax(dim=-1).item())
        return ID_TO_LABEL[predicted_id]

    def key_persona_scores(self, response: str, persona_sentence: str) -> list[tuple[str, float]]:
        """Eq. 2: C_k(y_i) = p(E|[P_j;Y]) - p(E|[P_j;Y_{/i}]) for each word
        y_i in the response Y. Returns (word, score) pairs in response
        order; higher score means removing that word hurts persona
        consistency more, i.e. it's a "key persona word"."""
        words = response.split()
        full_score = self.entail_prob(persona_sentence, response)

        scores = []
        for i in range(len(words)):
            without_i = " ".join(words[:i] + words[i + 1 :])
            without_score = self.entail_prob(persona_sentence, without_i) if without_i else 0.0
            scores.append((words[i], full_score - without_score))
        return scores

    def top_k_key_persona_words(self, response: str, persona_sentence: str, k: int) -> list[int]:
        """Word indices (into `response.split()`) of the top-k key persona
        words by `key_persona_scores`, used by negative_sampler.py."""
        scores = self.key_persona_scores(response, persona_sentence)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i][1], reverse=True)
        return ranked[:k]

    def is_persona_consistent(self, response: str, persona_sentence: str) -> bool:
        """Stage-2 gamma gate (Eq. 6): True (gamma=1) iff the response
        entails the persona sentence, i.e. is persona-consistent rather
        than persona-irrelevant/contradictory."""
        return self.predict_label(persona_sentence, response) == "positive"

    def average_consistency_score(self, response: str, persona_sentences: list[str]) -> float:
        """The paper's evaluation metric C (Madotto et al. 2019): entail=+1,
        contradict=-1, neutral=0, averaged over all persona sentences,
        reported as a percentage in Tables 2-5."""
        if not persona_sentences:
            return 0.0
        values = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        scores = [values[self.predict_label(p, response)] for p in persona_sentences]
        return 100.0 * sum(scores) / len(scores)


def train_entailment_scorer(
    pairs: list[DNLIPair],
    output_dir: str,
    base_model: str = _DEFAULT_BASE_MODEL,
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-5,
    device: str | None = None,
) -> None:
    """Fine-tune a 3-way (entail/neutral/contradiction) RoBERTa-large
    classifier on DNLI sentence pairs, saving to `output_dir` for
    `EntailmentScorer(output_dir)` to load."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    dataset = _PairDataset(pairs, tokenizer, max_length)
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
