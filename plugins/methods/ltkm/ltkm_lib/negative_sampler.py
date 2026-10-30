"""Negative Samples Generation Module (paper Fig. 3, "Revising"). Given a
golden consistent response and its persona sentence, finds the top-k "key
persona words" (via `entailment_scorer.EntailmentScorer.
top_k_key_persona_words`, Eq. 2) and regenerates each with a *pretrained,
not fine-tuned* masked language model -- if the MLM's top prediction is the
original word, its second-best prediction is used instead (guaranteeing an
actual change, per the paper: "If the MLM predicts the original word, the
second most likely word is used instead"). Varying which/how-many words are
masked (k=1..K) yields up to K negative responses per positive, used by
`training.train_stage2_fine`'s contrastive loss.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .entailment_scorer import EntailmentScorer

_DEFAULT_MLM = "bert-base-uncased"


class NegativeSampleGenerator:
    def __init__(self, mlm_name: str = _DEFAULT_MLM, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(mlm_name)
        self.model = AutoModelForMaskedLM.from_pretrained(mlm_name).to(self.device).eval()

    @torch.no_grad()
    def _regenerate_word(self, words: list[str], position: int) -> str:
        original = words[position]
        masked_words = list(words)
        masked_words[position] = self.tokenizer.mask_token
        masked_text = " ".join(masked_words)

        enc = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        mask_positions = (enc["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            # the original word tokenized into >1 piece and the mask token
            # collapsed differently than expected -- fail soft, keep original.
            return original
        mask_index = int(mask_positions[0].item())

        logits = self.model(**enc).logits[0, mask_index]
        top2 = torch.topk(logits, k=2).indices.tolist()
        candidates = [self.tokenizer.decode([tid]).strip() for tid in top2]

        for candidate in candidates:
            if candidate.lower() != original.lower() and candidate:
                return candidate
        return candidates[0]

    def generate_negatives(
        self,
        response: str,
        persona_sentence: str,
        entailment_scorer: EntailmentScorer,
        num_negatives: int = 3,
    ) -> list[str]:
        """Returns up to `num_negatives` negative responses, each with an
        increasing number (1..num_negatives) of key persona words replaced
        -- matching the paper's "Varying k from 1 to K, we can generate K
        negative response {Y_i^-}_{i=1}^K"."""
        words = response.split()
        if not words:
            return []

        key_indices = entailment_scorer.top_k_key_persona_words(response, persona_sentence, num_negatives)

        negatives = []
        current_words = list(words)
        for position in key_indices:
            current_words[position] = self._regenerate_word(current_words, position)
            negatives.append(" ".join(current_words))
        return negatives
