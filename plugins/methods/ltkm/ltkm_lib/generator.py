"""The base dialogue generator the coarse-to-fine framework trains on top
of. The paper applies its framework to four base models (Per-Seq2Seq,
Transformer, GPT2, DialoGPT) and reports DialoGPT+K+A as its strongest
config on ConvAI2/Persona-Chat -- this wrapper defaults to
`microsoft/DialoGPT-medium` accordingly, but accepts any causal-LM
`base_model` name (GPT2 works identically; a from-scratch RNN/Transformer
Seq2Seq like the paper's other two baselines is out of scope here since it
isn't a pretrained-and-fine-tuned HF model).

Input format matches the paper ("The input of the baseline models is the
concatenation of the persona sentences and the input query"): persona
sentences, then dialogue history turns, then the query, each on its own
line, with the response as the continuation to be generated/scored.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DEFAULT_BASE_MODEL = "microsoft/DialoGPT-medium"


def _build_prompt(persona: list[str], history: list[str], query: str) -> str:
    lines = list(persona) + list(history) + [query]
    return "\n".join(lines)


class PersonaDialogueGenerator:
    def __init__(self, model_or_path: str = _DEFAULT_BASE_MODEL, device: str | None = None, max_length: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_or_path).to(self.device)

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _encode_prompt_and_response(self, persona: list[str], history: list[str], query: str, response: str):
        prompt = _build_prompt(persona, history, query)
        prompt_ids = self.tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(
            response, add_special_tokens=False, truncation=True, max_length=self.max_length
        )["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        input_ids = input_ids[-self.max_length :]
        labels = [-100] * len(prompt_ids) + response_ids
        labels = labels[-self.max_length :]

        return torch.tensor([input_ids]), torch.tensor([labels])

    def training_example(self, persona: list[str], history: list[str], query: str, response: str) -> dict:
        """Encode one (persona, history, query, response) example for
        stage-1 MLE training (Eq. 1) -- `model(**inputs).loss` is already
        HF's mean per-token NLL over non-masked labels, which *is* Eq. 1."""
        input_ids, labels = self._encode_prompt_and_response(persona, history, query, response)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def sequence_log_prob(self, persona: list[str], history: list[str], query: str, response: str) -> torch.Tensor:
        """log p(Y | Q, P; theta), summed over response tokens (a scalar
        tensor with grad, since this is called on the *live* model being
        trained -- Eq. 4-5 explicitly do not use a frozen scorer). Divide
        by response length and pass through sigmoid for Eq. 5's D(Q,Y;theta)."""
        inputs = self.training_example(persona, history, query, response)
        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits[0, :-1, :]
        labels = inputs["labels"][0, 1:]

        mask = labels != -100
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * mask).sum()

    def response_length(self, response: str) -> int:
        return len(self.tokenizer(response, add_special_tokens=False)["input_ids"]) + 1  # +1 for eos

    @torch.no_grad()
    def generate(self, persona: list[str], history: list[str], query: str, max_new_tokens: int = 64) -> str:
        prompt = _build_prompt(persona, history, query) + "\n"
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(
            self.device
        )
        was_training = self.model.training
        self.model.eval()
        output_ids = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            length_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if was_training:
            self.model.train()
        generated = output_ids[0, enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
