"""SBYS -- Score Before You Speak (Saggar, Darling, Dimitrova, Sarikaya,
Hogg; ECAI 2025, "Score Before You Speak: Improving Persona Consistency in
Dialogue Generation using Response Quality Scores", arXiv:2508.06886,
https://github.com/arpita2512/score_before_you_speak). Family E (trained
persona-conditioned generation), same family as plugins/methods/paa/ltkm/
orig -- NOT a prompted-LLM baseline, does not go through
dialogue_benchmarks.llm.complete().

**The method**: training data is (persona, context, response,
response-quality-score) quadruples, not just (persona, context, response)
triples. Upstream builds the extra examples by masking content words out of
each golden response, refilling the mask with a BART-based infiller
(`preprocess/masking.py`), then scoring how persona-consistent each
refilled variant is against the persona (`preprocess/scoring.py`) -- so one
golden response yields several (response variant, score) pairs spanning
the consistency range, not just a single always-perfect example. The score
is serialized as literal text appended right after the *last* user turn
before generation begins: `"...Score: {score}<|sp2|>Bot: {response}"`
(`preprocess/convert_to_hf_dataset.py`'s `concatenate_and_tokenize` is the
literal ground truth for the prompt format used below, not the README's
example alone). At training time the model sees the full range of scores
paired with correspondingly (in)consistent responses; at generation time,
asking for the maximum score ("Score: 1.0") is what conditions the model
toward its most persona-consistent completion -- this repo always requests
1.0 (`response_score` config, default `"1.0"`), matching the paper's own
inference-time usage.

**Runnable out of the box, unlike `paa`/`ltkm`/`orig`**: upstream released
real fine-tuned checkpoints on the HuggingFace Hub (see the collection
linked from its README) -- `checkpoint_path` defaults to
`Arpita1/sbs_personachat_dialogpt` (DialoGPT-small fine-tuned on
PERSONA-CHAT; `Arpita1/sbs_convai2_dialogpt` is the ConvAI2 counterpart,
same architecture, just pass its repo id instead). `AutoModelForCausalLM.
from_pretrained()`/`AutoTokenizer.from_pretrained()` accept a HF Hub repo
id exactly like a local directory, so no vendoring or local training step
is needed for this baseline at all -- the first Family E baseline in this
repo that's actually runnable rather than blocked on "needs a trained
checkpoint nobody has produced yet". The two `sbs_*_llama3.1_lora`
checkpoints in the same collection (LoRA adapters over gated Llama 3.1,
needing `HF_TOKEN` + accepting Meta's license) are *not* wired up here --
deliberately out of scope, since they add real friction (gated-model
access) this repo's other baselines don't need, and the DialoGPT
checkpoints already cover this baseline's actual comparison point.

**No vendoring, no upstream inference code to wrap**: the source repo only
ships `preprocess/` (data construction) and `train/` (training loops) --
no `decoding.py`/generation script exists upstream at all (unlike `orig`,
which had one to reuse). The prompt-construction logic below
(`_build_prompt_ids`) is an original implementation matching upstream's
own `preprocess/convert_to_hf_dataset.py::concatenate_and_tokenize`
verbatim: `<|startoftext|>` + tokenize("Your persona: p1 p2 ...") + per
history turn (`<|sp1|>` before a user turn, `<|sp2|>` before a bot turn,
each turn's text itself already prefixed "User: "/"Bot: ") + tokenize
("Score: {score}") + `<|sp2|>`, then the model generates the continuation
(which upstream's own training data has starting with a literal "Bot: "
token sequence -- confirmed empirically, see below -- so this Method
strips a leading "Bot:" from the decoded generation).

**Licensing note**: arpita2512/score_before_you_speak has no LICENSE file
and GitHub's own detector reports none. Same situation as
plugins/methods/pathrag and plugins/methods/orig, reconfirmed with the
user this session on the same basis: internal reproduction only, not
redistribution.

Verification: this Method was run against the real
`Arpita1/sbs_personachat_dialogpt` checkpoint from the HuggingFace Hub (a
genuine, already-trained model, not a mock or tiny random-weight stand-in
-- the strongest verification of any Family E baseline here) with a
hand-written persona/history/query example, confirming the full pipeline
(prompt construction, generation, "Bot:" stripping) produces a real,
persona-grounded response.
"""
import time

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

_DEFAULT_CHECKPOINT = "Arpita1/sbs_personachat_dialogpt"


def _build_prompt_ids(tokenizer, persona_sentences, turns, score) -> list:
    """turns: alternating speaker-prefixed strings ("User: ...", "Bot: ...",
    "User: ...", ...), always ending on the current "User: {query}" turn --
    matches upstream's `preprocess/convert_to_hf_dataset.py::
    concatenate_and_tokenize` field-by-field (see module docstring)."""
    sp1_id, sp2_id = tokenizer.additional_special_tokens_ids

    persona_text = "Your persona:" + "".join(f" {p}" for p in persona_sentences)
    ids = [tokenizer.bos_token_id]
    ids += tokenizer(persona_text, add_special_tokens=False)["input_ids"]

    for i, turn_text in enumerate(turns):
        ids.append(sp1_id if i % 2 == 0 else sp2_id)
        ids += tokenizer(turn_text, add_special_tokens=False)["input_ids"]

    ids += tokenizer(f"Score: {score}", add_special_tokens=False)["input_ids"]
    ids.append(sp2_id)
    return ids


def _strip_bot_prefix(text: str) -> str:
    text = text.strip()
    if text.lower().startswith("bot:"):
        text = text[len("bot:") :].strip()
    return text


class Method(_MethodBase):
    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        response_score: str = "1.0",
        device: str = None,
        max_new_tokens: int = 40,
        num_beams: int = 5,
        **config,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            response_score=response_score,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **config,
        )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.response_score = response_score
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self._model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(self.device).eval()

        self._conversation = None

    def prepare(self, conversation) -> None:
        """No real per-conversation ingestion cost -- the model is already loaded.
        Just remembers the conversation so answer() can fall back to it for
        persona/history when item.metadata doesn't carry them directly."""
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        persona, history = self._extract_persona_and_history(item)
        turns = [f"{'User' if i % 2 == 0 else 'Bot'}: {h}" for i, h in enumerate(history)]
        turns.append(f"User: {item.question}")

        input_ids = _build_prompt_ids(self._tokenizer, persona, turns, self.response_score)
        input_ids = self._torch.tensor([input_ids]).to(self.device)
        attention_mask = self._torch.ones_like(input_ids)

        start = time.time()
        with self._torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, input_ids.shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return MethodOutput(answer=_strip_bot_prefix(text), latency=time.time() - start)

    def reset(self) -> None:
        self._conversation = None

    def _extract_persona_and_history(self, item):
        """Same contract/fallback logic as plugins/methods/paa, ltkm, and
        orig -- kept in sync deliberately, since all four are Family E
        persona-conditioned generators targeting the same msc/personachat
        conversation shapes."""
        meta = item.metadata or {}
        persona = meta.get("persona")
        history = meta.get("history")
        if persona is not None and history is not None:
            return list(persona), list(history)

        conv = self._conversation
        if conv is not None and hasattr(conv, "personality") and hasattr(conv, "turns"):
            turn_index = meta.get("turn_index", len(conv.turns) - 1 if conv.turns else None)
            turn = conv.turns[turn_index] if turn_index is not None else None
            return (
                list(persona) if persona is not None else list(conv.personality),
                list(history) if history is not None else (list(turn.history) if turn else []),
            )
        if conv is not None and hasattr(conv, "persona1") and hasattr(conv, "utterances"):
            up_to = meta.get("speaker_index", len(conv.utterances))
            speakers = conv.speakers or []
            next_speaker = speakers[up_to] if up_to < len(speakers) else (speakers[-1] if speakers else None)
            inferred_persona = conv.persona1 if next_speaker != "Speaker 2" else conv.persona2
            return (
                list(persona) if persona is not None else list(inferred_persona),
                list(history) if history is not None else list(conv.utterances[:up_to]),
            )
        raise ValueError(
            "sbys: could not determine persona/dialogue history for this item -- expected "
            "item.metadata['persona'] and ['history'], or a conversation shaped like "
            "personachat's PersonaChatDialogue or msc's MscSession"
        )
