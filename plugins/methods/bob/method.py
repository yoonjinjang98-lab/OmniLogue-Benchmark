"""BoB -- BERT Over BERT for Training Persona-based Dialogue Models from
Limited Personalized Data (Song, Wang, Zhang, Zhang, Liu; ACL 2021,
https://aclanthology.org/2021.acl-long.14/, https://github.com/songhaoyu/BoB).
Family E (trained persona-conditioned generation), same family as
plugins/methods/paa/ltkm/orig/sbys -- NOT a prompted-LLM baseline, does not
go through dialogue_benchmarks.llm.complete().

**The architecture**: one BERT encoder feeds TWO BERT decoders sharing its
output. `decoder1` is an ordinary autoregressive response decoder (persona+
query in, response out) trained with standard MLE. `decoder2` -- BoB's
actual contribution -- additionally cross-attends to *both* decoder1's own
hidden states *and* the raw persona token sequence directly (fed in
separately as `per_input_ids`, not through the shared encoder), and is
additionally trained with an unlikelihood objective on MNLI/DNLI-style
non-dialogue and dialogue NLI pairs so it learns to prefer
persona-consistent continuations over contradictory ones even from limited
persona-dense dialogue data. At inference, decoder2's output (not
decoder1's) is the model's actual answer -- this Method always requests it
(`model.generate(..., use_decoder2=True, ...)`).

**Licensing**: unlike plugins/methods/pathrag/orig/sbys, this repo ships a
real Apache-2.0 LICENSE file -- no licensing gap to flag or decide on here.

**Vendored**: `vendor/xlibs/` -- upstream's own renamed, modified snapshot
of `transformers==3.5.1` (already renamed to avoid colliding with this
venv's real `transformers` package, so vendoring it wholesale carries none
of the version-pinning risk plugins/methods/paa's actual `transformers<4.31`
pin does). The two-decoder architecture is a genuine, non-trivial patch to
`modeling_encoder_decoder.py`'s `EncoderDecoderModel` (a `decoder2`
constructor arg, `from_encoder_decoder_pretrained(encoder, decoder,
decoder2)` taking three model paths, and `generate(..., use_decoder2=,
per_input_ids=)`) -- there's no way to reproduce this by pointing at a
modern `transformers.EncoderDecoderModel` instead, so the whole (large,
but self-contained under its own `xlibs` namespace) library had to be
vendored, same precedent as plugins/methods/hipporag's vendoring of a
whole library for an equally load-bearing modification.

**Real bugs found and fixed at vendor time** (all confirmed by actually
running the two-decoder generate() path against a tiny BERT checkpoint,
not just read from source -- see verification note below):
- `xlibs/__init__.py` eagerly imports HF's old `pipelines`/`retrieval_rag`/
  `Trainer` modules, none of which `bertoverbert.py` (or this Method) ever
  uses. `pipelines` transitively imports `tokenization_auto` -> every known
  tokenizer type including SentencePiece-based ones -> `convert_slow_
  tokenizer.py`, which does `from transformers.utils import
  sentencepiece_model_pb2` -- reaching into the real, modern `transformers`
  package's pre-generated protobuf file, which is incompatible with the
  newer `protobuf` runtime in this shared venv (`TypeError: Descriptors
  cannot be created directly`). `Trainer` separately imports
  `torch.optim.lr_scheduler.SAVE_STATE_WARNING`, removed in modern torch.
  All three unused top-level imports were removed; the sentencepiece import
  was also moved from module level into the one function that actually
  needs it (`get_proto()`, only reachable from SentencePiece-based
  converters BoB never uses), rather than deleting it outright.
- `modeling_encoder_decoder.py`'s `forward()` hardcoded `.cuda()` on a
  tensor used by *every* forward pass (including plain inference), making
  the model unusable on a CPU-only machine. Changed to
  `.to(decoder2_input_ids.device)`.
- `generation_utils.py`'s `generate()` threads a `token_type_ids` argument
  through to `greedy_search()` correctly, but the `sample()`/
  `beam_search()`/`beam_sample()` call sites never pass it along --
  `EncoderDecoderModel.prepare_inputs_for_generation()` requires it as a
  positional argument with no default, so calling `generate(num_beams>1)`
  (i.e. *any* beam search -- upstream's own `predict()` always uses beam
  search) raised `TypeError: prepare_inputs_for_generation() missing 1
  required positional argument: 'token_type_ids'` unconditionally. Fixed
  by adding `token_type_ids=token_type_ids` to all three call sites,
  matching the pattern the already-correct `greedy_search()` call uses.
- Confirmed via the same investigation: `EncoderDecoderModel.forward()`
  (`return_dict=True`) always returns a 3-tuple (decoder1 output, decoder2
  output, an unlikelihood-training-only third output), but `sample()`/
  `beam_search()`/`beam_sample()` all unpacked only 2 values
  (`outputs_1, outputs_2 = self(...)`), raising `ValueError: too many
  values to unpack (expected 2)` unconditionally once the previous bug was
  fixed. `greedy_search()` (elsewhere in the same file) already correctly
  unpacks 3 values -- fixed the other three call sites to match
  (`outputs_1, outputs_2, _ = self(...)`). Together, these two bugs mean
  **upstream's own beam-search decoding path (the one `predict()` in the
  README actually demonstrates) cannot have been run since before whatever
  change last touched `generation_utils.py` or `modeling_encoder_decoder.py`**
  -- both failures are unconditional (not edge cases), so any beam-search
  generation call would hit them immediately.

Requires a locally trained checkpoint (`checkpoint_path` config, a
directory loadable via `EncoderDecoderModel.from_pretrained()`, i.e. what
`bertoverbert.py --do_train` saves) -- upstream ships no pretrained BoB
checkpoint, only base BERT weights + a training script
(`bertoverbert.py --do_train`) to fine-tune your own on ConvAI2
PersonaChat or ECDT2019 PersonalDialog. Not run this session (same status
as `paa`/`ltkm`/`orig`: real code, no trained checkpoint, nobody has run
the training yet).

**Modeling note**: matching upstream's own `dataloader.py`/
`preprocess.py` exactly, the encoder input is persona + the single current
query only -- there is no separate multi-turn history channel the way
plugins/methods/paa/orig/sbys build one (BoB's own ConvAI2 preprocessing
takes only the final query line paired with its response, per
`read_convai2_split`). This Method follows that faithfully: `item.question`
is used as the query and any `history` this repo's `_extract_persona_and_
history` fallback returns is intentionally ignored, not a bug.

Verification: after applying the three fixes above, this Method's
generation path (encoder + decoder1 + decoder2, beam search, `use_decoder2
=True`) was run end-to-end against a real (though untrained,
randomly-initialized) tiny BERT checkpoint
(`hf-internal-testing/tiny-random-bert`, re-saved locally in the old
`pytorch_model.bin` format `xlibs` expects, since safetensors postdates
this vendored library) -- confirmed both decoder1's and decoder2's
generate() calls produce correctly-shaped output with no errors. The
generated text itself is gibberish, as expected from untrained random
weights -- this confirms the vendored code (including all three fixes) is
mechanically correct, not that BoB's own reported numbers reproduce here.
"""
import sys
import time
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _load_vendor():
    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))
    import xlibs

    return xlibs


class Method(_MethodBase):
    """Inference-only wrapper around a checkpoint produced by upstream's
    `bertoverbert.py --do_train`, and won't run at all until one has
    actually been trained."""

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        max_length: int = 32,
        min_length: int = 3,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        persona_max_length: int = 64,
        query_max_length: int = 32,
        **config,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            persona_max_length=persona_max_length,
            query_max_length=query_max_length,
            **config,
        )
        if not checkpoint_path:
            raise ValueError(
                "bob: 'checkpoint_path' is required -- point it at a directory produced by "
                "upstream's `bertoverbert.py --do_train` (see this method's module docstring; "
                "upstream ships no pretrained checkpoint of its own)."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"bob: checkpoint not found at {checkpoint_path}")

        xlibs = _load_vendor()

        import torch

        self._torch = torch
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.persona_max_length = persona_max_length
        self.query_max_length = query_max_length
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self._tokenizer = xlibs.BertTokenizer.from_pretrained(str(checkpoint_path))
        self._tokenizer.bos_token = self._tokenizer.cls_token
        self._tokenizer.eos_token = self._tokenizer.sep_token

        model = xlibs.EncoderDecoderModel.from_pretrained(str(checkpoint_path))
        model.config.decoder_start_token_id = self._tokenizer.bos_token_id
        model.config.eos_token_id = self._tokenizer.eos_token_id
        model.config.pad_token_id = self._tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.output_hidden_states = True
        model.to(self.device)
        model.eval()
        self._model = model

        self._conversation = None

    def prepare(self, conversation) -> None:
        """No real per-conversation ingestion cost -- the model is already loaded.
        Just remembers the conversation so answer() can fall back to it for
        persona when item.metadata doesn't carry it directly."""
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        persona, _history = self._extract_persona_and_history(item)
        persona_text = " ".join(p.strip() for p in persona)

        persona_enc = self._tokenizer(
            persona_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.persona_max_length,
            truncation=True,
        )
        query_enc = self._tokenizer(
            item.question,
            return_tensors="pt",
            padding="max_length",
            max_length=self.query_max_length,
            truncation=True,
        )

        input_ids = self._torch.cat([persona_enc["input_ids"], query_enc["input_ids"]], -1).to(self.device)
        attention_mask = self._torch.cat(
            [persona_enc["attention_mask"], query_enc["attention_mask"]], -1
        ).to(self.device)
        type_ids = self._torch.cat(
            [self._torch.ones_like(persona_enc["input_ids"]), self._torch.zeros_like(query_enc["input_ids"])], -1
        ).to(self.device)
        persona_input_ids = persona_enc["input_ids"].to(self.device)

        start = time.time()
        with self._torch.no_grad():
            output = self._model.generate(
                input_ids,
                token_type_ids=type_ids,
                attention_mask=attention_mask,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                min_length=self.min_length,
                max_length=self.max_length,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                per_input_ids=persona_input_ids,
                use_decoder2=True,
            )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return MethodOutput(answer=text.strip(), latency=time.time() - start)

    def reset(self) -> None:
        self._conversation = None

    def _extract_persona_and_history(self, item):
        """Same contract/fallback logic as plugins/methods/paa, ltkm, orig,
        and sbys -- kept in sync deliberately, since all five are Family E
        persona-conditioned generators targeting the same msc/personachat
        conversation shapes. `history` is returned for interface
        consistency but intentionally unused in answer() -- see this
        module's docstring's "Modeling note"."""
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
            "bob: could not determine persona for this item -- expected item.metadata['persona'], "
            "or a conversation shaped like personachat's PersonaChatDialogue or msc's MscSession"
        )
