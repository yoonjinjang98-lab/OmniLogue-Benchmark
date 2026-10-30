"""LMEDR -- Learning to Memorize Entailment and Discourse Relations for
Persona-Consistent Dialogues (Chen, Chen, Zhang, Su, Sun; AAAI 2023,
https://arxiv.org/abs/2301.04871, https://github.com/Chenrj233/LMEDR).
Family E (trained persona-conditioned generation), same family as
plugins/methods/paa/ltkm/orig/sbys/bob -- NOT a prompted-LLM baseline, does
not go through dialogue_benchmarks.llm.complete().

**The architecture**: a BART encoder-decoder (`LMEDRModel`, a custom
subclass of `BartPretrainedModel`) augmented with two small learned
"memory" codebooks, each an `nn.Parameter` of shape
`(num_latent, d_model)`:
- `memory1` ("entailment" memory, `num_latent=10` rows by default): read by
  running the encoder over the persona sentences ALONE (`per_input_ids`),
  taking the first hidden state, projecting it through a classification
  head (`latent_head_m1`) to a distribution over the 10 rows, and taking
  the softmax-weighted sum of `memory1`'s rows.
- `memory2` ("discourse-relation" memory, `num_latent2=10` rows by
  default): same recipe, but read from the encoder pass over the FULL
  dialogue context (persona + history + query, `input_ids`) via a second
  head (`latent_head_m2`).
Both readouts are summed and added directly onto the decoder's first
hidden state (`hidden_states[:, 0, :] += latent_variable`, see
`vendor/lmedr_model/modeling_bart.py`'s `BartDecoder.forward`) before
decoding proceeds as ordinary BART generation. This is how the model
conditions generation on (a) how the response should relate to the
persona (entailment) and (b) discourse structure of the conversation,
without lengthening the input sequence. `MemoryLoss` (cosine-orthogonality
between `memory1`/`memory2`) is a training-time regularizer only, not used
at inference -- irrelevant to this Method.

**Vendored** (all under `vendor/`, matching the plugins/methods/bob/xlibs
precedent of vendoring a whole renamed transformers snapshot when a
baseline patches library internals directly):
- `vendor/xlibs_lmedr/` -- upstream requires `transformers==4.14.1`
  specifically because its own `model/modeling_bart.py` imports
  `_expand_mask`, `_make_causal_mask`, and `shift_tokens_right` directly
  from `transformers.models.bart.modeling_bart` -- non-public internal
  helpers that later transformers versions relocated/removed without
  deprecation (confirmed missing by direct import against this venv's
  installed `transformers==4.46.3`: `ImportError: cannot import name
  '_expand_mask'`). Downloaded `transformers==4.14.1`'s wheel directly
  (Apache-2.0, `vendor/LICENSE.txt`) and renamed the top-level package
  `xlibs_lmedr` -- deliberately NOT the same name as bob's `vendor/xlibs`
  (a different, incompatible `transformers==3.5.1` snapshot), since both
  plugins' vendor dirs get added to `sys.path` and Python caches modules
  by name in `sys.modules`: reusing `xlibs` here would make whichever of
  bob/lmedr imports first silently win, handing the other plugin the wrong
  library.
- `vendor/xlibs_lmedr/generation_utils.py` is overwritten with upstream's
  OWN patched copy (`other/generation_utils.py` in the source repo, which
  upstream's README instructs users to copy over their *real, shared*
  installed `transformers/generation_utils.py` -- refused here as
  unacceptably destructive to every other baseline in this repo that
  depends on `transformers`; applied only to this private vendored copy
  instead). Diffed against stock `transformers==4.14.1`: the patch is
  small and self-contained -- `generate()` gains `per_input_ids`/
  `per_attention_mask` parameters, and
  `_prepare_encoder_decoder_kwargs_for_generation` now calls
  `self.forward(..., generate=True)` (routing through `LMEDRModel`'s own
  forward to get both `encoder_outputs` AND the pre-computed
  `latent_variable`) instead of the stock version's plain
  `self.get_encoder()(input_ids, ...)`. The resulting `latent_variable` is
  threaded to every decoding strategy (greedy/sample/beam/beam-sample)
  uniformly through `model_kwargs` -> `prepare_inputs_for_generation()`'s
  returned dict (which `LMEDRModel` already includes a `latent_variable`
  key in) -- unlike plugins/methods/bob's `token_type_ids`, no per-strategy
  threading bug existed here to fix.
- `vendor/lmedr_model/modeling_bart.py`, `memory_loss.py` -- LMEDR's own
  model code (`model/` in the source repo), imports rewritten from
  `transformers`/`transformers.x.y` to `xlibs_lmedr`/`xlibs_lmedr.x.y` (was
  `from model.memory_loss import MemoryLoss` -> `from
  lmedr_model.memory_loss import MemoryLoss`; renamed the containing
  directory from `model` to `lmedr_model` for the same reason as
  `xlibs_lmedr`'s renaming -- `model` is far too generic a top-level module
  name to add to `sys.path` unnamespaced).

**Real bugs found and fixed at vendor time** (found by actually importing
and running the vendored code against a tiny random-weight BART config and
a real `facebook/bart-base` tokenizer -- see verification note below, not
just read from source):
- `xlibs_lmedr/file_utils.py` had `from transformers.utils.versions import
  importlib_metadata` -- an ABSOLUTE self-import of the package's own
  submodule. Harmless in the real, unrenamed `transformers` package (it
  resolves to itself), but once renamed to `xlibs_lmedr` it silently
  resolves to this venv's real, incompatible `transformers==4.46.3`
  instead and crashes on a since-removed `versions.importlib_metadata`
  name. Fixed to a relative import (`from .utils.versions import ...`).
  `xlibs_lmedr/onnx/{config,convert}.py` had four more instances of the
  exact same bug (all fixed to relative imports) -- reached because
  `models/bart/configuration_bart.py` (needed for `BartConfig` itself)
  imports `onnx.OnnxConfigWithPast`.
- `xlibs_lmedr/dependency_versions_check.py` hard-gates import on
  `tokenizers>=0.10.1,<0.11` (an eager version check run at package import
  time) -- incompatible with this shared venv's `tokenizers==0.20.3`
  (needed by hipporag/graphrag/paa's own newer transformers pins). This
  Method only ever uses `BartTokenizer` (the slow, pure-Python tokenizer,
  not `tokenizers`-backed), so the `tokenizers` entry was dropped from
  `pkgs_to_check_at_runtime` rather than pinning a second, conflicting
  `tokenizers` version.
- `xlibs_lmedr/models/__init__.py` eagerly imports EVERY model family's
  subpackage (95+ of them) at import time, not just `bart` -- reaching
  `models/layoutlm/configuration_layoutlm.py`, which imports the buggy
  `onnx` module above regardless of whether anything actually needs
  LayoutLM. Trimmed the import list down to `bart` only (the one model
  family this baseline uses), the same "delete the unreached eager import"
  fix as plugins/methods/bob's `xlibs/__init__.py` trim of
  `pipelines`/`retrieval_rag`/`trainer`.

**Licensing**: no repository-level LICENSE file (confirmed via GitHub's
API: `license: None`); per-file headers (e.g. `modeling_bart.py`'s) claim
Apache-2.0 inheritance from HuggingFace/Fairseq's own original files, but
that doesn't license LMEDR's own additions (the memory codebooks,
`memory_loss.py`, the training/eval scripts). Same resolution as
plugins/methods/pathrag/orig/sbys and this session's other no-LICENSE
repos: used for internal reproduction only, not redistribution -- per
user confirmation.

**Checkpoints**: upstream's README links real fine-tuned checkpoints
(`persona_original`, `persona_revised`, `dstc_model`) as Google Drive
*folders* -- not fetched (unlike plugins/methods/sbys's single HF Hub
file); this Method requires `checkpoint_path` and raises immediately if
not provided. **Trained from scratch instead (2026-09-14)**, following
upstream's own `train_PersonaChat.py` recipe -- see the training
completion note below.

**Modeling note**: `answer()` reconstructs upstream's own
`eval_PersonaChat_build.py::create_encoder_input` encoding exactly:
special tokens `<query>`/`<response>`/`<latent>`/`<persona>` (already in
the checkpoint's saved tokenizer vocab from training -- never added fresh
here) prefix persona sentences (each `sep_token`-terminated) and history
turns (alternating query/response starting with query, each
`eos_token`-terminated); `item.question` is appended as one more implicit
turn continuing that alternation. `per_input_ids` (persona sentences only,
same `<latent><persona>...` prefix) is passed alongside `input_ids` to
`generate()`, exactly as upstream's `evaluation_PersonaChat.py::act()`
does for its non-hits@1 (generation) path.

Verification: built a tiny random-weight `BartConfig`
(`d_model=16, encoder/decoder_layers=2`) plus a real
`facebook/bart-base` tokenizer (its vocab/merges only -- no pretrained
weights used) with the four special tokens added, constructed an
`LMEDRModel` from scratch, ran `save_pretrained(safe_serialization=False)`
+ `from_pretrained()` round-trip (mirroring how a real checkpoint from
upstream's `train_PersonaChat.py --do_train` would be loaded), then called
`generate()` with `per_input_ids`/`per_attention_mask` under greedy
decoding, sampling, beam search, and beam-sampling -- all four completed
without error and produced correctly-shaped output. The generated text
itself is gibberish, as expected from untrained random weights -- this
confirms the vendored+patched code (including all bugs fixed above) is
mechanically correct, not that LMEDR's own reported numbers reproduce
here.

**Trained for real (2026-09-14), real checkpoint now exists**:
`$DIALOGUE_BENCHMARKS_DATA/checkpoints/lmedr/final/` (a directory:
`pytorch_model.bin` + `config.json` + tokenizer files, exactly what
`from_pretrained()` above expects). Vendored upstream's own
`train_PersonaChat.py`/`build_data_PersonaChat.py` under `training/`
(imports repointed at the already-vendored `xlibs_lmedr`/`lmedr_model`).
Training data: `prepare_convai2_lmedr.py`/`prepare_dnli_lmedr.py`
(new scripts) built LMEDR's own 20-candidates-pipe-joined ConvAI2 format
(confirmed this repo's personachat parquet already carries the needed
`candidates` field, last one = gold) and reformatted the already-
downloaded `xksteven_dialogue_nli_train.jsonl` into upstream's expected
single-JSON-array shape (label values already exactly "positive"/
"negative"/"neutral" -- no new download needed). Real `facebook/bart-large`
weights downloaded via this venv's own modern `huggingface_hub` (the
vendored old `xlibs_lmedr`'s own download code 404s against a bare Hub
id -- pointed `--base_model` at the local snapshot dir instead, zero
repackaging needed). **Real OOM hit and fixed**: `train_batch_size=32`
(the initially-chosen batch, tuned against a short smoke sample) OOM'd
partway into the first real epoch -- real long conversations exceed the
smoke sample's memory footprint -- fixed by dropping to
`train_batch_size=16, gradient_accumulation_steps=2` (same effective
batch, safer headroom) plus `PYTORCH_CUDA_ALLOC_CONF=expandable_
segments:True`. Ran all 20 epochs successfully after that fix (~2 days
9 hours wall-clock, ~3h10m/epoch very consistently): the classification
head's dialogue-turn accuracy (gold-vs-candidate-response discrimination,
not LMEDR's own paper-reported headline metric) climbed 0.706 -> ~0.885
by epoch 14 and then plateaued in the 0.886-0.887 range through epoch 20
with no overfitting inflection, so the final epoch's checkpoint (which is
what upstream's own script saves as the final `WEIGHTS_NAME`-renamed
file) was kept as-is -- no earlier-epoch selection needed, same situation
`paa` ended up in. `mem` (the `MemoryLoss` codebook-orthogonality
regularizer) decayed toward ~0 as expected across training. **Verified
through the real `Method` end-to-end**: "i love pizza . i work as a
chef ." persona, question "what is your favorite food?" -> "pizza is my
favorite food."; a hunting/remodeling persona, question "what do you
like to do for fun?" -> "i like to go hunting and remodel homes." -- both
correctly persona-grounded, same verification tier as `ltkm`/`sbys`/
`bob`/`orig`/`paa`. `lmedr` is now the fifth and last genuinely-runnable
Family E baseline this session, closing out the family: all of
`ltkm`/`sbys`/`bob`/`orig`/`paa`/`lmedr` now run against real trained (or
upstream-released) weights.
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
    import xlibs_lmedr
    from lmedr_model.modeling_bart import LMEDRModel

    return xlibs_lmedr, LMEDRModel


class Method(_MethodBase):
    """Inference-only wrapper around a checkpoint produced by upstream's
    `train_PersonaChat.py --do_train` (or one of its released Google
    Drive checkpoints, downloaded and pointed at manually) -- won't run at
    all until one is actually available locally."""

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        max_length: int = 40,
        min_length: int = 1,
        num_beams: int = 2,
        max_history_turns: int = 12,
        num_latent: int = 10,
        num_latent2: int = 10,
        **config,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            max_history_turns=max_history_turns,
            num_latent=num_latent,
            num_latent2=num_latent2,
            **config,
        )
        if not checkpoint_path:
            raise ValueError(
                "lmedr: 'checkpoint_path' is required -- point it at a directory produced by "
                "upstream's `train_PersonaChat.py --do_train`, or one of its released checkpoints "
                "(persona_original/persona_revised, currently Google-Drive-hosted folders -- see "
                "this method's module docstring; not fetched automatically here)."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"lmedr: checkpoint not found at {checkpoint_path}")

        xlibs_lmedr, LMEDRModel = _load_vendor()

        import torch

        self._torch = torch
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.max_history_turns = max_history_turns
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        tokenizer = xlibs_lmedr.BartTokenizer.from_pretrained(str(checkpoint_path))
        self._tokenizer = tokenizer
        self._query_id, self._res_id, self._latent_id, self._persona_id = tokenizer.convert_tokens_to_ids(
            ["<query>", "<response>", "<latent>", "<persona>"]
        )
        self._sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self._eos_id = tokenizer.eos_token_id

        model = LMEDRModel.from_pretrained(str(checkpoint_path), num_latent=num_latent, num_latent2=num_latent2)
        model.to(self.device)
        model.eval()
        self._model = model

        self._conversation = None

    def prepare(self, conversation) -> None:
        """No real per-conversation ingestion cost -- the model is already
        loaded. Just remembers the conversation so answer() can fall back
        to it for persona/history when item.metadata doesn't carry them
        directly."""
        self._conversation = conversation

    def _tokenize(self, text: str):
        return self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text, add_prefix_space=True))

    def answer(self, item) -> MethodOutput:
        persona, history = self._extract_persona_and_history(item)
        if self.max_history_turns and len(history) > self.max_history_turns:
            history = history[-self.max_history_turns :]

        per_input_ids = [self._latent_id, self._persona_id]
        for sentence in persona:
            per_input_ids += self._tokenize(sentence) + [self._sep_id]

        encoder_input_ids = list(per_input_ids)
        turns = list(history) + [item.question]
        for i, turn_text in enumerate(turns):
            speaker_id = self._query_id if i % 2 == 0 else self._res_id
            encoder_input_ids += [speaker_id] + self._tokenize(turn_text) + [self._eos_id]

        input_ids = self._torch.tensor([encoder_input_ids], device=self.device)
        attention_mask = self._torch.ones_like(input_ids)
        per_ids = self._torch.tensor([per_input_ids], device=self.device)
        per_mask = self._torch.ones_like(per_ids)

        start = time.time()
        with self._torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                per_input_ids=per_ids,
                per_attention_mask=per_mask,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
            )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return MethodOutput(answer=text, latency=time.time() - start)

    def reset(self) -> None:
        self._conversation = None

    def _extract_persona_and_history(self, item):
        """Same contract/fallback logic as plugins/methods/paa, ltkm, orig,
        sbys, and bob -- kept in sync deliberately, since all six are
        Family E persona-conditioned generators targeting the same
        msc/personachat conversation shapes."""
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
            "lmedr: could not determine persona/dialogue history for this item -- expected "
            "item.metadata['persona'] and ['history'], or a conversation shaped like "
            "personachat's PersonaChatDialogue or msc's MscSession"
        )
