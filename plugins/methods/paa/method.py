"""PAA -- Persona-Adaptive Attention (Huang et al., AAAI 2023).

Wraps https://github.com/hqsiswiliam/persona-adaptive-attention: a GPT2-based
encoder-decoder that fuses persona and dialogue-context encodings via a gated
cross-attention mechanism, trained on ConvAI2. Targets personachat/msc-shaped
persona-conditioned next-turn generation (task_type="dialogue").

Vendored (trimmed to the inference path) under vendor/, with its top-level
`model`/`utils`/`config_loader` packages renamed to `paa_model`/`paa_utils`/
`paa_config_loader` so sys.path insertion below can't shadow another plugin's
own generically-named modules loaded in the same process.

IMPORTANT -- this is NOT a prompted-LLM baseline and does not go through
dialogue_benchmarks.llm.complete(). The upstream repo ships no pretrained PAA
checkpoint: only base GPT2 weights + the ConvAI2 corpus + a train.py to fine-tune
your own (see vendor/reference_configs/paa_transformer/*.yml for the configs the
paper used -- fusion_mode "((pr)(cr))" i.e. "pr-cr" is vanilla PAA). This wrapper
only runs inference from an already-trained checkpoint via `checkpoint_path`; it
does not train one for you.

Also requires the raw base-LM weights the checkpoint's base_model points at (e.g.
gpt2-pytorch_model.bin from https://huggingface.co/gpt2 -- the old-style .bin
state dict, not the modern .safetensors default) under `downloaded_LM/` next to
this file, or $PAA_DOWNLOADED_LM_DIR -- the vendored model constructor always
loads these to build the decoder before the fine-tuned checkpoint's own weights
are applied on top, even for pure inference.

**Trained for real (2026-09-14) -- both compatibility caveats below were
actually hit and fixed while training, not just theorized:**
- vendor/paa_model/paa_decoder/.../gpt_decoder_PAAGPT.py (the actual PAA
  transformer decoder) genuinely failed to import against this venv's real
  transformers==4.46.3 -- NOT the file_utils/model_parallel_utils imports
  this docstring used to warn about (those turned out to be fine), but
  `TypeError: add_code_sample_docstrings() got an unexpected keyword
  argument 'tokenizer_class'` at 3 call sites. Confirmed via
  `add_code_sample_docstrings`'s own purpose (Sphinx doc-string generation
  only, zero runtime effect) that deleting the 3 multi-line decorator
  blocks outright was safe -- done at vendor time, no pyproject.toml pin
  needed after all (runs fine against the shared venv's real transformers).
- torch.load() on the checkpoint here unpickles a DotMap config object,
  which needs weights_only=False (torch>=2.6 flipped the default to True,
  rejecting it with `WeightsUnpickler error: Unsupported global: GLOBAL
  dotmap.DotMap`) -- confirmed against a real trained checkpoint (this
  venv runs torch 2.14) and fixed at both call sites (here and
  `vendor/paa_utils/save_load_model.py::load_model`).

The persona/dialogue-history extraction below expects either
`item.metadata["persona"]`/`item.metadata["history"]`, or a conversation shaped
like personachat's PersonaChatDialogue or msc's MscSession (personachat/msc don't
implement eval_items() yet -- see BASELINE_METRIC_PLAN.md -- so this is a
best-guess contract pending that).

**Trained end to end (2026-09-14), real checkpoint now exists**:
`$DIALOGUE_BENCHMARKS_DATA/checkpoints/paa/final/model.pt` (a `best.pt`
that turned out identical to `last.pt` -- validation `token_ppl` decreased
monotonically the entire run, 20.66 -> 14.20 over 30 epochs, no
overfitting inflection like `orig` hit, so there was no "pick an earlier
checkpoint" decision to make). Training driver: vendored upstream's own
`train.py`/`trainer/causal_decoder_trainer.py`/`dataset/`/`evaluation/`
under `training/` (renamed `paa_trainer`/`paa_dataset`/`paa_evaluation` to
match this plugin's existing vendor-renaming convention; reused the
already-vendored `paa_utils.get_model_by_config`/`get_tokenizer`/
`save_load_model`/`eprint`, only `logger.py`/`setup_seed.py`/
`parser_helper.py` needed adding). **Two more real bugs found in
upstream's own `train.py`, confirmed by actually running it**: (1)
`config_loader/config.py::get_config()` hardcodes a bare relative
`open('config/default.yml')` -- only ever correct with CWD == upstream's
own repo root, never exercised before since inference never calls
`get_config()` -- fixed to resolve relative to the vendored file's own
location (`vendor/paa_config_loader/config.py`). (2) `train.py` reads
`args.btn_count`/`args.btn_fuse_layer` but never registers either as an
argparse argument -- an unconditional `AttributeError` on any invocation
as literally published -- added both with `default=None`, matching every
other optional arg's convention (`training/train.py`). Training data:
reused `lmedr`'s already-prepared `train_self_original.txt`/
`valid_self_original.txt` directly -- `paa`'s own `dataset_helper.py::
read_personachat_split` parser is nearly byte-identical to LMEDR's
`create_data()` (same raw-ConvAI2-with-pipe-joined-candidates format,
confirmed by direct code comparison before reusing, not assumed) --
via a symlinked `data/ConvAI2/` directory (upstream's own `--dataset
convai2` CLI flag unconditionally overwrites whatever the YAML config's
`dataset.train`/`test`/`valid` fields say with that hardcoded relative
path, so a symlink was simpler than patching the override away). Base
checkpoint: `gpt2-pytorch_model.bin`, confirmed to already be in the old
flat-key convention this vendored decoder expects (same file `orig`'s own
training already validated loads with `strict=True`, zero repackaging --
reused that finding directly), now kept permanently at
`$DIALOGUE_BENCHMARKS_DATA/checkpoints/paa/downloaded_LM/` (point
`$PAA_DOWNLOADED_LM_DIR` there -- this is a real ongoing runtime
dependency for every future inference call, not just training-time
staging, so it does NOT live under a `_work/` directory that later
cleanup might delete). Config: `paa_transformer-small.yml` (`((pr)(cr))`
fusion mode, upstream's own description of "vanilla PAA"), batch bumped
from upstream's default 8 to 16 (a hardware-efficiency bump only, ~13GB/
96GB GPU memory used -- plenty of headroom left unused, didn't push
further since ~17min/epoch was already tractable). **Verified through
the real `Method` end-to-end**: "i love pizza . i work as a chef ."
persona, history "what is your favorite food?" -> "hi , i'm a chef , how
about you ?"; a hunting/remodeling persona, history "what do you like to
do for fun?" -> "hi , i like to go hunting and remodel homes ." -- both
correctly persona-grounded, same verification tier as `ltkm`/`sbys`/
`bob`/`orig`. `paa` is now the fourth genuinely-runnable Family E
baseline this session (after `sbys`'s released checkpoint, `ltkm`'s and
`orig`'s from-scratch training), via locally-trained weights.
"""
import time
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_PUNCTUATION = ".,?"


def _load_vendor():
    """Imports torch/transformers/dotmap and the vendored PAA code lazily, so
    merely importing this module (e.g. `db validate`, `db list`) doesn't require
    the heavy `paa` extra -- only actually constructing Method() does."""
    import sys

    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))
    try:
        import torch  # noqa: F401
        from paa_config_loader.config import extend_compatibility_for_paa_transformer
        from paa_utils.get_model_by_config import get_model_via_config
        from paa_utils.get_tokenizer import get_tokenizer
        from paa_utils.save_load_model import load_model
    except ImportError as e:
        raise ImportError(
            "paa: missing a dependency for this baseline -- install it with "
            "`.venv/bin/pip install -e '.[paa]'` (torch/transformers/dotmap/etc, "
            "declared as an optional extra since they're heavy and paa-specific)"
        ) from e
    return torch, extend_compatibility_for_paa_transformer, get_model_via_config, get_tokenizer, load_model


def _preprocess_text(text: str) -> str:
    for punct in _PUNCTUATION:
        text = text.replace(punct, f" {punct} ")
    return " ".join(text.split())


def _persona_string(persona_sentences) -> str:
    joined = " ".join(s.strip() for s in persona_sentences)
    return _preprocess_text(f"P: {joined}")


def _context_string(history, sep_token: str) -> str:
    tagged = [f"{'Q:' if i % 2 == 0 else 'R:'} {utt}" for i, utt in enumerate(history)]
    return _preprocess_text(f" {sep_token} ".join(tagged))


class Method(_MethodBase):
    """Loads a locally trained PAA checkpoint once (in __init__, not prepare() --
    it's shared across every conversation in the run) and generates
    persona-conditioned next turns. See the module docstring: requires
    `checkpoint_path`, and won't run at all until one has actually been trained."""

    def __init__(self, checkpoint_path: str = None, device: str = None, max_len: int = 64, **config):
        super().__init__(checkpoint_path=checkpoint_path, device=device, max_len=max_len, **config)
        if not checkpoint_path:
            raise ValueError(
                "paa: 'checkpoint_path' is required -- point it at a checkpoint produced by "
                "training the vendored PAA architecture on ConvAI2 (see the module docstring); "
                "this baseline has no pretrained weights to fall back on."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"paa: checkpoint not found at {checkpoint_path}")

        import copy

        torch, extend_compatibility_for_paa_transformer, get_model_via_config, get_tokenizer, load_model = (
            _load_vendor()
        )
        self._torch = torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        # weights_only=False: the checkpoint's `config` field is a pickled
        # DotMap object, not just tensors -- torch>=2.6 flipped the default
        # to True, which rejects it (`WeightsUnpickler error: Unsupported
        # global: GLOBAL dotmap.DotMap`), confirmed by actually loading a
        # real trained checkpoint (this venv runs torch 2.14). Safe here:
        # the checkpoint is produced by this repo's own training run, not
        # an untrusted download.
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model_config = extend_compatibility_for_paa_transformer(checkpoint["config"])
        self._sep_token = model_config.dataset.persona_query_token

        self._tokenizer = get_tokenizer(model_config.tokenizer.vocab)
        self._left_tokenizer = copy.deepcopy(self._tokenizer)
        self._left_tokenizer.padding_side = "left"

        model = get_model_via_config(model_config, self._tokenizer)
        model.to(self.device)
        load_model(model, path=str(checkpoint_path), device=self.device)
        model.eval()
        self._model = model

        self._conversation = None

    def prepare(self, conversation) -> None:
        """No real per-conversation ingestion cost -- the model is already loaded.
        Just remembers the conversation so answer() can fall back to it for
        persona/history when item.metadata doesn't carry them directly."""
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        persona, history = self._extract_persona_and_history(item)
        persona_input = self._left_tokenizer(
            [_persona_string(persona)], return_tensors="pt", add_special_tokens=False, padding=True
        )
        query_input = self._left_tokenizer(
            [_context_string(history, self._sep_token)], return_tensors="pt", add_special_tokens=False, padding=True
        )
        persona_input = {k: v.to(self.device) for k, v in persona_input.items()}
        query_input = {k: v.to(self.device) for k, v in query_input.items()}

        start = time.time()
        with self._torch.no_grad():
            texts = self._model.generate(
                {"persona_input": persona_input, "query_input": query_input}, max_len=self.max_len
            )
        return MethodOutput(answer=texts[0].strip(), latency=time.time() - start)

    def reset(self) -> None:
        self._conversation = None

    def _extract_persona_and_history(self, item):
        meta = item.metadata or {}
        persona = meta.get("persona")
        history = meta.get("history")
        if persona is not None and history is not None:
            return list(persona), list(history)

        conv = self._conversation
        if conv is not None and hasattr(conv, "personality") and hasattr(conv, "turns"):
            # plugins/datasets/personachat.py: PersonaChatDialogue
            turn_index = meta.get("turn_index", len(conv.turns) - 1 if conv.turns else None)
            turn = conv.turns[turn_index] if turn_index is not None else None
            return (
                list(persona) if persona is not None else list(conv.personality),
                list(history) if history is not None else (list(turn.history) if turn else []),
            )
        if conv is not None and hasattr(conv, "persona1") and hasattr(conv, "utterances"):
            # plugins/datasets/msc.py: MscSession -- persona of whoever speaks next
            up_to = meta.get("speaker_index", len(conv.utterances))
            speakers = conv.speakers or []
            next_speaker = speakers[up_to] if up_to < len(speakers) else (speakers[-1] if speakers else None)
            inferred_persona = conv.persona1 if next_speaker != "Speaker 2" else conv.persona2
            return (
                list(persona) if persona is not None else list(inferred_persona),
                list(history) if history is not None else list(conv.utterances[:up_to]),
            )
        raise ValueError(
            "paa: could not determine persona/dialogue history for this item -- expected "
            "item.metadata['persona'] and ['history'], or a conversation shaped like "
            "personachat's PersonaChatDialogue or msc's MscSession"
        )
