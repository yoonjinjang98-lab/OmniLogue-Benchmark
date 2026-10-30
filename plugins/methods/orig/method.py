"""ORIG -- Order-Insensitive Representation Regularization (Chen, Wang,
Deng, Kwan, Wang, Wong; ACL 2023 Findings, "Towards Robust Personalized
Dialogue Generation via Order-Insensitive Representation Regularization",
arXiv:2305.12782, https://github.com/ChanLiang/ORIG). Family E (trained
persona-conditioned generation), same family as plugins/methods/paa and
plugins/methods/ltkm -- NOT a prompted-LLM baseline, does not go through
dialogue_benchmarks.llm.complete().

**The method**: fine-tunes DialoGPT (or plain GPT2) on persona-conditioned
dialogue as usual, but adds a second training signal targeting the actual
problem this baseline is for -- persona-prefix-prompting style baselines
answer differently depending on what *order* the persona sentences happen
to be concatenated in, which is an artifact of the prompt, not the
persona's actual content. Each training step runs the model on the SAME
example twice: once with persona sentences in their original order, once
with them randomly shuffled (`persona_data_loader.py`'s
`PersonaDataset(shuffle=True)`, upstream). A symmetric KL-divergence loss
between the two runs' output distributions (`persona_train_kl.py`'s
`get_distance_loss`, masked to the persona/context prefix positions) is
added to the normal per-token LM loss, weighted by `alpha`:
`loss = lm_loss + alpha * kl_loss`. The model is pushed to produce the
same predictions regardless of persona sentence order, without needing to
enumerate all `T!` permutations at training time (only 2 orderings per
step) -- this repo's comparison point for whether that actually reduces
order-sensitivity vs. a plain persona-prefix-prompting baseline like `paa`.

**Licensing note**: ChanLiang/ORIG has no LICENSE file and GitHub's own
detector reports none (confirmed via the GitHub API). This is the same
situation already hit and explicitly decided for plugins/methods/pathrag
and this repo's license-gap datasets (groupmembench/dialsim) -- reconfirmed
with the user this session on the same basis: internal reproduction only,
not redistribution. Individual vendored files carry per-file license
headers inherited from what they were built on top of (`decoding.py`'s
functions call into plain `transformers`, no header of its own; the
upstream repo's *training* scripts and `lsp_model/` carry MIT (Microsoft
DialoGPT) and Apache-2.0 (OpenAI/HuggingFace GPT-2) headers respectively --
none of which this Method actually vendors, see below) -- but the repository
as a whole, and ORIG's own new code specifically (`persona_data_loader.py`,
`decoding.py`, the KL-regularization logic in `persona_train_kl.py`), carry
no license at all.

**Vendored**: only `vendor/orig_decoding.py` (renamed from upstream's
`decoding.py` to avoid a generic `sys.modules["decoding"]` name colliding
with anything else loaded in the same process, the same precaution
plugins/methods/paa's own vendor renaming takes) -- specifically its
`make_example_inputs()` (builds the persona+context prompt: each sentence
tokenized and EOS-separated, `token_type_ids` distinguishing persona(0)/
context(1), left-padded), `fix_state_dict_namespace()` (renames a trained
checkpoint's `lm_head.decoder.weight` -> `lm_head.weight` and strips a
`module.` DDP prefix), and `load_model()` (loads that fixed state dict
into an already-constructed model, non-strict). **Nothing else is
vendored, deliberately**: `lsp_model/` (the training-time model class)
imports from `pytorch_pretrained_bert` -- the predecessor package to
`transformers`, unmaintained since ~2019 -- but that class is *only* used
by upstream's own training scripts (`persona_train.py`/
`persona_train_kl.py`), which this Method doesn't run (same "here's the
train script, go run it yourself" stance as `paa`). At inference time,
upstream's own `decoding.py` already loads a plain modern
`transformers.AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-
{size}")` and overlays the fine-tuned checkpoint's *state dict* onto it
(via the three vendored functions above) -- no `pytorch_pretrained_bert`
dependency exists on this path at all, so this Method reuses exactly that,
and needs no vendoring beyond it.

Requires a locally trained checkpoint (`checkpoint_path` config, a raw
PyTorch state-dict `.pkl` file, exactly what `persona_train_kl.py` saves
via `torch.save(model.state_dict(), ...)`) -- upstream ships no pretrained
ORIG weights, only `scripts/train_persona_gpt_kl.sh` + `persona_train_kl.py`
to fine-tune your own starting from base DialoGPT/GPT2 weights.

**Trained for real (2026-09-14)**, following upstream's own reference
recipe (`scripts/train_persona_gpt_kl.sh`) as closely as possible:
`training/` vendors `persona_train_kl.py`/`persona_data_loader.py`
verbatim from upstream plus its own `lsp_model`/`gpt2_training`
sub-packages (already bundled INSIDE the ChanLiang/ORIG repo itself, not a
separate Microsoft clone as first assumed -- only 212KB+56KB, depending on
the old, still-pip-installable `pytorch_pretrained_bert==0.6.2` package).
Base checkpoint: HF Hub's own `gpt2` repo's `config.json`/`vocab.json`/
`merges.txt`/`pytorch_model.bin`, used completely as-is -- confirmed its
`pytorch_model.bin` is *already* in the old flat (no `transformer.`
prefix) key convention `lsp_model.GPT2LMHeadModel` expects, so
`load_model()`'s own "keys don't start with `transformer.` -> load into
`.transformer` only" branch fires automatically and loads with
`strict=True`, no repackaging needed at all. Training data: upstream's own
repo ships real, ready-to-use ConvAI2-derived examples directly
(`data/{train,valid,test}/output`, 65719/7801/7512 examples, real JSON
arrays of `[persona_list, history, response, persona_label]` -- the valid
split's turn count (7801) matches this repo's own personachat validation
parsing exactly, corroborating it's the same standard ConvAI2 split), so
no new conversion script was needed (unlike `ltkm`/`bob`, which had to
build their own from this repo's parquet).

**Three real bugs found and fixed in upstream's own public release,
confirmed by actually running training, not by reading alone**:
1. `gpt2_training/train_utils.py` does `from env import END_OF_TEXT_TOKEN`,
   but no `env.py` exists anywhere in the repo (confirmed missing, not a
   shallow-clone artifact) -- added a 1-line local `training/env.py`
   stub with the same literal (`'<|endoftext|>'`) already hardcoded
   identically in this repo's own `decoding.py`/`persona_data_loader.py`,
   not a guess. The one function that actually needed it
   (`get_eval_list_same_length`) is itself dead code from
   `persona_train_kl.py`'s own perspective (imported, never called).
2. `gpt2_training/eval_utils.py` needs `pycocoevalcap` (BLEU/CIDEr, a
   Java-dependent package) just to import -- but the function it defines
   (`eval_model_loss`) is ALSO dead code (`persona_train_kl.py` defines
   and uses its own local `my_eval_model_loss` instead) -- dropped that
   one import line in the vendored copy rather than installing the extra
   dependency.
3. **The real one**: the KL training loop does
   `for batch, batch1 in zip(train_dataloader, train_dataloader1): assert
   torch.equal(position_ids0, position_ids1); assert torch.equal(label_ids0,
   label_ids1)` -- i.e. it assumes both loaders always yield the SAME
   underlying examples in the SAME order (original persona order vs.
   shuffled persona order of the *same* example, everything else
   identical, so labels/lengths must match). Upstream builds each loader
   with an independent `RandomSampler`, which does NOT guarantee the two
   shuffles line up -- confirmed empirically: this raises `AssertionError`
   on real data (not a synthetic edge case). Fixed by giving both loaders
   a plain `SequentialSampler` instead (`training/persona_train_kl.py`'s
   `get_dataloader()`/`get_train_dataloader()`) -- preserves the actual
   algorithmic contribution (comparing original vs. shuffled *persona*
   order within each example) and only removes an already-broken "shuffle
   example order across epochs" nicety neither loader could have actually
   delivered as originally written.

**Also discovered, not a code bug but a real gotcha worth remembering**:
`persona_data_loader.py`'s `read_file()` caches tokenized examples to a
sibling file keyed ONLY by path (`path.strip('output') + 'cached'`/
`'shuffle{1-7}'`), ignoring `with_persona_label`/`all_seq_loss`/etc. --
reusing the same data-file path across runs with DIFFERENT flag values
silently serves stale, mismatched cached content (this is what actually
produced several confusing, seemingly-random `AssertionError`s while
iterating on the fix above, before the real dataloader-sync bug was
isolated). Not worked around in the vendored file since a real, single-
config production run never revisits the same path with different flags,
but worth remembering before any future re-run against the same data
files with different arguments -- delete the `cached`/`shuffleN` siblings
first.

Matched upstream's own reference hyperparameters exactly (`--max_seq_length
180 --alpha 1.0 --train_batch_size 32 --gradient_accumulation_steps 2
--eval_batch_size 8 --num_epoch 6 --num_optim_steps 6000 --learning_rate
1e-5 --warmup_steps 600 --with_persona_label False --no_token_id True`).
Validation loss improved from 3.11 (step 200) to a minimum of **2.75 ppl
16.13 at step 2400** (epoch 2), then rose monotonically to 2.95-3.00
through the rest of training (steps 4600-6000) -- clear overfitting past
that point, so **the final checkpoint kept is step 2400's, not the last
one** (all 30 per-`valid_step` checkpoints were inspected via the run's
own `eval_log_*.txt`, not guessed). `$DIALOGUE_BENCHMARKS_DATA/checkpoints/
orig/final/model.pkl`. **Verified through the real `Method` end-to-end**
(not mock/tiny-random): "i love pizza . i work as a chef ." persona,
history "what is your favorite food?" -> "i like pizza. what about you?";
"i like to remodel homes . i like to go hunting . i like to shoot a bow .
my favorite holiday is halloween ." persona, history "what do you like to
do for fun?" -> "i like to go hunting. what about you?" -- both correctly
persona-grounded and on-topic, same verification tier as `ltkm`/`sbys`/
`bob`. `orig` is now the third Family E baseline this session that's
genuinely runnable (after `sbys`'s released checkpoint and `ltkm`'s own
from-scratch training), via locally-trained weights following upstream's
real recipe.
"""
import sys
import time
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_END_OF_TEXT = "<|endoftext|>"


def _load_vendor():
    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))
    import orig_decoding

    return orig_decoding


class Method(_MethodBase):
    """Inference-only wrapper around a checkpoint produced by upstream's
    `persona_train_kl.py`. A real one now exists at
    `$DIALOGUE_BENCHMARKS_DATA/checkpoints/orig/final/model.pkl` (see the
    module docstring for the training run and its two real bugs)."""

    def __init__(
        self,
        checkpoint_path: str = None,
        model_size: str = "medium",
        device: str = None,
        max_new_tokens: int = 40,
        num_beams: int = 10,
        padding_to: int = 128,
        **config,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            model_size=model_size,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            padding_to=padding_to,
            **config,
        )
        if not checkpoint_path:
            raise ValueError(
                "orig: 'checkpoint_path' is required -- point it at a state-dict .bin "
                "produced by upstream's persona_train_kl.py (see this method's module "
                "docstring; upstream ships no pretrained checkpoint of its own)."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"orig: checkpoint not found at {checkpoint_path}")

        orig_decoding = _load_vendor()

        import torch
        from transformers import AutoModelForCausalLM, GPT2Tokenizer

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.padding_to = padding_to
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        base_model_name = f"microsoft/DialoGPT-{model_size}"
        self._tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self._eos = self._tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = orig_decoding.load_model(model, str(checkpoint_path), self.device)
        model.eval()
        self._model = model
        self._make_example_inputs = orig_decoding.make_example_inputs

        self._conversation = None

    def prepare(self, conversation) -> None:
        """No real per-conversation ingestion cost -- the model is already loaded.
        Just remembers the conversation so answer() can fall back to it for
        persona/history when item.metadata doesn't carry them directly."""
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        persona, history = self._extract_persona_and_history(item)
        persona_ids = [self._tokenizer.encode(s) for s in persona]
        history_ids = [self._tokenizer.encode(s) for s in history]

        example = self._make_example_inputs(
            id=0,
            personas=persona_ids,
            context=history_ids,
            response="",
            persona_label=[],
            assigned_persona_label=[],
            eos=self._eos,
            device=self.device,
            input_ground_truth_persona_label=False,
            padding_to=self.padding_to,
        )

        start = time.time()
        with self._torch.no_grad():
            output = self._model.generate(
                input_ids=example["input_ids"],
                attention_mask=example["attention_mask"],
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self._eos,
                pad_token_id=self._eos,
            )
        generated = output[0, example["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return MethodOutput(answer=text, latency=time.time() - start)

    def reset(self) -> None:
        self._conversation = None

    def _extract_persona_and_history(self, item):
        """Same contract/fallback logic as plugins/methods/paa and
        plugins/methods/ltkm -- kept in sync deliberately, since all three
        are Family E persona-conditioned generators targeting the same
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
            "orig: could not determine persona/dialogue history for this item -- expected "
            "item.metadata['persona'] and ['history'], or a conversation shaped like "
            "personachat's PersonaChatDialogue or msc's MscSession"
        )
