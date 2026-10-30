"""ltkm -- "Learning to Know Myself: A Coarse-to-Fine Persona-Aware Training
Framework for Personalized Dialogue Generation" (Li, Hu, Sun, Xing, Guo,
Xie, Peng; AAAI 2023, https://ojs.aaai.org/index.php/AAAI/article/view/26545).
Family E (trained persona-conditioned generation), same family as
plugins/methods/paa -- NOT a prompted-LLM baseline, does not go through
dialogue_benchmarks.llm.complete().

**No official code was ever released for this paper.** Checked GitHub
search (`"Learning to Know Myself"`, `"Coarse-to-Fine Persona-Aware"`) and
arXiv -- no hits. Unlike every other compared method the paper itself
cites (TransferTransfo, LIC each have a GitHub link footnoted directly in
the paper), this paper's own reference section and footnotes never link a
repo, and its own footnote 6 ("The detailed appendix will be available
online") appears to have never materialized publicly either. Everything
under `ltkm_lib/` and the `train_*.py` scripts alongside this file is
therefore an **original implementation written from the paper's text**,
not a vendored/adapted repo -- there is nothing to vendor, unlike every
other baseline in plugins/methods/.

**Status: same tier as plugins/methods/paa -- real, structured code, not
yet trained/run.** Per explicit agreement with the user, this is written
to a "paa-level" bar: an actual architecture and training pipeline exist
(unlike the scaffolding-only state this file was in previously, which
just raised `NotImplementedError` with no implementation behind it), but
none of it has been run against real data yet -- no DNLI download, no
ConvAI2/Persona-Chat/DuLeMon preprocessing, no GPU training, no resulting
checkpoint. `Method.__init__` requires `checkpoint_path` (a directory
produced by `train_stage2_fine.py`) and raises if one isn't supplied,
exactly like paa's own gate on its missing checkpoint.

-----------------------------------------------------------------------
The method, and where each piece lives
-----------------------------------------------------------------------

The framework is **model-agnostic**: the paper applies it to four base
dialogue generators (Per-Seq2Seq, Transformer, GPT2, DialoGPT) and reports
gains for all four, with DialoGPT+K+A its strongest ConvAI2/Persona-Chat
config. `ltkm_lib/generator.py`'s `PersonaDialogueGenerator` wraps any HF
causal LM this way (defaults to DialoGPT-medium); a from-scratch RNN/
Transformer Seq2Seq (the paper's other two baselines) isn't implemented,
since "model-agnostic training recipe on top of a pretrained HF causal LM"
is the tractable target, not reimplementing 2018-era Seq2Seq architectures.

**Stage 1 -- "Learning to Know Myself" (coarse-grained)**, `train_stage1_coarse.py`:
- `ltkm_lib/relation_extractor.py` -- a RoBERTa-large classifier, fine-tuned
  on DNLI persona-sentence -> relation-triple annotations
  (`train_relation_extractor.py`), mapping a persona sentence to a DNLI
  relation label (paper: 82.3% micro-F1 on "41 filtered relations").
- `ltkm_lib/templates.py` -- per-relation question templates (this repo's
  own reconstruction: the paper never publishes its per-relation table,
  only one worked example of the generic `"{r1}_{r2}"` -> `"What {r2} do
  {e1} {r1}?"` pattern, which is kept here as the fallback for any
  DNLI relation not in the hand-written table). The 75+ DNLI relation
  names themselves are real, from Welleck et al. 2019 Appendix A.1.
- `ltkm_lib/question_construction.py` -- extracts the subject entity e1 via
  spaCy dependency parsing + first/second-person pronoun flipping (the
  paper says "keyword matching" without defining it further; spaCy's
  nsubj/nsubjpass + subtree is this repo's concrete choice), then renders
  the question via `templates.build_question`.
- `ltkm_lib/training.py::build_stage1_examples` augments the raw
  (persona, history, query, response) training set with one
  (persona-aware question, persona sentence) pair per persona sentence,
  per the paper; `train_stage1_coarse` then runs plain per-token MLE
  fine-tuning (Eq. 1 -- exactly HF's own `model(**inputs, labels=...).loss`).

**Stage 2 -- "Learning to Avoid Misidentification" (fine-grained)**, `train_stage2_fine.py`:
- `ltkm_lib/entailment_scorer.py` -- a *second*, separately fine-tuned
  DNLI RoBERTa-large (3-way entailment/neutral/contradiction), used for
  three things: `key_persona_scores` (Eq. 2's leave-one-word-out
  entailment-drop scoring), `is_persona_consistent` (Eq. 6's gamma gate),
  and `average_consistency_score` (the paper's own reported "C" metric).
- `ltkm_lib/negative_sampler.py` -- masks each of the top-k key persona
  words (from the entailment scorer) and regenerates them with a
  *pretrained, not fine-tuned* MLM (default `bert-base-uncased`), falling
  back to the 2nd-best MLM prediction when the top one reproduces the
  original word (Fig. 3's exact rule) -- producing up to K negative
  responses per positive.
- `ltkm_lib/training.py::train_stage2_fine` computes the length-normalized
  sigmoid contrastive loss (Eq. 4-5) against the *live* generator being
  trained (not a frozen scorer, per the paper) for the positive response
  and each negative, combined with the plain MLE loss gated by gamma
  (Eq. 6), and backprops the sum.

**Verification performed**: every code path was smoke-tested end-to-end
with tiny *randomly-initialized* HF checkpoints standing in for the real
pretrained models (`hf-internal-testing/tiny-random-roberta` for both the
relation extractor and the entailment scorer, `hf-internal-testing/
tiny-random-bert` for the negative-sampling MLM, `sshleifer/tiny-gpt2` for
the generator) against tiny hand-written examples (no real DNLI/ConvAI2
data): `train_relation_extractor` -> `RelationExtractor.predict` ->
`PersonaAwareQuestionConstructor.construct` (spaCy subject extraction +
pronoun flip + template rendering, using the real spaCy `en_core_web_sm`
model, not a tiny stand-in, since there's no "tiny" spaCy equivalent) ->
`train_entailment_scorer` -> `EntailmentScorer.key_persona_scores`/
`top_k_key_persona_words`/`is_persona_consistent`/`average_consistency_score`
-> `NegativeSampleGenerator.generate_negatives` (confirmed it actually
replaces masked words, including the top-2-fallback branch) ->
`PersonaDialogueGenerator.training_example`/`sequence_log_prob`/`generate`
-> `training.build_stage1_examples` + one `train_stage1_coarse` step ->
one `train_stage2_fine` step (contrastive + MLE loss, backward pass) ->
saving that checkpoint and loading it through `Method.__init__`/
`prepare`/`answer`. Every step ran without error and produced
correctly-shaped outputs; the *generated text itself* is gibberish, as
expected from untrained/randomly-initialized weights -- this confirms the
code is mechanically correct (shapes, gradients, control flow), not that
the paper's numbers are reproducible. That would need the real weights and
the real data pipeline below.

**What has NOT been done** (real GPU training with real weights and real
data, not just code): downloading DNLI (only distributed via Google
Drive, no pip/HF-hub package -- see `ltkm_lib/dnli.py`'s docstring on why
its exact JSON schema is a best-effort reconstruction from a third-party
loader rather than confirmed against the primary files), preprocessing
ConvAI2/Persona-Chat/DuLeMon into this repo's `DialogueExample` JSON shape
(no converter script exists), and actually running
`train_relation_extractor.py` -> `train_entailment_scorer.py` ->
`train_stage1_coarse.py` -> `train_stage2_fine.py` in sequence against
that real data with `roberta-large`/a real base generator. Only that would
confirm the paper's reported gains reproduce here.
"""
from pathlib import Path

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

_LIB_DIR = Path(__file__).resolve().parent


def _load_generator(checkpoint_path: str, device: str | None):
    import sys

    if str(_LIB_DIR) not in sys.path:
        sys.path.insert(0, str(_LIB_DIR))
    from ltkm_lib.generator import PersonaDialogueGenerator

    return PersonaDialogueGenerator(checkpoint_path, device=device)


class Method(_MethodBase):
    """Inference-only wrapper around a checkpoint produced by
    `train_stage2_fine.py` (the full K+A pipeline). Won't run at all until
    one has actually been trained -- see this module's docstring."""

    def __init__(self, checkpoint_path: str = None, device: str = None, max_new_tokens: int = 64, **config):
        super().__init__(checkpoint_path=checkpoint_path, device=device, max_new_tokens=max_new_tokens, **config)
        if not checkpoint_path:
            raise ValueError(
                "ltkm: 'checkpoint_path' is required -- point it at a directory produced by "
                "plugins/methods/ltkm/train_stage2_fine.py (see this method's module docstring "
                "for the full training pipeline; no checkpoint has been trained yet)."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ltkm: checkpoint not found at {checkpoint_path}")

        self.max_new_tokens = max_new_tokens
        self._generator = _load_generator(str(checkpoint_path), device)
        self._conversation = None

    def prepare(self, conversation) -> None:
        """No real per-conversation ingestion cost -- the model is already
        loaded. Just remembers the conversation so answer() can fall back
        to it for persona/history when item.metadata doesn't carry them."""
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        import time

        persona, history = self._extract_persona_and_history(item)
        start = time.time()
        text = self._generator.generate(persona, history, item.question, max_new_tokens=self.max_new_tokens)
        return MethodOutput(answer=text, latency=time.time() - start)

    def reset(self) -> None:
        self._conversation = None

    def _extract_persona_and_history(self, item):
        """Same contract/fallback logic as plugins/methods/paa's
        `_extract_persona_and_history` -- kept in sync deliberately, since
        both are Family E persona-conditioned generators targeting the
        same msc/personachat conversation shapes."""
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
            "ltkm: could not determine persona/dialogue history for this item -- expected "
            "item.metadata['persona'] and ['history'], or a conversation shaped like "
            "personachat's PersonaChatDialogue or msc's MscSession"
        )
