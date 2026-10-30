"""ss_mpc -- SS-MPC (Jang et al., 2025): multi-party response generation with
the dialogue structure encoded as *sequential* input to a plain BART, no
graph encoder. Input per instance, exactly as the vendored loader builds it
(loader_multi.BaseDataModule._build_enc_label):

    [i][parent of i][SPK speaker][SPK addressee] utterance_i  ... for each context utterance
    [n][answer's parent][SPK responder][SPK addressee]          -- the slot to generate

Train with train.py (the vendored code, unmodified: encoder post-training,
then fine-tuning); this method loads the final Lightning checkpoint and
generates with the same settings as the vendored test_step
(model.generate(max_length=50), BART's own generation config). For
ubuntu_irc_gen; reads the instance's raw GSN fields (relation_at, ctx_spk,
ctx_adr, ans_idx, ans_spk, ans_adr) off the conversation object.

Speaker ids: the vendored collate randomly permutes ids 1-8 per batch as
augmentation, at test time too; here the identity permutation is used so
outputs are deterministic (the model is trained to be invariant to it).
"""
from pathlib import Path

from dialogue_benchmarks._paths import DATA_ROOT
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

ADD_TOKENS = [f"[{i}]" for i in range(-1, 9)] + [f"[SPK{i}]" for i in range(-1, 9) if i != 0] + ["[IDX]", "[AD]"]


def build_label(context, rel, cspk, cadr, utterance_len, idx, spk, adr) -> str:
    """Verbatim port of loader_multi.BaseDataModule._build_enc_label."""
    return (
        "".join(f"[{i}][{rel[i]}][SPK{cspk[i]}][SPK{cadr[i]}]" + c for i, c in enumerate(context))
        + f"[{utterance_len}][{idx}][SPK{spk}][SPK{adr}]"
    )


def build_input(raw: dict, context: list, fits) -> str:
    """The vendored collate_fn for one instance (identity speaker map): parent
    indices from relation_at, and the same left-truncation -- drop the oldest
    utterance (and its rel/spk/adr entries, keeping every index absolute)
    until `fits(text)`."""
    rel = [y[1] for y in [[-1, -1]] + list(raw["relation_at"])]
    cspk, cadr = list(raw["ctx_spk"]), list(raw["ctx_adr"])
    context = list(context)
    utterance_len = len(context)
    while True:
        text = build_label(context, rel, cspk, cadr, utterance_len, raw["ans_idx"], raw["ans_spk"], raw["ans_adr"])
        if fits(text) or len(context) <= 1:
            return text
        context, rel, cspk, cadr = context[1:], rel[1:], cspk[1:], cadr[1:]


class Method(_MethodBase):
    def __init__(self, base_model: str = "facebook/bart-large", checkpoint: str = None, **config):
        config.pop("model", None)  # --model names the prompting LLM of other methods, not this one
        super().__init__(base_model=base_model, checkpoint=checkpoint, **config)
        self.base_model = base_model
        self.checkpoint = checkpoint
        self._model = None
        self._conversation = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, BartForConditionalGeneration

        from dialogue_benchmarks import llm  # noqa: F401 -- HF_HOME before downloads

        path = Path(self.checkpoint) if self.checkpoint else (
            DATA_ROOT / "checkpoints" / "ss_mpc" / self.base_model.replace("/", "__") / "final.ckpt"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"ss_mpc: no trained checkpoint at {path} -- run "
                f"`.venv/bin/python plugins/methods/ss_mpc/train.py --base {self.base_model}` first"
            )
        tok = AutoTokenizer.from_pretrained(self.base_model)
        tok.add_tokens(ADD_TOKENS)
        model = BartForConditionalGeneration.from_pretrained(self.base_model, dtype=torch.float32)
        model.resize_token_embeddings(len(tok))
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        # the Lightning module wraps the HF model as `self.model`
        model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}, strict=False)
        self._model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self._tok = tok
        self._torch = torch

    def prepare(self, conversation) -> None:
        self._load()
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        conv = self._conversation
        raw = getattr(conv, "raw", None)
        if not raw:
            raise ValueError("ss_mpc: conversation has no raw GSN fields (relation_at, ctx_spk, ...) -- use ubuntu_irc_gen")
        text = build_input(raw, [t.text for t in conv.turns], lambda t: len(self._tok.tokenize(t)) <= 1024 - 2)
        enc = self._tok([text], return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_length=50)
        return MethodOutput(answer=self._tok.decode(out[0], skip_special_tokens=True).strip())

    def reset(self) -> None:
        self._conversation = None
