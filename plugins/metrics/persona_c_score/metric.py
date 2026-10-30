"""persona_c_score -- NLI-classifier C-score against the speaker's persona.

score(item, prediction) = sum over persona sentences p of NLI(prediction, p):
+1 entailment, -1 contradiction, 0 neutral (premise = the response,
hypothesis = the persona sentence, as in Madotto et al. 2019). Papers fine-tune
the classifier on DNLI: when train_dnli.py's checkpoint exists
(DATA_ROOT/checkpoints/persona_c_score_dnli/) it is used; otherwise the
off-the-shelf FacebookAI/roberta-large-mnli stands in (noisier on chit-chat).
Loaded once per process on cuda:0 when available.
"""
from dialogue_benchmarks._paths import DATA_ROOT

_DNLI_CKPT = DATA_ROOT / "checkpoints" / "persona_c_score_dnli"
_MODEL = str(_DNLI_CKPT) if (_DNLI_CKPT / "config.json").exists() else "FacebookAI/roberta-large-mnli"
_state = {}


def _classifier():
    if "model" not in _state:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from dialogue_benchmarks import llm  # noqa: F401 -- sets HF_HOME before transformers downloads

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL).to(device).eval()
        labels = {i: l.lower() for i, l in model.config.id2label.items()}
        _state.update(model=model, tok=tok, device=device, labels=labels, torch=torch)
    return _state


def score(item, prediction) -> float:
    persona = (item.metadata or {}).get("persona") or []
    if isinstance(persona, str):
        persona = [persona]
    response = str(prediction).strip()
    if not persona or not response:
        return 0.0
    st = _classifier()
    torch = st["torch"]
    batch = st["tok"]([response] * len(persona), list(persona), return_tensors="pt", padding=True, truncation=True, max_length=256).to(st["device"])
    with torch.no_grad():
        pred = st["model"](**batch).logits.argmax(-1).tolist()
    value = {"entailment": 1.0, "contradiction": -1.0}
    return sum(value.get(st["labels"][p], 0.0) for p in pred)
