#!/usr/bin/env python3
"""Fine-tune the C-score classifier on Dialogue NLI -- the setup persona
papers' C-score uses (Welleck et al. 2019's DNLI; positive/negative/neutral =
entailment/contradiction/neutral), starting from roberta-large-mnli.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python plugins/metrics/persona_c_score/train_dnli.py [--max_train N]

Writes DATA_ROOT/checkpoints/persona_c_score_dnli/, which metric.py then
uses instead of the off-the-shelf MNLI model. Reports DNLI test accuracy.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dialogue_benchmarks import llm  # noqa: E402,F401 -- HF_HOME / token setup before transformers
from dialogue_benchmarks._paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "checkpoints" / "persona_c_score_dnli"
BASE = "FacebookAI/roberta-large-mnli"
# roberta-large-mnli's head: 0 contradiction, 1 neutral, 2 entailment
DNLI_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


def _pairs(split: str, limit=None):
    spec = importlib.util.spec_from_file_location("_dnli", ROOT / "plugins" / "datasets" / "dnli" / "adapter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pairs = mod._load_split(DATA_ROOT / "raw" / "dnli", split)
    pairs = [p for p in pairs if p.label in DNLI_TO_ID]
    return pairs[:limit] if limit else pairs


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    ap = argparse.ArgumentParser()
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE).cuda()

    def collate(batch):
        enc = tok([p.premise for p in batch], [p.hypothesis for p in batch], return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc["labels"] = torch.tensor([DNLI_TO_ID[p.label] for p in batch])
        return enc

    train = DataLoader(_pairs("train", args.max_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    for epoch in range(args.epochs):
        for i, batch in enumerate(train):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if i % 500 == 0:
                print(f"epoch {epoch} step {i}/{len(train)} loss {loss.item():.4f}", flush=True)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in DataLoader(_pairs("test"), batch_size=128, collate_fn=collate):
            batch = {k: v.cuda() for k, v in batch.items()}
            pred = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item()
            total += len(pred)
    print(f"DNLI test accuracy: {correct / total:.4f} ({total} pairs)")
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
