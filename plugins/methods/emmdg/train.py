#!/usr/bin/env python3
"""Fine-tune EMMDG (Li & Zhao, "EM Pre-training for Multi-party Dialogue
Response Generation", ACL 2023, github.com/EricLee8/MPDRG) from its released
EM-pre-trained checkpoint, with its own training script (vendor/mpdrg/
myTrain.py; the one patch is noted at its AdamW import) and the settings of
its run_finetune_ubuntu.sh: 15 epochs, lr 4e-5, the addressee-indicator
model (embindicator), addr mode 0, pre-trained indicator embeddings kept.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python plugins/methods/emmdg/train.py [--epochs 15] [--small]

Work dir: DATA_ROOT/checkpoints/emmdg/work/, with data/ubuntu/{train,valid,
test}.json symlinked to the shared Ubuntu IRC raw files (byte-identical to
the repo's own data.zip) and pretrain_models/mpdrg.pth = the released
checkpoint (download: `gdown 1y6N7L13kHlkC6t-E0tgkDV2UPD5OGu-z` into
DATA_ROOT/checkpoints/emmdg/pretrain_models/). The script evaluates valid +
test with coco-caption every epoch (needs Java for METEOR) and keeps the
best-valid model as <save_path>/best_model.pth, which method.py loads.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dialogue_benchmarks import llm  # noqa: E402,F401 -- HF_HOME / token for the child process via env
from dialogue_benchmarks._paths import DATA_ROOT  # noqa: E402

VENDOR = Path(__file__).resolve().parent / "vendor" / "mpdrg"
HOME = DATA_ROOT / "checkpoints" / "emmdg"
# vendor's run_finetune_ubuntu.sh
ARGS = ["--learning_rate", "4e-5", "--model_file", "embindicator", "--addr", "0", "--del_indicator_embs", "0",
        "--pretrain_path", "pretrain_models/mpdrg.pth", "--save_path", "mpdrg_save"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--small", action="store_true", help="vendor's 100-example mode (smoke test)")
    ap.add_argument("--work", type=Path, default=HOME / "work")
    args = ap.parse_args()

    pretrained = HOME / "pretrain_models" / "mpdrg.pth"
    if not pretrained.exists():
        raise SystemExit(f"missing {pretrained} -- `gdown 1y6N7L13kHlkC6t-E0tgkDV2UPD5OGu-z -O {pretrained}`")
    work = args.work
    (work / "data" / "ubuntu").mkdir(parents=True, exist_ok=True)
    (work / "pretrain_models").mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        link = work / "data" / "ubuntu" / f"{split}.json"
        if not link.exists():
            link.symlink_to(DATA_ROOT / "raw" / "ubuntu_irc" / f"ubuntu_data_{split}.json")
    if not (work / "pretrain_models" / "mpdrg.pth").exists():
        (work / "pretrain_models" / "mpdrg.pth").symlink_to(pretrained)

    cmd = [sys.executable, "-W", "ignore", str(VENDOR / "myTrain.py"), "--store_path", str(work),
           "--epochs", str(args.epochs), "--cuda", "0", *ARGS]
    if args.small:
        cmd += ["--small", "1"]
    subprocess.run(cmd, cwd=work, check=True)
    best = sorted(work.rglob("best_model.pth"), key=lambda p: p.stat().st_mtime)
    print(best[-1] if best else "no best_model.pth written")


if __name__ == "__main__":
    main()
