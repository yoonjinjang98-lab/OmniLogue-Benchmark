#!/usr/bin/env python3
"""Train SS-MPC (Jang et al., 2025) with its own code (vendor/ss_mpc/, run
unmodified): stage 1 encoder post-training (masked utterance-index /
speaker-token recovery), stage 2 response-generation fine-tuning from the
stage-1 weights -- the two commands in SS-MPC's enc_post.sh / train.sh.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python plugins/methods/ss_mpc/train.py \\
        [--base facebook/bart-large] [--epochs 10] [--max_steps -1]

The vendored code reads ./data/ubuntu_data_{train,valid}.json relative to
its working directory, so each run gets a work dir under
DATA_ROOT/checkpoints/ss_mpc/<base>/ with data/ symlinked to the shared
Ubuntu IRC (Hu et al. 2019) raw files -- the same split ubuntu_irc_gen
evaluates. Only training runs here (--do_train_only); evaluation happens in
the harness (method.py on the 5,000 test instances) with the harness's
metrics. --debug disables W&B logging, as in the vendored config.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dialogue_benchmarks import llm  # noqa: E402,F401 -- HF_HOME / token for the child processes via env
from dialogue_benchmarks._paths import DATA_ROOT  # noqa: E402

VENDOR = Path(__file__).resolve().parent / "vendor" / "ss_mpc"


def run_dir(base: str) -> Path:
    return DATA_ROOT / "checkpoints" / "ss_mpc" / base.replace("/", "__")


def _newest_ckpt(directory: Path) -> Path:
    ckpts = sorted(directory.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise SystemExit(f"no checkpoint written under {directory}")
    return ckpts[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="facebook/bart-large")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=-1, help="cap per stage (smoke tests)")
    ap.add_argument("--masking_probability", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    work = args.out or run_dir(args.base)
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        link = data / f"ubuntu_data_{split}.json"
        if not link.exists():
            link.symlink_to(DATA_ROOT / "raw" / "ubuntu_irc" / f"ubuntu_data_{split}.json")

    common = [
        sys.executable, str(VENDOR / "main_emb.py"),
        "--base_model", args.base, "--epoch", str(args.epochs), "--max_steps", str(args.max_steps),
        "--debug", "--do_train_only", "--gpu_devices", "1",
    ]
    stage1 = work / "stage1_post_training"
    subprocess.run(
        common + ["--enc_post_training", "--masking_probability", str(args.masking_probability),
                  "--checkpoint_dirpath", str(stage1)],
        cwd=work, check=True,
    )
    post = _newest_ckpt(stage1)
    stage2 = work / "stage2_fine_tuning"
    subprocess.run(common + ["--post_trained_model", str(post), "--checkpoint_dirpath", str(stage2)], cwd=work, check=True)
    final = _newest_ckpt(stage2)
    (work / "final.ckpt").unlink(missing_ok=True)
    (work / "final.ckpt").symlink_to(final)
    print(f"stage1 {post}\nstage2 {final}\n-> {work / 'final.ckpt'}")


if __name__ == "__main__":
    main()
