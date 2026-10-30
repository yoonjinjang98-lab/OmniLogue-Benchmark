"""CLI: fine-tune the DNLI entailment scorer (paper Eq. 2's phi model, and
also the model that computes the paper's "C" consistency metric and the
stage-2 gamma gate). Needs a real DNLI sentence-pair file -- see
ltkm_lib/dnli.py's module docstring for the expected JSON schema
(confirmed field names: sentence1/sentence2/label/id).

    .venv/bin/python plugins/methods/ltkm/train_entailment_scorer.py \\
        --pairs /path/to/dnli_train.json \\
        --output-dir /path/to/checkpoints/entailment_scorer
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ltkm_lib.dnli import load_dnli_pairs
from ltkm_lib.entailment_scorer import train_entailment_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Path to a DNLI sentence-pair JSON file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="roberta-large")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    pairs = load_dnli_pairs(args.pairs)
    train_entailment_scorer(
        pairs,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Saved entailment scorer to {args.output_dir}")


if __name__ == "__main__":
    main()
