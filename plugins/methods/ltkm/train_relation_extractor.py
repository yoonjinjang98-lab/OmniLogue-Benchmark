"""CLI: fine-tune the persona-sentence -> DNLI-relation classifier (paper
Sec. "Persona-aware Question Construction Module"). Needs a real DNLI
persona-triple annotation file -- see ltkm_lib/dnli.py's module docstring
for the expected (undocumented-upstream, best-effort) JSON schema.

    .venv/bin/python plugins/methods/ltkm/train_relation_extractor.py \\
        --triples /path/to/dnli_triples.json \\
        --output-dir /path/to/checkpoints/relation_extractor
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ltkm_lib.dnli import load_persona_triples
from ltkm_lib.relation_extractor import train_relation_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", required=True, help="Path to a DNLI persona-triple JSON file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="roberta-large")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    triples = load_persona_triples(args.triples)
    train_relation_extractor(
        triples,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Saved relation extractor to {args.output_dir}")


if __name__ == "__main__":
    main()
