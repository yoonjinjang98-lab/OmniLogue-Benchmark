"""CLI: build upstream BoB's `nli_positive.tsv`/`nli_negative.tsv` (used for
its decoder2 unlikelihood-training objective) from real MNLI -- upstream's
README lists MNLI (not DNLI) as the *training* NLI source, "entailment &
contradiction" specifically (DNLI is listed for *evaluation* only). MNLI's
official distribution is a standalone zip; `nyu-mll/glue`'s `mnli` config on
HuggingFace Hub is the same data (392,702 train pairs, `label` in
{0: entailment, 1: neutral, 2: contradiction}), fetched here instead since
it's already a maintained, versioned mirror. `dataloader.py::read_nli_split`
just needs `premise\\thypothesis` per line, no header -- confirmed by
reading the parser directly.

    .venv/bin/python plugins/methods/bob/prepare_mnli.py \\
        --output-dir /path/to/data/ConvAI2
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ds = load_dataset("nyu-mll/glue", "mnli", split="train")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = 0
    with open(out_dir / "nli_positive.tsv", "w", encoding="utf-8") as pos_f, open(
        out_dir / "nli_negative.tsv", "w", encoding="utf-8"
    ) as neg_f:
        for row in ds:
            premise = row["premise"].replace("\t", " ").replace("\n", " ")
            hypothesis = row["hypothesis"].replace("\t", " ").replace("\n", " ")
            if row["label"] == 0:  # entailment
                pos_f.write(f"{premise}\t{hypothesis}\n")
                n_pos += 1
            elif row["label"] == 2:  # contradiction
                neg_f.write(f"{premise}\t{hypothesis}\n")
                n_neg += 1

    print(f"Wrote {n_pos} positive (entailment) and {n_neg} negative (contradiction) pairs to {out_dir}")


if __name__ == "__main__":
    main()
