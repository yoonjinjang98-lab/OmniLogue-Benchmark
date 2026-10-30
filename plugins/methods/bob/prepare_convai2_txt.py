"""CLI: convert this repo's real ConvAI2 PersonaChat raw parquet
(data/raw/personachat/data/{train,validation}-00000-of-00001.parquet) into
upstream BoB's own `train_self_original_no_cands.txt`-style format, which
`dataloader.py::read_convai2_split` parses: `N your persona: <sentence>`
lines followed by `N <query>\\t<response>` lines per turn. The leading
number `N` and the exact `your persona: ` prefix offset (`line[16:]`) are
the only things the parser actually keys off -- confirmed by reading
`read_convai2_split` directly, not assumed from upstream's README.

    .venv/bin/python plugins/methods/bob/prepare_convai2_txt.py \\
        --parquet $DIALOGUE_BENCHMARKS_DATA/raw/personachat/data/train-00000-of-00001.parquet \\
        --output /path/to/train_self_original_no_cands.txt
"""
import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    lines = []
    n_examples = 0
    for _, row in df.iterrows():
        for i, persona_sentence in enumerate(row["personality"], start=1):
            lines.append(f"{i} your persona: {persona_sentence}")
        for utt in row["utterances"]:
            history = list(utt["history"])
            if not history:
                continue
            query = history[-1]
            response = utt["candidates"][-1]
            lines.append(f"1 {query}\t{response}")
            n_examples += 1

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(df)} conversations, {n_examples} turns to {args.output}")


if __name__ == "__main__":
    main()
