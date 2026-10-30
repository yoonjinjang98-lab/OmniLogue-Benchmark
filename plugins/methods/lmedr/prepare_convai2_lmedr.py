"""CLI: convert this repo's real ConvAI2 PersonaChat raw parquet
(data/raw/personachat/data/{train,validation}-00000-of-00001.parquet) into
upstream LMEDR's own `train_self_original.txt`/`valid_self_original.txt`
format, which `build_data_PersonaChat.py::create_data` parses: `N your
persona: <sentence>` lines, then `N <query>\\t<response>\\t\\t<cand1>|
<cand2>|...|<cand20>` lines per turn (candidates pipe-joined, gold response
last -- `build_dataloader` relies on `cand[:-1]` being the negative pool).
Confirmed by reading `create_data`/`build_dataloader` directly, not assumed
from upstream's README.

Unlike `plugins/methods/bob/prepare_convai2_txt.py` (which drops
candidates entirely, since BoB's own dataloader never needs them), LMEDR's
own training needs the full per-turn 20-candidate list intact -- this
repo's HF parquet mirror already carries it as each utterance's
`candidates` field (confirmed 20 entries per turn, last one equal to the
turn's gold response).

    .venv/bin/python plugins/methods/lmedr/prepare_convai2_lmedr.py \\
        --parquet $DIALOGUE_BENCHMARKS_DATA/raw/personachat/data/train-00000-of-00001.parquet \\
        --output /path/to/train_self_original.txt
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
    n_turns = 0
    for _, row in df.iterrows():
        for i, persona_sentence in enumerate(row["personality"], start=1):
            lines.append(f"{i} your persona: {persona_sentence}")
        for utt in row["utterances"]:
            history = list(utt["history"])
            if not history:
                continue
            query = history[-1]
            candidates = list(utt["candidates"])
            response = candidates[-1]
            lines.append(f"1 {query}\t{response}\t\t{'|'.join(candidates)}")
            n_turns += 1

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(df)} conversations, {n_turns} turns to {args.output}")


if __name__ == "__main__":
    main()
