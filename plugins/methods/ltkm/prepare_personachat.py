"""CLI: convert this repo's real ConvAI2 PersonaChat raw parquet
(data/raw/personachat/data/{train,validation}-00000-of-00001.parquet) into
the `{"persona": [...], "history": [...], "query": ..., "response": ...}`
JSON list `ltkm_lib/training.py::load_dialogue_examples` expects. Each
source conversation has `personality` (persona sentences) and `utterances`
(one entry per user turn, `history` = every turn up to and including that
turn, `candidates[-1]` = the gold response) -- standard ConvAI2 convention,
confirmed against the raw file directly (not assumed).

    .venv/bin/python plugins/methods/ltkm/prepare_personachat.py \\
        --parquet $DIALOGUE_BENCHMARKS_DATA/raw/personachat/data/train-00000-of-00001.parquet \\
        --output /path/to/dialogue_examples.json
"""
import argparse
import json

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    examples = []
    for _, row in df.iterrows():
        persona = list(row["personality"])
        for utt in row["utterances"]:
            history = list(utt["history"])
            if not history:
                continue
            examples.append(
                {
                    "persona": persona,
                    "history": history[:-1],
                    "query": history[-1],
                    "response": utt["candidates"][-1],
                }
            )

    with open(args.output, "w") as f:
        json.dump(examples, f)
    print(f"Wrote {len(examples)} dialogue examples to {args.output}")


if __name__ == "__main__":
    main()
