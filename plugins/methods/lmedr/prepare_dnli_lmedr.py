"""CLI: convert this repo's already-downloaded DNLI data (`data/raw/dnli/
xksteven_dialogue_nli_{split}.jsonl`, real JSONL with `premise`/`hypothesis`/
`label` fields, `label` in {"positive","negative","neutral"} -- same file
`plugins/methods/ltkm/prepare_dnli.py` didn't need triples from) into
upstream LMEDR's own `dialogue_nli_train.jsonl` shape: a single JSON *array*
(confirmed by reading `build_data_PersonaChat.py::build_infer_dataset`,
which calls `json.load(f)`, not line-by-line) of
`{"sentence1": ..., "sentence2": ..., "label": ...}` objects. Only
`label == "positive"` rows are actually used by `build_infer_dataset` (it
skips "neutral"/"negative"), but this script converts every row so the
same file can be reused if that ever changes.

    .venv/bin/python plugins/methods/lmedr/prepare_dnli_lmedr.py \\
        --input $DIALOGUE_BENCHMARKS_DATA/raw/dnli/xksteven_dialogue_nli_train.jsonl \\
        --output /path/to/dialogue_nli_train.jsonl
"""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"sentence1": obj["premise"], "sentence2": obj["hypothesis"], "label": obj["label"]})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    n_positive = sum(1 for r in rows if r["label"] == "positive")
    print(f"Wrote {len(rows)} pairs ({n_positive} positive) to {args.output}")


if __name__ == "__main__":
    main()
