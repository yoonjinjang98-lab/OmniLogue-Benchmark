"""CLI: build the two real-data JSON files `train_entailment_scorer.py`/
`train_relation_extractor.py` expect, from the `pietrolesci/dialogue_nli`
HuggingFace mirror of DNLI (Welleck et al. 2019) -- the only DNLI mirror
found (checked `xksteven/dialogue_nli`, already used by
`plugins/datasets/dnli`, and `pietrolesci/dialogue_nli`) that preserves the
`triple1`/`triple2` (e1, relation, e2) annotations `ltkm_lib/dnli.py`'s
`PersonaTriple` schema needs -- `xksteven/dialogue_nli` (the one already on
this repo's shared HDD raw data) only has the derived premise/hypothesis/
label pairs, not the triples, so it can't drive relation-extractor training
on its own.

    .venv/bin/python plugins/methods/ltkm/prepare_dnli.py \\
        --output-dir /path/to/prepared_dnli
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

_NON_RELATIONS = {"<none>", "<blank>", "other"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", help="pietrolesci/dialogue_nli split to use")
    args = parser.parse_args()

    path = hf_hub_download(
        "pietrolesci/dialogue_nli", f"data/{args.split}-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path)

    pairs = [
        {
            "id": str(row["id"]),
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "label": row["original_label"],
        }
        for _, row in df.iterrows()
    ]

    # Persona-sentence -> relation triples: every (sentence, triple) side of
    # every pair is itself a standalone (persona sentence, e1, relation, e2)
    # annotation, per DNLI's own construction -- dedup on
    # (persona, relation) since the same sentence recurs across many pairs.
    seen: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
    for _, row in df.iterrows():
        for sentence_key, triple_key in (("sentence1", "triple1"), ("sentence2", "triple2")):
            e1, relation, e2 = row[triple_key]
            if relation in _NON_RELATIONS:
                continue
            sentence = row[sentence_key]
            key = (sentence, relation)
            if key not in seen:
                seen[key] = {"persona": sentence, "e1": e1, "relation": relation, "e2": e2}
    triples = list(seen.values())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dnli_pairs.json").write_text(json.dumps(pairs))
    (out_dir / "persona_triples.json").write_text(json.dumps(triples))
    print(f"Wrote {len(pairs)} DNLI pairs and {len(triples)} persona triples to {out_dir}")


if __name__ == "__main__":
    main()
