#!/usr/bin/env python3
"""Generate EMMDG responses for the Ubuntu IRC test split with the vendored
code's own inference path (myTrain.evaluate(..., cal_metrics=False):
get_dataset features, addressee indicator embeddings, beam 4, max_length 50,
min_length 5, no_repeat_ngram_size 5) from a fine-tuned best_model.pth.
Writes {qid: response} JSON; method.py runs this once and reads the file.

    .venv/bin/python plugins/methods/emmdg/predict.py --work <train work dir> --best <best_model.pth> --out <json>

Runs as its own process because the vendored modules are named `utils` /
`models` and parse sys.argv at import -- not safe to load into the harness.
"""
import argparse
import json
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor" / "mpdrg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--best", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from dialogue_benchmarks import llm  # noqa: F401 -- HF_HOME before downloads

    # same arguments as train.py's fine-tuning run (vendor run_finetune_ubuntu.sh)
    sys.argv = [sys.argv[0], "--store_path", str(a.work), "--cuda", "0", "--learning_rate", "4e-5",
                "--model_file", "embindicator", "--addr", "0", "--del_indicator_embs", "0",
                "--pretrain_path", "pretrain_models/mpdrg.pth", "--save_path", "mpdrg_save"]
    sys.path.insert(0, str(VENDOR))
    import torch
    from torch.utils.data import DataLoader, SequentialSampler

    import myTrain as T  # noqa: E402

    tok = T.tokenizer_class.from_pretrained(T.args.model_name, cache_dir=T.args.cache_path)
    tok.truncation_side = "left"
    config = T.config_class.from_pretrained(T.args.model_name, cache_dir=T.args.cache_path)
    model = T.GenerationModel.from_pretrained(T.args.model_name, config=config, cache_dir=T.args.cache_path)
    model.load_state_dict(torch.load(a.best, map_location="cpu"))
    model = model.to(T.device)
    test = T.get_dataset(T.test_path, T.args.cache_path, tok, training=False)
    loader = DataLoader(test, sampler=SequentialSampler(test), batch_size=T.args.batch_size, collate_fn=T.collate_fn)
    name = "harness_test_predictions.json"
    T.evaluate(model, loader, tok, cur_best_result=None, is_test=True, file_name=name, cal_metrics=False)
    preds = json.loads((Path(T.args.save_path) / name).read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(preds))
    print(f"{len(preds)} predictions -> {a.out}")


if __name__ == "__main__":
    main()
