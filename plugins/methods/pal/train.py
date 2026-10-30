#!/usr/bin/env python3
"""Train PAL (see method.py) on PersonaChat's train split:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python plugins/methods/pal/train.py \\
        [--base meta-llama/Llama-3.1-8B-Instruct] [--sft_epochs 3] [--dpo_epochs 3] [--dpo_examples 20000]

Stages (outputs under DATA_ROOT/checkpoints/pal/<base>/):
  0. labels.json   persona-relevance label per training example
  1. stage1_sft/   LoRA (r=16, alpha=32, PAL's config) SFT on the select +
                   generate mixture, lr 2e-5
  2. dpo_pairs.json  rejected responses: the stage-1 model answering with
                   PAL's "generate_for_dpo" (no persona) prompt, beam 3
  3. stage2_dpo/   DPO LoRA on the merged stage-1 model, lr 1e-6, beta 0.1;
                   the reference is the stage-1 model (adapter disabled)

Deviation from PAL, data only: PAL's training files carry a hand-prepared
"relevant persona" index per example that the repository doesn't ship.
Stage 0 derives it: the persona sentence the gold response most strongly
entails under the C-score NLI classifier (persona_c_score's model, DNLI-tuned
when available), if its entailment probability exceeds 0.5, else none.
Everything else -- prompts, LoRA shape, learning rates, beta, decoding --
follows PAL's scripts/llama3.sh. DPO runs on a --dpo_examples subset of the
train split (PAL generated its negatives over its own split files).
"""
import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dialogue_benchmarks import llm  # noqa: E402,F401 -- HF_HOME / token setup before transformers
from dialogue_benchmarks.datasets.registry import _import_adapter, get_dataset_info  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("_pal_method", Path(__file__).with_name("method.py"))
P = M.P


def train_examples() -> list:
    adapter = _import_adapter("personachat")
    convs = adapter.load_split(get_dataset_info("personachat").raw_path, "train")
    out = []
    for conv in convs:
        for item in adapter.eval_items(conv):
            if item.reference:
                out.append({"persona": list(item.metadata["persona"]), "history": list(item.metadata["history"]), "response": item.reference})
    return out


def persona_labels(examples: list, cache: Path) -> list:
    if cache.exists():
        return json.loads(cache.read_text())
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cscore = _load("_pc", ROOT / "plugins" / "metrics" / "persona_c_score" / "metric.py")
    tok = AutoTokenizer.from_pretrained(cscore._MODEL)
    clf = AutoModelForSequenceClassification.from_pretrained(cscore._MODEL).cuda().eval()
    ent = next(i for i, l in clf.config.id2label.items() if l.lower() == "entailment")
    pairs = [(i, j, ex["response"], p) for i, ex in enumerate(examples) for j, p in enumerate(ex["persona"])]
    best = [(-1, 0.0)] * len(examples)
    for b in range(0, len(pairs), 512):
        chunk = pairs[b:b + 512]
        enc = tok([c[2] for c in chunk], [c[3] for c in chunk], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            probs = clf(**enc).logits.float().softmax(-1)[:, ent].tolist()
        for (i, j, _, _), pr in zip(chunk, probs):
            if pr > best[i][1]:
                best[i] = (j, pr)
    labels = [j if pr > 0.5 else -1 for j, pr in best]
    cache.write_text(json.dumps(labels))
    return labels


def _lora_model(base: str, stage1: Path = None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    if stage1 is not None:
        model = PeftModel.from_pretrained(model, str(stage1)).merge_and_unload()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # PAL: LoraConfig(r=16, lora_alpha=32, init_lora_weights="gaussian"), peft's default target modules
    return get_peft_model(model, LoraConfig(r=16, lora_alpha=32, init_lora_weights="gaussian", task_type="CAUSAL_LM"))


def _encode(tok, prompt: str, target: str, max_len: int = 1024):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target + "<|eot_id|>", add_special_tokens=False)["input_ids"]
    p = p[-(max_len - len(t)):]
    return p + t, [-100] * len(p) + t


def _batch(tok, rows):
    import torch

    width = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), width), tok.pad_token_id)
    lab = torch.full((len(rows), width), -100)
    att = torch.zeros((len(rows), width), dtype=torch.long)
    for k, (i, l) in enumerate(rows):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return ids.cuda(), lab.cuda(), att.cuda()


def stage1_sft(base, tok, examples, labels, out, epochs, lr, bs, accum):
    import torch
    from transformers import get_linear_schedule_with_warmup

    data = []
    for ex, lab in zip(examples, labels):
        related = P.persona_label_text(ex["persona"], lab)
        data.append((P.build("select", ex["persona"], ex["history"]), related))
        data.append((P.build("generate", ex["persona"], ex["history"], related), ex["response"]))
    model = _lora_model(base)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total = max(1, int(math.ceil(len(data) / (bs * accum)) * epochs))
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * total), total)
    model.train()
    step = micro = 0
    start = time.time()
    rng = random.Random(42)
    while step < total:
        rng.shuffle(data)
        for b in range(0, len(data), bs):
            ids, lab, att = _batch(tok, [_encode(tok, p, t) for p, t in data[b:b + bs]])
            (model(input_ids=ids, attention_mask=att, labels=lab).loss / accum).backward()
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"[sft] step {step}/{total} eta {(total - step) * (time.time() - start) / step / 60:.0f}m", flush=True)
                if step >= total:
                    break
    model.save_pretrained(out)
    del model
    torch.cuda.empty_cache()


def make_dpo_pairs(base, tok, stage1, examples, labels, n, cache: Path) -> list:
    if cache.exists():
        return json.loads(cache.read_text())
    import torch

    model, _ = M.load_model(base, stage1)
    rng = random.Random(7)
    picked = rng.sample(range(len(examples)), min(n, len(examples)))
    pairs = []
    for k, i in enumerate(picked):
        ex = examples[i]
        rejected = M.generate(model, tok, P.build("generate_for_dpo", ex["persona"], ex["history"])).split("\n")[0].strip()
        related = P.persona_label_text(ex["persona"], labels[i])
        prompt = P.build("generate", ex["persona"], ex["history"], related)  # PAL: golden_select_then_generate
        if rejected and rejected != ex["response"]:
            pairs.append({"prompt": prompt, "chosen": ex["response"], "rejected": rejected})
        if k % 200 == 0:
            print(f"[dpo-gen] {k}/{len(picked)}", flush=True)
    cache.write_text(json.dumps(pairs))
    del model
    torch.cuda.empty_cache()
    return pairs


def stage2_dpo(base, tok, stage1, pairs, out, epochs, lr, beta, bs, accum):
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup

    model = _lora_model(base, stage1)  # policy = merged stage-1 + fresh LoRA; reference = adapter disabled

    def logps(ids, lab, att):
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        mask = tgt != -100
        lp = torch.gather(logits.log_softmax(-1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (lp * mask).sum(-1)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total = max(1, int(math.ceil(len(pairs) / (bs * accum)) * epochs))
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * total), total)
    model.train()
    step = micro = 0
    start = time.time()
    rng = random.Random(42)
    while step < total:
        rng.shuffle(pairs)
        for b in range(0, len(pairs), bs):
            chunk = pairs[b:b + bs]
            rows = [_encode(tok, c["prompt"], c["chosen"]) for c in chunk] + [_encode(tok, c["prompt"], c["rejected"]) for c in chunk]
            ids, lab, att = _batch(tok, rows)
            pol = logps(ids, lab, att)
            with torch.no_grad(), model.disable_adapter():
                ref = logps(ids, lab, att)
            n = len(chunk)
            margin = (pol[:n] - ref[:n]) - (pol[n:] - ref[n:])
            loss = -F.logsigmoid(beta * margin).mean()
            (loss / accum).backward()
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    acc = (margin > 0).float().mean().item()
                    print(f"[dpo] step {step}/{total} loss {loss.item():.4f} reward_acc {acc:.2f} eta {(total - step) * (time.time() - start) / step / 60:.0f}m", flush=True)
                if step >= total:
                    break
    model.save_pretrained(out)


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sft_epochs", type=float, default=3)
    ap.add_argument("--dpo_epochs", type=float, default=3)
    ap.add_argument("--dpo_examples", type=int, default=20000)
    ap.add_argument("--max_examples", type=int, default=None, help="cap the train split (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or M.run_dir(args.base)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    examples = train_examples()
    if args.max_examples:
        examples = examples[: args.max_examples]
    labels = persona_labels(examples, out / "labels.json")
    print(f"{len(examples)} examples, {sum(l >= 0 for l in labels)} with a relevant persona", flush=True)

    stage1 = out / "stage1_sft"
    if not (stage1 / "adapter_config.json").exists():
        stage1_sft(args.base, tok, examples, labels, stage1, args.sft_epochs, 2e-5, args.batch_size, args.grad_accum)
    pairs = make_dpo_pairs(args.base, tok, stage1, examples, labels, args.dpo_examples, out / "dpo_pairs.json")
    print(f"{len(pairs)} DPO pairs", flush=True)
    stage2_dpo(args.base, tok, stage1, pairs, out / "stage2_dpo", args.dpo_epochs, 1e-6, 0.1, args.batch_size // 2 or 1, args.grad_accum)
    (out / "train_meta.json").write_text(json.dumps({**vars(args), "out": str(out), "n_examples": len(examples), "n_dpo_pairs": len(pairs)}, default=str, indent=2))
    print(out)


if __name__ == "__main__":
    main()
