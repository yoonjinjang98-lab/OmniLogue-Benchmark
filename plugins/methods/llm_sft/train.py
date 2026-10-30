#!/usr/bin/env python3
"""LoRA SFT for llm_sft: fine-tune an open LLM on a generation dataset's
*train* split, with exactly the prompt method.py uses at test time.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python plugins/methods/llm_sft/train.py \\
        --dataset ubuntu_irc_gen --base Qwen/Qwen3-8B [--max_examples N] [--epochs 1]

Datasets: any task_type="dialogue" adapter exposing load_split(raw_dir,
"train") -- ubuntu_irc_gen, mpchat_gen, personachat, msc_longterm,
conversation_chronicles. Training pairs = (build_messages(conversation,
item.metadata), item.reference) over every eval item of the train split;
loss only on the response tokens.

Defaults follow the usual LoRA recipe (r=16, alpha=32, dropout 0.05 on all
attention/MLP projections; lr 2e-4, linear decay, 3% warmup; bf16; gradient
checkpointing), 1 epoch. Saves the adapter + train_meta.json to
DATA_ROOT/checkpoints/llm_sft/<dataset>/<base>/, where method.py finds it.
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

_spec = importlib.util.spec_from_file_location("_llm_sft_method", Path(__file__).with_name("method.py"))
_method = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_method)


def training_pairs(dataset: str, split: str = "train", max_examples=None, seed: int = 42) -> list:
    adapter = _import_adapter(dataset)
    if not hasattr(adapter, "load_split"):
        raise SystemExit(f"{dataset}: adapter has no load_split(raw_dir, split) -- can't read its {split} split")
    convs = adapter.load_split(get_dataset_info(dataset).raw_path, split)
    pairs = []
    for conv in convs:
        for item in adapter.eval_items(conv):
            if str(item.reference).strip():
                pairs.append((_method.build_messages(conv, item.metadata or {}), str(item.reference).strip()))
    random.Random(seed).shuffle(pairs)
    return pairs[:max_examples] if max_examples else pairs


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or _method.adapter_dir(args.dataset, args.base)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    pairs = training_pairs(args.dataset, max_examples=args.max_examples)
    print(f"{args.dataset}: {len(pairs)} training pairs", flush=True)

    def encode(messages, target):
        prompt_ids = tok(_method.chat_prompt(tok, messages), add_special_tokens=False)["input_ids"]
        target_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-(args.max_len - len(target_ids)) :]  # truncate from the left: keep the recent context
        return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(pairs) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)

    model.train()
    step, micro, start = 0, 0, time.time()
    n_micro = total_steps * args.grad_accum
    while step < total_steps:
        for b in range(0, len(pairs), args.batch_size):
            batch = [encode(m, t) for m, t in pairs[b : b + args.batch_size]]
            width = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), width), tok.pad_token_id)
            labels = torch.full((len(batch), width), -100)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            for i, (ids, lab) in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids)
                labels[i, : len(lab)] = torch.tensor(lab)
                attn[i, : len(ids)] = 1
            loss = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), labels=labels.cuda()).loss
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 20 == 0 or step == total_steps:
                    rate = micro / (time.time() - start)
                    eta = (n_micro - micro) / rate / 60
                    print(f"step {step}/{total_steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} eta {eta:.0f}m", flush=True)
                if step >= total_steps:
                    break

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_meta.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "base": args.base,
                "n_pairs": len(pairs),
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lora_r": args.lora_r,
                "max_len": args.max_len,
                "steps": total_steps,
                "minutes": round((time.time() - start) / 60, 1),
            },
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
