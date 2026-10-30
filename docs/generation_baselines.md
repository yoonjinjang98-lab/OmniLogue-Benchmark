# Generation baselines (multi-party · long-term · persona)

Added 2026-10-16 from a survey of the 2024–2026 literature. The goal was to compare
response *generation* (BLEU-1~4 / METEOR / ROUGE-L, plus persona metrics) against
current baselines, not only the HeterMPC/MADNet-era ones. Published numbers for
every row, including methods without public code, are in each dataset's
`reported_results.json`. Those numbers are for reference only, because papers use
different evaluation toolkits.

## Datasets and protocols

| Axis | Dataset | Protocol |
|---|---|---|
| Multi-party | `ubuntu_irc_gen` | Ubuntu IRC (Hu et al. 2019), 5,000 test instances. Input: context + speaker + addressee. |
| Multi-party + persona | `mpchat_gen` | MPCHAT NRP test instances, 2,301. Input: context + persona post titles. |
| Long-term | `msc_longterm` | MSC test, 501 dialogues × target sessions 2–5. Earlier sessions are memory. The agent is Speaker 2 (LD-Agent / HAHT protocol). |
| Long-term | `conversation_chronicles` | Same protocol, first 501 test episodes (`CC_MAX_EPISODES`). |
| Persona | `personachat` | Next turn, given persona + history. |

## Baselines

| Baseline | Axis | Source | How it's run | Training |
|---|---|---|---|---|
| `llm_sft` | all | "LLM + SFT", the standard 2025–26 comparison point (DRCR ACL'26, LD-Agent NAACL'25, PAL'25) | LoRA SFT of Qwen3-8B (default) / Llama-3.2-3B / Qwen3-4B / Llama-3.1-8B; one prompt shared by train and test | `plugins/methods/llm_sft/train.py --dataset D --base B` |
| `ld_agent` | long-term | LD-Agent, NAACL 2025 (MIT) | prompts + retrieval scoring vendored verbatim; any LLM | none (zero-shot) |
| `ss_mpc` | multi-party | SS-MPC (Jang et al. 2025), cited in DRCR's table | the authors' code, vendored and run unmodified (post-training → fine-tuning) | `plugins/methods/ss_mpc/train.py` |
| `emmdg` | multi-party | EMMDG, ACL 2023 | released EM-pre-trained checkpoint + the authors' code (on the vendored transformers 4.51) | `plugins/methods/emmdg/train.py` |
| `pal` | persona | PAL (arXiv 2511.10215) | two-stage LoRA SFT + DPO, select-then-generate; prompts vendored | `plugins/methods/pal/train.py` |
| `persona_prompt` / `no_persona` | all | zero-/few-shot LLM prompting (the setting in DRCR's appendix and PAL's tables) | any LLM via `--model` | none |

**Reported only (no public code):** RL-TRC (EMNLP'24), DRCR (ACL'26, current best on
Ubuntu IRC), MuPaS (IJCNN'25), and SA-LLM (preprint; it uses a different Ubuntu
release).

**Not added:**
- COMEDY: Chinese-only data.
- HeterMPC / MADNet: no checkpoints, a 2022 environment, and EMMDG and SS-MPC already
  cover graph-free and EM-based approaches from the same line.
- ReBot: checkpoint availability is unverified.

## Metrics

- `bleu_n`: corpus BLEU-1~4 (nltk; multiply by 100 to compare with paper tables).
- `rouge_l`, `meteor`, `distinct_n`.
- `persona_c_score`: the C-score from an NLI classifier, as the persona literature
  reports it. A DNLI-fine-tuned RoBERTa is used once
  `plugins/metrics/persona_c_score/train_dnli.py` has run; until then the metric
  falls back to roberta-large-mnli.
- `persona_consistency`: the LLM-judged variant.

## Deviations from the originals

All deviations are recorded in each plugin's docstring.

- **PAL:** persona-relevance labels are derived with the NLI classifier, because
  PAL's labelled files aren't released.
- **EMMDG:**
  - per-epoch model selection scores with coco Bleu/Rouge directly, because its
    METEOR subprocess deadlocks;
  - transformers imports are repointed to the vendored 4.51 snapshot.
- **SS-MPC:** test-time speaker IDs use the identity permutation instead of a random
  one, to keep output deterministic. The vendored loader's own `test` stage reads
  `ubuntu_data_valid.json`; the harness evaluates on the 5,000 test instances.
- **LD-Agent:** there is no chromadb (with ≤ 4 memories, the noun-overlap pick is
  identical). MSC's HF release has no session gaps, so 1 week is assumed.

## Training

`scripts/train_generation_baselines.sh gpu0|gpu1` runs everything as two
resumable lanes. Expect about 70 GPU-hours in total.
