#!/usr/bin/env bash
# Train every generation baseline that needs weights (added 2026-10-16; see
# docs/generation_baselines.md), as two sequential lanes, one per GPU:
#
#   scripts/train_generation_baselines.sh gpu0    # C-score classifier, PAL, llm_sft x {personachat, msc_longterm, cc, mpchat_gen}
#   scripts/train_generation_baselines.sh gpu1    # llm_sft x ubuntu_irc_gen, SS-MPC, EMMDG
#
# Each job logs to DATA_ROOT/checkpoints/logs/<job>.log and is skipped if its
# output already exists, so a lane can be re-run after an interruption.
# Measured throughput (Qwen3-8B LoRA, RTX PRO 6000): ~670 examples/min.
# Rough wall time: gpu0 ~45 h, gpu1 ~35 h. In the lab, GPU 1 also hosts the
# Qwen3-Embedding gateway (port 8008) -- hypermem / Qwen3-embedding runs
# can't start while lane gpu1 holds it.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DATA="${DIALOGUE_BENCHMARKS_DATA:-$PWD/data}"
CK="$DATA/checkpoints"
LOGS="$CK/logs"
mkdir -p "$LOGS"
PY=.venv/bin/python

job() {  # job <name> <done-marker> <command...>
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then echo "[skip] $name ($marker exists)"; return; fi
  echo "[start] $name $(date -u +%FT%TZ) -> $LOGS/$name.log"
  if "$@" >"$LOGS/$name.log" 2>&1; then
    echo "[done]  $name $(date -u +%FT%TZ)"
  else
    echo "[FAIL]  $name -- see $LOGS/$name.log"
  fi
}

case "${1:-}" in
  gpu0)
    export CUDA_VISIBLE_DEVICES=0
    job persona_c_score_dnli "$CK/persona_c_score_dnli/config.json" \
      $PY plugins/metrics/persona_c_score/train_dnli.py
    job pal_llama31_8b "$CK/pal/meta-llama__Llama-3.1-8B-Instruct/stage2_dpo/adapter_config.json" \
      $PY plugins/methods/pal/train.py --base meta-llama/Llama-3.1-8B-Instruct
    for ds in personachat msc_longterm mpchat_gen; do
      job "llm_sft_${ds}_qwen3_8b" "$CK/llm_sft/$ds/Qwen__Qwen3-8B/adapter_config.json" \
        $PY plugins/methods/llm_sft/train.py --dataset "$ds" --base Qwen/Qwen3-8B
    done
    job llm_sft_conversation_chronicles_qwen3_8b "$CK/llm_sft/conversation_chronicles/Qwen__Qwen3-8B/adapter_config.json" \
      $PY plugins/methods/llm_sft/train.py --dataset conversation_chronicles --base Qwen/Qwen3-8B --max_examples 100000
    ;;
  gpu1)
    export CUDA_VISIBLE_DEVICES=1
    job llm_sft_ubuntu_irc_gen_qwen3_8b "$CK/llm_sft/ubuntu_irc_gen/Qwen__Qwen3-8B/adapter_config.json" \
      $PY plugins/methods/llm_sft/train.py --dataset ubuntu_irc_gen --base Qwen/Qwen3-8B
    job ss_mpc_bart_large "$CK/ss_mpc/facebook__bart-large/final.ckpt" \
      $PY plugins/methods/ss_mpc/train.py --base facebook/bart-large
    job emmdg "$CK/emmdg/work/ubuntu_saves" \
      $PY plugins/methods/emmdg/train.py
    ;;
  *)
    echo "usage: $0 gpu0|gpu1" >&2; exit 2 ;;
esac
