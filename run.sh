#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE=offline

run_job() {
  local script="$1"
  local model_name="$2"
  local optimizer="$3"

  echo "[$(date '+%F %T')] start ${model_name} optimizer3=${optimizer}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${script}" --optimizer3="${optimizer}"
  echo "[$(date '+%F %T')] done  ${model_name} optimizer3=${optimizer}"
}

for optimizer in coupled_muon; do
  run_job train_gpt.py dense "${optimizer}"
done

for optimizer in coupled_muon; do
  run_job train_gpt_moe.py moe "${optimizer}"
done
