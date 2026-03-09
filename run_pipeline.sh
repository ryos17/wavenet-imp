#!/usr/bin/env bash
set -euo pipefail

# Fixed args (edit as needed)
MODEL_CFG="model_cfg/ch16_ungated.json"

PRUNE_TYPES=("global" "local")
PRUNE_SCHEDULES=("exponential" "linear")

for PRUNE_TYPE in "${PRUNE_TYPES[@]}"; do
  for PRUNE_SCHEDULE in "${PRUNE_SCHEDULES[@]}"; do
    echo "==============================================="
    echo "Running with prune_type=${PRUNE_TYPE}, prune_schedule=${PRUNE_SCHEDULE}"
    echo "==============================================="

    python train_imp.py \
      --model_cfg "${MODEL_CFG}" \
      --prune_type "${PRUNE_TYPE}" \
      --prune_schedule "${PRUNE_SCHEDULE}"
  done
done