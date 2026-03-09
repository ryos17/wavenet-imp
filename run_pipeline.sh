#!/usr/bin/env bash
set -euo pipefail

# Fixed args (edit as needed)
MODEL_CFG="model_cfg/ch16_ungated.json"

for PRUNE_END in 250 500 750 1000 1250; do
  echo "==============================================="
  echo "Running with prune_end_epoch=${PRUNE_END}"
  echo "==============================================="

  python train_imp.py \
    --model_cfg "${MODEL_CFG}" \
    --prune_end_epoch "${PRUNE_END}"
done