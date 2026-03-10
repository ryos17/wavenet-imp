#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_pipeline.sh <DATA_ROOT>
# Example: bash run_pipeline.sh data_1
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DATA_ROOT>" >&2
  exit 1
fi
DATA_ROOT="$1"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: not a directory: ${DATA_ROOT}" >&2
  exit 1
fi

# Fixed args (edit as needed)
MODEL_CFG="model_cfg/ch16_ungated.json"

# Collect all input.wav / <name>.wav pairings under DATA_ROOT
# Example:
#   data_1/vBeat_fdr_high_bass/input.wav
#   data_1/vBeat_fdr_high_bass/vBeat_fdr_high_bass.wav
shopt -s nullglob
PAIRINGS=()
for input_wav in "${DATA_ROOT}"/*/input.wav; do
  [[ -f "$input_wav" ]] || continue
  dir=$(dirname "$input_wav")
  name=$(basename "$dir")
  target_wav="${dir}/${name}.wav"
  if [[ -f "$target_wav" ]]; then
    PAIRINGS+=("${input_wav}|${target_wav}")
  else
    echo "WARN: skip ${dir}: missing ${target_wav}" >&2
  fi
done
shopt -u nullglob

if [[ ${#PAIRINGS[@]} -eq 0 ]]; then
  echo "No pairings found under ${DATA_ROOT} (need */input.wav and */<dirname>.wav)." >&2
  exit 1
fi

echo "Found ${#PAIRINGS[@]} pairing(s) under ${DATA_ROOT}"

for pairing in "${PAIRINGS[@]}"; do
  INPUT_WAV="${pairing%%|*}"
  TARGET_WAV="${pairing#*|}"
  echo "-----------------------------------------------"
  echo "Pair: ${INPUT_WAV}  ->  ${TARGET_WAV}"
  echo "-----------------------------------------------"

  python train_imp.py \
    --model_cfg "${MODEL_CFG}" \
    --input_wav "${INPUT_WAV}" \
    --target_wav "${TARGET_WAV}"
done
