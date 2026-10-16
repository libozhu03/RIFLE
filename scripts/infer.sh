#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:?Usage: infer.sh <checkpoint> <images> <masks> <output> [extra args]}"
INPUT="${2:?Input image directory required}"
MASKS="${3:?Screen mask directory required}"
OUTPUT="${4:?Output directory required}"
shift 4
exec "${INFER_BIN:-${PYTHON_BIN:-python}}" infer.py \
  --pretrained_model_path "${PRETRAINED_MODEL_PATH:-preset/models/sd21}" \
  --pretrained_path "$CKPT" --input_path "$INPUT" \
  --mask_dir "$MASKS" --output_dir "$OUTPUT" "$@"
