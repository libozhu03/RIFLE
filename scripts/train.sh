#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python}" train.py \
  --config "${CONFIG_PATH:-configs/train.yaml}" "$@"
