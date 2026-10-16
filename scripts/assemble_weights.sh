#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${1:-weights/RIFLE.pkl}"
PART_PREFIX="${PART_PREFIX:-weights/RIFLE.pkl.part-}"
SHA_PATH="${SHA_PATH:-weights/RIFLE.pkl.sha256}"

parts=( "${PART_PREFIX}"* )
if [ "${#parts[@]}" -eq 0 ] || [ ! -e "${parts[0]}" ]; then
  echo "No split weight parts found with prefix: $PART_PREFIX" >&2
  exit 1
fi

cat "${parts[@]}" > "$OUT_PATH"

if [ -f "$SHA_PATH" ]; then
  expected="$(awk '{print $1}' "$SHA_PATH")"
  actual="$(shasum -a 256 "$OUT_PATH" | awk '{print $1}')"
  if [ "$expected" != "$actual" ]; then
    echo "Checksum mismatch for $OUT_PATH" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
  echo "Checksum verified for $OUT_PATH"
else
  echo "Reconstructed $OUT_PATH"
fi
