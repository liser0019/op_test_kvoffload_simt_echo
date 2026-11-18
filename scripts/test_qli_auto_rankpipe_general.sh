#!/usr/bin/env bash
# General fused comparison without hard-coding the resident batch around core count.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
OUTPUT=
TEMP=
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --temp-dir) TEMP=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ -n "$OUTPUT" ]] || { echo '--output is required' >&2; exit 2; }
[[ -n "$TEMP" ]] || TEMP="$OUTPUT/tmp"
mkdir -p "$OUTPUT" "$TEMP"
for BATCH in 6 8 10 32; do
  "$PYTHON" -B "$ROOT/profile_qli_fusion.py" \
    --requests "$BATCH" --output "$OUTPUT/batch$BATCH" \
    --temp-dir "$TEMP/batch$BATCH" --mtp 3 \
    --contexts 102400 262144 --scenarios cold_reset full_cache_75pct \
    --implementations peer c "${ARGS[@]}"
done
