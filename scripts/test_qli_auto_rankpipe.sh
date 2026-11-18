#!/usr/bin/env bash
# Keep the colleague's automatic QLI tiling and compare ABD512 with rankpipe AC.
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
for BATCH in 8 32; do
  "$PYTHON" -B "$ROOT/profile_qli_fusion.py" \
    --requests "$BATCH" --output "$OUTPUT/batch$BATCH" \
    --temp-dir "$TEMP/batch$BATCH" --mtp 3 \
    --contexts 102400 262144 --scenarios cold_reset full_cache_75pct \
    --implementations peer c "${ARGS[@]}"
done
