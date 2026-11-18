#!/usr/bin/env bash
# Source checks + the MTP3 factorial screening matrix. Results are never reused.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-0}
BUILD=${QLI_FUSION_BUILD_DIR:-$ROOT/build_qli_fusion}
OUT=${OUT:-$ROOT/results/factorial_screen}
TEMP=${TEMP_DIR:-$ROOT/tmp/factorial_screen}
WARMUP=${WARMUP:-2}
REPEAT=${REPEAT:-5}
CONTEXTS=${CONTEXTS:-"100000 262144"}
BATCHES=${BATCHES:-"8 17 32"}
IMPLEMENTATIONS=${IMPLEMENTATIONS:-"c0 A B C AB AC BC ABC"}
mkdir -p "$OUT" "$TEMP"
cd "$ROOT"
"$PYTHON" -B scripts/export_fusion_plan.py --check
"$PYTHON" -m checks.check_qli_factorial --upstream "$BUILD/upstream"
"$PYTHON" -m checks.check_qli_fusion_resources --upstream "$BUILD/upstream" > "$OUT/resources.json"
for batch in $BATCHES; do
  "$PYTHON" -B profile_qli_fusion.py \
    --fusion-build "$BUILD" --device "$DEVICE" --temp-dir "$TEMP" \
    --requests "$batch" --mtp 3 --contexts $CONTEXTS \
    --scenarios cold_reset full_cache_75pct \
    --implementations $IMPLEMENTATIONS \
    --warmup "$WARMUP" --repeat "$REPEAT" --regression-repeat 1 \
    --output "$OUT/batch$batch" 2>&1 | tee "$OUT/batch$batch.log"
done
echo QLI_FACTORIAL_MATRIX_COMPLETE
