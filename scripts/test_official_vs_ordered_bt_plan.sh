#!/usr/bin/env bash
# Official Indexer baseline versus ordered Indexer + frozen bt-direct-odd Plan.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "用法：$0 DEVICE QLI_BUILD PLAN_LIBRARY PLAN_IDENTITY [PERF_ARGS...]" >&2
  exit 2
fi
DEVICE=$1
QLI_BUILD=$(realpath "$2")
PLAN_LIBRARY=$(realpath "$3")
PLAN_IDENTITY=$(realpath "$4")
shift 4

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
OUTPUT=${OUTPUT:-/workspace/official_vs_ordered_bt_plan_results}
BATCH_START=${BATCH_START:-8}
BATCH_END=${BATCH_END:-16}
CONTEXTS=${CONTEXTS:-"102400 358400"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-50}
REGRESSION_REPEAT=${REGRESSION_REPEAT:-2}

mkdir -p "$OUTPUT"
printf '%s\n' "$DEVICE" > "$OUTPUT/device.txt"
git -C "$ROOT" rev-parse HEAD > "$OUTPUT/harness-commit.txt"
cat "$PLAN_IDENTITY" > "$OUTPUT/plan-identity.json"
sha256sum "$PLAN_LIBRARY" > "$OUTPUT/plan-library.sha256"

read -r -a CONTEXT_ARGS <<< "$CONTEXTS"
cd "$ROOT"
for BATCH in $(seq "$BATCH_START" "$BATCH_END"); do
  DEST="$OUTPUT/batch$BATCH"
  "$PYTHON" -B profile_qli_fusion.py \
    --device "$DEVICE" --fusion-build "$QLI_BUILD" --library "$PLAN_LIBRARY" \
    --library-identity "$PLAN_IDENTITY" --output "$DEST" --temp-dir "$OUTPUT/tmp" \
    --requests "$BATCH" --mtp 3 --contexts "${CONTEXT_ARGS[@]}" \
    --scenarios full_cache_90pct --implementations baseline ordered_bt_direct_odd \
    --warmup "$WARMUP" --repeat "$REPEAT" --regression-repeat "$REGRESSION_REPEAT" "$@"
done

printf '%s\n' '{"status":"complete","batches":[8,9,10,11,12,13,14,15,16],"contexts":[102400,358400],"scenario":"full_cache_90pct"}' > "$OUTPUT/MATRIX_COMPLETE.json"
