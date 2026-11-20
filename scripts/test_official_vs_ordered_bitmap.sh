#!/usr/bin/env bash
# Official Indexer baseline versus ordered Indexer plus bitmap/BT MTP3 Plans.
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
OUTPUT=${OUTPUT:-"/workspace/official_vs_ordered_bitmap_results_$(date +%Y%m%d_%H%M%S)"}
BATCH_START=${BATCH_START:-8}
BATCH_END=${BATCH_END:-16}
CONTEXTS=${CONTEXTS:-"102400 358400"}
WARMUP=${WARMUP:-10}
REPEAT=${REPEAT:-20}
REGRESSION_REPEAT=${REGRESSION_REPEAT:-2}
IMPLEMENTATIONS=${IMPLEMENTATIONS:-"baseline ordered_bitmap512 ordered_bt_direct_odd"}

mkdir -p "$OUTPUT"
printf '%s\n' "$DEVICE" > "$OUTPUT/device.txt"
if [[ -n "${HARNESS_COMMIT:-}" ]]; then
  printf '%s\n' "$HARNESS_COMMIT" > "$OUTPUT/harness-commit.txt"
elif git -C "$ROOT" rev-parse HEAD > "$OUTPUT/harness-commit.txt" 2>/dev/null; then
  :
else
  echo "源码 archive 场景必须设置 HARNESS_COMMIT" >&2
  exit 2
fi
cp "$PLAN_IDENTITY" "$OUTPUT/plan-identity.json"
sha256sum "$PLAN_LIBRARY" > "$OUTPUT/plan-library.sha256"

read -r -a CONTEXT_ARGS <<< "$CONTEXTS"
read -r -a IMPLEMENTATION_ARGS <<< "$IMPLEMENTATIONS"
cd "$ROOT"
for BATCH in $(seq "$BATCH_START" "$BATCH_END"); do
  DEST="$OUTPUT/batch$BATCH"
  "$PYTHON" -B profile_qli_fusion.py \
    --device "$DEVICE" --fusion-build "$QLI_BUILD" --library "$PLAN_LIBRARY" \
    --library-identity "$PLAN_IDENTITY" --output "$DEST" --temp-dir "$OUTPUT/tmp" \
    --requests "$BATCH" --mtp 3 --contexts "${CONTEXT_ARGS[@]}" \
    --scenarios full_cache_90pct --implementations "${IMPLEMENTATION_ARGS[@]}" \
    --warmup "$WARMUP" --repeat "$REPEAT" --regression-repeat "$REGRESSION_REPEAT" "$@"
done

"$PYTHON" - "$OUTPUT" "$BATCH_START" "$BATCH_END" "$CONTEXTS" <<'PY'
import json
import sys
from pathlib import Path
out=Path(sys.argv[1])
json.dump({
    'status':'complete',
    'batch_start':int(sys.argv[2]),
    'batch_end':int(sys.argv[3]),
    'contexts':[int(v) for v in sys.argv[4].split()],
    'scenario':'full_cache_90pct',
}, (out/'MATRIX_COMPLETE.json').open('w'), indent=2)
(out/'MATRIX_COMPLETE.json').write_text((out/'MATRIX_COMPLETE.json').read_text()+'\n')
PY
printf '%s\n' "$OUTPUT"
