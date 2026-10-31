#!/usr/bin/env bash
# Each stage is explicit, logged, fail-fast. Never edits/builds the failed checkout.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/usr/local/python3.11.10/bin/python3}"
STAGE="${1:?usage: test_bitmap_single_vf.sh old-control|candidate [old-checkout]}"
OUT_BASE="${RESULTS_ROOT:-/workspace/bitmap_single_vf_results}"
mkdir -p "$OUT_BASE"
OUT="$(mktemp -d "$OUT_BASE/${STAGE}_XXXXXX")"
export TMPDIR="${TMPDIR:-$OUT/tmp}"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
trap 'code=$?; if (( code != 0 )); then echo "STOP: stage=$STAGE exit=$code results=$OUT. Do not tune VF again; inspect ProcessRuntimeRow hash/owner/scan." >&2; fi' EXIT
cd "$ROOT"
git rev-parse HEAD > "$OUT/source_commit.txt"
echo "Results: $OUT"
if [[ "$STAGE" == old-control ]]; then
  OLD="${2:?old checkout containing its existing library is required}"
  [[ "$(git -C "$OLD" rev-parse HEAD)" == 84769ae5d30b9f1ee1cc04ee34e34b27c86c5ee2 ]]
  "$PY" -X faulthandler checks/run_bitmap_single_vf_probe.py --source-root "$OLD" \
    --implementation abcdf --repeat 1 --device 0 --temp-dir "$TMPDIR" \
    --report "$OUT/abcdf_old_control.json" 2>&1 | tee "$OUT/old_control.log"
elif [[ "$STAGE" == candidate ]]; then
  bash build_joint_plan.sh --soc=ascend950 2>&1 | tee "$OUT/build.log"
  "$PY" -X faulthandler checks/run_bitmap_single_vf_probe.py \
    --implementation joint_bitmap --repeat 5 --device 0 --temp-dir "$TMPDIR" \
    --report "$OUT/bitmap_mtp0.json" 2>&1 | tee "$OUT/bitmap_mtp0.log"
  "$PY" -X faulthandler checks/check_joint_plan.py --device 0 \
    --implementations abcdf joint_bitmap --regression-repeat 5 --temp-dir "$TMPDIR" \
    --report "$OUT/gate.json" 2>&1 | tee "$OUT/gate.log"
  "$PY" -X faulthandler profile_joint_mtp.py --device 0 \
    --implementations abcdf joint_bitmap --requests 32 --mtp 0 1 3 \
    --topk 2048 --capacity 4096 --max-token 262144 --block-size 64 --seed 2026 \
    --scenarios full_cache_75pct --overlaps independent --warmup 2 --repeat 5 \
    --regression-repeat 5 --correctness-report "$OUT/gate.json" --temp-dir "$TMPDIR" \
    --output "$OUT/profile" 2>&1 | tee "$OUT/profile.log"
else
  echo "unknown stage: $STAGE" >&2; exit 2
fi
