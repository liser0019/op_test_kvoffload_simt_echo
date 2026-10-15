#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
PLAN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DEVICE="${1:-1}"
if [[ ! "$PLAN_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "Device must be a non-negative logical NPU index" >&2
    exit 2
fi
cd "$PLAN_DIR"
PLAN_PYTHON="${PYTHON:-python3}"
if [[ ! -f build/libsimt_plan_variants_ops.so ]]; then
    echo "Run bash build.sh on A5 first" >&2
    exit 2
fi
PLAN_OUTPUT="${PLAN_PROFILE_OUTPUT:-/dev/shm/simt_plan_micro}"
mkdir -p "$PLAN_OUTPUT"
PLAN_SESSION="$(mktemp -d "$PLAN_OUTPUT/session_XXXXXXXX")"
unset ASCEND_LAUNCH_BLOCKING
PLAN_ARGS=(--device "$PLAN_DEVICE" --rows "${ROWS:-32}" --topk "${TOPK:-2048}"
           --capacity "${CAPACITY:-4096}" --max-token "${MAX_TOKEN:-262144}"
           --block-size "${BLOCK_SIZE:-64}" --reference baseline)
if [[ "${SHUFFLE_LRU:-0}" == "1" ]]; then PLAN_ARGS+=(--shuffle-lru); fi
# All 18 versions; edge audits and poisoned outputs precede performance capture.
"$PLAN_PYTHON" -X faulthandler compare.py "${PLAN_ARGS[@]}" \
    --audit-edges --warmup 2 --repeat 3 --verify-each-sample \
    --json "$PLAN_SESSION/audit.json" 2>&1 | tee "$PLAN_SESSION/audit.log"
# 72 calls = four full rotations through the 18 execution positions.
"$PLAN_PYTHON" -X faulthandler profile_plan_variants.py "${PLAN_ARGS[@]}" \
    --scenario all --warmup 18 --repeat "${REPEAT:-72}" \
    --output "$PLAN_SESSION/kernel_profile" 2>&1 | tee "$PLAN_SESSION/profile.log"
echo "Completed: $PLAN_SESSION"
echo "Collect profile.log, capture_metadata.json and each kernel_summary.csv."
