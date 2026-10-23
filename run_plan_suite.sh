#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 一张卡串行执行：生产正确性门禁 → MTP/Scan kernel profiler矩阵。
set -euo pipefail
PLAN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DEVICE="${1:-0}"
if [[ ! "$PLAN_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "Device必须是非负逻辑卡号" >&2
    exit 2
fi
cd "$PLAN_DIR"
PLAN_PYTHON="${PYTHON:-python3}"
PLAN_LIBRARY="${PLAN_LIBRARY:-$PLAN_DIR/build/libsimt_plan_variants_ops.so}"
if [[ ! -f "$PLAN_LIBRARY" ]]; then
    echo "请先在950服务器运行bash build.sh --soc=ascend950" >&2
    exit 2
fi
PLAN_OUTPUT="${PLAN_PROFILE_OUTPUT:-/dev/shm/abcdf_mtp_scan}"
mkdir -p "$PLAN_OUTPUT"
PLAN_SESSION="$(mktemp -d "$PLAN_OUTPUT/session_XXXXXXXX")"
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
# 必须全部通过，失败会立即停止；不会将SKIP当作通过。
"$PLAN_PYTHON" -X faulthandler checks/check_mtp_scan.py --device "$PLAN_DEVICE" \
    --library "$PLAN_LIBRARY" --production --report "$PLAN_SESSION/correctness.json" \
    2>&1 | tee "$PLAN_SESSION/correctness.log"
PLAN_ARGS=(--device "$PLAN_DEVICE" --library "$PLAN_LIBRARY"
           --requests "${REQUESTS:-32}" --fixed-query-rows "${FIXED_QUERY_ROWS:-32}"
           --topk "${TOPK:-2048}" --capacity "${CAPACITY:-4096}"
           --max-token "${MAX_TOKEN:-262144}" --group "${GROUP:-all}"
           --mtp 0 3 --comparison both --warmup "${WARMUP:-10}"
           --correctness-report "$PLAN_SESSION/correctness.json" --output "$PLAN_SESSION")
if [[ -n "${REPEAT:-}" ]]; then PLAN_ARGS+=(--repeat "$REPEAT"); fi
if [[ "${SHUFFLE_LRU:-0}" == "1" ]]; then PLAN_ARGS+=(--shuffle-lru); fi
"$PLAN_PYTHON" -X faulthandler profile_mtp_matrix.py "${PLAN_ARGS[@]}" \
    2>&1 | tee "$PLAN_SESSION/profile.log"
echo "完成: $PLAN_SESSION"
echo "保留整个目录，包括原始kernel_details.csv、capture_metadata.json与matrix_summary.json。"
