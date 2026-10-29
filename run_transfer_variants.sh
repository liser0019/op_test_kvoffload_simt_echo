#!/usr/bin/env bash
set -euo pipefail
FACTOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FACTOR_DEVICE="${1:-0}"
if [[ ! "$FACTOR_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "用法: bash run_transfer_variants.sh [逻辑卡号，默认0] [矩阵参数...]" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
cd "$FACTOR_ROOT"
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python3}" -X faulthandler profile_transfer_variants.py --device "$FACTOR_DEVICE" "$@"
