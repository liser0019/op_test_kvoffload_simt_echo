#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 无本地执行；交给950PR服务器。默认模型预设：MLA BF16 1024B+128B/token。
set -euo pipefail
TRANSFER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_DEVICE="${1:-0}"
if [[ ! "$TRANSFER_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "用法: bash run_plan_transfer.sh [逻辑卡号，默认0] [profile_plan_transfer.py参数...]" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
cd "$TRANSFER_ROOT"
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python3}" -X faulthandler profile_plan_transfer.py --device "$TRANSFER_DEVICE" "$@"
