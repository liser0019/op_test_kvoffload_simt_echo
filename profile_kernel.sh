#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# kernel 级 profile 采集包装：环境准备（/dev/shm 采集目录、TMP 重定向、清掉干扰变量）
# + 调用 profile_kernel.py + 日志落盘 + /dev/shm 空间检查。
# 用法: bash profile_kernel.sh [--device 1] [--scenario all|cold_reset|warm_75pct|warm_all_hit|full_cache_75pct]
#                           [--warmup 5] [--active 20] [--library 路径]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export KV_PROFILE_ROOT="${KV_PROFILE_ROOT:-$(mktemp -d /dev/shm/kvprof.XXXXXX)}"
mkdir -p "$KV_PROFILE_ROOT/tmp"
export TMPDIR="$KV_PROFILE_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

echo "本次采集目录: $KV_PROFILE_ROOT"
rc=0
python3 -X faulthandler profile_kernel.py "$@" 2>&1 | tee "$KV_PROFILE_ROOT/profile.log" || rc=$?
echo "Profiler 退出码: $rc"
df -h /dev/shm
exit "$rc"
