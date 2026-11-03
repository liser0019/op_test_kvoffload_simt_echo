#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/usr/local/python3.11.10/bin/python3}"
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -B -X faulthandler run_union_next_sweep.py --cold-compare "$@"
