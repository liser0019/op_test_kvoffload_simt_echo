#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${1:-1}"
[[ "$DEVICE" =~ ^[0-9]+$ ]] || { echo "Usage: bash run_abcdf_memfabric.sh [logical-device]" >&2; exit 2; }
PYTHON_BIN="${PYTHON:-python3}"
OUTPUT="${MEMFABRIC_PROFILE_OUTPUT:-$ROOT/results/abcdf_memfabric_profile}"
cd "$ROOT"
"$PYTHON_BIN" -X faulthandler compare_memfabric_plan.py --device "$DEVICE"
"$PYTHON_BIN" -X faulthandler profile_plan_variants.py --device "$DEVICE" \
    --comparison-module compare_memfabric_plan.py --scenario all \
    --implementations abcdf memfabric --reference memfabric \
    --shuffle-lru --warmup 10 --repeat 60 --output "$OUTPUT"
