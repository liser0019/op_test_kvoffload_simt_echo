#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build this repository first, then run the peer SIMT audit and kernel profiling suite.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PEER_PYTHON="${PYTHON:-python3}"
PEER_DEVICE="${1:-1}"
PEER_OUTPUT="${PEER_PROFILE_OUTPUT:-/dev/shm/peer_simt_profile}"

if [[ ! -f compare.py || ! -f peer_variants_manifest.json ]]; then
    echo "Run this script from the peer SIMT comparison checkout." >&2
    exit 2
fi
if [[ ! "$PEER_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "Usage: PYTHON=python3 bash run_peer_suite.sh [logical_NPU_device]" >&2
    exit 2
fi
if [[ ! -f build/libsimt_vs_echo_ops.so ]]; then
    echo "Build first: PYTHON=python3 JOBS=3 bash build.sh" >&2
    exit 2
fi

PEER_IMPLEMENTATIONS=(
    10_bcde
    combo_bcde_1024
    combo_bcde_pair_scan
    combo_bcde_fused
    combo_bcde_fused_1024
    peer_00_baseline
    peer_01_threads_1024
    peer_04_parallel_scan
    peer_05_fused_lookup
    peer_06_packed_lists
)
mkdir -p -- "$PEER_OUTPUT"
PEER_SESSION="$(mktemp -d "$PEER_OUTPUT/session_XXXXXX")"
echo "Session output: $PEER_SESSION"

# Short per-call audit. Its Event timing includes the public op interval.
# Frequent readback here is for correctness; no performance conclusion.
"$PEER_PYTHON" -X faulthandler compare.py \
    --device "$PEER_DEVICE" --scenario all \
    --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
    --implementations "${PEER_IMPLEMENTATIONS[@]}" --reference 10_bcde \
    --warmup 2 --repeat 3 --verify-each-sample \
    --json "$PEER_SESSION/short_audit.json" 2>&1 | tee "$PEER_SESSION/short_audit.log"

# Formal kernel capture; the collector excludes audit and warm-up calls.
"$PEER_PYTHON" -X faulthandler profile_peer_variants.py \
    --device "$PEER_DEVICE" --scenario all \
    --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
    --implementations "${PEER_IMPLEMENTATIONS[@]}" --reference 10_bcde \
    --warmup 13 --repeat 60 \
    --output "$PEER_SESSION/kernel_profile" 2>&1 | tee "$PEER_SESSION/profile.log"

echo "Completed: $PEER_SESSION"
echo "Send profile.log and capture_metadata.json; kernel_summary.csv is in each scenario directory."
