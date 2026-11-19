#!/usr/bin/env bash
# Narrow same-card 475K/context sweep; run after the parent build has registered all four names.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-0}"
OUTPUT="${OUTPUT:-/workspace/mtp3_bitmap_team_profile_$(date +%Y%m%d_%H%M%S)}"

if [[ "$OUTPUT" != /workspace/* ]]; then
  echo "OUTPUT must persist under /workspace: $OUTPUT" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/tmp"
export TMPDIR="$OUTPUT/tmp" PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

IMPLEMENTATIONS=(joint_mtp3_bitmap475_512 joint_mtp3_bitmap_ubmap_512
  joint_mtp3_bt_share_512 joint_mtp3_bt_direct_512)
COMMON=("$PYTHON" -B -X faulthandler profile_mtp3_lookup.py --device "$DEVICE"
  --implementations "${IMPLEMENTATIONS[@]}" --overlaps overlap40
  --warmup 3 --repeat 10 --temp-dir "$OUTPUT/tmp")

# 475K: cold and representative-hot at the requested batch shape.
"${COMMON[@]}" --output "$OUTPUT/475k" --contexts 486400 \
  --requests 1 8 9 10 16 28 32 --scenarios cold_reset full_cache_75pct

# Hot context boundary checks at batch 8 only.
"${COMMON[@]}" --output "$OUTPUT/other-contexts" \
  --contexts 102400 262144 358400 --requests 8 --scenarios full_cache_75pct

echo "profile results: $OUTPUT"
