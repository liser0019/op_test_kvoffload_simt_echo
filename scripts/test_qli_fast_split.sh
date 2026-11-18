#!/usr/bin/env bash
# Compare both S1Base=4 split chains with the same standalone rankpipe AC Plan.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/test_qli_fusion.sh" \
  --requests 32 --contexts 262144 --mtp 3 7 \
  --scenarios cold_reset full_cache_75pct --warmup 2 --repeat 5 \
  --implementations final4 split4 "$@"
