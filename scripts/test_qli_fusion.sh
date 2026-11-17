#!/usr/bin/env bash
# Same built libraries can be reused for smoke and confirmation; no auto rebuild.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
exec "${PYTHON:-python3}" -B -X faulthandler profile_qli_fusion.py --device 0 "$@"
