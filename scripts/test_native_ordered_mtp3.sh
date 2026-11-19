#!/usr/bin/env bash
# 本脚本只负责同机采集。请先构建独立 Plan 和原生保序融合包。
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "用法：$0 DEVICE FUSION_BUILD PLAN_LIBRARY OUTPUT [profile_qli_fusion.py 的实现和采样选项]" >&2
  exit 2
fi
DEVICE=$1
FUSION_BUILD=$(realpath "$2")
PLAN_LIBRARY=$(realpath "$3")
OUTPUT=$(realpath -m "$4")
shift 4

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
read -r -a BATCHES <<< "${MTP3_BATCHES:-6 8 10 16 17 32}"
mkdir "$OUTPUT"
mkdir "$OUTPUT/tmp"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
source "$FUSION_BUILD/env.sh"

cd "$ROOT"
for batch in "${BATCHES[@]}"; do
  "$PYTHON" -B profile_qli_fusion.py \
    --device "$DEVICE" --fusion-build "$FUSION_BUILD" --library "$PLAN_LIBRARY" \
    --output "$OUTPUT/bs$batch" --temp-dir "$OUTPUT/tmp" \
    --requests "$batch" --mtp 3 --contexts 262144 \
    --scenarios cold_reset full_cache_75pct "$@"
done
