#!/usr/bin/env bash
# 只测六路、六工况；失败即停，不展开旧候选/全overlap矩阵。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/usr/local/python3.11.10/bin/python3}"
OUTPUT="${OUTPUT:-/workspace/turbo_phase_scan_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$OUTPUT" ]]; then echo "拒绝覆盖已有结果: $OUTPUT" >&2; exit 2; fi
mkdir -p "$OUTPUT/tmp"
export TMPDIR="$OUTPUT/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHON
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
implementations=(abcdf private_turbo joint_turbo joint_turbo_a joint_turbo_b joint_turbo_ab)
bash build_joint_plan.sh --soc=ascend950 2>&1 | tee "$OUTPUT/build.log"
"$PYTHON" -B -X faulthandler checks/check_joint_plan.py --device 0 \
  --implementations "${implementations[@]}" --regression-repeat 5 \
  --temp-dir "$TMPDIR" --report "$OUTPUT/gate.json" 2>&1 | tee "$OUTPUT/gate.log"
"$PYTHON" -B -X faulthandler profile_joint_mtp.py --device 0 \
  --implementations "${implementations[@]}" --requests 32 --mtp 0 1 3 \
  --topk 2048 --capacity 4096 --max-token 262144 --block-size 64 --seed 2026 \
  --scenarios cold_reset full_cache_75pct --overlaps independent \
  --warmup "${WARMUP:-2}" --repeat "${REPEAT:-5}" --regression-repeat 5 \
  --correctness-report "$OUTPUT/gate.json" --temp-dir "$TMPDIR" \
  --output "$OUTPUT/profile" 2>&1 | tee "$OUTPUT/profile.log"
echo "结果: $OUTPUT；以退出码及 COMPLETE.json 为准，失败现场/原始采集不删除。"
