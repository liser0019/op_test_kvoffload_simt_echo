#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/usr/local/python3.11.10/bin/python3}"
OUTPUT="${OUTPUT:-/workspace/sorted_hardware_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$OUTPUT" ]]; then echo "拒绝覆盖 $OUTPUT" >&2; exit 2; fi
mkdir -p "$OUTPUT/tmp"
export PYTHON TMPDIR="$OUTPUT/tmp" PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
# 五路同场：基线、原Merge、已测BD、无Hash软件流式、无Hash硬件归并。
read -r -a names <<< "${CANDIDATES:-abcdf_pages joint_merge joint_merge_bd joint_sorted_stream joint_sorted_mrgsort}"
read -r -a scenes <<< "${SCENARIOS:-full_cache_75pct}"
bash build_joint_plan.sh --soc=ascend950 --sorted-only 2>&1 | tee "$OUTPUT/build.log"
"$PYTHON" -B -X faulthandler checks/check_merge_abcd_npu.py --device 0 \
 --implementations "${names[@]}" --regression-repeat 5 --temp-dir "$TMPDIR" \
 --report "$OUTPUT/gate.json" 2>&1 | tee "$OUTPUT/gate.log"
"$PYTHON" -B -X faulthandler profile_joint_mtp.py --device 0 \
 --implementations "${names[@]}" --sorted-topk --requests 32 --mtp 0 1 3 \
 --topk 2048 --capacity 4096 --max-token 262144 --block-size 64 --seed 2026 \
 --scenarios "${scenes[@]}" --overlaps overlap40 --warmup "${WARMUP:-2}" --repeat "${REPEAT:-5}" \
 --regression-repeat 5 --correctness-report "$OUTPUT/gate.json" \
 --temp-dir "$TMPDIR" --output "$OUTPUT/profile" 2>&1 | tee "$OUTPUT/profile.log"
echo "完成：$OUTPUT/profile/joint_matrix_*/report.md"
