#!/usr/bin/env bash
# 联合Plan独立构建，不要求安装原vLLM工程，也不编译Transfer/Scan实验。
set -euo pipefail
JOINT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOINT_DEBUG=OFF
JOINT_RESOURCES=OFF
JOINT_MERGE_ONLY=OFF
JOINT_SORTED_ONLY=OFF
JOINT_SWEEP_ONLY=OFF
JOINT_UNION_ONLY=OFF
JOINT_AC_THREADS_ONLY=OFF
JOINT_COLD_COMPARE_ONLY=OFF
JOINT_INTEGRATED_ONLY=OFF
JOINT_REFINE_ONLY=OFF
JOINT_EXEC_ONLY=OFF
JOINT_HOT_ONLY=OFF
JOINT_MTP8_ONLY=OFF
JOINT_MTP7_FIX_ONLY=OFF
for arg in "$@"; do
  case "$arg" in
    --soc=ascend950) ;;
    --merge-only) JOINT_MERGE_ONLY=ON ;;
    --sorted-only) JOINT_SORTED_ONLY=ON ;;
    --sweep-only) JOINT_SWEEP_ONLY=ON ;;
    --union-only) JOINT_UNION_ONLY=ON ;;
    --ac-threads-only) JOINT_AC_THREADS_ONLY=ON ;;
    --cold-compare-only) JOINT_COLD_COMPARE_ONLY=ON ;;
    --integrated-only) JOINT_INTEGRATED_ONLY=ON ;;
    --refine-only) JOINT_REFINE_ONLY=ON ;;
    --exec-only) JOINT_EXEC_ONLY=ON ;;
    --mtp7-fix-only) JOINT_MTP7_FIX_ONLY=ON ;;
    --mtp8-only) JOINT_MTP8_ONLY=ON ;;
    --hot-only) JOINT_HOT_ONLY=ON ;;
    --debug-info) JOINT_DEBUG=ON ;;
    --resource-usage) JOINT_RESOURCES=ON ;;
    --help|-h)
      echo "用法: PYTHON=python3 JOBS=3 bash build_joint_plan.sh [--soc=ascend950] [--debug-info] [--resource-usage] [--merge-only] [--sorted-only] [--sweep-only] [--union-only] [--ac-threads-only] [--cold-compare-only] [--integrated-only] [--refine-only] [--exec-only] [--hot-only] [--mtp8-only] [--mtp7-fix-only]"
      exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done
JOINT_BUILD_DIR="$JOINT_ROOT/build_joint_plan"
if [[ "$JOINT_DEBUG" == ON ]]; then JOINT_BUILD_DIR="$JOINT_ROOT/build_joint_plan_profile"; fi
JOINT_PYTHON="$(command -v -- "${PYTHON:-python3}")"
JOINT_ARGS=(-S "$JOINT_ROOT/joint_plan" -B "$JOINT_BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH=ascend950
  -DPython3_EXECUTABLE="$JOINT_PYTHON" -DJOINT_DEBUG_INFO="$JOINT_DEBUG"
  -DJOINT_RESOURCE_USAGE="$JOINT_RESOURCES" -DJOINT_MERGE_ONLY="$JOINT_MERGE_ONLY" -DJOINT_SORTED_ONLY="$JOINT_SORTED_ONLY" -DJOINT_SWEEP_ONLY="$JOINT_SWEEP_ONLY" -DJOINT_UNION_ONLY="$JOINT_UNION_ONLY" -DJOINT_AC_THREADS_ONLY="$JOINT_AC_THREADS_ONLY" -DJOINT_COLD_COMPARE_ONLY="$JOINT_COLD_COMPARE_ONLY" -DJOINT_INTEGRATED_ONLY="$JOINT_INTEGRATED_ONLY" -DJOINT_REFINE_ONLY="$JOINT_REFINE_ONLY" -DJOINT_EXEC_ONLY="$JOINT_EXEC_ONLY" -DJOINT_HOT_ONLY="$JOINT_HOT_ONLY" -DJOINT_MTP8_ONLY="$JOINT_MTP8_ONLY" -DJOINT_MTP7_FIX_ONLY="$JOINT_MTP7_FIX_ONLY")
for name in ASCEND_CANN_PACKAGE_HOME BISHENG_CXX ASCENDC_RUNTIME_LIBRARY; do
  if [[ -n "${!name:-}" ]]; then JOINT_ARGS+=("-D$name=${!name}"); fi
done
cmake "${JOINT_ARGS[@]}"
cmake --build "$JOINT_BUILD_DIR" --target mtp_joint_plan_ops --parallel "${JOBS:-3}"
echo "产物: $JOINT_BUILD_DIR/libmtp_joint_plan_ops.so"
if [[ "$JOINT_MTP7_FIX_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_mtp7_fix.sh --skip-build --device 0 --output /workspace/mtp7_fix_results"
elif [[ "$JOINT_MTP8_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_mtp8.sh --skip-build --device 0 --output /workspace/mtp8_results"
elif [[ "$JOINT_HOT_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_hot_sweep.sh --skip-build --device 0"
elif [[ "$JOINT_EXEC_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_exec_sweep.sh --skip-build --device 0"
elif [[ "$JOINT_REFINE_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_refine_sweep.sh --skip-build --device 0"
elif [[ "$JOINT_INTEGRATED_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_integrated_plan.sh --skip-build --device 0 (五路固定对照，含Phase0首尾判断)"
elif [[ "$JOINT_COLD_COMPARE_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_cold_finish.sh --skip-build --device 0 (六路固定对照)"
elif [[ "$JOINT_AC_THREADS_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_union_next.sh --skip-build --implementations abcdf_pages joint_hw_ac_512 joint_hw_ac_256 --device 0"
elif [[ "$JOINT_UNION_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_union_next.sh --skip-build (NPU0，新ABCD×1024/512/256，初筛与独立复测)"
elif [[ "$JOINT_SWEEP_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_hardware_sweep.sh (NPU0，一次启动筛选和复测)"
elif [[ "$JOINT_SORTED_ONLY" == ON ]]; then
  echo "下一步: bash scripts/test_sorted_hardware.sh (NPU0，专项门禁后同场profiling)"
elif [[ "$JOINT_MERGE_ONLY" == ON ]]; then
  echo "下一步: $JOINT_PYTHON checks/check_merge_abcd_npu.py --device 0 --library $JOINT_BUILD_DIR/libmtp_joint_plan_ops.so --report /workspace/merge_abcd_gate.json"
else
  echo "下一步: $JOINT_PYTHON checks/check_joint_plan.py --device 2 --library $JOINT_BUILD_DIR/libmtp_joint_plan_ops.so"
fi
