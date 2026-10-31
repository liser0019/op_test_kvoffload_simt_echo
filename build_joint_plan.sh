#!/usr/bin/env bash
# 联合Plan独立构建，不要求安装原vLLM工程，也不编译Transfer/Scan实验。
set -euo pipefail
JOINT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOINT_DEBUG=OFF
JOINT_RESOURCES=OFF
for arg in "$@"; do
  case "$arg" in
    --soc=ascend950) ;;
    --debug-info) JOINT_DEBUG=ON ;;
    --resource-usage) JOINT_RESOURCES=ON ;;
    --help|-h)
      echo "用法: PYTHON=python3 JOBS=3 bash build_joint_plan.sh [--soc=ascend950] [--debug-info] [--resource-usage]"
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
  -DJOINT_RESOURCE_USAGE="$JOINT_RESOURCES")
for name in ASCEND_CANN_PACKAGE_HOME BISHENG_CXX ASCENDC_RUNTIME_LIBRARY; do
  if [[ -n "${!name:-}" ]]; then JOINT_ARGS+=("-D$name=${!name}"); fi
done
cmake "${JOINT_ARGS[@]}"
cmake --build "$JOINT_BUILD_DIR" --target mtp_joint_plan_ops --parallel "${JOBS:-3}"
echo "产物: $JOINT_BUILD_DIR/libmtp_joint_plan_ops.so"
echo "下一步: $JOINT_PYTHON checks/check_joint_plan.py --device 2 --library $JOINT_BUILD_DIR/libmtp_joint_plan_ops.so"
