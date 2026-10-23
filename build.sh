#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CMake 配置 → 编译原版ABCDF与17个Scan候选，不依赖原工程。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PYTHON="${PYTHON:-python3}"
BUILD_JOBS="${JOBS:-4}"
SOC_VERSION="ascend950"
PROFILE_DEBUG="${PLAN_PROFILE_DEBUG_INFO:-OFF}"
RESOURCE_USAGE="${PLAN_RESOURCE_USAGE:-OFF}"
for arg in "$@"; do
    case "$arg" in
        --soc=*) SOC_VERSION="${arg#*=}" ;;
        --debug-info) PROFILE_DEBUG=ON ;;
        --resource-usage) RESOURCE_USAGE=ON ;;
        --help|-h)
            echo "用法: PYTHON=python3 JOBS=4 bash build.sh --soc=ascend950 [--debug-info] [--resource-usage]"
            echo "--debug-info: 保持-O3，给设备代码增加-g，独立输出到build_profile/"
            echo "可覆盖: ASCEND_CANN_PACKAGE_HOME, BISHENG_CXX, ASCENDC_RUNTIME_LIBRARY"
            exit 0 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done
if [[ "$SOC_VERSION" != "ascend950" ]]; then
    echo "仅支持 --soc=ascend950" >&2
    exit 2
fi
if [[ ! "$PROFILE_DEBUG" =~ ^(ON|OFF)$ || ! "$RESOURCE_USAGE" =~ ^(ON|OFF)$ ]]; then
    echo "PLAN_PROFILE_DEBUG_INFO / PLAN_RESOURCE_USAGE仅支持ON或OFF" >&2
    exit 2
fi
# 使用激活环境中的确切解释器，也让 CMake 使用同一个解释器。
BUILD_PYTHON="$(command -v -- "$BUILD_PYTHON")"
EXPERIMENT_BUILD_DIR="$SCRIPT_DIR/build"
if [[ "$PROFILE_DEBUG" == "ON" ]]; then EXPERIMENT_BUILD_DIR="$SCRIPT_DIR/build_profile"; fi
EXPERIMENT_BUILD_DIR="${PLAN_BUILD_DIR:-$EXPERIMENT_BUILD_DIR}"
CMAKE_ARGS=(
    -S "$SCRIPT_DIR" -B "$EXPERIMENT_BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="$SOC_VERSION"
    -DPLAN_PROFILE_DEBUG_INFO="$PROFILE_DEBUG" -DPLAN_RESOURCE_USAGE="$RESOURCE_USAGE"
    -DPython3_EXECUTABLE="$BUILD_PYTHON")
for name in ASCEND_CANN_PACKAGE_HOME BISHENG_CXX ASCENDC_RUNTIME_LIBRARY; do
    if [[ -n "${!name:-}" ]]; then
        CMAKE_ARGS+=("-D$name=${!name}")
    fi
done
cmake "${CMAKE_ARGS[@]}"
cmake --build "$EXPERIMENT_BUILD_DIR" --target simt_plan_variants_ops --parallel "$BUILD_JOBS"
echo "已构建: $EXPERIMENT_BUILD_DIR/libsimt_plan_variants_ops.so"
echo "下一步: $BUILD_PYTHON $SCRIPT_DIR/checks/check_mtp_scan.py --device 0 --library $EXPERIMENT_BUILD_DIR/libsimt_plan_variants_ops.so"
