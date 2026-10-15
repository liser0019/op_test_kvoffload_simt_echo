#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CMake 配置 → 编译 BCDE fused 1024 与 17 个优化候选，不依赖原工程。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PYTHON="${PYTHON:-python3}"
BUILD_JOBS="${JOBS:-4}"
SOC_VERSION="ascend950"
for arg in "$@"; do
    case "$arg" in
        --soc=*) SOC_VERSION="${arg#*=}" ;;
        --help|-h)
            echo "用法: PYTHON=python3 JOBS=4 bash build.sh --soc=ascend950"
            echo "可覆盖: ASCEND_CANN_PACKAGE_HOME, BISHENG_CXX, ASCENDC_RUNTIME_LIBRARY"
            exit 0 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done
if [[ "$SOC_VERSION" != "ascend950" ]]; then
    echo "仅支持 --soc=ascend950" >&2
    exit 2
fi
# 使用激活环境中的确切解释器，也让 CMake 使用同一个解释器。
BUILD_PYTHON="$(command -v -- "$BUILD_PYTHON")"
CMAKE_ARGS=(
    -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build"
    -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="$SOC_VERSION"
    -DPython3_EXECUTABLE="$BUILD_PYTHON")
for name in ASCEND_CANN_PACKAGE_HOME BISHENG_CXX ASCENDC_RUNTIME_LIBRARY; do
    if [[ -n "${!name:-}" ]]; then
        CMAKE_ARGS+=("-D$name=${!name}")
    fi
done
cmake "${CMAKE_ARGS[@]}"
cmake --build "$SCRIPT_DIR/build" --target simt_plan_variants_ops --parallel "$BUILD_JOBS"
echo "已构建: $SCRIPT_DIR/build/libsimt_plan_variants_ops.so"
echo "下一步: $BUILD_PYTHON $SCRIPT_DIR/compare.py --device 1 --warmup 2 --repeat 4 --verify-each-sample"
