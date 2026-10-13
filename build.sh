#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 只做两件事：CMake 配置 → 编译本目录两个算子的共享库，不依赖原工程。
# ECHO_PERF_STEP=ON（或 --perf-step）构建带分步打点的 Echo kernel 到 build_perf/，
# 供 echo_perf_step.py 使用；普通构建（compare.py 用）不受影响，产物在 build/。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PYTHON="${PYTHON:-python3}"
BUILD_JOBS="${JOBS:-4}"
SOC_VERSION="ascend950"
PERF_STEP="${ECHO_PERF_STEP:-OFF}"
for arg in "$@"; do
    case "$arg" in
        --soc=*) SOC_VERSION="${arg#*=}" ;;
        --perf-step) PERF_STEP=ON ;;
        --help|-h)
            echo "用法: PYTHON=python3 JOBS=4 bash build.sh --soc=ascend950 [--perf-step]"
            echo "或:   ECHO_PERF_STEP=ON bash build.sh   # 分步打点构建, 产物在 build_perf/"
            echo "可覆盖: ASCEND_CANN_PACKAGE_HOME, BISHENG_CXX, ASCENDC_RUNTIME_LIBRARY"
            exit 0 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done
if [[ "$SOC_VERSION" != "ascend950" ]]; then
    echo "仅支持 --soc=ascend950" >&2
    exit 2
fi
if [[ "$PERF_STEP" != "ON" && "$PERF_STEP" != "OFF" ]]; then
    echo "ECHO_PERF_STEP 只能为 ON/OFF" >&2
    exit 2
fi
# 使用激活环境中的确切解释器，也让 CMake 使用同一个解释器。
BUILD_PYTHON="$(command -v -- "$BUILD_PYTHON")"
BUILD_DIR="$SCRIPT_DIR/build"
if [[ "$PERF_STEP" == "ON" ]]; then
    BUILD_DIR="$SCRIPT_DIR/build_perf"
fi
CMAKE_ARGS=(
    -S "$SCRIPT_DIR" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="$SOC_VERSION"
    -DECHO_PERF_STEP="$PERF_STEP"
    -DPython3_EXECUTABLE="$BUILD_PYTHON")
for name in ASCEND_CANN_PACKAGE_HOME BISHENG_CXX ASCENDC_RUNTIME_LIBRARY; do
    if [[ -n "${!name:-}" ]]; then
        CMAKE_ARGS+=("-D$name=${!name}")
    fi
done
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target simt_vs_echo_ops --parallel "$BUILD_JOBS"
echo "已构建: $BUILD_DIR/libsimt_vs_echo_ops.so"
if [[ "$PERF_STEP" == "ON" ]]; then
    echo "下一步: $BUILD_PYTHON $SCRIPT_DIR/echo_perf_step.py --device 1"
else
    echo "下一步: $BUILD_PYTHON $SCRIPT_DIR/compare.py --device 1 --warmup 2 --repeat 4 --verify-each-sample"
fi
