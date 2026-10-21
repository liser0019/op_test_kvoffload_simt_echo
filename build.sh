#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 只做两件事：CMake 配置 → 编译本目录两个算子的共享库，不依赖原工程。
# ECHO_PERF_STEP=ON（或 --perf-step）构建带分步打点的 Echo kernel 到 build_perf/，
# 供 echo_perf_step.py 使用；普通构建（compare.py 用）不受影响，产物在 build/。
# ECHO_PERF_DRAIN=ON（[v2.1-prof 2026-10-20]，需配 ECHO_PERF_STEP=ON）：打点构建的
# 扰动测量模式——PerfEnd 读 clock 前先 SyncGrid 排空写缓冲，量"真实边界 vs 记账
# 边界"差；每标记 +一道栅栏价，强扰动、结论仅本二进制有效，产物进 build_v2dbg/。
# --threads=N（或环境变量 ECHO_LRU_THREADS，默认 1024）覆盖 Echo 每核线程数，
# 透传 -DECHO_LRU_THREADS=N 给 CMake；N != 1024 时产物改放 build_t{N} /
# build_perf_t{N}，避免不同线程数配置互相覆盖构建产物。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PYTHON="${PYTHON:-python3}"
BUILD_JOBS="${JOBS:-4}"
SOC_VERSION="ascend950"
PERF_STEP="${ECHO_PERF_STEP:-OFF}"
SIMT_PERF="${SIMT_PERF_STEP:-OFF}"
PERF_DRAIN="${ECHO_PERF_DRAIN:-OFF}"
ECHO_THREADS="${ECHO_LRU_THREADS:-1024}"
V2_CUTOFF="${ECHO_V2_DEBUG_CUTOFF:-0}"
# [v2.1 2026-10-20] ECHO_V2_SKIP_REBUILD 与 9 个 ECHO_V2_RB_* 旋钮已删: 它们服务
#   旧 LruRebuild 的栈溢出排查 (§9.18), 重建路径已随单趟分类重构消失。旋钮结论
#   本就不跨二进制 (每个旋钮=不同栈帧形态), 旧实验记录见 git 历史与 deep-dive。
V2_KOPT="${ECHO_V2_KOPT:--O3}"
for arg in "$@"; do
    case "$arg" in
        --soc=*) SOC_VERSION="${arg#*=}" ;;
        --perf-step) PERF_STEP=ON ;;
        --simt-perf-step) SIMT_PERF=ON ;;
        --threads=*) ECHO_THREADS="${arg#*=}" ;;
        --help|-h)
            echo "用法: PYTHON=python3 JOBS=4 bash build.sh --soc=ascend950 [--perf-step] [--threads=N]"
            echo "或:   ECHO_PERF_STEP=ON bash build.sh      # 分步打点构建, 产物在 build_perf/"
            echo "或:   SIMT_PERF_STEP=ON bash build.sh      # ABCDF 打点构建, 产物在 build_simt_perf/"
            echo "或:   ECHO_PERF_STEP=ON ECHO_PERF_DRAIN=ON bash build.sh  # 打点+每标记排空(扰动测量), 产物在 build_v2dbg/"
            echo "或:   ECHO_LRU_THREADS=512 bash build.sh   # 等价 --threads=512, 产物在 build_t512/"
            echo "--threads=N / 环境变量 ECHO_LRU_THREADS: Echo 每核线程数 (32 的倍数, [64,1024], 默认 1024);"
            echo "  N != 1024 时构建目录为 build_t{N} (打点构建为 build_perf_t{N}), 与默认产物互不覆盖。"
            echo "ECHO_PERF_DRAIN=ON: PerfEnd 读 clock 前先 SyncGrid 排空写缓冲 (量'真实边界 vs 记账边界'),"
            echo "  需配 ECHO_PERF_STEP=ON; 强扰动(每标记 +一道栅栏价), 结论仅本二进制内有效, 产物进 build_v2dbg/。"
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
if [[ "$SIMT_PERF" != "ON" && "$SIMT_PERF" != "OFF" ]]; then
    echo "SIMT_PERF_STEP 只能为 ON/OFF" >&2
    exit 2
fi
if [[ "$PERF_DRAIN" != "ON" && "$PERF_DRAIN" != "OFF" ]]; then
    echo "ECHO_PERF_DRAIN 只能为 ON/OFF" >&2
    exit 2
fi
if [[ "$PERF_DRAIN" == "ON" && "$PERF_STEP" != "ON" ]]; then
    # 排空语义依附于打点 (PerfEnd), 打点不开则旋钮无从生效 —— 显式拒绝而非静默忽略,
    #   防止"以为在测排空边界, 实际是普通构建"的错位结论 (测量工具最忌静默失效)。
    echo "ECHO_PERF_DRAIN=ON 需要 ECHO_PERF_STEP=ON (打点不开则排空无从谈起)" >&2
    exit 2
fi
if ! [[ "$ECHO_THREADS" =~ ^[0-9]+$ ]] || (( ECHO_THREADS % 32 != 0 || ECHO_THREADS < 64 || ECHO_THREADS > 1024 )); then
    echo "--threads / ECHO_LRU_THREADS 必须是 32 的倍数且在 [64,1024]: $ECHO_THREADS" >&2
    exit 2
fi
if ! [[ "$V2_KOPT" =~ ^-O[0123]$ ]]; then
    echo "ECHO_V2_KOPT 必须是 -O0/-O1/-O2/-O3: $V2_KOPT" >&2
    exit 2
fi
if ! [[ "$V2_CUTOFF" =~ ^[0-9]$ ]]; then
    echo "ECHO_V2_DEBUG_CUTOFF 必须是 0..9 (0=正式构建产物build/; 1..9=调试构建产物build_v2dbg/, 阶段: 1=prologue 2=Step0 3=Step1 4=Step2后 5=Step3后 6=Step4后 7=MissCumsum后 8=Step5.SyncGrid后 9=全程): $V2_CUTOFF" >&2
    exit 2
fi
# 使用激活环境中的确切解释器，也让 CMake 使用同一个解释器。
BUILD_PYTHON="$(command -v -- "$BUILD_PYTHON")"
BUILD_DIR="$SCRIPT_DIR/build"
if [[ "$PERF_STEP" == "ON" ]]; then
    BUILD_DIR="$SCRIPT_DIR/build_perf"
fi
if [[ "$V2_CUTOFF" != "0" || "$V2_KOPT" != "-O3" || "$PERF_DRAIN" == "ON" ]]; then
    # v2 调试构建 (阶段截断/优化档/打点排空扰动模式) 使用独立目录, 不覆盖正式产物。
    BUILD_DIR="$SCRIPT_DIR/build_v2dbg"
fi
if [[ "$ECHO_THREADS" != "1024" ]]; then
    # 线程数消融配置使用独立构建目录, 避免覆盖默认 1024 线程的产物。
    if [[ "$PERF_STEP" == "ON" ]]; then
        BUILD_DIR="$SCRIPT_DIR/build_perf_t${ECHO_THREADS}"
    else
        BUILD_DIR="$SCRIPT_DIR/build_t${ECHO_THREADS}"
    fi
fi
# ABCDF 打点独立存放；调试/线程上限亦编码在目录中，避免互相覆盖。
if [[ "$SIMT_PERF" == "ON" ]]; then
    BUILD_DIR="$SCRIPT_DIR/build_simt_perf"
    [[ "$PERF_STEP" == "ON" ]] && BUILD_DIR="$SCRIPT_DIR/build_both_perf"
    [[ "$ECHO_THREADS" != "1024" ]] && BUILD_DIR="${BUILD_DIR}_t${ECHO_THREADS}"
    if [[ "$V2_CUTOFF" != "0" || "$V2_KOPT" != "-O3" || "$PERF_DRAIN" == "ON" ]]; then
        BUILD_DIR="${BUILD_DIR}_debug"
    fi
fi
CMAKE_ARGS=(
    -S "$SCRIPT_DIR" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="$SOC_VERSION"
    -DECHO_PERF_STEP="$PERF_STEP" -DECHO_PERF_DRAIN="$PERF_DRAIN" -DECHO_LRU_THREADS="$ECHO_THREADS" -DECHO_V2_DEBUG_CUTOFF="$V2_CUTOFF"
    -DSIMT_PERF_STEP="$SIMT_PERF"
    -DECHO_KERNEL_OPT_LEVEL="$V2_KOPT"
    -DPython3_EXECUTABLE="$BUILD_PYTHON")
for name in ASCEND_CANN_PACKAGE_HOME BISHENG_CXX ASCENDC_RUNTIME_LIBRARY; do
    if [[ -n "${!name:-}" ]]; then
        CMAKE_ARGS+=("-D$name=${!name}")
    fi
done
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target simt_vs_echo_ops --parallel "$BUILD_JOBS"
echo "已构建: $BUILD_DIR/libsimt_vs_echo_ops.so"
if [[ "$SIMT_PERF" == "ON" ]]; then
    echo "下一步: $BUILD_PYTHON $SCRIPT_DIR/simt_perf_step.py --library $BUILD_DIR/libsimt_vs_echo_ops.so --device 1"
elif [[ "$PERF_STEP" == "ON" ]]; then
    echo "下一步: $BUILD_PYTHON $SCRIPT_DIR/echo_perf_step.py --device 1"
else
    echo "下一步: $BUILD_PYTHON $SCRIPT_DIR/compare.py --device 1 --warmup 2 --repeat 4 --verify-each-sample"
fi
