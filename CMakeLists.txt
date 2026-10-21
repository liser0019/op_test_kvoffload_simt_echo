# SPDX-License-Identifier: Apache-2.0
# 阅读顺序：依赖 → 两个 kernel 的编译选项 → 两个 plugin → 合并共享库。
# 不使用算子自动发现、子目录注册宏、setup.py 或 wheel 打包。
cmake_minimum_required(VERSION 3.16)
project(simt_vs_echo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 1. CANN 与 bisheng：覆盖入口与原 main 一致。
set(NPU_ARCH "ascend950" CACHE STRING "目标 SoC")
if(NOT NPU_ARCH STREQUAL "ascend950")
    message(FATAL_ERROR "本目录仅测试 Ascend 950 / dav-3510")
endif()
set(ASCEND_CANN_PACKAGE_HOME "" CACHE PATH "CANN 根目录")
set(_cann_candidates
    "${ASCEND_CANN_PACKAGE_HOME}"
    "$ENV{ASCEND_CANN_PACKAGE_HOME}"
    "$ENV{ASCEND_HOME_PATH}"
    "/usr/local/Ascend/ascend-toolkit/latest")
set(_cann_found "")
foreach(_candidate IN LISTS _cann_candidates)
    if(NOT _cann_found AND EXISTS "${_candidate}/include")
        set(_cann_found "${_candidate}")
    endif()
endforeach()
if(NOT _cann_found)
    message(FATAL_ERROR "未找到 CANN，请 source set_env.sh 或设置 ASCEND_CANN_PACKAGE_HOME")
endif()
set(ASCEND_CANN_PACKAGE_HOME "${_cann_found}" CACHE PATH "CANN 根目录" FORCE)
if(NOT BISHENG_CXX)
    find_program(BISHENG_CXX NAMES bisheng_compiler bisheng ccec_compiler
        PATHS "${ASCEND_CANN_PACKAGE_HOME}/bin"
              "${ASCEND_CANN_PACKAGE_HOME}/compiler/bin"
              "${ASCEND_CANN_PACKAGE_HOME}/compiler/bisheng/bin"
              "${ASCEND_CANN_PACKAGE_HOME}/compiler/tbe/bin")
endif()
if(NOT BISHENG_CXX)
    message(FATAL_ERROR "未找到 bisheng，可设置 BISHENG_CXX")
endif()
set(CANN_INCLUDE_DIRS "${ASCEND_CANN_PACKAGE_HOME}/include")
if(EXISTS "${ASCEND_CANN_PACKAGE_HOME}/asc/include")
    list(APPEND CANN_INCLUDE_DIRS "${ASCEND_CANN_PACKAGE_HOME}/asc/include")
endif()
set(CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_HOME}/lib64")
if(NOT EXISTS "${CANN_LIB_DIR}")
    set(CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_HOME}/lib")
endif()
# 保留服务器 CANN 9.1.0 的 runtime 静态库；另一个安装布局可显式覆盖。
set(ASCENDC_RUNTIME_LIBRARY
    "${ASCEND_CANN_PACKAGE_HOME}/aarch64-linux/lib64/libascendc_runtime.a"
    CACHE FILEPATH "AscendC runtime 静态库")
if(NOT EXISTS "${ASCENDC_RUNTIME_LIBRARY}")
    message(FATAL_ERROR "缺少 ${ASCENDC_RUNTIME_LIBRARY}，请设置 ASCENDC_RUNTIME_LIBRARY")
endif()
get_filename_component(CANN_RUNTIME_LIB_DIR "${ASCENDC_RUNTIME_LIBRARY}" DIRECTORY)

# 2. 始终查询本次选择的 Python 环境，避免链接另一套 torch / torch_npu。
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE _torch_result OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _torch_result EQUAL 0)
    message(FATAL_ERROR "当前 Python 无法 import torch")
endif()
find_package(Torch REQUIRED PATHS ${TORCH_CMAKE_PREFIX} NO_DEFAULT_PATH)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    RESULT_VARIABLE _torch_lib_result OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _torch_lib_result EQUAL 0 OR NOT EXISTS "${TORCH_LIB_DIR}/libtorch_python.so")
    message(FATAL_ERROR "当前 torch/lib 中找不到 libtorch_python.so: ${TORCH_LIB_DIR}")
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    RESULT_VARIABLE _npu_result OUTPUT_VARIABLE TORCH_NPU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _npu_result EQUAL 0)
    message(FATAL_ERROR "当前 Python 无法 import torch_npu")
endif()
set(TORCH_NPU_LIB_DIR "${TORCH_NPU_DIR}/lib")
if(NOT EXISTS "${TORCH_NPU_LIB_DIR}")
    set(TORCH_NPU_LIB_DIR "${TORCH_NPU_DIR}")
endif()
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "CANN: ${ASCEND_CANN_PACKAGE_HOME}")
message(STATUS "bisheng: ${BISHENG_CXX}")
message(STATUS "torch lib: ${TORCH_LIB_DIR}; torch_npu: ${TORCH_NPU_DIR}")

# 3. bisheng 单独生成两个 object。两种编程模式的选项原样保留 main。
# SIMT 是 mixed 外壳 + VF；Echo 是 pure SIMT，不能用同一组编译选项。
# ECHO_PERF_STEP=ON 时给 Echo kernel 与 plugin 同时定义 ECHO_PERF_STEP 宏：
# kernel 在 workspace 尾部写每核分步 clock() 周期（EchoWorkspaceElements 同步扩容），
# 供 echo_perf_step.py 读取；会改变计时特性，正式性能对比（compare.py）请保持 OFF。
option(ECHO_PERF_STEP "Instrument echo_lru per-step cycles into workspace" OFF)
option(SIMT_PERF_STEP "Instrument imported ABCDF per-row stage stamps" OFF)
# Echo 每核线程数（消融实验用，32 的倍数，[64,1024]）。
# ⚠ kernel（下方 ECHO_FLAGS）与 plugin（下方 target_compile_definitions）两处必须注入
# 同一个 ECHO_LRU_THREAD_NUM_OVERRIDE 值：plugin 用 ECHO_LRU_THREAD_NUM 做
# <<<blockNum, ECHO_LRU_THREAD_NUM>>> 的 blockDim，两处不同值会导致
# kernel __launch_bounds__ 与实际启动线程数静默错位。
set(ECHO_LRU_THREADS "1024" CACHE STRING "Echo 每核线程数（消融实验用，32 的倍数，[64,1024]）")
set(SIMT_FLAGS
    --cce-auto-sync=off -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false
    -mllvm -cce-aicore-hoist-movemask=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false)
if(SIMT_PERF_STEP)
    list(APPEND SIMT_FLAGS -DSIMT_PERF_STEP)
endif()
# 与 compare.simt_source_identity() 逐字同序。编译时冻结身份，禁止误用旧 so。
set(SIMT_IDENTITY_FILES
    csrc/simt/sparse_kv_plan_config.h
    csrc/simt/sparse_kv_plan_kernel.cpp
    csrc/simt/sparse_kv_plan_launch.h
    csrc/simt/sparse_kv_plan_perf.h
    csrc/simt/sparse_kv_plan_plugin.cpp
    csrc/simt/variant_config.h)
set(_simt_fingerprint "")
foreach(_file IN LISTS SIMT_IDENTITY_FILES)
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_file}" _sha)
    string(APPEND _simt_fingerprint "${_file}:${_sha}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
endforeach()
string(SHA256 SIMT_SOURCE_SHA256 "${_simt_fingerprint}")
set(ECHO_FLAGS --enable-simt -Wno-deprecated-declarations)
# 与 plugin 侧 target_compile_definitions 的 ECHO_LRU_THREAD_NUM_OVERRIDE 必须同值（见上）。
list(APPEND ECHO_FLAGS -DECHO_LRU_THREAD_NUM_OVERRIDE=${ECHO_LRU_THREADS})
if(ECHO_PERF_STEP)
    list(APPEND ECHO_FLAGS -DECHO_PERF_STEP)
endif()
# [v2.1-prof 2026-10-20] ECHO_PERF_DRAIN=ON (需配 ECHO_PERF_STEP=ON): PerfEnd 读
#   clock 前先 SyncGrid 排空写缓冲, 量"真实边界 vs 记账边界"(deep_dive §9.23)。
#   强扰动旋钮 (每标记 +一道栅栏价), 结论仅本二进制内有效; build.sh 将其产物
#   导向 build_v2dbg/。仅 kernel 侧需要该宏 (plugin 的 EchoWorkspaceElements
#   不引用), 正式构建 (OFF) 编译选项零变化。
set(ECHO_PERF_DRAIN "OFF" CACHE STRING "echo_lru perf: SyncGrid-drain before each PerfEnd stamp (ON/OFF)")
if(NOT ECHO_PERF_DRAIN MATCHES "^(ON|OFF)$")
    message(FATAL_ERROR "ECHO_PERF_DRAIN 只能为 ON/OFF")
endif()
if(ECHO_PERF_DRAIN STREQUAL "ON" AND NOT ECHO_PERF_STEP)
    message(FATAL_ERROR "ECHO_PERF_DRAIN=ON 需要 ECHO_PERF_STEP=ON")
endif()
if(ECHO_PERF_DRAIN STREQUAL "ON")
    list(APPEND ECHO_FLAGS -DECHO_PERF_DRAIN)
endif()
# [v2 debug] ECHO_V2_DEBUG_CUTOFF=N: Process 在第 N 阶段后提前 return (仅 kernel 侧,
#   用于二分定位设备侧 trap; 产物目录 build_v2dbg/, 不覆盖正式构建)。
#   阶段: 1=prologue 2=Step0 3=Step1 4=Step2 5=Step3 6=Step4 7/0=全程。
set(ECHO_V2_DEBUG_CUTOFF "0" CACHE STRING "echo_lru v2 debug: early-return after stage N (0=off)")
if(NOT ECHO_V2_DEBUG_CUTOFF STREQUAL "0")
    list(APPEND ECHO_FLAGS -DECHO_V2_DEBUG_CUTOFF=${ECHO_V2_DEBUG_CUTOFF})
endif()
# [v2.1 2026-10-20] ECHO_V2_SKIP_REBUILD 与 9 个 ECHO_V2_RB_* 旋钮已删: 它们服务
#   旧 LruRebuild 的栈溢出排查 (§9.18), 重建路径已随单趟分类重构消失。旋钮结论
#   本就不跨二进制 (每个旋钮=不同栈帧形态), 旧实验记录见 git 历史与 deep-dive。

# [v2 debug] Echo kernel 的 bisheng 优化级别 (默认 -O3)。v2 trap 排查: 同源码不同
#   优化级别 = 零语义改动的 codegen 判别实验 (§7.9 先例: 本栈存在编译单元整体
#   属性触发的设备 fault)。
set(ECHO_KERNEL_OPT_LEVEL "-O3" CACHE STRING "bisheng optimization level for echo kernel (-O0/-O1/-O2/-O3)")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(KERNEL_OPT_FLAGS ${ECHO_KERNEL_OPT_LEVEL})
else()
    set(KERNEL_OPT_FLAGS ${ECHO_KERNEL_OPT_LEVEL})
endif()
set(OP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/simt" "${CMAKE_CURRENT_SOURCE_DIR}/csrc/echo")
set(KERNEL_INCLUDE_ARGS "")
foreach(_dir IN LISTS CANN_INCLUDE_DIRS OP_INCLUDE_DIRS)
    list(APPEND KERNEL_INCLUDE_ARGS "-I${_dir}")
endforeach()
file(GLOB KERNEL_HEADERS CONFIGURE_DEPENDS "csrc/simt/*.h" "csrc/echo/*.h")
set(KERNEL_OBJECTS "")
foreach(_name IN ITEMS simt echo)
    if(_name STREQUAL "simt")
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/csrc/simt/sparse_kv_plan_kernel.cpp")
        set(_flags ${SIMT_FLAGS})
    else()
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/csrc/echo/echo_lru_launch.cpp")
        set(_flags ${ECHO_FLAGS})
    endif()
    set(_object "${CMAKE_CURRENT_BINARY_DIR}/kernel_${_name}.o")
    add_custom_command(OUTPUT "${_object}"
        COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
                ${KERNEL_OPT_FLAGS} ${_flags} ${KERNEL_INCLUDE_ARGS}
                -c "${_source}" -o "${_object}"
        DEPENDS "${_source}" ${KERNEL_HEADERS}
        COMMENT "bisheng 编译 ${_name} kernel" VERBATIM)
    list(APPEND KERNEL_OBJECTS "${_object}")
endforeach()
set_source_files_properties(${KERNEL_OBJECTS} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

# 4. 系统 C++ 编译两个 plugin，合并 kernel object，得到可被 torch.ops 加载的库。
# TORCH_LIBRARY 的静态初始化已负责注册，不需要 Python 包/pybind 入口。
add_library(simt_vs_echo_ops SHARED
    csrc/simt/sparse_kv_plan_plugin.cpp
    csrc/echo/echo_lru_plugin.cpp
    ${KERNEL_OBJECTS})
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(simt_vs_echo_ops PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_include_directories(simt_vs_echo_ops SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_DIR}/include"
    ${CANN_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${OP_INCLUDE_DIRS})
target_link_directories(simt_vs_echo_ops PRIVATE
    ${TORCH_LIB_DIR} ${TORCH_NPU_LIB_DIR} ${CANN_LIB_DIR} ${CANN_RUNTIME_LIB_DIR})
# 保留 main 已上机验证的库列表，尤其 platform / tiling_api / runtime。
target_link_libraries(simt_vs_echo_ops PRIVATE
    torch torch_cpu torch_python c10 torch_npu ascendcl tiling_api platform
    "${ASCENDC_RUNTIME_LIBRARY}")
if(ECHO_PERF_STEP)
    target_compile_definitions(simt_vs_echo_ops PRIVATE ECHO_PERF_STEP)
endif()
if(SIMT_PERF_STEP)
    target_compile_definitions(simt_vs_echo_ops PRIVATE SIMT_PERF_STEP)
endif()
target_compile_definitions(simt_vs_echo_ops PRIVATE SIMT_SOURCE_SHA256="${SIMT_SOURCE_SHA256}")
# 与 ECHO_FLAGS 里 kernel 侧的 -DECHO_LRU_THREAD_NUM_OVERRIDE 必须同值：
# plugin 用 ECHO_LRU_THREAD_NUM 做 <<<>>> 的 blockDim（echo_lru_launch.cpp）。
target_compile_definitions(simt_vs_echo_ops PRIVATE ECHO_LRU_THREAD_NUM_OVERRIDE=${ECHO_LRU_THREADS})
set_target_properties(simt_vs_echo_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
