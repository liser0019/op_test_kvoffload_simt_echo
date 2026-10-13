# SPDX-License-Identifier: Apache-2.0
# 阅读顺序：依赖 → 两种编程模式的选项 → 原算子及变体 → 合并共享库。
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

# 3. bisheng 分别生成原算子和各变体 object。两种编程模式的选项保留 main。
# SIMT 是 mixed 外壳 + VF；Echo 是 pure SIMT，不能用同一组编译选项。
set(SIMT_FLAGS
    --cce-auto-sync=off -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false
    -mllvm -cce-aicore-hoist-movemask=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false)
set(ECHO_FLAGS --enable-simt -Wno-deprecated-declarations)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(KERNEL_OPT_FLAGS -O0 -g)
else()
    set(KERNEL_OPT_FLAGS -O3)
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

# 11 个用户提供的 SIMT 版本，各自有独立 kernel/VF/launch/torch.ops 名称。
# 显式列出版本，新增版本必须同时更新 catalog，避免隐式加载其他代码。
set(SIMT_VARIANTS
    00_baseline 01_a_hit_short_circuit 02_b_parallel_scan 03_c_compact_partition
    04_d_cold_start 05_ab 06_abc 07_abcd 08_e_validity_markers 09_bce 10_bcde)
set(VARIANT_PLUGINS "")
foreach(_variant IN LISTS SIMT_VARIANTS)
    set(_variant_dir "${CMAKE_CURRENT_SOURCE_DIR}/csrc/variants/${_variant}")
    set(_source "${_variant_dir}/sparse_kv_plan_launch.cpp")
    set(_object "${CMAKE_CURRENT_BINARY_DIR}/kernel_${_variant}.o")
    add_custom_command(OUTPUT "${_object}"
        COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
                ${KERNEL_OPT_FLAGS} ${SIMT_FLAGS} ${KERNEL_INCLUDE_ARGS}
                -I${_variant_dir} -c "${_source}" -o "${_object}"
        DEPENDS "${_source}" "${_variant_dir}/sparse_kv_plan_compute.h"
                "${_variant_dir}/sparse_kv_plan_config.h" "csrc/variants/variant_tiling_data.h"
        COMMENT "bisheng 编译 ${_variant}" VERBATIM)
    set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    list(APPEND KERNEL_OBJECTS "${_object}")
    list(APPEND VARIANT_PLUGINS "${_variant_dir}/sparse_kv_plan_plugin.cpp")
endforeach()

# Peer sources and BCDE combinations have isolated kernel/VF/helper/op symbols.
set(PEER_VARIANTS
    peer_00_baseline
    peer_01_threads_1024
    peer_04_parallel_scan
    peer_05_fused_lookup
    peer_06_packed_lists
    combo_bcde_fused
    combo_bcde_1024
    combo_bcde_fused_1024
    combo_bcde_pair_scan)
foreach(_variant IN LISTS PEER_VARIANTS)
    set(_dir "${CMAKE_CURRENT_SOURCE_DIR}/csrc/peer_variants/${_variant}")
    file(GLOB _headers CONFIGURE_DEPENDS "${_dir}/*.h" "${_dir}/upstream/*")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${_headers} "${_dir}/kernel.cpp" "${_dir}/plugin.cpp")
    set(_object "${CMAKE_CURRENT_BINARY_DIR}/kernel_${_variant}.o")
    add_custom_command(OUTPUT "${_object}"
        COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
                ${KERNEL_OPT_FLAGS} ${SIMT_FLAGS} ${KERNEL_INCLUDE_ARGS}
                -I${_dir} -c "${_dir}/kernel.cpp" -o "${_object}"
        DEPENDS "${_dir}/kernel.cpp" ${_headers}
        COMMENT "bisheng compile ${_variant}" VERBATIM)
    set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    list(APPEND KERNEL_OBJECTS "${_object}")
    list(APPEND VARIANT_PLUGINS "${_dir}/plugin.cpp")
endforeach()
# Bake source identity into the library; refuse altered imports or a stale .so.
set(PEER_INFO_CPP "${CMAKE_CURRENT_BINARY_DIR}/peer_build_info.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/peer_variants_manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/checks/peer_build_info.py")
execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/checks/peer_build_info.py" --output "${PEER_INFO_CPP}"
    RESULT_VARIABLE _peer_info_result)
if(NOT _peer_info_result EQUAL 0)
    message(FATAL_ERROR "Peer source identity check failed")
endif()

# 4. 系统 C++ 编译各路 plugin，合并 kernel object，得到可被 torch.ops 加载的库。
# TORCH_LIBRARY 的静态初始化已负责注册，不需要 Python 包/pybind 入口。
add_library(simt_vs_echo_ops SHARED
    csrc/simt/sparse_kv_plan_plugin.cpp
    csrc/echo/echo_lru_plugin.cpp
    ${VARIANT_PLUGINS}
    "${PEER_INFO_CPP}"
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
set_target_properties(simt_vs_echo_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
