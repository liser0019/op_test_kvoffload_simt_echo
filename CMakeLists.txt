# SPDX-License-Identifier: Apache-2.0
# 阅读顺序：依赖 → mixed SIMT 编译选项 → 本轮18个版本 → 合并共享库。
# 不使用算子自动发现、子目录注册宏、setup.py 或 wheel 打包。
cmake_minimum_required(VERSION 3.16)
project(simt_plan_variants LANGUAGES CXX)
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

set(SIMT_FLAGS
    --cce-auto-sync=off -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false
    -mllvm -cce-aicore-hoist-movemask=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(KERNEL_OPT_FLAGS -O0 -g)
else()
    set(KERNEL_OPT_FLAGS -O3)
endif()
option(PLAN_PROFILE_DEBUG_INFO "Add device -g while keeping -O3" OFF)
option(PLAN_RESOURCE_USAGE "Ask bisheng to print register/stack usage" OFF)
if(PLAN_PROFILE_DEBUG_INFO)
    list(APPEND SIMT_FLAGS -g)
endif()
if(PLAN_RESOURCE_USAGE)
    list(APPEND SIMT_FLAGS --cce-res-usage)
endif()
set(OP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/plan_variants/common")
set(KERNEL_INCLUDE_ARGS "")
foreach(_dir IN LISTS CANN_INCLUDE_DIRS OP_INCLUDE_DIRS)
    list(APPEND KERNEL_INCLUDE_ARGS "-I${_dir}")
endforeach()
file(GLOB COMMON_HEADERS CONFIGURE_DEPENDS
    "csrc/plan_variants/common/*.h" "csrc/scan_experiments/common/*.h")
set(PLAN_VARIANTS
    csrc/plan_variants/13_abcdf
    csrc/scan_experiments/v100
    csrc/scan_experiments/v010
    csrc/scan_experiments/v001
    csrc/scan_experiments/v110
    csrc/scan_experiments/v101
    csrc/scan_experiments/v011
    csrc/scan_experiments/v111
    csrc/scan_experiments/s01
    csrc/scan_experiments/s02
    csrc/scan_experiments/s03
    csrc/scan_experiments/s04
    csrc/scan_experiments/s05
    csrc/scan_experiments/s06
    csrc/scan_experiments/s07
    csrc/scan_experiments/s08
    csrc/scan_experiments/s09
    csrc/scan_experiments/cold_ballot)
set(KERNEL_OBJECTS "")
set(VARIANT_PLUGINS "")
option(PLAN_TRANSFER_ONLY "Build only frozen ABCDF and Transfer for full-cache experiments" OFF)
if(PLAN_TRANSFER_ONLY)
    set(PLAN_BUILD_VARIANTS csrc/plan_variants/13_abcdf)
else()
    set(PLAN_BUILD_VARIANTS ${PLAN_VARIANTS})
endif()
foreach(_variant IN LISTS PLAN_BUILD_VARIANTS)
    set(_dir "${CMAKE_CURRENT_SOURCE_DIR}/${_variant}")
    get_filename_component(_variant_name "${_dir}" NAME)
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS "${_dir}/*.h" "${_dir}/upstream/*")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${_headers} "${_dir}/kernel.cpp" "${_dir}/plugin.cpp")
    set(_object "${CMAKE_CURRENT_BINARY_DIR}/kernel_${_variant_name}.o")
    add_custom_command(OUTPUT "${_object}"
        COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
                ${KERNEL_OPT_FLAGS} ${SIMT_FLAGS} ${KERNEL_INCLUDE_ARGS}
                -I${_dir} -c "${_dir}/kernel.cpp" -o "${_object}"
        DEPENDS "${_dir}/kernel.cpp" ${_headers} ${COMMON_HEADERS}
                "${CMAKE_CURRENT_SOURCE_DIR}/plan_variants_manifest.json"
        COMMENT "bisheng compile ${_variant}" VERBATIM)
    set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    list(APPEND KERNEL_OBJECTS "${_object}")
    list(APPEND VARIANT_PLUGINS "${_dir}/plugin.cpp")
endforeach()
set(PLAN_INFO_CPP "${CMAKE_CURRENT_BINARY_DIR}/plan_build_info.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/plan_variants_manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/checks/plan_build_info.py" ${COMMON_HEADERS})
execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/checks/plan_build_info.py" --output "${PLAN_INFO_CPP}"
    "--compiler=${BISHENG_CXX}" "--flags=${KERNEL_OPT_FLAGS};${SIMT_FLAGS}"
    "--debug-info=${PLAN_PROFILE_DEBUG_INFO}" "--resource-usage=${PLAN_RESOURCE_USAGE}"
    RESULT_VARIABLE _info_result)
if(NOT _info_result EQUAL 0)
    message(FATAL_ERROR "Plan source identity check failed")
endif()
add_library(simt_plan_variants_ops SHARED ${VARIANT_PLUGINS} "${PLAN_INFO_CPP}" ${KERNEL_OBJECTS})
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(simt_plan_variants_ops PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_include_directories(simt_plan_variants_ops SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_DIR}/include"
    ${CANN_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${OP_INCLUDE_DIRS})
target_link_directories(simt_plan_variants_ops PRIVATE
    ${TORCH_LIB_DIR} ${TORCH_NPU_LIB_DIR} ${CANN_LIB_DIR} ${CANN_RUNTIME_LIB_DIR})
# 保留 main 已上机验证的库列表，尤其 platform / tiling_api / runtime。
target_link_libraries(simt_plan_variants_ops PRIVATE
    torch torch_cpu torch_python c10 torch_npu ascendcl tiling_api platform
    "${ASCENDC_RUNTIME_LIBRARY}")
set_target_properties(simt_plan_variants_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# 3. 原SparseKvTransfer的SIMD/AIV编译：不套用SIMT专用同步/缓存选项。
set(TRANSFER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc/transfer")
set(TRANSFER_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kernel_transfer.o")
set(TRANSFER_FLAGS -Wno-deprecated-declarations)
if(PLAN_PROFILE_DEBUG_INFO)
    list(APPEND TRANSFER_FLAGS -g)
endif()
if(PLAN_RESOURCE_USAGE)
    list(APPEND TRANSFER_FLAGS --cce-res-usage)
endif()
add_custom_command(OUTPUT "${TRANSFER_OBJECT}"
    COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
            ${KERNEL_OPT_FLAGS} ${TRANSFER_FLAGS} ${KERNEL_INCLUDE_ARGS}
            -I${TRANSFER_DIR} -c "${TRANSFER_DIR}/kernel.cpp" -o "${TRANSFER_OBJECT}"
    DEPENDS "${TRANSFER_DIR}/kernel.cpp" "${TRANSFER_DIR}/launch.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/transfer_manifest.json"
    COMMENT "bisheng compile SparseKvTransfer" VERBATIM)
set_source_files_properties("${TRANSFER_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
set(TRANSFER_INFO_CPP "${CMAKE_CURRENT_BINARY_DIR}/transfer_build_info.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/transfer_manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/checks/transfer_build_info.py")
execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/checks/transfer_build_info.py" --output "${TRANSFER_INFO_CPP}"
    "--compiler=${BISHENG_CXX}" "--flags=${KERNEL_OPT_FLAGS};${TRANSFER_FLAGS}"
    "--plan-transfer-only=${PLAN_TRANSFER_ONLY}"
    RESULT_VARIABLE _transfer_info_result)
if(NOT _transfer_info_result EQUAL 0)
    message(FATAL_ERROR "Transfer source identity check failed")
endif()
target_sources(simt_plan_variants_ops PRIVATE
    "${TRANSFER_DIR}/plugin.cpp" "${TRANSFER_DIR}/prepared.cpp"
    "${TRANSFER_INFO_CPP}" "${TRANSFER_OBJECT}")

# Transfer独立因子kernel：单一计算模板，16个唯一入口；原版object继续保留。
set(TRANSFER_VARIANT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc/transfer_variants")
set(TRANSFER_VARIANT_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kernel_transfer_variants.o")
add_custom_command(OUTPUT "${TRANSFER_VARIANT_OBJECT}"
    COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
            ${KERNEL_OPT_FLAGS} ${TRANSFER_FLAGS} ${KERNEL_INCLUDE_ARGS}
            -I${TRANSFER_VARIANT_DIR} -c "${TRANSFER_VARIANT_DIR}/kernel.cpp" -o "${TRANSFER_VARIANT_OBJECT}"
    DEPENDS "${TRANSFER_VARIANT_DIR}/kernel.cpp" "${TRANSFER_VARIANT_DIR}/launch.h"
            "${TRANSFER_VARIANT_DIR}/original_class.h" "${TRANSFER_VARIANT_DIR}/entries.inc"
            "${TRANSFER_DIR}/launch.h" "${CMAKE_CURRENT_SOURCE_DIR}/transfer_manifest.json"
    COMMENT "bisheng compile Transfer A/B/C/D factorial kernels" VERBATIM)
set_source_files_properties("${TRANSFER_VARIANT_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
target_sources(simt_plan_variants_ops PRIVATE "${TRANSFER_VARIANT_OBJECT}")
