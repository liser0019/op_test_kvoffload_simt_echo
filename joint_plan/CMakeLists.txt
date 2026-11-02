# SPDX-License-Identifier: Apache-2.0
# 原独立ABCDF + 联合框架private/joint：相同-O3与SIMT编译选项。
# 不使用算子自动发现、子目录注册宏、setup.py 或 wheel 打包。
cmake_minimum_required(VERSION 3.16)
project(mtp_joint_plan LANGUAGES CXX)
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


get_filename_component(JOINT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(JOINT_DIR "${JOINT_ROOT}/csrc/joint_plan")
set(JOINT_FLAGS -O3 --cce-auto-sync=off -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false)
option(JOINT_DEBUG_INFO "Device -g with -O3" OFF)
option(JOINT_RESOURCE_USAGE "Compiler resource usage" OFF)
if(JOINT_DEBUG_INFO)
    list(APPEND JOINT_FLAGS -g)
endif()
if(JOINT_RESOURCE_USAGE)
    list(APPEND JOINT_FLAGS --cce-res-usage)
endif()
set(JOINT_INCLUDES "-I${JOINT_DIR}")
foreach(_dir IN LISTS CANN_INCLUDE_DIRS)
    list(APPEND JOINT_INCLUDES "-I${_dir}")
endforeach()
set(JOINT_INFO "${CMAKE_CURRENT_BINARY_DIR}/source_info.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${JOINT_ROOT}/joint_plan_manifest.json" "${JOINT_ROOT}/checks/joint_build_info.py")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${JOINT_ROOT}/checks/joint_build_info.py"
    --output "${JOINT_INFO}" "--compiler=${BISHENG_CXX}" "--flags=${JOINT_FLAGS}"
    RESULT_VARIABLE _identity_result)
if(NOT _identity_result EQUAL 0)
    message(FATAL_ERROR "Joint Plan source identity check failed")
endif()
set(JOINT_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/joint_plan.o")
add_custom_command(OUTPUT "${JOINT_OBJECT}"
    COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
        ${JOINT_FLAGS} ${JOINT_INCLUDES} -c "${JOINT_DIR}/kernel.cpp" -o "${JOINT_OBJECT}"
    DEPENDS "${JOINT_DIR}/kernel.cpp" "${JOINT_DIR}/launch.h"
        "${JOINT_DIR}/sparse_kv_joint_plan.h" "${JOINT_DIR}/sparse_kv_plan_config.h"
        "${JOINT_ROOT}/joint_plan_manifest.json"
    COMMENT "Compile preserved ABCDF + MTP joint Plan" VERBATIM)
set_source_files_properties("${JOINT_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
# Compile independent 13_abcdf; source-page validation has been removed from its shared ABI.
# This path never calls RunPrivateRow and never publishes joint metadata.
set(ABCDF_DIR "${JOINT_ROOT}/csrc/plan_variants/13_abcdf")
set(ABCDF_COMMON "${JOINT_ROOT}/csrc/plan_variants/common")
set(ABCDF_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/abcdf_original.o")
file(GLOB ABCDF_HEADERS CONFIGURE_DEPENDS "${ABCDF_DIR}/*.h" "${ABCDF_COMMON}/*.h")
add_custom_command(OUTPUT "${ABCDF_OBJECT}"
    COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
        ${JOINT_FLAGS} ${JOINT_INCLUDES} -I${ABCDF_DIR} -I${ABCDF_COMMON}
        -c "${ABCDF_DIR}/kernel.cpp" -o "${ABCDF_OBJECT}"
    DEPENDS "${ABCDF_DIR}/kernel.cpp" ${ABCDF_HEADERS} "${JOINT_ROOT}/joint_plan_manifest.json"
    COMMENT "Compile original independent 13_abcdf (mask47, 1024 threads)" VERBATIM)
set_source_files_properties("${ABCDF_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
# Independent experiment translation units; all active Plan variants share the no-source-page contract.
set(JOINT_VARIANT_DIR "${JOINT_ROOT}/csrc/joint_variants")
set(JOINT_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/joint_variants")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${JOINT_ROOT}/joint_variants.py" "${JOINT_ROOT}/checks/generate_joint_variants.py")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${JOINT_ROOT}/checks/generate_joint_variants.py"
    --output "${JOINT_GENERATED}" RESULT_VARIABLE _variant_result)
if(NOT _variant_result EQUAL 0)
    message(FATAL_ERROR "Failed to generate Joint variant symbols")
endif()
set(JOINT_VARIANT_OBJECTS)
foreach(_name a b ab c ac bc abc d ad bd abd cd acd bcd abcd bitmap turbo turbo_a turbo_b turbo_ab)
    set(_object "${JOINT_GENERATED}/variant_${_name}.o")
    add_custom_command(OUTPUT "${_object}"
        COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
            ${JOINT_FLAGS} ${JOINT_INCLUDES} -c "${JOINT_GENERATED}/variant_${_name}.cpp" -o "${_object}"
        DEPENDS "${JOINT_GENERATED}/variant_${_name}.cpp" "${JOINT_VARIANT_DIR}/kernel.cpp"
            "${JOINT_VARIANT_DIR}/sparse_kv_joint_plan.h" "${JOINT_VARIANT_DIR}/options.h" "${JOINT_VARIANT_DIR}/bitmap_plan.h"
            "${JOINT_VARIANT_DIR}/turbo_plan.h" "${JOINT_VARIANT_DIR}/turbo_union_experiments.h"
            "${JOINT_DIR}/launch.h" "${JOINT_DIR}/sparse_kv_plan_config.h"
            "${JOINT_ROOT}/joint_plan_manifest.json"
        COMMENT "Compile Joint experiment ${_name}" VERBATIM)
    set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    list(APPEND JOINT_VARIANT_OBJECTS "${_object}")
endforeach()
set(ABCDF_PAGES_DIR "${JOINT_ROOT}/csrc/abcdf_pages")
set(ABCDF_PAGES_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/abcdf_pages.o")
file(GLOB ABCDF_PAGES_HEADERS CONFIGURE_DEPENDS "${ABCDF_PAGES_DIR}/*.h")
add_custom_command(OUTPUT "${ABCDF_PAGES_OBJECT}"
    COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
        ${JOINT_FLAGS} ${JOINT_INCLUDES} -I${ABCDF_PAGES_DIR}
        -c "${ABCDF_PAGES_DIR}/kernel.cpp" -o "${ABCDF_PAGES_OBJECT}"
    DEPENDS "${ABCDF_PAGES_DIR}/kernel.cpp" ${ABCDF_PAGES_HEADERS}
        "${JOINT_ROOT}/joint_plan_manifest.json" VERBATIM)
set_source_files_properties("${ABCDF_PAGES_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
add_library(mtp_joint_plan_ops SHARED "${ABCDF_PAGES_DIR}/plugin.cpp" "${ABCDF_PAGES_OBJECT}" "${JOINT_DIR}/plugin.cpp" "${JOINT_INFO}"
    "${JOINT_OBJECT}" "${ABCDF_DIR}/plugin.cpp" "${ABCDF_OBJECT}"
    "${JOINT_VARIANT_DIR}/plugin.cpp" ${JOINT_VARIANT_OBJECTS})
target_include_directories(mtp_joint_plan_ops PRIVATE "${JOINT_GENERATED}")
separate_arguments(TORCH_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(mtp_joint_plan_ops PRIVATE ${TORCH_FLAGS_LIST})
target_include_directories(mtp_joint_plan_ops SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_DIR}/include" ${CANN_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_directories(mtp_joint_plan_ops PRIVATE ${TORCH_LIB_DIR} ${TORCH_NPU_LIB_DIR}
    ${CANN_LIB_DIR} ${CANN_RUNTIME_LIB_DIR})
target_link_libraries(mtp_joint_plan_ops PRIVATE torch torch_cpu torch_python c10 torch_npu
    ascendcl tiling_api platform "${ASCENDC_RUNTIME_LIBRARY}")
