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
    -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false)
option(JOINT_REFINE_ONLY "Build frozen bounds512 and refinement factorial" OFF)
option(JOINT_EXEC_ONLY "Build ABD512 execution factorial and frozen references" OFF)
option(JOINT_CAPACITY8K_ONLY "Build two bounded capacity8192 FUSED/AB adapters" OFF)
option(JOINT_FUSED_TUNE_ONLY "Build frozen FUSED and seven A/B/C refinement candidates" OFF)
option(JOINT_MERGE_UNIQUE_ONLY "Build frozen196 plus intermediate unique and fused count candidates" OFF)
option(JOINT_PREFIX_ONLY "Build frozen reuse_parallel and A/B/AB only" OFF)
option(JOINT_PEER_ONLY "Build peer standalone Plan and one current candidate" OFF)
option(JOINT_MTP7_PACK_ONLY "Build BD8 parallel/reuse/tile controls only" OFF)
option(JOINT_MTP7_FIX_ONLY "Build BD8 reference and2 MTP7 parallel candidates" OFF)
option(JOINT_MTP8_ONLY "Build only BD and8-row BD candidate plus independent ABCDF" OFF)
option(JOINT_HOT_ONLY "Build ABD512 hot pipeline factorial and frozen references" OFF)
option(JOINT_INTEGRATED_ONLY "Build integrated512, Phase0 bounds candidate and two cold references only" OFF)
option(JOINT_COLD_COMPARE_ONLY "Build six-way cold finish comparison only" OFF)
option(JOINT_AC_THREADS_ONLY "Build only AC512/AC256 thread controls and baselines" OFF)
option(JOINT_UNION_ONLY "Build AC512 reference + new ABCD/thread candidates" OFF)
option(JOINT_SWEEP_ONLY "Build hardware sweep and its validated reference" OFF)
option(JOINT_SORTED_ONLY "Build sorted union and Merge BD references" OFF)
option(JOINT_MERGE_ONLY "Build Merge candidates without old Joint experiments" OFF)
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
    --output "${JOINT_INFO}" "--compiler=${BISHENG_CXX}" "--flags=${JOINT_FLAGS};merge_only=${JOINT_MERGE_ONLY};sorted_only=${JOINT_SORTED_ONLY};sweep_only=${JOINT_SWEEP_ONLY};union_only=${JOINT_UNION_ONLY};ac_threads_only=${JOINT_AC_THREADS_ONLY};cold_compare_only=${JOINT_COLD_COMPARE_ONLY};integrated_only=${JOINT_INTEGRATED_ONLY};refine_only=${JOINT_REFINE_ONLY};exec_only=${JOINT_EXEC_ONLY};hot_only=${JOINT_HOT_ONLY};mtp8_only=${JOINT_MTP8_ONLY};mtp7_fix_only=${JOINT_MTP7_FIX_ONLY};mtp7_pack_only=${JOINT_MTP7_PACK_ONLY};peer_only=${JOINT_PEER_ONLY};prefix_only=${JOINT_PREFIX_ONLY};merge_unique_only=${JOINT_MERGE_UNIQUE_ONLY};fused_tune_only=${JOINT_FUSED_TUNE_ONLY};capacity8k_only=${JOINT_CAPACITY8K_ONLY}"
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
set(_merge_generate_args)
if(JOINT_CAPACITY8K_ONLY)
    list(APPEND _merge_generate_args --capacity8k-only)
elseif(JOINT_FUSED_TUNE_ONLY)
    list(APPEND _merge_generate_args --fused-tune-only)
elseif(JOINT_MERGE_UNIQUE_ONLY)
    list(APPEND _merge_generate_args --merge-unique-only)
elseif(JOINT_PREFIX_ONLY)
    list(APPEND _merge_generate_args --prefix-only)
elseif(JOINT_PEER_ONLY)
    list(APPEND _merge_generate_args --peer-only)
elseif(JOINT_MTP7_PACK_ONLY)
    list(APPEND _merge_generate_args --mtp7-pack-only)
elseif(JOINT_MTP7_FIX_ONLY)
    list(APPEND _merge_generate_args --mtp7-fix-only)
elseif(JOINT_MTP8_ONLY)
    list(APPEND _merge_generate_args --mtp8-only)
elseif(JOINT_HOT_ONLY)
    list(APPEND _merge_generate_args --hot-only)
elseif(JOINT_EXEC_ONLY)
    list(APPEND _merge_generate_args --exec-only)
elseif(JOINT_REFINE_ONLY)
    list(APPEND _merge_generate_args --refine-only)
elseif(JOINT_INTEGRATED_ONLY)
    list(APPEND _merge_generate_args --integrated-only)
elseif(JOINT_COLD_COMPARE_ONLY)
    list(APPEND _merge_generate_args --cold-compare-only)
elseif(JOINT_AC_THREADS_ONLY)
    list(APPEND _merge_generate_args --ac-threads-only)
elseif(JOINT_UNION_ONLY)
    list(APPEND _merge_generate_args --union-only)
elseif(JOINT_SWEEP_ONLY)
    list(APPEND _merge_generate_args --sweep-only)
elseif(JOINT_SORTED_ONLY)
    list(APPEND _merge_generate_args --sorted-only)
elseif(JOINT_MERGE_ONLY)
    list(APPEND _merge_generate_args --merge-only)
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" "${JOINT_ROOT}/checks/generate_joint_variants.py"
    --output "${JOINT_GENERATED}" ${_merge_generate_args} RESULT_VARIABLE _variant_result)
if(NOT _variant_result EQUAL 0)
    message(FATAL_ERROR "Failed to generate Joint variant symbols")
endif()
set(JOINT_VARIANT_OBJECTS)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import sys; sys.path.insert(0,r'${JOINT_ROOT}'); from joint_variants import MASKS; print(';'.join(MASKS))"
    OUTPUT_VARIABLE JOINT_VARIANT_NAMES OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _catalog_result)
if(NOT _catalog_result EQUAL 0)
    message(FATAL_ERROR "Cannot query variant catalog")
endif()
foreach(_name IN LISTS JOINT_VARIANT_NAMES)
    if(JOINT_CAPACITY8K_ONLY AND NOT _name MATCHES "^mtp7_fused_(8k_ref|ab_8k)_512$")
        continue()
    endif()
    if(JOINT_FUSED_TUNE_ONLY AND NOT _name MATCHES "^mtp7_fused_(count|a|b|c|ab|ac|bc|abc)_512$")
        continue()
    endif()
    if(JOINT_MERGE_UNIQUE_ONLY AND NOT _name MATCHES "^(mtp7_reuse_parallel_512|mtp7_mid_unique_512|mtp7_fused_count_512)$")
        continue()
    endif()
    if(JOINT_PREFIX_ONLY AND NOT _name MATCHES "^(mtp7_reuse_parallel_512|mtp7_prefix_a_512|mtp7_prefix_b_512|mtp7_prefix_ab_512)$")
        continue()
    endif()
    if(JOINT_PEER_ONLY AND NOT _name STREQUAL "mtp7_reuse_parallel_512")
        continue()
    endif()
    if(JOINT_MTP7_PACK_ONLY AND NOT _name MATCHES "^(hot_bd8_512|mtp7_parallel_512|mtp7_rank_512|mtp7_reuse_parallel_512|mtp7_reuse_rank_512|mtp7_tile_512|mtp7_reuse_tile_512)$")
        continue()
    endif()
    if(JOINT_MTP7_FIX_ONLY AND NOT _name MATCHES "^(hot_bd8_512|mtp7_parallel_512|mtp7_rank_512)$")
        continue()
    endif()
    if(JOINT_MTP8_ONLY AND NOT _name MATCHES "^(hot_bd_512|hot_bd8_512)$")
        continue()
    endif()
    if(JOINT_HOT_ONLY AND _name STREQUAL "hot_bd8_512")
        continue()
    endif()
    if(JOINT_HOT_ONLY AND NOT _name MATCHES "^(integrated_bounds_512|refine_abd_512|refine_bd_512|hot_.*)$")
        continue()
    endif()
    if(JOINT_EXEC_ONLY AND NOT _name MATCHES "^(integrated_bounds_512|refine_abd_512|refine_bd_512|exec_.*)$")
        continue()
    endif()
    if(JOINT_REFINE_ONLY AND NOT _name MATCHES "^(integrated_bounds_512|refine_.*)$")
        continue()
    endif()
    if(JOINT_INTEGRATED_ONLY AND NOT _name MATCHES "^(union_cold_512|hw_ac_cold_512|integrated_512|integrated_bounds_512)$")
        continue()
    endif()
    if(JOINT_COLD_COMPARE_ONLY AND NOT _name MATCHES "^(hw_ac_512|hw_ac_256|union_base_512|union_cold_512|hw_ac_cold_512)$")
        continue()
    endif()
    if(JOINT_AC_THREADS_ONLY AND NOT _name MATCHES "^(hw_ac_512|hw_ac_256)$")
        continue()
    endif()
    if(JOINT_UNION_ONLY AND NOT _name MATCHES "^(union_.*|hw_ac_512|hw_ac_256|hw_ac_cold_512)$")
        continue()
    endif()
    if(NOT JOINT_UNION_ONLY AND JOINT_SWEEP_ONLY AND NOT _name MATCHES "^(hw_.*|sorted_mrgsort)$")
        continue()
    endif()
    if(NOT JOINT_UNION_ONLY AND NOT JOINT_SWEEP_ONLY AND JOINT_SORTED_ONLY AND NOT _name MATCHES "^(merge|merge_bd|sorted_stream|sorted_mrgsort)$")
        continue()
    endif()
    if(NOT JOINT_SWEEP_ONLY AND NOT JOINT_SORTED_ONLY AND JOINT_MERGE_ONLY AND NOT _name MATCHES "^merge")
        continue()
    endif()
    set(_object "${JOINT_GENERATED}/variant_${_name}.o")
    add_custom_command(OUTPUT "${_object}"
        COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
            ${JOINT_FLAGS} ${JOINT_INCLUDES} -c "${JOINT_GENERATED}/variant_${_name}.cpp" -o "${_object}"
        DEPENDS "${JOINT_GENERATED}/variant_${_name}.cpp" "${JOINT_VARIANT_DIR}/kernel.cpp"
            "${JOINT_VARIANT_DIR}/sparse_kv_joint_plan.h" "${JOINT_VARIANT_DIR}/options.h" "${JOINT_VARIANT_DIR}/bitmap_plan.h"
            "${JOINT_VARIANT_DIR}/turbo_plan.h" "${JOINT_VARIANT_DIR}/turbo_union_experiments.h" "${JOINT_VARIANT_DIR}/merge_plan.h" "${JOINT_VARIANT_DIR}/merge_experiments.h" "${JOINT_VARIANT_DIR}/sorted_stage.h" "${JOINT_VARIANT_DIR}/sorted_controller.h" "${JOINT_VARIANT_DIR}/sorted_sweep_stage.h" "${JOINT_VARIANT_DIR}/sorted_sweep_controller.h" "${JOINT_VARIANT_DIR}/union_next_stage.h" "${JOINT_VARIANT_DIR}/union_next_controller.h" "${JOINT_VARIANT_DIR}/union_next_simd.h" "${JOINT_VARIANT_DIR}/cold_finish.h" "${JOINT_VARIANT_DIR}/cold_ac_stage.h" "${JOINT_VARIANT_DIR}/cold_union_stage.h" "${JOINT_VARIANT_DIR}/integrated_stage.h" "${JOINT_VARIANT_DIR}/phase0_bounds_stage.h" "${JOINT_VARIANT_DIR}/refine_options.h" "${JOINT_VARIANT_DIR}/refine_stage.h" "${JOINT_VARIANT_DIR}/refine_controller.h" "${JOINT_VARIANT_DIR}/refine_simd.h" "${JOINT_VARIANT_DIR}/exec_options.h" "${JOINT_VARIANT_DIR}/exec_stage.h" "${JOINT_VARIANT_DIR}/exec_simd.h" "${JOINT_VARIANT_DIR}/exec_controller.h" "${JOINT_VARIANT_DIR}/hot_plan_impl.h" "${JOINT_VARIANT_DIR}/mtp8_plan_impl.h" "${JOINT_VARIANT_DIR}/mtp7_merge_unique.h" "${JOINT_VARIANT_DIR}/mtp7_fused_refine.h"
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
set(PAGED_OPT_OBJECTS)
set(PAGED_OPT_PLUGINS)
foreach(_name a b ab)
    if(JOINT_FUSED_TUNE_ONLY OR JOINT_MERGE_UNIQUE_ONLY OR JOINT_PREFIX_ONLY OR JOINT_PEER_ONLY OR JOINT_MTP7_PACK_ONLY OR JOINT_MTP7_FIX_ONLY OR JOINT_MTP8_ONLY OR JOINT_UNION_ONLY OR JOINT_AC_THREADS_ONLY OR JOINT_COLD_COMPARE_ONLY)
        continue()
    endif()
    set(_dir "${JOINT_ROOT}/csrc/abcdf_paged_opt")
    set(_obj "${CMAKE_CURRENT_BINARY_DIR}/abcdf_pages_${_name}.o")
    file(GLOB _headers CONFIGURE_DEPENDS "${_dir}/*.h" "${_dir}/${_name}/*.h")
    add_custom_command(OUTPUT "${_obj}"
        COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
            ${JOINT_FLAGS} ${JOINT_INCLUDES} -I${_dir} -c "${_dir}/${_name}/kernel.cpp" -o "${_obj}"
        DEPENDS "${_dir}/${_name}/kernel.cpp" ${_headers} "${JOINT_ROOT}/joint_plan_manifest.json" VERBATIM)
    set_source_files_properties("${_obj}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    list(APPEND PAGED_OPT_OBJECTS "${_obj}")
    list(APPEND PAGED_OPT_PLUGINS "${_dir}/${_name}/plugin.cpp")
endforeach()
# Standalone Plan snapshot supplied by the peer QLI branch, same compiler flags.
set(PEER_PLAN_DIR "${JOINT_ROOT}/csrc/peer_qli_plan")
set(PEER_PLAN_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/peer_qli_plan.o")
file(GLOB_RECURSE PEER_PLAN_HEADERS CONFIGURE_DEPENDS "${PEER_PLAN_DIR}/*.h")
add_custom_command(OUTPUT "${PEER_PLAN_OBJECT}"
    COMMAND "${BISHENG_CXX}" --npu-arch=dav-3510 -xasc -std=c++17 -fPIC
        ${JOINT_FLAGS} ${JOINT_INCLUDES} -I${PEER_PLAN_DIR}
        -c "${PEER_PLAN_DIR}/kernel.cpp" -o "${PEER_PLAN_OBJECT}"
    DEPENDS "${PEER_PLAN_DIR}/kernel.cpp" ${PEER_PLAN_HEADERS}
        "${JOINT_ROOT}/joint_plan_manifest.json" VERBATIM)
set_source_files_properties("${PEER_PLAN_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
add_library(mtp_joint_plan_ops SHARED "${PEER_PLAN_OBJECT}" "${PEER_PLAN_DIR}/plugin.cpp" ${PAGED_OPT_OBJECTS} ${PAGED_OPT_PLUGINS} "${ABCDF_PAGES_DIR}/plugin.cpp" "${ABCDF_PAGES_OBJECT}" "${JOINT_DIR}/plugin.cpp" "${JOINT_INFO}"
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
