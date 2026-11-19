/*
 * Host 接入与缓冲区
 *
 *   R 为分配请求数，Q=4R，T=2048，C=8192，P=C/blockSize，已测 blockSize=128。
 *   reqIds                            int64[R]
 *   topkIndices                      int32[Q,T]，每行按 token ID 非递减
 *   stablePrefixLens, visibleSeqLens   int32[Q]，各行排他上界
 *   tokenToReq                        int32[Q]，当前 reqIds 下标，同请求行连续
 *   activeRows                        int32[1]，本次实际行数，双 AIV 需要 Q 行有效
 *   lastReqIds                        int64[Q]
 *   slotToToken, lruSlots              int32[Q,C]
 *   currentSlots                     int32[Q,T]
 *   missCount                        int32[Q]
 *   missTokens, missSlots            int32[Q,T]，读取每行 missCount 指定的前缀
 *   rowModes, residentRows            int32[Q]，模式为 0=独立、1=联合首行、2=跟随
 *   residentBlockTable               int32[Q,P]
 *   compactWorkspace                 int32[6144R]
 *   tiling                           80 字节的设备副本
 *
 * 按 MakeJointTiling 返回值分配 workspace 并上传 tiling，用
 * joint_plan_block_dim_mtp3_bt_direct_odd_512 查询 blockDim。
 * 从 Host 调用 launch_joint_plan_mtp3_bt_direct_odd_512(true, ...)。
 * dynamicUb 传 tiling.localMemoryBytes；tiling、张量与 workspace 在本次设备调用
 * 完成前保持有效。独立的并发调用各自使用独立的状态与 workspace。
 * 冷启动时 lastReqIds、slotToToken 填 -1，rowModes 置 0，各行 LRU 初始化为 [0,C)。
 * 联合组使用 residentRows 指定的共享缓存行来解释 currentSlots。
 *
 * CANN 9.2.1 / DAV_3510 编译参数沿用已上库版本：
 * bisheng --npu-arch=dav-3510 -xasc -std=c++17 -fPIC -O3 --cce-auto-sync=off
 *   -Wno-deprecated-declarations -mllvm -cce-vf-remove-membar=false
 *   -mllvm -cce-aicore-hoist-movemask=false -I<CANN/include> -I<CANN/asc/include>
 *   -c sparse_kv_plan_apt.cpp -o mtp3_bt_direct_odd_512.o
 * 按 CANN 安装布局设置 include 路径，链接使用调用工程已有的 CANN 平台与运行时。
 */

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef VLLM_ASCEND_SPARSE_KV_PLAN_CONFIG_H
#define VLLM_ASCEND_SPARSE_KV_PLAN_CONFIG_H

#include <climits>
#include <cstdint>

namespace sparse_kv_plan_config {

constexpr uint32_t PLAN_THREADS = 1024U;
constexpr uint32_t PLAN_WARP_SIZE = 32U;
constexpr uint32_t PLAN_WARP_COUNT = PLAN_THREADS / PLAN_WARP_SIZE;
constexpr uint32_t PLAN_CONTROL_ELEMENTS = 8U;
constexpr uint32_t PLAN_SCAN_ELEMENTS = 2U * PLAN_WARP_COUNT + PLAN_CONTROL_ELEMENTS;
constexpr uint32_t PLAN_SCAN_BYTES = (PLAN_SCAN_ELEMENTS * sizeof(int32_t) + 31U) & ~31U;
constexpr uint32_t PLAN_SCAN_STORAGE_ELEMENTS = PLAN_SCAN_BYTES / sizeof(int32_t);
constexpr uint32_t PLAN_ROW_UB_LIMIT_BYTES = 112U * 1024U;
constexpr uint32_t PLAN_ROW_UB_LIMIT_ELEMENTS = PLAN_ROW_UB_LIMIT_BYTES / sizeof(int32_t);
constexpr uint32_t PLAN_MAX_DYNAMIC_UB_BYTES = PLAN_SCAN_BYTES + PLAN_ROW_UB_LIMIT_BYTES;

static_assert(PLAN_THREADS > 0U, "Sparse KV Plan requires at least one thread");
static_assert(PLAN_THREADS % PLAN_WARP_SIZE == 0U, "Sparse KV Plan threads must contain whole warps");
static_assert(PLAN_SCAN_BYTES % 32U == 0U, "Sparse KV Plan scan workspace must be 32-byte aligned");
static_assert(PLAN_ROW_UB_LIMIT_BYTES % sizeof(int32_t) == 0U, "Sparse KV Plan row UB limit must be int32 aligned");
static_assert(PLAN_MAX_DYNAMIC_UB_BYTES <= 120U * 1024U, "Sparse KV Plan dynamic UB exceeds the full-DCache budget");

}  // namespace sparse_kv_plan_config


#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// A distinct host-side type avoids collision with the preserved kernel-local type.
struct JointDirectTilingData {
  uint64_t hashCapacity, workspaceRowElements;
  int64_t maxRows, topk, capacity, maxToken, maxRequests;
  int32_t blockSize;
  uint32_t localMemoryBytes;
  uint64_t jointHashCapacity, scratchElements;
};
static_assert(sizeof(JointDirectTilingData) == 80, "joint tiling ABI");
namespace sparse_kv_plan_config {

// 固定 MTP3 四行、TopK2048、capacity8192 的 Host tiling。
// 工作区前半为兼容路径保留，随后每个请求有 2048 个 int32 交接词。
inline bool MakeJointTiling(int64_t maxRequests, int64_t maxRows, int64_t capacity,
                            int64_t maxToken, int32_t blockSize,
                            JointDirectTilingData& tiling, uint64_t& workspaceInt32) {
  constexpr int64_t topK = 2048;
  constexpr int64_t residentCapacity = 8192;
  constexpr int64_t tokenLimit = 486400;
  if (maxRequests <= 0 || maxRows <= 0 || maxRows % 4 != 0 ||
      maxRows / 4 != maxRequests || capacity != residentCapacity ||
      maxToken <= 0 || maxToken > tokenLimit || blockSize <= 0 ||
      capacity % blockSize != 0 || maxRows > INT32_MAX / (capacity / blockSize)) {
    return false;
  }
  constexpr uint64_t hashCapacity = 4096;
  constexpr uint64_t privateWords = 3 * hashCapacity + residentCapacity + topK;
  constexpr uint64_t scratchWords = 128 + residentCapacity / 2 + residentCapacity +
                                    24576 + 8192 + 1728;
  constexpr uint32_t dynamicUb = PLAN_SCAN_BYTES + scratchWords * sizeof(int32_t);
  static_assert(scratchWords == 46912 && dynamicUb == 187936, "mask 264 UB layout");
  tiling = {hashCapacity, privateWords, maxRows, topK, residentCapacity, maxToken,
            maxRequests, blockSize, dynamicUb, 2 * residentCapacity, scratchWords};
  workspaceInt32 = static_cast<uint64_t>(maxRows) * topK / 2 +
                   static_cast<uint64_t>(maxRequests) * 2048U;
  return true;
}

}  // namespace sparse_kv_plan_config

extern "C" uint32_t joint_plan_block_dim_mtp3_bt_direct_odd_512();
extern "C" void launch_joint_plan_mtp3_bt_direct_odd_512(bool jointEntry, GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens,
    GM_ADDR tokenToReq, GM_ADDR activeRows, GM_ADDR lastReqIds,
    GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount,
    GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
    GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling,
    uint32_t blockDim, uint32_t dynamicUb, void* stream);

#endif
