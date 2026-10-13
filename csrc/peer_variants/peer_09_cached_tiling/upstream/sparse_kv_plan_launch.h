/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * (kernel 计算逻辑源自 MemFabric_Hybrid,Mulan PSL v2,见 sparse_kv_plan_kernel.cpp 头部声明)
 *
 * SparseKvPlan 直调(direct launch)共享头文件。
 * 本文件同时被 bisheng(kernel 侧)与 g++(plugin 侧)编译,只允许出现
 * 平台无关的纯 C++:POD tiling 结构、host 侧 tiling 计算、launch 声明。
 *
 * 与 vllm-ascend 原工程的对应关系:
 * - SparseKvPlanTilingData 对应 op_host/sparse_kv_plan_tiling.h 中
 *   TILING_DATA_FIELD_DEF 生成的结构(字段与顺序逐字一致,保证 tiling 缓冲
 *   字节布局兼容)。
 * - 三个 tiling 辅助函数逐行移植自 op_host/sparse_kv_plan_tiling.cpp。
 * - calc_sparse_kv_plan_block_dim 对应原 tiling 里的
 *   blockDim = min(PLAN_BLOCKS, GetCoreNumAiv())。
 */
#ifndef SPARSE_KV_PLAN_LAUNCH_H
#define SPARSE_KV_PLAN_LAUNCH_H

#include <cstdint>

#include "sparse_kv_plan_config.h"

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// direct launch 版 tiling 结构:字段、顺序与
// vllm-ascend op_host/sparse_kv_plan_tiling.h 的注册结构完全一致。
struct SparseKvPlanTilingData {
  uint64_t hashCapacity;
  uint64_t workspaceRowElements;
  int64_t maxRows;
  int64_t topk;
  int64_t capacity;
  int64_t maxToken;
  int64_t maxRequests;
  int64_t maxNumBlocks;
  int64_t hostNumBlocks;
  int32_t blockSize;
  uint32_t localMemoryBytes;
};

// 与原 tiling 的 PLAN_BLOCKS 常量一致:块数上限 64。
constexpr uint32_t SPARSE_KV_PLAN_MAX_BLOCKS = 64U;
// 与原 tiling 的 MIN_HASH_CAPACITY 一致。
constexpr uint64_t SPARSE_KV_PLAN_MIN_HASH_CAPACITY = 32U;

// HashCapacity:2*topk 向上取 2 的幂,下限 32。移植自 sparse_kv_plan_tiling.cpp。
inline uint64_t SparseKvPlanHashCapacity(int64_t topk) {
  uint64_t target = static_cast<uint64_t>(topk) * 2U;
  uint64_t capacity = SPARSE_KV_PLAN_MIN_HASH_CAPACITY;
  while (capacity < target) {
    capacity <<= 1U;
  }
  return capacity;
}

// 单行临时区元素数:keys[H] + firstPos[H] + owner[H] + evictSlots[C] + hitSlots[C]
// + missPositions[T],单位 int32。移植自 sparse_kv_plan_tiling.cpp。
inline uint64_t SparseKvPlanRowElements(int64_t topk, int64_t capacity) {
  return 3U * SparseKvPlanHashCapacity(topk) + 2U * static_cast<uint64_t>(capacity) +
         static_cast<uint64_t>(topk);
}

// 动态 UB 大小 = scan workspace + (行临时区 <= 112 KiB 时计入,否则走 GM workspace)。
inline uint32_t SparseKvPlanLocalMemoryBytes(uint64_t workspaceRowElements) {
  const uint64_t rowBytes = workspaceRowElements * sizeof(int32_t);
  const uint64_t alignedRowBytes = (rowBytes + 31U) & ~static_cast<uint64_t>(31U);
  const uint64_t localBytes = sparse_kv_plan::PLAN_SCAN_BYTES +
                              (alignedRowBytes <= sparse_kv_plan::PLAN_ROW_UB_LIMIT_BYTES ? alignedRowBytes : 0U);
  return static_cast<uint32_t>(localBytes);
}

// host 侧组装完整 tiling 结构(纯函数,不查平台)。
inline SparseKvPlanTilingData MakeSparseKvPlanTiling(int64_t maxRows, int64_t topk, int64_t capacity,
                                                    int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks,
                                                    int64_t hostNumBlocks, int64_t blockSize) {
  SparseKvPlanTilingData tiling{};
  const uint64_t hashCapacity = SparseKvPlanHashCapacity(topk);
  const uint64_t rowElements = SparseKvPlanRowElements(topk, capacity);
  tiling.hashCapacity = hashCapacity;
  tiling.workspaceRowElements = rowElements;
  tiling.maxRows = maxRows;
  tiling.topk = topk;
  tiling.capacity = capacity;
  tiling.maxToken = maxToken;
  tiling.maxRequests = maxRequests;
  tiling.maxNumBlocks = maxNumBlocks;
  tiling.hostNumBlocks = hostNumBlocks;
  tiling.blockSize = static_cast<int32_t>(blockSize);
  tiling.localMemoryBytes = SparseKvPlanLocalMemoryBytes(rowElements);
  return tiling;
}

#if defined(__cplusplus) && !defined(SPARSE_KV_PLAN_LAUNCH_DECL_ONLY)
extern "C" {
#endif

// 查询目标 SoC 的 AIV 核数并计算 blockDim = min(64, AIV 核数)。
// 在 bisheng 编译单元内实现(host 代码),供 plugin 调用。
uint32_t calc_sparse_kv_plan_block_dim(void);

// 直调入口:grid = blockDim 个 AIV block,tiling 为设备侧 tiling 结构缓冲。
// dynUBufSize 由 host 的 tiling.localMemoryBytes 提供,保留动态 UB。
// 参数顺序与 kernel 入口一致(去掉 GE 框架专用的 workspace 形参)。
void launch_sparse_kv_plan(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                           GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR blockTable,
                           GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots,
                           GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                           GM_ADDR compactWorkspace, GM_ADDR tiling, uint32_t blockDim,
                           uint32_t dynUBufSize, void* stream);

#if defined(__cplusplus) && !defined(SPARSE_KV_PLAN_LAUNCH_DECL_ONLY)
}
#endif

#endif  // SPARSE_KV_PLAN_LAUNCH_H
