/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * 昇腾 950PR/950DT：运行时 shape 的 Sparse KV Plan，64 个 AIV block，
 * Hot BD 联合路径使用 512-thread SIMT 与 SIMD VF，ABCDF 回退使用 1024 线程。
 *
 * 本实现通过 vLLM Ascend 自定义算子接口接收持久化的 NPU LRU 状态。
 * batch、topk、capacity、max_token 均来自运行时或 tiling 数据。
 *
 * 阅读顺序：sparse_kv_plan -> RunAbdPlan -> SortedStage / RefineUniqueMapVf。
 * 以下原始 ABCDF 说明与实现用于不满足 ABD 条件时的逐行回退。
 * 一条 query 行的 Plan 由一个 block 完成，避免引入跨 block 同步。
 * block b 处理 b、b+blockDim、b+2*blockDim……行；B 小于 blockDim 时，
 * 多余 block 在外壳直接返回。blockDim 上限为 64，实际值取决于目标 SKU 的 AIV 数量。
 *
 * 相对原版的主要修改：
 * 1. 1024 线程对应 32 个 warp；二级 scan 并行计算，调用者保证临时区复用同步。
 * 2. 单行临时数组不超过 112 KiB 时放进片上 UB，否则使用原 GM workspace。
 * 3. 用一个 32 位整数打包 hit/evict 两种标志，一次 scan 得到两种稳定排名。
 * 4. 复用插入所得 bucket，把 resident hit 回填与 miss 压缩融合。
 * 5. hit/evict 共用一个 capacity 数组；请求 ID 改变时走冷路径。
 * 6. 行内循环采用 32 位无符号下标；跨行偏移、地址和 workspace 大小保留 64 位。
 * 7. 删除 pending 中间写入、miss 输出前缀预清理和 missSlots 回读；owner 安全打包。
 * 8. 不超过 32 个 warp 时特化二级 scan；保留各 tile 末尾同步。
 *
 * 必须保留的业务语义：
 * - LRU 从旧到新排列，miss 优先使用最老的可淘汰 slot。
 * - 同一 token 重复出现在 TopK 时，仅第一次位置获得已有 resident hit；
 *   后续重复位置仍可能各自产生 miss。本文件没有擅自改成全部副本共享一个 slot。
 * - 同 token 驻留在多个 slot 时，旧 LRU 最后一个 owner 提供命中 slot，
 *   这些 resident slot 仍全部保留在 hit 段。
 * - missCount 只统计实际分到 slot 的 miss；未分配输出保持 -1。
 * - LRU 应由上层维护为 slot 的排列；保留原版对越界 LRU 项的跳过行为。
 *
 * 集成与调优：
 * - Plan 和 Transfer 需要在同一 stream 依次启动，或由上层建立等价事件依赖。
 * - UB 快路径不读写用户提供的 GM 临时区。
 * - 线程数是编译期配置，并不保证越多越快。应检查目标 CANN 的编译资源报告、
 *   寄存器溢出以及实机时延。
 *
 * API 依据（Huawei 官方开发仓文档，master 预览；以部署的 CANN 版本为准）：
 * https://asc.gitcode.com/guide/programming_guide/language_extension/simd_simt_hybrid_builtin_keywords.html
 * https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simt_programming/memory_hierarchy.html
 * https://asc.gitcode.com/api/SIMT-API/atomic_operations/asc_atomic_cas.html
 */

// Ported from AmourSec/vllmascend_sep_fif commit
// 382819673c5b9147d992a67f77dca0786369bc99, sparse_kv_plan_apt.cpp.
// The original algorithm is retained; input access and group dispatch are adapted.
#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
#include "sparse_kv_plan_config.h"

#ifndef QLI_FUSED_PLAN_HEADER
#define QLI_FUSED_PLAN_HEADER

namespace qli_fused_plan {

template <typename T> struct TopkSource {
  __gm__ T* allTopk;
  __ubuf__ T* localRow;
  int64_t localBegin;
  int64_t localCount;
  int64_t offset;

  __simt_callee__ __aicore__ inline T operator[](int64_t i) const {
    if (i >= localBegin && i < localBegin + localCount) return localRow[i - localBegin];
    return asc_ldcg(allTopk + offset + i);
  }
  __simt_callee__ __aicore__ inline TopkSource operator+(int64_t delta) const {
    return TopkSource{allTopk, localRow, localBegin - delta, localCount, offset + delta};
  }
};

using qli_fused_plan::config::PLAN_ROW_UB_LIMIT_ELEMENTS;
using qli_fused_plan::config::PLAN_SCAN_STORAGE_ELEMENTS;
using qli_fused_plan::config::PLAN_THREADS;
using qli_fused_plan::config::PLAN_WARP_COUNT;
using qli_fused_plan::config::PLAN_WARP_SIZE;
constexpr int32_t INVALID_TOKEN = -1;
constexpr int32_t HASH_EMPTY = -1;
constexpr int32_t HASH_POSITION_EMPTY = INT32_MAX;
constexpr uint32_t HASH_BUCKET_INVALID = UINT32_MAX;
constexpr uint32_t OWNER_SLOT_BITS = 16U;
constexpr int32_t OWNER_SLOT_MASK = (1U << OWNER_SLOT_BITS) - 1U;
constexpr uint32_t OWNER_PACK_CAPACITY = 1U << (31U - OWNER_SLOT_BITS);

// scan 的动态 UB 布局，单位为 int32。warp 区域均由线程数推导，
// control 之后的 padding 保证单行 workspace 从 32 字节边界开始。
constexpr uint32_t WARP_TOTALS_OFFSET = 0U;
constexpr uint32_t WARP_BASES_OFFSET = PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_OFFSET = 2U * PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_TOTAL = 0U;
constexpr uint32_t CONTROL_RESET = 1U;

// 这是主动选择的快路径上限，不是硬件 UB 容量上限。
// 动态 UB 最多为 112 KiB + scan workspace，为 SIMT DCache 和系统预留区域留下空间。
// 同一 block 顺序处理多行时复用这块内存，因此占用不随 batch 增长。
// 低 12 位表示 evict 数，高位表示 hit 数。
// 每个 tile 最多 PLAN_THREADS 项，而 2^12=4096，因此低位不会进位到 hit 字段。
constexpr uint32_t COUNT_SHIFT = 12U;
constexpr int32_t COUNT_MASK = (1U << COUNT_SHIFT) - 1U;

static_assert(PLAN_THREADS < (1U << COUNT_SHIFT), "packed scan would carry");

struct TilingData {
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
  uint64_t jointHashCapacity;
  uint64_t scratchElements;
  uint32_t maxJointRows;
};

// Only warp 0 participates. Every lane, including padding lanes, executes
// every shuffle. Process warp totals in 32-item groups for any warp count.
__simt_callee__ __aicore__ inline void PublishWarpBases(__ubuf__ int32_t* workspace, uint32_t thread) {
  if (thread >= PLAN_WARP_SIZE) {
    return;
  }
  __ubuf__ int32_t* warpTotals = workspace + WARP_TOTALS_OFFSET;
  __ubuf__ int32_t* warpBases = workspace + WARP_BASES_OFFSET;
  __ubuf__ int32_t* control = workspace + CONTROL_OFFSET;
  const uint32_t lane = thread;
  if constexpr (PLAN_WARP_COUNT <= PLAN_WARP_SIZE) {
    const int32_t value = lane < PLAN_WARP_COUNT ? warpTotals[lane] : 0;
    int32_t inclusive = value;
#pragma unroll
    for (uint32_t offset = 1U; offset < PLAN_WARP_SIZE; offset <<= 1U) {
      const int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
      if (lane >= offset) {
        inclusive += prior;
      }
    }
    if (lane < PLAN_WARP_COUNT) {
      warpBases[lane] = inclusive - value;
    }
    if (lane == PLAN_WARP_SIZE - 1U) {
      control[CONTROL_TOTAL] = inclusive;
    }
    return;
  }
  int32_t carry = 0;
  for (uint32_t base = 0U; base < PLAN_WARP_COUNT; base += PLAN_WARP_SIZE) {
    const uint32_t index = base + lane;
    const int32_t value = index < PLAN_WARP_COUNT ? warpTotals[index] : 0;
    int32_t inclusive = value;
    for (uint32_t offset = 1U; offset < PLAN_WARP_SIZE; offset <<= 1U) {
      const int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
      if (lane >= offset) {
        inclusive += prior;
      }
    }
    if (index < PLAN_WARP_COUNT) {
      warpBases[index] = carry + inclusive - value;
    }
    carry += asc_shfl(inclusive, PLAN_WARP_SIZE - 1U, PLAN_WARP_SIZE);
  }
  if (lane == 0U) {
    control[CONTROL_TOTAL] = carry;
  }
}

/**
 * 对完整 SIMT block 做独占前缀和，返回当前线程之前的输入总和。
 * 输入可以是 0/1，也可以是打包后的 evict + (hit << 12)。
 *
 * 第一级：每个 warp 的 32 个线程用 shuffle 计算 inclusive scan。
 * 第二级：warp 0 并行扫描 warp total，生成每个 warp 的起点。
 * warp 数完全由 PLAN_THREADS 推导，后续调整线程数无需重写 scan 布局。
 *
 * 所有线程必须参加所有 barrier。尾 tile 的无效线程输入 0，不能提前返回。
 * 调用者的 scatter 后 barrier 保证本轮全部读完，下一次 scan 才能复用。
 */
__simt_callee__ __aicore__ inline int32_t BlockExclusiveScan(__ubuf__ int32_t* workspace, int32_t value,
                                                             uint32_t thread) {
  __ubuf__ int32_t* warpTotals = workspace + WARP_TOTALS_OFFSET;
  __ubuf__ int32_t* warpBases = workspace + WARP_BASES_OFFSET;
  const uint32_t lane = thread & (PLAN_WARP_SIZE - 1U);
  const uint32_t warp = thread / PLAN_WARP_SIZE;
  int32_t inclusive = value;
  for (uint32_t offset = 1U; offset < PLAN_WARP_SIZE; offset <<= 1U) {
    const int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
    if (lane >= offset) {
      inclusive += prior;
    }
  }
  if (lane == PLAN_WARP_SIZE - 1U) {
    warpTotals[warp] = inclusive;
  }
  asc_syncthreads();

  PublishWarpBases(workspace, thread);
  asc_syncthreads();

  // Every call site consumes CONTROL_TOTAL, scatters its output, then
  // executes a block barrier before the next scan or scratch reuse.
  return warpBases[warp] + inclusive - value;
}

// H 是 2 的幂，线性探测可以用 & (H-1) 环绕。bucket 使用无符号类型，
// 不把合法高位索引误判为 -1。沿用原工程的 hash 容量与 token 混合函数。
__simt_callee__ __aicore__ inline uint32_t HashToken(int32_t token, uint32_t hashCapacity) {
  uint32_t value = static_cast<uint32_t>(token);
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value & (hashCapacity - 1U);
}

/**
 * ScratchPtr 由实参推导为 __ubuf__ int32_t* 或 __gm__ int32_t*。
 * 编译器分别生成两个地址空间的实现，不做 UB/GM 之间的指针强转。
 * 昇腾 950 的 int32 CAS/min/max 支持 UB 与 GM。
 * CAS 决定哪个 token 占桶，atomic_min 确保重复 token 保留最早 TopK 位置。
 */
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline uint32_t InsertTopkToken(ScratchPtr keys, ScratchPtr firstPositions, int32_t token,
                                                           int32_t position, uint32_t hashCapacity) {
  uint32_t bucket = HashToken(token, hashCapacity);
  const uint32_t mask = hashCapacity - 1U;
  for (uint32_t probe = 0U; probe < hashCapacity; ++probe) {
    const int32_t old = asc_atomic_cas(keys + bucket, HASH_EMPTY, token);
    if (old == HASH_EMPTY || old == token) {
      asc_atomic_min(firstPositions + bucket, position);
      return bucket;
    }
    bucket = (bucket + 1U) & mask;
  }
  // 合法 helper 产生至少 2*topk 个桶,因此不会填满;保留有界循环防止死循环。
  return HASH_BUCKET_INVALID;
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline uint32_t LookupTopkToken(ScratchPtr keys, int32_t token, uint32_t hashCapacity) {
  uint32_t bucket = HashToken(token, hashCapacity);
  const uint32_t mask = hashCapacity - 1U;
  for (uint32_t probe = 0U; probe < hashCapacity; ++probe) {
    const int32_t key = keys[bucket];
    if (key == token) {
      return bucket;
    }
    if (key == HASH_EMPTY) {
      return HASH_BUCKET_INVALID;
    }
    bucket = (bucket + 1U) & mask;
  }
  return HASH_BUCKET_INVALID;
}

__simt_callee__ __aicore__ inline bool IsSourceTokenValid(int64_t row, int32_t token, __gm__ int32_t* visibleSeqLens,
                                                          __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable,
                                                          int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks,
                                                          int64_t hostNumBlocks, int32_t blockSize) {
  if (token < 0 || token >= maxToken || token >= visibleSeqLens[row]) {
    return false;
  }
  const int32_t request = tokenToReq[row];
  if (request < 0 || request >= maxRequests || blockSize <= 0) {
    return false;
  }
  const int64_t blockId = token / blockSize;
  if (blockId < 0 || blockId >= maxNumBlocks) {
    return false;
  }
  const int64_t tableOffset = static_cast<int64_t>(request) * maxNumBlocks + blockId;
  const int32_t physicalBlock = blockTable[tableOffset];
  return physicalBlock >= 0 && physicalBlock < hostNumBlocks;
}

__simt_callee__ __aicore__ inline void ClearMissTail(int64_t base, uint32_t assigned, uint32_t topk, uint32_t thread,
                                                     __gm__ int32_t* missTokens, __gm__ int32_t* missSlots) {
  for (uint32_t pos = assigned + thread; pos < topk; pos += PLAN_THREADS) {
    missTokens[base + pos] = INVALID_TOKEN;
    missSlots[base + pos] = INVALID_TOKEN;
  }
}

// Request-ID changes reset the row to identity LRU. Duplicated valid TopK
// entries intentionally remain independent misses, matching the baseline.
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessColdRow(
    __ubuf__ int32_t* scanWorkspace, ScratchPtr missPositions, int64_t row, uint32_t thread, __gm__ int64_t* reqIds,
    __gm__ int64_t* lastReqIds, TopkSource<int32_t> topkSource, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, uint32_t topk, uint32_t capacity,
    int32_t maxToken, int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {
  const int64_t topkBase = row * static_cast<int64_t>(topk);
  const int64_t capacityBase = row * static_cast<int64_t>(capacity);
  __ubuf__ int32_t* control = scanWorkspace + CONTROL_OFFSET;
  for (uint32_t slot = thread; slot < capacity; slot += PLAN_THREADS) {
    slotToToken[capacityBase + slot] = INVALID_TOKEN;
  }
  int32_t count = 0;
  for (uint32_t tile = 0U; tile < topk; tile += PLAN_THREADS) {
    const uint32_t position = tile + thread;
    const int32_t flag = position < topk &&
                         IsSourceTokenValid(row, topkSource[topkBase + position], visibleSeqLens, tokenToReq,
                                            blockTable, maxToken, maxRequests, maxNumBlocks, hostNumBlocks, blockSize);
    const int32_t rank = BlockExclusiveScan(scanWorkspace, flag, thread);
    const int32_t total = control[CONTROL_TOTAL];
    if (flag != 0) {
      missPositions[count + rank] = static_cast<int32_t>(position);
    }
    asc_syncthreads();  // Includes slot invalidation and scan-control consumption.
    count += total;
  }
  const int32_t assigned = count < static_cast<int32_t>(capacity) ? count : static_cast<int32_t>(capacity);
  for (uint32_t miss = thread; miss < static_cast<uint32_t>(assigned); miss += PLAN_THREADS) {
    const int32_t position = missPositions[miss];
    const int32_t token = topkSource[topkBase + position];
    slotToToken[capacityBase + miss] = token;
    currentSlots[topkBase + position] = static_cast<int32_t>(miss);
    missTokens[topkBase + miss] = token;
    missSlots[topkBase + miss] = static_cast<int32_t>(miss);
  }
  ClearMissTail(topkBase, static_cast<uint32_t>(assigned), topk, thread, missTokens, missSlots);
  const uint32_t unused = capacity - static_cast<uint32_t>(assigned);
  for (uint32_t order = thread; order < capacity; order += PLAN_THREADS) {
    lruSlots[capacityBase + order] = static_cast<int32_t>(order < unused ? assigned + order : order - unused);
  }
  if (thread == 0U) {
    const int32_t request = tokenToReq[row];
    lastReqIds[row] = request >= 0 && request < maxRequests ? reqIds[request] : -1;
    missCount[row] = assigned;
  }
  asc_syncthreads();  // Cold return is block-uniform; finish writes before reusing scratch.
}

/**
 * 一次调用只处理一条 query 行。只有临时数组允许放 UB；reqId、resident、LRU 等
 * 持久状态始终保存在 GM，并由 row 定位，不能依赖上一轮由哪个物理核处理。
 *
 * rowWorkspace 的五段布局（单位 int32）：
 * keys[H]、firstPos[H]、owner[H]、partitionSlots[C]、missPositions[T]。
 * H=hashCapacity，C=capacity，T=topk，总大小为 (3H+C+T)*4 字节。
 */
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessRuntimeRow(
    __ubuf__ int32_t* scanWorkspace, int64_t row, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds,
    TopkSource<int32_t> topkSource, __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens,
    __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots,
    __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    ScratchPtr rowWorkspace, uint32_t hashCapacity, uint32_t topk, uint32_t capacity, int32_t maxToken,
    int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  __ubuf__ int32_t* control = scanWorkspace + CONTROL_OFFSET;
  const int64_t topkBase = row * static_cast<int64_t>(topk);
  const int64_t capacityBase = row * static_cast<int64_t>(capacity);
  ScratchPtr hashKeys = rowWorkspace;
  ScratchPtr hashFirstPos = hashKeys + hashCapacity;
  ScratchPtr hashOwner = hashFirstPos + hashCapacity;
  ScratchPtr evictableSlots = hashOwner + hashCapacity;
  // Evictable slots grow from the front; hits grow backwards from the end.
  // Their total never exceeds capacity, even when invalid LRU entries are skipped.
  ScratchPtr missPositions = evictableSlots + capacity;

  // 阶段 1：每个线程以 PLAN_THREADS 为步长初始化输出与 hash，支持任意合法 runtime T/H。
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    currentSlots[topkBase + pos] = INVALID_TOKEN;
  }
  if (thread == 0U) {
    missCount[row] = 0;
    const int32_t request = tokenToReq[row];
    const int64_t currentReqId = request >= 0 && request < maxRequests ? reqIds[request] : -1;
    control[CONTROL_RESET] = lastReqIds[row] != currentReqId ? 1 : 0;
  }
  asc_syncthreads();

  // CONTROL_RESET is published by thread 0 and read after a block barrier.
  // This branch is uniform across all threads; no MTP suffix-only shortcut.
  if (control[CONTROL_RESET] != 0) {
    ProcessColdRow(scanWorkspace, missPositions, row, thread, reqIds, lastReqIds, topkSource, visibleSeqLens,
                   tokenToReq, blockTable, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, topk,
                   capacity, maxToken, maxRequests, maxNumBlocks, hostNumBlocks, blockSize);
    return;
  }
  for (uint32_t i = thread; i < hashCapacity; i += PLAN_THREADS) {
    hashKeys[i] = HASH_EMPTY;
    hashFirstPos[i] = HASH_POSITION_EMPTY;
    hashOwner[i] = INVALID_TOKEN;
  }
  asc_syncthreads();

  // 阶段 3：仅插入合法 TopK token。所有插入结束后，才允许普通 load 查询 hash。
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    const int32_t token = topkSource[topkBase + pos];
    if (IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, blockTable, maxToken, maxRequests, maxNumBlocks,
                           hostNumBlocks, blockSize)) {
      missPositions[pos] =
          static_cast<int32_t>(InsertTopkToken(hashKeys, hashFirstPos, token, static_cast<int32_t>(pos), hashCapacity));
    } else {
      missPositions[pos] = static_cast<int32_t>(HASH_BUCKET_INVALID);
    }
  }
  asc_syncthreads();

  int32_t stablePrefix = stablePrefixLens[row];
  if (stablePrefix < 0) {
    stablePrefix = 0;
  } else if (stablePrefix > maxToken) {
    stablePrefix = maxToken;
  }

  // 阶段 4：按旧 LRU 顺序稳定分类。每个 tile 做一次打包 scan，
  // 同时产生 evictRank 和 hitRank；两者仍分别保持原来的相对顺序。
  const bool packedOwner = capacity <= OWNER_PACK_CAPACITY;
  int32_t evictableCount = 0;
  int32_t hitCount = 0;
  for (uint32_t tile = 0U; tile < capacity; tile += PLAN_THREADS) {
    const uint32_t order = tile + thread;
    int32_t slot = INVALID_TOKEN;
    int32_t evictFlag = 0;
    int32_t hitFlag = 0;
    if (order < capacity) {
      slot = lruSlots[capacityBase + order];
      if (slot >= 0 && static_cast<uint32_t>(slot) < capacity) {
        int32_t token = slotToToken[capacityBase + slot];
        // speculative suffix 不再可靠，先失效再判断 hit。
        if (token >= 0 && token < maxToken && token >= stablePrefix) {
          slotToToken[capacityBase + slot] = INVALID_TOKEN;
          token = INVALID_TOKEN;
        }
        uint32_t bucket = HASH_BUCKET_INVALID;
        if (token >= 0 && token < maxToken) {
          bucket = LookupTopkToken(hashKeys, token, hashCapacity);
        }
        if (bucket != HASH_BUCKET_INVALID) {
          hitFlag = 1;
          const int32_t owner = packedOwner
                                    ? static_cast<int32_t>((order << OWNER_SLOT_BITS) | static_cast<uint32_t>(slot))
                                    : static_cast<int32_t>(order);
          asc_atomic_max(hashOwner + bucket, owner);
          // 同一 token 占多个 slot 时，旧 LRU 中最后一个 owner 提供命中结果。
        } else {
          evictFlag = 1;
        }
      }
    }
    // 越界 LRU 项和尾部线程两种 flag 都为 0；不能把它们误当 evict。
    const int32_t packed = evictFlag + (hitFlag << COUNT_SHIFT);
    const int32_t ranks = BlockExclusiveScan(scanWorkspace, packed, thread);
    const int32_t totals = control[CONTROL_TOTAL];
    const int32_t evictRank = ranks & COUNT_MASK;
    const int32_t hitRank = ranks >> COUNT_SHIFT;
    if (evictFlag != 0) {
      evictableSlots[evictableCount + evictRank] = slot;
    }
    if (hitFlag != 0) {
      evictableSlots[capacity - 1U - static_cast<uint32_t>(hitCount + hitRank)] = slot;
    }
    asc_syncthreads();
    evictableCount += totals & COUNT_MASK;
    hitCount += totals >> COUNT_SHIFT;
  }

  // F: consume per-query bucket before in-place stable compaction.
  int32_t localMissCount = 0;
  for (uint32_t tile = 0U; tile < topk; tile += PLAN_THREADS) {
    const uint32_t position = tile + thread;
    int32_t missFlag = 0;
    if (position < topk) {
      const uint32_t bucket = static_cast<uint32_t>(missPositions[position]);
      if (bucket != HASH_BUCKET_INVALID) {
        const int32_t owner = hashOwner[bucket];
        if (owner >= 0 && hashFirstPos[bucket] == static_cast<int32_t>(position)) {
          currentSlots[topkBase + position] = packedOwner ? (owner & OWNER_SLOT_MASK) : lruSlots[capacityBase + owner];
        } else {
          missFlag = 1;
        }
      }
    }
    const int32_t rank = BlockExclusiveScan(scanWorkspace, missFlag, thread);
    const int32_t tileMissCount = control[CONTROL_TOTAL];
    if (missFlag != 0) {
      missPositions[localMissCount + rank] = static_cast<int32_t>(position);
    }
    asc_syncthreads();
    localMissCount += tileMissCount;
  }

  // 阶段 7：miss 多于可淘汰 slot 时，仅分配能够容纳的前缀。
  const int32_t assignCount = localMissCount < evictableCount ? localMissCount : evictableCount;

  for (uint32_t miss = thread; miss < static_cast<uint32_t>(assignCount); miss += PLAN_THREADS) {
    const int32_t slot = evictableSlots[miss];
    const int32_t position = missPositions[miss];
    const int32_t token = topkSource[topkBase + position];
    slotToToken[capacityBase + slot] = token;
    currentSlots[topkBase + position] = slot;
    missTokens[topkBase + miss] = token;
    missSlots[topkBase + miss] = slot;
  }
  ClearMissTail(topkBase, static_cast<uint32_t>(assignCount), topk, thread, missTokens, missSlots);

  // 阶段 8：LRU = 未使用 evictable + 新 miss + 原 hit，顺序与原算子一致。
  const int32_t unassignedCount = evictableCount - assignCount;
  const int32_t validLruCount = evictableCount + hitCount;
  for (uint32_t output = thread; output < static_cast<uint32_t>(validLruCount); output += PLAN_THREADS) {
    int32_t slot;
    if (output < static_cast<uint32_t>(unassignedCount)) {
      slot = evictableSlots[assignCount + output];
    } else if (output < static_cast<uint32_t>(evictableCount)) {
      slot = evictableSlots[output - static_cast<uint32_t>(unassignedCount)];
    } else {
      slot = evictableSlots[capacity - 1U - (output - static_cast<uint32_t>(evictableCount))];
    }
    lruSlots[capacityBase + output] = slot;
  }
  if (thread == 0U) {
    missCount[row] = assignCount;
  }
  // 下一行可能立刻复用 UB，必须等当前行全部线程写完。
  asc_syncthreads();
}

// ABD512 port from op_test_kvoffload_simt_echo commit 9987777ae1282d13ead447b03e00b6a7934c2d3c.
// The measured configuration is frozen here: refine A+B+D, parent AC512.
// Keep the original ABCDF helpers above for 1024-thread fallback.
struct JointPlanArgs {
  __gm__ int64_t *reqIds, *lastReqIds;
  __gm__ int32_t *stablePrefixLens, *visibleSeqLens, *tokenToReq, *blockTable;
  TopkSource<int32_t> topkSource;
  __gm__ int32_t *slotToToken, *lruSlots, *currentSlots, *missCount, *missTokens, *missSlots;
  __gm__ int32_t *rowModes, *residentRows, *residentBlockTable;
  uint32_t topk, capacity, hashCapacity, jointHashCapacity;
  int32_t maxToken, blockSize;
  int64_t maxRequests, maxNumBlocks, hostNumBlocks;
  uint64_t scratchElements;
  uint32_t maxJointRows;
};
constexpr int32_t ROW_PRIVATE = 0;
constexpr int32_t ROW_JOINT = 1;
constexpr int32_t ROW_FOLLOWER = 2;

// Publish both consumers' mapping: Transfer destination row and SFA page table.
__simt_callee__ __aicore__ inline void PublishResidentRow(const JointPlanArgs& a, int64_t row, int64_t owner,
                                                          uint32_t thread) {
  const uint32_t pages = a.capacity / a.blockSize;
  if (thread == 0) a.residentRows[row] = static_cast<int32_t>(owner);
  for (uint32_t page = thread; page < pages; page += PLAN_THREADS) {
    a.residentBlockTable[row * pages + page] = static_cast<int32_t>(owner * pages + page);
  }
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void RunPrivateRow(const JointPlanArgs& a, int64_t row, ScratchPtr scratch,
                                                     __ubuf__ int32_t* scan, uint32_t thread) {
  if (thread == 0) {
    // A former follower has no authoritative private payload. A joint leader
    // may also contain a different policy/layout; start cold on either transition.
    if (a.rowModes[row] != ROW_PRIVATE) a.lastReqIds[row] = -1;
    a.rowModes[row] = ROW_PRIVATE;
  }
  PublishResidentRow(a, row, row, thread);
  asc_syncthreads();
  ProcessRuntimeRow(scan, row, a.reqIds, a.lastReqIds, a.topkSource, a.stablePrefixLens, a.visibleSeqLens,
                    a.tokenToReq, a.blockTable, a.slotToToken, a.lruSlots, a.currentSlots, a.missCount, a.missTokens,
                    a.missSlots, scratch, a.hashCapacity, a.topk, a.capacity, a.maxToken, a.maxRequests, a.maxNumBlocks,
                    a.hostNumBlocks, a.blockSize);
}

// SPDX-License-Identifier: Apache-2.0
// SIMT-only reset completion. Caller has captured uniform reset and crossed
// a block barrier BEFORE any mode/request-ID write. Never call on mere all-miss.
// Sorted union and capacity checks remain in the unchanged parent stages.

template <uint32_t Threads, bool SingleAllowed, typename P>
__simt_callee__ __aicore__ inline void FinishColdSorted(const JointPlanArgs& a, int64_t first, int64_t end, P map,
                                                        P unique, __ubuf__ int32_t* ctx, uint32_t thread) {
  const uint32_t u = static_cast<uint32_t>(ctx[1]), rows = static_cast<uint32_t>(end - first);
  if (u > a.capacity) {
    if (thread == 0U) ctx[0] = 0;
    asc_syncthreads();
    return;
  }
  const int64_t qb = first * static_cast<int64_t>(a.topk), cb = first * static_cast<int64_t>(a.capacity);
  const uint32_t unused = a.capacity - u, q = rows * a.topk;
  // Reset has no valid resident/hit. Assign uid directly to identity slot uid.
  // This fuses cache initialization and fill; no intermediate owner/flags/partition.
  for (uint32_t slot = thread; slot < a.capacity; slot += Threads) {
    const int32_t token = slot < u ? unique[slot] : -1;
    a.slotToToken[cb + slot] = token;
    a.lruSlots[cb + slot] = static_cast<int32_t>(slot < unused ? u + slot : slot - unused);
    if (slot < u) {
      a.missTokens[qb + slot] = token;
      a.missSlots[qb + slot] = static_cast<int32_t>(slot);
    }
  }
  for (uint32_t i = thread; i < q; i += Threads) {
    const int32_t uid = map[i];
    a.currentSlots[qb + i] = uid >= 0 ? uid : -1;
  }
  // Preserve the descriptor-tail contract.
  for (uint32_t i = u + thread; i < q; i += Threads) {
    a.missTokens[qb + i] = -1;
    a.missSlots[qb + i] = -1;
  }
  const int32_t req = a.tokenToReq[first];
  const uint32_t pages = a.capacity / a.blockSize;
  for (uint32_t r = 0; r < rows; ++r) {
    const int64_t row = first + r;
    for (uint32_t page = thread; page < pages; page += Threads)
      a.residentBlockTable[row * pages + page] = static_cast<int32_t>(first * pages + page);
    if (thread == 0U) {
      a.residentRows[row] = static_cast<int32_t>(first);
      const int64_t left = static_cast<int64_t>(u) - static_cast<int64_t>(r) * a.topk;
      a.missCount[row] = left <= 0 ? 0 : left < a.topk ? static_cast<int32_t>(left) : static_cast<int32_t>(a.topk);
    }
    // Preserve the empirical thread-1 mode-publication workaround. No extra output.
    if (thread == 1U)
      a.rowModes[row] = SingleAllowed && rows == 1U ? ROW_PRIVATE : (r == 0U ? ROW_JOINT : ROW_FOLLOWER);
  }
  if (thread == 0U) {
    a.lastReqIds[first] = a.reqIds[req];
    ctx[0] = 2;
  }
  asc_syncthreads();  // finish all writes before this AIV reuses scratch for another group
}
// Frozen joint_refine_abd_512; source experiment mask72 / refine bits11.
constexpr uint32_t SORTED_THREADS = qli_fused_plan::config::PLAN_JOINT_THREADS;
constexpr uint32_t SORTED_WARPS = SORTED_THREADS / PLAN_WARP_SIZE;
static_assert(SORTED_THREADS % PLAN_WARP_SIZE == 0 && SORTED_WARPS <= PLAN_WARP_SIZE, "ABD warp layout");
// Uses existing (oversized) Scan storage. Warp count follows THIS VF's launch.
__simt_callee__ __aicore__ inline int32_t SortedScan(__ubuf__ int32_t* work, int32_t value, uint32_t thread,
                                                     uint32_t& parity, int32_t& total) {
  __ubuf__ int32_t* totals = work + (parity ? WARP_BASES_OFFSET : WARP_TOTALS_OFFSET);
  parity ^= 1U;
  uint32_t lane = thread & (PLAN_WARP_SIZE - 1U), warp = thread / PLAN_WARP_SIZE;
  int32_t inclusive = value;
#pragma unroll
  for (uint32_t d = 1; d < PLAN_WARP_SIZE; d <<= 1U) {
    int32_t x = asc_shfl_up(inclusive, d, PLAN_WARP_SIZE);
    if (lane >= d) inclusive += x;
  }
  if (lane == PLAN_WARP_SIZE - 1U) totals[warp] = inclusive;
  asc_syncthreads();
  int32_t own = lane < SORTED_WARPS ? totals[lane] : 0, sum = own;
#pragma unroll
  for (uint32_t d = 1; d < SORTED_WARPS; d <<= 1U) {
    int32_t x = asc_shfl_up(sum, d, PLAN_WARP_SIZE);
    if (lane >= d) sum += x;
  }
  total = asc_shfl(sum, SORTED_WARPS - 1U, PLAN_WARP_SIZE);
  return asc_shfl(sum - own, warp, PLAN_WARP_SIZE) + inclusive - value;
}
// B：只替换布尔 Scan 的 warp 内前缀；跨 warp 汇总、parity 与同步沿用参考版。
// Every logical tail lane participates with false; do not put ballot in i<n.
__simt_callee__ __aicore__ inline int32_t RefineBoolScan(__ubuf__ int32_t* work, bool flag, uint32_t thread,
                                                         uint32_t& parity, int32_t& total) {
  const uint32_t lane = thread & (PLAN_WARP_SIZE - 1U), warp = thread / PLAN_WARP_SIZE;
  const uint32_t votes = asc_ballot(flag ? 1 : 0), before = (uint32_t{1} << lane) - 1U;
  const int32_t ownRank = __popc(votes & before);
  __ubuf__ int32_t* totals = work + (parity ? WARP_BASES_OFFSET : WARP_TOTALS_OFFSET);
  parity ^= 1U;
  if (lane == PLAN_WARP_SIZE - 1U) totals[warp] = __popc(votes);
  asc_syncthreads();
  int32_t own = lane < SORTED_WARPS ? totals[lane] : 0, sum = own;
#pragma unroll
  for (uint32_t d = 1; d < SORTED_WARPS; d <<= 1U) {
    int32_t x = asc_shfl_up(sum, d, PLAN_WARP_SIZE);
    if (lane >= d) sum += x;
  }
  total = asc_shfl(sum, SORTED_WARPS - 1U, PLAN_WARP_SIZE);
  return asc_shfl(sum - own, warp, PLAN_WARP_SIZE) + ownRank;
}
// Hash-free sorted union planner. Helpers enter only through SortedStageVf.
// Lifetime: ctx[64] | queryToUid[W*T] | unique[C] | work[max(3C,64)].
// work is packet input/output during union construction, then owner/flags/LRU.
// GM scratch keeps packet input/output in the separate UB staging area.
template <typename P>
__simt_callee__ __aicore__ inline uint32_t SortedLower(P p, uint32_t n, int32_t token) {
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    uint32_t mid = lo + ((hi - lo) >> 1U);
    if (p[mid] < token)
      lo = mid + 1U;
    else
      hi = mid;
  }
  return lo;
}

// The four-row specialization retains the original 64-word context layout.
// Wide groups reuse spare words; hot compaction still owns words 43..45.
template <bool Wide>
struct SortedContext {
  static constexpr uint32_t starts = Wide ? 32U : 8U;
  static constexpr uint32_t ends = Wide ? 48U : 12U;
  static constexpr uint32_t active = Wide ? 8U : 16U;
  static constexpr uint32_t lengths = Wide ? 16U : 20U;
  static constexpr uint32_t consumed = Wide ? 56U : 24U;
};

template <int Phase, bool Wide, typename P>
__simt_callee__ __aicore__ inline void SortedStage(const JointPlanArgs& a, int64_t first, int64_t end, P data,
                                                   __ubuf__ int32_t* ctx, __ubuf__ int32_t* packet,
                                                   __ubuf__ int32_t* scan) {
  const uint32_t t = static_cast<uint32_t>(threadIdx.x), rows = static_cast<uint32_t>(end - first);
  using Layout = SortedContext<Wide>;
  constexpr bool wide = Wide;
  const uint32_t mapRows = rows <= a.maxJointRows && rows <= 8U ? rows : (a.maxJointRows <= 8U ? a.maxJointRows : 8U);
  P map = data, unique = map + mapRows * a.topk, owner = data + ((mapRows * a.topk + a.capacity + 7U) & ~7U);
  P flags = owner + a.capacity, partition = flags + a.capacity;
  const uint32_t q = rows * a.topk;
  const int64_t qb = first * static_cast<int64_t>(a.topk), cb = first * static_cast<int64_t>(a.capacity);
  if constexpr (Phase == 0) {
    if (t == 0U) {
      ctx[0] = 1;
      ctx[1] = 0;
      ctx[2] = -1;
      ctx[4] = 0;
      ctx[6] = 0;
      ctx[7] = 0;
      ctx[28] = 0;
      ctx[29] = 0;
      const uint32_t minWork = wide ? 128U : 64U;
      uint32_t chunk = (3U * a.capacity > minWork ? 3U * a.capacity : minWork) / 16U;
      if (chunk > 768U) chunk = 768U;
      ctx[3] = static_cast<int32_t>(chunk & ~3U);
      uint64_t words = 16ULL * static_cast<uint32_t>(ctx[3]);
      if (a.scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS)
        words = a.scratchElements - 64U - ((mapRows * a.topk + a.capacity + 7U) & ~7U);
      ctx[30] = static_cast<int32_t>((words / 2U) & ~uint64_t{7U});  // fixed input/output split

      const int32_t req = a.tokenToReq[first];
      uint32_t bits = 1;
      while (bits < 31U && (1ULL << bits) < q) ++bits;
      if (rows < 1U || rows > a.maxJointRows || rows > 8U || a.capacity > 32768U || q > 65535U || a.maxToken <= 0 || a.maxToken > 16777216 ||
          (static_cast<uint64_t>(a.maxToken - 1) << bits) > UINT32_MAX || req < 0 || req >= a.maxRequests)
        ctx[0] = 0;
      for (uint32_t r = 0; r < (wide ? 8U : 4U); ++r) {
        ctx[Layout::starts + r] = 0;
        ctx[Layout::ends + r] = 0;
      }
      if (ctx[0])
        for (uint32_t r = 0; r < rows; ++r) {
          const int64_t b = (first + r) * static_cast<int64_t>(a.topk);
          int32_t upper = a.visibleSeqLens[first + r];
          if (upper < 0) upper = 0;
          if (upper > a.maxToken) upper = a.maxToken;
          // Only Phase0 uses endpoint checks. SortedLower for resident lookup is unchanged.
          // upper remains row-specific and clipped; equality at upper is NOT visible.
          if (a.topk > 0U && a.topkSource[b] >= 0 && a.topkSource[b + a.topk - 1U] < upper) {
            ctx[Layout::starts + r] = 0;
            ctx[Layout::ends + r] = a.topk;
          } else {
            ctx[Layout::starts + r] = SortedLower(a.topkSource + b, a.topk, 0);
            ctx[Layout::ends + r] = SortedLower(a.topkSource + b, a.topk, upper);
          }
          if (ctx[Layout::starts + r] < ctx[Layout::ends + r]) ctx[6] = 1;
        }

      // A：严格递增单行的 UID 就是有效区间内的相对位置，无需打包或归并。
      // Duplicate detection below still vetoes this entire attempt before commit.
      if (ctx[0] && rows == 1U) {
        ctx[1] = ctx[Layout::ends + 0] - ctx[Layout::starts + 0];
        ctx[6] = 0;
      }
    }
    asc_syncthreads();
    // Reject oversized groups before touching the query map.
    if (ctx[0])
      for (uint32_t i = t; i < q; i += SORTED_THREADS) {
        // Validate while initializing query maps; no extra traversal or VF.
        const uint32_t pos = i % a.topk;
        const int64_t row = first + i / a.topk;
        const int32_t token = a.topkSource[qb + i];
        if (pos && token < a.topkSource[qb + i - 1U]) asc_atomic_min(ctx, 0);
        if (token >= 0 && token < a.maxToken && token < a.visibleSeqLens[row] &&
            !IsSourceTokenValid(row, token, a.visibleSeqLens, a.tokenToReq, a.blockTable, a.maxToken, a.maxRequests,
                                a.maxNumBlocks, a.hostNumBlocks, a.blockSize))
          asc_atomic_min(ctx, 0);
        // Independent semantics count duplicate query misses separately. Fall back;
        // never silently share their slot through the single-row unique list.

        if (rows == 1U) {
          const int32_t token = a.topkSource[qb + i];
          if (i && token == a.topkSource[qb + i - 1U]) asc_atomic_min(ctx, 0);
          const bool valid = i >= static_cast<uint32_t>(ctx[Layout::starts + 0]) &&
                             i < static_cast<uint32_t>(ctx[Layout::ends + 0]);
          map[i] = valid ? static_cast<int32_t>(i - static_cast<uint32_t>(ctx[Layout::starts + 0])) : -1;
          if (valid) unique[i - static_cast<uint32_t>(ctx[Layout::starts + 0])] = token;
        } else
          map[i] = -1;
      }
    asc_syncthreads();
  } else if constexpr (Phase == 1) {
    const uint32_t oldChunk = ctx[3];
    if (t == 0U) {
      uint32_t active = 0;
      for (uint32_t r = 0; r < rows; ++r)
        if (ctx[Layout::starts + r] < ctx[Layout::ends + r]) ctx[Layout::active + active++] = r;
      uint32_t chunk = oldChunk;
      if (active) {
        chunk = (static_cast<uint32_t>(ctx[30]) / (2U * active)) & ~3U;
        if (chunk > (wide ? 1020U : 4092U)) chunk = wide ? 1020U : 4092U;
      }
      ctx[3] = chunk;
      ctx[28] = active;
      ctx[7] = 0;
      uint32_t total = 0;
      for (uint32_t k = 0; k < active; ++k) {
        uint32_t r = ctx[Layout::active + k], n = ctx[Layout::ends + r] - ctx[Layout::starts + r];
        if (n > chunk) n = chunk;
        ctx[Layout::lengths + k] = n;
        ctx[Layout::consumed + k] = 0;
        total += n;
      }
      ctx[4] = total;
      ctx[5] = total;
      if constexpr (Wide) {
        if (active > 4U) {
          // A full packet merge is safe only through the last buffered token
          // of every truncated stream. Equal frontier tokens may span packets;
          // Phase3's existing previous-token state deduplicates that boundary.
          int32_t frontier = a.maxToken;
          for (uint32_t k = 0; k < active; ++k) {
            const uint32_t r = ctx[Layout::active + k], n = ctx[Layout::lengths + k];
            const uint32_t begin = ctx[Layout::starts + r];
            if (n < static_cast<uint32_t>(ctx[Layout::ends + r]) - begin) {
              const int32_t token = a.topkSource[qb + r * a.topk + begin + n - 1U];
              if (token < frontier) frontier = token;
            }
          }
          for (uint32_t k = 0; k < active; ++k) {
            const uint32_t r = ctx[Layout::active + k];
            ctx[Layout::consumed + k] = SortedLower(a.topkSource + qb + r * a.topk + ctx[Layout::starts + r],
                                                   ctx[Layout::lengths + k], frontier + 1);
          }
        }
      }
    }
    asc_syncthreads();
    const uint32_t chunk = ctx[3], active = ctx[28];
    auto* score = reinterpret_cast<__ubuf__ float*>(packet);
    for (uint32_t i = t; i < active * chunk; i += SORTED_THREADS) {
      uint32_t k = i / chunk, j = i % chunk;
      if (j < ctx[Layout::lengths + k]) {
        uint32_t r = ctx[Layout::active + k], pos = static_cast<uint32_t>(ctx[Layout::starts + r]) + j;
        score[2U * i] = -static_cast<float>(a.topkSource[qb + r * a.topk + pos]);
        packet[2U * i + 1U] = r * a.topk + pos;
      }
    }
    asc_syncthreads();
  } else if constexpr (Phase == 3) {
    const uint32_t n = ctx[4];
    auto* out = packet + (ctx[7] ? 0U : static_cast<uint32_t>(ctx[30]));
    auto* scores = reinterpret_cast<__ubuf__ float*>(out);
    uint32_t count = ctx[1], parity = 0;
    for (uint32_t tile = 0; tile < n; tile += SORTED_THREADS) {
      const uint32_t i = tile + t;
      int32_t token = -1, prev = -1;
      bool added = false;
      if (i < n) {
        token = static_cast<int32_t>(-scores[2U * i]);
        prev = i == 0U ? ctx[2] : static_cast<int32_t>(-scores[2U * (i - 1U)]);
        added = token != prev;
      }
      int32_t total = 0;
      const int32_t rank = RefineBoolScan(scan, added, t, parity, total);
      if (i < n) {
        uint32_t uid = count + static_cast<uint32_t>(rank) - (added ? 0U : 1U);
        if (uid < a.capacity && added) unique[uid] = token;
        map[static_cast<uint32_t>(out[2U * i + 1U])] = static_cast<int32_t>(uid);
      }
      asc_syncthreads();
      count += static_cast<uint32_t>(total);
    }
    if (t == 0U) {
      ctx[1] = count;
      if (n) ctx[2] = static_cast<int32_t>(-scores[2U * (n - 1U)]);
      if (count > a.capacity) ctx[0] = 0;
      ctx[29] = 1;
      ctx[6] = 0;
      for (uint32_t r = 0; r < rows; ++r)
        if (ctx[Layout::starts + r] < ctx[Layout::ends + r]) ctx[6] = 1;
    }
    asc_syncthreads();
  } else if constexpr (Phase == 5 || Phase == 6 || Phase == 9) {
    const uint32_t u = ctx[1];
    const int32_t req = a.tokenToReq[first];
    const int64_t reqid = a.reqIds[req];
    if constexpr (Phase == 5 || Phase == 6) {
      const bool reset = a.rowModes[first] != (rows == 1U ? ROW_PRIVATE : ROW_JOINT) || a.lastReqIds[first] != reqid;
      if (reset) {
        // Uniform reset: no thread may publish mode/ID until all have captured it.
        asc_syncthreads();
        FinishColdSorted<SORTED_THREADS, true>(a, first, end, map, unique, ctx, t);
        return;
      }
      int32_t stable = a.stablePrefixLens[first];
      for (int64_t row = first + 1; row < end; ++row)
        if (a.stablePrefixLens[row] < stable) stable = a.stablePrefixLens[row];
      if (stable < 0) stable = 0;
      if (stable > a.maxToken) stable = a.maxToken;
      for (uint32_t i = t; i < u; i += SORTED_THREADS) owner[i] = -1;
      asc_syncthreads();
      // One resident scan, no hash construction; every duplicate owner is protected.
      for (uint32_t order = t; order < a.capacity; order += SORTED_THREADS) {
        int32_t slot = a.lruSlots[cb + order];
        flags[order] = -1;
        if (slot < 0 || static_cast<uint32_t>(slot) >= a.capacity) continue;
        uint32_t hit = 0, invalidTag = 0;
        const int32_t token = a.slotToToken[cb + slot];
        if (token >= stable) invalidTag = 0x20000U;
        if (token >= 0 && token < stable) {
          const uint32_t pos = SortedLower(unique, u, token);
          if (pos < u && unique[pos] == token) {
            hit = 1;
            asc_atomic_max(owner + pos, static_cast<int32_t>((order << 16U) | static_cast<uint32_t>(slot)));
          }
        }
        flags[order] = slot | static_cast<int32_t>((hit << 16U) | invalidTag);
      }
      asc_syncthreads();
      if constexpr (Phase == 6) return;
    }
    uint32_t evicts = Phase == 9 ? ctx[43] : 0, hits = Phase == 9 ? ctx[44] : 0, misses = Phase == 9 ? ctx[45] : 0;
    if constexpr (Phase == 5) {
      evicts = 0;
      hits = 0;
      uint32_t parity = 0;
      for (uint32_t tile = 0; tile < a.capacity; tile += SORTED_THREADS) {
        uint32_t order = tile + t;
        int32_t f = order < a.capacity ? flags[order] : -1;
        int32_t hit = f >= 0 ? ((f >> 16U) & 1) : 0, evict = f >= 0 ? 1 - hit : 0, total = 0;
        int32_t rank = SortedScan(scan, evict + (hit << COUNT_SHIFT), t, parity, total);
        if (evict) partition[evicts + (static_cast<uint32_t>(rank) & COUNT_MASK)] = f & 0x2ffff;
        if (hit) partition[a.capacity - 1U - (hits + (static_cast<uint32_t>(rank) >> COUNT_SHIFT))] = f & 0x2ffff;
        asc_syncthreads();
        evicts += static_cast<uint32_t>(total) & COUNT_MASK;
        hits += static_cast<uint32_t>(total) >> COUNT_SHIFT;
      }
      if (evicts + hits != a.capacity) {
        if (t == 0U) ctx[0] = 0;
        return;
      }
      misses = 0;
      parity = 0;
      for (uint32_t tile = 0; tile < u; tile += SORTED_THREADS) {
        uint32_t pos = tile + t;
        bool missing = pos < u && owner[pos] < 0;
        int32_t total = 0;
        int32_t rank = RefineBoolScan(scan, missing, t, parity, total);
        if (pos < u) {
          flags[pos] = missing ? static_cast<int32_t>(misses) + rank : -1;
          if (missing && misses + static_cast<uint32_t>(rank) < evicts)
            owner[pos] = partition[misses + static_cast<uint32_t>(rank)];
        }
        asc_syncthreads();
        misses += static_cast<uint32_t>(total);
      }
      if (misses > evicts) {
        if (t == 0U) ctx[0] = 0;
        return;
      }
    }
    // All capacity checks precede persistent writes. Missing slots now hold final slot IDs.
    // Reuse the invalid-suffix tags captured during resident classification.
    for (uint32_t i = t; i < a.capacity; i += SORTED_THREADS) {
      const uint32_t entry = static_cast<uint32_t>(partition[i]);
      if (entry & 0x20000U) a.slotToToken[cb + (entry & 0xffffU)] = -1;
    }
    asc_syncthreads();
    for (uint32_t pos = t; pos < u; pos += SORTED_THREADS)
      if (flags[pos] >= 0) {
        const int32_t slot = owner[pos] & 0xffff, rank = flags[pos];
        a.slotToToken[cb + slot] = unique[pos];
        a.missTokens[qb + rank] = unique[pos];
        a.missSlots[qb + rank] = slot;
      }
    for (uint32_t i = t; i < q; i += SORTED_THREADS) {
      int32_t uid = map[i];
      a.currentSlots[qb + i] = uid >= 0 ? (owner[uid] & 0xffff) : -1;
    }
    for (uint32_t i = misses + t; i < q; i += SORTED_THREADS) {
      a.missTokens[qb + i] = -1;
      a.missSlots[qb + i] = -1;
    }
    const uint32_t unused = evicts - misses;
    for (uint32_t order = t; order < a.capacity; order += SORTED_THREADS)
      a.lruSlots[cb + order] = (order < unused   ? partition[misses + order]
                                : order < evicts ? partition[order - unused]
                                                 : partition[a.capacity - 1U - (order - evicts)]) &
                               0xffff;
    const uint32_t pages = a.capacity / a.blockSize;
    for (uint32_t r = 0; r < rows; ++r) {
      int64_t row = first + r;
      for (uint32_t page = t; page < pages; page += SORTED_THREADS)
        a.residentBlockTable[row * pages + page] = static_cast<int32_t>(first * pages + page);
      if (t == 0U) {
        a.residentRows[row] = first;
        int64_t left = static_cast<int64_t>(misses) - r * a.topk;
        a.missCount[row] = left <= 0 ? 0 : left < a.topk ? left : a.topk;
      }
      if (t == 1U) a.rowModes[row] = rows == 1U ? ROW_PRIVATE : (r == 0U ? ROW_JOINT : ROW_FOLLOWER);
    }
    if (t == 0U) {
      a.lastReqIds[first] = reqid;
      ctx[0] = 2;
    }
    asc_syncthreads();
  }
}

template <int Phase, bool Wide>
__simt_vf__ __aicore__ LAUNCH_BOUND((Phase == 4 ? PLAN_THREADS : SORTED_THREADS)) inline void SortedStageVf(
    __ubuf__ int32_t* scanWorkspace, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __ubuf__ int32_t* localTopk, int64_t localRow,
    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* compactWorkspace,
    uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart, int64_t rowStride, int64_t numReqs,
    int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks,
    int32_t blockSize, __gm__ int32_t* rowModes, __gm__ int32_t* residentRows, __gm__ int32_t* residentBlockTable,
    uint64_t jointHashCapacity, uint64_t scratchElements, uint32_t maxJointRows) {
  TopkSource<int32_t> topkSource{topkIndices, localTopk, localRow * topk, topk, 0};
  const JointPlanArgs args{reqIds,
                           lastReqIds,
                           stablePrefixLens,
                           visibleSeqLens,
                           tokenToReq,
                           blockTable,
                           topkSource,
                           slotToToken,
                           lruSlots,
                           currentSlots,
                           missCount,
                           missTokens,
                           missSlots,
                           rowModes,
                           residentRows,
                           residentBlockTable,
                           static_cast<uint32_t>(topk),
                           static_cast<uint32_t>(capacity),
                           static_cast<uint32_t>(hashCapacity),
                           static_cast<uint32_t>(jointHashCapacity),
                           static_cast<int32_t>(maxToken),
                           blockSize,
                           maxRequests,
                           maxNumBlocks,
                           hostNumBlocks,
                           scratchElements,
                           maxJointRows};

  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  if constexpr (Phase == 4) {
    // Unlike the standalone grid dispatcher, this invocation owns one bounded
    // group. The private fallback must never advance into another producer's rows.
    for (int64_t row = rowStart; row < rowStride; ++row) {
        if (scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS)
          RunPrivateRow(args, row, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, thread);
        else
          RunPrivateRow(args, row, compactWorkspace + static_cast<uint64_t>(row) * scratchElements, scanWorkspace,
                        thread);
      }
  } else {
    __ubuf__ int32_t* ctx = scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS;
    if (scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
      __ubuf__ int32_t* data = ctx + 64U;
      SortedStage<Phase, Wide>(
          args, rowStart, rowStride, data, ctx,
          data + ((((rowStride - rowStart <= args.maxJointRows ? static_cast<uint32_t>(rowStride - rowStart)
                                                               : args.maxJointRows) * args.topk +
                    args.capacity + 7U) &
                   ~7U)),
          scanWorkspace);
    } else {
      __gm__ int32_t* data = compactWorkspace + static_cast<uint64_t>(rowStart) * scratchElements + 64U;
      SortedStage<Phase, Wide>(args, rowStart, rowStride, data, ctx, ctx + 64U, scanWorkspace);
    }
  }
}
// SPDX-License-Identifier: Apache-2.0
// D：替换完整 UB Phase3；只修改临时映射，不访问 GM 或提交持久状态。
// Scatter bases are 32B-aligned (data/ctx); unique is addressed as data[q+uid],
// so odd TopK/capacity layouts do not require an aligned unique pointer.
__simd_vf__ inline void RefineUniqueMapVf(__ubuf__ int32_t* records, __ubuf__ int32_t* data, __ubuf__ int32_t* ctx,
                                          uint32_t n, uint32_t q, uint32_t capacity, uint32_t initialCount,
                                          int32_t initialLast, uint32_t more) {
  using namespace AscendC;
  Reg::RegTensor<uint32_t> lane, zero, one, even, odd, previousIndex, lastIndex;
  Reg::RegTensor<uint32_t> flags, prefix, carry, uid, payload, index, safeIndex, writeFlag, status, value;
  Reg::RegTensor<float> score, previous, lastScore, shiftScore, positive;
  Reg::RegTensor<int32_t> token, lastToken;
  Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
  Reg::MaskReg first, head, valid, fits, writeHead, chosen;
  // Unsqueeze(ALL) is exactly [0,1,...,63] for b32: no lane table in UB.
  Reg::Unsqueeze<uint32_t>(lane, full);
  Reg::Duplicate(zero, uint32_t{0});
  Reg::Duplicate(one, uint32_t{1});
  Reg::Duplicate(carry, initialCount);
  Reg::Duplicate(lastToken, initialLast);
  Reg::Duplicate(lastScore, -static_cast<float>(initialLast));
  Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
  Reg::Adds(previousIndex, lane, uint32_t(-1), full);
  Reg::Select(previousIndex, zero, previousIndex, first);  // no out-of-range register Gather
  static constexpr Reg::CastTrait toInt = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                           RoundMode::CAST_TRUNC};
  for (uint32_t base = 0; base < n; base += 64U) {
    uint32_t remaining = n - base;
    valid = Reg::UpdateMask<uint32_t>(remaining);
    const uint32_t count = n - base < 64U ? n - base : 64U;
    Reg::Adds(even, lane, base, full);
    Reg::ShiftLefts(even, even, int16_t{1}, full);
    Reg::Adds(odd, even, uint32_t{1}, full);
    Reg::Duplicate(score, 0.0f);
    Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float*>(records), even, valid);
    Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
    Reg::Select(previous, lastScore, shiftScore, first);
    Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
    Reg::Select(flags, one, zero, head);
    Reg::Select(flags, flags, zero, valid);
    // Rebuild all mask bits: no assumption about Compare's inactive lanes.
    Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
    Reg::Unsqueeze<uint32_t>(prefix, head);  // exclusive head count; last head not included here
    Reg::Add(prefix, prefix, flags, full);   // include every head, including lane63
    Reg::Add(prefix, prefix, carry, full);
    Reg::Adds(uid, prefix, uint32_t(-1), full);
    Reg::Neg(positive, score, full);
    Reg::Cast<int32_t, float, toInt>(token, positive, valid);  // exact integers in [0,2^24)
    Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
    Reg::Select(writeFlag, flags, zero, fits);
    Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
    Reg::Adds(index, uid, q, full);
    Reg::Adds(safeIndex, lane, q + capacity, full);
    Reg::Select(index, index, safeIndex, writeHead);  // even masked indices are unique
    Reg::Scatter<int32_t, uint32_t>(data, token, index, writeHead);
    Reg::Duplicate(payload, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(payload, reinterpret_cast<__ubuf__ uint32_t*>(records), odd, valid);
    Reg::Adds(safeIndex, lane, q, full);
    Reg::Select(payload, payload, safeIndex, valid);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data), uid, payload, valid);
    Reg::Duplicate(lastIndex, uint32_t{63});
    Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
    Reg::Duplicate(lastIndex, count - 1U);
    Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
    Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
  }
  // All progress Scatter indices are [0,63], unique and within the original ctx64.
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{1}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx), carry, lane, chosen);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{2}, full);
  Reg::Scatter<int32_t, uint32_t>(ctx, lastToken, lane, chosen);
  Reg::Compares<uint32_t, CMPMODE::GE>(fits, carry, capacity + 1U, full);
  Reg::Select(status, zero, one, fits);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{0}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx), status, lane, chosen);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{29}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx), one, lane, chosen);
  Reg::Duplicate(value, more);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{6}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx), value, lane, chosen);
}
__simd_callee__ inline void HotPutCtx(__ubuf__ int32_t* ctx, AscendC::Reg::RegTensor<uint32_t>& value,
                                      AscendC::Reg::RegTensor<uint32_t>& lane, uint32_t slot,
                                      AscendC::Reg::MaskReg& full) {
  AscendC::Reg::MaskReg chosen;
  AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::EQ>(chosen, lane, slot, full);
  AscendC::Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx), value, lane, chosen);
}
__simd_vf__ inline void HotCompactVf(__ubuf__ int32_t* data, __ubuf__ int32_t* ctx, uint32_t q, uint32_t capacity,
                                     uint32_t u) {
  using namespace AscendC;
  const uint32_t ow = (q + capacity + 7U) & ~7U, fl = ow + capacity, pa = fl + capacity;
  Reg::RegTensor<uint32_t> lane, zero, one, maskHit, keep, raw, hflag, eflag, vflag, hprefix, eprefix;
  Reg::RegTensor<uint32_t> hi, ei, hcarry, ecarry, mc, rank, inc, index, safe, slotTag, value, sum, status, tmp;
  Reg::RegTensor<int32_t> f, ownerValue, missRank;
  Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>(), valid, hit, evict, selected, fits;
  Reg::Unsqueeze<uint32_t>(lane, full);
  Reg::Duplicate(zero, uint32_t{0});
  Reg::Duplicate(one, uint32_t{1});
  Reg::Duplicate(mc, uint32_t{0});
  Reg::Duplicate(maskHit, uint32_t{65536});
  Reg::Duplicate(keep, uint32_t{0x2ffff});
  Reg::Duplicate(hcarry, uint32_t{0});
  Reg::Duplicate(ecarry, uint32_t{0});
  for (uint32_t base = 0; base < capacity; base += 64U) {
    uint32_t n = capacity - base;
    valid = Reg::UpdateMask<uint32_t>(n);
    Reg::Adds(index, lane, fl + base, full);
    Reg::Duplicate(f, int32_t{-1});
    Reg::Gather<int32_t, int32_t, uint32_t>(f, data, index, valid);
    Reg::Compares<int32_t, CMPMODE::GE>(selected, f, int32_t{0}, valid);
    Reg::Select(vflag, one, zero, selected);
    Reg::Select(vflag, vflag, zero, valid);
    Reg::And(raw, reinterpret_cast<Reg::RegTensor<uint32_t>&>(f), maskHit, full);
    Reg::Compares<uint32_t, CMPMODE::NE>(hit, raw, uint32_t{0}, full);
    Reg::Select(hflag, vflag, zero, hit);
    Reg::Sub(eflag, vflag, hflag, full);
    Reg::Compares<uint32_t, CMPMODE::NE>(hit, hflag, uint32_t{0}, full);
    Reg::Compares<uint32_t, CMPMODE::NE>(evict, eflag, uint32_t{0}, full);
    Reg::Unsqueeze<uint32_t>(hprefix, hit);
    Reg::Unsqueeze<uint32_t>(eprefix, evict);
    Reg::Add(hi, hprefix, hcarry, full);
    Reg::Add(ei, eprefix, ecarry, full);
    Reg::Duplicate(tmp, capacity - 1U);
    Reg::Sub(index, tmp, hi, full);
    Reg::Select(index, index, ei, hit);
    Reg::Adds(index, index, pa, full);
    Reg::Adds(safe, lane, capacity, full);
    Reg::Compares<uint32_t, CMPMODE::GE>(fits, lane, pa, full);
    Reg::Select(safe, safe, lane, fits);
    Reg::Compares<uint32_t, CMPMODE::NE>(selected, vflag, uint32_t{0}, full);
    Reg::Select(index, index, safe, selected);
    Reg::And(slotTag, reinterpret_cast<Reg::RegTensor<uint32_t>&>(f), keep, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data), slotTag, index, selected);
    Reg::Add(hi, hi, hflag, full);
    Reg::Add(ei, ei, eflag, full);
    Reg::Duplicate(tmp, uint32_t{63});
    Reg::Gather<uint32_t, uint32_t>(hcarry, hi, tmp);
    Reg::Gather<uint32_t, uint32_t>(ecarry, ei, tmp);
  }
  // partition刚写出，miss阶段将读取；同时flags原读生命周期结束后才可覆盖。
  Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
  for (uint32_t base = 0; base < u; base += 64U) {
    uint32_t n = u - base;
    valid = Reg::UpdateMask<uint32_t>(n);
    Reg::Adds(index, lane, ow + base, full);
    Reg::Duplicate(ownerValue, int32_t{0});
    Reg::Gather<int32_t, int32_t, uint32_t>(ownerValue, data, index, valid);
    Reg::Compares<int32_t, CMPMODE::LT>(selected, ownerValue, int32_t{0}, valid);
    Reg::Select(vflag, one, zero, selected);
    Reg::Select(vflag, vflag, zero, valid);
    Reg::Compares<uint32_t, CMPMODE::NE>(selected, vflag, uint32_t{0}, full);
    Reg::Unsqueeze<uint32_t>(rank, selected);
    Reg::Add(rank, rank, mc, full);
    Reg::Duplicate(missRank, int32_t{-1});
    Reg::Select(missRank, reinterpret_cast<Reg::RegTensor<int32_t>&>(rank), missRank, selected);
    Reg::Adds(index, lane, fl + base, full);
    Reg::Adds(safe, lane, capacity, full);
    Reg::Compares<uint32_t, CMPMODE::GE>(hit, lane, fl, full);
    Reg::Select(safe, safe, lane, hit);
    Reg::Select(index, index, safe, valid);
    Reg::Scatter<int32_t, uint32_t>(data, missRank, index, valid);
    Reg::Compare<uint32_t, CMPMODE::LT>(fits, rank, ecarry, full);
    Reg::Select(vflag, vflag, zero, fits);
    Reg::Compares<uint32_t, CMPMODE::NE>(fits, vflag, uint32_t{0}, full);
    Reg::Adds(index, rank, pa, full);
    Reg::Duplicate(tmp, pa);
    Reg::Select(index, index, tmp, fits);
    Reg::Duplicate(value, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(value, reinterpret_cast<__ubuf__ uint32_t*>(data), index, fits);
    Reg::Adds(index, lane, ow + base, full);
    Reg::Adds(safe, lane, capacity, full);
    Reg::Compares<uint32_t, CMPMODE::GE>(hit, lane, ow, full);
    Reg::Select(safe, safe, lane, hit);
    Reg::Select(index, index, safe, fits);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data), value, index, fits);
    Reg::Compares<int32_t, CMPMODE::GE>(selected, missRank, int32_t{0}, valid);
    Reg::Select(vflag, one, zero, selected);
    Reg::Select(vflag, vflag, zero, valid);
    Reg::Add(inc, rank, vflag, full);
    Reg::Duplicate(tmp, uint32_t{63});
    Reg::Gather<uint32_t, uint32_t>(mc, inc, tmp);
  }
  Reg::Add(sum, ecarry, hcarry, full);
  Reg::Compares<uint32_t, CMPMODE::EQ>(selected, sum, capacity, full);
  Reg::Select(status, one, zero, selected);
  Reg::Adds(tmp, ecarry, uint32_t{1}, full);
  Reg::Compare<uint32_t, CMPMODE::LT>(selected, mc, tmp, full);
  Reg::Select(status, status, zero, selected);
  HotPutCtx(ctx, ecarry, lane, 43, full);
  HotPutCtx(ctx, hcarry, lane, 44, full);
  HotPutCtx(ctx, mc, lane, 45, full);
  HotPutCtx(ctx, status, lane, 0, full);
}

__aicore__ inline void SortedVectorToScalar(AscendC::TPipe& pipe) {
  auto id = pipe.AllocEventID<AscendC::HardEvent::V_S>();
  AscendC::SetFlag<AscendC::HardEvent::V_S>(id);
  AscendC::WaitFlag<AscendC::HardEvent::V_S>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::V_S>(id);
}
__aicore__ inline void SortedScalarToVector(AscendC::TPipe& pipe) {
  auto id = pipe.AllocEventID<AscendC::HardEvent::S_V>();
  AscendC::SetFlag<AscendC::HardEvent::S_V>(id);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::S_V>(id);
}
template <bool Wide>
__aicore__ inline void RunAbdPlanImpl(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens,
                                  GM_ADDR tokenToReq, GM_ADDR blockTable, GM_ADDR activeRows, GM_ADDR lastReqIds,
                                  GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount,
                                  GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
                                  GM_ADDR residentRows, GM_ADDR residentBlockTable, const TilingData& d,
                                  __ubuf__ int32_t* localTopk, int64_t localRow, AscendC::TPipe& pipe, AscendC::LocalTensor<int32_t> storage, int64_t firstRow,
                                  int64_t endRow, int64_t rowCount) {
  using Layout = SortedContext<Wide>;
  constexpr bool wide = Wide;
  const int64_t rows = rowCount;
  const int64_t start = firstRow, stride = endRow;
  if (start < 0 || endRow <= start || endRow > rows) return;
  auto* scan = reinterpret_cast<__ubuf__ int32_t*>(storage.GetPhyAddr());
  auto* ctx = scan + PLAN_SCAN_STORAGE_ELEMENTS;
#define SORTED_CALL(PHASE)                                                                                           \
  asc_vf_call<SortedStageVf<PHASE, Wide>>(                                                                                 \
      dim3((PHASE) == 4 ? PLAN_THREADS : SORTED_THREADS), scan, reinterpret_cast<__gm__ int64_t*>(reqIds),           \
      reinterpret_cast<__gm__ int64_t*>(lastReqIds), reinterpret_cast<__gm__ int32_t*>(topkIndices), localTopk, localRow, \
      reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), reinterpret_cast<__gm__ int32_t*>(visibleSeqLens),        \
      reinterpret_cast<__gm__ int32_t*>(tokenToReq), reinterpret_cast<__gm__ int32_t*>(blockTable),                  \
      reinterpret_cast<__gm__ int32_t*>(slotToToken), reinterpret_cast<__gm__ int32_t*>(lruSlots),                   \
      reinterpret_cast<__gm__ int32_t*>(currentSlots), reinterpret_cast<__gm__ int32_t*>(missCount),                 \
      reinterpret_cast<__gm__ int32_t*>(missTokens), reinterpret_cast<__gm__ int32_t*>(missSlots),                   \
      reinterpret_cast<__gm__ int32_t*>(compactWorkspace), d.hashCapacity, d.workspaceRowElements, first, end, rows, \
      d.topk, d.capacity, d.maxToken, d.maxRequests, d.maxNumBlocks, d.hostNumBlocks, d.blockSize,                   \
      reinterpret_cast<__gm__ int32_t*>(rowModes), reinterpret_cast<__gm__ int32_t*>(residentRows),                  \
      reinterpret_cast<__gm__ int32_t*>(residentBlockTable), d.jointHashCapacity, d.scratchElements, d.maxJointRows)
  if (d.jointHashCapacity == 0U) {
    int64_t first = start, end = stride;
    SORTED_CALL(4);
    return;
  }
  for (int64_t first = start; first < stride;) {
    int64_t end = first + 1;
    const int32_t req = reinterpret_cast<__gm__ int32_t*>(tokenToReq)[first];
    while (end < stride && reinterpret_cast<__gm__ int32_t*>(tokenToReq)[end] == req) ++end;
    {
      SORTED_CALL(0);
      SortedVectorToScalar(pipe);
      while (ctx[0] == 1 && (ctx[6])) {
        SORTED_CALL(1);
        SortedVectorToScalar(pipe);
        if (ctx[28] == 0) break;
        if (ctx[28] == 1) {
          ctx[4] = ctx[Layout::lengths + 0];
          ctx[Layout::consumed + 0] = ctx[Layout::lengths + 0];
          const uint32_t r = ctx[Layout::active + 0];
          ctx[Layout::starts + r] = ctx[Layout::starts + r] + ctx[Layout::lengths + 0];
          ctx[7] = 1;
          SortedScalarToVector(pipe);
        } else if (ctx[28] > 1) {
          const uint32_t chunk = ctx[3];
          const uint32_t groupRows = static_cast<uint32_t>(end - first);
          const uint32_t mapRows = groupRows <= d.maxJointRows ? groupRows : d.maxJointRows;
          const uint32_t inputOffset =
              d.scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS
                  ? PLAN_SCAN_STORAGE_ELEMENTS + 64U +
                        ((mapRows * d.topk + d.capacity + 7U) & ~7U)
                  : PLAN_SCAN_STORAGE_ELEMENTS + 64U;
          auto floats = storage.ReinterpretCast<float>();
          uint16_t lens[Wide ? 8 : 4] = {};
          uint32_t consumed[Wide ? 8 : 4] = {};
          const uint32_t active = static_cast<uint32_t>(ctx[28]);
          for (uint32_t k = 0; k < active; ++k) lens[k] = static_cast<uint16_t>(ctx[Layout::lengths + k]);
          bool complete = true;
          for (uint32_t k = 0; k < active; ++k) {
            const uint32_t r = ctx[Layout::active + k];
            if (ctx[Layout::lengths + k] !=
                static_cast<uint32_t>(ctx[Layout::ends + r] - ctx[Layout::starts + r])) complete = false;
          }
          if (!wide || active <= 4U) {
            auto out = floats[inputOffset + static_cast<uint32_t>(ctx[30])];
            AscendC::MrgSortSrcList<float> lists{floats[inputOffset], floats[inputOffset + 2U * chunk],
                                                 floats[inputOffset + (active > 2U ? 4U * chunk : 0U)],
                                                 floats[inputOffset + (active > 3U ? 6U * chunk : 0U)]};
            if (complete) {
              AscendC::MrgSort<float, false>(out, lists, lens, consumed, static_cast<uint16_t>((1U << active) - 1U), 1);
              for (uint32_t k = 0; k < active; ++k) consumed[k] = lens[k];
            } else {
              AscendC::MrgSort<float, true>(out, lists, lens, consumed, static_cast<uint16_t>((1U << active) - 1U), 1);
            }
          } else {
            const uint32_t left = active / 2U, right = active - left, rightOffset = 2U * left * chunk;
            uint16_t leftLens[4] = {0, 0, 0, 0}, rightLens[4] = {0, 0, 0, 0};
            for (uint32_t k = 0; k < left; ++k) leftLens[k] = lens[k];
            for (uint32_t k = 0; k < right; ++k) rightLens[k] = lens[left + k];
            AscendC::MrgSortSrcList<float> leftLists{
                floats[inputOffset], floats[inputOffset + 2U * chunk],
                floats[inputOffset + (left > 2U ? 4U * chunk : 0U)],
                floats[inputOffset + (left > 3U ? 6U * chunk : 0U)]};
            AscendC::MrgSortSrcList<float> rightLists{
                floats[inputOffset + rightOffset], floats[inputOffset + rightOffset + 2U * chunk],
                floats[inputOffset + rightOffset + (right > 2U ? 4U * chunk : 0U)],
                floats[inputOffset + rightOffset + (right > 3U ? 6U * chunk : 0U)]};
            auto halfB = floats[inputOffset + static_cast<uint32_t>(ctx[30])];
            AscendC::MrgSort<float, false>(halfB, leftLists, leftLens, consumed, static_cast<uint16_t>((1U << left) - 1U), 1);
            uint32_t rightConsumed[4] = {0, 0, 0, 0};
            AscendC::MrgSort<float, false>(floats[inputOffset + static_cast<uint32_t>(ctx[30]) + rightOffset], rightLists,
                                           rightLens, rightConsumed, static_cast<uint16_t>((1U << right) - 1U), 1);
            SortedVectorToScalar(pipe);
            SortedScalarToVector(pipe);
            uint16_t mergedLens[4] = {0, 0, 0, 0};
            for (uint32_t k = 0; k < left; ++k) mergedLens[0] += leftLens[k];
            for (uint32_t k = 0; k < right; ++k) mergedLens[1] += rightLens[k];
            AscendC::MrgSortSrcList<float> mergedLists{
                floats[inputOffset + static_cast<uint32_t>(ctx[30])],
                floats[inputOffset + static_cast<uint32_t>(ctx[30]) + rightOffset],
                floats[inputOffset + static_cast<uint32_t>(ctx[30])],
                floats[inputOffset + static_cast<uint32_t>(ctx[30])]};
            uint32_t mergedConsumed[4] = {0, 0, 0, 0};
            AscendC::MrgSort<float, false>(floats[inputOffset], mergedLists, mergedLens, mergedConsumed, 3U, 1);
            for (uint32_t k = 0; k < active; ++k) consumed[k] = ctx[Layout::consumed + k];
            ctx[7] = 1;
          }
          SortedVectorToScalar(pipe);
          uint32_t total = 0;
          for (uint32_t k = 0; k < active; ++k) {
            if (consumed[k] > lens[k]) {
              ctx[0] = 0;
              break;
            }
            const uint32_t r = ctx[Layout::active + k];
            total += consumed[k];
            ctx[Layout::consumed + k] = consumed[k];
            ctx[Layout::starts + r] += consumed[k];
          }
          if (total == 0U) ctx[0] = 0;
          ctx[4] = static_cast<int32_t>(total);
          SortedScalarToVector(pipe);
        }

        if (d.scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS && ctx[0] == 1) {
          const uint32_t q = static_cast<uint32_t>(end - first) * static_cast<uint32_t>(d.topk);
          const uint32_t inputOffset = PLAN_SCAN_STORAGE_ELEMENTS + 64U + ((q + d.capacity + 7U) & ~7U);
          auto* records = scan + inputOffset + (ctx[7] ? 0U : static_cast<uint32_t>(ctx[30]));
          uint32_t more = 0;
          for (int64_t r = 0; r < end - first; ++r)
            if (ctx[Layout::starts + static_cast<uint32_t>(r)] < ctx[Layout::ends + static_cast<uint32_t>(r)]) more = 1;
          asc_vf_call<RefineUniqueMapVf>(records, ctx + 64U, ctx, static_cast<uint32_t>(ctx[4]), q,
                                         static_cast<uint32_t>(d.capacity), static_cast<uint32_t>(ctx[1]), ctx[2],
                                         more);
        } else {
          SORTED_CALL(3);
        }  // GM scratch retains SIMT, including B if enabled.

        SortedVectorToScalar(pipe);
      }
      if (ctx[0] == 1) {
        if (d.scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
          SORTED_CALL(6);
          SortedVectorToScalar(pipe);
          if (ctx[0] == 1) {
            const uint32_t q = static_cast<uint32_t>(end - first) * static_cast<uint32_t>(d.topk);
            asc_vf_call<HotCompactVf>(ctx + 64U, ctx, q, static_cast<uint32_t>(d.capacity),
                                      static_cast<uint32_t>(ctx[1]));
            SortedVectorToScalar(pipe);
            if (ctx[0] == 1) {
              SORTED_CALL(9);
              SortedVectorToScalar(pipe);
            }
          }
        } else {
          SORTED_CALL(5);
          SortedVectorToScalar(pipe);
        }
      }
      if (ctx[0] != 2) {
        SORTED_CALL(4);
      }
    }
    first = end;
  }
#undef SORTED_CALL
}


struct Buffers {
  GM_ADDR reqIds; GM_ADDR topkIndices; GM_ADDR stablePrefixLens; GM_ADDR visibleSeqLens;
  GM_ADDR tokenToReq; GM_ADDR blockTable; GM_ADDR activeRows; GM_ADDR lastReqIds;
  GM_ADDR slotToToken; GM_ADDR lruSlots; GM_ADDR currentSlots; GM_ADDR missCount;
  GM_ADDR missTokens; GM_ADDR missSlots; GM_ADDR compactWorkspace; GM_ADDR rowModes;
  GM_ADDR residentRows; GM_ADDR residentBlockTable;
  // Persistent producer counter.  It is deliberately kept out of the plan
  // payload so the caller can reset/elect producers without touching state.
  GM_ADDR readyCounts;
};

__aicore__ inline void RunGroup(const Buffers& b, const TilingData& d, int64_t first, int64_t end, int64_t rows,
                                AscendC::TPipe& pipe, AscendC::LocalTensor<int32_t> storage,
                                __ubuf__ int32_t* localTopk, int64_t localRow) {
  if (first < 0 || end <= first || end > rows) return;
#define RUN_GROUP(WIDE) RunAbdPlanImpl<WIDE>(b.reqIds,b.topkIndices,b.stablePrefixLens,b.visibleSeqLens,b.tokenToReq,b.blockTable,b.activeRows,b.lastReqIds,b.slotToToken,b.lruSlots,b.currentSlots,b.missCount,b.missTokens,b.missSlots,b.compactWorkspace,b.rowModes,b.residentRows,b.residentBlockTable,d,localTopk,localRow,pipe,storage,first,end,rows)
  if (d.maxJointRows > 4U) RUN_GROUP(true); else RUN_GROUP(false);
#undef RUN_GROUP
}


}  // namespace qli_fused_plan

#endif  // QLI_FUSED_PLAN_HEADER
