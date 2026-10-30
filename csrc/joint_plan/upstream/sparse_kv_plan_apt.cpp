/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * 昇腾 950PR/950DT：运行时 shape 的 Sparse KV Plan，64 个 AIV block，
 * 每个有任务的 block 调用一个 1024-thread SIMT VF。
 *
 * 本实现通过 vLLM Ascend 自定义算子接口接收持久化的 NPU LRU 状态。
 * batch、topk、capacity、max_token 均来自运行时或 tiling 数据。
 *
 * 阅读顺序：Host 启动函数 -> mixed kernel 外壳 -> SIMT VF -> ProcessRuntimeRow。
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

#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "sparse_kv_plan_config.h"

namespace {

using sparse_kv_plan_config::PLAN_ROW_UB_LIMIT_ELEMENTS;
using sparse_kv_plan_config::PLAN_SCAN_STORAGE_ELEMENTS;
using sparse_kv_plan_config::PLAN_THREADS;
using sparse_kv_plan_config::PLAN_WARP_COUNT;
using sparse_kv_plan_config::PLAN_WARP_SIZE;
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
  uint64_t jointHashCapacity;
  uint64_t scratchElements;
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

__simt_callee__ __aicore__ inline void ClearMissTail(
    int64_t base, uint32_t assigned, uint32_t topk, uint32_t thread,
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
    __ubuf__ int32_t* scanWorkspace, ScratchPtr missPositions, int64_t row, uint32_t thread,
    __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable,
    __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    uint32_t topk, uint32_t capacity, int32_t maxToken, int64_t maxRequests,
    int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {
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
        IsSourceTokenValid(row, topkIndices[topkBase + position], visibleSeqLens, tokenToReq,
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
    const int32_t token = topkIndices[topkBase + position];
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
    __gm__ int32_t* topkIndices, __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens,
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
    ProcessColdRow(scanWorkspace, missPositions, row, thread, reqIds, lastReqIds, topkIndices,
                   visibleSeqLens, tokenToReq, blockTable, slotToToken, lruSlots, currentSlots,
                   missCount, missTokens, missSlots, topk, capacity, maxToken, maxRequests,
                   maxNumBlocks, hostNumBlocks, blockSize);
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
    const int32_t token = topkIndices[topkBase + pos];
    if (IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, blockTable, maxToken, maxRequests, maxNumBlocks,
                           hostNumBlocks, blockSize)) {
      missPositions[pos] = static_cast<int32_t>(
          InsertTopkToken(hashKeys, hashFirstPos, token, static_cast<int32_t>(pos), hashCapacity));
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
          currentSlots[topkBase + position] =
              packedOwner ? (owner & OWNER_SLOT_MASK) : lruSlots[capacityBase + owner];
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
    const int32_t token = topkIndices[topkBase + position];
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

#include "sparse_kv_joint_plan.h"

/**
 * SIMT 调度层。是否使用 UB 对整个 block 一致，不会在 barrier 两侧发生线程分歧。
 * UB 路径复用当前 AIV 的单行 scratch；GM 路径按 row 切分用户的完整 workspace。
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(PLAN_THREADS) inline void SparseKvPlanRuntimeVf(
    __ubuf__ int32_t* scanWorkspace, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* compactWorkspace,
    uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart, int64_t rowStride, int64_t numReqs,
    int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks,
    int32_t blockSize, __gm__ int32_t* rowModes, __gm__ int32_t* residentRows, __gm__ int32_t* residentBlockTable,
    uint64_t jointHashCapacity, uint64_t scratchElements) {
  const JointPlanArgs args{reqIds,
                           lastReqIds,
                           topkIndices,
                           stablePrefixLens,
                           visibleSeqLens,
                           tokenToReq,
                           blockTable,
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
                           hostNumBlocks};
  int64_t group = 0;
  for (int64_t row = jointHashCapacity != 0U ? 0 : rowStart; row < numReqs;) {
    int64_t end = row + 1;
    if (jointHashCapacity != 0U) {
      while (end < numReqs && tokenToReq[end] == tokenToReq[row]) ++end;
    }
    const bool assigned = jointHashCapacity == 0U || group % rowStride == rowStart;
    if (assigned) {
      if (scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
        ProcessRequestRun(args, row, end, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace);
      } else {
        ProcessRequestRun(args, row, end, compactWorkspace + static_cast<uint64_t>(row) * scratchElements,
                          scanWorkspace);
      }
    }
    ++group;
    row = jointHashCapacity != 0U ? end : row + rowStride;
  }
}

}  // namespace

extern "C" __global__ __aicore__ void sparse_kv_plan(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                                     GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR blockTable,
                                                     GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken,
                                                     GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount,
                                                     GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace,
                                                     GM_ADDR rowModes, GM_ADDR residentRows, GM_ADDR residentBlockTable,
                                                     GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SparseKvPlanTilingData);
  GET_TILING_DATA_WITH_STRUCT(SparseKvPlanTilingData, tilingData, tiling);
  const int32_t requestedRows = *reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t numRows = requestedRows > 0 && requestedRows < tilingData.maxRows
                              ? requestedRows
                              : (requestedRows > 0 ? tilingData.maxRows : 0);
  const int64_t rowStart = static_cast<int64_t>(AscendC::GetBlockIdx());
  const int64_t rowStride = static_cast<int64_t>(AscendC::GetBlockNum());
  if (rowStart >= numRows) {
    return;  // 外层 AIV 标量分支，尚未进入 SIMT，不会破坏 block barrier。
  }
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> localWorkspace;
  pipe.InitBuffer(localWorkspace, tilingData.localMemoryBytes);
  __ubuf__ int32_t* scanWorkspace = reinterpret_cast<__ubuf__ int32_t*>(localWorkspace.Get<int32_t>().GetPhyAddr());
  asc_vf_call<SparseKvPlanRuntimeVf>(
      dim3(PLAN_THREADS), scanWorkspace, reinterpret_cast<__gm__ int64_t*>(reqIds),
      reinterpret_cast<__gm__ int64_t*>(lastReqIds), reinterpret_cast<__gm__ int32_t*>(topkIndices),
      reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), reinterpret_cast<__gm__ int32_t*>(visibleSeqLens),
      reinterpret_cast<__gm__ int32_t*>(tokenToReq), reinterpret_cast<__gm__ int32_t*>(blockTable),
      reinterpret_cast<__gm__ int32_t*>(slotToToken), reinterpret_cast<__gm__ int32_t*>(lruSlots),
      reinterpret_cast<__gm__ int32_t*>(currentSlots), reinterpret_cast<__gm__ int32_t*>(missCount),
      reinterpret_cast<__gm__ int32_t*>(missTokens), reinterpret_cast<__gm__ int32_t*>(missSlots),
      reinterpret_cast<__gm__ int32_t*>(compactWorkspace), tilingData.hashCapacity, tilingData.workspaceRowElements,
      rowStart, rowStride, numRows, tilingData.topk, tilingData.capacity, tilingData.maxToken, tilingData.maxRequests,
      tilingData.maxNumBlocks, tilingData.hostNumBlocks, tilingData.blockSize,
      reinterpret_cast<__gm__ int32_t*>(rowModes), reinterpret_cast<__gm__ int32_t*>(residentRows),
      reinterpret_cast<__gm__ int32_t*>(residentBlockTable), tilingData.jointHashCapacity, tilingData.scratchElements);
}
