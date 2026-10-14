/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * 本文件为 SparseKvPlan 的 direct launch(直调)单算子版本。
 *
 * Sparse KV Plan 微优化对照：原 BCDE + fused + 1024 的派生版本。
 * 计算源自 Sparse KV Plan；本版本的算法差异见 plan_variants_manifest.json。
 * 当前配置为 1024 个 SIMT 线程，launch bound、循环步长和 Scan 空间
 * 均使用公共配置头中的 PLAN_THREADS / PLAN_WARP_COUNT。
 * Host 显式上传 80 字节 tiling，通过 mixed kernel 外壳调用 SIMT VF。
 * block 按 query 行分配任务，并以 blockDim 为步长处理后续行；核数上限为 64。
 * 行临时区不超过 112 KiB 时使用 UB，否则使用 compactWorkspace 的 GM 行区。
 * LRU 更新、重复 TopK 处理和投机后缀失效语义保留；实机性能仍需测量。
 */

#include "variant_config.h"

#include <algorithm>
#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "sparse_kv_plan_launch.h"

// 部分直调工具链不带任务类型注册宏:已定义则使用,未定义则置空
//(直调下任务类型由启动配置决定,不影响 AIV-only kernel 的正确性)。
#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(type)
#endif

namespace {

using sparse_kv_plan::PLAN_ROW_UB_LIMIT_ELEMENTS;
using sparse_kv_plan::PLAN_SCAN_STORAGE_ELEMENTS;
using sparse_kv_plan::PLAN_THREADS;
using sparse_kv_plan::PLAN_WARP_COUNT;
using sparse_kv_plan::PLAN_WARP_SIZE;
// Options are compile-time constants, not per-call runtime switches.
constexpr bool OPT_A = (PLAN_OPT_MASK & (1U << 0U)) != 0U;
constexpr bool OPT_B = (PLAN_OPT_MASK & (1U << 1U)) != 0U;
constexpr bool OPT_C = (PLAN_OPT_MASK & (1U << 2U)) != 0U;
constexpr bool OPT_D = (PLAN_OPT_MASK & (1U << 3U)) != 0U;
constexpr bool OPT_E = (PLAN_OPT_MASK & (1U << 4U)) != 0U;
constexpr bool OPT_F = (PLAN_OPT_MASK & (1U << 5U)) != 0U;
constexpr bool OPT_G = (PLAN_OPT_MASK & (1U << 6U)) != 0U;
constexpr bool OPT_H = (PLAN_OPT_MASK & (1U << 7U)) != 0U;

constexpr int32_t INVALID_TOKEN = -1;
constexpr int32_t PENDING_SLOT = -2;
constexpr int32_t HASH_EMPTY = -1;
constexpr int32_t HASH_POSITION_EMPTY = INT32_MAX;
constexpr uint32_t HASH_BUCKET_INVALID = UINT32_MAX;

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
// 每个 tile 最多 2048 项，而 2^12=4096，因此低位不会进位到 hit 字段。
constexpr uint32_t COUNT_SHIFT = 12U;
constexpr int32_t COUNT_MASK = (1U << COUNT_SHIFT) - 1U;

static_assert(PLAN_THREADS < (1U << COUNT_SHIFT), "packed scan would carry");


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
  if constexpr (OPT_F && PLAN_WARP_COUNT <= PLAN_WARP_SIZE) {
    const int32_t value = lane < PLAN_WARP_COUNT ? warpTotals[lane] : 0;
    int32_t inclusive = value;
#pragma unroll
    for (uint32_t offset = 1U; offset < PLAN_WARP_SIZE; offset <<= 1U) {
      const int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
      if (lane >= offset) inclusive += prior;
    }
    if (lane < PLAN_WARP_COUNT) warpBases[lane] = inclusive - value;
    // Padding lanes participate, so lane 31 also holds the full total.
    if (lane == PLAN_WARP_SIZE - 1U) control[CONTROL_TOTAL] = inclusive;
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
 * H 版利用下一轮首个 barrier 保护结果消费；最后一轮仍保留 scatter 后同步。
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

  // Callers consume the bases/total before the next first scan barrier.
  // The final tile must also drain its scatter before stage/scratch reuse.
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

// The next scan writes only warpTotals before its first barrier. That
// barrier drains all previous warpBases/CONTROL_TOTAL consumers before
// PublishWarpBases overwrites them. Keep the final tile's scatter barrier.
__simt_callee__ __aicore__ inline void FinishTile(uint32_t tile, uint32_t extent) {
  if constexpr (OPT_H) {
    if (extent - tile <= PLAN_THREADS) asc_syncthreads();
  } else {
    asc_syncthreads();
  }
}

__simt_callee__ __aicore__ inline void ClearMissTail(
    int64_t base, uint32_t assigned, uint32_t topk, uint32_t thread,
    __gm__ int32_t* missTokens, __gm__ int32_t* missSlots) {
  if constexpr (OPT_C) {
    for (uint32_t pos = assigned + thread; pos < topk; pos += PLAN_THREADS) {
      missTokens[base + pos] = INVALID_TOKEN;
      missSlots[base + pos] = INVALID_TOKEN;
    }
  }
}

// Row-local arithmetic remains int64 for table offsets. A negative visible
// length means no visible tokens, rather than a large unsigned upper bound.
struct RowSourceContext {
  int64_t tableBase = 0;
  uint32_t upper = 0;
  uint32_t blockSize = 0;
  uint32_t blockShift = UINT32_MAX;
  bool valid = false;
};

__simt_callee__ __aicore__ inline RowSourceContext PrepareRowSource(
    int64_t row, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    int32_t maxToken, int64_t maxRequests, int64_t maxNumBlocks, int32_t blockSize) {
  RowSourceContext context;
  if constexpr (OPT_E) {
    const int32_t request = tokenToReq[row];
    context.valid = request >= 0 && request < maxRequests && blockSize > 0;
    const int32_t visible = visibleSeqLens[row];
    context.upper = visible <= 0 ? 0U : static_cast<uint32_t>(visible < maxToken ? visible : maxToken);
    if (context.valid) {
      context.tableBase = static_cast<int64_t>(request) * maxNumBlocks;
      context.blockSize = static_cast<uint32_t>(blockSize);
      if ((context.blockSize & (context.blockSize - 1U)) == 0U) {
        context.blockShift = 0U;
        for (uint32_t size = context.blockSize; size > 1U; size >>= 1U) ++context.blockShift;
      }
    }
  }
  return context;
}

__simt_callee__ __aicore__ inline bool IsRowSourceValid(
    const RowSourceContext& context, int64_t row, int32_t token,
    __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable,
    int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks,
    int64_t hostNumBlocks, int32_t blockSize) {
  if constexpr (OPT_E) {
    const uint32_t unsignedToken = static_cast<uint32_t>(token);
    if (!context.valid || unsignedToken >= context.upper) return false;
    const uint32_t block = context.blockShift == UINT32_MAX
        ? unsignedToken / context.blockSize : unsignedToken >> context.blockShift;
    if (static_cast<int64_t>(block) >= maxNumBlocks) return false;
    const int32_t physical = blockTable[context.tableBase + block];
    return physical >= 0 && static_cast<int64_t>(physical) < hostNumBlocks;
  } else {
    return IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, blockTable,
                              maxToken, maxRequests, maxNumBlocks, hostNumBlocks, blockSize);
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
  const RowSourceContext source = PrepareRowSource(row, visibleSeqLens, tokenToReq, maxToken,
                                                    maxRequests, maxNumBlocks, blockSize);
  int32_t count = 0;
  for (uint32_t tile = 0U; tile < topk; tile += PLAN_THREADS) {
    const uint32_t position = tile + thread;
    const int32_t flag = position < topk &&
        IsRowSourceValid(source, row, topkIndices[topkBase + position], visibleSeqLens, tokenToReq,
                           blockTable, maxToken, maxRequests, maxNumBlocks, hostNumBlocks, blockSize);
    const int32_t rank = BlockExclusiveScan(scanWorkspace, flag, thread);
    const int32_t total = control[CONTROL_TOTAL];
    if (flag != 0) {
      missPositions[count + rank] = static_cast<int32_t>(position);
    }
    FinishTile(tile, topk);  // Final tile drains compressed-position writes.
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
 * 一次调用只处理一条请求。只有临时数组允许放 UB；reqId、resident、LRU 等
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
    if constexpr (!OPT_C) {
      missTokens[topkBase + pos] = INVALID_TOKEN;
      missSlots[topkBase + pos] = INVALID_TOKEN;
    }
  }
  bool reset = false;
  if constexpr (OPT_G) {
    const int32_t request = tokenToReq[row];
    const int64_t id = request >= 0 && request < maxRequests ? reqIds[request] : -1;
    reset = lastReqIds[row] != id;
    if (thread == 0U) missCount[row] = 0;
    // Every cold thread reaches its first scan barrier before lastReqIds
    // is changed. Keep output initialization above this uniform branch.
  } else {
    if (thread == 0U) {
      missCount[row] = 0;
      const int32_t request = tokenToReq[row];
      const int64_t id = request >= 0 && request < maxRequests ? reqIds[request] : -1;
      control[CONTROL_RESET] = lastReqIds[row] != id ? 1 : 0;
    }
    asc_syncthreads();
    reset = control[CONTROL_RESET] != 0;
  }
  if (reset) {
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

  const RowSourceContext source = PrepareRowSource(row, visibleSeqLens, tokenToReq, maxToken,
                                                    maxRequests, maxNumBlocks, blockSize);
  // 阶段 3：仅插入合法 TopK token。所有插入结束后，才允许普通 load 查询 hash。
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    const int32_t token = topkIndices[topkBase + pos];
    if (IsRowSourceValid(source, row, token, visibleSeqLens, tokenToReq, blockTable,
                           maxToken, maxRequests, maxNumBlocks,
                           hostNumBlocks, blockSize)) {
      if constexpr (!OPT_A) currentSlots[topkBase + pos] = PENDING_SLOT;
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
  // Signed atomic_max requires the packed order to fit its 15 high bits.
  const bool packedOwner = OPT_D && capacity <= 32768U;
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
        uint32_t bucket = HASH_BUCKET_INVALID;
        if constexpr (OPT_E) {
          if (static_cast<uint32_t>(token) < static_cast<uint32_t>(stablePrefix)) {
            bucket = LookupTopkToken(hashKeys, token, hashCapacity);
          } else if (static_cast<uint32_t>(token) < static_cast<uint32_t>(maxToken)) {
            slotToToken[capacityBase + slot] = INVALID_TOKEN;
          }
        } else {
          if (token >= 0 && token < maxToken && token >= stablePrefix) {
            slotToToken[capacityBase + slot] = INVALID_TOKEN;
            token = INVALID_TOKEN;
          }
          if (token >= 0 && token < maxToken) bucket = LookupTopkToken(hashKeys, token, hashCapacity);
        }
        if (bucket != HASH_BUCKET_INVALID) {
          hitFlag = 1;
          const int32_t owner = packedOwner
              ? static_cast<int32_t>((order << 16U) | static_cast<uint32_t>(slot))
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
    FinishTile(tile, capacity);
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
          currentSlots[topkBase + position] = packedOwner ? (owner & 0xffff) : lruSlots[capacityBase + owner];
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
    FinishTile(tile, topk);
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
  // A removes cross-thread pending cleanup; B removes missSlots readback.
  // Only their combination makes this assignment barrier redundant.
  if constexpr (!(OPT_A && OPT_B)) asc_syncthreads();

  // 阶段 8：LRU = 未使用 evictable + 新 miss + 原 hit，顺序与原算子一致。
  const int32_t unassignedCount = evictableCount - assignCount;
  const int32_t validLruCount = evictableCount + hitCount;
  for (uint32_t output = thread; output < static_cast<uint32_t>(validLruCount); output += PLAN_THREADS) {
    int32_t slot;
    if (output < static_cast<uint32_t>(unassignedCount)) {
      slot = evictableSlots[assignCount + output];
    } else if (output < static_cast<uint32_t>(evictableCount)) {
      if constexpr (OPT_B) slot = evictableSlots[output - static_cast<uint32_t>(unassignedCount)];
      else slot = missSlots[topkBase + output - unassignedCount];
    } else {
      slot = evictableSlots[capacity - 1U - (output - static_cast<uint32_t>(evictableCount))];
    }
    lruSlots[capacityBase + output] = slot;
  }
  if constexpr (!OPT_A) {
    // Assignment writes may come from other threads, hence the barrier above.
    for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
      if (currentSlots[topkBase + pos] == PENDING_SLOT) currentSlots[topkBase + pos] = INVALID_TOKEN;
    }
  }
  if (thread == 0U) {
    missCount[row] = assignCount;
  }
  // 下一行可能立刻复用 UB，必须等当前行全部线程写完。
  asc_syncthreads();
}

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
    int32_t blockSize) {
  for (int64_t row = rowStart; row < numReqs; row += rowStride) {
    if (workspaceRowElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
      __ubuf__ int32_t* rowWorkspace = scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS;
      ProcessRuntimeRow(scanWorkspace, row, reqIds, lastReqIds, topkIndices, stablePrefixLens, visibleSeqLens,
                        tokenToReq, blockTable, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                        rowWorkspace, static_cast<uint32_t>(hashCapacity), static_cast<uint32_t>(topk),
                        static_cast<uint32_t>(capacity), static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks,
                        hostNumBlocks, blockSize);
    } else {
      __gm__ int32_t* rowWorkspace = compactWorkspace + static_cast<uint64_t>(row) * workspaceRowElements;
      ProcessRuntimeRow(scanWorkspace, row, reqIds, lastReqIds, topkIndices, stablePrefixLens, visibleSeqLens,
                        tokenToReq, blockTable, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                        rowWorkspace, static_cast<uint32_t>(hashCapacity), static_cast<uint32_t>(topk),
                        static_cast<uint32_t>(capacity), static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks,
                        hostNumBlocks, blockSize);
    }
  }
}

}  // namespace

extern "C" __global__ __aicore__ void sparse_kv_plan_kernel(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                                     GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR blockTable,
                                                     GM_ADDR activeRows, GM_ADDR lastReqIds,
                                                     GM_ADDR slotToToken,
                                                     GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount,
                                                     GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace,
                                                     GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  const __gm__ SparseKvPlanTilingData* tilingGm =
      reinterpret_cast<const __gm__ SparseKvPlanTilingData*>(tiling);
  SparseKvPlanTilingData tilingData;
  tilingData.hashCapacity = tilingGm->hashCapacity;
  tilingData.workspaceRowElements = tilingGm->workspaceRowElements;
  tilingData.maxRows = tilingGm->maxRows;
  tilingData.topk = tilingGm->topk;
  tilingData.capacity = tilingGm->capacity;
  tilingData.maxToken = tilingGm->maxToken;
  tilingData.maxRequests = tilingGm->maxRequests;
  tilingData.maxNumBlocks = tilingGm->maxNumBlocks;
  tilingData.hostNumBlocks = tilingGm->hostNumBlocks;
  tilingData.blockSize = tilingGm->blockSize;
  tilingData.localMemoryBytes = tilingGm->localMemoryBytes;
  const int32_t requestedRows = *reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t numRows = requestedRows > 0 && requestedRows < tilingData.maxRows
                              ? requestedRows
                              : (requestedRows > 0 ? tilingData.maxRows : 0);
  const int64_t rowStart = static_cast<int64_t>(AscendC::GetBlockIdx());
  const int64_t rowStride = static_cast<int64_t>(AscendC::GetBlockNum());
  if (rowStart >= numRows) {
    return;  // 外层 AIV 标量分支,尚未进入 SIMT,不会破坏 block barrier。
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
      tilingData.maxNumBlocks, tilingData.hostNumBlocks, tilingData.blockSize);
}

// ---------------------------------------------------------------------------
// 以下为 host 侧直调支持代码(bisheng 编译,与 kernel 同一目标文件)。
// ---------------------------------------------------------------------------

#include "platform/platform_ascendc.h"

extern "C" uint32_t calc_sparse_kv_plan_block_dim(void) {
  uint32_t coreNum = 0U;
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (platform != nullptr) {
    coreNum = static_cast<uint32_t>(platform->GetCoreNumAiv());
  }
  if (coreNum == 0U) {
    // 平台查询失败时退回 64:多余 block 在 kernel 外壳直接返回,结果仍正确。
    coreNum = SPARSE_KV_PLAN_MAX_BLOCKS;
  }
  return std::min(SPARSE_KV_PLAN_MAX_BLOCKS, coreNum);
}

extern "C" void launch_sparse_kv_plan(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                      GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR blockTable,
                                      GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots,
                                      GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                                      GM_ADDR compactWorkspace, GM_ADDR tiling, uint32_t blockDim,
                                      uint32_t dynUBufSize, void* stream) {
  sparse_kv_plan_kernel<<<blockDim, dynUBufSize, stream>>>(
      reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable, activeRows, lastReqIds,
      slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, tiling);
}
