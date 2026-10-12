/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * 昇腾 950PR/950DT：运行时 shape 的 Sparse KV Plan，64 个 AIV block，
 * 每个有任务的 block 调用一个 2048-thread SIMT VF。
 *
 * 本实现通过 vLLM Ascend 自定义算子接口接收持久化的 NPU LRU 状态。
 * batch、topk、capacity、max_token 均来自运行时或 tiling 数据。
 *
 * 阅读顺序：Host 启动函数 -> mixed kernel 外壳 -> SIMT VF -> ProcessRuntimeRow。
 * 一条 request 的 Plan 由一个 block 完成，避免引入跨 block 同步。
 * block b 处理 b、b+blockDim、b+2*blockDim……行；B 小于 blockDim 时，
 * 多余 block 在外壳直接返回。blockDim 上限为 64，实际值取决于目标 SKU 的 AIV 数量。
 *
 * 相对原版的主要修改：
 * 1. 2048 线程对应 64 个 warp；重新实现两层 scan，不能只修改线程数常量。
 * 2. 单行临时数组不超过 112 KiB 时放进片上 UB，否则使用原 GM workspace。
 * 3. 用一个 32 位整数打包 hit/evict 两种标志，一次 scan 得到两种稳定排名。
 * 4. 直接遍历 hashOwner 发布 resident hit，省去原来的第二轮 resident 哈希查询。
 * 5. 行内循环采用 32 位无符号下标；跨行偏移、地址和 workspace 大小保留 64 位。
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
 * - 2048 是线程配置，并不保证比 1024 快。应检查目标 CANN 的编译资源报告、
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

using sparse_kv_plan::PLAN_ROW_UB_LIMIT_ELEMENTS;
using sparse_kv_plan::PLAN_SCAN_STORAGE_ELEMENTS;
using sparse_kv_plan::PLAN_THREADS;
using sparse_kv_plan::PLAN_WARP_COUNT;
using sparse_kv_plan::PLAN_WARP_SIZE;
constexpr int32_t INVALID_TOKEN = -1;
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

/**
 * 对完整 SIMT block 做独占前缀和，返回当前线程之前的输入总和。
 * 输入可以是 0/1，也可以是打包后的 evict + (hit << 12)。
 *
 * 第一级：每个 warp 的 32 个线程用 shuffle 计算 inclusive scan。
 * 第二级：线程 0 顺序扫描所有 warp total，生成每个 warp 的起点。
 * warp 数完全由 PLAN_THREADS 推导，后续调整线程数无需重写 scan 布局。
 *
 * 所有线程必须参加所有 barrier。尾 tile 的无效线程输入 0，不能提前返回。
 * 最后的 barrier 还保证所有线程已读取 warpBases，下一次 scan 才能复用它。
 */
__simt_callee__ __aicore__ inline int32_t BlockExclusiveScan(__ubuf__ int32_t* workspace, int32_t value,
                                                             uint32_t thread) {
  __ubuf__ int32_t* warpTotals = workspace + WARP_TOTALS_OFFSET;
  __ubuf__ int32_t* warpBases = workspace + WARP_BASES_OFFSET;
  __ubuf__ int32_t* control = workspace + CONTROL_OFFSET;
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

  if (thread == 0U) {
    int32_t blockTotal = 0;
    for (uint32_t warpIndex = 0U; warpIndex < PLAN_WARP_COUNT; ++warpIndex) {
      warpBases[warpIndex] = blockTotal;
      blockTotal += warpTotals[warpIndex];
    }
    control[CONTROL_TOTAL] = blockTotal;
  }
  asc_syncthreads();

  const int32_t exclusive = warpBases[warp] + inclusive - value;
  asc_syncthreads();
  return exclusive;
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
__simt_callee__ __aicore__ inline void InsertTopkToken(ScratchPtr keys, ScratchPtr firstPositions, int32_t token,
                                                       int32_t position, uint32_t hashCapacity) {
  uint32_t bucket = HashToken(token, hashCapacity);
  const uint32_t mask = hashCapacity - 1U;
  for (uint32_t probe = 0U; probe < hashCapacity; ++probe) {
    const int32_t old = asc_atomic_cas(keys + bucket, HASH_EMPTY, token);
    if (old == HASH_EMPTY || old == token) {
      asc_atomic_min(firstPositions + bucket, position);
      return;
    }
    bucket = (bucket + 1U) & mask;
  }
  // 合法 helper 产生至少 2*topk 个桶，因此不会填满；保留有界循环防止死循环。
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

/**
 * 一次调用只处理一条请求。只有临时数组允许放 UB；reqId、resident、LRU 等
 * 持久状态始终保存在 GM，并由 row 定位，不能依赖上一轮由哪个物理核处理。
 *
 * rowWorkspace 的六段布局（单位 int32）：
 * keys[H]、firstPos[H]、owner[H]、evictSlots[C]、hitSlots[C]、missPositions[T]。
 * H=hashCapacity，C=capacity，T=topk，总大小为 (3H+2C+T)*4 字节。
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
  ScratchPtr hitSlots = evictableSlots + capacity;
  ScratchPtr missPositions = hitSlots + capacity;

  // 阶段 1：每个线程以 2048 为步长初始化输出与 hash，支持任意合法 runtime T/H。
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    currentSlots[topkBase + pos] = INVALID_TOKEN;
    missTokens[topkBase + pos] = INVALID_TOKEN;
    missSlots[topkBase + pos] = INVALID_TOKEN;
  }
  for (uint32_t i = thread; i < hashCapacity; i += PLAN_THREADS) {
    hashKeys[i] = HASH_EMPTY;
    hashFirstPos[i] = HASH_POSITION_EMPTY;
    hashOwner[i] = INVALID_TOKEN;
  }
  if (thread == 0U) {
    missCount[row] = 0;
    const int32_t request = tokenToReq[row];
    const int64_t currentReqId = request >= 0 && request < maxRequests ? reqIds[request] : -1;
    control[CONTROL_RESET] = lastReqIds[row] != currentReqId ? 1 : 0;
  }
  asc_syncthreads();

  // 阶段 2：同一 batch 行换成新 request 时，必须重置该行缓存和 LRU。
  if (control[CONTROL_RESET] != 0) {
    for (uint32_t slot = thread; slot < capacity; slot += PLAN_THREADS) {
      slotToToken[capacityBase + slot] = INVALID_TOKEN;
      lruSlots[capacityBase + slot] = static_cast<int32_t>(slot);
    }
    if (thread == 0U) {
      const int32_t request = tokenToReq[row];
      lastReqIds[row] = request >= 0 && request < maxRequests ? reqIds[request] : -1;
    }
  }
  asc_syncthreads();

  // 阶段 3：仅插入合法 TopK token。所有插入结束后，才允许普通 load 查询 hash。
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    const int32_t token = topkIndices[topkBase + pos];
    if (IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, blockTable, maxToken, maxRequests, maxNumBlocks,
                           hostNumBlocks, blockSize)) {
      InsertTopkToken(hashKeys, hashFirstPos, token, static_cast<int32_t>(pos), hashCapacity);
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
          asc_atomic_max(hashOwner + bucket, static_cast<int32_t>(order));
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
      hitSlots[hitCount + hitRank] = slot;
    }
    asc_syncthreads();
    evictableCount += totals & COUNT_MASK;
    hitCount += totals >> COUNT_SHIFT;
  }

  // 阶段 5：owner 已经是旧 LRU 中最后一个命中位置，直接按桶发布结果。
  // 同桶只有一个线程写 firstPos，避免重复 resident token 的非确定性覆盖。
  // 与原版相比，去掉遍历 capacity 后再次 LookupTopkToken 的随机探测。
  for (uint32_t bucket = thread; bucket < hashCapacity; bucket += PLAN_THREADS) {
    const int32_t owner = hashOwner[bucket];
    if (owner >= 0) {
      const int32_t position = hashFirstPos[bucket];
      if (position >= 0 && static_cast<uint32_t>(position) < topk) {
        currentSlots[topkBase + position] = lruSlots[capacityBase + owner];
      }
    }
  }
  asc_syncthreads();

  // 阶段 6：按 TopK 原顺序压缩 miss 位置。跨 tile 的 carry 累加到 localMissCount。
  int32_t localMissCount = 0;
  for (uint32_t tile = 0U; tile < topk; tile += PLAN_THREADS) {
    const uint32_t position = tile + thread;
    int32_t missFlag = 0;
    if (position < topk) {
      const int32_t token = topkIndices[topkBase + position];
      missFlag = currentSlots[topkBase + position] < 0 &&
                 IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, blockTable, maxToken, maxRequests,
                                    maxNumBlocks, hostNumBlocks, blockSize);
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
  asc_syncthreads();

  // 阶段 8：LRU = 未使用 evictable + 新 miss + 原 hit，顺序与原算子一致。
  const int32_t unassignedCount = evictableCount - assignCount;
  const int32_t validLruCount = evictableCount + hitCount;
  for (uint32_t output = thread; output < static_cast<uint32_t>(validLruCount); output += PLAN_THREADS) {
    int32_t slot;
    if (output < static_cast<uint32_t>(unassignedCount)) {
      slot = evictableSlots[assignCount + output];
    } else if (output < static_cast<uint32_t>(evictableCount)) {
      slot = missSlots[topkBase + output - unassignedCount];
    } else {
      slot = hitSlots[output - evictableCount];
    }
    lruSlots[capacityBase + output] = slot;
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

// End of the unmodified upstream compute/VF section.
