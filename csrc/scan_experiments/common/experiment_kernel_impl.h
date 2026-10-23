/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * 本文件为 SparseKvPlan 的 direct launch(直调)单算子版本。
 *
 * 固定 ABCDF 为基线；本文件仅实现本轮独立 Scan 策略组合。
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
#include "experiment_launch.h"
#include "simt_api/device_functions.h"

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
constexpr uint32_t CONTROL_HIT_TOTAL = 2U;  // 整段LRU使用独立hit计数，避免12位进位。

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

// 本轮字母与历史 ABCDF 分开编号；历史 PLAN_OPT_MASK 固定为47。
struct ExperimentScanResult { int32_t rank; int32_t total; };

template <bool Ballot, bool Packed, bool SelfReduce>
__simt_callee__ __aicore__ inline ExperimentScanResult ExperimentScan(
    __ubuf__ int32_t* workspace, int32_t value, uint32_t thread) {
  if constexpr (!Ballot && !SelfReduce) {
    // 未开启该阶段优化时直接调用原 ABCDF Scan，保持对照的改动范围。
    const int32_t rank = BlockExclusiveScan(workspace, value, thread);
    return {rank, workspace[CONTROL_OFFSET + CONTROL_TOTAL]};
  }
  const uint32_t lane = thread & 31U;
  const uint32_t warp = thread / 32U;
  int32_t localRank;
  int32_t warpTotal;
  if constexpr (Ballot) {
    const uint32_t before = (1U << lane) - 1U;  // lane=0..31，不会移位32位。
    const uint32_t low = asc_ballot(Packed ? ((value & COUNT_MASK) != 0) : (value != 0));
    localRank = __popc(low & before);
    warpTotal = __popc(low);
    if constexpr (Packed) {
      const uint32_t high = asc_ballot((value >> COUNT_SHIFT) != 0);
      localRank += __popc(high & before) << COUNT_SHIFT;
      warpTotal += __popc(high) << COUNT_SHIFT;
    }
  } else {
    int32_t inclusive = value;
    for (uint32_t offset = 1U; offset < 32U; offset <<= 1U) {
      const int32_t prior = asc_shfl_up(inclusive, offset, 32U);
      if (lane >= offset) inclusive += prior;
    }
    localRank = inclusive - value;
    warpTotal = inclusive;  // 只有 lane31 写入，那里已经是完整 warp 总数。
  }
  if (lane == 31U) workspace[WARP_TOTALS_OFFSET + warp] = warpTotal;
  asc_syncthreads();  // 生产者→消费者：所有 warpTotals 已发布。
  if constexpr (SelfReduce) {
    static_assert(PLAN_WARP_COUNT <= 32U, "SelfReduce requires <=32 warps");
    const int32_t count = lane < PLAN_WARP_COUNT ? workspace[WARP_TOTALS_OFFSET + lane] : 0;
    const int32_t base = asc_reduce_add(lane < warp ? count : 0);
    const int32_t total = asc_reduce_add(count);
    // 各 warp 自持 base/total，不再向其他 warp 发布结果。
    // 调用者保留 scatter 后 barrier，防止下一轮提前覆盖 warpTotals。
    return {base + localRank, total};
  } else {
    PublishWarpBases(workspace, thread);
    asc_syncthreads();
    return {workspace[WARP_BASES_OFFSET + warp] + localRank,
            workspace[CONTROL_OFFSET + CONTROL_TOTAL]};
  }
}

// 整段前缀：warp0 依次处理32个 chunk 一组，组间 carry 保存在寄存器。
// 不在组间做 block barrier；调用者负责 mask 发布与 base 消费的同步。
// 两套计数独立保存，避免整段4096项使12位 packed低字段进位。
template <bool Dual, typename ScratchPtr>
__simt_callee__ __aicore__ inline void PrefixChunks(
    ScratchPtr masksLow, ScratchPtr masksHigh, ScratchPtr basesLow, ScratchPtr basesHigh,
    uint32_t chunks, __ubuf__ int32_t* workspace, uint32_t thread) {
  if (thread >= 32U) return;
  const uint32_t lane = thread;
  int32_t carryLow = 0;
  int32_t carryHigh = 0;
  for (uint32_t group = 0; group < chunks; group += 32U) {
    const uint32_t chunk = group + lane;
    const int32_t low = chunk < chunks ? __popc(static_cast<uint32_t>(masksLow[chunk])) : 0;
    int32_t high = 0;
    if constexpr (Dual) {
      high = chunk < chunks ? __popc(static_cast<uint32_t>(masksHigh[chunk])) : 0;
    }
    int32_t inclusiveLow = low;
    int32_t inclusiveHigh = high;
    for (uint32_t offset = 1U; offset < 32U; offset <<= 1U) {
      const int32_t prevLow = asc_shfl_up(inclusiveLow, offset, 32U);
      int32_t prevHigh = 0;
      if constexpr (Dual) prevHigh = asc_shfl_up(inclusiveHigh, offset, 32U);
      if (lane >= offset) {
        inclusiveLow += prevLow;
        if constexpr (Dual) inclusiveHigh += prevHigh;
      }
    }
    if (chunk < chunks) {
      basesLow[chunk] = carryLow + inclusiveLow - low;
      if constexpr (Dual) basesHigh[chunk] = carryHigh + inclusiveHigh - high;
    }
    carryLow += asc_shfl(inclusiveLow, 31U, 32U);
    if constexpr (Dual) carryHigh += asc_shfl(inclusiveHigh, 31U, 32U);
  }
  if (lane == 0U) {
    workspace[CONTROL_OFFSET + CONTROL_TOTAL] = carryLow;
    if constexpr (Dual) workspace[CONTROL_OFFSET + CONTROL_HIT_TOTAL] = carryHigh;
  }
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
    int32_t rank;
    int32_t total;
    if constexpr (PLAN_COLD_BALLOT != 0) {
      const ExperimentScanResult scan = ExperimentScan<true, false, false>(scanWorkspace, flag, thread);
      rank = scan.rank;
      total = scan.total;
    } else {
      rank = BlockExclusiveScan(scanWorkspace, flag, thread);
      total = control[CONTROL_TOTAL];
    }
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
 * 本helper一次处理一条query行。只有临时数组允许放 UB；reqId、resident、LRU 等
 * 持久状态始终保存在 GM，并由 row 定位，不能依赖上一轮由哪个物理核处理。
 *
 * rowWorkspace 的五段布局（单位 int32）：
 * keys[H]、firstPos[H]、owner[H]、partitionSlots[C]、missPositions[T]。
 * H=hashCapacity，C=capacity，T=topk；D/E附加区接在这五段之后。
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
  if constexpr (PLAN_LRU_MODE == 2) {
    const uint32_t chunks = (capacity + 31U) / 32U;
    ScratchPtr savedSlots = missPositions + topk;
    ScratchPtr evictMasks = savedSlots + capacity;
    ScratchPtr hitMasks = evictMasks + chunks;
    ScratchPtr evictBases = hitMasks + chunks;
    ScratchPtr hitBases = evictBases + chunks;
    const uint32_t lane = thread & 31U;
    const uint32_t before = (1U << lane) - 1U;
    // Pass1：快照原始槽号和分类位图；保留原 hashOwner 的 atomic_max 规则。
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

      if (order < capacity) savedSlots[order] = slot;
      const uint32_t evicts = asc_ballot(evictFlag);
      const uint32_t hits = asc_ballot(hitFlag);
      const uint32_t chunk = order / 32U;
      if (lane == 0U && chunk < chunks) {
        evictMasks[chunk] = static_cast<int32_t>(evicts);
        hitMasks[chunk] = static_cast<int32_t>(hits);
      }
    }
    asc_syncthreads();
    PrefixChunks<true>(evictMasks, hitMasks, evictBases, hitBases, chunks, scanWorkspace, thread);
    asc_syncthreads();
    evictableCount = control[CONTROL_TOTAL];
    hitCount = control[CONTROL_HIT_TOTAL];
    for (uint32_t order = thread; order < capacity; order += PLAN_THREADS) {
      const uint32_t chunk = order / 32U;
      const uint32_t evicts = static_cast<uint32_t>(evictMasks[chunk]);
      const uint32_t hits = static_cast<uint32_t>(hitMasks[chunk]);
      const uint32_t bit = 1U << lane;
      const int32_t slot = savedSlots[order];
      if ((evicts & bit) != 0U) {
        evictableSlots[evictBases[chunk] + __popc(evicts & before)] = slot;
      }
      if ((hits & bit) != 0U) {
        const uint32_t rank = static_cast<uint32_t>(hitBases[chunk] + __popc(hits & before));
        evictableSlots[capacity - 1U - rank] = slot;
      }
    }
    asc_syncthreads();  // 所有 scatter 完成，才进入 hit/miss 和下一阶段 scratch 使用。
  } else {
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
      const ExperimentScanResult scan = ExperimentScan<(PLAN_LRU_MODE == 1), true,
          (PLAN_SELF_REDUCE != 0)>(scanWorkspace, packed, thread);
      const int32_t ranks = scan.rank;
      const int32_t totals = scan.total;
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

  }
  // F: consume per-query bucket before in-place stable compaction.
  int32_t localMissCount = 0;
  if constexpr (PLAN_MISS_MODE == 2) {
    const uint32_t chunks = (topk + 31U) / 32U;
    const uint32_t lruExtra = PLAN_LRU_MODE == 2 ? capacity + 4U * ((capacity + 31U) / 32U) : 0U;
    ScratchPtr masks = missPositions + topk + lruExtra;
    ScratchPtr bases = masks + chunks;
    const uint32_t lane = thread & 31U;
    const uint32_t before = (1U << lane) - 1U;
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

      const uint32_t mask = asc_ballot(missFlag);
      const uint32_t chunk = position / 32U;
      if (lane == 0U && chunk < chunks) masks[chunk] = static_cast<int32_t>(mask);
    }
    // 所有旧 bucket 均已消费，才允许压缩写回 missPositions。
    asc_syncthreads();
    PrefixChunks<false>(masks, masks, bases, bases, chunks, scanWorkspace, thread);
    asc_syncthreads();
    localMissCount = control[CONTROL_TOTAL];
    for (uint32_t position = thread; position < topk; position += PLAN_THREADS) {
      const uint32_t chunk = position / 32U;
      const uint32_t mask = static_cast<uint32_t>(masks[chunk]);
      if ((mask & (1U << lane)) != 0U) {
        missPositions[bases[chunk] + __popc(mask & before)] = static_cast<int32_t>(position);
      }
    }
    asc_syncthreads();
  } else {
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
      const ExperimentScanResult scan = ExperimentScan<(PLAN_MISS_MODE == 1), false,
          (PLAN_SELF_REDUCE != 0)>(scanWorkspace, missFlag, thread);
      const int32_t rank = scan.rank;
      const int32_t tileMissCount = scan.total;
      if (missFlag != 0) {
        missPositions[localMissCount + rank] = static_cast<int32_t>(position);
      }
      FinishTile(tile, topk);
      localMissCount += tileMissCount;
    }

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
