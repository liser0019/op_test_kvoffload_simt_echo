/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Sparse KV Plan with ordered queries and bitmap resident lookup.
 * Source: op_test_kvoffload_simt_echo@ba4c5dd9bb6aaa432abf4822607ce224bec51862.
 * One AIV per request group; MTP0/1/3 correspond to 1/2/4 query rows.
 * TopK=2048; capacity=4096/4096/8192; maxToken<=358400.
 *
 * Reading order: RunOrderedBitmapPlan -> RunPlanStage -> InitializeColdGroup or
 * ProcessHotGroup and AssignMissSlots -> CopyStagedOutputs.
 * The 1024-thread PlanPrivateRow is called only for insufficient eviction slots.
 * Host integration and tiling are in sparse_kv_plan_config.h.
 */

#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
#include "sparse_kv_plan_config.h"

#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(type)
#endif

namespace {

// ==================== 1. 1024-thread fallback and shared definitions ====================

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

constexpr uint32_t WARP_TOTALS_OFFSET = 0U;
constexpr uint32_t WARP_BASES_OFFSET = PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_OFFSET = 2U * PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_TOTAL = 0U;
constexpr uint32_t CONTROL_RESET = 1U;

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
  int32_t blockSize;
  uint32_t localMemoryBytes;
  uint64_t jointHashCapacity;
  uint64_t scratchElements;
};

__simt_callee__ __aicore__ inline uint32_t HashToken(int32_t token, uint32_t hashCapacity) {
  uint32_t value = static_cast<uint32_t>(token);
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value & (hashCapacity - 1U);
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline uint32_t InsertTopkToken(ScratchPtr keys, ScratchPtr firstPositions,
                                                           int32_t token, int32_t position, uint32_t hashCapacity) {
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

__simt_callee__ __aicore__ inline void ClearMissTail(int64_t base, uint32_t assigned, uint32_t topk,
                                                     uint32_t thread, __gm__ int32_t* missTokens,
                                                     __gm__ int32_t* missSlots) {
  for (uint32_t pos = assigned + thread; pos < topk; pos += PLAN_THREADS) {
    missTokens[base + pos] = INVALID_TOKEN;
    missSlots[base + pos] = INVALID_TOKEN;
  }
}

struct PlanBuffers {
  __gm__ int64_t *reqIds, *lastReqIds;
  __gm__ int32_t *topkIndices, *stablePrefixLens, *visibleSeqLens, *tokenToReq;
  __gm__ int32_t *slotToToken, *lruSlots, *currentSlots, *missCount, *missTokens, *missSlots;
  __gm__ int32_t *rowModes, *residentRows, *residentBlockTable;
  uint32_t topk, capacity, hashCapacity, jointHashCapacity;
  uint64_t scratchElements;
  int32_t maxToken, blockSize;
  int64_t maxRequests;
};
constexpr int32_t ROW_PRIVATE = 0;
constexpr int32_t ROW_JOINT = 1;
constexpr int32_t ROW_FOLLOWER = 2;
struct RequestSource {
  int32_t request = -1;
  int32_t groupMaxToken = 0;
  uint32_t blockSize = 1U;
  uint32_t blockShift = UINT32_MAX;
  bool reqValid = false;
  __simt_callee__ __aicore__ inline uint32_t RowUpper(int32_t visible) const {
    if (!reqValid || visible <= 0 || groupMaxToken <= 0) {
      return 0U;
    }
    return static_cast<uint32_t>(visible < groupMaxToken ? visible : groupMaxToken);
  }
  __simt_callee__ __aicore__ inline bool IsValid(int32_t token, uint32_t rowUpper) const {
    return static_cast<uint32_t>(token) < rowUpper;
  }
  __simt_callee__ __aicore__ inline uint32_t Pages(uint32_t capacity) const {
    return blockShift != UINT32_MAX ? (capacity >> blockShift) : (capacity / blockSize);
  }
};
__simt_callee__ __aicore__ inline RequestSource GetRequestSource(const PlanBuffers& buffers, int32_t request) {
  RequestSource src;
  src.request = request;
  const uint32_t safeBlockSize = static_cast<uint32_t>(buffers.blockSize > 0 ? buffers.blockSize : 1);
  src.blockSize = safeBlockSize;
  if ((safeBlockSize & (safeBlockSize - 1U)) == 0U) {
    uint32_t shift = 0U;
    for (uint32_t v = safeBlockSize; v > 1U; v >>= 1U) {
      ++shift;
    }
    src.blockShift = shift;
  }
  src.reqValid = request >= 0 && static_cast<int64_t>(request) < buffers.maxRequests && buffers.blockSize > 0 &&
                 buffers.maxToken > 0;
  if (src.reqValid) {
    src.groupMaxToken = buffers.maxToken;
  }
  return src;
}
static_assert(PLAN_WARP_COUNT <= PLAN_WARP_SIZE && (PLAN_WARP_COUNT & (PLAN_WARP_COUNT - 1U)) == 0U,
              "ExclusiveScanRowTile requires power-of-two PLAN_WARP_COUNT <= PLAN_WARP_SIZE");
// Single-barrier ping-pong block exclusive scan:
// Alternates between WARP_TOTALS_OFFSET and WARP_BASES_OFFSET across consecutive tiles so the
// next tile's warp-total write never aliases the current tile's warp-total read, eliminating
// PublishWarpBases and cutting internal scan barriers from 2 to 1 per tile.
__simt_callee__ __aicore__ inline int32_t ExclusiveScanRowTile(__ubuf__ int32_t* workspace, int32_t value,
                                                               uint32_t thread, uint32_t& parity,
                                                               int32_t& outTotal) {
  __ubuf__ int32_t* warpTotals = workspace + (parity != 0U ? WARP_BASES_OFFSET : WARP_TOTALS_OFFSET);
  parity ^= 1U;
  const uint32_t lane = thread & (PLAN_WARP_SIZE - 1U);
  const uint32_t warp = thread / PLAN_WARP_SIZE;
  int32_t inclusive = value;
#pragma unroll
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
  const int32_t w = lane < PLAN_WARP_COUNT ? warpTotals[lane] : 0;
  int32_t warpInclusive = w;
#pragma unroll
  for (uint32_t offset = 1U; offset < PLAN_WARP_COUNT; offset <<= 1U) {
    const int32_t prior = asc_shfl_up(warpInclusive, offset, PLAN_WARP_SIZE);
    if (lane >= offset) {
      warpInclusive += prior;
    }
  }
  const int32_t warpBase = asc_shfl(warpInclusive - w, warp, PLAN_WARP_SIZE);
  outTotal = asc_shfl(warpInclusive, PLAN_WARP_COUNT - 1U, PLAN_WARP_SIZE);
  return warpBase + inclusive - value;
}
// Each tile's ExclusiveScanRowTile alternates warpTotals buffers, so only the final tile of a stage
// requires a post-scatter barrier before downstream stages read the scattered UB array.
__simt_callee__ __aicore__ inline void SyncAfterLastRowTile(uint32_t tile, uint32_t extent) {
  if (tile + PLAN_THREADS >= extent) {
    asc_syncthreads();
  }
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void PlanPrivateRow(const PlanBuffers& buffers, int64_t row, ScratchPtr scratch,
                                                      __ubuf__ int32_t* scanWorkspace, uint32_t thread) {
  const int64_t topkBase = row * static_cast<int64_t>(buffers.topk);
  const int64_t capacityBase = row * static_cast<int64_t>(buffers.capacity);
  const RequestSource src = GetRequestSource(buffers, buffers.tokenToReq[row]);
  const uint32_t rowUpper = src.RowUpper(buffers.visibleSeqLens[row]);
  const int64_t currentReqId = src.reqValid ? buffers.reqIds[src.request] : -1;
  const bool reset = buffers.rowModes[row] != ROW_PRIVATE || buffers.lastReqIds[row] != currentReqId;
  const uint32_t pages = src.Pages(buffers.capacity);
  const int32_t ownerPageBase = static_cast<int32_t>(row * static_cast<int64_t>(pages));
  if (thread == 0U) {
    buffers.residentRows[row] = static_cast<int32_t>(row);
  }
  for (uint32_t page = thread; page < pages; page += PLAN_THREADS) {
    buffers.residentBlockTable[row * static_cast<int64_t>(pages) + page] =
        ownerPageBase + static_cast<int32_t>(page);
  }
  ScratchPtr hashKeys = scratch;
  ScratchPtr hashFirstPos = hashKeys + buffers.hashCapacity;
  ScratchPtr hashOwner = hashFirstPos + buffers.hashCapacity;
  ScratchPtr evictableSlots = hashOwner + buffers.hashCapacity;
  ScratchPtr missPositions = evictableSlots + buffers.capacity;
  if (!reset) {
    for (uint32_t pos = thread; pos < buffers.topk; pos += PLAN_THREADS) {
      buffers.currentSlots[topkBase + pos] = INVALID_TOKEN;
    }
    for (uint32_t i = thread; i < buffers.hashCapacity; i += PLAN_THREADS) {
      hashKeys[i] = HASH_EMPTY;
      hashFirstPos[i] = HASH_POSITION_EMPTY;
      hashOwner[i] = INVALID_TOKEN;
    }
  }
  asc_syncthreads();
  if (thread == 0U) {
    buffers.rowModes[row] = ROW_PRIVATE;
  }
  uint32_t scanParity = 0U;
  if (reset) {
    int32_t count = 0;
    for (uint32_t tile = 0U; tile < buffers.topk; tile += PLAN_THREADS) {
      const uint32_t position = tile + thread;
      int32_t token = INVALID_TOKEN;
      int32_t flag = 0;
      if (position < buffers.topk) {
        token = buffers.topkIndices[topkBase + position];
        flag = src.IsValid(token, rowUpper) ? 1 : 0;
      }
      int32_t total = 0;
      const int32_t rank = ExclusiveScanRowTile(scanWorkspace, flag, thread, scanParity, total);
      if (position < buffers.topk) {
        const int32_t miss = count + rank;
        if (flag != 0 && miss < static_cast<int32_t>(buffers.capacity)) {
          buffers.slotToToken[capacityBase + miss] = token;
          buffers.currentSlots[topkBase + position] = miss;
          buffers.missTokens[topkBase + miss] = token;
          buffers.missSlots[topkBase + miss] = miss;
        } else {
          buffers.currentSlots[topkBase + position] = INVALID_TOKEN;
        }
      }
      count += total;
    }
    const int32_t assigned =
        count < static_cast<int32_t>(buffers.capacity) ? count : static_cast<int32_t>(buffers.capacity);
    for (uint32_t slot = static_cast<uint32_t>(assigned) + thread; slot < buffers.capacity; slot += PLAN_THREADS) {
      buffers.slotToToken[capacityBase + slot] = INVALID_TOKEN;
    }
    ClearMissTail(topkBase, static_cast<uint32_t>(assigned), buffers.topk, thread, buffers.missTokens,
                  buffers.missSlots);
    const uint32_t unused = buffers.capacity - static_cast<uint32_t>(assigned);
    for (uint32_t order = thread; order < buffers.capacity; order += PLAN_THREADS) {
      buffers.lruSlots[capacityBase + order] =
          static_cast<int32_t>(order < unused ? static_cast<uint32_t>(assigned) + order : order - unused);
    }
    if (thread == 0U) {
      buffers.lastReqIds[row] = currentReqId;
      buffers.missCount[row] = assigned;
    }
    asc_syncthreads();
    return;
  }
  for (uint32_t pos = thread; pos < buffers.topk; pos += PLAN_THREADS) {
    const int32_t token = buffers.topkIndices[topkBase + pos];
    if (src.IsValid(token, rowUpper)) {
      missPositions[pos] = static_cast<int32_t>(
          InsertTopkToken(hashKeys, hashFirstPos, token, static_cast<int32_t>(pos), buffers.hashCapacity));
    } else {
      missPositions[pos] = static_cast<int32_t>(HASH_BUCKET_INVALID);
    }
  }
  asc_syncthreads();
  int32_t stablePrefix = buffers.stablePrefixLens[row];
  if (stablePrefix < 0) {
    stablePrefix = 0;
  } else if (stablePrefix > buffers.maxToken) {
    stablePrefix = buffers.maxToken;
  }
  const uint32_t uStable = static_cast<uint32_t>(stablePrefix);
  const uint32_t uMaxToken = static_cast<uint32_t>(buffers.maxToken > 0 ? buffers.maxToken : 0);
  const bool packedOwner = buffers.capacity <= OWNER_PACK_CAPACITY;
  int32_t evictableCount = 0;
  int32_t hitCount = 0;
  for (uint32_t tile = 0U; tile < buffers.capacity; tile += PLAN_THREADS) {
    const uint32_t order = tile + thread;
    int32_t slot = INVALID_TOKEN;
    int32_t evictFlag = 0;
    int32_t hitFlag = 0;
    if (order < buffers.capacity) {
      slot = buffers.lruSlots[capacityBase + order];
      if (slot >= 0 && static_cast<uint32_t>(slot) < buffers.capacity) {
        const int32_t token = buffers.slotToToken[capacityBase + slot];
        uint32_t bucket = HASH_BUCKET_INVALID;
        if (static_cast<uint32_t>(token) < uStable) {
          bucket = LookupTopkToken(hashKeys, token, buffers.hashCapacity);
        } else if (static_cast<uint32_t>(token) < uMaxToken) {
          buffers.slotToToken[capacityBase + slot] = INVALID_TOKEN;
        }
        if (bucket != HASH_BUCKET_INVALID) {
          hitFlag = 1;
          const int32_t owner = packedOwner
                                    ? static_cast<int32_t>((order << OWNER_SLOT_BITS) | static_cast<uint32_t>(slot))
                                    : static_cast<int32_t>(order);
          asc_atomic_max(hashOwner + bucket, owner);
        } else {
          evictFlag = 1;
        }
      }
    }
    const int32_t packed = evictFlag + (hitFlag << COUNT_SHIFT);
    int32_t totals = 0;
    const int32_t ranks = ExclusiveScanRowTile(scanWorkspace, packed, thread, scanParity, totals);
    const int32_t evictRank = ranks & COUNT_MASK;
    const int32_t hitRank = ranks >> COUNT_SHIFT;
    if (evictFlag != 0) {
      evictableSlots[evictableCount + evictRank] = slot;
    }
    if (hitFlag != 0) {
      evictableSlots[buffers.capacity - 1U - static_cast<uint32_t>(hitCount + hitRank)] = slot;
    }
    SyncAfterLastRowTile(tile, buffers.capacity);
    evictableCount += totals & COUNT_MASK;
    hitCount += totals >> COUNT_SHIFT;
  }
  const bool packMissBucket = buffers.topk <= UINT16_MAX && buffers.hashCapacity <= UINT16_MAX;
  int32_t localMissCount = 0;
  for (uint32_t tile = 0U; tile < buffers.topk; tile += PLAN_THREADS) {
    const uint32_t position = tile + thread;
    int32_t missFlag = 0;
    uint32_t bucket = HASH_BUCKET_INVALID;
    if (position < buffers.topk) {
      bucket = static_cast<uint32_t>(missPositions[position]);
      if (bucket != HASH_BUCKET_INVALID) {
        const int32_t owner = hashOwner[bucket];
        if (owner >= 0 && hashFirstPos[bucket] == static_cast<int32_t>(position)) {
          buffers.currentSlots[topkBase + position] =
              packedOwner ? (owner & OWNER_SLOT_MASK) : buffers.lruSlots[capacityBase + owner];
        } else {
          missFlag = 1;
        }
      }
    }
    int32_t tileMissCount = 0;
    const int32_t rank = ExclusiveScanRowTile(scanWorkspace, missFlag, thread, scanParity, tileMissCount);
    if (missFlag != 0) {
      missPositions[localMissCount + rank] =
          packMissBucket ? static_cast<int32_t>((bucket << 16U) | position) : static_cast<int32_t>(position);
    }
    SyncAfterLastRowTile(tile, buffers.topk);
    localMissCount += tileMissCount;
  }
  const int32_t assignCount = localMissCount < evictableCount ? localMissCount : evictableCount;
  for (uint32_t miss = thread; miss < static_cast<uint32_t>(assignCount); miss += PLAN_THREADS) {
    const int32_t slot = evictableSlots[miss];
    const uint32_t packedMiss = static_cast<uint32_t>(missPositions[miss]);
    const int32_t position =
        packMissBucket ? static_cast<int32_t>(packedMiss & 0xFFFFU) : static_cast<int32_t>(packedMiss);
    const int32_t token = packMissBucket ? hashKeys[packedMiss >> 16U] : buffers.topkIndices[topkBase + position];
    buffers.slotToToken[capacityBase + slot] = token;
    buffers.currentSlots[topkBase + position] = slot;
    buffers.missTokens[topkBase + miss] = token;
    buffers.missSlots[topkBase + miss] = slot;
  }
  ClearMissTail(topkBase, static_cast<uint32_t>(assignCount), buffers.topk, thread, buffers.missTokens,
                buffers.missSlots);
  const int32_t unassignedCount = evictableCount - assignCount;
  const int32_t validLruCount = evictableCount + hitCount;
  for (uint32_t output = thread; output < static_cast<uint32_t>(validLruCount); output += PLAN_THREADS) {
    int32_t slot;
    if (output < static_cast<uint32_t>(unassignedCount)) {
      slot = evictableSlots[assignCount + output];
    } else if (output < static_cast<uint32_t>(evictableCount)) {
      slot = evictableSlots[output - static_cast<uint32_t>(unassignedCount)];
    } else {
      slot = evictableSlots[buffers.capacity - 1U - (output - static_cast<uint32_t>(evictableCount))];
    }
    buffers.lruSlots[capacityBase + output] = slot;
  }
  if (thread == 0U) {
    buffers.missCount[row] = assignCount;
  }
  asc_syncthreads();
}

// ==================== 2. Bitmap layout and input bounds ====================

constexpr uint64_t kMaxScratchWords = 204U * 1024U / 4U;
constexpr uint32_t kGroupControlWords = 128U;
constexpr uint32_t kPlanThreads = PLAN_THREADS / 2U;

template <typename P>
__simt_callee__ __aicore__ inline uint32_t LowerBoundToken(P p, uint32_t n, int32_t token) {
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    uint32_t mid = lo + ((hi - lo) >> 1U);
    if (p[mid] < token) {
      lo = mid + 1U;
    } else {
      hi = mid;
    }
  }
  return lo;
}
enum PlanStage : int {
  kPrepareQueries = 0,
  kPackMergeInputs = 1,
  kPlanPrivateRows = 4,
  kClassifyResident = 6,
  kCommitHotGroup = 9,
};

// Indices in the shared UB control region used by Scalar, SIMT, and SIMD.
enum GroupWord : uint32_t {
  kGroupStatus = 0,
  kUniqueCount = 1,
  kPreviousUniqueToken = 2,
  kRecordsPerSource = 3,
  kMergeRecordCount = 4,
  kPacketRecordCount = 5,
  kNeedsMerge = 6,
  kUsesInputPacket = 7,
  kActiveSourceCount = 28,
  kPacketProcessed = 29,
  kMergeOutputOffset = 30,
  kEvictableSlots = 43,
  kHitSlots = 44,
  kMissingTokens = 45,
  kDescriptorTailCount = 46,
  kDescriptorCount = 47,
  kValidBegin = 48,
  kValidEnd = 56,
  kSourceRow = 64,
  kSourceLength = 72,
  kSourceConsumed = 80,
  kReservedSourceTailLength = 88,
  kReservedSourceTailOffset = 96,
  kOutputsStaged = 120,
};

constexpr int32_t kGroupRejected = 0;
constexpr int32_t kGroupReady = 1;
constexpr int32_t kGroupCommitted = 2;
constexpr uint32_t kInvalidUid = 65535U;
constexpr uint32_t kSlotFlagsWordOffset = 12288U;
constexpr uint32_t kPackedSlotIndexMask = 0x3fffU;
constexpr uint32_t kResidentHitFlag = 0x4000U;
constexpr uint32_t kResidentInvalidFlag = 0x8000U;
constexpr uint32_t kPartitionInvalidFlag = 0x4000U;
constexpr uint32_t kMissOwnerFlag = 0x40000000U;

constexpr int32_t kInsufficientSlots = -2;
constexpr uint32_t kBitmapWords = 358400U / 32U;
static_assert(2U * kBitmapWords * sizeof(uint32_t) + 4096U * sizeof(uint16_t) <= 96U * 1024U);
static_assert(2U * kBitmapWords * sizeof(uint32_t) + 8192U * sizeof(uint16_t) <= 128U * 1024U);

// ==================== 3. Cold initialization ====================

// C reset producer: same UID/slot and protocol, stage LRU and descriptor tails for MTE3.
__simt_callee__ __aicore__ inline void InitializeColdGroup(const PlanBuffers& buffers, int64_t first, int64_t end,
                                                           __ubuf__ uint16_t* queryToUid,
                                                           __ubuf__ int32_t* uniqueTokens,
                                                           __ubuf__ int32_t* lruStaging,
                                                           __ubuf__ int32_t* groupControl, uint32_t thread) {
  constexpr uint32_t threadCount = kPlanThreads;
  const uint32_t uniqueCount = static_cast<uint32_t>(groupControl[kUniqueCount]),
                 rows = static_cast<uint32_t>(end - first);
  if (uniqueCount > buffers.capacity) {
    if (thread == 0U) {
      groupControl[kGroupStatus] = kGroupRejected;
    }
    asc_syncthreads();
    return;
  }
  const int64_t queryBase = first * static_cast<int64_t>(buffers.topk),
                slotBase = first * static_cast<int64_t>(buffers.capacity);
  const uint32_t unused = buffers.capacity - uniqueCount, queryCount = rows * buffers.topk;
  // Reset has no valid resident/hit. Assign uid directly to identity slot uid.
  // This fuses cache initialization and fill; no intermediate owner/flags/partition.
  for (uint32_t slot = thread; slot < buffers.capacity; slot += threadCount) {
    const int32_t token = slot < uniqueCount ? uniqueTokens[slot] : -1;
    buffers.slotToToken[slotBase + slot] = token;
    lruStaging[slot] = static_cast<int32_t>(slot < unused ? uniqueCount + slot : slot - unused);
    if (slot < uniqueCount) {
      buffers.missTokens[queryBase + slot] = token;
      buffers.missSlots[queryBase + slot] = static_cast<int32_t>(slot);
    }
  }
  for (uint32_t i = thread; i < queryCount; i += threadCount) {
    const uint32_t uid = queryToUid[i];
    buffers.currentSlots[queryBase + i] = uid != kInvalidUid ? static_cast<int32_t>(uid) : -1;
  }
  // Stage the unused descriptor tail for the bulk writeback.
  asc_syncthreads();  // finish every unique token consumer before reusing unique as -1 buffer
  for (uint32_t i = thread; i < queryCount - uniqueCount; i += threadCount) {
    uniqueTokens[i] = -1;
  }
  const int32_t requestIndex = buffers.tokenToReq[first];
  const RequestSource src = GetRequestSource(buffers, requestIndex);
  const uint32_t pages = src.Pages(buffers.capacity);
  for (uint32_t rowInGroup = 0; rowInGroup < rows; ++rowInGroup) {
    const int64_t row = first + rowInGroup;
    for (uint32_t page = thread; page < pages; page += threadCount) {
      buffers.residentBlockTable[row * pages + page] = static_cast<int32_t>(first * pages + page);
    }
    if (thread == 0U) {
      buffers.residentRows[row] = static_cast<int32_t>(first);
      const int64_t left = static_cast<int64_t>(uniqueCount) - static_cast<int64_t>(rowInGroup) * buffers.topk;
      buffers.missCount[row] = left <= 0             ? 0
                               : left < buffers.topk ? static_cast<int32_t>(left)
                                                     : static_cast<int32_t>(buffers.topk);
    }
    // Preserve the empirical thread-1 mode-publication workaround. No extra output.
    if (thread == 1U) {
      buffers.rowModes[row] = rows == 1U ? ROW_PRIVATE : (rowInGroup == 0U ? ROW_JOINT : ROW_FOLLOWER);
    }
  }
  if (thread == 0U) {
    buffers.lastReqIds[first] = buffers.reqIds[requestIndex];
    groupControl[kDescriptorTailCount] = queryCount - uniqueCount;
    groupControl[kDescriptorCount] = uniqueCount;
    groupControl[kOutputsStaged] = 1;
    groupControl[kGroupStatus] = kGroupCommitted;
  }
  asc_syncthreads();  // finish all writes before this AIV reuses scratch for another group
}
// SPDX-License-Identifier: Apache-2.0

// ==================== 4. Hot-cache classification and commit ====================

// Classification writes scratch. Commit runs after SIMD has checked slot availability.
template <int Phase>
__simt_callee__ __aicore__ inline void ProcessHotGroup(const PlanBuffers& buffers, int64_t first, int64_t end,
                                                       __ubuf__ int32_t* data, __ubuf__ int32_t* groupControl,
                                                       __ubuf__ int32_t* scan, __gm__ int32_t* workspace) {
  const uint32_t thread = threadIdx.x, capacity = buffers.capacity, topk = buffers.topk,
                 queryCount = (end - first) * topk, uniqueCount = groupControl[kUniqueCount];
  auto* queryToUid = reinterpret_cast<__ubuf__ uint16_t*>(data);
  auto* uniqueTokens = data + capacity / 2U;
  auto* uidToSlot = uniqueTokens + capacity;
  auto* partition = reinterpret_cast<__ubuf__ uint16_t*>(uidToSlot + capacity);
  // Reuse the first-UID storage after all reads of the bitmap index complete.
  auto* flags = reinterpret_cast<__ubuf__ uint16_t*>(uidToSlot + kSlotFlagsWordOffset);
  const int64_t queryBase = first * topk, slotBase = first * capacity;
  const uint32_t rows = end - first;
  const int32_t requestIndex = buffers.tokenToReq[first];
  const int64_t requestId = buffers.reqIds[requestIndex];
  const uint32_t evicts = Phase == kCommitHotGroup ? groupControl[kEvictableSlots] : 0;
  const uint32_t misses = Phase == kCommitHotGroup ? groupControl[kMissingTokens] : 0;
  if constexpr (Phase != kCommitHotGroup) {
    const bool reset =
        buffers.rowModes[first] != (rows == 1 ? ROW_PRIVATE : ROW_JOINT) || buffers.lastReqIds[first] != requestId;
    if (reset) {
      asc_syncthreads();
      InitializeColdGroup(buffers, first, end, queryToUid, uniqueTokens, uidToSlot, groupControl, thread);
      return;
    }
    int32_t stableEnd = buffers.stablePrefixLens[first];
    for (int64_t rowInGroup = first + 1; rowInGroup < end; ++rowInGroup) {
      if (buffers.stablePrefixLens[rowInGroup] < stableEnd) {
        stableEnd = buffers.stablePrefixLens[rowInGroup];
      }
    }
    if (stableEnd < 0) {
      stableEnd = 0;
    }
    if (stableEnd > buffers.maxToken) {
      stableEnd = buffers.maxToken;
    }
    // The bitmap index and slot lookup reuse the completed merge packet.
    auto* slotLookup = reinterpret_cast<__ubuf__ uint16_t*>(uidToSlot + 2U * kBitmapWords);
    {
      auto* masks = reinterpret_cast<__ubuf__ uint32_t*>(uidToSlot);
      auto* firstUid = reinterpret_cast<__ubuf__ uint32_t*>(uidToSlot + kBitmapWords);
      const uint32_t bitmapExtent = static_cast<uint32_t>(buffers.maxToken);
      const uint32_t words = (bitmapExtent + 31U) / 32U;
      for (uint32_t w = thread; w < words; w += kPlanThreads) {
        masks[w] = 0U;
      }
      asc_syncthreads();  // zero bitmap before the single-writer heads publish words
      for (uint32_t uid = thread; uid < uniqueCount; uid += kPlanThreads) {
        const uint32_t token = static_cast<uint32_t>(uniqueTokens[uid]), word = token >> 5U;
        if (uid == 0U || (static_cast<uint32_t>(uniqueTokens[uid - 1U]) >> 5U) != word) {
          uint32_t value = 0U;
          // Sorted unique tokens make each word a contiguous run of at most32 tokens.
          // One head owns BOTH32-bit writes; no atomic OR or halfword RMW required.
          for (uint32_t j = uid; j < uniqueCount && (static_cast<uint32_t>(uniqueTokens[j]) >> 5U) == word; ++j) {
            value |= uint32_t{1} << (static_cast<uint32_t>(uniqueTokens[j]) & 31U);
          }
          firstUid[word] = uid;
          masks[word] = value;
        }
      }
      asc_syncthreads();
      for (uint32_t slot = thread; slot < capacity; slot += kPlanThreads) {
        const int32_t token = buffers.slotToToken[slotBase + slot];
        uint32_t encoded = token >= stableEnd ? kResidentInvalidFlag : 0U;
        if (token >= 0 && token < stableEnd) {
          const uint32_t word = static_cast<uint32_t>(token) >> 5U,
                         bit = uint32_t{1} << (static_cast<uint32_t>(token) & 31U);
          const uint32_t mask = masks[word];
          if (mask & bit) {
            const uint32_t uid = firstUid[word] + __popc(mask & (bit - 1U));
            encoded = kResidentHitFlag | uid;
          }
        }
        slotLookup[slot] = static_cast<uint16_t>(encoded);
      }
      asc_syncthreads();  // index is dead before owner initialization reuses its first64KiB
    }
    for (uint32_t i = thread; i < uniqueCount; i += kPlanThreads) {
      uidToSlot[i] = -1;
    }
    asc_syncthreads();
    for (uint32_t tile = 0; tile < capacity; tile += kPlanThreads) {
      const uint32_t order = tile + thread;
      int32_t slot = order < capacity ? buffers.lruSlots[slotBase + order] : -1;
      const bool valid = slot >= 0 && static_cast<uint32_t>(slot) < capacity;

      if (order < capacity && !valid) {
        asc_atomic_min(groupControl + kGroupStatus, kGroupRejected);
      }
      bool hit = false, invalid = false;
      if (valid) {
        const uint32_t encoded = slotLookup[slot];
        invalid = (encoded & kResidentInvalidFlag) != 0;
        hit = (encoded & kResidentHitFlag) != 0;
        if (hit) {
          asc_atomic_max(uidToSlot + (encoded & kPackedSlotIndexMask), static_cast<int32_t>((order << 16) | slot));
        }
      }
      if (order < capacity) {
        flags[order] = valid ? static_cast<uint16_t>(slot | (hit ? kResidentHitFlag : 0U) |
                                                     (invalid ? kResidentInvalidFlag : 0U))
                             : uint16_t{65535};
      }
    }
    asc_syncthreads();
    if constexpr (Phase == kClassifyResident) {
      return;
    }
  }
  // All rejection checks above precede persistent state mutation.
  for (uint32_t uid = thread; uid < uniqueCount; uid += kPlanThreads) {
    const bool missing = (static_cast<uint32_t>(uidToSlot[uid]) & kMissOwnerFlag) != 0;
    const uint32_t rank = (static_cast<uint32_t>(uidToSlot[uid]) >> 16) & kPackedSlotIndexMask;
    if (missing) {
      int32_t slot = uidToSlot[uid] & 0xffff;
      buffers.slotToToken[slotBase + slot] = uniqueTokens[uid];
      buffers.missTokens[queryBase + rank] = uniqueTokens[uid];
      buffers.missSlots[queryBase + rank] = slot;
    }
  }
  // Read two query UIDs with one packed UB load.
  for (uint32_t pair = thread; pair < queryCount / 2; pair += kPlanThreads) {
    const uint32_t packed = reinterpret_cast<__ubuf__ uint32_t*>(data)[pair];
    const uint32_t lo = packed & kInvalidUid, hi = packed >> 16;
    buffers.currentSlots[queryBase + 2 * pair] = lo == kInvalidUid ? -1 : uidToSlot[lo] & 0xffff;
    buffers.currentSlots[queryBase + 2 * pair + 1] = hi == kInvalidUid ? -1 : uidToSlot[hi] & 0xffff;
  }
  // All readers of owner/unique have finished before output staging overwrites them.
  asc_syncthreads();
  for (uint32_t i = thread; i < queryCount - misses; i += kPlanThreads) {
    uniqueTokens[i] = -1;
  }
  const uint32_t unused = evicts - misses;
  for (uint32_t order = thread; order < capacity; order += kPlanThreads) {
    const uint32_t index = order < unused   ? misses + order
                           : order < evicts ? order - unused
                                            : capacity - 1 - (order - evicts);
    const uint32_t entry = partition[index], slot = entry & kPackedSlotIndexMask;

    if (index >= misses && (entry & kPartitionInvalidFlag)) {
      buffers.slotToToken[slotBase + slot] = -1;
    }
    uidToSlot[order] = static_cast<int32_t>(slot);
  }
  const RequestSource src = GetRequestSource(buffers, requestIndex);
  const uint32_t pages = src.Pages(capacity);
  for (uint32_t rowInGroup = 0; rowInGroup < rows; ++rowInGroup) {
    int64_t row = first + rowInGroup;
    for (uint32_t p = thread; p < pages; p += kPlanThreads) {
      buffers.residentBlockTable[row * pages + p] = first * pages + p;
    }
    if (thread == 0) {
      buffers.residentRows[row] = first;
      int64_t left = static_cast<int64_t>(misses) - rowInGroup * topk;
      buffers.missCount[row] = left <= 0 ? 0 : left < topk ? left : topk;
    }
    if (thread == 1) {
      buffers.rowModes[row] = rows == 1 ? ROW_PRIVATE : (rowInGroup == 0 ? ROW_JOINT : ROW_FOLLOWER);
    }
  }
  if (thread == 0) {
    groupControl[kDescriptorTailCount] = queryCount - misses;
    groupControl[kDescriptorCount] = misses;
    groupControl[kOutputsStaged] = 1;
    buffers.lastReqIds[first] = requestId;
    groupControl[kGroupStatus] = kGroupCommitted;
  }
  asc_syncthreads();
}

// ==================== 5. Input preparation and SIMT entry ====================

template <int Phase, typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessPlanStage(const PlanBuffers& buffers, int64_t first, int64_t end,
                                                        ScratchPtr data, __ubuf__ int32_t* groupControl,
                                                        __ubuf__ int32_t* packet, __ubuf__ int32_t* scan,
                                                        __gm__ int32_t* mapWorkspace) {
  const uint32_t topk = buffers.topk;
  const uint32_t capacity = buffers.capacity;
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x), rows = static_cast<uint32_t>(end - first);
  const uint32_t mapWords = capacity / 2U;
  ScratchPtr uniqueTokens = data + mapWords;
  auto* queryToUid = reinterpret_cast<__ubuf__ uint16_t*>(groupControl + kGroupControlWords);
  const uint32_t queryCount = rows * topk;
  const int64_t queryBase = first * static_cast<int64_t>(topk), slotBase = first * static_cast<int64_t>(capacity);
  if constexpr (Phase == kPrepareQueries) {
    if (thread == 0U) {
      groupControl[kOutputsStaged] = 0;
      groupControl[kGroupStatus] = kGroupReady;
      groupControl[kUniqueCount] = 0;
      groupControl[kPreviousUniqueToken] = -1;
      groupControl[kMergeRecordCount] = 0;
      groupControl[kNeedsMerge] = 0;
      groupControl[kUsesInputPacket] = 0;
      groupControl[kActiveSourceCount] = 0;
      groupControl[kPacketProcessed] = 0;
      uint32_t recordsPerSource = (3U * capacity > 64U ? 3U * capacity : 64U) / 16U;
      if (recordsPerSource > 768U) {
        recordsPerSource = 768U;
      }
      groupControl[kRecordsPerSource] = static_cast<int32_t>(recordsPerSource & ~3U);
      uint64_t words = 16ULL * static_cast<uint32_t>(groupControl[kRecordsPerSource]);
      if (buffers.scratchElements <= kMaxScratchWords) {
        words = (rows == 4U && topk == 2048U && capacity == 8192U) ? 32768U : 24576U;
      }
      groupControl[kMergeOutputOffset] =
          static_cast<int32_t>((words / 2U) & ~uint64_t{7U});  // fixed input/output split
      const int32_t requestIndex = buffers.tokenToReq[first];
      uint32_t bits = 1;
      while (bits < 31U && (1ULL << bits) < queryCount) {
        ++bits;
      }
      if (rows < 1U || rows > 8U || (rows & (rows - 1U)) != 0U || capacity > 32768U || queryCount > capacity ||
          queryCount > kInvalidUid || buffers.maxToken <= 0 || buffers.maxToken > 16777216 ||
          (static_cast<uint64_t>(buffers.maxToken - 1) << bits) > UINT32_MAX || requestIndex < 0 ||
          requestIndex >= buffers.maxRequests) {
        groupControl[kGroupStatus] = kGroupRejected;
      }
      // The launch grid addresses fixed-width, contiguous request groups.
      for (int64_t row = first + 1; row < end; ++row) {
        if (buffers.tokenToReq[row] != requestIndex) {
          groupControl[kGroupStatus] = kGroupRejected;
        }
      }
      for (uint32_t r = 0; r < 8U; ++r) {
        groupControl[kValidBegin + r] = 0;
        groupControl[kValidEnd + r] = 0;
      }
      if (groupControl[kGroupStatus]) {
        for (uint32_t r = 0; r < rows; ++r) {
          const int64_t rowQueryBase = (first + r) * static_cast<int64_t>(topk);
          int32_t upper = buffers.visibleSeqLens[first + r];
          if (upper < 0) {
            upper = 0;
          }
          if (upper > buffers.maxToken) {
            upper = buffers.maxToken;
          }
          if (buffers.topkIndices[rowQueryBase + topk - 1U] < 0) {
            // Find the first trailing padding entry.
            uint32_t lo = 0, hi = topk;
            while (lo < hi) {
              const uint32_t mid = lo + (hi - lo) / 2U;
              if (buffers.topkIndices[rowQueryBase + mid] >= 0) {
                lo = mid + 1;
              } else {
                hi = mid;
              }
            }
            groupControl[kValidEnd + r] =
                static_cast<int32_t>(LowerBoundToken(buffers.topkIndices + rowQueryBase, lo, upper));
          } else if (topk > 0U && buffers.topkIndices[rowQueryBase] >= 0 &&
                     buffers.topkIndices[rowQueryBase + topk - 1U] < upper) {
            groupControl[kValidBegin + r] = 0;
            groupControl[kValidEnd + r] = static_cast<int32_t>(topk);
          } else {
            groupControl[kValidBegin + r] =
                static_cast<int32_t>(LowerBoundToken(buffers.topkIndices + rowQueryBase, topk, 0));
            groupControl[kValidEnd + r] =
                static_cast<int32_t>(LowerBoundToken(buffers.topkIndices + rowQueryBase, topk, upper));
          }
          if (groupControl[kValidBegin + r] < groupControl[kValidEnd + r]) {
            groupControl[kNeedsMerge] = 1;
          }
        }
      }
      if (groupControl[kGroupStatus] && rows == 1U) {
        groupControl[kUniqueCount] = groupControl[kValidEnd] - groupControl[kValidBegin];
        groupControl[kNeedsMerge] = 0;
      }
    }
    asc_syncthreads();
    // Full multi-row packets overwrite every query map entry below.
    const bool full = rows > 1U && groupControl[kGroupStatus] && groupControl[kValidBegin] == 0 &&
                      groupControl[kValidEnd] == topk && groupControl[kValidBegin + 1] == 0 &&
                      groupControl[kValidEnd + 1] == topk &&
                      (rows == 2U || (groupControl[kValidBegin + 2] == 0 && groupControl[kValidEnd + 2] == topk &&
                                      groupControl[kValidBegin + 3] == 0 && groupControl[kValidEnd + 3] == topk));
    if (full) {
      return;
    }
    if (groupControl[kGroupStatus]) {
      for (uint32_t queryIndex = thread; queryIndex < queryCount; queryIndex += kPlanThreads) {
        if (rows == 1U) {
          const uint32_t validBegin = static_cast<uint32_t>(groupControl[kValidBegin]);
          const bool valid =
              queryIndex >= validBegin && queryIndex < static_cast<uint32_t>(groupControl[kValidEnd]);
          const int32_t token = buffers.topkIndices[queryBase + queryIndex];
          // Check only the valid interval so repeated padding stays legal.
          if (valid && queryIndex > validBegin && token <= buffers.topkIndices[queryBase + queryIndex - 1U]) {
            asc_atomic_min(groupControl + kGroupStatus, kGroupRejected);
          }
          queryToUid[queryIndex] = valid ? static_cast<uint16_t>(queryIndex - validBegin) : uint16_t{65535};
          if (valid) {
            uniqueTokens[queryIndex - validBegin] = token;
          }
        } else {
          queryToUid[queryIndex] = kInvalidUid;
        }
      }
    }
    asc_syncthreads();
  } else if constexpr (Phase == kPackMergeInputs) {
    const uint32_t defaultRecordsPerSource = groupControl[kRecordsPerSource];
    if (thread == 0U) {
      // Initialize the reserved packet state in the existing UB layout.
      for (uint32_t row = 0; row < 8U; ++row) {
        groupControl[88 + row] = 0;
        groupControl[96 + row] = 0;
      }
      uint32_t activeSourceCount = 0;
      for (uint32_t r = 0; r < rows; ++r) {
        if (groupControl[kValidBegin + r] < groupControl[kValidEnd + r]) {
          groupControl[kSourceRow + activeSourceCount++] = r;
        }
      }
      uint32_t recordsPerSource = defaultRecordsPerSource;
      if (activeSourceCount) {
        recordsPerSource =
            (static_cast<uint32_t>(groupControl[kMergeOutputOffset]) / (2U * activeSourceCount)) & ~3U;
        if (recordsPerSource > 4092U) {
          recordsPerSource = 4092U;
        }
        if (rows > 4U && recordsPerSource > 512U) {
          recordsPerSource = 512U;
        }
      }
      if (rows > 4U) {
        groupControl[kMergeOutputOffset] =
            32U * recordsPerSource;  // input8, intermediate8, final8 records per chunk
      }
      groupControl[kRecordsPerSource] = recordsPerSource;
      groupControl[kActiveSourceCount] = activeSourceCount;
      groupControl[kUsesInputPacket] = 0;
      uint32_t total = 0;
      for (uint32_t k = 0; k < activeSourceCount; ++k) {
        uint32_t r = groupControl[kSourceRow + k], n = groupControl[kValidEnd + r] - groupControl[kValidBegin + r];
        if (n > recordsPerSource) {
          n = recordsPerSource;
        }
        groupControl[kSourceLength + k] = n;
        groupControl[kSourceConsumed + k] = 0;
        total += n;
      }
      groupControl[kMergeRecordCount] = total;
      groupControl[kPacketRecordCount] = total;
    }
    asc_syncthreads();
    const uint32_t recordsPerSource = groupControl[kRecordsPerSource],
                   activeSourceCount = groupControl[kActiveSourceCount];
    auto* score = reinterpret_cast<__ubuf__ float*>(packet);
    for (uint32_t i = thread; i < activeSourceCount * recordsPerSource; i += kPlanThreads) {
      uint32_t k = i / recordsPerSource, j = i % recordsPerSource;
      if (j < static_cast<uint32_t>(groupControl[kSourceLength + k])) {
        uint32_t r = groupControl[kSourceRow + k], pos = static_cast<uint32_t>(groupControl[kValidBegin + r]) + j;
        score[2U * i] = -static_cast<float>(buffers.topkIndices[queryBase + r * topk + pos]);
        packet[2U * i + 1U] = r * topk + pos;
      }
    }
    asc_syncthreads();
  } else if constexpr (Phase == kClassifyResident || Phase == kCommitHotGroup) {
    ProcessHotGroup<Phase>(buffers, first, end, groupControl + kGroupControlWords, groupControl, scan,
                           mapWorkspace);
  }
}
template <int Phase>
__simt_vf__ __aicore__
LAUNCH_BOUND((Phase == kPlanPrivateRows ? PLAN_THREADS : kPlanThreads)) inline void RunPlanStage(
    __ubuf__ int32_t* scanWorkspace, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds,
    __gm__ int32_t* topkIndices, __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens,
    __gm__ int32_t* tokenToReq, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    __gm__ int32_t* compactWorkspace, uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart,
    int64_t rowStride, int64_t numReqs, int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests,
    int32_t blockSize, __gm__ int32_t* rowModes, __gm__ int32_t* residentRows, __gm__ int32_t* residentBlockTable,
    uint64_t jointHashCapacity, uint64_t scratchElements) {
  const PlanBuffers args{reqIds,
                         lastReqIds,
                         topkIndices,
                         stablePrefixLens,
                         visibleSeqLens,
                         tokenToReq,
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
                         scratchElements,
                         static_cast<int32_t>(maxToken),
                         blockSize,
                         maxRequests};
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  if constexpr (Phase == kPlanPrivateRows) {
    if (jointHashCapacity == 0U) {
      for (int64_t row = rowStart; row < numReqs; row += rowStride) {
        if (scratchElements <= kMaxScratchWords) {
          PlanPrivateRow(args, row, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, thread);
        } else {
          PlanPrivateRow(args, row, compactWorkspace + static_cast<uint64_t>(row) * scratchElements, scanWorkspace,
                         thread);
        }
      }
    } else {
      for (int64_t row = rowStart; row < rowStride; ++row) {
        if (scratchElements <= kMaxScratchWords) {
          PlanPrivateRow(args, row, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, thread);
        } else {
          PlanPrivateRow(args, row, compactWorkspace + static_cast<uint64_t>(row) * scratchElements, scanWorkspace,
                         thread);
        }
      }
    }
  } else {
    __ubuf__ int32_t* groupControl = scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS;
    if (scratchElements <= kMaxScratchWords) {
      __ubuf__ int32_t* data = groupControl + kGroupControlWords;
      const uint32_t words = args.capacity / 2U;
      auto* packet = data + ((words + args.capacity + 7U) & ~7U);
      ProcessPlanStage<Phase>(args, rowStart, rowStride, data, groupControl, packet, scanWorkspace,
                              compactWorkspace);
    } else {
      __gm__ int32_t* data =
          compactWorkspace + static_cast<uint64_t>(rowStart) * scratchElements + kGroupControlWords;
      ProcessPlanStage<Phase>(args, rowStart, rowStride, data, groupControl, groupControl + kGroupControlWords,
                              scanWorkspace, compactWorkspace);
    }
  }
}

// ==================== 6. SIMD deduplication and query mapping ====================

__simd_vf__ inline void DeduplicateMergePacket(__ubuf__ int32_t* records, __ubuf__ int32_t* data,
                                               __ubuf__ int32_t* groupControl, uint32_t recordCount,
                                               uint32_t queryCount, uint32_t capacity, uint32_t initialCount,
                                               int32_t initialLast, uint32_t more, uint32_t mapWords) {
  using namespace AscendC;
  Reg::RegTensor<uint32_t> lane, zero, one, scoreIndices, queryIndices, previousIndex, lastIndex;
  Reg::RegTensor<uint32_t> flags, prefix, uniqueCount, uid, queryIndex, index, safeIndex, writeFlag, status, value;
  Reg::RegTensor<float> score, previous, lastScore, shiftScore, positive;
  Reg::RegTensor<int32_t> token, lastToken;
  Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
  Reg::MaskReg first, newToken, valid, fits, writeHead, chosen;
  // Unsqueeze(ALL) is exactly [0,1,...,63] for b32: no lane table in UB.
  Reg::Unsqueeze<uint32_t>(lane, full);
  Reg::Duplicate(zero, uint32_t{0});
  Reg::Duplicate(one, uint32_t{1});
  Reg::Duplicate(uniqueCount, initialCount);
  Reg::Duplicate(lastToken, initialLast);
  Reg::Duplicate(lastScore, -static_cast<float>(initialLast));
  Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
  Reg::Adds(previousIndex, lane, uint32_t(-1), full);
  Reg::Select(previousIndex, zero, previousIndex, first);  // no out-of-range register Gather
  static constexpr Reg::CastTrait toInt = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
  for (uint32_t base = 0; base < recordCount; base += 64U) {
    uint32_t remaining = recordCount - base;
    valid = Reg::UpdateMask<uint32_t>(remaining);
    const uint32_t count = recordCount - base < 64U ? recordCount - base : 64U;
    Reg::Adds(scoreIndices, lane, base, full);
    Reg::ShiftLefts(scoreIndices, scoreIndices, int16_t{1}, full);
    Reg::Adds(queryIndices, scoreIndices, uint32_t{1}, full);
    Reg::Duplicate(score, 0.0f);
    Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float*>(records), scoreIndices, valid);
    Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
    Reg::Select(previous, lastScore, shiftScore, first);
    Reg::Compare<float, CMPMODE::NE>(newToken, score, previous, valid);
    Reg::Select(flags, one, zero, newToken);
    Reg::Select(flags, flags, zero, valid);
    // Rebuild all mask bits: no assumption about Compare's inactive lanes.
    Reg::Compares<uint32_t, CMPMODE::NE>(newToken, flags, uint32_t{0}, full);
    Reg::Unsqueeze<uint32_t>(prefix, newToken);  // exclusive head count; last head not included here
    Reg::Add(prefix, prefix, flags, full);       // include every head, including lane63
    Reg::Add(prefix, prefix, uniqueCount, full);
    Reg::Adds(uid, prefix, uint32_t(-1), full);
    Reg::Neg(positive, score, full);
    Reg::Cast<int32_t, float, toInt>(token, positive, valid);  // exact integers in [0,2^24)
    Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
    Reg::Select(writeFlag, flags, zero, fits);
    Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
    Reg::Adds(index, uid, mapWords, full);
    Reg::Adds(safeIndex, lane, mapWords + capacity, full);
    Reg::Select(index, index, safeIndex, writeHead);  // even masked indices are unique
    Reg::Scatter<int32_t, uint32_t>(data, token, index, writeHead);
    Reg::Duplicate(queryIndex, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(queryIndex, reinterpret_cast<__ubuf__ uint32_t*>(records),
                                              queryIndices, valid);
    Reg::Adds(safeIndex, lane, queryCount, full);
    Reg::Select(queryIndex, queryIndex, safeIndex, valid);
    // B32 mask selects even B16 lanes. Each low half stores one UID;
    // inactive odd-lane indices are distinct and inside allocated scratch.
    Reg::Adds(safeIndex, lane, queryCount + 64U, full);
    Reg::ShiftLefts(index, safeIndex, int16_t{16}, full);
    Reg::Or(index, index, queryIndex, full);
    Reg::Scatter<uint16_t, uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data),
                                     reinterpret_cast<Reg::RegTensor<uint16_t>&>(uid),
                                     reinterpret_cast<Reg::RegTensor<uint16_t>&>(index), valid);
    Reg::Duplicate(lastIndex, uint32_t{63});
    Reg::Gather<uint32_t, uint32_t>(uniqueCount, prefix, lastIndex);
    Reg::Duplicate(lastIndex, count - 1U);
    Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
    Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
  }
  // All progress Scatter indices are [0,63], unique and within the original ctx64.
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{kUniqueCount}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(groupControl), uniqueCount, lane, chosen);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{kPreviousUniqueToken}, full);
  Reg::Scatter<int32_t, uint32_t>(groupControl, lastToken, lane, chosen);
  Reg::Compares<uint32_t, CMPMODE::GE>(fits, uniqueCount, capacity + 1U, full);
  Reg::Select(status, zero, one, fits);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{kGroupStatus}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(groupControl), status, lane, chosen);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{kPacketProcessed}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(groupControl), one, lane, chosen);
  Reg::Duplicate(value, more);
  Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{kNeedsMerge}, full);
  Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(groupControl), value, lane, chosen);
}

// ==================== 7. SIMD compaction and MTE3 writeback ====================

// Packed halfword stores preserve the LRU order on each side of the partition.
__simd_callee__ inline void WriteGroupWord(__ubuf__ int32_t* groupControl, AscendC::Reg::RegTensor<uint32_t>& value,
                                           AscendC::Reg::RegTensor<uint32_t>& lane, uint32_t at,
                                           AscendC::Reg::MaskReg& full) {
  AscendC::Reg::MaskReg chosen;
  AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::EQ>(chosen, lane, at, full);
  AscendC::Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(groupControl), value, lane,
                                            chosen);
}
__simd_vf__ inline void AssignMissSlots(__ubuf__ int32_t* data, __ubuf__ int32_t* groupControl, uint32_t capacity,
                                        uint32_t uniqueCount) {
  using namespace AscendC;
  const uint32_t ownerOffset = capacity / 2U + capacity, partitionOffset = ownerOffset + capacity;
  const uint32_t flagsOffset = ownerOffset + kSlotFlagsWordOffset;
  Reg::RegTensor<uint32_t> lane, zero, one, idx, oddBit, packed, high, slotFlags, mask, slot, invalid;
  Reg::RegTensor<uint32_t> hitFlags, evictFlags, hitPrefix, ep, hitCount, evictableCount, hitRank, evictionRank,
      last, entry, safe, tmp, missCount, rank, inc, owner, index16, status;
  Reg::RegTensor<int32_t> signedOwner;
  Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>(), odd, valid, hit, evict, selected, fits;
  Reg::Unsqueeze<uint32_t>(lane, full);
  Reg::Duplicate(zero, uint32_t{0});
  Reg::Duplicate(one, uint32_t{1});
  Reg::Duplicate(hitCount, uint32_t{0});
  Reg::Duplicate(evictableCount, uint32_t{0});
  Reg::Duplicate(last, uint32_t{63});
  Reg::Duplicate(mask, uint32_t{1});
  Reg::And(oddBit, lane, mask, full);
  Reg::Compares<uint32_t, CMPMODE::NE>(odd, oddBit, uint32_t{0}, full);
  for (uint32_t base = 0; base < capacity; base += 64U) {
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_UNPACK_B16>(
        slotFlags, reinterpret_cast<__ubuf__ uint32_t*>(data + flagsOffset) + (base / 2U));
    Reg::Compares<uint32_t, CMPMODE::NE>(valid, slotFlags, uint32_t{kInvalidUid}, full);
    Reg::Duplicate(mask, uint32_t{kResidentHitFlag});
    Reg::And(tmp, slotFlags, mask, full);
    Reg::Compares<uint32_t, CMPMODE::NE>(hit, tmp, uint32_t{0}, valid);
    Reg::Select(hitFlags, one, zero, hit);
    Reg::Select(evictFlags, one, zero, valid);
    Reg::Sub(evictFlags, evictFlags, hitFlags, full);
    Reg::Compares<uint32_t, CMPMODE::NE>(evict, evictFlags, uint32_t{0}, full);
    Reg::Unsqueeze<uint32_t>(hitPrefix, hit);
    Reg::Add(hitRank, hitPrefix, hitCount, full);

    Reg::Adds(evictionRank, lane, base, full);
    Reg::Sub(evictionRank, evictionRank, hitRank, full);
    Reg::Duplicate(tmp, capacity - 1U);
    Reg::Sub(idx, tmp, hitRank, full);
    Reg::Select(idx, idx, evictionRank, hit);
    Reg::Duplicate(mask, uint32_t{kPackedSlotIndexMask});
    Reg::And(entry, slotFlags, mask, full);
    Reg::Duplicate(mask, uint32_t{kResidentInvalidFlag});
    Reg::And(invalid, slotFlags, mask, full);
    Reg::ShiftRights(invalid, invalid, int16_t{1}, full);
    Reg::Or(entry, entry, invalid, full);
    // Only low halfwords selected by the B32 mask; inactive high-half indices safe.
    Reg::Adds(idx, idx, 2U * capacity, full);
    Reg::Select(idx, idx, lane, valid);
    Reg::Adds(safe, lane, uint32_t{64}, full);
    Reg::ShiftLefts(index16, safe, int16_t{16}, full);
    Reg::Or(index16, index16, idx, full);
    Reg::Scatter<uint16_t, uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data + ownerOffset),
                                     reinterpret_cast<Reg::RegTensor<uint16_t>&>(entry),
                                     reinterpret_cast<Reg::RegTensor<uint16_t>&>(index16), valid);
    Reg::Add(hitRank, hitRank, hitFlags, full);
    Reg::Gather<uint32_t, uint32_t>(hitCount, hitRank, last);
    Reg::Duplicate(tmp, base + 64U);
    Reg::Sub(evictableCount, tmp, hitCount, full);
  }
  Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
  Reg::Duplicate(missCount, uint32_t{0});
  for (uint32_t base = 0; base < uniqueCount; base += 64U) {
    uint32_t remaining = uniqueCount - base;
    valid = Reg::UpdateMask<uint32_t>(remaining);

    Reg::LoadAlign<int32_t>(signedOwner, data + ownerOffset + base);
    Reg::Compares<int32_t, CMPMODE::LT>(selected, signedOwner, int32_t{0}, valid);
    Reg::Select(inc, one, zero, selected);
    Reg::Unsqueeze<uint32_t>(rank, selected);
    Reg::Add(rank, rank, missCount, full);
    Reg::Compare<uint32_t, CMPMODE::LT>(fits, rank, evictableCount, selected);
    Reg::ShiftRights(idx, rank, int16_t{1}, full);
    Reg::Adds(idx, idx, partitionOffset, full);
    Reg::Duplicate(tmp, partitionOffset);
    Reg::Select(idx, idx, tmp, fits);
    Reg::Duplicate(packed, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(packed, reinterpret_cast<__ubuf__ uint32_t*>(data), idx, fits);
    Reg::Duplicate(mask, uint32_t{1});
    Reg::And(tmp, rank, mask, full);
    Reg::Compares<uint32_t, CMPMODE::NE>(odd, tmp, uint32_t{0}, full);
    Reg::ShiftRights(high, packed, int16_t{16}, full);
    Reg::Select(slot, high, packed, odd);
    Reg::Duplicate(mask, uint32_t{kPackedSlotIndexMask});
    Reg::And(slot, slot, mask, full);
    Reg::ShiftLefts(owner, rank, int16_t{16}, full);
    Reg::Or(owner, owner, slot, full);
    Reg::Duplicate(mask, uint32_t{kMissOwnerFlag});
    Reg::Or(owner, owner, mask, full);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM_B32>(
        reinterpret_cast<__ubuf__ uint32_t*>(data) + ownerOffset + base, owner, fits);
    Reg::Add(tmp, rank, inc, full);
    Reg::Gather<uint32_t, uint32_t>(missCount, tmp, last);
  }
  Reg::Add(tmp, evictableCount, hitCount, full);
  Reg::Compares<uint32_t, CMPMODE::EQ>(valid, tmp, capacity, full);
  Reg::Adds(tmp, evictableCount, uint32_t{1}, full);
  Reg::Compare<uint32_t, CMPMODE::LT>(selected, missCount, tmp, valid);
  Reg::Duplicate(status, static_cast<uint32_t>(kInsufficientSlots));
  Reg::Select(status, status, zero, valid);
  Reg::Select(status, one, status, selected);
  WriteGroupWord(groupControl, evictableCount, lane, kEvictableSlots, full);
  WriteGroupWord(groupControl, hitCount, lane, kHitSlots, full);
  WriteGroupWord(groupControl, missCount, lane, kMissingTokens, full);
  WriteGroupWord(groupControl, status, lane, kGroupStatus, full);
}

// SPDX-License-Identifier: Apache-2.0

// MTE3 consumer of the LRU and descriptor tails staged in UB by SIMT.
__aicore__ inline void CopyStagedOutputs(AscendC::TPipe& pipe, AscendC::LocalTensor<int32_t> local,
                                         __ubuf__ int32_t* groupControl, int64_t first, uint32_t topk,
                                         uint32_t capacity, GM_ADDR lruSlots, GM_ADDR missTokens,
                                         GM_ADDR missSlots) {
  const uint32_t descriptorTailCount = static_cast<uint32_t>(groupControl[kDescriptorTailCount]),
                 misses = static_cast<uint32_t>(groupControl[kDescriptorCount]);
  const uint32_t uniqueOffset = PLAN_SCAN_STORAGE_ELEMENTS + kGroupControlWords + capacity / 2U;
  const uint32_t ownerOffset = uniqueOffset + capacity;
  AscendC::GlobalTensor<int32_t> lru, missTokenOutput, missSlotOutput;
  lru.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lruSlots) + first * capacity, capacity);
  const int64_t tail = first * static_cast<int64_t>(topk) + misses;
  missTokenOutput.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missTokens) + tail, descriptorTailCount);
  missSlotOutput.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missSlots) + tail, descriptorTailCount);
  // Explicit SIMT/V -> MTE3 visibility; V_S only publishes the control values.
  auto ready = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ready);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ready);
  pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(ready);
  AscendC::DataCopyExtParams lruCopy(1, capacity * 4U, 0, 0, 0);
  AscendC::DataCopyPad(lru, local[ownerOffset], lruCopy);
  if (descriptorTailCount) {
    AscendC::DataCopyExtParams tailCopy(1, descriptorTailCount * 4U, 0, 0, 0);
    AscendC::DataCopyPad(missTokenOutput, local[uniqueOffset], tailCopy);
    AscendC::DataCopyPad(missSlotOutput, local[uniqueOffset], tailCopy);
  }
  // Both sources stay intact until ALL output copies finish. Includes the group
  // tail and kernel tail; no next VF/group may reuse these UB regions early.
  auto done = pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(done);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(done);
  pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(done);
}

// ==================== 8. Controller, hardware merge and fallback ====================

__aicore__ inline void WaitForVector(AscendC::TPipe& pipe) {
  auto id = pipe.AllocEventID<AscendC::HardEvent::V_S>();
  AscendC::SetFlag<AscendC::HardEvent::V_S>(id);
  AscendC::WaitFlag<AscendC::HardEvent::V_S>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::V_S>(id);
}
__aicore__ inline void PublishScalarToVector(AscendC::TPipe& pipe) {
  auto id = pipe.AllocEventID<AscendC::HardEvent::S_V>();
  AscendC::SetFlag<AscendC::HardEvent::S_V>(id);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::S_V>(id);
}

template <bool AllowJoint>
__aicore__ inline void RunOrderedBitmapPlan(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                            GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR activeRows,
                                            GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots,
                                            GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens,
                                            GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
                                            GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling) {
  JointDirectTilingData planTiling;
  const auto* gm = reinterpret_cast<const __gm__ JointDirectTilingData*>(tiling);
  planTiling.hashCapacity = gm->hashCapacity;
  planTiling.workspaceRowElements = gm->workspaceRowElements;
  planTiling.maxRows = gm->maxRows;
  planTiling.topk = gm->topk;
  planTiling.capacity = gm->capacity;
  planTiling.maxToken = gm->maxToken;
  planTiling.maxRequests = gm->maxRequests;
  planTiling.blockSize = gm->blockSize;
  planTiling.localMemoryBytes = gm->localMemoryBytes;
  planTiling.jointHashCapacity = gm->jointHashCapacity;
  planTiling.scratchElements = gm->scratchElements;
  const int32_t requestedRows = *reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t rows = requestedRows <= 0                   ? 0
                       : requestedRows < planTiling.maxRows ? requestedRows
                                                            : planTiling.maxRows;
  int64_t blockIndex = AscendC::GetBlockIdx(), blockCount = AscendC::GetBlockNum();
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
  pipe.InitBuffer(storage, planTiling.localMemoryBytes);
  auto local = storage.Get<int32_t>();
  auto* scan = reinterpret_cast<__ubuf__ int32_t*>(local.GetPhyAddr());
  auto* groupControl = scan + PLAN_SCAN_STORAGE_ELEMENTS;
#define RUN_PLAN_STAGE(PHASE)                                                                              \
  asc_vf_call<RunPlanStage<PHASE>>(                                                                        \
      dim3((PHASE) == kPlanPrivateRows ? PLAN_THREADS : kPlanThreads), scan,                               \
      reinterpret_cast<__gm__ int64_t*>(reqIds), reinterpret_cast<__gm__ int64_t*>(lastReqIds),            \
      reinterpret_cast<__gm__ int32_t*>(topkIndices), reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), \
      reinterpret_cast<__gm__ int32_t*>(visibleSeqLens), reinterpret_cast<__gm__ int32_t*>(tokenToReq),    \
      reinterpret_cast<__gm__ int32_t*>(slotToToken), reinterpret_cast<__gm__ int32_t*>(lruSlots),         \
      reinterpret_cast<__gm__ int32_t*>(currentSlots), reinterpret_cast<__gm__ int32_t*>(missCount),       \
      reinterpret_cast<__gm__ int32_t*>(missTokens), reinterpret_cast<__gm__ int32_t*>(missSlots),         \
      reinterpret_cast<__gm__ int32_t*>(compactWorkspace), planTiling.hashCapacity,                        \
      planTiling.workspaceRowElements, first, end, rows, planTiling.topk, planTiling.capacity,             \
      planTiling.maxToken, planTiling.maxRequests, planTiling.blockSize,                                   \
      reinterpret_cast<__gm__ int32_t*>(rowModes), reinterpret_cast<__gm__ int32_t*>(residentRows),        \
      reinterpret_cast<__gm__ int32_t*>(residentBlockTable), planTiling.jointHashCapacity,                 \
      planTiling.scratchElements)
  // The Host accepts only 1, 2 or 4 allocated rows per request. A partial
  // active request group is an input error. An inactive trailing group is OK.
  const int64_t rowsPerRequest =
      planTiling.capacity == 8192U ? 4 : (planTiling.maxRows == planTiling.maxRequests ? 1 : 2);
  auto* requestIndices = reinterpret_cast<const __gm__ int32_t*>(tokenToReq);
  auto* errors = reinterpret_cast<__gm__ int32_t*>(compactWorkspace);
  if (requestedRows < 0 || requestedRows > planTiling.maxRows || (rows & (rowsPerRequest - 1)) != 0) {
    if (blockIndex == 0) {
      errors[0] = -1;
    }
    return;
  }
  const uint32_t topk = static_cast<uint32_t>(planTiling.topk);
  const uint32_t capacity = static_cast<uint32_t>(planTiling.capacity);
  const uint32_t queryCount = static_cast<uint32_t>(rowsPerRequest) * topk;
  const uint32_t mapWords = capacity / 2U;
  const uint32_t packetOffset = PLAN_SCAN_STORAGE_ELEMENTS + kGroupControlWords + mapWords + capacity;
  auto mergeStorage = storage.Get<float>();
  for (int64_t first = blockIndex * rowsPerRequest; first < rows; first += blockCount * rowsPerRequest) {
    const int64_t end = first + rowsPerRequest;
    const int32_t request = requestIndices[first];
    if ((first > 0 && requestIndices[first - 1] == request) || (end < rows && requestIndices[end] == request)) {
      errors[first] = -1;
      continue;
    }
    RUN_PLAN_STAGE(kPrepareQueries);
    WaitForVector(pipe);
    if (groupControl[kGroupStatus] == kGroupReady && groupControl[kNeedsMerge]) {
      RUN_PLAN_STAGE(kPackMergeInputs);
      WaitForVector(pipe);
      const uint32_t activeSourceCount = static_cast<uint32_t>(groupControl[kActiveSourceCount]);
      const uint32_t recordsPerSource = static_cast<uint32_t>(groupControl[kRecordsPerSource]);
      auto* records = scan + packetOffset;
      if (activeSourceCount > 1U) {
        const uint32_t outputOffset = static_cast<uint32_t>(groupControl[kMergeOutputOffset]);
        AscendC::MrgSortSrcList<float> lists{
            mergeStorage[packetOffset], mergeStorage[packetOffset + 2U * recordsPerSource],
            mergeStorage[packetOffset + (activeSourceCount > 2U ? 4U * recordsPerSource : 0U)],
            mergeStorage[packetOffset + (activeSourceCount > 3U ? 6U * recordsPerSource : 0U)]};
        uint16_t sourceLengths[4] = {};
        uint32_t sourceConsumed[4] = {};
        for (uint32_t k = 0; k < activeSourceCount; ++k) {
          sourceLengths[k] = static_cast<uint16_t>(groupControl[kSourceLength + k]);
        }
        // All visible records fit the packet. Keep the established merge -> VF
        // event handoff, with no readback of consumptions or refill loop.
        AscendC::MrgSort<float, false>(mergeStorage[packetOffset + outputOffset], lists, sourceLengths,
                                       sourceConsumed, static_cast<uint16_t>((1U << activeSourceCount) - 1U), 1);
        WaitForVector(pipe);
        PublishScalarToVector(pipe);
        records += outputOffset;
      }
      asc_vf_call<DeduplicateMergePacket>(records, groupControl + kGroupControlWords, groupControl,
                                          static_cast<uint32_t>(groupControl[kMergeRecordCount]), queryCount,
                                          capacity, 0U, -1, 0U, mapWords);
      WaitForVector(pipe);
    }
    if (groupControl[kGroupStatus] == kGroupReady) {
      RUN_PLAN_STAGE(kClassifyResident);
      WaitForVector(pipe);
      if (groupControl[kGroupStatus] == kGroupReady) {
        asc_vf_call<AssignMissSlots>(groupControl + kGroupControlWords, groupControl, capacity,
                                     static_cast<uint32_t>(groupControl[kUniqueCount]));
        WaitForVector(pipe);
        if (groupControl[kGroupStatus] == kGroupReady) {
          RUN_PLAN_STAGE(kCommitHotGroup);
          WaitForVector(pipe);
        }
      }
    }
    if (groupControl[kGroupStatus] == kGroupCommitted) {
      CopyStagedOutputs(pipe, local, groupControl, first, topk, capacity, lruSlots, missTokens, missSlots);
    } else if (groupControl[kGroupStatus] == kInsufficientSlots) {
      RUN_PLAN_STAGE(kPlanPrivateRows);
    } else {
      errors[first] = -1;
    }
  }
#undef RUN_PLAN_STAGE
}

}  // namespace

// ==================== 9. Device and host launch entries ====================

// Direct launch adaptation only: fieldwise GM tiling load and unchanged VF call.
static_assert(sizeof(SparseKvPlanTilingData) == sizeof(JointDirectTilingData), "device/host tiling ABI mismatch");
template <bool AllowJoint>
__aicore__ inline void JointDirectBody_ordered_bitmap_512(
    GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens, GM_ADDR tokenToReq,
    GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots,
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
    GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling) {
  RunOrderedBitmapPlan<AllowJoint>(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, activeRows,
                                   lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens,
                                   missSlots, compactWorkspace, rowModes, residentRows, residentBlockTable, tiling);
}
extern "C" __global__ __aicore__ void sparse_kv_plan_joint_kernel_ordered_bitmap_512(
    GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens, GM_ADDR tokenToReq,
    GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots,
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
    GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  JointDirectBody_ordered_bitmap_512<true>(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq,
                                           activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount,
                                           missTokens, missSlots, compactWorkspace, rowModes, residentRows,
                                           residentBlockTable, tiling);
}
extern "C" __global__ __aicore__ void sparse_kv_plan_private_kernel_ordered_bitmap_512(
    GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens, GM_ADDR tokenToReq,
    GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots,
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
    GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  JointDirectBody_ordered_bitmap_512<false>(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq,
                                            activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount,
                                            missTokens, missSlots, compactWorkspace, rowModes, residentRows,
                                            residentBlockTable, tiling);
}

#include "platform/platform_ascendc.h"

extern "C" uint32_t joint_plan_block_dim_ordered_bitmap_512() {
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const uint32_t n = platform == nullptr ? 0 : static_cast<uint32_t>(platform->GetCoreNumAiv());
  return n < 64U ? n : 64U;
}
extern "C" void launch_joint_plan_ordered_bitmap_512(
    bool jointEntry, GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens,
    GM_ADDR tokenToReq, GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots,
    GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace,
    GM_ADDR rowModes, GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling, uint32_t blockDim,
    uint32_t dynamicUb, void* stream) {
  if (jointEntry) {
    sparse_kv_plan_joint_kernel_ordered_bitmap_512<<<blockDim, dynamicUb, stream>>>(
        reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, activeRows, lastReqIds, slotToToken,
        lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, rowModes, residentRows,
        residentBlockTable, tiling);
  } else {
    sparse_kv_plan_private_kernel_ordered_bitmap_512<<<blockDim, dynamicUb, stream>>>(
        reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, activeRows, lastReqIds, slotToToken,
        lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, rowModes, residentRows,
        residentBlockTable, tiling);
  }
}
