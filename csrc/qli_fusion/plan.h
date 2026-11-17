/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Original MemFabric_Hybrid calculation: Mulan PSL v2.
 * Experiment helpers and storage adapters: Apache-2.0.
 * SPDX-License-Identifier: MulanPSL-2.0 AND Apache-2.0
 */
// Generated from frozen rankpipe AC; storage/dispatch port: scripts/export_fusion_plan.py.
#pragma once
#include <climits>
#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
#include "sparse_kv_plan_config.h"
#include "storage.h"
#define JOINT_OPT_MASK 72
#define JOINT_COLD_FINISH 1
#define JOINT_REFINE 1
#define JOINT_REFINE_BITS 11
#define JOINT_HOT 1
#define JOINT_HOT_BITS 10
#define JOINT_HOT_PART_ONLY 0
#define JOINT_HOT_MISS_ONLY 0
#define JOINT_MTP8 1
#define JOINT_MTP7_OPT 1
#define JOINT_MTP7_REUSE 1
#define JOINT_MTP7_TILE 0
#define JOINT_MTP7_CAPACITY8K 0
#define JOINT_MTP7_MID_UNIQUE 0
#define JOINT_MTP7_FUSED_COUNT 1
#define JOINT_MTP7_FUSED_TUNE_BITS 3
#define JOINT_MTP7_PREFIX 0
#define JOINT_MTP7_MAP16 1
#define JOINT_LAYOUT16 1
#define JOINT_LAYOUT_HOT_BITS 15
#define JOINT_STAGE_EXEC 1
#define JOINT_STAGE_EXEC_BITS 15
#define JOINT_RANKPIPE 1
#define JOINT_RANKPIPE_BITS 5
#define EIGHT_MAP_ARGUMENT(q,t) , (d.capacity/2U)
#define QLI_OWNER_OFFSET(value) ((static_cast<uint32_t>(value)/2U)+(QLI_PLAN_STORAGE==3?2048U:static_cast<uint32_t>(value)/2U))
#define QLI_MAP_WORDS(requests,topk) (static_cast<uint64_t>(requests)*8U*static_cast<uint64_t>(topk)/2U)
#define QLI_UNIQUE_GM_WORDS(requests,capacity) (static_cast<uint64_t>(requests)*8U*static_cast<uint64_t>(capacity))
namespace qli_fused_plan {
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
  return warpBases[warp] + inclusive - value;
}
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
                                                          __gm__ int32_t* tokenToReq, int64_t maxToken,
                                                          int64_t maxRequests) {
  if (token < 0 || token >= maxToken || token >= visibleSeqLens[row]) {
    return false;
  }
  const int32_t request = tokenToReq[row];
  return request >= 0 && request < maxRequests;
}
__simt_callee__ __aicore__ inline void ClearMissTail(
    int64_t base, uint32_t assigned, uint32_t topk, uint32_t thread,
    __gm__ int32_t* missTokens, __gm__ int32_t* missSlots) {
  for (uint32_t pos = assigned + thread; pos < topk; pos += PLAN_THREADS) {
    missTokens[base + pos] = INVALID_TOKEN;
    missSlots[base + pos] = INVALID_TOKEN;
  }
}
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessColdRow(
    __ubuf__ int32_t* scanWorkspace, ScratchPtr missPositions, int64_t row, uint32_t thread,
    __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    uint32_t topk, uint32_t capacity, int32_t maxToken, int64_t maxRequests) {
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
                           maxToken, maxRequests);
    const int32_t rank = BlockExclusiveScan(scanWorkspace, flag, thread);
    const int32_t total = control[CONTROL_TOTAL];
    if (flag != 0) {
      missPositions[count + rank] = static_cast<int32_t>(position);
    }
    asc_syncthreads();
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
  asc_syncthreads();
}
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessRuntimeRow(
    __ubuf__ int32_t* scanWorkspace, int64_t row, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds,
    __gm__ int32_t* topkIndices, __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens,
    __gm__ int32_t* tokenToReq, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots,
    __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    ScratchPtr rowWorkspace, uint32_t hashCapacity, uint32_t topk, uint32_t capacity, int32_t maxToken,
    int64_t maxRequests) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  __ubuf__ int32_t* control = scanWorkspace + CONTROL_OFFSET;
  const int64_t topkBase = row * static_cast<int64_t>(topk);
  const int64_t capacityBase = row * static_cast<int64_t>(capacity);
  ScratchPtr hashKeys = rowWorkspace;
  ScratchPtr hashFirstPos = hashKeys + hashCapacity;
  ScratchPtr hashOwner = hashFirstPos + hashCapacity;
  ScratchPtr evictableSlots = hashOwner + hashCapacity;
  ScratchPtr missPositions = evictableSlots + capacity;
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
  if (control[CONTROL_RESET] != 0) {
    ProcessColdRow(scanWorkspace, missPositions, row, thread, reqIds, lastReqIds, topkIndices,
                   visibleSeqLens, tokenToReq, slotToToken, lruSlots, currentSlots,
                   missCount, missTokens, missSlots, topk, capacity, maxToken, maxRequests);
    return;
  }
  for (uint32_t i = thread; i < hashCapacity; i += PLAN_THREADS) {
    hashKeys[i] = HASH_EMPTY;
    hashFirstPos[i] = HASH_POSITION_EMPTY;
    hashOwner[i] = INVALID_TOKEN;
  }
  asc_syncthreads();
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    const int32_t token = topkIndices[topkBase + pos];
    if (IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, maxToken, maxRequests)) {
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
        } else {
          evictFlag = 1;
        }
      }
    }
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
  asc_syncthreads();
}

static_assert(72 > 0 && 72 <= 121, "invalid joint experiment mask");
struct JointPlanArgs {
  __gm__ int64_t *reqIds, *lastReqIds;
  FusionTopk topkIndices;
  __gm__ int32_t *stablePrefixLens, *visibleSeqLens, *tokenToReq;
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
struct FastGroupSource {
  int32_t request = -1;
  int32_t groupMaxToken = 0;
  uint32_t blockSize = 1U;
  uint32_t blockShift = UINT32_MAX;
  bool reqValid = false;
  __simt_callee__ __aicore__ inline uint32_t RowUpper(int32_t visible) const {
    if (!reqValid || visible <= 0 || groupMaxToken <= 0) return 0U;
    return static_cast<uint32_t>(visible < groupMaxToken ? visible : groupMaxToken);
  }
  __simt_callee__ __aicore__ inline bool IsValid(int32_t token, uint32_t rowUpper) const {
    return static_cast<uint32_t>(token) < rowUpper;
  }
  __simt_callee__ __aicore__ inline uint32_t Pages(uint32_t capacity) const {
    return blockShift != UINT32_MAX ? (capacity >> blockShift) : (capacity / blockSize);
  }
};
__simt_callee__ __aicore__ inline FastGroupSource MakeFastGroupSource(const JointPlanArgs& a, int32_t request) {
  FastGroupSource src;
  src.request = request;
  const uint32_t safeBlockSize = static_cast<uint32_t>(a.blockSize > 0 ? a.blockSize : 1);
  src.blockSize = safeBlockSize;
  if ((safeBlockSize & (safeBlockSize - 1U)) == 0U) {
    uint32_t shift = 0U;
    for (uint32_t v = safeBlockSize; v > 1U; v >>= 1U) ++shift;
    src.blockShift = shift;
  }
  src.reqValid = request >= 0 && static_cast<int64_t>(request) < a.maxRequests &&
                 a.blockSize > 0 && a.maxToken > 0;
  if (src.reqValid) {
    src.groupMaxToken = a.maxToken;
  }
  return src;
}
static_assert(PLAN_WARP_COUNT <= PLAN_WARP_SIZE && (PLAN_WARP_COUNT & (PLAN_WARP_COUNT - 1U)) == 0U,
              "TurboExclusiveScan requires power-of-two PLAN_WARP_COUNT <= PLAN_WARP_SIZE");
__simt_callee__ __aicore__ inline int32_t TurboExclusiveScan(__ubuf__ int32_t* workspace, int32_t value,
                                                             uint32_t thread, uint32_t& parity, int32_t& outTotal) {
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
__simt_callee__ __aicore__ inline void FinishTileTurbo(uint32_t tile, uint32_t extent) {
  if (tile + PLAN_THREADS >= extent) {
    asc_syncthreads();
  }
}
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void RunPrivateRow(const JointPlanArgs& a, int64_t row, ScratchPtr scratch,
                                                     __ubuf__ int32_t* scanWorkspace, uint32_t thread) {
  const int64_t topkBase = row * static_cast<int64_t>(a.topk);
  const int64_t capacityBase = row * static_cast<int64_t>(a.capacity);
  const FastGroupSource src = MakeFastGroupSource(a, a.tokenToReq[row]);
  const uint32_t rowUpper = src.RowUpper(a.visibleSeqLens[row]);
  const int64_t currentReqId = src.reqValid ? a.reqIds[src.request] : -1;
  const bool reset = a.rowModes[row] != ROW_PRIVATE || a.lastReqIds[row] != currentReqId;
  const uint32_t pages = src.Pages(a.capacity);
  const int32_t ownerPageBase = static_cast<int32_t>(row * static_cast<int64_t>(pages));
  if (thread == 0U) {
    a.residentRows[row] = static_cast<int32_t>(row);
  }
  for (uint32_t page = thread; page < pages; page += PLAN_THREADS) {
    a.residentBlockTable[row * static_cast<int64_t>(pages) + page] = ownerPageBase + static_cast<int32_t>(page);
  }
  ScratchPtr hashKeys = scratch;
  ScratchPtr hashFirstPos = hashKeys + a.hashCapacity;
  ScratchPtr hashOwner = hashFirstPos + a.hashCapacity;
  ScratchPtr evictableSlots = hashOwner + a.hashCapacity;
  ScratchPtr missPositions = evictableSlots + a.capacity;
  if (!reset) {
    for (uint32_t pos = thread; pos < a.topk; pos += PLAN_THREADS) {
      a.currentSlots[topkBase + pos] = INVALID_TOKEN;
    }
    for (uint32_t i = thread; i < a.hashCapacity; i += PLAN_THREADS) {
      hashKeys[i] = HASH_EMPTY;
      hashFirstPos[i] = HASH_POSITION_EMPTY;
      hashOwner[i] = INVALID_TOKEN;
    }
  }
  asc_syncthreads();
  if (thread == 0U) {
    a.rowModes[row] = ROW_PRIVATE;
  }
  uint32_t scanParity = 0U;
  if (reset) {
    int32_t count = 0;
    for (uint32_t tile = 0U; tile < a.topk; tile += PLAN_THREADS) {
      const uint32_t position = tile + thread;
      int32_t token = INVALID_TOKEN;
      int32_t flag = 0;
      if (position < a.topk) {
        token = a.topkIndices[topkBase + position];
        flag = src.IsValid(token, rowUpper) ? 1 : 0;
      }
      int32_t total = 0;
      const int32_t rank = TurboExclusiveScan(scanWorkspace, flag, thread, scanParity, total);
      if (position < a.topk) {
        const int32_t miss = count + rank;
        if (flag != 0 && miss < static_cast<int32_t>(a.capacity)) {
          a.slotToToken[capacityBase + miss] = token;
          a.currentSlots[topkBase + position] = miss;
          a.missTokens[topkBase + miss] = token;
          a.missSlots[topkBase + miss] = miss;
        } else {
          a.currentSlots[topkBase + position] = INVALID_TOKEN;
        }
      }
      count += total;
    }
    const int32_t assigned = count < static_cast<int32_t>(a.capacity) ? count : static_cast<int32_t>(a.capacity);
    for (uint32_t slot = static_cast<uint32_t>(assigned) + thread; slot < a.capacity; slot += PLAN_THREADS) {
      a.slotToToken[capacityBase + slot] = INVALID_TOKEN;
    }
    ClearMissTail(topkBase, static_cast<uint32_t>(assigned), a.topk, thread, a.missTokens, a.missSlots);
    const uint32_t unused = a.capacity - static_cast<uint32_t>(assigned);
    for (uint32_t order = thread; order < a.capacity; order += PLAN_THREADS) {
      a.lruSlots[capacityBase + order] =
          static_cast<int32_t>(order < unused ? static_cast<uint32_t>(assigned) + order : order - unused);
    }
    if (thread == 0U) {
      a.lastReqIds[row] = currentReqId;
      a.missCount[row] = assigned;
    }
    asc_syncthreads();
    return;
  }
  for (uint32_t pos = thread; pos < a.topk; pos += PLAN_THREADS) {
    const int32_t token = a.topkIndices[topkBase + pos];
    if (src.IsValid(token, rowUpper)) {
      missPositions[pos] = static_cast<int32_t>(
          InsertTopkToken(hashKeys, hashFirstPos, token, static_cast<int32_t>(pos), a.hashCapacity));
    } else {
      missPositions[pos] = static_cast<int32_t>(HASH_BUCKET_INVALID);
    }
  }
  asc_syncthreads();
  int32_t stablePrefix = a.stablePrefixLens[row];
  if (stablePrefix < 0) {
    stablePrefix = 0;
  } else if (stablePrefix > a.maxToken) {
    stablePrefix = a.maxToken;
  }
  const uint32_t uStable = static_cast<uint32_t>(stablePrefix);
  const uint32_t uMaxToken = static_cast<uint32_t>(a.maxToken > 0 ? a.maxToken : 0);
  const bool packedOwner = a.capacity <= OWNER_PACK_CAPACITY;
  int32_t evictableCount = 0;
  int32_t hitCount = 0;
  for (uint32_t tile = 0U; tile < a.capacity; tile += PLAN_THREADS) {
    const uint32_t order = tile + thread;
    int32_t slot = INVALID_TOKEN;
    int32_t evictFlag = 0;
    int32_t hitFlag = 0;
    if (order < a.capacity) {
      slot = a.lruSlots[capacityBase + order];
      if (slot >= 0 && static_cast<uint32_t>(slot) < a.capacity) {
        const int32_t token = a.slotToToken[capacityBase + slot];
        uint32_t bucket = HASH_BUCKET_INVALID;
        if (static_cast<uint32_t>(token) < uStable) {
          bucket = LookupTopkToken(hashKeys, token, a.hashCapacity);
        } else if (static_cast<uint32_t>(token) < uMaxToken) {
          a.slotToToken[capacityBase + slot] = INVALID_TOKEN;
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
    const int32_t ranks = TurboExclusiveScan(scanWorkspace, packed, thread, scanParity, totals);
    const int32_t evictRank = ranks & COUNT_MASK;
    const int32_t hitRank = ranks >> COUNT_SHIFT;
    if (evictFlag != 0) {
      evictableSlots[evictableCount + evictRank] = slot;
    }
    if (hitFlag != 0) {
      evictableSlots[a.capacity - 1U - static_cast<uint32_t>(hitCount + hitRank)] = slot;
    }
    FinishTileTurbo(tile, a.capacity);
    evictableCount += totals & COUNT_MASK;
    hitCount += totals >> COUNT_SHIFT;
  }
  const bool packMissBucket = a.topk <= 65535U && a.hashCapacity <= 65535U;
  int32_t localMissCount = 0;
  for (uint32_t tile = 0U; tile < a.topk; tile += PLAN_THREADS) {
    const uint32_t position = tile + thread;
    int32_t missFlag = 0;
    uint32_t bucket = HASH_BUCKET_INVALID;
    if (position < a.topk) {
      bucket = static_cast<uint32_t>(missPositions[position]);
      if (bucket != HASH_BUCKET_INVALID) {
        const int32_t owner = hashOwner[bucket];
        if (owner >= 0 && hashFirstPos[bucket] == static_cast<int32_t>(position)) {
          a.currentSlots[topkBase + position] =
              packedOwner ? (owner & OWNER_SLOT_MASK) : a.lruSlots[capacityBase + owner];
        } else {
          missFlag = 1;
        }
      }
    }
    int32_t tileMissCount = 0;
    const int32_t rank = TurboExclusiveScan(scanWorkspace, missFlag, thread, scanParity, tileMissCount);
    if (missFlag != 0) {
      missPositions[localMissCount + rank] =
          packMissBucket ? static_cast<int32_t>((bucket << 16U) | position) : static_cast<int32_t>(position);
    }
    FinishTileTurbo(tile, a.topk);
    localMissCount += tileMissCount;
  }
  const int32_t assignCount = localMissCount < evictableCount ? localMissCount : evictableCount;
  for (uint32_t miss = thread; miss < static_cast<uint32_t>(assignCount); miss += PLAN_THREADS) {
    const int32_t slot = evictableSlots[miss];
    const uint32_t packedMiss = static_cast<uint32_t>(missPositions[miss]);
    const int32_t position =
        packMissBucket ? static_cast<int32_t>(packedMiss & 0xFFFFU) : static_cast<int32_t>(packedMiss);
    const int32_t token = packMissBucket ? hashKeys[packedMiss >> 16U] : a.topkIndices[topkBase + position];
    a.slotToToken[capacityBase + slot] = token;
    a.currentSlots[topkBase + position] = slot;
    a.missTokens[topkBase + miss] = token;
    a.missSlots[topkBase + miss] = slot;
  }
  ClearMissTail(topkBase, static_cast<uint32_t>(assignCount), a.topk, thread, a.missTokens, a.missSlots);
  const int32_t unassignedCount = evictableCount - assignCount;
  const int32_t validLruCount = evictableCount + hitCount;
  for (uint32_t output = thread; output < static_cast<uint32_t>(validLruCount); output += PLAN_THREADS) {
    int32_t slot;
    if (output < static_cast<uint32_t>(unassignedCount)) {
      slot = evictableSlots[assignCount + output];
    } else if (output < static_cast<uint32_t>(evictableCount)) {
      slot = evictableSlots[output - static_cast<uint32_t>(unassignedCount)];
    } else {
      slot = evictableSlots[a.capacity - 1U - (output - static_cast<uint32_t>(evictableCount))];
    }
    a.lruSlots[capacityBase + output] = slot;
  }
  if (thread == 0U) {
    a.missCount[row] = assignCount;
  }
  asc_syncthreads();
}
static_assert(11>=0 && 11<16,"refine bits A/B/C/D");
constexpr bool REFINE_A=(11&1U)!=0;
constexpr bool REFINE_B=(11&2U)!=0;
constexpr bool REFINE_C=(11&4U)!=0;
constexpr bool REFINE_D=(11&8U)!=0;

static_assert(1>=0 && 1<=2,"MTP7 candidate selector");
constexpr uint64_t EIGHT_UB_WORDS=216U*1024U/4U;
constexpr uint32_t EIGHT_CTX=128U;
static_assert(10>=0 && 10<32,"hot factor bits");
constexpr bool HOT_FILTER=(10&1U)!=0;
constexpr bool HOT_PART=(10&2U)!=0 || 0;
constexpr bool HOT_MISS=(10&2U)!=0 || 0;
constexpr bool HOT_PACKET=(10&4U)!=0;
constexpr bool HOT_REUSE=(10&8U)!=0;
constexpr bool HOT_FIXED=(10&16U)!=0;
static_assert(REFINE_A && REFINE_B && REFINE_D && !REFINE_C,"frozen ABD parent");
constexpr uint32_t HW_BITS = 72 <= 69 ? 72 -38U :
    72==70 ? 15U : 72==71 ? 14U : 72==72 ? 5U : 4U;
constexpr bool HW_A=(HW_BITS&1U)!=0, HW_B=(HW_BITS&2U)!=0;
constexpr bool HW_C=(HW_BITS&4U)!=0, HW_D=(HW_BITS&8U)!=0;
static_assert(!HW_D,"reused input cannot retain old packet tails");
constexpr uint32_t HW_DIV = 72>=70 ? 2U : (HW_BITS&16U)!=0 ? 4U : 1U;
constexpr uint32_t SORTED_THREADS = PLAN_THREADS/HW_DIV >= PLAN_WARP_SIZE ? PLAN_THREADS/HW_DIV : PLAN_WARP_SIZE;
constexpr uint32_t SORTED_WARPS = SORTED_THREADS/PLAN_WARP_SIZE;
static_assert(SORTED_THREADS<=PLAN_THREADS && (SORTED_WARPS&(SORTED_WARPS-1U))==0U,"warp layout");
__simt_callee__ __aicore__ inline int32_t SortedScan(__ubuf__ int32_t* work,int32_t value,
    uint32_t thread,uint32_t& parity,int32_t& total){
  __ubuf__ int32_t* totals=work+(parity?WARP_BASES_OFFSET:WARP_TOTALS_OFFSET);parity^=1U;
  uint32_t lane=thread&(PLAN_WARP_SIZE-1U),warp=thread/PLAN_WARP_SIZE;
  int32_t inclusive=value;
#pragma unroll
  for(uint32_t d=1;d<PLAN_WARP_SIZE;d<<=1U){int32_t x=asc_shfl_up(inclusive,d,PLAN_WARP_SIZE);if(lane>=d)inclusive+=x;}
  if(lane==PLAN_WARP_SIZE-1U)totals[warp]=inclusive;asc_syncthreads();
  int32_t own=lane<SORTED_WARPS?totals[lane]:0,sum=own;
#pragma unroll
  for(uint32_t d=1;d<SORTED_WARPS;d<<=1U){int32_t x=asc_shfl_up(sum,d,PLAN_WARP_SIZE);if(lane>=d)sum+=x;}
  total=asc_shfl(sum,SORTED_WARPS-1U,PLAN_WARP_SIZE);
  return asc_shfl(sum-own,warp,PLAN_WARP_SIZE)+inclusive-value;
}
__simt_callee__ __aicore__ inline int32_t RefineBoolScan(__ubuf__ int32_t* work,bool flag,
    uint32_t thread,uint32_t& parity,int32_t& total){
  if constexpr(!REFINE_B)return SortedScan(work,flag?1:0,thread,parity,total);
  const uint32_t lane=thread&(PLAN_WARP_SIZE-1U),warp=thread/PLAN_WARP_SIZE;
  const uint32_t votes=asc_ballot(flag?1:0),before=(uint32_t{1}<<lane)-1U;
  const int32_t ownRank=__popc(votes&before);
  __ubuf__ int32_t* totals=work+(parity?WARP_BASES_OFFSET:WARP_TOTALS_OFFSET);parity^=1U;
  if(lane==PLAN_WARP_SIZE-1U)totals[warp]=__popc(votes);
  asc_syncthreads();
  int32_t own=lane<SORTED_WARPS?totals[lane]:0,sum=own;
#pragma unroll
  for(uint32_t d=1;d<SORTED_WARPS;d<<=1U){int32_t x=asc_shfl_up(sum,d,PLAN_WARP_SIZE);if(lane>=d)sum+=x;}
  total=asc_shfl(sum,SORTED_WARPS-1U,PLAN_WARP_SIZE);
  return asc_shfl(sum-own,warp,PLAN_WARP_SIZE)+ownRank;
}
__simt_callee__ __aicore__ inline bool EightMapIsPacked(uint32_t rows, uint32_t topk) {
  return true;
}
__aicore__ inline uint32_t EightMapWords(uint32_t queries, uint32_t topk, uint32_t capacity) {
  return capacity/2U;
}
struct EightMap16View {
  __ubuf__ uint16_t* values;
  __simt_callee__ __aicore__ inline int32_t operator[](uint32_t i) const {
    const uint16_t value = values[i];
    return value == 65535U ? -1 : static_cast<int32_t>(value);
  }
};
template <typename P>
__simt_callee__ __aicore__ inline uint32_t SortedLower(P p, uint32_t n, int32_t token) {
  uint32_t lo=0,hi=n;
  while(lo<hi){uint32_t mid=lo+((hi-lo)>>1U);if(p[mid]<token)lo=mid+1U;else hi=mid;}
  return lo;
}

__simt_callee__ __aicore__ inline void RankPipeCold(const JointPlanArgs& a,
    int64_t first,int64_t end,__ubuf__ uint16_t* map,FusionUnique unique,
    __ubuf__ int32_t* owner,__ubuf__ int32_t* ctx,uint32_t thread){
  constexpr uint32_t Threads=SORTED_THREADS;
  const uint32_t u=static_cast<uint32_t>(ctx[1]),rows=static_cast<uint32_t>(end-first);
  if(u>a.capacity){if(thread==0U)ctx[0]=0;asc_syncthreads();return;}
  const int64_t qb=first*static_cast<int64_t>(a.topk),cb=first*static_cast<int64_t>(a.capacity);
  const uint32_t unused=a.capacity-u,q=rows*a.topk;
  for(uint32_t slot=thread;slot<a.capacity;slot+=Threads){
    const int32_t token=slot<u?unique[slot]:-1;
    a.slotToToken[cb+slot]=token;
    owner[slot]=static_cast<int32_t>(slot<unused?u+slot:slot-unused);
    if(slot<u){a.missTokens[qb+slot]=token;a.missSlots[qb+slot]=static_cast<int32_t>(slot);}
  }
  for(uint32_t i=thread;i<q;i+=Threads){
    const uint32_t uid=map[i];a.currentSlots[qb+i]=uid!=65535U?static_cast<int32_t>(uid):-1;
  }
  asc_syncthreads();
  for(uint32_t i=thread;i<2048U;i+=Threads)reinterpret_cast<__ubuf__ int32_t*>(map)[i]=-1;
  const int32_t req=a.tokenToReq[first];
  const FastGroupSource src=MakeFastGroupSource(a,req);const uint32_t pages=src.Pages(a.capacity);
  for(uint32_t r=0;r<rows;++r){const int64_t row=first+r;
    for(uint32_t page=thread;page<pages;page+=Threads)
      a.residentBlockTable[row*pages+page]=static_cast<int32_t>(first*pages+page);
    if(thread==0U){
      a.residentRows[row]=static_cast<int32_t>(first);
      const int64_t left=static_cast<int64_t>(u)-static_cast<int64_t>(r)*a.topk;
      a.missCount[row]=left<=0?0:left<a.topk?static_cast<int32_t>(left):static_cast<int32_t>(a.topk);
    }
    if(thread==1U)a.rowModes[row]=rows==1U?ROW_PRIVATE:(r==0U?ROW_JOINT:ROW_FOLLOWER);
  }
  if(thread==0U){a.lastReqIds[first]=a.reqIds[req];ctx[46]=q-u;ctx[47]=u;ctx[120]=1;ctx[0]=2;}
  asc_syncthreads();
}

template<int Phase>
__simt_callee__ __aicore__ inline void LayoutHotFinish(const JointPlanArgs& a,
    int64_t first,int64_t end,__ubuf__ int32_t* data,__ubuf__ int32_t* ctx,
    __ubuf__ int32_t* scan,__gm__ int32_t* workspace) {
  const uint32_t t=threadIdx.x,c=a.capacity,topk=a.topk,q=(end-first)*topk,u=ctx[1];
  auto* map=reinterpret_cast<__ubuf__ uint16_t*>(data);
  auto unique=FusionUnique{data,workspace+QLI_MAP_WORDS(a.maxRequests,a.topk)+first*c,a.topkIndices,first*a.topk,c,0U,end-first==1,ctx+48};
  auto* owner=data+QLI_OWNER_OFFSET(c);
  auto* part=reinterpret_cast<__ubuf__ uint16_t*>(owner+c);
  auto* bits=reinterpret_cast<__ubuf__ uint32_t*>(part+c);
  auto* flags=map;
  const bool directRank=(QLI_PLAN_STORAGE!=2) && a.maxToken<=262144;
  auto* slotLookup=directRank?reinterpret_cast<__ubuf__ uint16_t*>(owner+16384U):part;
  const int64_t qb=first*topk,cb=first*c;
  const uint32_t rows=end-first;
  const int32_t req=a.tokenToReq[first];const int64_t reqid=a.reqIds[req];
  uint32_t evicts=Phase==9?ctx[43]:0,hits=Phase==9?ctx[44]:0,misses=Phase==9?ctx[45]:0,parity=0;
  if constexpr(Phase!=9){
  const bool reset=a.rowModes[first]!=(rows==1?ROW_PRIVATE:ROW_JOINT)||a.lastReqIds[first]!=reqid;
  if(reset){asc_syncthreads();
    RankPipeCold(a,first,end,map,unique,owner,ctx,t);
    return;
  }
  int32_t stable=a.stablePrefixLens[first];
  for(int64_t r=first+1;r<end;++r)if(a.stablePrefixLens[r]<stable)stable=a.stablePrefixLens[r];
  if(stable<0)stable=0;if(stable>a.maxToken)stable=a.maxToken;
  for(uint32_t i=t;i<q/2;i+=SORTED_THREADS)workspace[qb/2+i]=data[i];
  asc_syncthreads();
  if(directRank){
    auto* masks=reinterpret_cast<__ubuf__ uint32_t*>(owner);
    auto* firstUid=reinterpret_cast<__ubuf__ uint32_t*>(owner+8192U);
    const uint32_t words=(static_cast<uint32_t>(a.maxToken)+31U)/32U;
    for(uint32_t w=t;w<words;w+=SORTED_THREADS)masks[w]=0U;
    asc_syncthreads();
    for(uint32_t uid=t;uid<u;uid+=SORTED_THREADS){
      const uint32_t token=static_cast<uint32_t>(unique[uid]),word=token>>5U;
      if(uid==0U || (static_cast<uint32_t>(unique[uid-1U])>>5U)!=word){
        uint32_t value=0U;
        for(uint32_t j=uid;j<u && (static_cast<uint32_t>(unique[j])>>5U)==word;++j)
          value|=uint32_t{1}<<(static_cast<uint32_t>(unique[j])&31U);
        firstUid[word]=uid;masks[word]=value;
      }
    }
    asc_syncthreads();
    for(uint32_t slot=t;slot<c;slot+=SORTED_THREADS){
      const int32_t token=a.slotToToken[cb+slot];uint32_t encoded=token>=stable?0x8000U:0U;
      if(token>=0 && token<stable){
        const uint32_t word=static_cast<uint32_t>(token)>>5U,bit=uint32_t{1}<<(static_cast<uint32_t>(token)&31U);
        const uint32_t mask=masks[word];
        if(mask&bit){const uint32_t uid=firstUid[word]+__popc(mask&(bit-1U));encoded=0x4000U|uid;}
      }
      slotLookup[slot]=static_cast<uint16_t>(encoded);
    }
    asc_syncthreads();
  }else{
  const uint32_t directoryBins=(15&8U) && c==4096U ? 512U : 1024U;
  uint32_t bucketShift=0;
  while((static_cast<uint64_t>(directoryBins)<<bucketShift)<static_cast<uint32_t>(a.maxToken))++bucketShift;
  for(uint32_t i=t;i<=directoryBins;i+=SORTED_THREADS)owner[i]=SortedLower(unique,u,static_cast<int32_t>(i<<bucketShift));
  asc_syncthreads();
  for(uint32_t slot=t;slot<c;slot+=SORTED_THREADS){
    const int32_t token=a.slotToToken[cb+slot];uint32_t encoded=token>=stable?0x8000U:0U;
    if(token>=0&&token<stable){uint32_t pos;
      const uint32_t bucket=static_cast<uint32_t>(token)>>bucketShift;
      const uint32_t lo=owner[bucket],hi=owner[bucket+1U];
      pos=lo+SortedLower(unique+lo,hi-lo,token);
      if(pos<u&&unique[pos]==token)encoded=0x4000U|pos;
    }
    part[slot]=static_cast<uint16_t>(encoded);
  }
  asc_syncthreads();
  }
  for(uint32_t i=t;i<u;i+=SORTED_THREADS)owner[i]=-1;
  asc_syncthreads();
  for(uint32_t tile=0;tile<c;tile+=SORTED_THREADS){
    const uint32_t order=tile+t;int32_t slot=order<c?a.lruSlots[cb+order]:-1;
    const bool valid=slot>=0&&static_cast<uint32_t>(slot)<c;
    if(order<c && !valid)asc_atomic_min(ctx,0);
    bool hit=false,invalid=false;
    if(valid){
      const uint32_t encoded=slotLookup[slot];invalid=(encoded&0x8000U)!=0;hit=(encoded&0x4000U)!=0;
      if(hit)asc_atomic_max(owner+(encoded&0x3fffU),static_cast<int32_t>((order<<16)|slot));
    }
    if(order<c)flags[order]=valid?static_cast<uint16_t>(slot|(hit?0x4000U:0U)|(invalid?0x8000U:0U)):uint16_t{65535};
  }
  asc_syncthreads();
  if constexpr(Phase==6)return;
  for(uint32_t tile=0;tile<c;tile+=SORTED_THREADS){
    const uint32_t order=tile+t;bool valid=false,hit=false,invalid=false;uint32_t slot=0;
    if(order<c){
      uint32_t f=flags[order];valid=f!=65535U;hit=valid&&(f&0x4000U);invalid=valid&&(f&0x8000U);slot=f&0x3fffU;
    }
    const int32_t evict=valid&&!hit,totalFlag=evict+(hit?(1U<<COUNT_SHIFT):0U);int32_t total=0;
    const uint32_t rank=SortedScan(scan,totalFlag,t,parity,total);
    const uint16_t entry=static_cast<uint16_t>(slot|(invalid?0x4000U:0U));
    if(evict)part[evicts+(rank&COUNT_MASK)]=entry;
    if(hit)part[c-1-(hits+(rank>>COUNT_SHIFT))]=entry;
    asc_syncthreads();evicts+=static_cast<uint32_t>(total)&COUNT_MASK;hits+=static_cast<uint32_t>(total)>>COUNT_SHIFT;
  }
  if(evicts+hits!=c){if(t==0)ctx[0]=0;return;}
  parity=0;
  for(uint32_t tile=0;tile<u;tile+=SORTED_THREADS){
    uint32_t uid=tile+t;bool missing=uid<u&&owner[uid]<0;int32_t total=0;
    uint32_t rank=RefineBoolScan(scan,missing,t,parity,total)+misses;
    if(uid<u)flags[uid]=missing?static_cast<uint16_t>(rank):uint16_t{65535};
    if(missing&&rank<evicts){uint32_t slot=part[rank]&0x3fffU;
      owner[uid]=slot;
    }
    asc_syncthreads();misses+=total;
  }
  if(misses>evicts){if(t==0)ctx[0]=0;return;}
  }
  for(uint32_t uid=t;uid<u;uid+=SORTED_THREADS){
    const bool missing=(static_cast<uint32_t>(owner[uid])&0x40000000U)!=0;
    const uint32_t rank=(static_cast<uint32_t>(owner[uid])>>16)&0x3fffU;
    if(missing){int32_t slot=owner[uid]&0xffff;a.slotToToken[cb+slot]=unique[uid];a.missTokens[qb+rank]=unique[uid];a.missSlots[qb+rank]=slot;}
  }
  for(uint32_t pair=t;pair<q/2;pair+=SORTED_THREADS){
    const uint32_t packed=static_cast<uint32_t>(workspace[qb/2+pair]);
    const uint32_t lo=packed&65535U,hi=packed>>16;
    a.currentSlots[qb+2*pair]=lo==65535U?-1:owner[lo]&0xffff;
    a.currentSlots[qb+2*pair+1]=hi==65535U?-1:owner[hi]&0xffff;
  }
  asc_syncthreads();
  for(uint32_t i=t;i<2048U;i+=SORTED_THREADS)data[i]=-1;
  const uint32_t unused=evicts-misses;
  for(uint32_t order=t;order<c;order+=SORTED_THREADS){
    const uint32_t index=order<unused?misses+order:order<evicts?order-unused:c-1-(order-evicts);
    const uint32_t entry=part[index],slot=entry&0x3fffU;
    if(index>=misses && (entry&0x4000U))a.slotToToken[cb+slot]=-1;
    owner[order]=static_cast<int32_t>(slot);
  }
  const FastGroupSource src=MakeFastGroupSource(a,req);const uint32_t pages=src.Pages(c);
  for(uint32_t r=0;r<rows;++r){int64_t row=first+r;
    for(uint32_t p=t;p<pages;p+=SORTED_THREADS)a.residentBlockTable[row*pages+p]=first*pages+p;
    if(t==0){a.residentRows[row]=first;int64_t left=static_cast<int64_t>(misses)-r*topk;a.missCount[row]=left<=0?0:left<topk?left:topk;}
    if(t==1)a.rowModes[row]=rows==1?ROW_PRIVATE:(r==0?ROW_JOINT:ROW_FOLLOWER);
  }
  if(t==0){
    ctx[46]=q-misses;ctx[47]=misses;ctx[120]=1;
    a.lastReqIds[first]=reqid;ctx[0]=2;
  }asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void RankPipeCompactVf(
    __ubuf__ int32_t* data,__ubuf__ int32_t* ctx,__ubuf__ int32_t* scan,
    uint32_t c,uint32_t u){
  const uint32_t t=threadIdx.x;
  auto* flags=reinterpret_cast<__ubuf__ uint16_t*>(data);
  auto* owner=data+QLI_OWNER_OFFSET(c);
  auto* part=reinterpret_cast<__ubuf__ uint16_t*>(owner+c);
  uint32_t hits=0U,misses=0U,parity=0U;
  for(uint32_t block=0;block<c;block+=32U*SORTED_THREADS){
    const uint32_t begin=block+t*32U;uint32_t hitMask=0U;
    for(uint32_t i=0;i<32U;++i)if(begin+i<c){
      const uint32_t f=flags[begin+i];
      if(f==65535U)asc_atomic_min(ctx,0);
      else if(f&0x4000U)hitMask|=uint32_t{1}<<i;
    }
    int32_t total=0;
    const uint32_t baseHit=hits+SortedScan(scan,__popc(hitMask),t,parity,total);
    for(uint32_t i=0;i<32U;++i)if(begin+i<c){
      const uint32_t f=flags[begin+i],bit=uint32_t{1}<<i;
      const uint32_t before=baseHit+__popc(hitMask&(bit-1U));
      if(f!=65535U){
        const uint16_t entry=static_cast<uint16_t>((f&0x3fffU)|((f&0x8000U)>>1U));
        part[(hitMask&bit)?c-1U-before:begin+i-before]=entry;
      }
    }
    asc_syncthreads();
    if(ctx[0]==0)return;
    hits+=static_cast<uint32_t>(total);
  }
  const uint32_t evicts=c-hits;
  for(uint32_t block=0;block<u;block+=32U*SORTED_THREADS){
    const uint32_t begin=block+t*32U;uint32_t missMask=0U;
    for(uint32_t i=0;i<32U;++i)if(begin+i<u && owner[begin+i]<0)missMask|=uint32_t{1}<<i;
    int32_t total=0;
    const uint32_t baseMiss=misses+SortedScan(scan,__popc(missMask),t,parity,total);
    for(uint32_t i=0;i<32U;++i){const uint32_t bit=uint32_t{1}<<i;
      if(missMask&bit){const uint32_t rank=baseMiss+__popc(missMask&(bit-1U));
        if(rank<evicts)owner[begin+i]=static_cast<int32_t>(0x40000000U|(rank<<16U)|(part[rank]&0x3fffU));}
    }
    asc_syncthreads();misses+=static_cast<uint32_t>(total);
  }
  if(t==0){ctx[43]=evicts;ctx[44]=hits;ctx[45]=misses;ctx[0]=misses<=evicts?1:0;}
  asc_syncthreads();
}
template <int Phase, bool Fixed, typename P>
__simt_callee__ __aicore__ inline void SortedStage(const JointPlanArgs& a,
    int64_t first,int64_t end,P data,__ubuf__ int32_t* ctx,__ubuf__ int32_t* packet,
    __ubuf__ int32_t* scan, __gm__ int32_t* mapWorkspace) {
  const uint32_t topk=Fixed?2048U:a.topk,capacity=Fixed?4096U:a.capacity;
  const uint32_t t=static_cast<uint32_t>(threadIdx.x), rows=static_cast<uint32_t>(end-first);
  const bool packedMap = EightMapIsPacked(rows, topk);
  const uint32_t mapWords = capacity/2U;
  P map=data, owner=data+QLI_OWNER_OFFSET(capacity);
  auto unique=FusionUnique{data,mapWorkspace+QLI_MAP_WORDS(a.maxRequests,topk)+first*capacity,a.topkIndices,first*static_cast<int64_t>(topk),capacity,0U,end-first==1,ctx+48};
  auto* map16 = reinterpret_cast<__ubuf__ uint16_t*>(ctx + EIGHT_CTX);
  const uint32_t q=rows*topk;
  const int64_t qb=first*static_cast<int64_t>(topk),cb=first*static_cast<int64_t>(capacity);
  if constexpr(Phase==0) {
    if(t==0U){
      ctx[120]=0;
      ctx[0]=1;ctx[1]=0;ctx[2]=-1;ctx[4]=0;ctx[6]=0;ctx[7]=0;ctx[28]=0;ctx[29]=0;
      uint32_t chunk=(3U*capacity>64U?3U*capacity:64U)/16U;
      if(chunk>768U)chunk=768U;ctx[3]=static_cast<int32_t>(chunk&~3U);
      uint64_t words=16ULL*static_cast<uint32_t>(ctx[3]);
      if(HW_C && a.scratchElements<=EIGHT_UB_WORDS)
        words=24576U;
      ctx[30]=static_cast<int32_t>((words/2U)&~uint64_t{7U});
      const int32_t req=a.tokenToReq[first];
      uint32_t bits=1;while(bits<31U&&(1ULL<<bits)<q)++bits;
      if(rows<1U||rows>8U||(rows&(rows-1U))!=0U||capacity>32768U||q>capacity||q>65535U||a.maxToken<=0||a.maxToken>16777216||
         (static_cast<uint64_t>(a.maxToken-1)<<bits)>UINT32_MAX||req<0||req>=a.maxRequests)ctx[0]=0;
      for(uint32_t r=0;r<8U;++r){ctx[48+r]=0;ctx[56+r]=0;}
      if(ctx[0])for(uint32_t r=0;r<rows;++r){
        const int64_t b=(first+r)*static_cast<int64_t>(topk);
        int32_t upper=a.visibleSeqLens[first+r];if(upper<0)upper=0;if(upper>a.maxToken)upper=a.maxToken;
        if(topk && a.topkIndices[b]>=0 && a.topkIndices[b+topk-1U]>=0 && a.topkIndices[b+topk-1U]<upper){
          ctx[48+r]=0;ctx[56+r]=static_cast<int32_t>(topk);
        }else{
        uint32_t validEnd=topk;
        if(topk && a.topkIndices[b]>=0 && a.topkIndices[b+topk-1U]<0){
          uint32_t lo=0,hi=topk;
          while(lo<hi){uint32_t mid=lo+(hi-lo)/2U;if(a.topkIndices[b+mid]>=0)lo=mid+1U;else hi=mid;}
          validEnd=lo;
        }
        ctx[48+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,validEnd,0));
        ctx[56+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,validEnd,upper));
        }
        if(ctx[48+r]<ctx[56+r])ctx[6]=1;
      }
      if constexpr(REFINE_A){
        if(ctx[0] && rows==1U){ctx[1]=ctx[56]-ctx[48];ctx[6]=0;}
      }
    }
    asc_syncthreads();
    if(ctx[0])for(uint32_t i=t;i<q;i+=SORTED_THREADS){
      if constexpr(REFINE_A){
        if(rows==1U){
          const int32_t token=a.topkIndices[qb+i];
          if(i && token>=0 && token==a.topkIndices[qb+i-1U])asc_atomic_min(ctx,0);
          const bool valid=i>=static_cast<uint32_t>(ctx[48]) && i<static_cast<uint32_t>(ctx[56]);
          map16[i]=valid?static_cast<uint16_t>(i-static_cast<uint32_t>(ctx[48])):uint16_t{65535};
          if(valid)unique.Set(i-static_cast<uint32_t>(ctx[48]),token,i);
        }else {
          if (packedMap) { map16[i] = 65535U; }
          else map[i] = -1;
        }
      }else{
        if(rows==1U && i && a.topkIndices[qb+i]==a.topkIndices[qb+i-1U])asc_atomic_min(ctx,0);
        if (packedMap) { map16[i] = 65535U; }
        else map[i] = -1;
      }
    }
    asc_syncthreads();
  } else if constexpr(Phase==1) {
    const uint32_t oldChunk=ctx[3],oldActive=ctx[28],outOffset=ctx[30];
    if constexpr(HW_D){
      if(ctx[29])for(uint32_t z=t;z<oldActive*oldChunk;z+=SORTED_THREADS){
        uint32_t k=z/oldChunk,j=z%oldChunk;
        uint32_t used=ctx[80+k],left=static_cast<uint32_t>(ctx[72+k])-used;
        if(j<left){uint32_t src=k*oldChunk+used+j,dst=k*oldChunk+j;
          packet[outOffset+2U*dst]=packet[2U*src];packet[outOffset+2U*dst+1U]=packet[2U*src+1U];}
      }
      asc_syncthreads();
    }
    if(t==0U){
      for(uint32_t r=0;r<8U;++r){ctx[88+r]=0;ctx[96+r]=0;}
      if(HW_D&&ctx[29])for(uint32_t k=0;k<oldActive;++k){
        uint32_t r=ctx[64+k];ctx[88+r]=ctx[72+k]-ctx[80+k];ctx[96+r]=k*oldChunk;
      }
      uint32_t active=0;
      for(uint32_t r=0;r<rows;++r)if(ctx[48+r]<ctx[56+r])ctx[64+active++]=r;
      uint32_t chunk=oldChunk;
      if(rows>4U){
        const uint64_t work = 24576U;
        const uint32_t maximum = 612U;
        chunk = static_cast<uint32_t>(work / 40U) & (packedMap ? ~7U : ~3U);
        if (chunk > maximum) chunk = maximum;
        ctx[30]=24U*chunk;
      }else if(HW_C&&active){chunk=(static_cast<uint32_t>(ctx[30])/(2U*active))&~3U;if(chunk>4092U)chunk=4092U;}
      ctx[3]=chunk;ctx[28]=active;ctx[7]=0;
      ctx[107]=0;
      uint32_t total=0;
      for(uint32_t k=0;k<active;++k){uint32_t r=ctx[64+k],n=ctx[56+r]-ctx[48+r];
        if(n>chunk)n=chunk;ctx[72+k]=n;ctx[80+k]=0;total+=n;}
      ctx[4]=total;ctx[5]=total;
    }
    asc_syncthreads();
    const uint32_t chunk=ctx[3],active=ctx[28];
    auto* score=reinterpret_cast<__ubuf__ float*>(packet);
    for(uint32_t i=t;i<active*chunk;i+=SORTED_THREADS){uint32_t k=i/chunk,j=i%chunk;
      if(j<static_cast<uint32_t>(ctx[72+k])){uint32_t r=ctx[64+k],pos=static_cast<uint32_t>(ctx[48+r])+j;
        if(HW_D&&j<static_cast<uint32_t>(ctx[88+r])){uint32_t saved=ctx[96+r]+j;
          packet[2U*i]=packet[outOffset+2U*saved];packet[2U*i+1U]=packet[outOffset+2U*saved+1U];
        }else{score[2U*i]=-static_cast<float>(a.topkIndices[qb+r*topk+pos]);packet[2U*i+1U]=r*topk+pos;}
      }
    }
    asc_syncthreads();
  } else if constexpr(Phase==2) {
    const uint32_t chunk=ctx[3],active=ctx[28];
    auto* score=reinterpret_cast<__ubuf__ float*>(packet);
    auto* out=packet+static_cast<uint32_t>(ctx[30]);
    for(uint32_t z=t;z<active*chunk;z+=SORTED_THREADS){
      const uint32_t k=z/chunk,j=z%chunk,n=ctx[72+k];if(j>=n)continue;
      const float key=score[2U*z];const int32_t id=packet[2U*z+1U];uint32_t rank=j;
      for(uint32_t s=0;s<active;++s)if(s!=k){
        uint32_t lo=0,hi=ctx[72+s];
        while(lo<hi){uint32_t mid=lo+((hi-lo)>>1U),p=s*chunk+mid;
          const float other=score[2U*p];
          if(other>key||(other==key&&packet[2U*p+1U]<id))lo=mid+1U;else hi=mid;
        }rank+=lo;
      }
      out[2U*rank]=packet[2U*z];out[2U*rank+1U]=id;
      if(j+1U==n)asc_atomic_min(ctx+5,static_cast<int32_t>(rank+1U));
    }
    asc_syncthreads();
    if(t==0U){
      for(uint32_t i=0;i<static_cast<uint32_t>(ctx[5]);++i){
        uint32_t row=static_cast<uint32_t>(out[2U*i+1U])/topk;
        for(uint32_t k=0;k<active;++k)if(static_cast<uint32_t>(ctx[64+k])==row)++ctx[80+k];
      }
      for(uint32_t k=0;k<active;++k)ctx[48+ctx[64+k]]+=ctx[80+k];
      ctx[4]=ctx[5];
    }
    asc_syncthreads();
  } else if constexpr(Phase==3) {
    const uint32_t n=ctx[4],chunk=ctx[3];auto* out=packet+(ctx[7]?0U:static_cast<uint32_t>(ctx[30]));
    auto* scores=reinterpret_cast<__ubuf__ float*>(out);
    uint32_t count=ctx[1],parity=0;
    for(uint32_t tile=0;tile<n;tile+=SORTED_THREADS){
      const uint32_t i=tile+t;int32_t token=-1,prev=-1;bool added=false;
      if(i<n){token=static_cast<int32_t>(-scores[2U*i]);
        prev=i==0U?ctx[2]:static_cast<int32_t>(-scores[2U*(i-1U)]);added=token!=prev;}
      int32_t total=0;const int32_t rank=RefineBoolScan(scan,added,t,parity,total);
      if(i<n){uint32_t uid=count+static_cast<uint32_t>(rank)-(added?0U:1U);
        if(uid<capacity&&added)unique.Set(uid,token,static_cast<uint32_t>(out[2U*i+1U]));
        if (packedMap) map16[static_cast<uint32_t>(out[2U*i+1U])] = static_cast<uint16_t>(uid);
        else map[static_cast<uint32_t>(out[2U*i+1U])] = static_cast<int32_t>(uid);
      }
      asc_syncthreads();count+=static_cast<uint32_t>(total);
    }
    if(t==0U){ctx[1]=count;if(n)ctx[2]=static_cast<int32_t>(-scores[2U*(n-1U)]);if(count>capacity)ctx[0]=0;
      ctx[29]=1;ctx[6]=0;for(uint32_t r=0;r<rows;++r)if(ctx[48+r]<ctx[56+r])ctx[6]=1;}
    asc_syncthreads();
  } else if constexpr(Phase==5 || Phase==6 || Phase==9) {
    LayoutHotFinish<Phase>(a,first,end,ctx+EIGHT_CTX,ctx,scan,mapWorkspace);
  }
}
template <int Phase>
__simt_vf__ __aicore__ LAUNCH_BOUND((Phase==4?PLAN_THREADS:SORTED_THREADS)) inline void SortedStageVf(
    __ubuf__ int32_t* scanWorkspace, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* compactWorkspace,
    uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart, int64_t rowStride, int64_t numReqs,
    int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests,
    int32_t blockSize, __gm__ int32_t* rowModes, __gm__ int32_t* residentRows, __gm__ int32_t* residentBlockTable,
    uint64_t jointHashCapacity, uint64_t scratchElements, __ubuf__ int32_t* localTopk, int64_t localRow) {
  const JointPlanArgs args{reqIds,
                           lastReqIds,
                           FusionTopk{topkIndices,localTopk,localRow*topk,topk},
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
  const uint32_t thread=static_cast<uint32_t>(threadIdx.x);
  if constexpr(Phase==4) {
    if(jointHashCapacity==0U){
      for(int64_t row=rowStart;row<numReqs;row+=rowStride){
        if(false)RunPrivateRow(args,row,scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS,scanWorkspace,thread);
        else RunPrivateRow(args,row,compactWorkspace+QLI_MAP_WORDS(maxRequests,topk)+QLI_UNIQUE_GM_WORDS(maxRequests,capacity)+static_cast<uint64_t>(row)*workspaceRowElements,scanWorkspace,thread);
      }
    }else for(int64_t row=rowStart;row<rowStride;++row){
      if(false)RunPrivateRow(args,row,scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS,scanWorkspace,thread);
      else RunPrivateRow(args,row,compactWorkspace+QLI_MAP_WORDS(maxRequests,topk)+QLI_UNIQUE_GM_WORDS(maxRequests,capacity)+static_cast<uint64_t>(row)*workspaceRowElements,scanWorkspace,thread);
    }
  } else {
    __ubuf__ int32_t* ctx=scanWorkspace+PLAN_SCAN_STORAGE_ELEMENTS;
    if(scratchElements<=EIGHT_UB_WORDS){
      __ubuf__ int32_t* data=ctx+EIGHT_CTX;
      auto* packet=data+QLI_OWNER_OFFSET(args.capacity);
      if constexpr(HOT_FIXED){if(args.topk==2048U&&args.capacity==4096U){
        SortedStage<Phase,true>(args,rowStart,rowStride,data,ctx,packet,scanWorkspace,compactWorkspace);return;}}
      SortedStage<Phase,false>(args,rowStart,rowStride,data,ctx,packet,scanWorkspace,compactWorkspace);
    }
  }
}
__simd_vf__ inline void RefineUniqueMapVf(__ubuf__ int32_t* records,
    __ubuf__ int32_t* data,__ubuf__ int32_t* ctx,uint32_t n,uint32_t q,
    uint32_t capacity,uint32_t initialCount,int32_t initialLast,uint32_t more , uint32_t mapWords){
  using namespace AscendC;
  Reg::RegTensor<uint32_t> lane,zero,one,even,odd,previousIndex,lastIndex;
  Reg::RegTensor<uint32_t> flags,prefix,carry,uid,payload,index,safeIndex,writeFlag,status,value;
  Reg::RegTensor<float> score,previous,lastScore,shiftScore,positive;
  Reg::RegTensor<int32_t> token,lastToken;
  Reg::MaskReg full=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();
  Reg::MaskReg first,head,valid,fits,writeHead,chosen;
  Reg::Unsqueeze<uint32_t>(lane,full);
  Reg::Duplicate(zero,uint32_t{0});Reg::Duplicate(one,uint32_t{1});
  Reg::Duplicate(carry,initialCount);Reg::Duplicate(lastToken,initialLast);
  Reg::Duplicate(lastScore,-static_cast<float>(initialLast));
  Reg::Compares<uint32_t,CMPMODE::EQ>(first,lane,uint32_t{0},full);
  Reg::Adds(previousIndex,lane,uint32_t(-1),full);
  Reg::Select(previousIndex,zero,previousIndex,first);
  static constexpr Reg::CastTrait toInt={Reg::RegLayout::UNKNOWN,Reg::SatMode::NO_SAT,
      Reg::MaskMergeMode::ZEROING,RoundMode::CAST_TRUNC};
  for(uint32_t base=0;base<n;base+=64U){
    uint32_t remaining=n-base;valid=Reg::UpdateMask<uint32_t>(remaining);
    const uint32_t count=n-base<64U?n-base:64U;
    Reg::Adds(even,lane,base,full);Reg::ShiftLefts(even,even,int16_t{1},full);
    Reg::Adds(odd,even,uint32_t{1},full);
    Reg::Duplicate(score,0.0f);
    Reg::Gather<float,float,uint32_t>(score,reinterpret_cast<__ubuf__ float*>(records),even,valid);
    Reg::Gather<float,uint32_t>(shiftScore,score,previousIndex);
    Reg::Select(previous,lastScore,shiftScore,first);
    Reg::Compare<float,CMPMODE::NE>(head,score,previous,valid);
    Reg::Select(flags,one,zero,head);Reg::Select(flags,flags,zero,valid);
    Reg::Compares<uint32_t,CMPMODE::NE>(head,flags,uint32_t{0},full);
    Reg::Unsqueeze<uint32_t>(prefix,head);
    Reg::Add(prefix,prefix,flags,full);
    Reg::Add(prefix,prefix,carry,full);
    Reg::Adds(uid,prefix,uint32_t(-1),full);
    Reg::Neg(positive,score,full);
    Reg::Cast<int32_t,float,toInt>(token,positive,valid);
    Reg::Compares<uint32_t,CMPMODE::LT>(fits,uid,capacity,full);
    Reg::Select(writeFlag,flags,zero,fits);
    Reg::Compares<uint32_t,CMPMODE::NE>(writeHead,writeFlag,uint32_t{0},full);
    Reg::Adds(index,uid,mapWords,full);
    Reg::Adds(safeIndex,lane,mapWords+capacity,full);
    Reg::Select(index,index,safeIndex,writeHead);
    Reg::Duplicate(payload,uint32_t{0});
    Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);
    FusionStoreUnique(data,token,payload,uid,lane,writeHead,full,mapWords,capacity,initialCount);

    Reg::Adds(safeIndex,lane,q,full);Reg::Select(payload,payload,safeIndex,valid);
    if (true) {
      Reg::Adds(safeIndex,lane,q+64U,full);
      Reg::ShiftLefts(index,safeIndex,int16_t{16},full);
      Reg::Or(index,index,payload,full);
      Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data),
          reinterpret_cast<Reg::RegTensor<uint16_t>&>(uid),
          reinterpret_cast<Reg::RegTensor<uint16_t>&>(index),valid);
    } else {
      Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(data),uid,payload,valid);
    }
    Reg::Duplicate(lastIndex,uint32_t{63});Reg::Gather<uint32_t,uint32_t>(carry,prefix,lastIndex);
    Reg::Duplicate(lastIndex,count-1U);
    Reg::Gather<float,uint32_t>(lastScore,score,lastIndex);
    Reg::Gather<int32_t,uint32_t>(lastToken,token,lastIndex);
  }
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{1},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),carry,lane,chosen);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{2},full);
  Reg::Scatter<int32_t,uint32_t>(ctx,lastToken,lane,chosen);
  Reg::Compares<uint32_t,CMPMODE::GE>(fits,carry,capacity+1U,full);
  Reg::Select(status,zero,one,fits);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{0},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),status,lane,chosen);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{29},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),one,lane,chosen);
  Reg::Duplicate(value,more);
  Reg::Compares<uint32_t,CMPMODE::EQ>(chosen,lane,uint32_t{6},full);
  Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),value,lane,chosen);
}
__aicore__ inline void SortedVectorToScalar(AscendC::TPipe& pipe) {
  auto id=pipe.AllocEventID<AscendC::HardEvent::V_S>();
  AscendC::SetFlag<AscendC::HardEvent::V_S>(id);AscendC::WaitFlag<AscendC::HardEvent::V_S>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::V_S>(id);
}
__aicore__ inline void SortedScalarToVector(AscendC::TPipe& pipe) {
  auto id=pipe.AllocEventID<AscendC::HardEvent::S_V>();
  AscendC::SetFlag<AscendC::HardEvent::S_V>(id);AscendC::WaitFlag<AscendC::HardEvent::S_V>(id);
  pipe.ReleaseEventID<AscendC::HardEvent::S_V>(id);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightCopyVf(
    __ubuf__ int32_t* dst,__ubuf__ int32_t* src,uint32_t words){
  for(uint32_t i=threadIdx.x;i<words;i+=SORTED_THREADS)dst[i]=src[i];
  asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightProgressVf(
    __ubuf__ int32_t* records,__ubuf__ int32_t* ctx,uint32_t n,uint32_t topk){
  const uint32_t t=threadIdx.x;
  if(t<8U){
    uint32_t consumed=0;
    for(uint32_t i=0;i<n;++i)if(static_cast<uint32_t>(records[2U*i+1U])/topk==t)++consumed;
    ctx[48U+t]+=consumed;
  }
  asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void EightParallelProgressVf(
    __ubuf__ int32_t* records,__ubuf__ int32_t* ctx,uint32_t n,uint32_t topk){
  const uint32_t t=threadIdx.x;
  auto* partials=records-static_cast<uint32_t>(ctx[30]);
  uint32_t bins[8]={0,0,0,0,0,0,0,0};
  for(uint32_t i=t;i<n;i+=SORTED_THREADS){
    const uint32_t qid=static_cast<uint32_t>(records[2U*i+1U]);
    const uint32_t row=topk==2048U?qid>>11U:qid/topk;
    if(row>=8U)asc_atomic_min(ctx,0);
#pragma unroll
    for(uint32_t r=0;r<8U;++r)bins[r]+=row==r?1U:0U;
  }
  const uint32_t lane=t&31U,warp=t/32U;
#pragma unroll
  for(uint32_t r=0;r<8U;++r){
    uint32_t v=bins[r];
#pragma unroll
    for(uint32_t delta=16U;delta;delta>>=1U)v+=static_cast<uint32_t>(asc_shfl(static_cast<int32_t>(v),lane^delta,32));
    if(lane==0U)partials[warp*8U+r]=static_cast<int32_t>(v);
  }
  asc_syncthreads();
  if(t<8U){uint32_t total=0;
    for(uint32_t w=0;w<SORTED_WARPS;++w)total+=static_cast<uint32_t>(partials[w*8U+t]);
    ctx[48U+t]+=total;
  }
  asc_syncthreads();
}

template <uint32_t Bits, int KnownHalf>
__simd_vf__ inline void RefineFusedTuneVf(__ubuf__ int32_t *records, __ubuf__ int32_t *data,
                                          __ubuf__ int32_t *ctx, uint32_t n, uint32_t q, uint32_t capacity,
                                          uint32_t initialCount, int32_t initialLast, uint32_t topk,
                                          uint32_t knownTotal, uint32_t unknownTotal, uint32_t mapWords=8192U) {
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane, zero, one, even, odd, previousIndex, lastIndex;
    Reg::RegTensor<uint32_t> flags, prefix, carry, uid, payload, index, safeIndex, writeFlag, status, value;
    Reg::RegTensor<float> score, previous, lastScore, shiftScore, positive;
    Reg::RegTensor<int32_t> token, lastToken;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg first, head, valid, fits, writeHead, chosen;
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Duplicate(carry, initialCount);
    Reg::Duplicate(lastToken, initialLast);
    Reg::Duplicate(lastScore, -static_cast<float>(initialLast));
    Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
    Reg::Adds(previousIndex, lane, uint32_t(-1), full);
    Reg::Select(previousIndex, zero, previousIndex, first);
    static constexpr Reg::CastTrait toInt = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    static_assert(KnownHalf >= -1 && KnownHalf <= 2);
    constexpr bool PackedCursor = (Bits & 4U) != 0;
    constexpr uint32_t CountRows = KnownHalf < 0 ? 8U : KnownHalf == 2 ? 0U : 4U;
    constexpr uint32_t Counters = CountRows - (PackedCursor && CountRows ? 1U : 0U);
    Reg::RegTensor<uint32_t> bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7;
    if constexpr (0U < Counters)
        Reg::Duplicate(bin0, uint32_t{0});
    if constexpr (1U < Counters)
        Reg::Duplicate(bin1, uint32_t{0});
    if constexpr (2U < Counters)
        Reg::Duplicate(bin2, uint32_t{0});
    if constexpr (3U < Counters)
        Reg::Duplicate(bin3, uint32_t{0});
    if constexpr (4U < Counters)
        Reg::Duplicate(bin4, uint32_t{0});
    if constexpr (5U < Counters)
        Reg::Duplicate(bin5, uint32_t{0});
    if constexpr (6U < Counters)
        Reg::Duplicate(bin6, uint32_t{0});
    if constexpr (7U < Counters)
        Reg::Duplicate(bin7, uint32_t{0});
    if constexpr ((Bits & 2U) != 0) {
        for (uint32_t base = 0; base < (n & ~63U); base += 64U) {
            uint32_t remaining = 64U;
            valid = Reg::UpdateMask<uint32_t>(remaining);
            const uint32_t count = n - base < 64U ? n - base : 64U;
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(
                reinterpret_cast<Reg::RegTensor<uint32_t> &>(score), payload,
                reinterpret_cast<__ubuf__ uint32_t *>(records) + 2U * base);
            Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
            Reg::Select(previous, lastScore, shiftScore, first);
            Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
            Reg::Select(flags, one, zero, head);
            Reg::Select(flags, flags, zero, valid);
            Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
            Reg::Unsqueeze<uint32_t>(prefix, head);
            Reg::Add(prefix, prefix, flags, full);
            Reg::Add(prefix, prefix, carry, full);
            Reg::Adds(uid, prefix, uint32_t(-1), full);
            Reg::Neg(positive, score, full);
            Reg::Cast<int32_t, float, toInt>(token, positive, valid);
            Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
            Reg::Select(writeFlag, flags, zero, fits);
            Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
            Reg::Adds(index, uid, mapWords, full);
            Reg::Adds(safeIndex, lane, mapWords + capacity, full);
            Reg::Select(index, index, safeIndex, writeHead);

            FusionStoreUnique(data,token,payload,uid,lane,writeHead,full,mapWords,capacity,initialCount);
            Reg::Adds(safeIndex, lane, q, full);
            Reg::Select(payload, payload, safeIndex, valid);
            if constexpr (0U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 0U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin0, bin0, value, full);
            }
            if constexpr (1U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 1U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin1, bin1, value, full);
            }
            if constexpr (2U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 2U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin2, bin2, value, full);
            }
            if constexpr (3U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 3U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin3, bin3, value, full);
            }
            if constexpr (4U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 4U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin4, bin4, value, full);
            }
            if constexpr (5U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 5U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin5, bin5, value, full);
            }
            if constexpr (6U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 6U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin6, bin6, value, full);
            }
            if constexpr (7U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 7U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin7, bin7, value, full);
            }
            Reg::Adds(safeIndex,lane,q+64U,full);
            Reg::ShiftLefts(index,safeIndex,int16_t{16},full);
            Reg::Or(index,index,payload,full);
            Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(uid),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(index),valid);
            Reg::Duplicate(lastIndex, uint32_t{63});
            Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
            Reg::Duplicate(lastIndex, count - 1U);
            Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
            Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
        }
        for (uint32_t base = (n & ~63U); base < n; base += 64U) {
            uint32_t remaining = n - base;
            valid = Reg::UpdateMask<uint32_t>(remaining);
            const uint32_t count = n - base < 64U ? n - base : 64U;
            Reg::Adds(even, lane, base, full);
            Reg::ShiftLefts(even, even, int16_t{1}, full);
            Reg::Adds(odd, even, uint32_t{1}, full);
            Reg::Duplicate(score, 0.0f);
            Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(records), even,
                                                valid);
            Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
            Reg::Select(previous, lastScore, shiftScore, first);
            Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
            Reg::Select(flags, one, zero, head);
            Reg::Select(flags, flags, zero, valid);
            Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
            Reg::Unsqueeze<uint32_t>(prefix, head);
            Reg::Add(prefix, prefix, flags, full);
            Reg::Add(prefix, prefix, carry, full);
            Reg::Adds(uid, prefix, uint32_t(-1), full);
            Reg::Neg(positive, score, full);
            Reg::Cast<int32_t, float, toInt>(token, positive, valid);
            Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
            Reg::Select(writeFlag, flags, zero, fits);
            Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
            Reg::Adds(index, uid, mapWords, full);
            Reg::Adds(safeIndex, lane, mapWords + capacity, full);
            Reg::Select(index, index, safeIndex, writeHead);
            Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);
            FusionStoreUnique(data,token,payload,uid,lane,writeHead,full,mapWords,capacity,initialCount);
            Reg::Duplicate(payload, uint32_t{0});
            Reg::Gather<uint32_t, uint32_t, uint32_t>(payload, reinterpret_cast<__ubuf__ uint32_t *>(records),
                                                      odd, valid);
            Reg::Adds(safeIndex, lane, q, full);
            Reg::Select(payload, payload, safeIndex, valid);
            if constexpr (0U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 0U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin0, bin0, value, full);
            }
            if constexpr (1U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 1U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin1, bin1, value, full);
            }
            if constexpr (2U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 2U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin2, bin2, value, full);
            }
            if constexpr (3U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 3U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin3, bin3, value, full);
            }
            if constexpr (4U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 4U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin4, bin4, value, full);
            }
            if constexpr (5U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 5U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin5, bin5, value, full);
            }
            if constexpr (6U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 6U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin6, bin6, value, full);
            }
            if constexpr (7U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 7U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin7, bin7, value, full);
            }
            Reg::Adds(safeIndex,lane,q+64U,full);
            Reg::ShiftLefts(index,safeIndex,int16_t{16},full);
            Reg::Or(index,index,payload,full);
            Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(uid),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(index),valid);
            Reg::Duplicate(lastIndex, uint32_t{63});
            Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
            Reg::Duplicate(lastIndex, count - 1U);
            Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
            Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
        }
    } else {
        for (uint32_t base = 0; base < n; base += 64U) {
            uint32_t remaining = n - base;
            valid = Reg::UpdateMask<uint32_t>(remaining);
            const uint32_t count = n - base < 64U ? n - base : 64U;
            Reg::Adds(even, lane, base, full);
            Reg::ShiftLefts(even, even, int16_t{1}, full);
            Reg::Adds(odd, even, uint32_t{1}, full);
            Reg::Duplicate(score, 0.0f);
            Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(records), even,
                                                valid);
            Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
            Reg::Select(previous, lastScore, shiftScore, first);
            Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
            Reg::Select(flags, one, zero, head);
            Reg::Select(flags, flags, zero, valid);
            Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
            Reg::Unsqueeze<uint32_t>(prefix, head);
            Reg::Add(prefix, prefix, flags, full);
            Reg::Add(prefix, prefix, carry, full);
            Reg::Adds(uid, prefix, uint32_t(-1), full);
            Reg::Neg(positive, score, full);
            Reg::Cast<int32_t, float, toInt>(token, positive, valid);
            Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
            Reg::Select(writeFlag, flags, zero, fits);
            Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
            Reg::Adds(index, uid, mapWords, full);
            Reg::Adds(safeIndex, lane, mapWords + capacity, full);
            Reg::Select(index, index, safeIndex, writeHead);
            Reg::Gather<uint32_t,uint32_t,uint32_t>(payload,reinterpret_cast<__ubuf__ uint32_t*>(records),odd,valid);
            FusionStoreUnique(data,token,payload,uid,lane,writeHead,full,mapWords,capacity,initialCount);
            Reg::Duplicate(payload, uint32_t{0});
            Reg::Gather<uint32_t, uint32_t, uint32_t>(payload, reinterpret_cast<__ubuf__ uint32_t *>(records),
                                                      odd, valid);
            Reg::Adds(safeIndex, lane, q, full);
            Reg::Select(payload, payload, safeIndex, valid);
            if constexpr (0U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 0U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin0, bin0, value, full);
            }
            if constexpr (1U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 1U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin1, bin1, value, full);
            }
            if constexpr (2U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 2U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin2, bin2, value, full);
            }
            if constexpr (3U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 3U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin3, bin3, value, full);
            }
            if constexpr (4U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 4U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin4, bin4, value, full);
            }
            if constexpr (5U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 5U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin5, bin5, value, full);
            }
            if constexpr (6U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 6U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin6, bin6, value, full);
            }
            if constexpr (7U < Counters) {
                constexpr uint32_t Boundary = (KnownHalf == 0 ? 5U : 1U) + 7U;
                Reg::Compares<uint32_t, CMPMODE::LT>(chosen, payload, Boundary * topk, full);
                Reg::Select(value, one, zero, chosen);
                Reg::Add(bin7, bin7, value, full);
            }
            Reg::Adds(safeIndex,lane,q+64U,full);
            Reg::ShiftLefts(index,safeIndex,int16_t{16},full);
            Reg::Or(index,index,payload,full);
            Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(uid),
                reinterpret_cast<Reg::RegTensor<uint16_t>&>(index),valid);
            Reg::Duplicate(lastIndex, uint32_t{63});
            Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
            Reg::Duplicate(lastIndex, count - 1U);
            Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
            Reg::Gather<int32_t, uint32_t>(lastToken, token, lastIndex);
        }
    }
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{1}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), carry, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{2}, full);
    Reg::Scatter<int32_t, uint32_t>(ctx, lastToken, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::GE>(fits, carry, capacity + 1U, full);
    Reg::Select(status, zero, one, fits);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{0}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), status, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{29}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), one, lane, chosen);
    Reg::RegTensor<uint32_t> reduced, previousCount, cumulative, delta, oldCursor, endCursor;
    Reg::RegTensor<uint32_t> knownCounts, allDeltas, updated, knownOffset;
    Reg::Duplicate(previousCount, uint32_t{0});
    Reg::Duplicate(even, uint32_t{0});
    Reg::Duplicate(allDeltas, uint32_t{0});
    uint32_t eight = 8U;
    valid = Reg::UpdateMask<uint32_t>(eight);
    if constexpr (KnownHalf >= 0) {
        Reg::Duplicate(knownCounts, uint32_t{0});
        Reg::Adds(index, lane, 112U, full);
        Reg::Select(index, index, lane, valid);
        Reg::Gather<uint32_t, uint32_t, uint32_t>(knownCounts, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, valid);
    }
    Reg::Duplicate(knownOffset, KnownHalf == 0 ? knownTotal : 0U);
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 0U : 0U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{0}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 48U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{1});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 1U : 1U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{1}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 49U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{2});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 2U : 2U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{2}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 50U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 0) {
        Reg::Duplicate(lastIndex, uint32_t{3});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 3U : 3U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{3}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 51U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{4});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 4U : 0U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{4}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 52U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{5});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 5U : 1U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{5}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 53U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{6});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 6U : 2U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{6}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 54U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (KnownHalf == 2 || KnownHalf == 1) {
        Reg::Duplicate(lastIndex, uint32_t{7});
        Reg::Gather<uint32_t, uint32_t>(delta, knownCounts, lastIndex);
    } else {
        constexpr uint32_t J = KnownHalf < 0 ? 7U : 3U;
        if constexpr (J == CountRows - 1U && PackedCursor) {
            Reg::Duplicate(cumulative, KnownHalf < 0 ? n : unknownTotal);
        } else {
            if constexpr (J == 0U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin0, full);
            if constexpr (J == 1U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin1, full);
            if constexpr (J == 2U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin2, full);
            if constexpr (J == 3U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin3, full);
            if constexpr (J == 4U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin4, full);
            if constexpr (J == 5U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin5, full);
            if constexpr (J == 6U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin6, full);
            if constexpr (J == 7U)
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, bin7, full);
            Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
            if constexpr (KnownHalf == 0)
                Reg::Sub(cumulative, cumulative, knownOffset, full);
        }
        Reg::Sub(delta, cumulative, previousCount, full);
        previousCount = cumulative;
    }
    if constexpr (PackedCursor) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{7}, full);
        Reg::Select(allDeltas, delta, allDeltas, chosen);
    } else {
        Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
        Reg::Adds(index, lane, 55U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, first);
        Reg::Add(oldCursor, oldCursor, delta, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, first);
    }
    if constexpr (PackedCursor) {
        Reg::Adds(index, lane, 48U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, valid);
        Reg::Add(oldCursor, oldCursor, allDeltas, full);
        Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), oldCursor, index, valid);
    } else {
        Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
        Reg::Adds(index, lane, 48U, full);
        Reg::Duplicate(oldCursor, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(oldCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx),
                                                  index, valid);
    }
    Reg::Adds(index, lane, 56U, full);
    Reg::Duplicate(endCursor, uint32_t{0});
    Reg::Gather<uint32_t, uint32_t, uint32_t>(endCursor, reinterpret_cast<__ubuf__ uint32_t *>(ctx), index,
                                              valid);
    Reg::Compare<uint32_t, CMPMODE::LT>(chosen, oldCursor, endCursor, valid);
    Reg::Select(value, one, zero, chosen);
    Reg::Select(value, value, zero, valid);
    Reg::Reduce<Reg::ReduceType::SUM>(reduced, value, full);
    Reg::Gather<uint32_t, uint32_t>(cumulative, reduced, even);
    Reg::Compares<uint32_t, CMPMODE::NE>(chosen, cumulative, uint32_t{0}, full);
    Reg::Select(value, one, zero, chosen);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{6}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(ctx), value, lane, chosen);
}

__simd_callee__ inline void LayoutPut(__ubuf__ int32_t* ctx,
    AscendC::Reg::RegTensor<uint32_t>& value,AscendC::Reg::RegTensor<uint32_t>& lane,
    uint32_t at,AscendC::Reg::MaskReg& full){
  AscendC::Reg::MaskReg chosen;AscendC::Reg::Compares<uint32_t,AscendC::CMPMODE::EQ>(chosen,lane,at,full);
  AscendC::Reg::Scatter<uint32_t,uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctx),value,lane,chosen);
}
__simd_vf__ inline void LayoutHotCompactVf(__ubuf__ int32_t* data,__ubuf__ int32_t* ctx,
    uint32_t c,uint32_t u){
  using namespace AscendC;
  const uint32_t ow=QLI_OWNER_OFFSET(c),pa=ow+c;
  Reg::RegTensor<uint32_t> lane,zero,one,idx,oddBit,packed,high,f,mask,slot,invalid;
  Reg::RegTensor<uint32_t> hf,ef,hp,ep,hc,ec,hi,ei,last,entry,safe,tmp,mc,rank,inc,owner,index16,status;
  Reg::RegTensor<int32_t> signedOwner;
  Reg::MaskReg full=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>(),odd,valid,hit,evict,selected,fits;
  Reg::Unsqueeze<uint32_t>(lane,full);Reg::Duplicate(zero,uint32_t{0});Reg::Duplicate(one,uint32_t{1});
  Reg::Duplicate(hc,uint32_t{0});Reg::Duplicate(ec,uint32_t{0});Reg::Duplicate(last,uint32_t{63});
  Reg::Duplicate(mask,uint32_t{1});Reg::And(oddBit,lane,mask,full);Reg::Compares<uint32_t,CMPMODE::NE>(odd,oddBit,uint32_t{0},full);
  for(uint32_t base=0;base<c;base+=64U){
    Reg::LoadAlign<uint32_t,Reg::LoadDist::DIST_UNPACK_B16>(f,reinterpret_cast<__ubuf__ uint32_t*>(data)+(base/2U));
    Reg::Compares<uint32_t,CMPMODE::NE>(valid,f,uint32_t{65535},full);
    Reg::Duplicate(mask,uint32_t{0x4000});Reg::And(tmp,f,mask,full);Reg::Compares<uint32_t,CMPMODE::NE>(hit,tmp,uint32_t{0},valid);
    Reg::Select(hf,one,zero,hit);Reg::Select(ef,one,zero,valid);Reg::Sub(ef,ef,hf,full);
    Reg::Compares<uint32_t,CMPMODE::NE>(evict,ef,uint32_t{0},full);
    Reg::Unsqueeze<uint32_t>(hp,hit);
    Reg::Add(hi,hp,hc,full);
    Reg::Adds(ei,lane,base,full);Reg::Sub(ei,ei,hi,full);
    Reg::Duplicate(tmp,c-1U);Reg::Sub(idx,tmp,hi,full);Reg::Select(idx,idx,ei,hit);
    Reg::Duplicate(mask,uint32_t{0x3fff});Reg::And(entry,f,mask,full);
    Reg::Duplicate(mask,uint32_t{0x8000});Reg::And(invalid,f,mask,full);Reg::ShiftRights(invalid,invalid,int16_t{1},full);Reg::Or(entry,entry,invalid,full);
    Reg::Adds(idx,idx,2U*c,full);Reg::Select(idx,idx,lane,valid);
    Reg::Adds(safe,lane,uint32_t{64},full);Reg::ShiftLefts(index16,safe,int16_t{16},full);Reg::Or(index16,index16,idx,full);
    Reg::Scatter<uint16_t,uint16_t>(reinterpret_cast<__ubuf__ uint16_t*>(data+ow),
      reinterpret_cast<Reg::RegTensor<uint16_t>&>(entry),reinterpret_cast<Reg::RegTensor<uint16_t>&>(index16),valid);
    Reg::Add(hi,hi,hf,full);Reg::Gather<uint32_t,uint32_t>(hc,hi,last);
    Reg::Duplicate(tmp,base+64U);Reg::Sub(ec,tmp,hc,full);
  }
  Reg::LocalMemBar<Reg::MemType::VEC_ALL,Reg::MemType::VEC_ALL>();
  Reg::Duplicate(mc,uint32_t{0});
  for(uint32_t base=0;base<u;base+=64U){
    uint32_t remaining=u-base;valid=Reg::UpdateMask<uint32_t>(remaining);
    Reg::LoadAlign<int32_t>(signedOwner,data+ow+base);
    Reg::Compares<int32_t,CMPMODE::LT>(selected,signedOwner,int32_t{0},valid);
    Reg::Select(inc,one,zero,selected);Reg::Unsqueeze<uint32_t>(rank,selected);Reg::Add(rank,rank,mc,full);
    Reg::Compare<uint32_t,CMPMODE::LT>(fits,rank,ec,selected);
    Reg::ShiftRights(idx,rank,int16_t{1},full);Reg::Adds(idx,idx,pa,full);
    Reg::Duplicate(tmp,pa);Reg::Select(idx,idx,tmp,fits);Reg::Duplicate(packed,uint32_t{0});
    Reg::Gather<uint32_t,uint32_t,uint32_t>(packed,reinterpret_cast<__ubuf__ uint32_t*>(data),idx,fits);
    Reg::Duplicate(mask,uint32_t{1});Reg::And(tmp,rank,mask,full);Reg::Compares<uint32_t,CMPMODE::NE>(odd,tmp,uint32_t{0},full);
    Reg::ShiftRights(high,packed,int16_t{16},full);Reg::Select(slot,high,packed,odd);
    Reg::Duplicate(mask,uint32_t{0x3fff});Reg::And(slot,slot,mask,full);
    Reg::ShiftLefts(owner,rank,int16_t{16},full);Reg::Or(owner,owner,slot,full);Reg::Duplicate(mask,uint32_t{0x40000000});Reg::Or(owner,owner,mask,full);
    Reg::StoreAlign<uint32_t,Reg::StoreDist::DIST_NORM_B32>(reinterpret_cast<__ubuf__ uint32_t*>(data)+ow+base,owner,fits);
    Reg::Add(tmp,rank,inc,full);Reg::Gather<uint32_t,uint32_t>(mc,tmp,last);
  }
  Reg::Add(tmp,ec,hc,full);Reg::Compares<uint32_t,CMPMODE::EQ>(valid,tmp,c,full);
  Reg::Adds(tmp,ec,uint32_t{1},full);Reg::Compare<uint32_t,CMPMODE::LT>(selected,mc,tmp,valid);
  Reg::Select(status,one,zero,selected);
  LayoutPut(ctx,ec,lane,43,full);LayoutPut(ctx,hc,lane,44,full);LayoutPut(ctx,mc,lane,45,full);LayoutPut(ctx,status,lane,0,full);
}

__aicore__ inline void RankPipeCopyOutputs(AscendC::TPipe& pipe,
    AscendC::LocalTensor<int32_t> local,__ubuf__ int32_t* ctx,
    int64_t first,uint32_t topk,uint32_t c,__gm__ uint8_t* lruSlots,__gm__ uint8_t* missTokens,__gm__ uint8_t* missSlots){
  const uint32_t n=static_cast<uint32_t>(ctx[46]),misses=static_cast<uint32_t>(ctx[47]);
  const uint32_t uniqueOffset=0U; // map is dead: bounded descriptor-tail staging
  const uint32_t ownerOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+QLI_OWNER_OFFSET(c);
  AscendC::GlobalTensor<int32_t> lru,mt,ms;
  lru.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lruSlots)+first*c,c);
  const int64_t tail=first*static_cast<int64_t>(topk)+misses;
  mt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missTokens)+tail,n);
  ms.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missSlots)+tail,n);
  auto ready=pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ready);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ready);
  pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(ready);
  AscendC::DataCopyExtParams lruCopy(1,c*4U,0,0,0);
  AscendC::DataCopyPad(lru,local[ownerOffset],lruCopy);
  for(uint32_t offset=0;offset<n;offset+=2048U){
    const uint32_t count=n-offset<2048U?n-offset:2048U;
    AscendC::DataCopyExtParams tailCopy(1,count*4U,0,0,0);
    mt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missTokens)+tail+offset,count);
    ms.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missSlots)+tail+offset,count);
    AscendC::DataCopyPad(mt,local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX],tailCopy);
    AscendC::DataCopyPad(ms,local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX],tailCopy);
  }
  auto done=pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(done);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(done);
  pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(done);
}
// ABI retained from the pinned ordered Indexer hook; source-page validation is
// not reintroduced into Plan. QLI's own block_table remains fully operational.
struct TilingData {
  uint64_t hashCapacity,workspaceRowElements;
  int64_t maxRows,topk,capacity,maxToken,maxRequests,maxNumBlocks,hostNumBlocks;
  int32_t blockSize;uint32_t localMemoryBytes;
  uint64_t jointHashCapacity,scratchElements;uint32_t maxJointRows;
};
struct Buffers {
  GM_ADDR reqIds;GM_ADDR topkIndices;GM_ADDR stablePrefixLens;GM_ADDR visibleSeqLens;
  GM_ADDR tokenToReq;GM_ADDR blockTable;GM_ADDR activeRows;GM_ADDR lastReqIds;
  GM_ADDR slotToToken;GM_ADDR lruSlots;GM_ADDR currentSlots;GM_ADDR missCount;
  GM_ADDR missTokens;GM_ADDR missSlots;GM_ADDR compactWorkspace;
  GM_ADDR rowModes;GM_ADDR residentRows;GM_ADDR residentBlockTable;GM_ADDR readyCounts;
};
// C flushes at most 2048 newly unique records per pass. Importantly, this is
// DURING merge, not a spill after materializing a capacity-sized UB unique list.
__aicore__ inline void FusionRefinePacket(__ubuf__ int32_t* records,
    __ubuf__ int32_t* data,__ubuf__ int32_t* ctx,uint32_t n,uint32_t q,
    uint32_t c,uint32_t topk,uint32_t more,bool fused,int64_t first,
    __gm__ int32_t* workspace,int64_t requests,AscendC::TPipe& pipe,AscendC::LocalTensor<int32_t> local){
  const uint32_t step=QLI_PLAN_STORAGE==3?2048U:n;
  if(n==0U)return;
  for(uint32_t base=0;base<n;base+=step){
    const uint32_t count=n-base<step?n-base:step,old=ctx[1];
    const uint32_t packetMore=base+count<n?1U:more;
    if(fused){
      if constexpr(QLI_PLAN_STORAGE!=3){
        if(ctx[108]==2)asc_vf_call<RefineFusedTuneVf<3U,2>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,ctx[109],ctx[110],c/2U);
        else if(ctx[108]==0)asc_vf_call<RefineFusedTuneVf<3U,0>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,ctx[109],ctx[110],c/2U);
        else if(ctx[108]==1)asc_vf_call<RefineFusedTuneVf<3U,1>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,ctx[110],ctx[109],c/2U);
        else asc_vf_call<RefineFusedTuneVf<3U,-1>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,0U,0U,c/2U);
      }else{
        asc_vf_call<RefineFusedTuneVf<3U,-1>>(records+2U*base,data,ctx,count,q,c,old,ctx[2],topk,0U,0U,c/2U);
      }
    }else asc_vf_call<RefineUniqueMapVf>(records+2U*base,data,ctx,count,q,c,old,ctx[2],packetMore,c/2U);
    SortedVectorToScalar(pipe);
    if(ctx[0]!=1)return;
    if constexpr(QLI_PLAN_STORAGE==3){
      const uint32_t delta=static_cast<uint32_t>(ctx[1])-old;
      if(delta){
        auto source=local[PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+c/2U];
        AscendC::GlobalTensor<int32_t> target;
        target.SetGlobalBuffer(workspace+QLI_MAP_WORDS(requests,topk)+first*c+old,delta);
        auto event=pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopyExtParams copy(1,delta*4U,0,0,0);AscendC::DataCopyPad(target,source,copy);
        // WAR: this bounded staging will be overwritten by the next VF.
        event=pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(event);
      }
    }
  }
}
__aicore__ inline void RunFusionGroupImpl(const Buffers& b,const TilingData& d,
    int64_t first,int64_t end,int64_t rows,AscendC::TPipe& pipe,
    AscendC::LocalTensor<int32_t> local,__ubuf__ int32_t* localTopk,int64_t localRow){
  auto storage=local;
  auto* scan=reinterpret_cast<__ubuf__ int32_t*>(local.GetPhyAddr());
  auto* ctx=scan+PLAN_SCAN_STORAGE_ELEMENTS;
  auto reqIds=b.reqIds,topkIndices=b.topkIndices,stablePrefixLens=b.stablePrefixLens,visibleSeqLens=b.visibleSeqLens;
  auto tokenToReq=b.tokenToReq,lastReqIds=b.lastReqIds,slotToToken=b.slotToToken,lruSlots=b.lruSlots;
  auto currentSlots=b.currentSlots,missCount=b.missCount,missTokens=b.missTokens,missSlots=b.missSlots;
  auto compactWorkspace=b.compactWorkspace,rowModes=b.rowModes,residentRows=b.residentRows,residentBlockTable=b.residentBlockTable;
#define SORTED_CALL(PHASE) asc_vf_call<SortedStageVf<PHASE>>(dim3((PHASE)==4?PLAN_THREADS:SORTED_THREADS), scan, reinterpret_cast<__gm__ int64_t*>(reqIds), reinterpret_cast<__gm__ int64_t*>(lastReqIds), \
      reinterpret_cast<__gm__ int32_t*>(topkIndices),reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), \
      reinterpret_cast<__gm__ int32_t*>(visibleSeqLens),reinterpret_cast<__gm__ int32_t*>(tokenToReq), \
      reinterpret_cast<__gm__ int32_t*>(slotToToken),reinterpret_cast<__gm__ int32_t*>(lruSlots), \
      reinterpret_cast<__gm__ int32_t*>(currentSlots),reinterpret_cast<__gm__ int32_t*>(missCount), \
      reinterpret_cast<__gm__ int32_t*>(missTokens),reinterpret_cast<__gm__ int32_t*>(missSlots), \
      reinterpret_cast<__gm__ int32_t*>(compactWorkspace),d.hashCapacity,d.workspaceRowElements, \
      first,end,rows,d.topk,d.capacity,d.maxToken,d.maxRequests,d.blockSize, \
      reinterpret_cast<__gm__ int32_t*>(rowModes),reinterpret_cast<__gm__ int32_t*>(residentRows), \
      reinterpret_cast<__gm__ int32_t*>(residentBlockTable),d.jointHashCapacity,d.scratchElements,localTopk,localRow)
      SORTED_CALL(0);SortedVectorToScalar(pipe);
      while(ctx[0]==1&&(!HW_A||ctx[6])){
        SORTED_CALL(1);SortedVectorToScalar(pipe);if(ctx[28]==0)break;
#if JOINT_OPT_MASK >= 39
        if(end-first>4 && ctx[28]>4){
          const uint32_t chunk=ctx[3];
#if JOINT_MTP7_MAP16
          const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+
              QLI_OWNER_OFFSET(d.capacity);
#else
          const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+
              ((static_cast<uint32_t>(end-first)*d.topk+d.capacity+7U)&~7U);
#endif
          auto floats=storage.ReinterpretCast<float>();auto* packet=scan+inputOffset;
          uint32_t lengths[2]={0,0}; bool allComplete[2]={true,true};
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
          ctx[108]=-1; //0=completeA,1=completeB,2=both; -1=general8-row histogram
#endif
          for(uint32_t half=0;half<2U;++half){
            const uint32_t begin=half*4U, count=(static_cast<uint32_t>(ctx[28])-begin<4U?static_cast<uint32_t>(ctx[28])-begin:4U);
            uint16_t lens[4]={0,0,0,0};uint32_t used[4]={0,0,0,0};
            for(uint32_t k=0;k<count;++k){
              lens[k]=static_cast<uint16_t>(ctx[72+begin+k]);
#if !JOINT_MTP7_PREFIX
              if(ctx[72+begin+k]!=ctx[56+ctx[64+begin+k]]-ctx[48+ctx[64+begin+k]])allComplete[half]=false;
#endif
            }
#if JOINT_MTP7_REUSE
            const uint32_t mid=half==0U?16U*chunk:0U;
#else
            const uint32_t mid=16U*chunk+half*8U*chunk;
#endif
            if(count==1U){
              asc_vf_call<EightCopyVf>(dim3(SORTED_THREADS),packet+mid,packet+2U*begin*chunk,2U*lens[0]);
              lengths[half]=lens[0];
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
              if(ctx[28]==8)ctx[112+begin]=lens[0];
#endif
              SortedVectorToScalar(pipe);
            }else{
              AscendC::MrgSortSrcList<float> lists{floats[inputOffset+2U*begin*chunk],floats[inputOffset+2U*(begin+1U)*chunk],
                floats[inputOffset+2U*(begin+(count>2?2U:0U))*chunk],floats[inputOffset+2U*(begin+(count>3?3U:0U))*chunk]};
              auto dst=floats[inputOffset+mid];
              if(allComplete[half])AscendC::MrgSort<float,false>(dst,lists,lens,used,static_cast<uint16_t>((1U<<count)-1U),1);
              else AscendC::MrgSort<float,true>(dst,lists,lens,used,static_cast<uint16_t>((1U<<count)-1U),1);
              SortedVectorToScalar(pipe);
              for(uint32_t k=0;k<count;++k){
                if(allComplete[half])used[k]=lens[k];
                if(used[k]>lens[k])ctx[0]=0;
                lengths[half]+=used[k];
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
                if(ctx[28]==8)ctx[112+begin+k]=static_cast<int32_t>(used[k]);
#endif
              }
            }
          }
#if JOINT_MTP7_MID_UNIQUE
          // Compact both four-way prefixes without dropping any query's association.
          asc_vf_call<EightMidUniqueVf>(packet,ctx+EIGHT_CTX,ctx,
              static_cast<uint32_t>(end-first)*static_cast<uint32_t>(d.topk),chunk,lengths[0],lengths[1]);
          SortedVectorToScalar(pipe);
          lengths[0]=static_cast<uint32_t>(ctx[104]);lengths[1]=static_cast<uint32_t>(ctx[105]);
#endif
#if JOINT_MTP7_OPT == 2
          if(!lengths[0]||!lengths[1])ctx[0]=0;
          if(ctx[0]){
            ctx[104]=lengths[0];ctx[105]=lengths[1];
            ctx[106]=(allComplete[0]?1U:0U)|(allComplete[1]?2U:0U);
            SortedScalarToVector(pipe);
            asc_vf_call<EightRankMergeVf>(dim3(SORTED_THREADS),packet,ctx,chunk,static_cast<uint32_t>(d.topk));
            SortedVectorToScalar(pipe);
          }
#else
          uint16_t lens[4]={static_cast<uint16_t>(lengths[0]),static_cast<uint16_t>(lengths[1]),0,0};
          uint32_t used[4]={0,0,0,0};
          if(!lengths[0]||!lengths[1])ctx[0]=0;
          if(ctx[0]){
#if JOINT_MTP7_MID_UNIQUE
            AscendC::MrgSortSrcList<float> lists{floats[inputOffset+24U*chunk],floats[inputOffset+32U*chunk],floats[inputOffset+24U*chunk],floats[inputOffset+24U*chunk]};
            auto dst=floats[inputOffset];
#elif JOINT_MTP7_REUSE
            AscendC::MrgSortSrcList<float> lists{floats[inputOffset+16U*chunk],floats[inputOffset],floats[inputOffset+16U*chunk],floats[inputOffset+16U*chunk]};
            auto dst=floats[inputOffset+24U*chunk];
#else
            AscendC::MrgSortSrcList<float> lists{floats[inputOffset+16U*chunk],floats[inputOffset+24U*chunk],floats[inputOffset+16U*chunk],floats[inputOffset+16U*chunk]};
            auto dst=floats[inputOffset+32U*chunk];
#endif
            if(allComplete[0]&&allComplete[1]){
              AscendC::MrgSort<float,false>(dst,lists,lens,used,3,1);used[0]=lens[0];used[1]=lens[1];
            }else AscendC::MrgSort<float,true>(dst,lists,lens,used,3,1);
            SortedVectorToScalar(pipe);
            if(used[0]>lens[0]||used[1]>lens[1]||used[0]+used[1]==0U)ctx[0]=0;
            else{
              ctx[4]=used[0]+used[1];
#if (JOINT_MTP7_FUSED_TUNE_BITS & 1)
              // Only full8 active sources have contiguous row0..3 / row4..7 halves.
              // A partial second-layer prefix MUST NOT inherit first-layer counts.
              if(ctx[28]==8){
                const bool fullA=used[0]==lens[0],fullB=used[1]==lens[1];
                ctx[108]=fullA?(fullB?2:0):(fullB?1:-1);
                ctx[109]=static_cast<int32_t>(used[0]);ctx[110]=static_cast<int32_t>(used[1]);
              }
#endif
#if JOINT_MTP7_MID_UNIQUE || JOINT_MTP7_FUSED_COUNT
              ctx[107]=1; // the following SIMD/map stage will publish original-query consumption
#elif JOINT_MTP7_PREFIX
              uint32_t expected = 0U;
              for (uint32_t k=0; k<static_cast<uint32_t>(ctx[28]); ++k) {
                const uint32_t row=ctx[64U+k], n=ctx[72U+k];
                ctx[48U+row] += n;
                expected += n;
              }
              if (expected != static_cast<uint32_t>(ctx[4])) ctx[0]=0;
#else
              SortedScalarToVector(pipe);
#if JOINT_MTP7_OPT >= 1
#if JOINT_MTP7_REUSE
              asc_vf_call<EightParallelProgressVf>(dim3(SORTED_THREADS),packet+24U*chunk,ctx,static_cast<uint32_t>(ctx[4]),static_cast<uint32_t>(d.topk));
#else
              asc_vf_call<EightParallelProgressVf>(dim3(SORTED_THREADS),packet+32U*chunk,ctx,static_cast<uint32_t>(ctx[4]),static_cast<uint32_t>(d.topk));
#endif
#else
              asc_vf_call<EightProgressVf>(dim3(SORTED_THREADS),packet+32U*chunk,ctx,static_cast<uint32_t>(ctx[4]),static_cast<uint32_t>(d.topk));
#endif
              SortedVectorToScalar(pipe);
#endif
            }
          }
#endif
          SortedScalarToVector(pipe);
        }else
        if(HW_A&&ctx[28]==1){
          ctx[4]=ctx[72];ctx[80]=ctx[72];ctx[48+ctx[64]]+=ctx[72];ctx[7]=1;
          SortedScalarToVector(pipe);
        }else
        if(ctx[28]>1){
          const uint32_t chunk=ctx[3];
#if JOINT_MTP7_MAP16
          const uint32_t inputOffset=d.scratchElements<=EIGHT_UB_WORDS?
              PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+QLI_OWNER_OFFSET(d.capacity):PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX;
#else
          const uint32_t inputOffset=d.scratchElements<=EIGHT_UB_WORDS?
              PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+((((HW_C?static_cast<uint32_t>(end-first):8U)*d.topk+d.capacity+7U)&~7U)):PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX;
#endif
          auto floats=storage.ReinterpretCast<float>();auto out=floats[inputOffset+static_cast<uint32_t>(ctx[30])];
          AscendC::MrgSortSrcList<float> lists{floats[inputOffset],floats[inputOffset+2U*chunk],
              floats[inputOffset+(ctx[28]>2?4U*chunk:0U)],floats[inputOffset+(ctx[28]>3?6U*chunk:0U)]};
          uint16_t lens[4]={0,0,0,0};uint32_t consumed[4]={0,0,0,0};
          for(int k=0;k<ctx[28];++k)lens[k]=static_cast<uint16_t>(ctx[72+k]);
          bool complete=HW_C;
#if JOINT_MTP7_PREFIX
          if (end-first != 8) {
          for(int k=0;k<ctx[28];++k)if(ctx[72+k]!=ctx[56+ctx[64+k]]-ctx[48+ctx[64+k]])complete=false;
          }
#else
          for(int k=0;k<ctx[28];++k)if(ctx[72+k]!=ctx[56+ctx[64+k]]-ctx[48+ctx[64+k]])complete=false;
#endif
          if(complete){
            AscendC::MrgSort<float,false>(out,lists,lens,consumed,static_cast<uint16_t>((1U<<ctx[28])-1U),1);
            for(int k=0;k<ctx[28];++k)consumed[k]=lens[k];
          }else AscendC::MrgSort<float,true>(out,lists,lens,consumed,static_cast<uint16_t>((1U<<ctx[28])-1U),1);
          SortedVectorToScalar(pipe);
          uint32_t total=0;
          for(int k=0;k<ctx[28];++k){if(consumed[k]>lens[k]){ctx[0]=0;break;}total+=consumed[k];ctx[80+k]=consumed[k];ctx[48+ctx[64+k]]+=consumed[k];}
          if(total==0U)ctx[0]=0;ctx[4]=static_cast<int32_t>(total);
          SortedScalarToVector(pipe);
        }else { SORTED_CALL(2); }
#else
        SORTED_CALL(2);
#endif
#if (JOINT_REFINE_BITS & 8)
          if(d.scratchElements<=EIGHT_UB_WORDS && ctx[0]==1){
            const uint32_t q=static_cast<uint32_t>(end-first)*static_cast<uint32_t>(d.topk);
#if JOINT_MTP7_MAP16
            const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+QLI_OWNER_OFFSET(d.capacity);
#else
            const uint32_t inputOffset=PLAN_SCAN_STORAGE_ELEMENTS+EIGHT_CTX+((q+d.capacity+7U)&~7U);
#endif
            auto* records=scan+inputOffset+(ctx[7]?0U:static_cast<uint32_t>(ctx[30]));
            uint32_t more=0;for(int64_t r=0;r<end-first;++r)if(ctx[48+r]<ctx[56+r])more=1;          FusionRefinePacket(records,ctx+EIGHT_CTX,ctx,static_cast<uint32_t>(ctx[4]),q,
            static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(d.topk),more,ctx[107]!=0,
            first,reinterpret_cast<__gm__ int32_t*>(compactWorkspace),d.maxRequests,pipe,local);
          }else {SORTED_CALL(3);}

#else
        SORTED_CALL(3);
#endif
        SortedVectorToScalar(pipe);
      }
      if(ctx[0]==1){
        if constexpr(JOINT_LAYOUT16==1 && (JOINT_LAYOUT_HOT_BITS&2U)){
          SORTED_CALL(6);SortedVectorToScalar(pipe);
          if(ctx[0]==1){
#if JOINT_RANKPIPE && (JOINT_RANKPIPE_BITS & 2)
            asc_vf_call<RankPipeCompactVf>(dim3(SORTED_THREADS),ctx+EIGHT_CTX,ctx,scan,static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(ctx[1]));
#else
            asc_vf_call<LayoutHotCompactVf>(ctx+EIGHT_CTX,ctx,static_cast<uint32_t>(d.capacity),static_cast<uint32_t>(ctx[1]));
#endif
            SortedVectorToScalar(pipe);
            if(ctx[0]==1){SORTED_CALL(9);SortedVectorToScalar(pipe);}
          }
        }else{SORTED_CALL(5);SortedVectorToScalar(pipe);}
      }
#if JOINT_RANKPIPE && (JOINT_RANKPIPE_BITS & 4)
      // Inspect completion BEFORE fallback: the private path reuses the entire
      // ctx/scratch region, so its bytes are no longer rank-pipeline metadata.
      if(ctx[0]==2 && ctx[120]==1)
        RankPipeCopyOutputs(pipe,local,ctx,first,static_cast<uint32_t>(d.topk),static_cast<uint32_t>(d.capacity),reinterpret_cast<__gm__ uint8_t*>(lruSlots),reinterpret_cast<__gm__ uint8_t*>(missTokens),reinterpret_cast<__gm__ uint8_t*>(missSlots));
#endif
      if(ctx[0]!=2){SORTED_CALL(4);}
#undef SORTED_CALL
}


__aicore__ inline void RunGroup(const Buffers& b,const TilingData& d,int64_t first,int64_t end,int64_t rows,
    AscendC::TPipe& pipe,AscendC::LocalTensor<int32_t> storage,__ubuf__ int32_t* localTopk,int64_t localRow){
  if(first<0||end<=first||end>rows)return;
  RunFusionGroupImpl(b,d,first,end,rows,pipe,storage,localTopk,localRow);
}
} // namespace qli_fused_plan
