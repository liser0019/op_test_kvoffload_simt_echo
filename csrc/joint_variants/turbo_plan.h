/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
// Included after options.h inside the kernel's anonymous namespace when JOINT_OPT_MASK == 17.
// High-performance MTP Joint Plan ("turbo"):
// 1. Fast group discovery: register-cached tokenToReq[row] + early-break bound once remaining rows
//    cannot reach this AIV core's next assigned group.
// 2. FastGroupSource: caches request identity, clamps causal visibility to [0, maxToken] for a
//    single unsigned bound check (source page-table validity is guaranteed by upstream), and
//    replaces 64-bit division in residentBlockTable page calculation with a 32-bit unsigned shift
//    when blockSize is a power of two.
// 3. Dedicated queryBuckets (UB at production shape; GM when scratch exceeds budget)
//    + active-bucket partition[capacity] packing:
//    - Scratch layout: keys[jointHashCapacity], bucketSlots[jointHashCapacity], partition[capacity],
//      queryBuckets[scratchElements - 2 * jointHashCapacity - capacity].
//    - Both cold (reset) and warm (!reset) paths store per-query savedBucket in UB queryBuckets
//      during Stage 0 and read queryBuckets + bucketSlots in Stage 4 to write final slots to GM
//      currentSlots once (eliminating the Stage 0 -> Stage 4 GM currentSlots write+read round-trip
//      and high/low 16-bit packing on bucketSlots).
//    - On warm steps (!reset), high 16 bits of partition[uid] record the exact U active hash buckets
//      (eliminating the blind 8192-bucket Stage 2 scan), while low 16 bits of partition[...] hold
//      Stage 1 evict/hit slots.
// 4. Cold/mode-reset Joint fast path: skips Stage 1 (LRU scan) and Stage 2 (Miss scan) completely
//    when reset is true (0 Stage 1/2 scans instead of 12 scans).
// 5. Single-barrier ping-pong TurboExclusiveScan + FinishTileTurbo: alternates WARP_TOTALS_OFFSET and
//    WARP_BASES_OFFSET across consecutive tiles and computes warpBase/tileTotal via warp shuffles,
//    trading per-warp shuffle instructions for 1 fewer internal barrier per tile.
// 6. Private Stage 5/6 UB token forwarding: packs (bucket << 16) | position into missPositions so
//    Stage 6 reads missed tokens directly from UB hashKeys[bucket] instead of re-reading GM topkIndices.

struct JointPlanArgs {
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

// Single-barrier ping-pong block exclusive scan:
// Alternates between WARP_TOTALS_OFFSET and WARP_BASES_OFFSET across consecutive tiles so the
// next tile's warp-total write never aliases the current tile's warp-total read, eliminating
// PublishWarpBases and cutting internal scan barriers from 2 to 1 per tile.
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

// Each tile's TurboExclusiveScan alternates warpTotals buffers, so only the final tile of a stage
// requires a post-scatter barrier before downstream stages read the scattered UB array.
__simt_callee__ __aicore__ inline void FinishTileTurbo(uint32_t tile, uint32_t extent) {
  if (tile + PLAN_THREADS >= extent) {
    asc_syncthreads();
  }
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline int32_t InsertJointToken(ScratchPtr keys, int32_t token, uint32_t size,
                                                           int32_t& savedBucket) {
  uint32_t bucket = HashToken(token, size);
  const uint32_t mask = size - 1U;
  for (uint32_t probe = 0; probe < size; ++probe) {
    const int32_t old = asc_atomic_cas(keys + bucket, HASH_EMPTY, token);
    if (old == token || old == HASH_EMPTY) {
      savedBucket = static_cast<int32_t>(bucket);
      return old == HASH_EMPTY ? 1 : 0;
    }
    bucket = (bucket + 1U) & mask;
  }
  savedBucket = INVALID_TOKEN;
  return 1;
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline bool BuildJointUnionTurbo(
    const JointPlanArgs& a, const FastGroupSource& groupSrc, int64_t first, int64_t end,
    bool reset, bool compactPartition, ScratchPtr keys, ScratchPtr bucketSlots, ScratchPtr partition,
    ScratchPtr queryBuckets, __ubuf__ int32_t* scan, uint32_t thread, int32_t& outUnionCount) {
  for (uint32_t i = thread; i < a.jointHashCapacity; i += PLAN_THREADS) {
    keys[i] = HASH_EMPTY;
    bucketSlots[i] = INVALID_TOKEN;
    if (compactPartition && !reset && i < a.capacity) {
      partition[i] = 0;
    }
  }
  asc_syncthreads();
  uint32_t scanParity = 0U;
  int32_t count = 0;
  for (int64_t row = first; row < end; ++row) {
    const uint32_t rowUpper = groupSrc.RowUpper(a.visibleSeqLens[row]);
    const int64_t rowBase = row * static_cast<int64_t>(a.topk);
    const uint32_t queryRowOffset = static_cast<uint32_t>(row - first) * a.topk;
    for (uint32_t tile = 0; tile < a.topk; tile += PLAN_THREADS) {
      const uint32_t pos = tile + thread;
      int32_t added = 0;
      int32_t token = INVALID_TOKEN;
      int32_t savedBucket = INVALID_TOKEN;
      if (pos < a.topk) {
        token = a.topkIndices[rowBase + pos];
        if (groupSrc.IsValid(token, rowUpper)) {
          added = InsertJointToken(keys, token, a.jointHashCapacity, savedBucket);
        }
        queryBuckets[queryRowOffset + pos] = savedBucket;
      }
      int32_t tileTotal = 0;
      const int32_t rank = TurboExclusiveScan(scan, added, thread, scanParity, tileTotal);
      const int32_t nextCount = count + tileTotal;
      if (nextCount > static_cast<int32_t>(a.capacity)) {
        asc_syncthreads();
        outUnionCount = nextCount;
        return false;
      }
      if (added != 0) {
        const int32_t uid = count + rank;
        if (reset) {
          bucketSlots[savedBucket] = uid;
          partition[uid] = token;
        } else if (compactPartition) {
          partition[uid] = static_cast<int32_t>(static_cast<uint32_t>(savedBucket) << 16U);
        }
      }
      count = nextCount;
    }
  }
  asc_syncthreads();
  outUnionCount = count;
  return true;
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void PlanJointUnionTurbo(
    const JointPlanArgs& a, const FastGroupSource& groupSrc, int64_t first, int64_t end,
    int64_t reqId, bool reset, bool compactPartition, int32_t unionCount,
    ScratchPtr keys, ScratchPtr bucketSlots, ScratchPtr partition, ScratchPtr queryBuckets,
    __ubuf__ int32_t* scan, __gm__ int32_t* modeOutput, uint32_t thread) {
  const int64_t base = first * static_cast<int64_t>(a.capacity);
  const int64_t missBase = first * static_cast<int64_t>(a.topk);
  int32_t misses = 0;

  if (reset) {
    misses = unionCount;
    const uint32_t uCount = static_cast<uint32_t>(unionCount);
    for (uint32_t uid = thread; uid < uCount; uid += PLAN_THREADS) {
      const int32_t token = partition[uid];
      a.slotToToken[base + uid] = token;
      a.missTokens[missBase + uid] = token;
      a.missSlots[missBase + uid] = static_cast<int32_t>(uid);
    }
    for (uint32_t slot = uCount + thread; slot < a.capacity; slot += PLAN_THREADS) {
      a.slotToToken[base + slot] = INVALID_TOKEN;
    }
    const uint32_t unused = a.capacity - uCount;
    for (uint32_t order = thread; order < a.capacity; order += PLAN_THREADS) {
      a.lruSlots[base + order] = static_cast<int32_t>(order < unused ? (uCount + order) : (order - unused));
    }
  } else {
    int32_t stable = a.stablePrefixLens[first];
    for (int64_t row = first + 1; row < end; ++row) {
      const int32_t rowStable = a.stablePrefixLens[row];
      if (rowStable < stable) stable = rowStable;
    }
    if (stable < 0) stable = 0;

    uint32_t scanParity = 0U;
    int32_t evicts = 0, hits = 0;
    for (uint32_t tile = 0; tile < a.capacity; tile += PLAN_THREADS) {
      const uint32_t order = tile + thread;
      int32_t slot = -1, hit = 0, evict = 0;
      if (order < a.capacity) {
        slot = a.lruSlots[base + order];
        int32_t token = a.slotToToken[base + slot];
        if (token >= stable) {
          token = INVALID_TOKEN;
          a.slotToToken[base + slot] = INVALID_TOKEN;
        }
        const uint32_t bucket = token >= 0 ? LookupTopkToken(keys, token, a.jointHashCapacity) : HASH_BUCKET_INVALID;
        hit = bucket != HASH_BUCKET_INVALID;
        evict = 1 - hit;
        if (hit) {
          bucketSlots[bucket] = slot;
        }
      }
      int32_t totals = 0;
      const int32_t ranks = TurboExclusiveScan(scan, evict + (hit << COUNT_SHIFT), thread, scanParity, totals);
      if (evict) {
        if (compactPartition) {
          partition[evicts + (ranks & COUNT_MASK)] |= slot;
        } else {
          partition[evicts + (ranks & COUNT_MASK)] = slot;
        }
      }
      if (hit) {
        const uint32_t hitIdx = a.capacity - 1U - static_cast<uint32_t>(hits + (ranks >> COUNT_SHIFT));
        if (compactPartition) {
          partition[hitIdx] |= slot;
        } else {
          partition[hitIdx] = slot;
        }
      }
      FinishTileTurbo(tile, a.capacity);
      evicts += totals & COUNT_MASK;
      hits += totals >> COUNT_SHIFT;
    }

    if (compactPartition) {
      const int32_t expectedMisses = unionCount - hits;
      if (expectedMisses > 0) {
        const uint32_t uCount = static_cast<uint32_t>(unionCount);
        for (uint32_t tile = 0; tile < uCount && misses < expectedMisses; tile += PLAN_THREADS) {
          const uint32_t uid = tile + thread;
          uint32_t bucket = 0U;
          bool missing = false;
          if (uid < uCount) {
            bucket = static_cast<uint32_t>(partition[uid]) >> 16U;
            missing = bucketSlots[bucket] < 0;
          }
          int32_t total = 0;
          const int32_t rank = TurboExclusiveScan(scan, missing ? 1 : 0, thread, scanParity, total);
          if (missing) {
            const int32_t index = misses + rank;
            const int32_t slot = partition[index] & 0xFFFF;
            const int32_t token = keys[bucket];
            bucketSlots[bucket] = slot;
            a.slotToToken[base + slot] = token;
            a.missTokens[missBase + index] = token;
            a.missSlots[missBase + index] = slot;
          }
          misses += total;
          if (tile + PLAN_THREADS >= uCount || misses >= expectedMisses) {
            asc_syncthreads();
          }
        }
      }
    } else {
      for (uint32_t tile = 0; tile < a.jointHashCapacity; tile += PLAN_THREADS) {
        const uint32_t bucket = tile + thread;
        const bool missing = bucket < a.jointHashCapacity && keys[bucket] >= 0 && bucketSlots[bucket] < 0;
        int32_t total = 0;
        const int32_t rank = TurboExclusiveScan(scan, missing ? 1 : 0, thread, scanParity, total);
        if (missing) {
          const int32_t index = misses + rank;
          const int32_t slot = partition[index];
          const int32_t token = keys[bucket];
          bucketSlots[bucket] = slot;
          a.slotToToken[base + slot] = token;
          a.missTokens[missBase + index] = token;
          a.missSlots[missBase + index] = slot;
        }
        asc_syncthreads();
        misses += total;
      }
    }

    const int32_t unused = evicts - misses;
    const int32_t slotMask = compactPartition ? 0xFFFF : -1;
    for (uint32_t order = thread; order < a.capacity; order += PLAN_THREADS) {
      const int32_t packedSlot =
          order < static_cast<uint32_t>(unused)
              ? partition[misses + order]
              : (order < static_cast<uint32_t>(evicts) ? partition[order - unused]
                                                       : partition[a.capacity - 1U - (order - evicts)]);
      a.lruSlots[base + order] = packedSlot & slotMask;
    }
  }

  const uint32_t pages = groupSrc.Pages(a.capacity);
  const int32_t ownerPageBase = static_cast<int32_t>(first * static_cast<int64_t>(pages));
  for (int64_t row = first; row < end; ++row) {
    for (uint32_t page = thread; page < pages; page += PLAN_THREADS) {
      a.residentBlockTable[row * static_cast<int64_t>(pages) + page] = ownerPageBase + static_cast<int32_t>(page);
    }
    if (thread == 0) {
      a.residentRows[row] = static_cast<int32_t>(first);
      const int64_t remaining = misses - (row - first) * static_cast<int64_t>(a.topk);
      a.missCount[row] = static_cast<int32_t>(
          remaining <= 0 ? 0 : (remaining < static_cast<int64_t>(a.topk) ? remaining : static_cast<int64_t>(a.topk)));
    }
    // Device experiments reproduced this mode value in a read-only input.
    // Explicit output + thread 1 is an empirical workaround; root cause is
    // unconfirmed. This is not a synchronization repair: no barrier is added.
    if (thread == 1) {
      modeOutput[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;
    }
    const int64_t rowQueryBase = row * static_cast<int64_t>(a.topk);
    const uint32_t ubRowOffset = static_cast<uint32_t>(row - first) * a.topk;
    for (uint32_t pos = thread; pos < a.topk; pos += PLAN_THREADS) {
      const int32_t bucket = queryBuckets[ubRowOffset + pos];
      a.currentSlots[rowQueryBase + pos] = bucket >= 0 ? bucketSlots[bucket] : INVALID_TOKEN;
    }
  }
  if (thread == 0) a.lastReqIds[first] = reqId;
  asc_syncthreads();
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

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessRequestRun(const JointPlanArgs& a, int64_t first, int64_t end,
                                                         ScratchPtr scratch, __ubuf__ int32_t* scan,
                                                         __gm__ int32_t* modeOutput) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  if (end - first > 1) {
    const FastGroupSource groupSrc = MakeFastGroupSource(a, a.tokenToReq[first]);
    const uint64_t totalQueries = static_cast<uint64_t>(end - first) * a.topk;
    const uint64_t neededElements = 2ULL * a.jointHashCapacity + a.capacity + totalQueries;
    if (groupSrc.reqValid && neededElements <= a.scratchElements) {
      ScratchPtr keys = scratch;
      ScratchPtr bucketSlots = keys + a.jointHashCapacity;
      ScratchPtr partition = bucketSlots + a.jointHashCapacity;
      ScratchPtr queryBuckets = partition + a.capacity;
      const int64_t reqId = a.reqIds[groupSrc.request];
      const bool reset = a.rowModes[first] != ROW_JOINT || a.lastReqIds[first] != reqId;
      const bool compactPartition = a.jointHashCapacity <= 65535U && a.capacity <= 65535U;
      int32_t unionCount = 0;
      if (BuildJointUnionTurbo(a, groupSrc, first, end, reset, compactPartition, keys, bucketSlots, partition,
                               queryBuckets, scan, thread, unionCount)) {
        PlanJointUnionTurbo(a, groupSrc, first, end, reqId, reset, compactPartition, unionCount,
                            keys, bucketSlots, partition, queryBuckets, scan, modeOutput, thread);
        return;
      }
    }
  }
  // Overflow or single-row run never mutates the joint cache; process rows via fast private path.
  for (int64_t row = first; row < end; ++row) {
    RunPrivateRow(a, row, scratch, scan, thread);
  }
}

__simt_callee__ __aicore__ inline void RunJointScheduleTurbo(
    const JointPlanArgs& args, int64_t rowStart, int64_t rowStride, int64_t numReqs,
    __gm__ int32_t* tokenToReq, __gm__ int32_t* rowModes, __ubuf__ int32_t* scanWorkspace,
    __gm__ int32_t* compactWorkspace, uint64_t jointHashCapacity, uint64_t scratchElements) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  if (jointHashCapacity == 0U) {
    for (int64_t row = rowStart; row < numReqs; row += rowStride) {
      if (scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
        RunPrivateRow(args, row, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, thread);
      } else {
        RunPrivateRow(args, row, compactWorkspace + static_cast<uint64_t>(row) * scratchElements,
                      scanWorkspace, thread);
      }
    }
    return;
  }
  int64_t group = 0;
  int64_t nextTarget = rowStart;
  for (int64_t row = 0; row + (nextTarget - group) < numReqs;) {
    const int32_t req = tokenToReq[row];
    int64_t end = row + 1;
    while (end < numReqs && tokenToReq[end] == req) ++end;
    if (group == nextTarget) {
      if (scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
        ProcessRequestRun(args, row, end, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, rowModes);
      } else {
        ProcessRequestRun(args, row, end, compactWorkspace + static_cast<uint64_t>(row) * scratchElements,
                          scanWorkspace, rowModes);
      }
      nextTarget += rowStride;
    }
    ++group;
    row = end;
  }
}
