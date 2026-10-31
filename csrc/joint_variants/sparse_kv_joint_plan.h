/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
// Included after the ABCDF helpers, inside the kernel's anonymous namespace.
// A contiguous request run belongs to one AIV. No inter-AIV barriers or atomics.
struct JointPlanArgs {
  __gm__ int64_t *reqIds, *lastReqIds;
  __gm__ int32_t *topkIndices, *stablePrefixLens, *visibleSeqLens, *tokenToReq, *blockTable;
  __gm__ int32_t *slotToToken, *lruSlots, *currentSlots, *missCount, *missTokens, *missSlots;
  __gm__ int32_t *rowModes, *residentRows, *residentBlockTable;
  uint32_t topk, capacity, hashCapacity, jointHashCapacity;
  int32_t maxToken, blockSize;
  int64_t maxRequests, maxNumBlocks, hostNumBlocks;
};
constexpr int32_t ROW_PRIVATE = 0;
constexpr int32_t ROW_JOINT = 1;
constexpr int32_t ROW_FOLLOWER = 2;

__simt_callee__ __aicore__ inline bool JointTokenValid(const JointPlanArgs& a, int64_t row, int32_t token) {
  return IsSourceTokenValid(row, token, a.visibleSeqLens, a.tokenToReq, a.blockTable, a.maxToken, a.maxRequests,
                            a.maxNumBlocks, a.hostNumBlocks, a.blockSize);
}

// Each valid token has one bucket. Stop after a tile whose union exceeds C;
// insertion is bounded even when a small-capacity test fills the hash table.
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline int32_t InsertJointToken(ScratchPtr keys, int32_t token, uint32_t size, int32_t& savedBucket) {
  uint32_t bucket = HashToken(token, size);
  for (uint32_t probe = 0; probe < size; ++probe) {
    const int32_t old = asc_atomic_cas(keys + bucket, HASH_EMPTY, token);
    if (old == token || old == HASH_EMPTY) {
      savedBucket = static_cast<int32_t>(bucket);
      return old == HASH_EMPTY ? 1 : 0;
    }
    bucket = (bucket + 1U) & (size - 1U);
  }
  savedBucket = INVALID_TOKEN;
  return 1;  // Table full implies union overflow; never index an invalid bucket.
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline bool BuildJointUnion(const JointPlanArgs& a, int64_t first, int64_t end,
                                                       ScratchPtr keys, ScratchPtr slots, __ubuf__ int32_t* scan,
                                                       uint32_t thread) {
  for (uint32_t i = thread; i < a.jointHashCapacity; i += PLAN_THREADS) {
    keys[i] = HASH_EMPTY;
    slots[i] = INVALID_TOKEN;
  }
  asc_syncthreads();
  int32_t count = 0;
  for (int64_t row = first; row < end; ++row) {
#if JOINT_OPT_MASK & 8
    // D: 跨 tile 只有原子插入，不消费其他线程的普通 load。
    // 本行开始 U<=C，本行新增<=T<=C，H>=2C；有界探测不会无限等待。
    int32_t rowAdded = 0;
#endif
    for (uint32_t tile = 0; tile < a.topk; tile += PLAN_THREADS) {
      const uint32_t pos = tile + thread;
      int32_t added = 0;
      if (pos < a.topk) {
        const int32_t token = a.topkIndices[row * a.topk + pos];
        int32_t savedBucket = INVALID_TOKEN;
        if (JointTokenValid(a, row, token)) {
          added = InsertJointToken(keys, token, a.jointHashCapacity, savedBucket);
        }
#if JOINT_OPT_MASK & 2
        // B: 记录的是该 query 通过因果可见性和来源检查后的桶号。
        // 各行可以完全不同、无序、含重复项；非法位置保留 -1。
        a.currentSlots[row * a.topk + pos] = savedBucket;
#endif
      }
#if JOINT_OPT_MASK & 8
      rowAdded += added;
#else
      BlockExclusiveScan(scan, added, thread);
      count += scan[CONTROL_OFFSET + CONTROL_TOTAL];
      asc_syncthreads();
      if (count > static_cast<int32_t>(a.capacity)) return false;
#endif
    }
#if JOINT_OPT_MASK & 8
    BlockExclusiveScan(scan, rowAdded, thread);
    count += scan[CONTROL_OFFSET + CONTROL_TOTAL];
    asc_syncthreads();  // 所有插入完成、所有线程消费 total，才允许读取哈希或回退。
    if (count > static_cast<int32_t>(a.capacity)) return false;
#endif
  }
  return true;
}

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
  ProcessRuntimeRow(scan, row, a.reqIds, a.lastReqIds, a.topkIndices, a.stablePrefixLens, a.visibleSeqLens,
                    a.tokenToReq, a.blockTable, a.slotToToken, a.lruSlots, a.currentSlots, a.missCount, a.missTokens,
                    a.missSlots, scratch, a.hashCapacity, a.topk, a.capacity, a.maxToken, a.maxRequests, a.maxNumBlocks,
                    a.hostNumBlocks, a.blockSize);
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void PlanJointUnion(const JointPlanArgs& a, int64_t first, int64_t end,
                                                      ScratchPtr keys, ScratchPtr slots, ScratchPtr partition,
                                                      __ubuf__ int32_t* scan, __gm__ int32_t* modeOutput,
                                                      uint32_t thread) {
  const int64_t base = first * a.capacity;
  const int64_t reqId = a.reqIds[a.tokenToReq[first]];
  const bool reset = a.rowModes[first] != ROW_JOINT || a.lastReqIds[first] != reqId;
  int32_t stable = a.stablePrefixLens[first];
  for (int64_t row = first + 1; row < end; ++row) {
    if (a.stablePrefixLens[row] < stable) stable = a.stablePrefixLens[row];
  }
  if (stable < 0) stable = 0;
  // Read reset on all threads before thread 0 changes persistent mode/identity.
  asc_syncthreads();
  int32_t evicts = 0, hits = 0;
  for (uint32_t tile = 0; tile < a.capacity; tile += PLAN_THREADS) {
    const uint32_t order = tile + thread;
    int32_t slot = -1, hit = 0, evict = 0;
    if (order < a.capacity) {
      slot = reset ? static_cast<int32_t>(order) : a.lruSlots[base + order];
      int32_t token = reset ? INVALID_TOKEN : a.slotToToken[base + slot];
#if JOINT_OPT_MASK & 4
      // C: 未失效槽内容不变，省去同值 GM 写回；reset 必须逐槽清空。
      if (reset || token >= stable) {
        token = INVALID_TOKEN;
        a.slotToToken[base + slot] = token;
      }
#else
      if (token >= stable) token = INVALID_TOKEN;
      a.slotToToken[base + slot] = token;
#endif
      const uint32_t bucket = token >= 0 ? LookupTopkToken(keys, token, a.jointHashCapacity) : HASH_BUCKET_INVALID;
      hit = bucket != HASH_BUCKET_INVALID;
      evict = 1 - hit;
      if (hit) slots[bucket] = slot;  // Joint state contains each token at most once.
    }
    const int32_t ranks = BlockExclusiveScan(scan, evict + (hit << COUNT_SHIFT), thread);
    const int32_t totals = scan[CONTROL_OFFSET + CONTROL_TOTAL];
    if (evict) partition[evicts + (ranks & COUNT_MASK)] = slot;
    if (hit) partition[a.capacity - 1U - (hits + (ranks >> COUNT_SHIFT))] = slot;
    asc_syncthreads();
    evicts += totals & COUNT_MASK;
    hits += totals >> COUNT_SHIFT;
  }
  int32_t misses = 0;
  for (uint32_t tile = 0; tile < a.jointHashCapacity; tile += PLAN_THREADS) {
    const uint32_t bucket = tile + thread;
    const bool missing = bucket < a.jointHashCapacity && keys[bucket] >= 0 && slots[bucket] < 0;
    const int32_t rank = BlockExclusiveScan(scan, missing ? 1 : 0, thread);
    const int32_t total = scan[CONTROL_OFFSET + CONTROL_TOTAL];
    if (missing) {
      const int32_t index = misses + rank;
      const int32_t slot = partition[index];
      const int32_t token = keys[bucket];
      slots[bucket] = slot;
      a.slotToToken[base + slot] = token;
      // Reuse W*T entries already allocated to this run; U <= W*T and U <= C.
      a.missTokens[first * a.topk + index] = token;
      a.missSlots[first * a.topk + index] = slot;
    }
    asc_syncthreads();
    misses += total;
  }
  const int32_t unused = evicts - misses;
  for (uint32_t order = thread; order < a.capacity; order += PLAN_THREADS) {
    a.lruSlots[base + order] =
        order < static_cast<uint32_t>(unused)
            ? partition[misses + order]
            : (order < static_cast<uint32_t>(evicts) ? partition[order - unused]
                                                     : partition[a.capacity - 1U - (order - evicts)]);
  }
  for (int64_t row = first; row < end; ++row) {
    PublishResidentRow(a, row, first, thread);
    if (thread == 0) {
      int64_t remaining = misses - (row - first) * a.topk;
      a.missCount[row] = remaining <= 0 ? 0 : (remaining < a.topk ? remaining : a.topk);
    }
    // Device experiments reproduced this mode value in a read-only input.
    // Explicit output + thread 1 is an empirical workaround; root cause is
    // unconfirmed. This is not a synchronization repair: no barrier is added.
    if (thread == 1) {
      modeOutput[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;
    }
    for (uint32_t pos = thread; pos < a.topk; pos += PLAN_THREADS) {
#if JOINT_OPT_MASK & 2
      // 哈希构建完成后不再移动/删除 keys；桶号在整个调用内稳定。
      const int32_t bucket = a.currentSlots[row * a.topk + pos];
      a.currentSlots[row * a.topk + pos] = bucket >= 0 ? slots[bucket] : INVALID_TOKEN;
#else
      const int32_t token = a.topkIndices[row * a.topk + pos];
      int32_t slot = INVALID_TOKEN;
      if (JointTokenValid(a, row, token)) {
        const uint32_t bucket = LookupTopkToken(keys, token, a.jointHashCapacity);
        slot = slots[bucket];
      }
      a.currentSlots[row * a.topk + pos] = slot;
#endif
    }
  }
  if (thread == 0) a.lastReqIds[first] = reqId;
  asc_syncthreads();
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessRequestRun(const JointPlanArgs& a, int64_t first, int64_t end,
                                                         ScratchPtr scratch, __ubuf__ int32_t* scan,
                                                         __gm__ int32_t* modeOutput) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  ScratchPtr keys = scratch;
  ScratchPtr slots = keys + a.jointHashCapacity;
  ScratchPtr partition = slots + a.jointHashCapacity;
  const int32_t request = a.tokenToReq[first];
  if (end - first > 1 && request >= 0 && request < a.maxRequests &&
      BuildJointUnion(a, first, end, keys, slots, scan, thread)) {
    PlanJointUnion(a, first, end, keys, slots, partition, scan, modeOutput, thread);
  } else {
    // Overflow never mutates the cache. Preserve every row using ABCDF.
    for (int64_t row = first; row < end; ++row) RunPrivateRow(a, row, scratch, scan, thread);
  }
}
