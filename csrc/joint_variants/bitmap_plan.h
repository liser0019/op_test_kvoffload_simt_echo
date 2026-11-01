/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
// Included after the ABCDF helpers, inside the kernel's anonymous namespace.
// A contiguous request run belongs to one AIV. No inter-AIV barriers or atomics.
struct JointPlanArgs {
  __gm__ int64_t *reqIds, *lastReqIds;
  __gm__ int32_t *topkIndices, *stablePrefixLens, *visibleSeqLens, *tokenToReq;
  __gm__ int32_t *slotToToken, *lruSlots, *currentSlots, *missCount, *missTokens, *missSlots;
  __gm__ int32_t *rowModes, *residentRows, *residentBlockTable;
  uint32_t topk, capacity, hashCapacity, jointHashCapacity;
  __gm__ int32_t* directWorkspace;
  uint64_t scratchElements;
  int32_t maxToken, blockSize;
  int64_t maxRequests;
};
constexpr int32_t ROW_PRIVATE = 0;
constexpr int32_t ROW_JOINT = 1;
constexpr int32_t ROW_FOLLOWER = 2;

__simt_callee__ __aicore__ inline bool JointTokenValid(const JointPlanArgs& a, int64_t row, int32_t token) {
  return IsSourceTokenValid(row, token, a.visibleSeqLens, a.tokenToReq, a.maxToken, a.maxRequests);
}

// OR 是并集：旧位为0的线程是该token唯一写者，不做热点全局计数。
// bitmap和unique都在当前AIV的scratch；map仅写本次有效并集，无需清零整个maxToken表。
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline int32_t BuildBitmapUnion(
    const JointPlanArgs& a, int64_t first, int64_t end, ScratchPtr bitmap,
    ScratchPtr unique, __gm__ int32_t* map, __ubuf__ int32_t* scan, uint32_t thread) {
  const uint32_t words = (static_cast<uint32_t>(a.maxToken) + 31U) / 32U;
  for (uint32_t i = thread; i < words; i += PLAN_THREADS) bitmap[i] = 0;
  asc_syncthreads();
  int32_t count = 0;
  // 将所有query位置展平成一个并行序列；每个位置按自己的row判断可见性。
  const int64_t n = (end - first) * a.topk;
  for (int64_t tile = 0; tile < n; tile += PLAN_THREADS) {
    const int64_t index = tile + thread;
    int32_t token = -1, added = 0;
    if (index < n) {
      const int64_t row = first + index / a.topk;
      token = a.topkIndices[first * a.topk + index];
      if (JointTokenValid(a, row, token)) {
        const uint32_t bit = 1U << (static_cast<uint32_t>(token) & 31U);
        const uint32_t old = static_cast<uint32_t>(asc_atomic_or(bitmap + token / 32, static_cast<int32_t>(bit)));
        added = (old & bit) == 0;
      } else token = -1;
      a.currentSlots[first * a.topk + index] = token; // 暂存合法token，不是槽号。
    }
    const int32_t rank = BlockExclusiveScan(scan, added, thread);
    const int32_t total = scan[CONTROL_OFFSET + CONTROL_TOTAL];
    // 超容量的tile不用发布列表；先结束所有原子/临时写，再整组回退。
    if (count + total <= static_cast<int32_t>(a.capacity) && added) {
      unique[count + rank] = token;
      map[token] = -1;
    }
    asc_syncthreads();
    count += total;
    if (count > static_cast<int32_t>(a.capacity)) return -1;
  }
  return count;
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
                    a.tokenToReq, a.slotToToken, a.lruSlots, a.currentSlots, a.missCount, a.missTokens,
                    a.missSlots, scratch, a.hashCapacity, a.topk, a.capacity, a.maxToken, a.maxRequests);
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void PlanJointUnion(const JointPlanArgs& a, int64_t first, int64_t end,
                                                      ScratchPtr keys, __gm__ int32_t* slots, ScratchPtr partition,
                                                      ScratchPtr unique, int32_t unionCount,
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
      if (reset || token >= stable) {
        token = INVALID_TOKEN;
        a.slotToToken[base + slot] = token;
      }
      hit = token >= 0 && token < a.maxToken &&
          (static_cast<uint32_t>(keys[token / 32]) & (1U << (static_cast<uint32_t>(token) & 31U))) != 0;
      evict = 1 - hit;
      if (hit) slots[token] = slot;  // Joint state contains each token at most once.
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
  for (uint32_t tile = 0; tile < static_cast<uint32_t>(unionCount); tile += PLAN_THREADS) {
    const uint32_t indexInUnion = tile + thread;
    const int32_t token = indexInUnion < static_cast<uint32_t>(unionCount) ? unique[indexInUnion] : -1;
    const bool missing = token >= 0 && slots[token] < 0;
    const int32_t rank = BlockExclusiveScan(scan, missing ? 1 : 0, thread);
    const int32_t total = scan[CONTROL_OFFSET + CONTROL_TOTAL];
    if (missing) {
      const int32_t index = misses + rank;
      const int32_t slot = partition[index];
      slots[token] = slot;
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
      const int32_t token = a.currentSlots[row * a.topk + pos];
      a.currentSlots[row * a.topk + pos] = token >= 0 ? slots[token] : INVALID_TOKEN;
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
  const uint32_t words = (static_cast<uint32_t>(a.maxToken) + 31U) / 32U;
  ScratchPtr bitmap = scratch;
  ScratchPtr unique = bitmap + words;
  ScratchPtr partition = unique + a.capacity;
  __gm__ int32_t* map = a.directWorkspace + static_cast<uint64_t>(first) *
      (a.scratchElements + static_cast<uint64_t>(a.maxToken)) + a.scratchElements;
  const int32_t request = a.tokenToReq[first];
  int32_t count = -1;
  if (end - first > 1 && request >= 0 && request < a.maxRequests)
    count = BuildBitmapUnion(a, first, end, bitmap, unique, map, scan, thread);
  if (count >= 0) {
    PlanJointUnion(a, first, end, bitmap, map, partition, unique, count, scan, modeOutput, thread);
  } else {
    for (int64_t row = first; row < end; ++row) RunPrivateRow(a, row, scratch, scan, thread);
  }
}
