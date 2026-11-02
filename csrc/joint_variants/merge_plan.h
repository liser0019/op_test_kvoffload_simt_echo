// Merge Joint Plan V1: sorted TopK -> resident-first filtering -> stable 4-way merge.
// Intended to be included *inside the same anonymous namespace* AFTER the existing
// turbo definitions of JointPlanArgs, FastGroupSource, TurboExclusiveScan,
// InsertJointToken, LookupTopkToken and RunPrivateRow.
//
// Target: Ascend 950PR / 950DT, CANN 9.1+, PLAN_THREADS unchanged.
// This version deliberately performs the merge in SIMT (parallel rank scatter).
// It does NOT yet call the vector MrgSort instruction; that is a later isolated
// optimization once the exact kernel/memory contract is verified on-device.
//
// Host preconditions:
//  - all TopK rows nondecreasing by TOKEN POSITION (not attention score);
//  - group rows are contiguous and tokenToReq is constant within each group;
//  - resident LRU is a permutation and all resident physical slots for the
//    request group share the owner's first-row state;
//  - source page-table legality is validated upstream as in FastGroupSource;
//  - before invoking this path, the scratch buffer contains at least
//      2 * next_pow2(2 * capacity) + groupRows * topk + capacity int32 elements.
//    The available scratch pointer may be either __ubuf__ or __gm__.
//
// Joint semantics:
//  - Duplicate resident tokens remain protected in ALL occupied slots; the
//    LRU-last owner answers hits.
//  - Each distinct missing token is fetched ONCE; duplicate queries share slot.
//  - Miss allocation order is TOKEN ASCENDING, not first row/query occurrence.
//  - Overflow refuses joint commit and falls back to the existing private path.
//  - No persistent resident/LRU/metadata write before the capacity precheck.

constexpr uint32_t MERGE_MAX_ROWS = 4U;
constexpr uint32_t MERGE_OWNER_SELECTED = 0x80000000U;
constexpr uint32_t MERGE_OWNER_VALUE_MASK = 0x7fffffffU;

__simt_callee__ __aicore__ inline uint32_t MergeNextPow2Double(uint32_t c) {
  // Caller prechecks c <= 32768.
  uint32_t h = 1U;
  while (h < 2U * c) h <<= 1U;
  return h;
}

__simt_callee__ __aicore__ inline uint32_t MergeRowLen(uint32_t row,
    uint32_t n0, uint32_t n1, uint32_t n2, uint32_t n3) {
  return row == 0U ? n0 : row == 1U ? n1 : row == 2U ? n2 : n3;
}

// sorted[lo:hi] is a nondecreasing sequence of unsigned packed token/query keys.
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline uint32_t MergeLowerBound(
    ScratchPtr sorted, uint32_t base, uint32_t length, uint32_t key) {
  uint32_t lo = 0, hi = length;
  while (lo < hi) {
    const uint32_t mid = lo + ((hi - lo) >> 1U);
    if (static_cast<uint32_t>(sorted[base + mid]) < key) lo = mid + 1U;
    else hi = mid;
  }
  return lo;
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline bool BuildAndCommitSortedJoint(
    const JointPlanArgs& a, const FastGroupSource& src,
    int64_t first, int64_t end, int64_t reqId, bool reset,
    ScratchPtr scratch, uint32_t h,
    __ubuf__ int32_t* scan, __gm__ int32_t* modeOutput,
    uint32_t thread, uint32_t queryBits) {
  const uint32_t queryMask = (1U << queryBits) - 1U;
  const uint32_t rows = static_cast<uint32_t>(end - first);
  const uint32_t totalQueries = rows * a.topk;
  const int64_t stateBase = first * static_cast<int64_t>(a.capacity);
  const int64_t queryBase = first * static_cast<int64_t>(a.topk);

  // Phase lifetime layout (units: int32):
  //   hashKeys[h] | owner[h] | packedMiss[rows * topk] | partition[capacity]
  // After LRU classification, hashKeys+owner becomes the sorted merge output.
  ScratchPtr hashKeys = scratch;
  ScratchPtr owner = hashKeys + h;
  ScratchPtr packedMiss = owner + h;
  ScratchPtr partition = packedMiss + totalQueries;
  ScratchPtr merged = hashKeys; // reused ONLY after hash is no longer queried.

  int32_t stable = a.stablePrefixLens[first];
  for (int64_t row = first + 1; row < end; ++row) {
    const int32_t cur = a.stablePrefixLens[row];
    if (cur < stable) stable = cur;
  }
  if (stable < 0) stable = 0;
  if (stable > a.maxToken) stable = a.maxToken;

  // PHASE 1: Build resident token -> LRU-last (order,slot) index.
  // A reset has no valid shared resident state, so skip all resident reads.
  if (!reset) {
    for (uint32_t i = thread; i < h; i += PLAN_THREADS) {
      hashKeys[i] = HASH_EMPTY;
      owner[i] = INVALID_TOKEN;
    }
    asc_syncthreads();
    for (uint32_t order = thread; order < a.capacity; order += PLAN_THREADS) {
      const int32_t slot = a.lruSlots[stateBase + order];
      if (slot < 0 || static_cast<uint32_t>(slot) >= a.capacity) continue;
      const int32_t token = a.slotToToken[stateBase + slot];
      if (token < 0 || token >= stable) continue;
      int32_t bucket = INVALID_TOKEN;
      InsertJointToken(hashKeys, token, h, bucket);
      if (bucket >= 0) {
        // capacity <= 32768: order[0:14] and slot[0:15] fit in 31 bits.
        const int32_t packedOwner = static_cast<int32_t>(
            (order << 16U) | static_cast<uint32_t>(slot));
        asc_atomic_max(owner + bucket, packedOwner);
      }
    }
    asc_syncthreads();
  }

  // PHASE 2: Filter resident hits. Per-row stable compaction keeps miss lists
  // sorted because the incoming TopK row is sorted by token position.
  uint32_t n0 = 0U, n1 = 0U, n2 = 0U, n3 = 0U;
  uint32_t scanParity = 0U;
  for (uint32_t rowOff = 0; rowOff < rows; ++rowOff) {
    const int64_t row = first + static_cast<int64_t>(rowOff);
    const uint32_t upper = src.RowUpper(a.visibleSeqLens[row]);
    uint32_t count = 0U;
    for (uint32_t tile = 0U; tile < a.topk; tile += PLAN_THREADS) {
      const uint32_t pos = tile + thread;
      const uint32_t qIdx = rowOff * a.topk + pos;
      bool missing = false;
      int32_t token = INVALID_TOKEN;
      if (pos < a.topk) {
        token = a.topkIndices[queryBase + qIdx];
        a.currentSlots[queryBase + qIdx] = INVALID_TOKEN;
        if (src.IsValid(token, upper)) {
          uint32_t bucket = HASH_BUCKET_INVALID;
          if (!reset) bucket = LookupTopkToken(hashKeys, token, h);
          if (bucket != HASH_BUCKET_INVALID) {
            // Atomic OR is BOTH an atomic hit marker and a race-free read of
            // the immutable owner bits (other threads only modify bit 31).
            const uint32_t before = static_cast<uint32_t>(
                asc_atomic_or(owner + bucket,
                              static_cast<int32_t>(MERGE_OWNER_SELECTED)));
            a.currentSlots[queryBase + qIdx] = static_cast<int32_t>(before & OWNER_SLOT_MASK);
          } else {
            missing = true;
          }
        }
      }
      int32_t tileCount = 0;
      const int32_t rank = TurboExclusiveScan(
          scan, missing ? 1 : 0, thread, scanParity, tileCount);
      if (missing) {
        // Packed key preserves token ascending and then query-index ascending.
        const uint32_t key = (static_cast<uint32_t>(token) << queryBits) | qIdx;
        packedMiss[rowOff * a.topk + count + static_cast<uint32_t>(rank)] =
            static_cast<int32_t>(key);
      }
      // Complete the scatter before the next scan/tile can reuse the workspace.
      asc_syncthreads();
      count += static_cast<uint32_t>(tileCount);
    }
    if (rowOff == 0U) n0 = count;
    else if (rowOff == 1U) n1 = count;
    else if (rowOff == 2U) n2 = count;
    else n3 = count;
  }

  // PHASE 3: Classify old LRU slots by selected resident token. This protects
  // every duplicate resident slot, not just the last owner chosen for queries.
  uint32_t evicts = 0U, hits = 0U;
  if (!reset) {
    for (uint32_t tile = 0U; tile < a.capacity; tile += PLAN_THREADS) {
      const uint32_t order = tile + thread;
      int32_t slot = INVALID_TOKEN;
      int32_t selected = 0, evict = 0;
      if (order < a.capacity) {
        slot = a.lruSlots[stateBase + order];
        if (slot >= 0 && static_cast<uint32_t>(slot) < a.capacity) {
          const int32_t token = a.slotToToken[stateBase + slot];
          if (token >= 0 && token < stable) {
            const uint32_t b = LookupTopkToken(hashKeys, token, h);
            selected = b != HASH_BUCKET_INVALID &&
                (static_cast<uint32_t>(owner[b]) & MERGE_OWNER_SELECTED) != 0U;
          }
          evict = 1 - selected;
        }
      }
      int32_t tileTotal = 0;
      const int32_t rank = TurboExclusiveScan(
          scan, evict + (selected << COUNT_SHIFT), thread,
          scanParity, tileTotal);
      if (evict) {
        partition[evicts + (static_cast<uint32_t>(rank) & COUNT_MASK)] = slot;
      }
      if (selected) {
        const uint32_t idx = a.capacity - 1U -
            (hits + (static_cast<uint32_t>(rank) >> COUNT_SHIFT));
        partition[idx] = slot;
      }
      asc_syncthreads();
      evicts += static_cast<uint32_t>(tileTotal) & COUNT_MASK;
      hits += static_cast<uint32_t>(tileTotal) >> COUNT_SHIFT;
    }
    // Private fallback is safer than continuing with a corrupt LRU permutation.
    if (evicts + hits != a.capacity) return false;
  } else {
    for (uint32_t slot = thread; slot < a.capacity; slot += PLAN_THREADS) {
      partition[slot] = static_cast<int32_t>(slot);
    }
    asc_syncthreads();
    evicts = a.capacity;
  }

  // PHASE 4: Merge up to four ALREADY SORTED miss streams in parallel.
  // The rank of a key in the merged list is its local index plus the count of
  // smaller keys from every other stream. The qIdx tie-break makes keys unique.
  // After phase 3, hashKeys+owner is dead and can be reused as merged[] .
  const uint32_t rawMisses = n0 + n1 + n2 + n3;
  for (uint32_t r = 0; r < rows; ++r) {
    const uint32_t len = MergeRowLen(r, n0, n1, n2, n3);
    for (uint32_t j = thread; j < len; j += PLAN_THREADS) {
      const uint32_t key = static_cast<uint32_t>(packedMiss[r * a.topk + j]);
      uint32_t rank = j;
      for (uint32_t s = 0; s < rows; ++s) {
        if (s == r) continue;
        rank += MergeLowerBound(packedMiss, s * a.topk,
                               MergeRowLen(s, n0, n1, n2, n3), key);
      }
      merged[rank] = static_cast<int32_t>(key);
    }
  }
  asc_syncthreads();

  // PHASE 5: Deterministic adjacent-unique + Scan. Reuse packedMiss as uid[]
  // AFTER the merged array has been fully published. No atomic 'ticket' order.
  scanParity = 0U;
  uint32_t distinct = 0U;
  for (uint32_t tile = 0; tile < rawMisses; tile += PLAN_THREADS) {
    const uint32_t i = tile + thread;
    bool firstOccurrence = false;
    if (i < rawMisses) {
      const uint32_t token = static_cast<uint32_t>(merged[i]) >> queryBits;
      firstOccurrence = i == 0U ||
          token != (static_cast<uint32_t>(merged[i - 1U]) >> queryBits);
    }
    int32_t tileTotal = 0;
    const int32_t rank = TurboExclusiveScan(
        scan, firstOccurrence ? 1 : 0, thread, scanParity, tileTotal);
    if (i < rawMisses) {
      // On a duplicate, exclusive rank points one AFTER its unique owner.
      const uint32_t uid = distinct + static_cast<uint32_t>(rank) -
                           (firstOccurrence ? 0U : 1U);
      packedMiss[i] = static_cast<int32_t>(uid);
    }
    asc_syncthreads();
    distinct += static_cast<uint32_t>(tileTotal);
  }
  // Never update the persistent shared cache on insufficient victims.
  if (distinct > evicts) return false;

  // PHASE 6: Commit. On reset, old slotToToken has no valid entries.
  if (reset) {
    for (uint32_t slot = thread; slot < a.capacity; slot += PLAN_THREADS) {
      a.slotToToken[stateBase + slot] = INVALID_TOKEN;
    }
  } else {
    for (uint32_t order = thread; order < a.capacity; order += PLAN_THREADS) {
      const int32_t slot = a.lruSlots[stateBase + order];
      const int32_t token = a.slotToToken[stateBase + slot];
      if (token >= stable) a.slotToToken[stateBase + slot] = INVALID_TOKEN;
    }
  }
  asc_syncthreads(); // invalidate before repopulating victim slots

  for (uint32_t i = thread; i < rawMisses; i += PLAN_THREADS) {
    const uint32_t key = static_cast<uint32_t>(merged[i]);
    const uint32_t uid = static_cast<uint32_t>(packedMiss[i]);
    const int32_t slot = partition[uid];
    const uint32_t qIdx = key & queryMask;
    const int32_t token = static_cast<int32_t>(key >> queryBits);
    a.currentSlots[queryBase + qIdx] = slot;
    if (i == 0U || token != static_cast<int32_t>(
            static_cast<uint32_t>(merged[i - 1U]) >> queryBits)) {
      a.slotToToken[stateBase + slot] = token;
      a.missTokens[queryBase + uid] = token;
      a.missSlots[queryBase + uid] = slot;
    }
  }
  for (uint32_t i = distinct + thread; i < totalQueries; i += PLAN_THREADS) {
    a.missTokens[queryBase + i] = INVALID_TOKEN;
    a.missSlots[queryBase + i] = INVALID_TOKEN;
  }

  const uint32_t unused = evicts - distinct;
  for (uint32_t order = thread; order < a.capacity; order += PLAN_THREADS) {
    const int32_t slot = order < unused
        ? partition[distinct + order]
        : (order < evicts
            ? partition[order - unused]
            : partition[a.capacity - 1U - (order - evicts)]);
    a.lruSlots[stateBase + order] = slot;
  }
  const uint32_t pages = src.Pages(a.capacity);
  const int32_t ownerPageBase = static_cast<int32_t>(
      first * static_cast<int64_t>(pages));
  for (uint32_t r = 0; r < rows; ++r) {
    const int64_t row = first + static_cast<int64_t>(r);
    for (uint32_t page = thread; page < pages; page += PLAN_THREADS) {
      a.residentBlockTable[row * static_cast<int64_t>(pages) + page] =
          ownerPageBase + static_cast<int32_t>(page);
    }
    if (thread == 0U) {
      a.residentRows[row] = static_cast<int32_t>(first);
      const int64_t remain = static_cast<int64_t>(distinct) -
                             static_cast<int64_t>(r) * a.topk;
      a.missCount[row] = remain <= 0 ? 0 : static_cast<int32_t>(
          remain < a.topk ? remain : a.topk);
    }
    // Keep the original tested metadata writer convention; not an alternative
    // to synchronization, and requires the previous joint-mode integration.
    if (thread == 1U) {
      modeOutput[row] = r == 0U ? ROW_JOINT : ROW_FOLLOWER;
    }
  }
  if (thread == 0U) a.lastReqIds[first] = reqId;
  asc_syncthreads();
  return true;
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void ProcessRequestRunMerge(
    const JointPlanArgs& a, int64_t first, int64_t end,
    ScratchPtr scratch, __ubuf__ int32_t* scan,
    __gm__ int32_t* modeOutput) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  const int64_t rowCount = end - first;
  // 生产形状: token 18 bits + query index 13 bits = 31 bits。
  // 动态分配低位，不能沿用 prototype 的固定 16+16 编码。
  const uint64_t q = static_cast<uint64_t>(rowCount) * a.topk;
  uint32_t queryBits = 1U;
  while (queryBits < 31U && (1ULL << queryBits) < q) ++queryBits;
  const bool packable = q > 0U && q <= (1ULL << queryBits) &&
      a.maxToken > 0 &&
      (static_cast<uint64_t>(a.maxToken - 1) << queryBits) <= UINT32_MAX;
  if (rowCount > 1 && rowCount <= static_cast<int64_t>(MERGE_MAX_ROWS) &&
      a.capacity > 0U && a.capacity <= 32768U &&
      packable &&
      a.topk > 0U &&
      static_cast<uint64_t>(rowCount) * a.topk <= 65535ULL &&
      a.capacity < (1U << COUNT_SHIFT) * 16U) {
    const FastGroupSource src = MakeFastGroupSource(a, a.tokenToReq[first]);
    const uint32_t h = MergeNextPow2Double(a.capacity);
    const uint64_t queryElements = static_cast<uint64_t>(rowCount) * a.topk;
    const uint64_t need = 2ULL * h + queryElements + a.capacity;
    if (src.reqValid && h <= a.jointHashCapacity &&
        queryElements <= 2ULL * h && need <= a.scratchElements) {
      const int64_t reqId = a.reqIds[src.request];
      bool reset = a.rowModes[first] != ROW_JOINT ||
                   a.lastReqIds[first] != reqId;
      // 沿用当前 Joint 协议：rowModes/lastReqIds 是持久状态。
      // residentRows 是输出，审计会污染它；不能用输出脏值触发 reset。
      // 分组拓扑改变须由调用方重置持久模式，与原 Joint 的约定相同。
      if (BuildAndCommitSortedJoint(a, src, first, end, reqId, reset,
                                    scratch, h, scan, modeOutput, thread, queryBits)) {
        return;
      }
    }
  }
  // The only writes made before a failed joint attempt were currentSlots.
  // RunPrivateRow overwrites those as it establishes new independent state.
  for (int64_t row = first; row < end; ++row) {
    RunPrivateRow(a, row, scratch, scan, thread);
  }
}

// Scheduling matches the existing RunJointScheduleTurbo; no host ABI changes.
__simt_callee__ __aicore__ inline void RunJointScheduleMerge(
    const JointPlanArgs& args, int64_t rowStart, int64_t rowStride,
    int64_t numReqs, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* rowModes, __ubuf__ int32_t* scanWorkspace,
    __gm__ int32_t* compactWorkspace,
    uint64_t jointHashCapacity, uint64_t scratchElements) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  if (jointHashCapacity == 0U) {
    for (int64_t row = rowStart; row < numReqs; row += rowStride) {
      if (scratchElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
        RunPrivateRow(args, row, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS,
                      scanWorkspace, thread);
      } else {
        RunPrivateRow(args, row,
                      compactWorkspace + static_cast<uint64_t>(row) * scratchElements,
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
        ProcessRequestRunMerge(args, row, end,
            scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, rowModes);
      } else {
        ProcessRequestRunMerge(args, row, end,
            compactWorkspace + static_cast<uint64_t>(row) * scratchElements,
            scanWorkspace, rowModes);
      }
      nextTarget += rowStride;
    }
    ++group;
    row = end;
  }
}
