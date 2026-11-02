// SPDX-License-Identifier: Apache-2.0
// 只由 Turbo mask18/19/20 包含。A/B 与旧 joint_a/joint_b 的含义无关。
// A：先只读查询已冻结的前序行 Hash，再统一进入 CAS 阶段。
// B：先累计一行的新增项，再做一次 block Scan；借 queryBuckets 暂存新增标记。
// 不改变持久状态发布、LRU、回退或 Scan 实现。性能收益必须由 NPU 验证。

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void PublishTurboUnique(
    bool reset, bool compactPartition, ScratchPtr bucketSlots,
    ScratchPtr partition, int32_t bucket, int32_t token, int32_t uid) {
  if (reset) {
    bucketSlots[bucket] = uid;
    partition[uid] = token;
  } else if (compactPartition) {
    partition[uid] = static_cast<int32_t>(static_cast<uint32_t>(bucket) << 16U);
  }
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline bool BuildJointUnionExperiment(
    const JointPlanArgs& a, const FastGroupSource& groupSrc, int64_t first, int64_t end,
    bool reset, bool compactPartition, ScratchPtr keys, ScratchPtr bucketSlots, ScratchPtr partition,
    ScratchPtr queryBuckets, __ubuf__ int32_t* scan, uint32_t thread, int32_t& outUnionCount) {
  constexpr bool frozenLookup = JOINT_OPT_MASK == 18 || JOINT_OPT_MASK == 20;
  constexpr bool rowScan = JOINT_OPT_MASK == 19 || JOINT_OPT_MASK == 20;
  // -1 = 无效 query；-2 = A 的待插入项。B 插入结束后用 -(bucket+2) 表示新项。
  // 两种负值含义分属不同阶段，Scatter 后全部恢复为原始桶号或 -1。
  constexpr int32_t pending = -2;
  for (uint32_t i = thread; i < a.jointHashCapacity; i += PLAN_THREADS) {
    keys[i] = HASH_EMPTY;
    bucketSlots[i] = INVALID_TOKEN;
    if (compactPartition && !reset && i < a.capacity) partition[i] = 0;
  }
  asc_syncthreads();
  uint32_t scanParity = 0U;
  int32_t count = 0;
  for (int64_t row = first; row < end; ++row) {
    const uint32_t rowUpper = groupSrc.RowUpper(a.visibleSeqLens[row]);
    const int64_t rowBase = row * static_cast<int64_t>(a.topk);
    const uint32_t queryRowOffset = static_cast<uint32_t>(row - first) * a.topk;
    const bool lookupRow = frozenLookup && row != first;
    if (lookupRow) {
      // 上一行尾部 barrier 已结束所有 CAS；此循环内没有任何 keys 写入。
      for (uint32_t pos = thread; pos < a.topk; pos += PLAN_THREADS) {
        const int32_t token = a.topkIndices[rowBase + pos];
        int32_t saved = INVALID_TOKEN;
        if (groupSrc.IsValid(token, rowUpper)) {
          const uint32_t bucket = LookupTopkToken(keys, token, a.jointHashCapacity);
          saved = bucket == HASH_BUCKET_INVALID ? pending : static_cast<int32_t>(bucket);
        }
        queryBuckets[queryRowOffset + pos] = saved;
      }
      // 必须等待全 block 完成普通读取后才允许 CAS；不能改成逐线程先读后 CAS。
      asc_syncthreads();
    }

    // 防止极大 J 下 -(bucket+2) 溢出；超出标记范围时退回原分 tile Scan。
    if (rowScan && a.jointHashCapacity <= (1U << 30U)) {
      int32_t addedByThread = 0;
      for (uint32_t pos = thread; pos < a.topk; pos += PLAN_THREADS) {
        int32_t bucket = lookupRow ? queryBuckets[queryRowOffset + pos] : pending;
        int32_t added = 0;
        if (bucket == pending) {
          const int32_t token = a.topkIndices[rowBase + pos];
          bucket = INVALID_TOKEN;
          if (groupSrc.IsValid(token, rowUpper))
            added = InsertJointToken(keys, token, a.jointHashCapacity, bucket);
        }
        // Hash 满时仍累计 added，整行的 union>capacity 检查先于任何持久写入。
        queryBuckets[queryRowOffset + pos] = added && bucket >= 0 ? -(bucket + 2) : bucket;
        addedByThread += added;
      }
      int32_t rowTotal = 0;
      int32_t rank = TurboExclusiveScan(scan, addedByThread, thread, scanParity, rowTotal);
      if (count + rowTotal > static_cast<int32_t>(a.capacity)) {
        asc_syncthreads();
        outUnionCount = count + rowTotal;
        return false;
      }
      for (uint32_t pos = thread; pos < a.topk; pos += PLAN_THREADS) {
        const int32_t encoded = queryBuckets[queryRowOffset + pos];
        if (encoded < INVALID_TOKEN) {
          const int32_t bucket = -encoded - 2;
          queryBuckets[queryRowOffset + pos] = bucket;
          PublishTurboUnique(reset, compactPartition, bucketSlots, partition, bucket, reset ? keys[bucket] : INVALID_TOKEN, count + rank);
          ++rank;
        }
      }
      count += rowTotal;
    } else {
      // A 单点保留每 PLAN_THREADS 项一次 Scan，以便单独衡量查表阶段的效果。
      for (uint32_t tile = 0; tile < a.topk; tile += PLAN_THREADS) {
        const uint32_t pos = tile + thread;
        int32_t added = 0, bucket = INVALID_TOKEN, token = INVALID_TOKEN;
        if (pos < a.topk) {
          bucket = lookupRow ? queryBuckets[queryRowOffset + pos] : pending;
          if (bucket == pending) {
            token = a.topkIndices[rowBase + pos];
            bucket = INVALID_TOKEN;
            if (groupSrc.IsValid(token, rowUpper))
              added = InsertJointToken(keys, token, a.jointHashCapacity, bucket);
          }
          queryBuckets[queryRowOffset + pos] = bucket;
        }
        int32_t tileTotal = 0;
        const int32_t rank = TurboExclusiveScan(scan, added, thread, scanParity, tileTotal);
        if (count + tileTotal > static_cast<int32_t>(a.capacity)) {
          asc_syncthreads();
          outUnionCount = count + tileTotal;
          return false;
        }
        if (added != 0)
          PublishTurboUnique(reset, compactPartition, bucketSlots, partition, bucket, token, count + rank);
        count += tileTotal;
      }
    }
    // A：冻结整行的 Hash 后下一行才可普通读取。B：完成标记解包/Scatter。
    // 不新增跨 AIV 同步；所有线程以同一 row 控制流经过这里。
    asc_syncthreads();
  }
  outUnionCount = count;
  return true;
}
