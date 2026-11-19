// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mtp3_lookup_config.h"

__simt_callee__ __aicore__ inline __gm__ int32_t* Mtp3LookupScratch(
    __gm__ int32_t* workspace, int64_t maxRequests, int64_t firstRow) {
  return workspace + maxRequests * mtp3_lookup::kMapWords +
         (firstRow / mtp3_lookup::kRowsPerRequest) * mtp3_lookup::kRequestWords;
}

__aicore__ inline bool Mtp3LookupAlignedGroup(
    const __gm__ int32_t* tokenToReq, int64_t first, int64_t rows) {
  const int64_t end = first + mtp3_lookup::kRowsPerRequest;
  if (end > rows) return false;
  const int32_t request = tokenToReq[first];
  return (first == 0 || tokenToReq[first - 1] != request) &&
         tokenToReq[first + 1] == request && tokenToReq[first + 2] == request &&
         tokenToReq[first + 3] == request && (end == rows || tokenToReq[end] != request);
}

// Prepare publishes temporary state only. Commit is the only phase that writes
// the persistent resident/LRU state, including the original fallback path.
template<bool Save>
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void Mtp3LookupTransferVf(
    __ubuf__ int32_t* ctx, __gm__ int32_t* workspace, int64_t maxRequests,
    int64_t first, uint32_t maxToken, bool hot) {
  using namespace mtp3_lookup;
  const uint32_t thread = threadIdx.x;
  __gm__ int32_t* shard = Mtp3LookupScratch(workspace, maxRequests, first);
  __gm__ int32_t* queryMap = workspace + first * (kMapWords / kRowsPerRequest);
  __gm__ int32_t* tokens = shard + kRecordsOffset;
  __ubuf__ int32_t* data = ctx + EIGHT_CTX;
  __ubuf__ int32_t* unique = data + kMapWords;
  if constexpr (Save) {
    const bool valid = ctx[0] == 1 && static_cast<uint32_t>(ctx[1]) <= kCapacity;
    const uint32_t count = valid ? static_cast<uint32_t>(ctx[1]) : 0;
    uint32_t shift = 0;
    while ((kDirectoryBins << shift) < maxToken) ++shift;
    if (thread == 0U) {
      shard[0] = valid ? 1 : 0;
      shard[1] = static_cast<int32_t>(count);
      shard[2] = hot ? 1 : 0;
      shard[3] = static_cast<int32_t>(shift);
    }
    if (!valid) return;
    for (uint32_t i = thread; i < kMapWords; i += SORTED_THREADS) queryMap[i] = data[i];
    for (uint32_t i = thread; i < count; i += SORTED_THREADS) tokens[i] = unique[i];
    if (hot) {
      __gm__ int32_t* directory = shard + kDirectoryOffset;
      for (uint32_t i = thread; i <= kDirectoryBins; i += SORTED_THREADS) {
        directory[i] = static_cast<int32_t>(SortedLower(unique, count, static_cast<int32_t>(i << shift)));
      }
    }
  } else {
    const uint32_t count = static_cast<uint32_t>(shard[1]);
    const bool valid = shard[0] == 1 && count <= kCapacity;
    if (thread == 0U) {
      ctx[0] = valid ? 1 : 0;
      ctx[1] = static_cast<int32_t>(count);
      ctx[120] = 0;
    }
    if (!valid) return;
    for (uint32_t i = thread; i < kMapWords; i += SORTED_THREADS) data[i] = queryMap[i];
    for (uint32_t i = thread; i < count; i += SORTED_THREADS) unique[i] = tokens[i];
  }
}

__simt_callee__ __aicore__ inline uint32_t Mtp3LookupEncode(
    int32_t token, int32_t stable, uint32_t shift, uint32_t count,
    const __ubuf__ int32_t* unique, const __ubuf__ int32_t* directory) {
  if (token >= stable) return 0x8000U;
  if (token < 0) return 0;
  const uint32_t bucket = static_cast<uint32_t>(token) >> shift;
  const uint32_t lo = static_cast<uint32_t>(directory[bucket]);
  const uint32_t hi = static_cast<uint32_t>(directory[bucket + 1]);
  const uint32_t pos = lo + SortedLower(unique + lo, hi - lo, token);
  return pos < count && unique[pos] == token ? (0x4000U | pos) : 0;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void Mtp3LookupSlotsVf(
    __ubuf__ int32_t* local, const __gm__ int32_t* resident, __gm__ int32_t* shard,
    uint32_t start, uint32_t end, uint32_t count, int32_t stable, uint32_t shift) {
  using namespace mtp3_lookup;
  const uint32_t thread = threadIdx.x;
  __ubuf__ int32_t* unique = local;
  __ubuf__ int32_t* directory = local + kCapacity;
  for (uint32_t i = thread; i < count; i += SORTED_THREADS) unique[i] = shard[kRecordsOffset + i];
  for (uint32_t i = thread; i <= kDirectoryBins; i += SORTED_THREADS) directory[i] = shard[kDirectoryOffset + i];
  asc_syncthreads();

  __gm__ uint32_t* results = reinterpret_cast<__gm__ uint32_t*>(shard + kSlotsOffset);
  // One writer per 32-bit word. Consecutive shards own disjoint aligned ranges.
  for (uint32_t i = start + 2U * thread; i < end; i += 2U * SORTED_THREADS) {
    const uint32_t lo = Mtp3LookupEncode(resident[i], stable, shift, count, unique, directory);
    const uint32_t hi = Mtp3LookupEncode(resident[i + 1], stable, shift, count, unique, directory);
    results[i / 2U] = lo | (hi << 16U);
  }
}

__aicore__ inline void Mtp3LookupSlotBody(
    GM_ADDR tokenToReq, GM_ADDR activeRows, GM_ADDR stablePrefixLens,
    GM_ADDR slotToToken, GM_ADDR workspace, GM_ADDR tiling) {
  using namespace mtp3_lookup;
  const auto* d = reinterpret_cast<const __gm__ JointDirectTilingData*>(tiling);
  const int32_t requested = *reinterpret_cast<const __gm__ int32_t*>(activeRows);
  const int64_t rows = requested <= 0 ? 0 : (requested < d->maxRows ? requested : d->maxRows);
  const int64_t tasks = (rows / kRowsPerRequest) * JOINT_MTP3_LOOKUP_SHARDS;
  const int64_t start = AscendC::GetBlockIdx();
  if (start >= tasks) return;

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
  pipe.InitBuffer(storage, kLookupUbBytes);
  auto local = reinterpret_cast<__ubuf__ int32_t*>(storage.Get<int32_t>().GetPhyAddr());
  auto* tokenRows = reinterpret_cast<const __gm__ int32_t*>(tokenToReq);
  auto* stableRows = reinterpret_cast<const __gm__ int32_t*>(stablePrefixLens);
  auto* resident = reinterpret_cast<const __gm__ int32_t*>(slotToToken);
  constexpr uint32_t slots = kCapacity / JOINT_MTP3_LOOKUP_SHARDS;
  for (int64_t task = start; task < tasks; task += AscendC::GetBlockNum()) {
    const int64_t first = (task / JOINT_MTP3_LOOKUP_SHARDS) * kRowsPerRequest;
    if (!Mtp3LookupAlignedGroup(tokenRows, first, rows)) continue;
    auto* shard = reinterpret_cast<__gm__ int32_t*>(workspace) + d->maxRequests * kMapWords +
                  (first / kRowsPerRequest) * kRequestWords;
    if (shard[0] != 1 || shard[2] == 0) continue;
    int32_t stable = stableRows[first];
    for (uint32_t i = 1; i < kRowsPerRequest; ++i) {
      if (stableRows[first + i] < stable) stable = stableRows[first + i];
    }
    if (stable < 0) stable = 0;
    if (stable > d->maxToken) stable = static_cast<int32_t>(d->maxToken);
    const uint32_t offset = static_cast<uint32_t>(task % JOINT_MTP3_LOOKUP_SHARDS) * slots;
    asc_vf_call<Mtp3LookupSlotsVf>(dim3(SORTED_THREADS), local, resident + first * kCapacity, shard,
        offset, offset + slots, static_cast<uint32_t>(shard[1]), stable, static_cast<uint32_t>(shard[3]));
    // A core can own another shard on a subsequent iteration. Drain the VF
    // before it reuses the same local unique/directory buffers.
    auto event = pipe.AllocEventID<AscendC::HardEvent::V_S>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(event);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(event);
    pipe.ReleaseEventID<AscendC::HardEvent::V_S>(event);
  }
}
