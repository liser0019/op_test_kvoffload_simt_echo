// SPDX-License-Identifier: Apache-2.0
#pragma once

// Valid after the merge packet is consumed. The leader retains its unique
// tokens and packed query map in UB, even while the helper is running.
namespace bitmap_team {
constexpr uint32_t kCapacity = 8192;
constexpr uint32_t kHalf = kCapacity / 2;
constexpr uint32_t kResultWords = kHalf / 2;  // 2 packed slots per int32.
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kHeader = 8;
constexpr uint32_t kUniqueOffset = PLAN_SCAN_STORAGE_ELEMENTS + EIGHT_CTX + kCapacity / 2;
constexpr uint32_t kOwnerOffset = kUniqueOffset + kCapacity;
constexpr uint32_t kBufferOffset = PLAN_SCAN_STORAGE_ELEMENTS + EIGHT_CTX;
constexpr uint32_t kRequestWords = kResultWords +
    (JOINT_MTP3_BITMAP_TEAM == 1 ? kHeader + kCapacity : 0);
constexpr uint32_t kResultOffset = kRequestWords - kResultWords;
}  // namespace bitmap_team

__simt_callee__ __aicore__ inline uint32_t BitmapTeamEncode(
    int32_t token, int32_t stable, const __ubuf__ uint32_t* masks,
    const __ubuf__ uint32_t* firstUid) {
  if (token >= stable) return 0x8000U;
  if (token < 0) return 0;
  const uint32_t word = static_cast<uint32_t>(token) >> 5U;
  const uint32_t bit = uint32_t{1} << (static_cast<uint32_t>(token) & 31U);
  const uint32_t mask = masks[word];
  return (mask & bit) ? (0x4000U | (firstUid[word] + __popc(mask & (bit - 1U)))) : 0;
}

// Share rebuilds an exact index from a compact sorted unique list.
// Direct independently derives the same ranks from the original TopK.
template<bool Direct>
__simt_vf__ __aicore__ LAUNCH_BOUND(SORTED_THREADS) inline void BitmapTeamLookupVf(
    __ubuf__ int32_t* scan, const __gm__ int32_t* topk, const __gm__ int32_t* visible,
    const __gm__ int32_t* resident, int32_t stable, uint32_t maxToken, uint32_t count) {
  using namespace bitmap_team;
  const uint32_t t = threadIdx.x;
  const uint32_t words = (maxToken + 31U) / 32U;
  const uint32_t span = (words + 7U) & ~7U;
  __ubuf__ int32_t* list = scan + kBufferOffset;
  auto* masks = reinterpret_cast<__ubuf__ uint32_t*>(list + kCapacity);
  auto* firstUid = masks + span;
  auto* result = firstUid + span;
  for (uint32_t w = t; w < words; w += SORTED_THREADS) masks[w] = 0;
  asc_syncthreads();
  if constexpr (Direct) {
    for (uint32_t i = t; i < 4U * kTopK; i += SORTED_THREADS) {
      const int32_t token = topk[i];
      if (token >= 0 && token < visible[i / kTopK] && static_cast<uint32_t>(token) < maxToken) {
        asc_atomic_or(masks + (static_cast<uint32_t>(token) >> 5U),
                      uint32_t{1} << (static_cast<uint32_t>(token) & 31U));
      }
    }
    asc_syncthreads();
    // Each lane scans a consecutive range. A single block scan provides the
    // prefix across ranges, preserving the ascending token UID used by merge.
    uint32_t perLane = (words + SORTED_THREADS - 1U) / SORTED_THREADS;
#if JOINT_MTP3_BITMAP_ODD_STRIDE
    // Keep the ranges consecutive while varying the per-lane UB access stride.
    perLane |= 1U;
#endif
    const uint32_t begin = t * perLane;
    const uint32_t end = begin + perLane < words ? begin + perLane : words;
    int32_t own = 0;
    for (uint32_t w = begin; w < end; ++w) {
      firstUid[w] = static_cast<uint32_t>(own);
      own += __popc(masks[w]);
    }
    uint32_t parity = 0;
    int32_t total = 0;
    const uint32_t base = SortedScan(scan, own, t, parity, total);
    for (uint32_t w = begin; w < end; ++w) firstUid[w] += base;
  } else {
    for (uint32_t uid = t; uid < count; uid += SORTED_THREADS) {
      const uint32_t word = static_cast<uint32_t>(list[uid]) >> 5U;
      if (uid != 0U && (static_cast<uint32_t>(list[uid - 1U]) >> 5U) == word) continue;
      uint32_t value = 0U;
      for (uint32_t j = uid; j < count && (static_cast<uint32_t>(list[j]) >> 5U) == word; ++j) {
        value |= uint32_t{1} << (static_cast<uint32_t>(list[j]) & 31U);
      }
      firstUid[word] = uid;
      masks[word] = value;
    }
  }
  asc_syncthreads();
  for (uint32_t i = t; i < kResultWords; i += SORTED_THREADS) {
    const uint32_t lo = BitmapTeamEncode(resident[kHalf + 2U*i], stable, masks, firstUid);
    const uint32_t hi = BitmapTeamEncode(resident[kHalf + 2U*i + 1U], stable, masks, firstUid);
    result[i] = lo | (hi << 16U);
  }
}

template<AscendC::HardEvent Event>
__aicore__ inline void BitmapTeamFence(AscendC::TPipe& pipe) {
  const auto id = pipe.AllocEventID<Event>();
  AscendC::SetFlag<Event>(id);
  AscendC::WaitFlag<Event>(id);
  pipe.ReleaseEventID<Event>(id);
}

__aicore__ inline void BitmapTeamWrite(AscendC::TPipe& pipe, __gm__ int32_t* dst,
    AscendC::LocalTensor<int32_t> src, uint32_t count) {
  AscendC::GlobalTensor<int32_t> g;
  g.SetGlobalBuffer(dst, count);
  BitmapTeamFence<AscendC::HardEvent::V_MTE3>(pipe);
  AscendC::DataCopy(g, src, count);
  BitmapTeamFence<AscendC::HardEvent::MTE3_S>(pipe);
}

__aicore__ inline void BitmapTeamRead(AscendC::TPipe& pipe, AscendC::LocalTensor<int32_t> dst,
    __gm__ int32_t* src, uint32_t count) {
  AscendC::GlobalTensor<int32_t> g;
  g.SetGlobalBuffer(src, count);
  AscendC::DataCopy(dst, g, count);
  BitmapTeamFence<AscendC::HardEvent::MTE2_V>(pipe);
  BitmapTeamFence<AscendC::HardEvent::MTE2_S>(pipe);
}

__aicore__ inline bool BitmapTeamEligible(
    const __gm__ int32_t* rowToReq, int64_t rows, int64_t requests, int64_t blocks) {
  if (requests * 2 > blocks || rows != requests * 4) return false;
  for (int64_t first = 0; first < rows; first += 4) {
    const int32_t req = rowToReq[first];
    if (req < 0 || req >= requests || (first != 0 && rowToReq[first - 1] == req) ||
        rowToReq[first + 1] != req || rowToReq[first + 2] != req || rowToReq[first + 3] != req) {
      return false;
    }
  }
  return true;
}

__aicore__ inline bool BitmapTeamHot(int64_t first, const __gm__ int32_t* rowToReq,
    const __gm__ int32_t* modes, const __gm__ int64_t* last, const __gm__ int64_t* requests) {
  return modes[first] == ROW_JOINT && last[first] == requests[rowToReq[first]];
}

__aicore__ inline void BitmapTeamHelper(AscendC::TPipe& pipe, AscendC::LocalTensor<int32_t> local,
    __ubuf__ int32_t* scan, __gm__ int32_t* exchange, int64_t first, uint32_t maxToken,
    const __gm__ int32_t* topk, const __gm__ int32_t* visible, const __gm__ int32_t* stable,
    const __gm__ int32_t* resident) {
  using namespace bitmap_team;
  uint32_t count = 0;
  if constexpr (JOINT_MTP3_BITMAP_TEAM == 1) {
    BitmapTeamRead(pipe, local[kBufferOffset], exchange, kHeader);
    count = static_cast<uint32_t>(scan[kBufferOffset]);
    if (count > kCapacity) return;
    if (count != 0U) {
      BitmapTeamRead(pipe, local[kBufferOffset], exchange + kHeader, (count + 7U) & ~7U);
    }
  }
  int32_t limit = stable[first];
  for (uint32_t r = 1; r < 4; ++r) if (stable[first + r] < limit) limit = stable[first + r];
  if (limit < 0) limit = 0;
  if (limit > static_cast<int32_t>(maxToken)) limit = static_cast<int32_t>(maxToken);
  asc_vf_call<BitmapTeamLookupVf<(JOINT_MTP3_BITMAP_TEAM == 2)>>(
      dim3(SORTED_THREADS), scan, topk + first*kTopK, visible + first,
      resident + first*kCapacity, limit, maxToken, count);
  const uint32_t span = ((maxToken + 31U) / 32U + 7U) & ~7U;
  BitmapTeamWrite(pipe, exchange + kResultOffset, local[kBufferOffset + kCapacity + 2U*span],
                  kResultWords);
}
