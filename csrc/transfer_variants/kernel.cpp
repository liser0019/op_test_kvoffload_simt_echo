/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * 派生自SparseKvTransfer：A/B/C/D通过模板位独立组合，保留原版另行编译。
 */
#include "kernel_operator.h"
#include "launch.h"
#include "original_class.h"
#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(type)
#endif

namespace {
constexpr uint32_t BUFFER_BYTES = 16U * 1024U;
constexpr uint32_t BUFFER_COUNT = 2U;

struct TokenAddress {
  uint64_t sourceK, sourceV, destinationK, destinationV;
};

// A: K/V共用queue记录；B:批8token；C:生产常量；D:全局连续miss区间。
// 输入source/destination仍可任意散布。批次减少queue/event管理，未假装合成连续DMA。
template <uint32_t Mask, bool Production>
class TransferOptimized {
  static constexpr bool Pair = (Mask & 1U) != 0U;
  static constexpr bool Contiguous = (Mask & 8U) != 0U;
  static constexpr uint32_t Batch = (Mask & 2U) ? 8U : 1U;
 public:
  __aicore__ inline void Init(
      GM_ADDR counts, GM_ADDR tokens, GM_ADDR slots, GM_ADDR requests, GM_ADDR table, GM_ADDR bases,
      GM_ADDR destinationK, GM_ADDR destinationV, int64_t rows, const SparseKvTransferDirectTilingData& data) {
    counts_ = reinterpret_cast<__gm__ int32_t*>(counts);
    tokens_ = reinterpret_cast<__gm__ int32_t*>(tokens);
    slots_ = reinterpret_cast<__gm__ int32_t*>(slots);
    requests_ = reinterpret_cast<__gm__ int32_t*>(requests);
    table_ = reinterpret_cast<__gm__ int32_t*>(table);
    const auto* source = reinterpret_cast<const __gm__ int64_t*>(bases);
    hostK_ = static_cast<uint64_t>(source[0]);
    hostV_ = static_cast<uint64_t>(source[1]);
    deviceK_ = reinterpret_cast<uint64_t>(destinationK);
    deviceV_ = reinterpret_cast<uint64_t>(destinationV);
    rows_ = rows; data_ = data;
    core_ = AscendC::GetBlockIdx(); cores_ = AscendC::GetBlockNum();
    pitchK_ = (BytesK() + 31U) & ~31U;
    pitchV_ = (BytesV() + 31U) & ~31U;
    pipe_.InitBuffer(queue_, BUFFER_COUNT, BUFFER_BYTES);
  }

  __aicore__ inline void Process() {
    int64_t rangeBegin = 0, rangeEnd = 0;
    if constexpr (Contiguous) {
      int64_t total = 0;
      for (int64_t row = 0; row < rows_; ++row) total += Count(row);
      const int64_t perCore = total / cores_, extra = total % cores_;
      // 避免total*core溢出；每核分配数量相差至多1。
      rangeBegin = perCore * core_ + (core_ < extra ? core_ : extra);
      rangeEnd = rangeBegin + perCore + (core_ < extra ? 1 : 0);
    }
    int64_t prefix = 0;
    uint32_t prefixMod = 0;
    for (int64_t row = 0; row < rows_; ++row) {
      const int32_t count = Count(row);
      int64_t begin, end = count, step;
      if constexpr (Contiguous) {
        begin = rangeBegin > prefix ? rangeBegin - prefix : 0;
        end = rangeEnd < prefix + count ? rangeEnd - prefix : count;
        prefix += count;
        step = 1;
      } else {
        begin = (core_ + cores_ - prefixMod) % cores_;
        prefixMod = (prefixMod + static_cast<uint32_t>(count) % cores_) % cores_;
        step = cores_;
      }
      const int32_t request = requests_[row];
      if (request < 0 || request >= data_.maxRequests || begin >= end) continue;
      TokenAddress addresses[Batch];
      uint32_t valid = 0;
      for (int64_t index = begin; index < end; index += step) {
        TokenAddress address;
        if (!Resolve(row, request, index, address)) continue;
        addresses[valid++] = address;
        if (valid == Batch) {
          SubmitBatch(addresses, valid);
          valid = 0;
        }
      }
      if (valid != 0) SubmitBatch(addresses, valid);  // 尾批只搬已解析合法项。
    }
    while (pending_ != 0U) Drain();
    AscendC::PipeBarrier<PIPE_ALL>();
  }

 private:
  __aicore__ inline uint32_t BytesK() const {
    if constexpr (Production) return 1024U;
    return static_cast<uint32_t>(data_.tokenSizeBytesK);
  }
  __aicore__ inline uint32_t BytesV() const {
    if constexpr (Production) return 128U;
    return static_cast<uint32_t>(data_.tokenSizeBytesV);
  }
  __aicore__ inline int32_t Count(int64_t row) const {
    const int32_t value = counts_[row];
    return value < 0 ? 0 : (value > data_.topk ? static_cast<int32_t>(data_.topk) : value);
  }
  __aicore__ inline bool Resolve(int64_t row, int32_t request, int64_t index, TokenAddress& out) {
    const int64_t offset = row * data_.topk + index;
    const int32_t token = tokens_[offset], slot = slots_[offset];
    if (token < 0 || slot < 0 || slot >= data_.capacity) return false;
    int32_t block, within;
    if constexpr (Production) {
      block = token >> 6; within = token & 63;
    } else {
      block = token / data_.blockSize;
      within = token - block * data_.blockSize;
    }
    if (block >= data_.maxNumBlocks) return false;
    const int32_t physical = table_[static_cast<int64_t>(request) * data_.maxNumBlocks + block];
    if (physical < 0 || physical >= data_.hostNumBlocks) return false;
    const uint64_t sourceToken = static_cast<uint64_t>(physical) *
        (Production ? 64U : static_cast<uint32_t>(data_.blockSize)) + within;
    const uint64_t destinationToken = static_cast<uint64_t>(row) * data_.capacity + slot;
    out.sourceK = hostK_ + sourceToken * BytesK();
    out.sourceV = hostV_ + sourceToken * BytesV();
    out.destinationK = deviceK_ + destinationToken * BytesK();
    out.destinationV = deviceV_ + destinationToken * BytesV();
    return true;
  }
  __aicore__ inline void Read(AscendC::LocalTensor<uint8_t> local, uint32_t offset,
                              uint64_t source, uint32_t bytes) {
    AscendC::GlobalTensor<uint8_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source), bytes);
    AscendC::DataCopyExtParams params(1, bytes, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> pad{};
    AscendC::DataCopyPad(local[offset], gm, params, pad);
  }
  __aicore__ inline void Write(AscendC::LocalTensor<uint8_t> local, uint32_t offset,
                               uint64_t destination, uint32_t bytes) {
    AscendC::GlobalTensor<uint8_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destination), bytes);
    AscendC::DataCopyExtParams params(1, bytes, 0, 0, 0);
    AscendC::DataCopyPad(gm, local[offset], params);
  }
  __aicore__ inline void SubmitBatch(const TokenAddress* addresses, uint32_t count) {
    if constexpr (Pair) {
      Submit(addresses, count, 2U);  // 一个queue记录覆盖K/V。
    } else {
      Submit(addresses, count, 0U);
      Submit(addresses, count, 1U);
    }
  }
  __aicore__ inline void Submit(const TokenAddress* addresses, uint32_t count, uint32_t kind) {
    auto local = queue_.template AllocTensor<uint8_t>();
    const uint32_t vBase = Pair ? Batch * pitchK_ : 0U;
    for (uint32_t i = 0; i < count; ++i) {
      if (kind != 1U) {
        Read(local, i * pitchK_, addresses[i].sourceK, BytesK());
        destinationsK_[tail_][i] = addresses[i].destinationK;
      }
      if (kind != 0U) {
        Read(local, vBase + i * pitchV_, addresses[i].sourceV, BytesV());
        destinationsV_[tail_][i] = addresses[i].destinationV;
      }
    }
    kinds_[tail_] = kind; countsInQueue_[tail_] = count;
    queue_.EnQue(local);
    tail_ = (tail_ + 1U) & 1U;
    if (++pending_ == BUFFER_COUNT) Drain();
  }
  __aicore__ inline void Drain() {
    const uint32_t kind = kinds_[head_], count = countsInQueue_[head_];
    const uint32_t vBase = Pair ? Batch * pitchK_ : 0U;
    auto local = queue_.template DeQue<uint8_t>();
    for (uint32_t i = 0; i < count; ++i) {
      if (kind != 1U) Write(local, i * pitchK_, destinationsK_[head_][i], BytesK());
      if (kind != 0U) Write(local, vBase + i * pitchV_, destinationsV_[head_][i], BytesV());
    }
    queue_.FreeTensor(local);  // 等待相应搬出后才可复用，不删除队列同步。
    head_ = (head_ + 1U) & 1U; --pending_;
  }

  AscendC::TPipe pipe_;
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> queue_;
  SparseKvTransferDirectTilingData data_{};
  __gm__ int32_t *counts_, *tokens_, *slots_, *requests_, *table_;
  uint64_t hostK_, hostV_, deviceK_, deviceV_;
  int64_t rows_;
  uint32_t core_, cores_, pitchK_, pitchV_;
  uint32_t head_ = 0, tail_ = 0, pending_ = 0;
  uint64_t destinationsK_[2][Batch], destinationsV_[2][Batch];
  uint32_t kinds_[2], countsInQueue_[2];
};

__aicore__ inline SparseKvTransferDirectTilingData Load(GM_ADDR tiling) {
  const auto* gm = reinterpret_cast<const __gm__ SparseKvTransferDirectTilingData*>(tiling);
  SparseKvTransferDirectTilingData d;
  d.maxRows = gm->maxRows; d.maxRequests = gm->maxRequests; d.maxNumBlocks = gm->maxNumBlocks;
  d.topk = gm->topk; d.capacity = gm->capacity; d.hostNumBlocks = gm->hostNumBlocks;
  d.blockSize = gm->blockSize; d.tokenSizeBytesK = gm->tokenSizeBytesK; d.tokenSizeBytesV = gm->tokenSizeBytesV;
  return d;  // 已逐字段读入本地POD；返回不跨GM地址空间。
}

template <uint32_t Mask, bool Production>
__aicore__ inline void Process(
    GM_ADDR counts, GM_ADDR tokens, GM_ADDR slots, GM_ADDR requests, GM_ADDR table, GM_ADDR bases,
    GM_ADDR active, GM_ADDR k, GM_ADDR v, const SparseKvTransferDirectTilingData& d) {
  const int32_t requested = *reinterpret_cast<const __gm__ int32_t*>(active);
  const int64_t rows = requested > 0 ? (requested < d.maxRows ? requested : d.maxRows) : 0;
  constexpr uint32_t batch = (Mask & 2U) ? 8U : 1U;
  const uint64_t pitchK = (static_cast<uint64_t>(d.tokenSizeBytesK) + 31U) & ~uint64_t{31U};
  const uint64_t pitchV = (static_cast<uint64_t>(d.tokenSizeBytesV) + 31U) & ~uint64_t{31U};
  const uint64_t required = batch * ((Mask & 1U) ? pitchK + pitchV :
                                     (pitchK > pitchV ? pitchK : pitchV));
  if (required > BUFFER_BYTES) {
    transfer_original::SparseKvTransferRuntimeKernel original;
    original.Init(counts, tokens, slots, requests, table, bases, k, v, rows, d.maxRequests,
                  d.topk, d.capacity, d.maxNumBlocks, d.hostNumBlocks, d.blockSize,
                  d.tokenSizeBytesK, d.tokenSizeBytesV);
    original.Process();  // 大尺寸保留原始16KiB分块，不越界、不修改语义。
  } else {
    TransferOptimized<Mask, Production> op;
    op.Init(counts, tokens, slots, requests, table, bases, k, v, rows, d);
    op.Process();
  }
}
}  // namespace

// 每个组合有独立设备kernel名，profiler不会将不同版本混计。
#define TRANSFER_VARIANT(mask, name) \
extern "C" __global__ __aicore__ void sparse_kv_transfer_kernel_opt_##name( \
    GM_ADDR counts, GM_ADDR tokens, GM_ADDR slots, GM_ADDR requests, GM_ADDR table, GM_ADDR bases, \
    GM_ADDR active, GM_ADDR k, GM_ADDR v, GM_ADDR tiling) { \
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
  AscendC::InitSocState(); \
  const auto data = Load(tiling); \
  if constexpr (((mask) & 4U) != 0U) { \
    if (data.blockSize == 64 && data.tokenSizeBytesK == 1024 && data.tokenSizeBytesV == 128) \
      Process<mask, true>(counts, tokens, slots, requests, table, bases, active, k, v, data); \
    else Process<mask, false>(counts, tokens, slots, requests, table, bases, active, k, v, data); \
  } else Process<mask, false>(counts, tokens, slots, requests, table, bases, active, k, v, data); \
}
#include "entries.inc"
#undef TRANSFER_VARIANT

extern "C" void launch_sparse_kv_transfer_variant(
    int32_t mask, GM_ADDR counts, GM_ADDR tokens, GM_ADDR slots, GM_ADDR requests, GM_ADDR table,
    GM_ADDR bases, GM_ADDR active, GM_ADDR k, GM_ADDR v, GM_ADDR tiling, uint32_t blockDim, void* stream) {
  switch (mask) {
#define TRANSFER_VARIANT(maskValue, name) \
    case maskValue: sparse_kv_transfer_kernel_opt_##name<<<blockDim, TRANSFER_DIRECT_UB_BYTES, stream>>>( \
        counts, tokens, slots, requests, table, bases, active, k, v, tiling); return;
#include "entries.inc"
#undef TRANSFER_VARIANT
  }
}
