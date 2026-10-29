/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

// 接入说明：计算类逐字保留；仅替换注册入口/tiling读取并增加direct launch。
// 原始文件与计算区指纹由transfer_manifest.json核对。
#include <cstdint>
#include "kernel_operator.h"
#include "launch.h"
#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(type)
#endif

namespace {

constexpr uint32_t TRANSFER_BUFFER_COUNT = 2U;
constexpr uint32_t TRANSFER_CHUNK_BYTES = 16U * 1024U;
static_assert(TRANSFER_CHUNK_BYTES % 32U == 0U, "UB chunk must be 32-byte aligned");


class SparseKvTransferRuntimeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR tokenToReq,
                              GM_ADDR blockTable, GM_ADDR hostCacheBases, GM_ADDR deviceKBase, GM_ADDR deviceVBase,
                              int64_t numRows, int64_t maxRequests, int64_t topk, int64_t capacity,
                              int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize, int32_t tokenSizeBytesK,
                              int32_t tokenSizeBytesV) {
    aivNum_ = AscendC::GetBlockNum();
    aivIndex_ = AscendC::GetBlockIdx();
    missCount_ = reinterpret_cast<__gm__ int32_t*>(missCount);
    missTokens_ = reinterpret_cast<__gm__ int32_t*>(missTokens);
    missSlots_ = reinterpret_cast<__gm__ int32_t*>(missSlots);
    tokenToReq_ = reinterpret_cast<__gm__ int32_t*>(tokenToReq);
    blockTable_ = reinterpret_cast<__gm__ int32_t*>(blockTable);
    const __gm__ int64_t* bases = reinterpret_cast<__gm__ int64_t*>(hostCacheBases);
    hostKBase_ = static_cast<uint64_t>(bases[0]);
    hostVBase_ = static_cast<uint64_t>(bases[1]);
    deviceKBase_ = reinterpret_cast<uint64_t>(deviceKBase);
    deviceVBase_ = reinterpret_cast<uint64_t>(deviceVBase);
    numRows_ = numRows;
    maxRequests_ = maxRequests;
    topk_ = topk;
    capacity_ = capacity;
    maxNumBlocks_ = maxNumBlocks;
    hostNumBlocks_ = hostNumBlocks;
    blockSize_ = blockSize;
    tokenSizeBytesK_ = tokenSizeBytesK;
    tokenSizeBytesV_ = tokenSizeBytesV;
    blockBytesK_ = static_cast<uint64_t>(blockSize) * tokenSizeBytesK;
    blockBytesV_ = static_cast<uint64_t>(blockSize) * tokenSizeBytesV;

    pipe_.InitBuffer(copyQueue_, TRANSFER_BUFFER_COUNT, TRANSFER_CHUNK_BYTES);
    head_ = 0U;
    tail_ = 0U;
    pending_ = 0U;
  }

  __aicore__ inline void Process() {
    uint32_t prefixMod = 0U;
    for (int64_t row = 0; row < numRows_; ++row) {
      int32_t count = missCount_[row];
      if (count < 0) {
        count = 0;
      } else if (static_cast<int64_t>(count) > topk_) {
        count = static_cast<int32_t>(topk_);
      }
      const uint32_t first = (aivIndex_ + aivNum_ - prefixMod) % aivNum_;
      prefixMod = (prefixMod + static_cast<uint32_t>(count) % aivNum_) % aivNum_;
      const int64_t rowMissBase = row * topk_;
      const int32_t request = tokenToReq_[row];
      if (request < 0 || request >= maxRequests_) {
        continue;
      }
      const int64_t rowBlockBase = static_cast<int64_t>(request) * maxNumBlocks_;

      for (int64_t index = first; index < count; index += aivNum_) {
        const int32_t token = missTokens_[rowMissBase + index];
        const int32_t slot = missSlots_[rowMissBase + index];
        if (token < 0 || slot < 0 || static_cast<int64_t>(slot) >= capacity_) {
          continue;
        }
        const int32_t blockId = token / blockSize_;
        if (static_cast<int64_t>(blockId) >= maxNumBlocks_) {
          continue;
        }
        const int32_t blockIndex = blockTable_[rowBlockBase + blockId];
        if (blockIndex < 0 || blockIndex >= hostNumBlocks_) {
          continue;
        }
        const int32_t offsetInBlock = token - blockId * blockSize_;
        const uint64_t sourceK = hostKBase_ + static_cast<uint64_t>(blockIndex) * blockBytesK_ +
                                 static_cast<uint64_t>(offsetInBlock) * tokenSizeBytesK_;
        const uint64_t sourceV = hostVBase_ + static_cast<uint64_t>(blockIndex) * blockBytesV_ +
                                 static_cast<uint64_t>(offsetInBlock) * tokenSizeBytesV_;
        const uint64_t linearSlot = static_cast<uint64_t>(row) * capacity_ + slot;
        const uint64_t destinationK = deviceKBase_ + linearSlot * tokenSizeBytesK_;
        const uint64_t destinationV = deviceVBase_ + linearSlot * tokenSizeBytesV_;

        CopyBytes(sourceK, destinationK, static_cast<uint32_t>(tokenSizeBytesK_));
        CopyBytes(sourceV, destinationV, static_cast<uint32_t>(tokenSizeBytesV_));
      }
    }
    while (pending_ != 0U) {
      DrainOne();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }

 private:
  __aicore__ inline void CopyBytes(uint64_t source, uint64_t destination, uint32_t bytes) {
    uint32_t offset = 0U;
    while (offset < bytes) {
      const uint32_t remaining = bytes - offset;
      const uint32_t chunk = remaining < TRANSFER_CHUNK_BYTES ? remaining : TRANSFER_CHUNK_BYTES;
      SubmitChunk(source + offset, destination + offset, chunk);
      offset += chunk;
    }
  }

  __aicore__ inline void SubmitChunk(uint64_t source, uint64_t destination, uint32_t bytes) {
    AscendC::GlobalTensor<uint8_t> sourceGm;
    sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source), bytes);
    AscendC::LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams copyParams(1, bytes, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> padParams{};
    AscendC::DataCopyPad(local, sourceGm, copyParams, padParams);
    copyQueue_.EnQue(local);

    destinations_[tail_] = destination;
    lengths_[tail_] = bytes;
    tail_ = (tail_ + 1U) & 1U;
    ++pending_;
    if (pending_ == TRANSFER_BUFFER_COUNT) {
      DrainOne();
    }
  }

  __aicore__ inline void DrainOne() {
    const uint32_t bytes = lengths_[head_];
    AscendC::GlobalTensor<uint8_t> destinationGm;
    destinationGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destinations_[head_]), bytes);
    AscendC::LocalTensor<uint8_t> local = copyQueue_.DeQue<uint8_t>();
    AscendC::DataCopyExtParams copyParams(1, bytes, 0, 0, 0);
    AscendC::DataCopyPad(destinationGm, local, copyParams);
    copyQueue_.FreeTensor(local);
    head_ = (head_ + 1U) & 1U;
    --pending_;
  }

  AscendC::TPipe pipe_;
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> copyQueue_;
  uint64_t destinations_[TRANSFER_BUFFER_COUNT] = {};
  uint32_t lengths_[TRANSFER_BUFFER_COUNT] = {};
  uint32_t head_ = 0U;
  uint32_t tail_ = 0U;
  uint32_t pending_ = 0U;
  uint32_t aivNum_ = 0U;
  uint32_t aivIndex_ = 0U;
  __gm__ int32_t* missCount_ = nullptr;
  __gm__ int32_t* missTokens_ = nullptr;
  __gm__ int32_t* missSlots_ = nullptr;
  __gm__ int32_t* tokenToReq_ = nullptr;
  __gm__ int32_t* blockTable_ = nullptr;
  uint64_t hostKBase_ = 0U;
  uint64_t hostVBase_ = 0U;
  uint64_t deviceKBase_ = 0U;
  uint64_t deviceVBase_ = 0U;
  int64_t numRows_ = 0;
  int64_t maxRequests_ = 0;
  int64_t topk_ = 0;
  int64_t capacity_ = 0;
  int64_t maxNumBlocks_ = 0;
  int64_t hostNumBlocks_ = 0;
  int32_t blockSize_ = 0;
  int32_t tokenSizeBytesK_ = 0;
  int32_t tokenSizeBytesV_ = 0;
  uint64_t blockBytesK_ = 0U;
  uint64_t blockBytesV_ = 0U;
};

}

extern "C" __global__ __aicore__ void sparse_kv_transfer_kernel(
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR tokenToReq,
    GM_ADDR blockTable, GM_ADDR hostCacheBases, GM_ADDR activeRows,
    GM_ADDR residentK, GM_ADDR residentV, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  const __gm__ SparseKvTransferDirectTilingData* gm =
      reinterpret_cast<const __gm__ SparseKvTransferDirectTilingData*>(tiling);
  // 逐字段加载，避免CANN9.1跨地址空间整结构体复制的问题。
  SparseKvTransferDirectTilingData data;
  data.maxRows = gm->maxRows;
  data.maxRequests = gm->maxRequests;
  data.maxNumBlocks = gm->maxNumBlocks;
  data.topk = gm->topk;
  data.capacity = gm->capacity;
  data.hostNumBlocks = gm->hostNumBlocks;
  data.blockSize = gm->blockSize;
  data.tokenSizeBytesK = gm->tokenSizeBytesK;
  data.tokenSizeBytesV = gm->tokenSizeBytesV;
  const int32_t requested = *reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t rows = requested > 0 ? (requested < data.maxRows ? requested : data.maxRows) : 0;
  SparseKvTransferRuntimeKernel kernel;
  kernel.Init(missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases,
              residentK, residentV, rows, data.maxRequests, data.topk, data.capacity,
              data.maxNumBlocks, data.hostNumBlocks, data.blockSize,
              data.tokenSizeBytesK, data.tokenSizeBytesV);
  kernel.Process();  // 0 miss也真实下发，测量其空工作控制开销。
}

#include "platform/platform_ascendc.h"
extern "C" uint32_t sparse_kv_transfer_block_dim() {
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const uint32_t cores = platform == nullptr ? 0U : static_cast<uint32_t>(platform->GetCoreNumAiv());
  // 与上游tiling相同的min(64, AIV数)，查询失败则明确拒绝。
  return cores < 64U ? cores : 64U;
}
extern "C" void launch_sparse_kv_transfer(
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR tokenToReq, GM_ADDR blockTable,
    GM_ADDR hostCacheBases, GM_ADDR activeRows, GM_ADDR residentK, GM_ADDR residentV,
    GM_ADDR tiling, uint32_t blockDim, void* stream) {
  sparse_kv_transfer_kernel<<<blockDim, TRANSFER_DIRECT_UB_BYTES, stream>>>(
      missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases,
      activeRows, residentK, residentV, tiling);
}
