// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Host编译为void*；设备编译保留kernel_operator.h定义的GM地址空间。
#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// 仅接入层POD，避免与上游匿名命名空间的类型同名；字段顺序完全保留。
struct SparseKvTransferDirectTilingData {
  int64_t maxRows;
  int64_t maxRequests;
  int64_t maxNumBlocks;
  int64_t topk;
  int64_t capacity;
  int64_t hostNumBlocks;
  int32_t blockSize;
  int32_t tokenSizeBytesK;
  int32_t tokenSizeBytesV;
};
static_assert(sizeof(SparseKvTransferDirectTilingData) == 64, "Transfer tiling ABI");
constexpr uint32_t TRANSFER_DIRECT_UB_BYTES = 2U * 16U * 1024U;
extern "C" uint32_t sparse_kv_transfer_block_dim();
extern "C" void launch_sparse_kv_transfer(
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR tokenToReq, GM_ADDR blockTable,
    GM_ADDR hostCacheBases, GM_ADDR activeRows, GM_ADDR residentK, GM_ADDR residentV,
    GM_ADDR tiling, uint32_t blockDim, void* stream);
