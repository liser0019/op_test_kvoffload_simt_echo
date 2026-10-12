// SPDX-License-Identifier: Apache-2.0
// 注册版 tiling 宏的 POD 等价布局；字段类型和顺序不变。
#pragma once
#include <cstdint>
struct SparseKvPlanVariantTilingData {
  uint64_t hashCapacity;
  uint64_t workspaceRowElements;
  int64_t maxRows;
  int64_t topk;
  int64_t capacity;
  int64_t maxToken;
  int64_t maxRequests;
  int64_t maxNumBlocks;
  int64_t hostNumBlocks;
  int32_t blockSize;
  uint32_t localMemoryBytes;
};
static_assert(sizeof(SparseKvPlanVariantTilingData) == 80, "Unexpected tiling ABI");
