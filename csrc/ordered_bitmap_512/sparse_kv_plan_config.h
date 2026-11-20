/*
 * Sparse KV Plan with ordered queries and bitmap resident lookup.
 * Source revision: ba4c5dd9bb6aaa432abf4822607ce224bec51862.
 *
 * R requests, W=1/2/4 consecutive query rows per request, Q=R*W, T=2048,
 * C=4096/4096/8192 respectively, P=C/blockSize. maxToken <= 358400.
 * Valid row token IDs are strictly ascending. Padding -1 may occupy one
 * contiguous run at either end. visibleSeqLens and stablePrefixLens are
 * exclusive token bounds. tokenToReq holds the request index for each row.
 *
 * Input tensors:
 *   reqIds                            int64[R]
 *   topkIndices                       int32[Q,T]
 *   stablePrefixLens, visibleSeqLens   int32[Q]
 *   tokenToReq                        int32[Q]
 *   activeRows                        int32[1], whole groups, 0 <= value <= Q
 * Mutable tensors:
 *   lastReqIds                        int64[Q]
 *   slotToToken, lruSlots              int32[Q,C]
 *   currentSlots, missTokens, missSlots int32[Q,T]
 *   missCount, rowModes, residentRows  int32[Q]
 *   residentBlockTable                int32[Q,P]
 *   compactWorkspace                  int32[Q*T/2]
 *
 * Use MakeJointTiling to obtain the workspace size and an 80-byte tiling
 * value. Upload tiling to the device, calculate blockDim with
 * OrderedBitmapBlockDim(R), and launch with jointEntry=true even for MTP0.
 * Pass tiling.localMemoryBytes for dynamicUb. Keep tensors, workspace and
 * tiling alive until this stream completes. Concurrent calls need separate
 * mutable state and workspace.
 *
 * Once per workspace initialization, set its first Q words to INT32_MAX.
 * Device failures write -1 at the rejected group's first row (or at index 0
 * for invalid activeRows). Check these words at the caller's synchronization
 * boundary before consuming results. A failed workspace stays marked until
 * initialized again. Successful paths, including capacity fallback, leave
 * these words untouched. The 4K/8K hot paths keep query-to-UID data in UB.
 *
 * A fresh cache has lastReqIds/slotToToken=-1, rowModes=0, and identity LRUs.
 * Existing cache state may be passed through using the same mode/slot ABI.
 * rowModes 0/1/2 denote private/joint leader/follower. Interpret currentSlots
 * against the cache row in residentRows and consume only each missCount
 * prefix of the descriptors. Use stream order or an event to feed Transfer.
 *
 * Compile on CANN 9.2.1 / DAV_3510:
 *   bisheng --npu-arch=dav-3510 -xasc -std=c++17 -fPIC -O3 --cce-auto-sync=off
 *     -Wno-deprecated-declarations -mllvm -cce-vf-remove-membar=false
 *     -mllvm -cce-aicore-hoist-movemask=false -I<CANN/include> -I<CANN/asc/include>
 *     -c sparse_kv_plan.cpp -o ordered_bitmap_512.o
 * Link with the caller's CANN platform and runtime libraries.
 */

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef VLLM_ASCEND_SPARSE_KV_PLAN_CONFIG_H
#define VLLM_ASCEND_SPARSE_KV_PLAN_CONFIG_H

#include <climits>
#include <cstdint>

namespace sparse_kv_plan_config {

constexpr uint32_t PLAN_THREADS = 1024U;
constexpr uint32_t PLAN_WARP_SIZE = 32U;
constexpr uint32_t PLAN_WARP_COUNT = PLAN_THREADS / PLAN_WARP_SIZE;
constexpr uint32_t PLAN_CONTROL_ELEMENTS = 8U;
constexpr uint32_t PLAN_SCAN_ELEMENTS = 2U * PLAN_WARP_COUNT + PLAN_CONTROL_ELEMENTS;
constexpr uint32_t PLAN_SCAN_BYTES = (PLAN_SCAN_ELEMENTS * sizeof(int32_t) + 31U) & ~31U;
constexpr uint32_t PLAN_SCAN_STORAGE_ELEMENTS = PLAN_SCAN_BYTES / sizeof(int32_t);
constexpr uint32_t PLAN_ROW_UB_LIMIT_BYTES = 112U * 1024U;
constexpr uint32_t PLAN_ROW_UB_LIMIT_ELEMENTS = PLAN_ROW_UB_LIMIT_BYTES / sizeof(int32_t);
constexpr uint32_t PLAN_MAX_DYNAMIC_UB_BYTES = PLAN_SCAN_BYTES + PLAN_ROW_UB_LIMIT_BYTES;

static_assert(PLAN_THREADS > 0U, "Sparse KV Plan requires at least one thread");
static_assert(PLAN_THREADS % PLAN_WARP_SIZE == 0U, "Sparse KV Plan threads must contain whole warps");
static_assert(PLAN_SCAN_BYTES % 32U == 0U, "Sparse KV Plan scan workspace must be 32-byte aligned");
static_assert(PLAN_ROW_UB_LIMIT_BYTES % sizeof(int32_t) == 0U, "Sparse KV Plan row UB limit must be int32 aligned");
static_assert(PLAN_MAX_DYNAMIC_UB_BYTES <= 120U * 1024U,
              "Sparse KV Plan dynamic UB exceeds the full-DCache budget");

}  // namespace sparse_kv_plan_config

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// A distinct host-side type avoids collision with the preserved kernel-local type.
struct JointDirectTilingData {
  uint64_t hashCapacity;
  uint64_t workspaceRowElements;
  int64_t maxRows;
  int64_t topk;
  int64_t capacity;
  int64_t maxToken;
  int64_t maxRequests;
  int32_t blockSize;
  uint32_t localMemoryBytes;
  uint64_t jointHashCapacity;
  uint64_t scratchElements;
};
static_assert(sizeof(JointDirectTilingData) == 80, "joint tiling ABI");
extern "C" uint32_t joint_plan_block_dim_ordered_bitmap_512();
extern "C" void launch_joint_plan_ordered_bitmap_512(
    bool jointEntry, GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens,
    GM_ADDR tokenToReq, GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots,
    GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace,
    GM_ADDR rowModes, GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling, uint32_t blockDim,
    uint32_t dynamicUb, void* stream);

namespace sparse_kv_plan_config {

inline bool MakeJointTiling(int64_t maxRequests, int64_t maxRows, int64_t capacity, int64_t maxToken,
                            int32_t blockSize, JointDirectTilingData& tiling, uint64_t& workspaceInt32) {
  if (maxRequests <= 0 || maxRows <= 0 || maxRows % maxRequests != 0 || maxToken <= 0 || maxToken > 358400 ||
      blockSize <= 0) {
    return false;
  }
  const int64_t rowsPerRequest = maxRows / maxRequests;
  if ((rowsPerRequest != 1 && rowsPerRequest != 2 && rowsPerRequest != 4) ||
      capacity != (rowsPerRequest == 4 ? 8192 : 4096) || capacity % blockSize != 0 ||
      maxRows > INT32_MAX / (capacity / blockSize)) {
    return false;
  }

  constexpr uint64_t topK = 2048;
  constexpr uint64_t hashCapacity = 4096;
  const uint64_t residentCapacity = static_cast<uint64_t>(capacity);
  const uint64_t mergeWorkspaceWords = rowsPerRequest == 4 ? 32768U : 24576U;
  const uint64_t scratchWords = 128U + residentCapacity / 2U + residentCapacity + mergeWorkspaceWords;
  const uint32_t dynamicUb = PLAN_SCAN_BYTES + static_cast<uint32_t>(scratchWords * sizeof(int32_t));
  tiling = {hashCapacity, 3U * hashCapacity + residentCapacity + topK,
            maxRows,      topK,
            capacity,     maxToken,
            maxRequests,  blockSize,
            dynamicUb,    2U * residentCapacity,
            scratchWords};
  workspaceInt32 = static_cast<uint64_t>(maxRows) * topK / 2U;
  return true;
}

inline uint32_t OrderedBitmapBlockDim(int64_t maxRequests) {
  const uint32_t available = joint_plan_block_dim_ordered_bitmap_512();
  return maxRequests <= 0 ? 0U : (maxRequests < available ? static_cast<uint32_t>(maxRequests) : available);
}

}  // namespace sparse_kv_plan_config

#endif
