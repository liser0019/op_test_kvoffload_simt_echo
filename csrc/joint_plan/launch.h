// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// A distinct host-side type avoids collision with the preserved kernel-local type.
struct JointDirectTilingData {
  uint64_t hashCapacity, workspaceRowElements;
  int64_t maxRows, topk, capacity, maxToken, maxRequests;
  int32_t blockSize;
  uint32_t localMemoryBytes;
  uint64_t jointHashCapacity, scratchElements;
};
static_assert(sizeof(JointDirectTilingData) == 80, "joint tiling ABI");
extern "C" uint32_t joint_plan_block_dim();
extern "C" void launch_joint_plan(bool jointEntry, GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens,
    GM_ADDR tokenToReq, GM_ADDR activeRows, GM_ADDR lastReqIds,
    GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount,
    GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
    GM_ADDR residentRows, GM_ADDR residentBlockTable, GM_ADDR tiling,
    uint32_t blockDim, uint32_t dynamicUb, void* stream);
