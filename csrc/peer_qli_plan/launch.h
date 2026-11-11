// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif
// Exact public layout of upstream qli_fused_plan::TilingData.
struct PeerPlanTiling {
  uint64_t hashCapacity, workspaceRowElements;
  int64_t maxRows, topk, capacity, maxToken, maxRequests, maxNumBlocks, hostNumBlocks;
  int32_t blockSize;
  uint32_t localMemoryBytes;
  uint64_t jointHashCapacity, scratchElements;
  uint32_t maxJointRows;
};
static_assert(sizeof(PeerPlanTiling) == 104, "peer plan tiling ABI");
extern "C" uint32_t joint_plan_block_dim();
extern "C" void launch_peer_qli_plan(
    GM_ADDR reqIds,
    GM_ADDR topkIndices,
    GM_ADDR stablePrefixLens,
    GM_ADDR visibleSeqLens,
    GM_ADDR tokenToReq,
    GM_ADDR activeRows,
    GM_ADDR lastReqIds,
    GM_ADDR slotToToken,
    GM_ADDR lruSlots,
    GM_ADDR currentSlots,
    GM_ADDR missCount,
    GM_ADDR missTokens,
    GM_ADDR missSlots,
    GM_ADDR compactWorkspace,
    GM_ADDR rowModes,
    GM_ADDR residentRows,
    GM_ADDR residentBlockTable,
    GM_ADDR blockTable,
    GM_ADDR tiling,
    uint32_t blocks, uint32_t dynamicUb, void* stream);
