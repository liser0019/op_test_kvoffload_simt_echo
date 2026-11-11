// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "launch.h"
#include "upstream/sparse_kv_plan_config.h"
namespace sparse_kv_plan_config = qli_fused_plan::config;
#include "standalone_body.h"
static_assert(sizeof(PeerPlanTiling) == sizeof(SparseKvPlanTilingData), "peer ABI size");
extern "C" __global__ __aicore__ void sparse_kv_plan_peer_qli_kernel(
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
    GM_ADDR tiling) {
  const auto* gm = reinterpret_cast<const __gm__ PeerPlanTiling*>(tiling);
  SparseKvPlanTilingData local{};
  local.hashCapacity = gm->hashCapacity;
  local.workspaceRowElements = gm->workspaceRowElements;
  local.maxRows = gm->maxRows;
  local.topk = gm->topk;
  local.capacity = gm->capacity;
  local.maxToken = gm->maxToken;
  local.maxRequests = gm->maxRequests;
  local.maxNumBlocks = gm->maxNumBlocks;
  local.hostNumBlocks = gm->hostNumBlocks;
  local.blockSize = gm->blockSize;
  local.localMemoryBytes = gm->localMemoryBytes;
  local.jointHashCapacity = gm->jointHashCapacity;
  local.scratchElements = gm->scratchElements;
  local.maxJointRows = gm->maxJointRows;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  RunAbdPlan(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq,
      blockTable, activeRows, lastReqIds, slotToToken, lruSlots, currentSlots,
      missCount, missTokens, missSlots, compactWorkspace, rowModes,
      residentRows, residentBlockTable, local);
}
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
    uint32_t blocks, uint32_t dynamicUb, void* stream) {
  sparse_kv_plan_peer_qli_kernel<<<blocks, dynamicUb, stream>>>(
      reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, rowModes, residentRows, residentBlockTable, blockTable, tiling);
}
