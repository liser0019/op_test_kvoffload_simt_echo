// SPDX-License-Identifier: Apache-2.0
// 仅做直调适配：算法头逐字提取；独立 VF/kernel/launch 符号避免多版本冲突。
#define SparseKvPlanRuntimeVf SparseKvPlanRuntimeVf_v05
#include "sparse_kv_plan_compute.h"
#include "../variant_tiling_data.h"
extern "C" __global__ __aicore__ void sparse_kv_plan_v05_kernel(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                                     GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR blockTable,
                                                     GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken,
                                                     GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount,
                                                     GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace,
                                                     GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  const __gm__ SparseKvPlanTilingData* tilingGm =
      reinterpret_cast<const __gm__ SparseKvPlanTilingData*>(tiling);
  SparseKvPlanTilingData tilingData;
  tilingData.hashCapacity = tilingGm->hashCapacity;
  tilingData.workspaceRowElements = tilingGm->workspaceRowElements;
  tilingData.maxRows = tilingGm->maxRows;
  tilingData.topk = tilingGm->topk;
  tilingData.capacity = tilingGm->capacity;
  tilingData.maxToken = tilingGm->maxToken;
  tilingData.maxRequests = tilingGm->maxRequests;
  tilingData.maxNumBlocks = tilingGm->maxNumBlocks;
  tilingData.hostNumBlocks = tilingGm->hostNumBlocks;
  tilingData.blockSize = tilingGm->blockSize;
  tilingData.localMemoryBytes = tilingGm->localMemoryBytes;
  const int32_t requestedRows = *reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t numRows = requestedRows > 0 && requestedRows < tilingData.maxRows
                              ? requestedRows
                              : (requestedRows > 0 ? tilingData.maxRows : 0);
  const int64_t rowStart = static_cast<int64_t>(AscendC::GetBlockIdx());
  const int64_t rowStride = static_cast<int64_t>(AscendC::GetBlockNum());
  if (rowStart >= numRows) {
    return;  // 外层 AIV 标量分支，尚未进入 SIMT，不会破坏 block barrier。
  }
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> localWorkspace;
  pipe.InitBuffer(localWorkspace, tilingData.localMemoryBytes);
  __ubuf__ int32_t* scanWorkspace = reinterpret_cast<__ubuf__ int32_t*>(localWorkspace.Get<int32_t>().GetPhyAddr());
  asc_vf_call<SparseKvPlanRuntimeVf>(
      dim3(PLAN_THREADS), scanWorkspace, reinterpret_cast<__gm__ int64_t*>(reqIds),
      reinterpret_cast<__gm__ int64_t*>(lastReqIds), reinterpret_cast<__gm__ int32_t*>(topkIndices),
      reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), reinterpret_cast<__gm__ int32_t*>(visibleSeqLens),
      reinterpret_cast<__gm__ int32_t*>(tokenToReq), reinterpret_cast<__gm__ int32_t*>(blockTable),
      reinterpret_cast<__gm__ int32_t*>(slotToToken), reinterpret_cast<__gm__ int32_t*>(lruSlots),
      reinterpret_cast<__gm__ int32_t*>(currentSlots), reinterpret_cast<__gm__ int32_t*>(missCount),
      reinterpret_cast<__gm__ int32_t*>(missTokens), reinterpret_cast<__gm__ int32_t*>(missSlots),
      reinterpret_cast<__gm__ int32_t*>(compactWorkspace), tilingData.hashCapacity, tilingData.workspaceRowElements,
      rowStart, rowStride, numRows, tilingData.topk, tilingData.capacity, tilingData.maxToken, tilingData.maxRequests,
      tilingData.maxNumBlocks, tilingData.hostNumBlocks, tilingData.blockSize);
}

#undef SparseKvPlanRuntimeVf

extern "C" void launch_sparse_kv_plan_v05(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR blockTable, GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR tiling, uint32_t blockDim, uint32_t dynUBufSize, void* stream) {
  sparse_kv_plan_v05_kernel<<<blockDim, dynUBufSize, stream>>>(
      reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable, activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, tiling);
}
