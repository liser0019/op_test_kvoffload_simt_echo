/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * (kernel 计算逻辑源自 MemFabric_Hybrid,Mulan PSL v2)
 *
 * SparseKvPlan 直调算子 torch 注册层(g++ 编译)。
 *
 * torch.ops.sparse_kv_plan_op.sparse_kv_plan 的 schema 与 vllm-ascend 的
 * aclnnSparseKvPlan 原型逐参数对应(15 个张量 + 5 个整型属性),差异:
 * - 本算子为带持久状态的 in-place 算子,8 个被 kernel 写写的张量带 (a!)..(h!)
 *   变更标注;
 * - 以 direct launch 方式启动 kernel,不经过 GE/aclnn。
 *
 * 属性含义(与 vllm-ascend 原工程一致):
 *   topk          每行 TopK 宽度 T(topk_indices.size(1))
 *   capacity      每行驻留 slot 数 C
 *   max_token     token 值域上界
 *   block_size    KV block 大小(token/block)
 *   host_num_blocks host 侧 KV 缓存物理 block 数(physical block 合法上界)
 */
#include <cstddef>
#include <cstring>
#include <limits>

#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "sparse_kv_plan_launch.h"

namespace sparse_kv_plan_op {
namespace {

// 输入校验:取自 vllm-ascend sparse_kv_lru_torch_adpt.h 的
// CheckSparseKvRuntimeInputs 并补齐状态张量形状检查。
// 单算子测试场景要求失败足够响亮,故不做 NDEBUG 裁剪。
void CheckSparseKvPlanInputs(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                             const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                             const at::Tensor& tokenToReq, const at::Tensor& blockTable,
                             const at::Tensor& activeRows, const at::Tensor& lastReqIds,
                             const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                             const at::Tensor& currentSlots, const at::Tensor& missCount,
                             const at::Tensor& missTokens, const at::Tensor& missSlots,
                             const at::Tensor& compactWorkspace, int64_t topk, int64_t capacity,
                             int64_t maxToken, int64_t blockSize, int64_t hostNumBlocks, bool requireNpu) {
  TORCH_CHECK(topk > 0 && capacity > 0 && maxToken > 0 && blockSize > 0 && hostNumBlocks > 0,
              "topk/capacity/max_token/block_size/host_num_blocks must be positive");
  TORCH_CHECK(topk <= std::numeric_limits<int32_t>::max() / 2 &&
                  capacity <= std::numeric_limits<int32_t>::max() &&
                  maxToken <= std::numeric_limits<int32_t>::max() &&
                  blockSize <= std::numeric_limits<int32_t>::max(),
              "SparseKvPlan shape or attribute exceeds int32 kernel limits");
  TORCH_CHECK(reqIds.dim() == 1 && reqIds.scalar_type() == at::kLong, "req_ids must be 1-D int64");
  TORCH_CHECK(topkIndices.dim() == 2 && topkIndices.scalar_type() == at::kInt, "topk_indices must be 2-D int32");
  const int64_t maxRows = topkIndices.size(0);
  TORCH_CHECK(maxRows > 0 && reqIds.numel() > 0, "max_rows and max_requests must be positive");
  TORCH_CHECK(topkIndices.size(1) == topk, "topk_indices.size(1) must equal topk attr, got ",
              topkIndices.size(1), " vs ", topk);
  TORCH_CHECK(stablePrefixLens.numel() == maxRows && visibleSeqLens.numel() == maxRows &&
                  tokenToReq.numel() == maxRows,
              "row metadata (stable_prefix_lens/visible_seq_lens/token_to_req) length must be max_rows");
  TORCH_CHECK(stablePrefixLens.scalar_type() == at::kInt && visibleSeqLens.scalar_type() == at::kInt &&
                  tokenToReq.scalar_type() == at::kInt,
              "row metadata must be int32");
  TORCH_CHECK(blockTable.dim() == 2 && blockTable.scalar_type() == at::kInt, "block_table must be 2-D int32");
  TORCH_CHECK(blockTable.size(0) == reqIds.size(0), "req_ids and block_table row counts must match");
  TORCH_CHECK(blockTable.size(1) > 0, "max_num_blocks must be positive");
  TORCH_CHECK(activeRows.numel() == 1 && activeRows.scalar_type() == at::kInt,
              "active_rows must be a scalar int32 tensor");
  TORCH_CHECK(lastReqIds.dim() == 1 && lastReqIds.scalar_type() == at::kLong && lastReqIds.numel() == maxRows,
              "last_req_ids must be 1-D int64 [max_rows]");
  TORCH_CHECK(slotToToken.dim() == 2 && slotToToken.scalar_type() == at::kInt && slotToToken.size(0) == maxRows &&
                  slotToToken.size(1) == capacity,
              "slot_to_token must be int32 [max_rows, capacity]");
  TORCH_CHECK(lruSlots.dim() == 2 && lruSlots.scalar_type() == at::kInt && lruSlots.size(0) == maxRows &&
                  lruSlots.size(1) == capacity,
              "lru_slots must be int32 [max_rows, capacity]");
  TORCH_CHECK(currentSlots.dim() == 2 && currentSlots.scalar_type() == at::kInt && currentSlots.size(0) == maxRows &&
                  currentSlots.size(1) == topk,
              "current_slots must be int32 [max_rows, topk]");
  TORCH_CHECK(missCount.dim() == 1 && missCount.scalar_type() == at::kInt && missCount.numel() == maxRows,
              "miss_count must be 1-D int32 [max_rows]");
  TORCH_CHECK(missTokens.sizes() == currentSlots.sizes() && missTokens.scalar_type() == at::kInt,
              "miss_tokens must be int32 [max_rows, topk]");
  TORCH_CHECK(missSlots.sizes() == currentSlots.sizes() && missSlots.scalar_type() == at::kInt,
              "miss_slots must be int32 [max_rows, topk]");
  TORCH_CHECK(compactWorkspace.scalar_type() == at::kInt && compactWorkspace.dim() == 1,
              "compact_workspace must be 1-D int32");
  const at::Tensor* tensors[] = {&reqIds,          &topkIndices,  &stablePrefixLens, &visibleSeqLens,
                                 &tokenToReq,      &blockTable,   &activeRows,       &lastReqIds,
                                 &slotToToken,     &lruSlots,     &currentSlots,     &missCount,
                                 &missTokens,      &missSlots,    &compactWorkspace};
  const char* names[] = {"req_ids",          "topk_indices", "stable_prefix_lens", "visible_seq_lens",
                         "token_to_req",     "block_table",  "active_rows",       "last_req_ids",
                         "slot_to_token",    "lru_slots",    "current_slots",     "miss_count",
                         "miss_tokens",      "miss_slots",   "compact_workspace"};
  if (requireNpu) {
    TORCH_CHECK(reqIds.device().type() == c10::DeviceType::PrivateUse1, "req_ids must be on NPU");
  }
  for (std::size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
    TORCH_CHECK(tensors[i]->is_contiguous(), names[i], " must be contiguous");
    if (requireNpu) {
      TORCH_CHECK(tensors[i]->device() == reqIds.device(), names[i], " must be on the same NPU as req_ids");
    }
  }
  // workspace 容量要求:UB 快路径只需占位 1 个元素;GM 路径需 max_rows * rowElements。
  const uint64_t rowElements = SparseKvPlanRowElements(topk, capacity);
  TORCH_CHECK(rowElements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / maxRows),
              "compact_workspace element count overflows int64");
  const int64_t required =
      rowElements <= sparse_kv_plan::PLAN_ROW_UB_LIMIT_ELEMENTS ? 1 : static_cast<int64_t>(maxRows * rowElements);
  TORCH_CHECK(compactWorkspace.numel() >= required, "compact_workspace requires >= ", required,
              " elements for this topk/capacity, got ", compactWorkspace.numel());
}

void sparse_kv_plan_meta(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                         const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                         const at::Tensor& tokenToReq, const at::Tensor& blockTable, const at::Tensor& activeRows,
                         const at::Tensor& lastReqIds, const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                         const at::Tensor& currentSlots, const at::Tensor& missCount, const at::Tensor& missTokens,
                         const at::Tensor& missSlots, const at::Tensor& compactWorkspace, int64_t topk,
                         int64_t capacity, int64_t maxToken, int64_t blockSize, int64_t hostNumBlocks) {
  CheckSparseKvPlanInputs(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable, activeRows,
                          lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                          compactWorkspace, topk, capacity, maxToken, blockSize, hostNumBlocks, false);
}

void sparse_kv_plan_npu(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                        const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                        const at::Tensor& tokenToReq, const at::Tensor& blockTable, const at::Tensor& activeRows,
                        const at::Tensor& lastReqIds, const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                        const at::Tensor& currentSlots, const at::Tensor& missCount, const at::Tensor& missTokens,
                        const at::Tensor& missSlots, const at::Tensor& compactWorkspace, int64_t topk,
                        int64_t capacity, int64_t maxToken, int64_t blockSize, int64_t hostNumBlocks) {
  CheckSparseKvPlanInputs(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable, activeRows,
                         lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                         compactWorkspace, topk, capacity, maxToken, blockSize, hostNumBlocks, true);
  const c10::OptionalDeviceGuard guard(reqIds.device());

  const int64_t maxRows = topkIndices.size(0);
  const int64_t maxRequests = reqIds.size(0);
  const int64_t maxNumBlocks = blockTable.size(1);

  // host 侧组装 tiling(与原 op_host/sparse_kv_plan_tiling.cpp 同源的纯函数实现)。
  const SparseKvPlanTilingData tiling =
      MakeSparseKvPlanTiling(maxRows, topk, capacity, maxToken, maxRequests, maxNumBlocks, hostNumBlocks, blockSize);

  // tiling 结构体经小缓冲 H2D:pageable 拷贝阻塞完成后再入队 kernel,同流保序。
  at::Tensor tilingCpu = at::empty({static_cast<int64_t>(sizeof(SparseKvPlanTilingData))},
                                   at::TensorOptions().dtype(at::kByte));
  std::memcpy(tilingCpu.data_ptr(), &tiling, sizeof(tiling));
  at::Tensor tilingDev = tilingCpu.to(reqIds.device(), /*non_blocking=*/false);

  const uint32_t blockDim = calc_sparse_kv_plan_block_dim();
  auto stream = c10_npu::getCurrentNPUStream().stream(false);

  auto launch = [=]() -> int {
    launch_sparse_kv_plan(
        const_cast<GM_ADDR>(reqIds.data_ptr()), const_cast<GM_ADDR>(topkIndices.data_ptr()),
        const_cast<GM_ADDR>(stablePrefixLens.data_ptr()), const_cast<GM_ADDR>(visibleSeqLens.data_ptr()),
        const_cast<GM_ADDR>(tokenToReq.data_ptr()), const_cast<GM_ADDR>(blockTable.data_ptr()),
        const_cast<GM_ADDR>(activeRows.data_ptr()), const_cast<GM_ADDR>(lastReqIds.data_ptr()),
        const_cast<GM_ADDR>(slotToToken.data_ptr()), const_cast<GM_ADDR>(lruSlots.data_ptr()),
        const_cast<GM_ADDR>(currentSlots.data_ptr()), const_cast<GM_ADDR>(missCount.data_ptr()),
        const_cast<GM_ADDR>(missTokens.data_ptr()), const_cast<GM_ADDR>(missSlots.data_ptr()),
        const_cast<GM_ADDR>(compactWorkspace.data_ptr()), const_cast<GM_ADDR>(tilingDev.data_ptr()), blockDim,
        tiling.localMemoryBytes, reinterpret_cast<void*>(stream));
    return 0;
  };
  // RunOpApi 仅提供算子名标签(直调不真正走 aclnn)。
  at_npu::native::OpCommand::RunOpApi("SparseKvPlan", launch);
}

}  // namespace

// schema 与 vllm-ascend aclnnSparseKvPlan 原型逐参数对应;
// (a!)..(h!) 标注 kernel 会原地写入的状态/临时张量。
TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("sparse_kv_plan(Tensor req_ids, Tensor topk_indices, Tensor stable_prefix_lens, "
        "Tensor visible_seq_lens, Tensor token_to_req, Tensor block_table, Tensor active_rows, "
        "Tensor(a!) last_req_ids, Tensor(b!) slot_to_token, Tensor(c!) lru_slots, Tensor(d!) current_slots, "
        "Tensor(e!) miss_count, Tensor(f!) miss_tokens, Tensor(g!) miss_slots, Tensor(h!) compact_workspace, "
        "int topk, int capacity, int max_token, int block_size, int host_num_blocks) -> ()");
}

TORCH_LIBRARY_IMPL(sparse_kv_plan_op, Meta, m) {
  m.impl("sparse_kv_plan", sparse_kv_plan_meta);
}

TORCH_LIBRARY_IMPL(sparse_kv_plan_op, PrivateUse1, m) {
  m.impl("sparse_kv_plan", sparse_kv_plan_npu);
}

}  // namespace sparse_kv_plan_op
