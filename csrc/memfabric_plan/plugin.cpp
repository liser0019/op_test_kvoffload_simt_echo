// SPDX-License-Identifier: Apache-2.0
// PyTorch glue only. The archive's Plan kernel/launch remain unmodified.
#include <cstdint>
#include <limits>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void OffloadOpsLruCompact(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    int64_t, int64_t, int64_t, int64_t, void*);
namespace {
void Check(const at::Tensor& req, const at::Tensor& last, const at::Tensor& query,
           const at::Tensor& prefix, const at::Tensor& resident, const at::Tensor& lru,
           const at::Tensor& current, const at::Tensor& count, const at::Tensor& tokens,
           const at::Tensor& slots, const at::Tensor& mark, const at::Tensor& pos,
           const at::Tensor& epoch, int64_t rows, int64_t topk, int64_t capacity,
           int64_t maxToken, bool npu) {
  const int64_t limit = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(rows > 0 && topk > 0 && topk <= capacity && maxToken > 0 &&
              rows <= limit && topk <= limit && capacity <= limit && maxToken <= limit / 8,
              "MemFabric Plan: invalid/unsupported shape");
  // Same allocations as upstream Init: 6 TopK + 4 capacity buffers + clear + 2 scalars.
  const uint64_t tBytes = (static_cast<uint64_t>(topk) * 4 + 31) & ~uint64_t(31);
  const uint64_t cBytes = (static_cast<uint64_t>(capacity) * 4 + 31) & ~uint64_t(31);
  TORCH_CHECK(6 * tBytes + 4 * cBytes + 4096 + 64 <= 120 * 1024,
              "MemFabric Plan: benchmark UB budget exceeded (120 KiB)");
  TORCH_CHECK(req.dim() == 1 && last.dim() == 1 && req.numel() == rows && last.numel() == rows &&
              req.scalar_type() == at::kLong && last.scalar_type() == at::kLong, "request IDs: int64 [rows]");
  const at::Tensor* ints[] = {&query, &prefix, &resident, &lru, &current, &count, &tokens, &slots, &mark, &pos, &epoch};
  for (auto* tensor : ints) TORCH_CHECK(tensor->scalar_type() == at::kInt, "Plan arrays must be int32");
  TORCH_CHECK(query.dim() == 2 && query.size(0) == rows && query.size(1) == topk, "query shape");
  TORCH_CHECK(prefix.dim() == 1 && prefix.numel() == rows && count.dim() == 1 && count.numel() == rows, "row scalar shape");
  TORCH_CHECK(resident.dim() == 2 && resident.size(0) == rows && resident.size(1) == capacity &&
              lru.sizes() == resident.sizes(), "resident/LRU shape");
  TORCH_CHECK(current.sizes() == query.sizes() && tokens.sizes() == query.sizes() && slots.sizes() == query.sizes(), "output shape");
  TORCH_CHECK(mark.dim() == 2 && mark.size(0) == 8 && mark.size(1) == maxToken && pos.sizes() == mark.sizes() &&
              epoch.dim() == 1 && epoch.numel() == 8, "upstream requires mark/pos [8,max_token], epochs [8]");
  const at::Tensor* all[] = {&req,&last,&query,&prefix,&resident,&lru,&current,&count,&tokens,&slots,&mark,&pos,&epoch};
  if (npu) TORCH_CHECK(req.device().type() == c10::DeviceType::PrivateUse1, "requires NPU tensors");
  for (auto* tensor : all) TORCH_CHECK(tensor->is_contiguous() && tensor->device() == req.device(), "tensors must be contiguous on one device");
}
#define PARAMS const at::Tensor& req, const at::Tensor& last, const at::Tensor& query, const at::Tensor& prefix, \
 const at::Tensor& resident, const at::Tensor& lru, const at::Tensor& current, const at::Tensor& count, \
 const at::Tensor& tokens, const at::Tensor& slots, const at::Tensor& mark, const at::Tensor& pos, \
 const at::Tensor& epoch, int64_t rows, int64_t topk, int64_t capacity, int64_t maxToken
#define ARGS req,last,query,prefix,resident,lru,current,count,tokens,slots,mark,pos,epoch,rows,topk,capacity,maxToken
void Meta(PARAMS) { Check(ARGS, false); }
void Npu(PARAMS) {
  Check(ARGS, true);
  const c10::OptionalDeviceGuard guard(req.device());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  // Own all tensors until the queued launch completes, as with the SIMT plugin.
  auto launch = [=]() -> int {
    OffloadOpsLruCompact(
      reinterpret_cast<uint64_t>(req.data_ptr()), reinterpret_cast<uint64_t>(last.data_ptr()),
      reinterpret_cast<uint64_t>(query.data_ptr()), reinterpret_cast<uint64_t>(prefix.data_ptr()),
      reinterpret_cast<uint64_t>(resident.data_ptr()), reinterpret_cast<uint64_t>(lru.data_ptr()),
      reinterpret_cast<uint64_t>(current.data_ptr()), reinterpret_cast<uint64_t>(count.data_ptr()),
      reinterpret_cast<uint64_t>(tokens.data_ptr()), reinterpret_cast<uint64_t>(slots.data_ptr()),
      reinterpret_cast<uint64_t>(mark.data_ptr()), reinterpret_cast<uint64_t>(pos.data_ptr()),
      reinterpret_cast<uint64_t>(epoch.data_ptr()), rows,topk,capacity,maxToken,reinterpret_cast<void*>(stream));
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("MemFabricLruPlan", launch);
}
#undef PARAMS
#undef ARGS
}
TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("memfabric_lru_plan(Tensor req_ids, Tensor(a!) last_req_ids, Tensor topk_indices, Tensor stable_prefix_lens, "
        "Tensor(b!) slot_to_token, Tensor(c!) lru_slots, Tensor(d!) current_slots, Tensor(e!) miss_count, "
        "Tensor(f!) miss_tokens, Tensor(g!) miss_slots, Tensor(h!) token_mark, Tensor(i!) token_pos, Tensor(j!) epochs, "
        "int rows, int topk, int capacity, int max_token) -> ()");
}
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, Meta, m) { m.impl("memfabric_lru_plan", Meta); }
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, PrivateUse1, m) { m.impl("memfabric_lru_plan", Npu); }
