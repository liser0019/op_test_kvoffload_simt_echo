#include <algorithm>
#include <cstring>
#include <limits>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "echo_lru_launch.h"

namespace {
int64_t EchoCoreCount() {
  int32_t device = 0;
  TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "echo_lru: cannot query current device");
  int64_t cores = 0;
  auto ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
  TORCH_CHECK(ret == ACL_SUCCESS && cores > 0, "echo_lru: cannot query vector cores, ret=", ret);
  return cores;
}

ascend_kernel::EchoLruTiling Check(const at::Tensor& pos, const at::Tensor& spl, const at::Tensor& resetMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& pri, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& curSlots, const at::Tensor& missHostPos, const at::Tensor& missAllocFlat, const at::Tensor& missMask, const at::Tensor& workspace, int64_t blocks, bool specEnabled, bool npu) {
  TORCH_CHECK(pos.dim() == 2 && pos.size(0) > 0 && pos.size(1) > 0,
              "echo_lru: pos must have shape [rows, topk]");
  const int64_t rows = pos.size(0), topk = pos.size(1);
  TORCH_CHECK(dth.dim() == 2 && dth.size(0) == rows, "echo_lru: invalid dth shape");
  const int64_t capacity = dth.size(1) - 1;
  TORCH_CHECK(capacity > 0 && capacity <= ascend_kernel::ARGTOPK_B_MAX,
              "echo_lru: capacity must be in [1, 4096]");
  TORCH_CHECK(htd.dim() == 2 && htd.size(0) == rows && htd.size(1) > 1,
              "echo_lru: invalid htd shape");
  const int64_t maxToken = htd.size(1);
  const int64_t limit = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(rows <= limit && topk <= limit && maxToken <= limit &&
              rows <= limit / topk && rows <= limit / maxToken &&
              rows <= limit / (capacity + 1) &&
              rows <= limit / (capacity + std::min(topk, capacity)),
              "echo_lru: shape exceeds 32-bit kernel indexing");
  TORCH_CHECK(blocks > 0 && blocks <= rows, "echo_lru: invalid block count");
  const int64_t chunk = (rows + blocks - 1) / blocks;
  TORCH_CHECK((blocks - 1) * chunk < rows, "echo_lru: last block would be empty; use echo_partition");
  auto tensor = [&](const at::Tensor& value, at::ScalarType type,
                    at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(value.scalar_type() == type && value.sizes() == shape && value.is_contiguous(),
                "echo_lru: shape/dtype/contiguity mismatch for ", name);
    TORCH_CHECK(value.device() == pos.device(), "echo_lru: tensors must share a device");
  };
  tensor(pos, at::kInt, {rows, topk}, "pos");
  tensor(spl, at::kInt, {rows}, "spl");
  tensor(resetMask, at::kByte, {rows}, "reset_mask");
  tensor(htd, at::kInt, {rows, maxToken}, "htd");
  tensor(dth, at::kInt, {rows, capacity + 1}, "dth");
  tensor(pri, at::kInt, {rows, capacity + 1}, "pri");
  tensor(freeSlots, at::kInt, {rows, capacity + std::min(topk, capacity)}, "free_slots");
  tensor(avail, at::kInt, {rows}, "avail");
  tensor(fifo, at::kInt, {rows}, "fifo");
  tensor(curSlots, at::kInt, {rows, topk}, "current_slots");
  tensor(missHostPos, at::kInt, {rows, topk}, "miss_host_pos");
  tensor(missAllocFlat, at::kInt, {rows, topk}, "miss_alloc_flat");
  tensor(missMask, at::kByte, {rows, topk}, "miss_mask");
  const uint64_t required = EchoWorkspaceElements(rows, topk, capacity, blocks);
  TORCH_CHECK(required <= uint64_t(limit) && workspace.dim() == 1 &&
              workspace.numel() >= static_cast<int64_t>(required) &&
              workspace.scalar_type() == at::kInt && workspace.is_contiguous() &&
              workspace.device() == pos.device(), "echo_lru: workspace is invalid or too small");
  if (npu) {
    TORCH_CHECK(pos.device().type() == c10::DeviceType::PrivateUse1, "echo_lru: expected NPU");
  }
  ascend_kernel::EchoLruTiling tiling{};
  tiling.blockNum = blocks;
  tiling.rowsPerCore = chunk;
  tiling.tailRows = rows - chunk * (blocks - 1);
  tiling.numTokens = rows;
  tiling.topkBufferB = capacity;
  tiling.freeStackSize = capacity + std::min(topk, capacity);
  tiling.maxModelLenP1 = maxToken;
  tiling.topkDim = topk;
  tiling.effTopk = std::min(topk, capacity);
  tiling.specEnabled = specEnabled;
  tiling.argtopkDebugRow = 0xFFFFFFFFu;
  return tiling;
}

void EchoMeta(const at::Tensor& pos, const at::Tensor& spl, const at::Tensor& resetMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& pri, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& curSlots, const at::Tensor& missHostPos, const at::Tensor& missAllocFlat, const at::Tensor& missMask, const at::Tensor& workspace, int64_t blocks, bool specEnabled) {
  Check(pos, spl, resetMask, htd, dth, pri, freeSlots, avail, fifo, curSlots, missHostPos, missAllocFlat, missMask, workspace, blocks, specEnabled, false);
}

void EchoNpu(const at::Tensor& pos, const at::Tensor& spl, const at::Tensor& resetMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& pri, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& curSlots, const at::Tensor& missHostPos, const at::Tensor& missAllocFlat, const at::Tensor& missMask, const at::Tensor& workspace, int64_t blocks, bool specEnabled) {
  const c10::OptionalDeviceGuard guard(pos.device());
  const auto tiling = Check(pos, spl, resetMask, htd, dth, pri, freeSlots, avail, fifo, curSlots, missHostPos, missAllocFlat, missMask, workspace, blocks, specEnabled, true);
  TORCH_CHECK(blocks <= EchoCoreCount(), "echo_lru: block count exceeds available vector cores");
  at::Tensor tilingCpu = at::empty({static_cast<int64_t>(sizeof(tiling))},
                                  at::TensorOptions().dtype(at::kByte));
  std::memcpy(tilingCpu.data_ptr(), &tiling, sizeof(tiling));
  at::Tensor tilingDev = tilingCpu.to(pos.device(), false);
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  // Own tensors and tiling until an asynchronously scheduled callback completes.
  auto launch = [=]() -> int {
    launch_echo_lru(htd.data_ptr<int32_t>(), dth.data_ptr<int32_t>(), pri.data_ptr<int32_t>(),
                    freeSlots.data_ptr<int32_t>(), avail.data_ptr<int32_t>(), fifo.data_ptr<int32_t>(),
                    pos.data_ptr<int32_t>(), spl.data_ptr<int32_t>(), resetMask.data_ptr<uint8_t>(),
                    curSlots.data_ptr<int32_t>(), missHostPos.data_ptr<int32_t>(),
                    missAllocFlat.data_ptr<int32_t>(), missMask.data_ptr<uint8_t>(),
                    workspace.data_ptr(), tilingDev.data_ptr(), blocks, reinterpret_cast<void*>(stream));
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("EchoLru", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("echo_lru_core_count() -> int", EchoCoreCount);
  m.def("echo_lru(Tensor pos, Tensor spl, Tensor reset_mask, Tensor(a!) htd, "
        "Tensor(b!) dth, Tensor(c!) pri, Tensor(d!) free_slots, Tensor(e!) avail, "
        "Tensor(f!) fifo, Tensor(g!) current_slots, Tensor(h!) miss_host_pos, "
        "Tensor(i!) miss_alloc_flat, Tensor(j!) miss_mask, Tensor(k!) workspace, "
        "int blocks, bool spec_enabled) -> ()");
}
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, Meta, m) { m.impl("echo_lru", EchoMeta); }
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, PrivateUse1, m) { m.impl("echo_lru", EchoNpu); }
