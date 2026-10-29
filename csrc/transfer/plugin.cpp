// SPDX-License-Identifier: Apache-2.0
// PyTorch接入与Host映射只在Host侧执行；Transfer本体仍是上游AscendC kernel。
#include <acl/acl.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "launch.h"
#include "../transfer_variants/launch.h"

namespace {
void CheckAcl(aclError code, const char* operation) {
  TORCH_CHECK(code == ACL_SUCCESS, operation, " failed, ACL code=", code);
}

// from_blob的deleter持有owner。调用者必须保留返回的CPU tensor直至NPU同步完成。
// 保存原分配地址：映射地址可能为了4KiB对齐而位于分配区内部。
struct HostMapping {
  void* allocation = nullptr;
  void* host = nullptr;
  aclrtContext context = nullptr;
  bool registered = false;
  ~HostMapping() {
    aclrtContext previous = nullptr;
    (void)aclrtGetCurrentContext(&previous);
    if (context != nullptr && context != previous) {
      const auto result = aclrtSetCurrentContext(context);
      if (result != ACL_SUCCESS) {
        std::fprintf(stderr, "Host mapping cleanup: set context failed %d; retain allocation\n", result);
        return;
      }
    }
    bool mayFree = true;
    if (registered) {
      // 清理在计时外；异常路径也不能先释放仍可能被设备读取的Host映射。
      const auto syncResult = aclrtSynchronizeDevice();
      if (syncResult != ACL_SUCCESS) {
        std::fprintf(stderr, "Host mapping cleanup: device sync failed %d; retain mapping/allocation\n", syncResult);
        if (previous != nullptr && previous != context) (void)aclrtSetCurrentContext(previous);
        return;
      }
      const auto result = aclrtHostUnregister(host);
      if (result != ACL_SUCCESS) {
        std::fprintf(stderr, "Host mapping cleanup: unregister failed %d; retain allocation\n", result);
        mayFree = false;
      }
    }
    if (allocation != nullptr && mayFree) {
      const auto result = aclrtFreeHost(allocation);
      if (result != ACL_SUCCESS) std::fprintf(stderr, "Host mapping cleanup: free failed %d\n", result);
    }
    if (previous != nullptr && previous != context) (void)aclrtSetCurrentContext(previous);
  }
};

std::tuple<at::Tensor, int64_t> MappedHostEmpty(int64_t bytes) {
  constexpr uint64_t page = 4096U;
  TORCH_CHECK(bytes > 0 && bytes < std::numeric_limits<int64_t>::max() - 2 * page,
              "Host allocation size is invalid");
  auto owner = std::make_shared<HostMapping>();
  CheckAcl(aclrtGetCurrentContext(&owner->context), "aclrtGetCurrentContext");
  TORCH_CHECK(owner->context != nullptr, "Select and initialize the NPU before allocating mapped Host memory");
  const uint64_t mappedBytes = (static_cast<uint64_t>(bytes) + page - 1U) & ~(page - 1U);
  CheckAcl(aclrtMallocHost(&owner->allocation, mappedBytes + page - 1U), "aclrtMallocHost");
  TORCH_CHECK(owner->allocation != nullptr, "aclrtMallocHost returned a null pointer");
  const uintptr_t aligned = (reinterpret_cast<uintptr_t>(owner->allocation) + page - 1U) & ~(page - 1U);
  owner->host = reinterpret_cast<void*>(aligned);
  // aclrtMallocHost已锁页，不重复使用PINNED注册。禁止普通CPU指针直接作为GM地址。
  CheckAcl(aclrtHostRegisterV2(owner->host, mappedBytes, ACL_HOST_REG_MAPPED), "aclrtHostRegisterV2");
  owner->registered = true;
  void* device = nullptr;
  CheckAcl(aclrtHostGetDevicePointer(owner->host, &device, 0), "aclrtHostGetDevicePointer");
  TORCH_CHECK(device != nullptr, "Host mapping returned a null DVA");
  at::Tensor tensor = at::from_blob(owner->host, {bytes}, [owner](void*) {},
                                   at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  return {tensor, static_cast<int64_t>(reinterpret_cast<uintptr_t>(device))};
}

int64_t CheckedProduct(int64_t left, int64_t right) {
  TORCH_CHECK(left > 0 && right > 0 && left <= std::numeric_limits<int64_t>::max() / right,
              "Transfer buffer size overflow");
  return left * right;
}

int64_t TransferBlocks(int64_t coreLimit) {
  TORCH_CHECK(coreLimit > 0 && coreLimit <= 64, "core_limit must be within [1,64]");
  const auto available = sparse_kv_transfer_block_dim();
  TORCH_CHECK(available > 0, "Cannot query Transfer AIV cores");
  return coreLimit < available ? coreLimit : available;
}

void Transfer(
    const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots,
    const at::Tensor& tokenToReq, const at::Tensor& blockTable, const at::Tensor& hostCacheBases,
    const at::Tensor& activeRows, const at::Tensor& hostK, const at::Tensor& hostV,
    const at::Tensor& residentK, const at::Tensor& residentV,
    int64_t topk, int64_t capacity, int64_t blockSize, int64_t hostNumBlocks,
    int64_t bytesK, int64_t bytesV, int64_t variantMask, int64_t coreLimit) {
  TORCH_CHECK(variantMask >= -1 && variantMask <= 15, "Transfer variant_mask must be -1..15");
  const auto i32 = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(topk > 0 && topk <= i32 && capacity > 0 && capacity <= i32 &&
              blockSize > 0 && blockSize <= i32 && hostNumBlocks > 0 && hostNumBlocks <= i32 &&
              bytesK > 0 && bytesK <= i32 && bytesV > 0 && bytesV <= i32,
              "Transfer attributes must be positive and fit int32");
  TORCH_CHECK(missCount.dim() == 1 && missCount.numel() > 0, "miss_count must have positive rows");
  const int64_t rows = missCount.numel();
  const at::Tensor* npu[] = {&missCount, &missTokens, &missSlots, &tokenToReq, &blockTable,
                             &hostCacheBases, &activeRows, &residentK, &residentV};
  TORCH_CHECK(missCount.device().type() == c10::DeviceType::PrivateUse1, "Transfer expects NPU tensors");
  for (const auto* tensor : npu) {
    TORCH_CHECK(tensor->device() == missCount.device() && tensor->is_contiguous(),
                "Transfer NPU tensors must be contiguous on the same device");
  }
  for (const auto* tensor : {&missCount, &missTokens, &missSlots, &tokenToReq, &blockTable, &activeRows}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "Transfer metadata must be int32");
  }
  TORCH_CHECK(missTokens.dim() == 2 && missTokens.size(0) == rows && missTokens.size(1) == topk &&
              missSlots.sizes() == missTokens.sizes(), "miss_tokens/miss_slots shape mismatch");
  TORCH_CHECK(tokenToReq.dim() == 1 && tokenToReq.numel() == rows && activeRows.numel() == 1,
              "row metadata shape mismatch");
  TORCH_CHECK(blockTable.dim() == 2 && blockTable.size(0) > 0 && blockTable.size(1) > 0,
              "block_table must have requests and blocks");
  TORCH_CHECK(hostCacheBases.scalar_type() == at::kLong && hostCacheBases.numel() == 2,
              "host_cache_bases must contain two mapped DVA int64 values");
  const int64_t sourceTokens = CheckedProduct(hostNumBlocks, blockSize);
  for (const auto& pair : {std::make_pair(&hostK, bytesK), std::make_pair(&hostV, bytesV)}) {
    TORCH_CHECK(pair.first->device().is_cpu() && pair.first->scalar_type() == at::kByte &&
                pair.first->is_contiguous() && pair.first->numel() == CheckedProduct(sourceTokens, pair.second),
                "Host buffers must be mapped contiguous CPU uint8 pools of the declared size");
  }
  for (const auto& pair : {std::make_pair(&residentK, bytesK), std::make_pair(&residentV, bytesV)}) {
    const auto& tensor = *pair.first;
    TORCH_CHECK(tensor.scalar_type() == at::kByte && tensor.dim() == 3 &&
                tensor.size(0) == rows && tensor.size(1) == capacity && tensor.size(2) == pair.second,
                "resident K/V must be uint8 [rows,capacity,token_bytes]");
  }
  TORCH_CHECK(!residentK.is_alias_of(residentV) && !hostK.is_alias_of(hostV),
              "K/V buffers must not alias");

  SparseKvTransferDirectTilingData tiling{};
  tiling.maxRows = rows;
  tiling.maxRequests = blockTable.size(0);
  tiling.maxNumBlocks = blockTable.size(1);
  tiling.topk = topk;
  tiling.capacity = capacity;
  tiling.hostNumBlocks = hostNumBlocks;
  tiling.blockSize = static_cast<int32_t>(blockSize);
  tiling.tokenSizeBytesK = static_cast<int32_t>(bytesK);
  tiling.tokenSizeBytesV = static_cast<int32_t>(bytesV);
  auto cpuTiling = at::empty({static_cast<int64_t>(sizeof(tiling))}, at::TensorOptions().dtype(at::kByte));
  std::memcpy(cpuTiling.data_ptr(), &tiling, sizeof(tiling));
  auto deviceTiling = cpuTiling.to(missCount.device(), false);
  const uint32_t blocks = static_cast<uint32_t>(TransferBlocks(coreLimit));
  TORCH_CHECK(blocks > 0, "Cannot query AIV cores for SparseKvTransfer");
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto launch = [=]() -> int {
    // CPU所有权也被按值捕获；runner另持有映射，直到最终NPU同步结束。
    (void)hostK;
    (void)hostV;
    if (variantMask < 0) {
      launch_sparse_kv_transfer(missCount.data_ptr(), missTokens.data_ptr(), missSlots.data_ptr(),
        tokenToReq.data_ptr(), blockTable.data_ptr(), hostCacheBases.data_ptr(), activeRows.data_ptr(),
        residentK.data_ptr(), residentV.data_ptr(), deviceTiling.data_ptr(), blocks,
        reinterpret_cast<void*>(stream));
    } else {
      launch_sparse_kv_transfer_variant(static_cast<int32_t>(variantMask),
          missCount.data_ptr(), missTokens.data_ptr(), missSlots.data_ptr(), tokenToReq.data_ptr(),
          blockTable.data_ptr(), hostCacheBases.data_ptr(), activeRows.data_ptr(), residentK.data_ptr(),
          residentV.data_ptr(), deviceTiling.data_ptr(), blocks, reinterpret_cast<void*>(stream));
    }
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("sparse_kv_transfer", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("mapped_host_empty(int bytes) -> (Tensor, int)", MappedHostEmpty);
  m.def("transfer_block_dim_info(int core_limit) -> int", TransferBlocks);
  m.def("sparse_kv_transfer(Tensor miss_count, Tensor miss_tokens, Tensor miss_slots, "
        "Tensor token_to_req, Tensor block_table, Tensor host_cache_bases, Tensor active_rows, "
        "Tensor host_k, Tensor host_v, Tensor(a!) resident_k, Tensor(b!) resident_v, "
        "int topk, int capacity, int block_size, int host_num_blocks, "
        "int token_bytes_k, int token_bytes_v, int variant_mask=-1, int core_limit=64) -> ()");
}
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, PrivateUse1, m) {
  m.impl("sparse_kv_transfer", Transfer);
}
