// SPDX-License-Identifier: Apache-2.0
// Prepared direct launch: no allocation/upload or device-value read inside Run.
// Experimental prepared class: identical bindings/tiling/lifetime checks, plus immutable variant mask.
// Baseline Prepared and original ABCDF remain in their original translation units.
#include <acl/acl.h>
#include <algorithm>
#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../joint_plan/launch.h"
#include "variant_dispatch.h"
#include "../joint_plan/sparse_kv_plan_config.h"

namespace {
using Tensors = std::vector<at::Tensor>;
using Attrs = std::vector<int64_t>;
struct Binding {
  at::Tensor tensor;
  void* address;
  c10::TensorImpl* impl;
  std::vector<int64_t> sizes, strides;
  explicit Binding(const at::Tensor& t) : tensor(t), address(t.data_ptr()),
      impl(t.unsafeGetTensorImpl()), sizes(t.sizes().vec()), strides(t.strides().vec()) {}
  void Check(const at::Tensor& t) const {
    TORCH_CHECK(t.defined() && t.unsafeGetTensorImpl() == impl && t.data_ptr() == address &&
        t.device() == tensor.device() && t.scalar_type() == tensor.scalar_type() &&
        t.sizes().vec() == sizes && t.strides().vec() == strides && t.is_contiguous(),
        "joint prepared tensor binding changed");
  }
};
struct Resources {
  Tensors tensors;
  std::vector<Binding> bindings;
  at::Tensor tiling;
  Attrs attrs;
  uint32_t blocks = 0, dynamicUb = 0;
  bool jointEntry = false;
  int64_t variantMask = 0;
  aclrtContext context = nullptr;
};
void Release(Resources* p) {
  aclrtContext previous = nullptr;
  (void)aclrtGetCurrentContext(&previous);
  auto status = p->context == nullptr ? ACL_SUCCESS : aclrtSetCurrentContext(p->context);
  if (status == ACL_SUCCESS) status = aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr, "joint cleanup failed %d; retain resources for device safety\n", status);
    if (previous != nullptr) (void)aclrtSetCurrentContext(previous);
    return;
  }
  delete p;
  if (previous != nullptr) (void)aclrtSetCurrentContext(previous);
}
uint64_t HashSize(uint64_t n) {
  uint64_t size = 32;
  while (size < n * 2U) size <<= 1U;
  return size;
}
void Shape(const at::Tensor& t, std::initializer_list<int64_t> shape) {
  TORCH_CHECK(t.sizes().vec() == std::vector<int64_t>(shape), "joint tensor shape mismatch");
}

class Prepared : public torch::CustomClassHolder {
 public:
  Prepared(Tensors p, Attrs attrs, bool jointEntry, int64_t variantMask) {
    TORCH_CHECK(variantMask > 0 && variantMask <= 16, "joint variant mask must be 1..16");
    // topk,capacity,maxToken,blockSize,hostNumBlocks,effectiveJoint.
    TORCH_CHECK(p.size() == 18 && attrs.size() == 6, "joint expects 18 tensors / 6 attrs");
    for (size_t i = 0; i < 5; ++i)
      TORCH_CHECK(attrs[i] > 0 && attrs[i] <= std::numeric_limits<int32_t>::max(), "joint attrs out of int32 range");
    TORCH_CHECK((attrs[5] == 0 || attrs[5] == 1) && (jointEntry || attrs[5] == 0), "joint selector invalid");
    const int64_t t = attrs[0], c = attrs[1], block = attrs[3];
    TORCH_CHECK(t <= c && c <= std::numeric_limits<int32_t>::max() / 2 && c % block == 0,
                "joint requires topk<=capacity<=INT32_MAX/2 and capacity%block_size==0");
    TORCH_CHECK(p[0].defined() && p[0].dim() == 1 && p[0].numel() > 0 &&
                p[1].defined() && p[1].dim() == 2 && p[1].size(0) > 0,
                "joint request/query shape");
    const auto device = p[0].device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "joint requires NPU");
    const c10::OptionalDeviceGuard guard(device);
    for (size_t i = 0; i < p.size(); ++i)
      TORCH_CHECK(p[i].defined() && p[i].is_contiguous() && p[i].device() == device &&
          p[i].scalar_type() == ((i == 0 || i == 7) ? at::kLong : at::kInt),
          "joint tensor dtype/device/contiguity");
    const int64_t rows = p[1].size(0), requests = p[0].numel();
    TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max() / (c / block), "resident page index exceeds int32");
    Shape(p[1], {rows, t});
    for (int i : {2, 3, 4, 7, 11, 15, 16}) Shape(p[i], {rows});
    TORCH_CHECK(p[5].dim() == 2 && p[5].size(0) == requests && p[5].size(1) > 0,
                "joint block table shape");
    TORCH_CHECK(p[6].numel() == 1, "joint active_rows shape");
    Shape(p[8], {rows, c}); Shape(p[9], {rows, c});
    for (int i : {10, 12, 13}) Shape(p[i], {rows, t});
    Shape(p[17], {rows, c / block});
    // Mutable buffers must not alias inputs or one another.
    for (size_t i = 7; i < p.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        TORCH_CHECK(!p[i].is_alias_of(p[j]), "joint mutable tensor alias");

    JointDirectTilingData d{};
    d.hashCapacity = HashSize(t);
    d.workspaceRowElements = 3U * d.hashCapacity + c + t;
    d.jointHashCapacity = attrs[5] ? HashSize(c) : 0U;
    const uint64_t jointElements = 2U * d.jointHashCapacity + c;
    d.scratchElements = d.workspaceRowElements > jointElements ? d.workspaceRowElements : jointElements;
    if (variantMask == 16 && attrs[5]) {
      const uint64_t bitmapElements = (static_cast<uint64_t>(attrs[2]) + 31U) / 32U + 2U * c;
      d.scratchElements = std::max(d.workspaceRowElements, bitmapElements);
    }
    const uint64_t alignedBytes = (d.scratchElements * 4U + 31U) & ~uint64_t{31U};
    const bool ub = alignedBytes <= sparse_kv_plan_config::PLAN_ROW_UB_LIMIT_BYTES;
    d.localMemoryBytes = sparse_kv_plan_config::PLAN_SCAN_BYTES + (ub ? alignedBytes : 0U);
    TORCH_CHECK(d.scratchElements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / rows),
                "joint workspace size overflow");
    const uint64_t requiredWorkspace = variantMask == 16 ?
        static_cast<uint64_t>(rows) * (d.scratchElements + static_cast<uint64_t>(attrs[2])) :
        (ub ? 1U : static_cast<uint64_t>(rows) * d.scratchElements);
    TORCH_CHECK(requiredWorkspace <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "bitmap workspace overflow");
    TORCH_CHECK(p[14].dim() == 1 && p[14].numel() >= static_cast<int64_t>(requiredWorkspace),
                "joint workspace too small");
    d.maxRows = rows; d.topk = t; d.capacity = c; d.maxToken = attrs[2];
    d.maxRequests = requests; d.maxNumBlocks = p[5].size(1); d.hostNumBlocks = attrs[4];
    d.blockSize = static_cast<int32_t>(block);

    auto data = std::shared_ptr<Resources>(new Resources, Release);
    TORCH_CHECK(aclrtGetCurrentContext(&data->context) == ACL_SUCCESS && data->context != nullptr,
                "joint ACL context unavailable");
    auto cpu = at::empty({static_cast<int64_t>(sizeof(d))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(cpu.data_ptr(), &d, sizeof(d));
    data->tiling = cpu.to(device, false);  // Setup only, outside target capture.
    data->dynamicUb = d.localMemoryBytes;
    data->blocks = joint_plan_block_dim();
    TORCH_CHECK(data->blocks > 0, "joint cannot query AIV count");
    data->tensors = std::move(p); data->attrs = std::move(attrs); data->jointEntry = jointEntry;
    data->variantMask = variantMask;
    for (const auto& tensor : data->tensors) data->bindings.emplace_back(tensor);
    data_ = std::move(data);
  }
  void Run(Tensors p, Attrs attrs) {
    const auto data = Get();
    TORCH_CHECK(data != nullptr, "joint context closed");
    TORCH_CHECK(p.size() == 18 && attrs == data->attrs, "joint context attrs/list changed");
    for (size_t i = 0; i < p.size(); ++i) data->bindings[i].Check(p[i]);
    const c10::OptionalDeviceGuard guard(data->tensors[0].device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [data, stream]() -> int {
      const auto& p = data->tensors;
      LaunchJointVariant(data->variantMask, data->jointEntry,
          p[0].data_ptr(), p[1].data_ptr(), p[2].data_ptr(), p[3].data_ptr(), p[4].data_ptr(),
          p[5].data_ptr(), p[6].data_ptr(), p[7].data_ptr(), p[8].data_ptr(), p[9].data_ptr(),
          p[10].data_ptr(), p[11].data_ptr(), p[12].data_ptr(), p[13].data_ptr(), p[14].data_ptr(),
          p[15].data_ptr(), p[16].data_ptr(), p[17].data_ptr(), data->tiling.data_ptr(),
          data->blocks, data->dynamicUb, reinterpret_cast<void*>(stream));
      return 0;
    };
    at_npu::native::OpCommand::RunOpApi("mtp_joint_plan", launch);
  }
  std::string Info() const {
    const auto data = Get();
    TORCH_CHECK(data != nullptr, "joint context closed");
    return "{\"blocks\":" + std::to_string(data->blocks) + ",\"dynamic_ub_bytes\":" +
        std::to_string(data->dynamicUb) + ",\"joint_enabled\":" + std::to_string(data->attrs[5]) + ",\"variant_mask\":" + std::to_string(data->variantMask) + "}";
  }
  void Close() {
    std::shared_ptr<Resources> previous;
    { std::lock_guard<std::mutex> lock(mutex_); previous.swap(data_); }
  }
 private:
  std::shared_ptr<Resources> Get() const {
    std::lock_guard<std::mutex> lock(mutex_); return data_;
  }
  mutable std::mutex mutex_;
  std::shared_ptr<Resources> data_;
};
static auto registration = torch::class_<Prepared>("mtp_joint_plan", "PreparedVariant")
    .def(torch::init<Tensors, Attrs, bool, int64_t>()).def("run", &Prepared::Run)
    .def("info", &Prepared::Info).def("close", &Prepared::Close);
}  // namespace
