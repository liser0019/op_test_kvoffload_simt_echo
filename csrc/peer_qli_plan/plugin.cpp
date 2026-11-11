// SPDX-License-Identifier: Apache-2.0
// Plan-only adapter, owns tiling and bound tensors through async completion.
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <acl/acl.h>
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
#include "launch.h"


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
        "peer Plan prepared tensor binding changed");
  }
};
struct Resources {
  Tensors tensors;
  std::vector<Binding> bindings;
  at::Tensor tiling;
  Attrs attrs;
  uint32_t blocks = 0, dynamicUb = 0;
  bool jointEntry = false;
  aclrtContext context = nullptr;
};
void Release(Resources* p) {
  aclrtContext previous = nullptr;
  (void)aclrtGetCurrentContext(&previous);
  auto status = p->context == nullptr ? ACL_SUCCESS : aclrtSetCurrentContext(p->context);
  if (status == ACL_SUCCESS) status = aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr, "peer Plan cleanup failed %d; retain resources for device safety\n", status);
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
  TORCH_CHECK(t.sizes().vec() == std::vector<int64_t>(shape), "peer Plan tensor shape mismatch");
}

class Prepared : public torch::CustomClassHolder {
 public:
  Prepared(Tensors p, Attrs attrs, bool jointEntry) {
    // topk,capacity,maxToken,blockSize,effectiveJoint,maxJointRows.
    TORCH_CHECK(p.size() == 18 && attrs.size() == 6, "peer Plan expects 18 tensors / 6 attrs");
    for (size_t i = 0; i < 4; ++i)
      TORCH_CHECK(attrs[i] > 0 && attrs[i] <= std::numeric_limits<int32_t>::max(), "peer Plan attrs out of int32 range");
    TORCH_CHECK((attrs[4] == 0 || attrs[4] == 1) && (jointEntry || attrs[4] == 0), "peer Plan selector invalid");
    const int64_t t = attrs[0], c = attrs[1], block = attrs[3];
    TORCH_CHECK(t <= c && c <= std::numeric_limits<int32_t>::max() / 2 && c % block == 0,
                "peer Plan requires topk<=capacity<=INT32_MAX/2 and capacity%block_size==0");
    TORCH_CHECK(p[0].defined() && p[0].dim() == 1 && p[0].numel() > 0 &&
                p[1].defined() && p[1].dim() == 2 && p[1].size(0) > 0,
                "peer Plan request/query shape");
    const auto device = p[0].device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "peer Plan requires NPU");
    const c10::OptionalDeviceGuard guard(device);
    for (size_t i = 0; i < p.size(); ++i)
      TORCH_CHECK(p[i].defined() && p[i].is_contiguous() && p[i].device() == device &&
          p[i].scalar_type() == ((i == 0 || i == 6) ? at::kLong : at::kInt),
          "peer Plan tensor dtype/device/contiguity");
    const int64_t rows = p[1].size(0), requests = p[0].numel();
    TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max() / (c / block), "resident page index exceeds int32");
    Shape(p[1], {rows, t});
    for (int i : {2, 3, 4, 6, 10, 14, 15}) Shape(p[i], {rows});
    TORCH_CHECK(p[5].numel() == 1, "peer Plan active_rows shape");
    Shape(p[7], {rows, c}); Shape(p[8], {rows, c});
    for (int i : {9, 11, 12}) Shape(p[i], {rows, t});
    Shape(p[16], {rows, c / block});
    // Mutable buffers must not alias inputs or one another.
    for (size_t i = 6; i < p.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        TORCH_CHECK(!p[i].is_alias_of(p[j]), "peer Plan mutable tensor alias");

    TORCH_CHECK(attrs[5] == 1 || attrs[5] == 2 || attrs[5] == 4 || attrs[5] == 8, "maxJointRows must be1/2/4/8");
    const int64_t logicalBlocks = (attrs[2] + block - 1) / block;
    Shape(p[17], {requests, logicalBlocks});
    TORCH_CHECK(requests <= std::numeric_limits<int32_t>::max() / logicalBlocks, "source page id overflows int32");
    PeerPlanTiling d{};
    d.hashCapacity = HashSize(t);
    d.workspaceRowElements = 3U * d.hashCapacity + c + t;
    d.jointHashCapacity = attrs[4] ? HashSize(c) : 0U;
    d.maxJointRows = static_cast<uint32_t>(attrs[5]);
    const uint64_t sorted = 64U + ((d.maxJointRows * t + c + 7U) & ~uint64_t{7U}) +
        std::max<uint64_t>({3U * static_cast<uint64_t>(c), 64U, 16U * d.maxJointRows});
    d.scratchElements = attrs[4] ? std::max(d.workspaceRowElements, sorted) : d.workspaceRowElements;
    const uint64_t alignedBytes = (d.scratchElements * 4U + 31U) & ~uint64_t{31U};
    const bool ub = alignedBytes <= 112U * 1024U;
    const uint64_t chunk = (std::min<uint64_t>(768U, std::max<uint64_t>(64U, 3U * c) / 16U) / 4U) * 4U;
    const uint64_t stageWords = std::max<uint64_t>(16U * chunk, 16U * d.maxJointRows);
    const uint64_t stageBytes = attrs[4] ? (64U + stageWords) * 4U : 0U;
    // Upstream CANN tiler:112KiB UB scratch threshold, otherwise GM plus local merge buffers.
    d.localMemoryBytes = 288U + (ub ? alignedBytes : stageBytes);
    d.maxNumBlocks = logicalBlocks;
    d.hostNumBlocks = requests * logicalBlocks;
    TORCH_CHECK(d.scratchElements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / rows),
                "peer Plan workspace size overflow");
    TORCH_CHECK(p[13].dim() == 1 && p[13].numel() >= (ub ? 1 : static_cast<int64_t>(rows * d.scratchElements)),
                "peer Plan workspace too small");
    d.maxRows = rows; d.topk = t; d.capacity = c; d.maxToken = attrs[2];
    d.maxRequests = requests;
    d.blockSize = static_cast<int32_t>(block);

    auto data = std::shared_ptr<Resources>(new Resources, Release);
    TORCH_CHECK(aclrtGetCurrentContext(&data->context) == ACL_SUCCESS && data->context != nullptr,
                "peer Plan ACL context unavailable");
    auto cpu = at::empty({static_cast<int64_t>(sizeof(d))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(cpu.data_ptr(), &d, sizeof(d));
    data->tiling = cpu.to(device, false);  // Setup only, outside target capture.
    data->dynamicUb = d.localMemoryBytes;
    data->blocks = joint_plan_block_dim();
    TORCH_CHECK(data->blocks > 0, "peer Plan cannot query AIV count");
    data->tensors = std::move(p); data->attrs = std::move(attrs); data->jointEntry = jointEntry;
    for (const auto& tensor : data->tensors) data->bindings.emplace_back(tensor);
    data_ = std::move(data);
  }
  void Run(Tensors p, Attrs attrs) {
    const auto data = Get();
    TORCH_CHECK(data != nullptr, "peer Plan context closed");
    TORCH_CHECK(p.size() == 18 && attrs == data->attrs, "peer Plan context attrs/list changed");
    for (size_t i = 0; i < p.size(); ++i) data->bindings[i].Check(p[i]);
    const c10::OptionalDeviceGuard guard(data->tensors[0].device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [data, stream]() -> int {
      const auto& p = data->tensors;
      launch_peer_qli_plan(
          p[0].data_ptr(), p[1].data_ptr(), p[2].data_ptr(), p[3].data_ptr(), p[4].data_ptr(),
          p[5].data_ptr(), p[6].data_ptr(), p[7].data_ptr(), p[8].data_ptr(), p[9].data_ptr(),
          p[10].data_ptr(), p[11].data_ptr(), p[12].data_ptr(), p[13].data_ptr(), p[14].data_ptr(),
          p[15].data_ptr(), p[16].data_ptr(), p[17].data_ptr(), data->tiling.data_ptr(),
          data->blocks, data->dynamicUb, reinterpret_cast<void*>(stream));
      return 0;
    };
    at_npu::native::OpCommand::RunOpApi("peer_qli_plan", launch);
  }
  std::string Info() const {
    const auto data = Get();
    TORCH_CHECK(data != nullptr, "peer Plan context closed");
    return "{\"blocks\":" + std::to_string(data->blocks) + ",\"dynamic_ub_bytes\":" +
        std::to_string(data->dynamicUb) + ",\"joint_enabled\":" + std::to_string(data->attrs[4]) + "}";
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
static auto registration = torch::class_<Prepared>("mtp_joint_plan", "PreparedPeerQli")
    .def(torch::init<Tensors, Attrs, bool>()).def("run", &Prepared::Run)
    .def("info", &Prepared::Info).def("close", &Prepared::Close);
}  // namespace
