// SPDX-License-Identifier: Apache-2.0
// Prepared adapter for the ordered-query 512-thread bitmap Plan.
#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
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
#include "sparse_kv_plan_config.h"

namespace {
using Tensors = std::vector<at::Tensor>;
using Attrs = std::vector<int64_t>;

struct Binding {
  at::Tensor tensor;
  void* address;
  c10::TensorImpl* impl;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  explicit Binding(const at::Tensor& value)
      : tensor(value), address(value.data_ptr()), impl(value.unsafeGetTensorImpl()),
        sizes(value.sizes().vec()), strides(value.strides().vec()) {}
  void Check(const at::Tensor& value) const {
    TORCH_CHECK(value.defined() && value.unsafeGetTensorImpl() == impl &&
                    value.data_ptr() == address && value.device() == tensor.device() &&
                    value.scalar_type() == tensor.scalar_type() &&
                    value.sizes().vec() == sizes && value.strides().vec() == strides &&
                    value.is_contiguous(),
                "ordered bitmap Plan prepared tensor binding changed");
  }
};

struct Resources {
  Tensors tensors;
  std::vector<Binding> bindings;
  at::Tensor tiling;
  Attrs attrs;
  uint32_t blocks = 0;
  uint32_t dynamicUb = 0;
  aclrtContext context = nullptr;
};

void Release(Resources* resources) {
  aclrtContext previous = nullptr;
  (void)aclrtGetCurrentContext(&previous);
  auto status = resources->context == nullptr ? ACL_SUCCESS : aclrtSetCurrentContext(resources->context);
  if (status == ACL_SUCCESS) {
    status = aclrtSynchronizeDevice();
  }
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr, "ordered bitmap Plan cleanup failed %d; retain resources for device safety\n", status);
    if (previous != nullptr) {
      (void)aclrtSetCurrentContext(previous);
    }
    return;
  }
  delete resources;
  if (previous != nullptr) {
    (void)aclrtSetCurrentContext(previous);
  }
}

void Shape(const at::Tensor& tensor, std::initializer_list<int64_t> expected) {
  TORCH_CHECK(tensor.sizes().vec() == std::vector<int64_t>(expected),
              "ordered bitmap Plan tensor shape mismatch");
}

class Prepared : public torch::CustomClassHolder {
 public:
  Prepared(Tensors tensors, Attrs attrs) {
    TORCH_CHECK(tensors.size() == 17 && attrs.size() == 4,
                "ordered bitmap Plan expects 17 tensors / 4 attrs");
    for (size_t index = 0; index < attrs.size(); ++index) {
      TORCH_CHECK(attrs[index] > 0 && attrs[index] <= std::numeric_limits<int32_t>::max(),
                  "ordered bitmap Plan attrs out of int32 range");
    }
    const int64_t topk = attrs[0];
    const int64_t capacity = attrs[1];
    const int64_t maxToken = attrs[2];
    const int64_t blockSize = attrs[3];
    TORCH_CHECK(tensors[0].defined() && tensors[0].dim() == 1 && tensors[0].numel() > 0 &&
                    tensors[1].defined() && tensors[1].dim() == 2 && tensors[1].size(0) > 0,
                "ordered bitmap Plan request/query shape");
    const auto device = tensors[0].device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
                "ordered bitmap Plan requires NPU");
    const c10::OptionalDeviceGuard guard(device);
    for (size_t index = 0; index < tensors.size(); ++index) {
      TORCH_CHECK(tensors[index].defined() && tensors[index].is_contiguous() &&
                      tensors[index].device() == device &&
                      tensors[index].scalar_type() == ((index == 0 || index == 6) ? at::kLong : at::kInt),
                  "ordered bitmap Plan tensor dtype/device/contiguity");
    }
    const int64_t rows = tensors[1].size(0);
    const int64_t requests = tensors[0].numel();
    TORCH_CHECK(rows == 4 * requests && topk == 2048 && capacity == 8192,
                "ordered bitmap Plan MTP3 requires four rows per request, TopK2048 and capacity8192");
    TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max() / (capacity / blockSize),
                "ordered bitmap Plan resident page index exceeds int32");
    Shape(tensors[1], {rows, topk});
    for (int index : {2, 3, 4, 6, 10, 14, 15}) {
      Shape(tensors[index], {rows});
    }
    TORCH_CHECK(tensors[5].numel() == 1, "ordered bitmap Plan active_rows shape");
    Shape(tensors[7], {rows, capacity});
    Shape(tensors[8], {rows, capacity});
    for (int index : {9, 11, 12}) {
      Shape(tensors[index], {rows, topk});
    }
    Shape(tensors[16], {rows, capacity / blockSize});

    for (size_t mutableIndex = 6; mutableIndex < tensors.size(); ++mutableIndex) {
      for (size_t inputIndex = 0; inputIndex < mutableIndex; ++inputIndex) {
        TORCH_CHECK(!tensors[mutableIndex].is_alias_of(tensors[inputIndex]),
                    "ordered bitmap Plan mutable tensor alias");
      }
    }

    JointDirectTilingData tiling{};
    uint64_t workspaceElements = 0;
    TORCH_CHECK(sparse_kv_plan_config::MakeJointTiling(
                    requests, rows, capacity, maxToken, static_cast<int32_t>(blockSize),
                    tiling, workspaceElements),
                "ordered bitmap Plan tiling rejected shape");
    TORCH_CHECK(tensors[13].dim() == 1 && tensors[13].numel() >= static_cast<int64_t>(workspaceElements),
                "ordered bitmap Plan workspace too small");
    // The attached source has one supported hot shape: four rows per request,
    // capacity 8192. Any other value would mean the host/device layout drifted.
    TORCH_CHECK(tiling.localMemoryBytes == 181024U,
                "ordered bitmap Plan dynamic UB differs from pinned 181024-byte layout");

    auto resources = std::shared_ptr<Resources>(new Resources, Release);
    TORCH_CHECK(aclrtGetCurrentContext(&resources->context) == ACL_SUCCESS && resources->context != nullptr,
                "ordered bitmap Plan ACL context unavailable");
    auto cpuTiling = at::empty({static_cast<int64_t>(sizeof(tiling))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(cpuTiling.data_ptr(), &tiling, sizeof(tiling));
    resources->tiling = cpuTiling.to(device, false);
    resources->dynamicUb = tiling.localMemoryBytes;
    resources->blocks = sparse_kv_plan_config::OrderedBitmapBlockDim(requests);
    TORCH_CHECK(resources->blocks > 0, "ordered bitmap Plan cannot query AIV count");
    resources->tensors = std::move(tensors);
    resources->attrs = std::move(attrs);
    for (const auto& tensor : resources->tensors) {
      resources->bindings.emplace_back(tensor);
    }
    data_ = std::move(resources);
  }

  void Run(Tensors tensors, Attrs attrs) {
    const auto resources = Get();
    TORCH_CHECK(resources != nullptr, "ordered bitmap Plan context closed");
    TORCH_CHECK(tensors.size() == 17 && attrs == resources->attrs,
                "ordered bitmap Plan context attrs/list changed");
    for (size_t index = 0; index < tensors.size(); ++index) {
      resources->bindings[index].Check(tensors[index]);
    }
    const c10::OptionalDeviceGuard guard(resources->tensors[0].device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [resources, stream]() -> int {
      const auto& p = resources->tensors;
      launch_joint_plan_ordered_bitmap_512(
          true, p[0].data_ptr(), p[1].data_ptr(), p[2].data_ptr(), p[3].data_ptr(), p[4].data_ptr(),
          p[5].data_ptr(), p[6].data_ptr(), p[7].data_ptr(), p[8].data_ptr(), p[9].data_ptr(),
          p[10].data_ptr(), p[11].data_ptr(), p[12].data_ptr(), p[13].data_ptr(), p[14].data_ptr(),
          p[15].data_ptr(), p[16].data_ptr(), resources->tiling.data_ptr(), resources->blocks,
          resources->dynamicUb, reinterpret_cast<void*>(stream));
      return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ordered_bitmap_512_plan", launch);
  }

  std::string Info() const {
    const auto resources = Get();
    TORCH_CHECK(resources != nullptr, "ordered bitmap Plan context closed");
    return "{\"blocks\":" + std::to_string(resources->blocks) +
           ",\"dynamic_ub_bytes\":" + std::to_string(resources->dynamicUb) +
           ",\"joint_enabled\":1,\"algorithm\":\"ordered_bitmap_512\"}";
  }

  void Close() {
    std::shared_ptr<Resources> previous;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      previous.swap(data_);
    }
  }

 private:
  std::shared_ptr<Resources> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_;
  }

  mutable std::mutex mutex_;
  std::shared_ptr<Resources> data_;
};

static auto registration = torch::class_<Prepared>("mtp_joint_plan", "PreparedOrderedBitmap512")
    .def(torch::init<Tensors, Attrs>()).def("run", &Prepared::Run)
    .def("info", &Prepared::Info).def("close", &Prepared::Close);
}  // namespace
