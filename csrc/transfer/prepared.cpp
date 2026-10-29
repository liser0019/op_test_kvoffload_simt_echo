// SPDX-License-Identifier: Apache-2.0
// 仅新增Host接入；ABCDF与Transfer计算kernel均不修改。
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
#include "../plan_variants/13_abcdf/symbol_aliases.h"
#include "../plan_variants/common/sparse_kv_plan_launch.h"
#include "launch.h"
#include "../transfer_variants/launch.h"

namespace {
using Tensors = std::vector<at::Tensor>;
using Attrs = std::vector<int64_t>;

// 只读取CPU侧Tensor元信息；运行时不读设备内容。
// TensorImpl、地址、尺寸、步长、dtype、device均固定；内容允许被恢复/更新。
struct Binding {
  at::Tensor tensor;
  void* address;
  c10::TensorImpl* impl;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  at::ScalarType dtype;
  c10::Device device;
  explicit Binding(const at::Tensor& t)
      : tensor(t), address(t.data_ptr()), impl(t.unsafeGetTensorImpl()),
        sizes(t.sizes().vec()), strides(t.strides().vec()),
        dtype(t.scalar_type()), device(t.device()) {}
  void Check(const at::Tensor& t) const {
    TORCH_CHECK(t.defined() && t.unsafeGetTensorImpl() == impl &&
                t.device() == device && t.scalar_type() == dtype &&
                t.sizes().vec() == sizes && t.strides().vec() == strides &&
                t.data_ptr() == address && t.is_contiguous(),
                "prepared context binding changed: tensor identity/address/shape/stride/dtype/device");
  }
};

struct Resources {
  Tensors p, x;  // p:原Plan的15张量；x:bases/HostK/HostV/residentK/residentV。
  Attrs attrs;   // T,C,maxToken,blockSize,hostNumBlocks,bytesK,bytesV。
  std::vector<Binding> bindings;
  at::Tensor planTiling, transferTiling;
  uint32_t planBlocks = 0, transferBlocks = 0, dynUB = 0;
  int64_t basesVersion = 0;
  aclrtContext context = nullptr;
};

// caller与RunOpApi闭包共同持有资源，无全局缓存。
// 最后一份所有权释放时等待设备，保证异步kernel不访问已释放的tiling/映射/缓冲。
// 该等待不在两条kernel之间；正常runner在校验/同步后才释放上下文。
void ReleaseResources(Resources* resource) {
  aclrtContext previous = nullptr;
  (void)aclrtGetCurrentContext(&previous);
  auto status = resource->context == nullptr ? ACL_SUCCESS : aclrtSetCurrentContext(resource->context);
  if (status == ACL_SUCCESS) status = aclrtSynchronizeDevice();
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr, "prepared cleanup failed %d; retain all resources for device safety\n", status);
    if (previous != nullptr) (void)aclrtSetCurrentContext(previous);
    return;
  }
  delete resource;
  if (previous != nullptr) (void)aclrtSetCurrentContext(previous);
}

void Shape(const at::Tensor& t, std::initializer_list<int64_t> shape) {
  TORCH_CHECK(t.sizes().vec() == std::vector<int64_t>(shape), "prepared tensor shape mismatch");
}
int64_t Product(int64_t a, int64_t b) {
  TORCH_CHECK(a > 0 && b > 0 && a <= std::numeric_limits<int64_t>::max() / b,
              "prepared size overflow");
  return a * b;
}
template <typename T>
at::Tensor Upload(const T& value, c10::Device device) {
  auto cpu = at::empty({static_cast<int64_t>(sizeof(T))}, at::TensorOptions().dtype(at::kByte));
  std::memcpy(cpu.data_ptr(), &value, sizeof(T));
  return cpu.to(device, false);  // 仅constructor调用，采集前完成。
}

class PreparedPlanTransfer : public torch::CustomClassHolder {
 public:
  PreparedPlanTransfer(Tensors p, Tensors x, Attrs attrs) {
    TORCH_CHECK(p.size() == 15 && x.size() == 5 && attrs.size() == 9,
                "prepared expects 15 Plan tensors, 5 Transfer tensors and 9 attributes");
    TORCH_CHECK(attrs[7] >= -1 && attrs[7] <= 15 && attrs[8] > 0 && attrs[8] <= 64,
                "prepared variant/core limit invalid");
    for (const auto& t : p) TORCH_CHECK(t.defined(), "undefined Plan tensor");
    for (const auto& t : x) TORCH_CHECK(t.defined(), "undefined Transfer tensor");
    const auto device = p[0].device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "prepared requires NPU");
    const c10::OptionalDeviceGuard guard(device);
    for (size_t index = 0; index < 7; ++index) {
      const auto value = attrs[index];
      TORCH_CHECK(value > 0 && value <= std::numeric_limits<int32_t>::max(),
                  "prepared attributes must be positive int32");
    }
    const auto t = attrs[0], c = attrs[1], length = attrs[2], block = attrs[3], hostBlocks = attrs[4];
    TORCH_CHECK(t <= std::numeric_limits<int32_t>::max() / 2, "prepared topk exceeds Plan limit");
    TORCH_CHECK(p[0].dim() == 1 && p[0].numel() > 0 &&
                p[1].dim() == 2 && p[1].size(0) > 0 && p[5].dim() == 2 &&
                p[5].size(0) == p[0].numel() && p[5].size(1) > 0, "prepared request/query/page shape");
    const int64_t rows = p[1].size(0), requests = p[0].numel(), blocks = p[5].size(1);
    for (size_t i = 0; i < p.size(); ++i) {
      TORCH_CHECK(p[i].device() == device && p[i].is_contiguous() &&
                  p[i].scalar_type() == ((i == 0 || i == 7) ? at::kLong : at::kInt),
                  "prepared Plan dtype/device/contiguity");
    }
    Shape(p[1], {rows, t});
    for (int i : {2, 3, 4, 7, 11}) Shape(p[i], {rows});
    TORCH_CHECK(p[6].numel() == 1, "prepared active_rows shape");
    Shape(p[8], {rows, c}); Shape(p[9], {rows, c});
    for (int i : {10, 12, 13}) Shape(p[i], {rows, t});
    const auto tiling = MakeSparseKvPlanTiling(rows, t, c, length, requests, blocks, hostBlocks, block);
    const auto rowElements = tiling.workspaceRowElements;
    TORCH_CHECK(rowElements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / rows),
                "prepared workspace overflow");
    const auto required = rowElements <= sparse_kv_plan::PLAN_ROW_UB_LIMIT_ELEMENTS ?
        int64_t{1} : static_cast<int64_t>(rowElements * rows);
    TORCH_CHECK(p[14].dim() == 1 && p[14].numel() >= required, "prepared workspace too small");
    for (int i : {0, 3, 4}) {
      TORCH_CHECK(x[i].device() == device && x[i].is_contiguous(), "prepared Transfer device/stride");
    }
    TORCH_CHECK(x[0].scalar_type() == at::kLong && x[0].numel() == 2, "prepared bases shape/dtype");
    const int64_t sourceTokens = Product(hostBlocks, block);
    for (int i : {1, 2}) {
      TORCH_CHECK(x[i].device().is_cpu() && x[i].scalar_type() == at::kByte && x[i].is_contiguous() &&
                  x[i].numel() == Product(sourceTokens, attrs[4 + i]), "prepared Host pool shape/dtype");
      TORCH_CHECK(x[i + 2].scalar_type() == at::kByte, "prepared resident dtype");
      Shape(x[i + 2], {rows, c, attrs[4 + i]});
    }
    TORCH_CHECK(!x[1].is_alias_of(x[2]) && !x[3].is_alias_of(x[4]), "prepared K/V alias");

    // 仅setup读取一次bases，验证真正Host映射与DVA一致。目标调用无D2H。
    const auto bases = x[0].cpu().contiguous();
    const auto* values = bases.data_ptr<int64_t>();
    for (int i : {1, 2}) {
      void* mapped = nullptr;
      const auto status = aclrtHostGetDevicePointer(x[i].data_ptr(), &mapped, 0);
      TORCH_CHECK(status == ACL_SUCCESS && mapped != nullptr &&
                  reinterpret_cast<uintptr_t>(mapped) == static_cast<uint64_t>(values[i - 1]),
                  "prepared Host pool is not registered/mapped or DVA does not match");
    }

    auto data = std::shared_ptr<Resources>(new Resources, ReleaseResources);
    TORCH_CHECK(aclrtGetCurrentContext(&data->context) == ACL_SUCCESS && data->context != nullptr,
                "prepared cannot obtain ACL context");
    data->p = std::move(p); data->x = std::move(x); data->attrs = std::move(attrs);
    data->planTiling = Upload(tiling, device);
    data->dynUB = tiling.localMemoryBytes;
    SparseKvTransferDirectTilingData transfer{};
    transfer.maxRows = rows; transfer.maxRequests = requests; transfer.maxNumBlocks = blocks;
    transfer.topk = t; transfer.capacity = c; transfer.hostNumBlocks = hostBlocks;
    transfer.blockSize = static_cast<int32_t>(block);
    transfer.tokenSizeBytesK = static_cast<int32_t>(data->attrs[5]);
    transfer.tokenSizeBytesV = static_cast<int32_t>(data->attrs[6]);
    data->transferTiling = Upload(transfer, device);
    data->planBlocks = calc_sparse_kv_plan_block_dim();
    data->transferBlocks = sparse_kv_transfer_block_dim();
    if (data->attrs[8] < data->transferBlocks) data->transferBlocks = static_cast<uint32_t>(data->attrs[8]);
    TORCH_CHECK(data->planBlocks > 0 && data->transferBlocks > 0, "prepared AIV query failed");
    for (const auto& tensor : data->p) data->bindings.emplace_back(tensor);
    for (const auto& tensor : data->x) data->bindings.emplace_back(tensor);
    data->basesVersion = data->x[0]._version();
    data_ = std::move(data);
  }

  void Run(Tensors p, Tensors x, Attrs attrs, std::string mode) {
    auto data = Get();
    TORCH_CHECK(data != nullptr, "prepared context is closed");
    TORCH_CHECK(mode == "plan" || mode == "transfer" || mode == "chain", "prepared invalid mode");
    TORCH_CHECK(p.size() == 15 && x.size() == 5 && attrs == data->attrs,
                "prepared context attributes/list lengths changed");
    for (size_t i = 0; i < p.size(); ++i) data->bindings[i].Check(p[i]);
    for (size_t i = 0; i < x.size(); ++i) data->bindings[p.size() + i].Check(x[i]);
    TORCH_CHECK(x[0]._version() == data->basesVersion, "prepared mapped DVA tensor was modified");
    const c10::OptionalDeviceGuard guard(data->p[0].device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const bool plan = mode != "transfer", transfer = mode != "plan";
    auto launch = [data, stream, plan, transfer]() -> int {
      const auto& p = data->p;
      const auto& x = data->x;
      if (plan) {
        launch_sparse_kv_plan(
            p[0].data_ptr(), p[1].data_ptr(), p[2].data_ptr(), p[3].data_ptr(),
            p[4].data_ptr(), p[5].data_ptr(), p[6].data_ptr(), p[7].data_ptr(),
            p[8].data_ptr(), p[9].data_ptr(), p[10].data_ptr(), p[11].data_ptr(),
            p[12].data_ptr(), p[13].data_ptr(), p[14].data_ptr(), data->planTiling.data_ptr(),
            data->planBlocks, data->dynUB, reinterpret_cast<void*>(stream));
      }
      // 连续原kernel下发：此处没有tiling申请/上传、设备值读取或显式同步。
      if (transfer) {
        if (data->attrs[7] < 0) {
          launch_sparse_kv_transfer(
              p[11].data_ptr(), p[12].data_ptr(), p[13].data_ptr(), p[4].data_ptr(), p[5].data_ptr(),
              x[0].data_ptr(), p[6].data_ptr(), x[3].data_ptr(), x[4].data_ptr(),
              data->transferTiling.data_ptr(), data->transferBlocks, reinterpret_cast<void*>(stream));
        } else {
          launch_sparse_kv_transfer_variant(static_cast<int32_t>(data->attrs[7]),
              p[11].data_ptr(), p[12].data_ptr(), p[13].data_ptr(), p[4].data_ptr(), p[5].data_ptr(),
              x[0].data_ptr(), p[6].data_ptr(), x[3].data_ptr(), x[4].data_ptr(),
              data->transferTiling.data_ptr(), data->transferBlocks, reinterpret_cast<void*>(stream));
        }
      }
      return 0;
    };
    at_npu::native::OpCommand::RunOpApi("abcdf_plan_transfer_prepared", launch);
  }

  void Close() {
    std::shared_ptr<Resources> previous;
    { std::lock_guard<std::mutex> lock(mutex_); previous.swap(data_); }
    // 在锁外销毁，析构可能等待设备。
  }
  std::string Info() const {
    const auto data = Get();
    TORCH_CHECK(data != nullptr, "prepared context is closed");
    return "{\"plan_blocks\":" + std::to_string(data->planBlocks) +
           ",\"transfer_blocks\":" + std::to_string(data->transferBlocks) +
           ",\"variant_mask\":" + std::to_string(data->attrs[7]) +
           ",\"tiling_upload\":\"setup_only\",\"ownership\":\"caller_and_queued_callback\"}";
  }

 private:
  std::shared_ptr<Resources> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_;
  }
  mutable std::mutex mutex_;
  std::shared_ptr<Resources> data_;
};

static auto registration = torch::class_<PreparedPlanTransfer>("sparse_kv_plan_op", "PreparedPlanTransfer")
    .def(torch::init<Tensors, Tensors, Attrs>())
    .def("run", &PreparedPlanTransfer::Run)
    .def("close", &PreparedPlanTransfer::Close)
    .def("info", &PreparedPlanTransfer::Info);
}  // namespace
