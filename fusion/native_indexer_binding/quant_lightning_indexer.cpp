#include <dlfcn.h>
#include <string>
#include <vector>

#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include "aclnn_common.h"

namespace {
using namespace at_npu::native;

at::Tensor Run(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
              const at::Tensor &queryScale, const at::Tensor &keyScale,
              const at::Tensor &queryLengths, const at::Tensor &keyLengths,
              const at::Tensor &blockTable, int64_t queryQuantMode, int64_t keyQuantMode,
              std::string layoutQuery, std::string layoutKey, int64_t sparseCount,
              int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
              const at::Tensor &output) {
    const c10::OptionalDeviceGuard deviceGuard(query.device());
    TORCH_CHECK(layoutQuery == "TND" && query.dim() == 3 &&
                layoutKey == "PA_BSND" && key.dim() == 4,
                "native Indexer benchmark requires TND query and PA_BSND key");
    const std::vector<int64_t> shape{query.size(0), key.size(2), sparseCount};
    TORCH_CHECK(output.sizes() == at::IntArrayRef(shape) && output.is_contiguous() &&
                output.device() == query.device() && output.scalar_type() == at::kInt,
                "native Indexer output shape, layout, device or dtype mismatch");
    for (const auto *input : {&query, &key, &weights, &queryScale, &keyScale,
                              &queryLengths, &keyLengths, &blockTable}) {
        TORCH_CHECK(!output.is_alias_of(*input), "native Indexer output aliases an input");
    }
    char *layoutQueryPtr = layoutQuery.data();
    char *layoutKeyPtr = layoutKey.data();
    ACLNN_CMD(aclnnQuantLightningIndexer, query, key, weights, queryScale, keyScale,
              queryLengths, keyLengths, blockTable, queryQuantMode, keyQuantMode,
              layoutQueryPtr, layoutKeyPtr, sparseCount, sparseMode, preTokens, nextTokens, output);
    return output;
}

std::vector<std::string> OpApiPaths() {
    std::vector<std::string> paths;
    for (const auto *symbol : {"aclnnQuantLightningIndexerGetWorkspaceSize",
                                "aclnnQuantLightningIndexer"}) {
        void *address = GetOpApiFuncAddr(symbol);
        Dl_info info{};
        TORCH_CHECK(address != nullptr && dladdr(address, &info) != 0 && info.dli_fname,
                    "cannot resolve native Indexer symbol ", symbol);
        paths.emplace_back(info.dli_fname);
    }
    return paths;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("_fusion_identity", []() { return "@FUSION_SOURCE_ID@"; });
    m.def("_op_api_paths", &OpApiPaths);
    m.def("run", &Run, py::arg("query"), py::arg("key"), py::arg("weights"),
          py::arg("query_dequant_scale"), py::arg("key_dequant_scale"),
          py::arg("actual_seq_lengths_query"), py::arg("actual_seq_lengths_key"),
          py::arg("block_table"), py::arg("query_quant_mode"), py::arg("key_quant_mode"),
          py::arg("layout_query"), py::arg("layout_key"), py::arg("sparse_count"),
          py::arg("sparse_mode"), py::arg("pre_tokens"), py::arg("next_tokens"),
          py::arg("topk_out"));
}
