// CPU shim driver: execute the real VF/helpers, no Ascend address-space/timing model.
#include <climits>
#include "cpu_runtime.h"
#include "kernel_under_test.h"
extern "C" __attribute__((visibility("default"))) void run_variant(void** p, const int64_t* d, int32_t* scan) {
#if JOINT_OPT_MASK >= 37
  JointDirectTilingData tiling{};
  tiling.hashCapacity=d[1];tiling.workspaceRowElements=d[2];tiling.maxRows=d[3];
  tiling.topk=d[4];tiling.capacity=d[5];tiling.maxToken=d[6];tiling.maxRequests=d[7];
  tiling.blockSize=d[8];tiling.jointHashCapacity=d[9];tiling.scratchElements=d[10];
  const uint64_t chunk=(std::min(uint64_t{768},std::max(uint64_t{64},uint64_t{3}*d[5])/16U)/4U)*4U;
  tiling.localMemoryBytes=288U+ (d[10]<=112*1024/4?((d[10]*4+31)&~31ULL):(64+16*chunk)*4);
  SortedCpuBlocks=d[0];
  for(SortedCpuBlock=0;SortedCpuBlock<SortedCpuBlocks;++SortedCpuBlock)
    SortedDirectBody<true>(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],&tiling);
  return;
#else
#if JOINT_OPT_MASK == 0
  CpuWorkers workers(sparse_kv_plan::PLAN_THREADS);
#else
  CpuWorkers workers(sparse_kv_plan_config::PLAN_THREADS);
#endif
  for (int64_t start = 0; start < d[0]; ++start) {
    workers.Run([&] {
#define VF_ARGUMENTS scan, (int64_t*)p[0], (int64_t*)p[6], (int32_t*)p[1], \
    (int32_t*)p[2], (int32_t*)p[3], (int32_t*)p[4], \
    (int32_t*)p[7], (int32_t*)p[8], (int32_t*)p[9], (int32_t*)p[10], \
    (int32_t*)p[11], (int32_t*)p[12], (int32_t*)p[13], \
    d[1], d[2], start, d[0], d[3], d[4], d[5], d[6], d[7], \
    (int32_t)d[8], (int32_t*)p[14], (int32_t*)p[15], (int32_t*)p[16], d[9], d[10]
#if JOINT_OPT_MASK == 0
      SparseKvPlanRuntimeVf(scan, (int64_t*)p[0], (int64_t*)p[6], (int32_t*)p[1],
          (int32_t*)p[2], (int32_t*)p[3], (int32_t*)p[4], (int32_t*)p[7], (int32_t*)p[8],
          (int32_t*)p[9], (int32_t*)p[10], (int32_t*)p[11], (int32_t*)p[12], (int32_t*)p[13],
          d[1], d[2], start, d[0], d[3], d[4], d[5], d[6], d[7]);
#elif JOINT_OPT_MASK >= 16
      if (d[11] == 0) SparseKvPlanRuntimeVf<false>(VF_ARGUMENTS);
      else SparseKvPlanRuntimeVf<true>(VF_ARGUMENTS);
#elif JOINT_OPT_MASK & 1
      if (d[9] == 0) SparseKvPlanRuntimeVf<false>(VF_ARGUMENTS);
      else SparseKvPlanRuntimeVf<true>(VF_ARGUMENTS);
#else
      SparseKvPlanRuntimeVf(VF_ARGUMENTS);
#endif
#undef VF_ARGUMENTS
    });
  }
#endif
}

#if JOINT_OPT_MASK >= 22
extern "C" __attribute__((visibility("default"))) void run_merge_pair(int32_t* a, uint32_t na, int32_t* b, uint32_t nb, int32_t* out) {
  for (uint32_t t = 0; t < sparse_kv_plan_config::PLAN_THREADS; ++t)
    MergePairChunks(a, na, b, nb, out, t);
}
#endif
