// CPU shim driver: execute the real VF/helpers, no Ascend address-space/timing model.
#include <climits>
#include "cpu_runtime.h"
#include "kernel_under_test.h"
extern "C" __attribute__((visibility("default"))) void run_variant(void** p, const int64_t* d, int32_t* scan) {
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
}
