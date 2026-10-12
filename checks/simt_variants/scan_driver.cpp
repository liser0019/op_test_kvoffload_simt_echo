#include <climits>
#include <iostream>
#include <random>
#include <stdexcept>

#include "cpu_runtime.h"
#include "kernel_under_test.h"

int main() {
  CpuWorkers workers(32);
  std::mt19937 rng(20261012);
  std::vector<int32_t> workspace(sparse_kv_plan::PLAN_SCAN_STORAGE_ELEMENTS, -777);
  for (int test = 0; test < 128; ++test) {
    int32_t total = 0;
    std::vector<int32_t> expected(sparse_kv_plan::PLAN_WARP_COUNT);
    for (uint32_t warp = 0; warp < sparse_kv_plan::PLAN_WARP_COUNT; ++warp) {
      const int32_t hits = rng() % 33;
      const int32_t evicts = rng() % (33 - hits);
      const int32_t value = test % 2 ? evicts + (hits << COUNT_SHIFT) : hits;
      workspace[WARP_TOTALS_OFFSET + warp] = value;
      expected[warp] = total;
      total += value;
    }
    workers.Run([&] { PublishWarpBases(workspace.data(), threadIdx.x); });
    for (uint32_t warp = 0; warp < sparse_kv_plan::PLAN_WARP_COUNT; ++warp) {
      if (workspace[WARP_BASES_OFFSET + warp] != expected[warp]) throw std::runtime_error("warp base mismatch");
    }
    if (workspace[CONTROL_OFFSET + CONTROL_TOTAL] != total) throw std::runtime_error("block total mismatch");
  }
  std::cout << "PASS 128 warp-total scans; production-config threads=" << sparse_kv_plan::PLAN_THREADS
            << " warps=" << sparse_kv_plan::PLAN_WARP_COUNT << '\n';
}
