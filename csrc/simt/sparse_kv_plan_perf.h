#pragma once
#include <cstdint>
#include "sparse_kv_plan_config.h"

// 每条逻辑 row 独占 12 个 int32 槽。冷/热路径使用不同槽，提前 return
// 不会伪装成后续阶段耗时为零；未走到的槽必须保留 host 哨兵。
constexpr uint32_t SIMT_PERF_NUM_STAMPS = 12U;
// 计时写入单独按 512B 分隔，避免不同 block 的 row 戳和 scratch 共用缓存行。
// 这是观测区的保守隔离布局，不改变算法 scratch 或动态 UB。
constexpr uint32_t SIMT_PERF_ROW_STRIDE = 128U;  // int32 elements
enum SimtPerfSlot : uint32_t {
  SIMT_ENTRY = 0, SIMT_PREPARE = 1,
  SIMT_HASH_INIT = 2, SIMT_HASH_INSERT = 3, SIMT_CLASSIFY = 4,
  SIMT_HIT_MISS_SCAN = 5, SIMT_ASSIGN = 6, SIMT_LRU_END = 7,
  SIMT_COLD_CLEAR = 8, SIMT_COLD_SCAN = 9,
  SIMT_COLD_ASSIGN = 10, SIMT_COLD_END = 11
};

// ABCDF scratch 为 3H+C+T；UB 路径仍保留一个 GM 占位元素。
// 打点区在完整 maxRows scratch 之后，不能按 activeRows 计算偏移。
inline uint64_t SimtScratchElements(uint64_t rows, uint64_t rowElements) {
  return rowElements <= sparse_kv_plan::PLAN_ROW_UB_LIMIT_ELEMENTS ? 1U : rows * rowElements;
}
inline uint64_t SimtWorkspaceElements(uint64_t rows, uint64_t rowElements) {
  uint64_t required = SimtScratchElements(rows, rowElements);
#ifdef SIMT_PERF_STEP
  required = (required + SIMT_PERF_ROW_STRIDE - 1U) & ~uint64_t(SIMT_PERF_ROW_STRIDE - 1U);
  required += rows * SIMT_PERF_ROW_STRIDE;
#endif
  return required;
}
