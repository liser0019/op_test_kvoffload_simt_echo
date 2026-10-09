#pragma once
#include <cstdint>
#include "echo_lru_tiling.h"

// Pure SIMT: four launch arguments, with statically allocated 128 KiB UB.
void launch_echo_lru(int32_t* htd, int32_t* dth, int32_t* pri, int32_t* freeSlots,
                     int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                     uint8_t* resetMask, int32_t* curSlots, int32_t* missHostPos,
                     int32_t* missAllocFlat, uint8_t* missMask, void* workspace,
                     void* tiling, uint32_t blockNum, void* stream);

inline uint64_t EchoWorkspaceElements(uint32_t rows, uint32_t topk,
                                       uint32_t capacity, uint32_t blocks) {
  uint64_t npad = 1;
  while (npad < uint64_t(capacity) + 1) npad <<= 1;
  uint64_t base = uint64_t(rows) * (capacity + 1ULL + 4ULL + 3ULL * topk);
  base = (base + 1) & ~uint64_t(1);
  return base + uint64_t(blocks) * (4 * npad +
      ascend_kernel::ARGTOPK_HIST_THREADS * 256ULL +
      ascend_kernel::ARGTOPK_SCAT_THREADS * 256ULL + 256ULL +
      ascend_kernel::ARGTOPK_GM_NPAD_MAX);
}
