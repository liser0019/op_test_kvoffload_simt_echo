#include "echo_lru_launch.h"
#include "echo_lru_kernel.h"

void launch_echo_lru(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* freeSlots,
                     int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                     uint8_t* resetMask, int32_t* curSlots, int32_t* missHostPos,
                     int32_t* missAllocFlat, uint8_t* missMask, void* workspace,
                     void* tiling, uint32_t blockNum, uint32_t blockDim, void* stream) {
  ascend_kernel::echo_lru_kernel<<<blockNum, blockDim, 0, stream>>>(
      htd, dth, lru, freeSlots, avail, fifo, pos, spl, resetMask, curSlots,
      missHostPos, missAllocFlat, missMask, workspace,
      reinterpret_cast<ascend_kernel::EchoLruTiling*>(tiling));
}
