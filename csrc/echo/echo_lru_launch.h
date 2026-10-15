#pragma once
#include <cstdint>
#include "echo_lru_tiling.h"

// blockDim 由 plugin 按 tiling.topkDim 自适应传入 (ceil(topk/4) 取整到 32, 夹在
// [64, ECHO_LRU_THREAD_NUM]): topk=2048 → 512 (消融实测全面甜点), topk=4096 → 1024
// (保证 OpMissCumsum 4 元素并行门 4*blockDim >= topk). kernel 内部全部 blockDim.x
// 自适应, __launch_bounds__(ECHO_LRU_THREAD_NUM) 是上限, 允许以更少线程启动.
void launch_echo_lru(int32_t* htd, int32_t* dth, int32_t* pri, int32_t* freeSlots,
                     int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                     uint8_t* resetMask, int32_t* curSlots, int32_t* missHostPos,
                     int32_t* missAllocFlat, uint8_t* missMask, void* workspace,
                     void* tiling, uint32_t blockNum, uint32_t blockDim, void* stream);

// GM workspace 布局 (与 EchoLruKernel::Init 逐项一致):
//   [0, base)           每行: wsCumsum(B+1) + pushCount/numHits/numMisses/freeSize(4)
//                       + freeIdx(topk) + rawSlots(topk) + wsMissRank(topk)
//   [align2(base), +bn*256)                      [诊断3] 每核 pass0 归并直方图 dump
//   [.., +bn*ARGTOPK_GM_NPAD_MAX)                [诊断3] 每核打包 srcKey dump
//   [.., +bn*ECHO_PERF_NUM_STEPS)                [ECHO_PERF_STEP] 每核分步周期 (仅 perf 构建分配)
inline uint64_t EchoWorkspaceElements(uint32_t rows, uint32_t topk,
                                       uint32_t capacity, uint32_t blocks) {
  uint64_t base = uint64_t(rows) * (capacity + 1ULL + 4ULL + 3ULL * topk);
  base = (base + 1) & ~uint64_t(1);
  uint64_t perBlock = 256ULL + ascend_kernel::ARGTOPK_GM_NPAD_MAX;
#ifdef ECHO_PERF_STEP
  perBlock += ascend_kernel::ECHO_PERF_NUM_STEPS;
#endif
  return base + uint64_t(blocks) * perBlock;
}
