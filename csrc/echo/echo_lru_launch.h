#pragma once
#include <cstdint>
#include "echo_lru_tiling.h"

// blockDim 由 plugin 按 tiling.topkDim 自适应传入 (ceil(topk/4) 取整到 32, 夹在
// [64, ECHO_LRU_THREAD_NUM]): topk=2048 → 512 (消融实测全面甜点), topk=4096 → 1024
// (保证 OpMissCumsum 4 元素并行门 4*blockDim >= topk). kernel 内部全部 blockDim.x
// 自适应, __launch_bounds__(ECHO_LRU_THREAD_NUM) 是上限, 允许以更少线程启动.
// [v2] 参数 pri → lru (秩→槽位顺序数组, 替代 v1 时间戳; 形状不变 [rows, B+1]).
void launch_echo_lru(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* freeSlots,
                     int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                     uint8_t* resetMask, int32_t* curSlots, int32_t* missHostPos,
                     int32_t* missAllocFlat, uint8_t* missMask, void* workspace,
                     void* tiling, uint32_t blockNum, uint32_t blockDim, void* stream);

// GM workspace 布局 (与 EchoLruKernel::Init 逐项一致, [v2]):
//   [0, base)           每行: wsCumsum(B+1) + pushCount/numHits/numMisses/freeSize(4)
//                       + freeIdx(topk) + rawSlots(topk) + wsMissRank(topk)
//                       + flags(B+1)   [v2.2] 分类位 (hit=gen / spec 移除=genSpec,
//                       世代号等值链; 值域自洽, 无 per-invocation 清零 —— 分配时清零一次即可)
//   [align2(base), +bn*ECHO_PERF_NUM_STEPS)      [ECHO_PERF_STEP] 每核分步打点区
//                       ([v2.1-prof] t0 绝对时间戳低 32 位, 仅 perf 构建分配)
//   (v1 的 argtopk 诊断 dump 区 [bn*256 + bn*ARGTOPK_GM_NPAD_MAX] 已随基数选择删除)
inline uint64_t EchoWorkspaceElements(uint32_t rows, uint32_t topk,
                                       uint32_t capacity, uint32_t blocks) {
  uint64_t base = uint64_t(rows) * (2ULL * (capacity + 1ULL) + 4ULL + 3ULL * topk);
  base = (base + 1) & ~uint64_t(1);
  uint64_t perBlock = 0;
#ifdef ECHO_PERF_STEP
  perBlock += ascend_kernel::ECHO_PERF_NUM_STEPS;
#endif
  return base + uint64_t(blocks) * perBlock;
}
