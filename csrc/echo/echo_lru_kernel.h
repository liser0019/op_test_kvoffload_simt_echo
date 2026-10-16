// Adapted from peer_code/echo_lru_kernel.asc; computational stages retained.
// echo_lru 纯 SIMT 版本 (Path A: 按行分块) — LRU 元数据多步 kernel
//   (Step0 reset → Step6 set_update, 原 Step1 计数子步与 Step7 final slots
//    已分别并入 warp 归约与 OpSetUpdate, 见 [P0]/[P2] 标记)。
//
// 纯 SIMT 范式 (对照 examples/03_simt_api gather_2d / grid_config):
//   - 不 include kernel_operator.h (那是 SIMD 头树, 纯 SIMT 后端找不到 kernel_tpipe.h).
//   - kernel 参数与设备指针都用裸指针 (float*/int32_t*), 不用 GM_ADDR / __gm__.
//     __gm__ 是混合编程才需要的地址空间限定符, 纯 SIMT 默认指针即 GM.
//   - 设备函数 __aicore__ inline (被 __global__ 调用, 见 reduce_sum.asc/stack_overflow.asc); 入口 __global__ __launch_bounds__(N).
//   - host 四参 <<<blockNum, 512, 0, stream>>>; 子 Op 直接调用 (无 VF_CALL).
//
// 排布 (Path A): 算法每行完全独立 (golden.py 全是 for row in range(R), 行间零依赖).
//   每核只处理自己的连续行段 [blockIdx.x*rowsPerCore, +rowCount), 行内 ECHO_LRU_THREAD_NUM
//   线程 grid-stride 扫列. 每核读/写的都是自己负责的行 → 无跨核写后读.
// 同步: 步间用块内 asc_syncthreads (解决行内跨线程 RAW, 如 OpProtect 以 topkDim 步长写
//   pri[row,s]、OpArgtopkRow 以 B+1 步长读 pri[row,i], 同地址被不同线程访问);
//   asc_dcci_entire 刷本核 D-cache 防单核内写后读 stale. 行间零依赖故无需跨核栅栏.
// argtopk: 4 趟 MSB-first 基数选择 (radix select), 见 OpArgtopkRow 注释.

#include "echo_lru_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"     // asc_syncthreads
#include "simt_api/device_atomic_functions.h"   // asc_atomic_add
#include "simt_api/device_functions.h"          // asc_dcci_entire
#include "simt_api/device_warp_functions.h"     // asc_shfl_up / asc_reduce_add ([P0/P1/P3] warp 协作)
// clock() (PerfBegin/PerfEnd 分步打点) 由 common_functions.h → utils/debug/asc_time.h 提供
// (3510 纯 SIMT 下返回 uint64_t 周期计数).
// Path A does not use cooperative_groups; avoid an unnecessary CANN 9.2 dependency.

namespace ascend_kernel {

constexpr int32_t I32_MAX = 2147483647;

// ===========================================================================
// [P0/P1/P3/P4 2026-10-13] warp/block 协作辅助 (两级结构对照 SIMT 基线 BlockExclusiveScan)。
// dav-3510 SIMT 实测特性:
//   - 同地址 GM atomic 串行化严重 (~9K 周期/次, 1024 线程打同一地址时 OpHitMiss 高达
//     ~9.4M 周期), 块内计数一律走 warp 归约/扫描 + t0 汇总;
//   - 1024 线程 asc_syncthreads ~6-8K 周期 (fs=0 早退路径实测), 栅栏次数本身是
//     大成本, 优先 warp 内解决、并逐个证明可删的栅栏。
// warp 数按 blockDim.x 运行时推导 (设备 1024→32 warp; CPU 回归 128→4 warp)。
// ===========================================================================

// 打包块内独占扫描: value = 字段A | (字段B << shift), 两个字段各自的块内总和都
//   必须小于 (1 << shift) (打包加法无跨字段进位, 由调用方保证)。返回本线程的
//   exclusive (同打包格式); t0 另把全块 grand total (打包) 写入 control[0]。
//   2 次栅栏: warp shuffle 一级 + t0 扫 warp total 二级。
//   [P4] 基线的第 3 次"保护 scratch 复用"栅栏已删, 顺序论证 (所有调用方都是
//   统一控制流的逐行循环):
//   (a) 线程 X 读 warpBases/control (b2 之后、返回前) vs t0 下一轮写它们
//       (b1(next) 之后): t0 须等全部线程到达 b1(next), X 的读在其到达之前 → 读先于写;
//   (b) t0 读 warpTotals (b1~b2 之间) vs lane31 下一轮写它 (b2 之后): t0 的读
//       在 t0 到达 b2 之前, lane31 的写在 b2 释放之后 → 读先于写。两对均安全。
__aicore__ inline int32_t PackedBlockScan(int32_t packedValue, int32_t* warpTotals) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5;
    uint32_t warpCount = blockDim.x >> 5;
    int32_t inclusive = packedValue;
    for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
        int32_t prior = asc_shfl_up(inclusive, offset, 32u);
        if (lane >= offset) inclusive += prior;
    }
    if (lane == 31u) warpTotals[warp] = inclusive;
    asc_syncthreads();
    // [Scan-v2] B2 已删: base 是 warpTotals 的确定性函数, 每线程本地重算自己
    //   warp 的前缀即可 (W ≤ 32 次 UB 读), 无需 t0 广播 warpBases。
    int32_t base = 0;
    for (uint32_t w = 0; w < warp; ++w) base += warpTotals[w];
    return base + inclusive - packedValue;
}

// ---- argtopk 缓冲 (定义见 echo_lru_tiling.h, host/device 共享) ----
// [radix select] OpArgtopkRow 用 4 趟 MSB-first 基数选择: key/hist/globalCnt/scalar
//   全在单大 __ubuf__ 数组 argtopkUb 分段 (见 EchoLruKernel::Init), 无 GM ping-pong.
//   GM workspace 尾部仅 [诊断3] debug dump 区 (+ ECHO_PERF_STEP 分步周期区).
//   host/torch 侧 CHECK(B <= ARGTOPK_B_MAX).

// ===========================================================================
// 行段宏: 每核负责 [startRow, startRow+rowCount). R 以内才有效 (尾核 tailRows).
// ===========================================================================
// 元素级 step: 外层按核遍历自己行段, 内层 ECHO_LRU_THREAD_NUM(=1024) 线程 grid-stride 扫该行 cols.
// 行内串行子步: threadIdx.x==0 串行扫自己行段 (cumsum / freeSize / fifo++ 等).
// ===========================================================================

// ---- Step 0: reset ----
__aicore__ inline void OpResetRows(
    uint32_t R, uint32_t maxModelLenP1, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* htdGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        if (resetMaskGm[row] == 0) continue;
        uint32_t rowOff = row * maxModelLenP1;
        for (uint32_t col = threadIdx.x; col < maxModelLenP1; col += blockDim.x) {
            htdGm[rowOff + col] = (col == 0) ? 0 : I32_MAX;
        }
    }
}

__aicore__ inline void OpResetDthPri(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* dthGm, int32_t* priGm) {
    uint32_t cols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        if (resetMaskGm[row] == 0) continue;
        uint32_t rowOff = row * cols;
        for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            dthGm[rowOff + col] = I32_MAX;
            priGm[rowOff + col] = (col == 0) ? I32_MAX : -1;
        }
    }
}

__aicore__ inline void OpResetFree(
    uint32_t R, uint32_t B, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* freeGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        if (resetMaskGm[row] == 0) continue;
        uint32_t rowOff = row * freeStackSize;
        for (uint32_t col = threadIdx.x; col < freeStackSize; col += blockDim.x) {
            freeGm[rowOff + col] = (col < B) ? (int32_t)(B - col) : 0;
        }
    }
}

__aicore__ inline void OpResetAvailFifo(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* availGm, int32_t* fifoGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            if (resetMaskGm[row] != 0) {
                availGm[row] = (int32_t)B;
                fifoGm[row] = 1;
            }
        }
    }
}

// ---- Step 0.5: invalidate speculative suffix ----
__aicore__ inline void OpInvalidateSpecSuffix(
    uint32_t R, uint32_t B, uint32_t maxModelLenP1, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* splGm, int32_t* dthGm, int32_t* priGm,
    int32_t* htdGm, int32_t* wsCumsumGm) {
    uint32_t cols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t spl = splGm[row];
        uint32_t rowOff = row * cols;
        uint32_t htdRowOff = row * maxModelLenP1;
        for (uint32_t slot = threadIdx.x; slot < cols; slot += blockDim.x) {
            int32_t tok = dthGm[rowOff + slot];
            bool spec = (tok >= spl) && (tok != I32_MAX);
            wsCumsumGm[rowOff + slot] = spec ? 1 : 0;
            if (spec) {
                dthGm[rowOff + slot] = I32_MAX;
                priGm[rowOff + slot] = -1;
                int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
                htdGm[htdRowOff + tokSafe] = I32_MAX;
            }
        }
    }
}

__aicore__ inline void OpSpecCumsum(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* wsCumsumGm, int32_t* pushCountGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t acc = 0;
            for (uint32_t slot = 0; slot <= B; ++slot) {
                int32_t m = wsCumsumGm[row * (B + 1) + slot];
                acc += m;
                wsCumsumGm[row * (B + 1) + slot] = acc;
            }
            pushCountGm[row] = acc;
        }
    }
}

__aicore__ inline void OpSpecPushFree(
    uint32_t R, uint32_t B, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* wsCumsumGm, int32_t* availGm, int32_t* freeGm) {
    uint32_t cols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t avail = availGm[row];
        uint32_t rowOff = row * cols;
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t slot = threadIdx.x; slot < cols; slot += blockDim.x) {
            // wsCumsum 经 OpSpecCumsum 前缀和后 = 到 slot 为止的 spec 总数 (inclusive).
            //   判 "本 slot 是否 spec" 须用差分: cum[slot]-cum[slot-1] == 1 (slot 0 时 cum[-1]=0).
            //   ⚠ 不能用 `cum>0`: 前缀和后末段非 spec slot 的 cum 也 >0 (= 总 spec 数), 会误当 spec,
            //   与真 spec slot 抢写同一 writeIdx (cum 相同) → 并发覆盖, free 栈值错乱
            //   (实测 B=64: slot 32/64 覆盖 slot 30/60 的位置, aslot 错). 见 echo-lru-argtopk-debug-status.
            int32_t cum = wsCumsumGm[rowOff + slot];
            int32_t prevCum = (slot == 0) ? 0 : wsCumsumGm[rowOff + slot - 1];
            if (cum - prevCum == 1) {
                int32_t writeIdx = avail + cum - 1;
                if (writeIdx < 0) writeIdx = 0;
                if (writeIdx >= (int32_t)freeStackSize) writeIdx = (int32_t)freeStackSize - 1;
                freeGm[freeRowOff + writeIdx] = (int32_t)slot;
            }
        }
    }
}

__aicore__ inline void OpSpecAvailAdd(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* availGm, int32_t* pushCountGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            availGm[row] = availGm[row] + pushCountGm[row];
        }
    }
}

// ---- Step 1: hit/miss + miss 计数 + freeSize ----
// [P0 2026-10-13] miss 计数改 warp 归约 + t0 汇总写 GM。原实现每线程一次
//   asc_atomic_add 到同一 GM 地址 numHitsGm[row]: warm 场景 ~1024 次/行, 同地址
//   atomic 串行 ~9K 周期/次 → OpHitMiss 实测 ~9.4M 周期; cold 场景 0 hit 无 atomic
//   仅 ~10K (耗时与 hit 数几乎无关、与 atomic 次数强相关, 已证实根因)。
// [P4 2026-10-13] 原 OpFreeSize (独立 t0 步 + SyncGrid) 折叠进 t0 汇总段:
//   fs = nm - avail (clamp ≥0) 顺手写入 freeSizeGm。avail 此时恒为本轮初值
//   (写入点仅 Step0/0.5 reset 与 Step5 尾折叠更新), 经 Step1 尾 SyncGrid 的
//   dcci 后对 Step3 全员读者可见。计数语义不变: nm 直接数 effTopk 内
//   curSlots==I32_MAX 的个数, 与原公式 effTopk - numHits 恒等。
//   numHitsGm 不再读写 (槽位保留防 workspace 布局漂移)。
// [C1/C2 2026-10-13] 栅栏 2 次/行 → 末行省 1 次 (复用保护); t0 另在 warpScratch[64]
//   写淘汰标志, 供 Process 整段跳过 fs 全 0 的淘汰流水线。
__aicore__ inline void OpHitMiss(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t maxModelLenP1,
    uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* htdGm, int32_t* posGm, int32_t* curSlotsGm,
    int32_t* numMissesGm, int32_t* availGm, int32_t* freeSizeGm,
    int32_t* warpScratch) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    uint32_t t = threadIdx.x;
    uint32_t lane = t & 31u;
    uint32_t warpCount = blockDim.x >> 5;
    int32_t fsAny = 0;   // [C1] t0 跨行累计: 本核任一行是否需要淘汰 (仅 t0 写)
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        uint32_t rowOff = row * topkDim;
        uint32_t htdRowOff = row * maxModelLenP1;
        uint32_t localMiss = 0;
        for (uint32_t j = t; j < topkDim; j += blockDim.x) {
            if (j >= effTopk) { curSlotsGm[rowOff + j] = 0; continue; }
            int32_t tok = posGm[rowOff + j];
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            int32_t slot = htdGm[htdRowOff + tokSafe];
            curSlotsGm[rowOff + j] = slot;
            if (slot == I32_MAX) localMiss++;
        }
        // [Scan-v2] warp 归约 (无栅栏) → lane0 写 UB (行奇偶双缓冲) → 1 栅栏 →
        //   t0 本地 Σ 汇总写 numMisses + fs。每行仅 1 次栅栏 (原第 2 栅栏的
        //   warpScratch 复用保护由双缓冲承担: 行 r+2 复用 buf(r&1), 中间隔行
        //   r+1 的 B1, 读先于写)。t0 的 GM 写可见性由 Step1 尾 SyncGrid 兜底。
        int32_t warpSum = asc_reduce_add((int32_t)localMiss);
        if (lane == 0u) warpScratch[(r & 1u) * 32u + (t >> 5)] = warpSum;
        asc_syncthreads();
        if (t == 0u) {
            int32_t nm = 0;
            for (uint32_t w = 0; w < warpCount; ++w) nm += warpScratch[(r & 1u) * 32u + w];
            numMissesGm[row] = nm;
            int32_t fs = nm - availGm[row];
            if (fs < 0) fs = 0;
            freeSizeGm[row] = fs;
            if (fs != 0) fsAny = 1;
        }
    }
    // [C1] 淘汰流水线整段跳过标志: 全行 fs==0 时 Argtopk/FreeRelease/FreePush 对
    //   状态零影响 (freeIdx 无读者 —— fs=0 时 OpAllocPop 恒走老栈路径), 见 Process。
    //   槽位 64 与双缓冲 [0,64) 不冲突; 经 Step1 尾 SyncGrid 对全员可见。
    if (t == 0u) warpScratch[64] = fsAny;
}

// ---- Step 2: protect + fifo++ ----
__aicore__ inline void OpProtect(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t B,
    uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* curSlotsGm, int32_t* priGm, int32_t* fifoGm) {
    uint32_t priCols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t fifo = fifoGm[row];
        uint32_t rowOff = row * topkDim;
        uint32_t priRowOff = row * priCols;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if (j >= effTopk) continue;
            int32_t slot = curSlotsGm[rowOff + j];
            if (slot != I32_MAX) {
                int32_t s = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
                priGm[priRowOff + s] = fifo;
            }
        }
    }
}

__aicore__ inline void OpFifoInc(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows, int32_t* fifoGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            fifoGm[row] = fifoGm[row] + 1;
        }
    }
}

// Step5 尾: fifo++ 与 avail 折叠更新 (替代原 OpFifoInc + OpFreeAvailAdd + OpAllocAvailSub
//   两个独立 t0 步及其前后 SyncGrid)。
// [P4 2026-10-13] avail 终值 = availOld + fs - nm, 与旧版两步序列 (+fs 后 -nm) 恒等,
//   且恒 ≥ 0: fs = max(nm - availOld, 0) → nm > availOld 时终值 0, 否则 availOld - nm ≥ 0。
//   t0 读的 fifo/freeSize/numMisses 均为 t0 自己在 Step0~1 写入 (同线程 GM RAW 安全);
//   avail 的全员读者 (OpFreePush/OpAllocPop) 已被 Step4 尾 SyncGrid 与本步前 SyncBlock
//   排空, t0 写不与他们竞争; fifo 的全员读者 (OpAllocPriority) 同理。
__aicore__ inline void OpFifoAvailUpdate(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* fifoGm, int32_t* availGm, int32_t* freeSizeGm, int32_t* numMissesGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            fifoGm[row] = fifoGm[row] + 1;
            availGm[row] = availGm[row] + freeSizeGm[row] - numMissesGm[row];
        }
    }
}

// ---- Step 3: free (argtopk select + release + push) ----
// [P4 2026-10-13] OpFreeSize (独立 t0 步 + SyncGrid) 已折进 OpHitMiss 的 t0 汇总段。

// per-row 基数选择 (radix select, MSB-first 逐字节收敛, 全定长, 无递归, 无动态分配).
// 目标: 从打包 key 中选出 pri 最小的 freeSize 个槽 (平局按槽号升序), 写 freeIdxOut.
//   key[i] = (i==0 || i>B || pri<0) ? I32_MAX : pri  → 全非负, 无效/填充沉底.
// [B 2026-10-15] 栅栏与工作量重构 (重构前 full_cache 实测 ~410K 周期 → 199K):
//   (1) 首趟单相位: pack(全员) 与 clear+hist(G 路, 直读 priGm) 无交叉 RAW;
//   (2) merge 改 256 桶并行 (每桶归并 G 个切片, 墙钟 G 次读);
//   (3) hist/pack/emission 只扫有效区 [0, B+1); P = Npad-(B+1) 个 padding 键恒为
//       I32_MAX, 在 merge 相位解析式计入 padByte 桶 (padIn = 选择前缀走在 I32_MAX
//       路径上); emission 越过有效区时按槽号升序补 padding 索引, 逐位一致。
// [方向1 2026-10-15] 自适应跳趟: pri ≤ fifo 恒成立 (t0 写 pri 的唯一来源是 fifo),
//   fifo 的前导零字节数 z ∈ {0,1,2,3} → 跳过 pass 0..z-1, 从 byte(3-z) 起跑
//   (快路径前缀 prefix=0: 高字节全 0 的键; I32_MAX 键 byte3=0x7F 被候选过滤天然
//   排除)。健壮性不依赖启发式成立: 若选择在 prefix=0 候选内放不下 (b 扫到 256,
//   即 fsClamp 超出小键总数), 回退 pass 0 完整重跑一次 —— 任何输入下输出与
//   全趟版逐位一致。基准 fifo=4097 → 跳 2 趟。
// [方向2 2026-10-15] 决策本地化: 桶决策是 globalCnt 的确定性函数, 全员各自重算
//   b/prefix/remaining (本数据 break 在前几个桶, 每线程 ≤4 次 UB 读), 取代 t0
//   串行决策 + scalar 广播 → 每趟 3→2 栅栏 ([clear+hist]→B1→[merge]→B2→[本地决策]
//   →下一趟)。scalar 不再有读者 (emission 直接用寄存器 T/quota), 栅栏间无任何
//   跨线程写。快路径+全趟合计: 基准只跑 2 趟 × 2 栅栏 + emission 扫描 2 + 尾 2
//   = 8 次栅栏 (B 前为 16)。
// 不变量 (计数含 padding): |{key < T}| + quota == fsClamp; remaining' ≥ 1 恒成立
//   (b-loop 在 cum+c ≥ remaining 时 break, cum < remaining)。
// 输出: 先全部 key<T (槽号升序), 再前 quota 个 key==T (槽号升序, 有效区优先,
//   越界部分补 padding 索引), 尾部补 0 (下游按 freeSize 裁剪)。key==T 平局取
//   最小槽号, 与稳定排序 (pri, slot) 字典序一致。
__aicore__ inline void OpArgtopkRow(
    uint32_t B, uint32_t Npad, uint32_t topkDim, uint32_t G, uint32_t fifoVal,
    int32_t* priGm, uint32_t row, int32_t* freeIdxOut, int32_t freeSize,
    bool freeIdxIsGm,
    int32_t* key,
    int32_t* hist,
    int32_t* globalCnt,
    int32_t* segLocal,
    int32_t* segBase,
    int32_t* scalar,
    uint32_t* dbgCount,
    int32_t* dbgKey) {

    uint32_t t = threadIdx.x;
    uint32_t rowOff = row * (B + 1);
    uint32_t fsClamp = freeSize < 0 ? 0u : (freeSize > (int32_t)topkDim ? topkDim : (uint32_t)freeSize);

    if (fsClamp == 0u) {
        asc_syncthreads();
        return;
    }

    bool active = (t < G);
    const uint32_t validN = B + 1u;         // 有效区 [0, B], 含 slot 0
    const uint32_t padN = Npad - validN;    // padding 区 [B+1, Npad), 键恒为 I32_MAX

    // [方向1] fifo 前导零字节数 z → 起跑趟 startPass = z (跳过 pass 0..z-1, 至少跑 byte0)。
    //   fifo=4097 (0x1001): byte3/byte2 为 0 → z=2。fifo=0 → 全部 pri 无效, z=3
    //   (候选空 → 必然回退, 仅浪费一趟, 结果仍正确)。
    uint32_t startPass = 3u;
    if ((fifoVal >> 24) != 0u) startPass = 0u;
    else if ((fifoVal >> 16) != 0u) startPass = 1u;
    else if ((fifoVal >> 8) != 0u) startPass = 2u;

    uint32_t T = 0u, quota = 0u;            // 选择结果 (emission 用; 全员寄存器)
    {
        uint32_t pass = startPass;
        uint32_t remaining = fsClamp;
        uint32_t prefix = 0u;
        bool firstIter = true;              // 首迭代: pack + 直读 priGm
        // 外层 attempt: 0 = 快路径 (startPass 起跑), 1 = [方向1] 回退全趟。
        //   ⚠ 不用 while(true)+continue: 数据依赖 continue 跨越栅栏结构,
        //   bisheng 前端收敛分析会段错误; 有界 for + 迭代边界 break 与旧版
        //   scalar[3] break 同构, 已验证可编译。
        for (uint32_t attempt = 0; attempt < 2u; ++attempt) {
            for (; pass < 4u; ++pass) {
                uint32_t shift = (3u - pass) * 8u;
                bool const first = firstIter;
                firstIter = false;
                // ---- (a) 首迭代: [P-D] pack 融合 + clear(仅首次, 覆盖 UB 初始垃圾)
                //      + hist(G 路, 直读 priGm, 顺手写 key —— stride=G 循环覆盖全部
                //      [0,validN), 独立的全员 pack 循环已删除); 后续趟: hist(G 路,
                //      读 key; 不 clear —— [P-B] 上一趟 merge 已破坏性清零)。
                //      过滤用 if/else 分支而不是 pass==0 短路 —— pass 0 时
                //      (v >> 32) 是文本可见 UB, SIMT 后端可能投机求值。----
                if (active) {
                    int32_t* myHist = hist + t * 256u;
                    if (first) {
                        for (uint32_t b = 0; b < 256; ++b) myHist[b] = 0;
                        // 首趟: pass==startPass。pass==0 无过滤; startPass>0 时按
                        //   prefix=0 过滤被跳过的高字节 (I32_MAX 键 byte3=0x7F 被排除)。
                        for (uint32_t i = t; i < validN; i += G) {
                            int32_t p = priGm[rowOff + i];
                            uint32_t v = (uint32_t)((p < 0 || i == 0) ? I32_MAX : p);
                            key[i] = (int32_t)v;
                            if (pass == 0u) {
                                myHist[(v >> shift) & 0xFFu] += 1;
                            } else if ((v >> (shift + 8u)) == 0u) {
                                myHist[(v >> shift) & 0xFFu] += 1;
                            }
                        }
                    } else {
                        for (uint32_t i = t; i < validN; i += G) {
                            uint32_t v = (uint32_t)key[i];
                            if (pass == 0u) {
                                myHist[(v >> shift) & 0xFFu] += 1;   // 回退重跑的 pass 0
                            } else if ((v >> (shift + 8u)) == (prefix >> (shift + 8u))) {
                                myHist[(v >> shift) & 0xFFu] += 1;
                            }
                        }
                    }
                }
                asc_syncthreads();          // B1: hist 切片写 → merge 读
                // ---- (b) merge: 256 桶并行 (每桶归并 G 个切片); padding 解析式计入。
                //      [P-B] 破坏性归并: 读后清零, 下一趟 hist 无需 clear (首趟除外,
                //      见 (a)); 清零写与下一趟 hist 增量之间有 B2 隔开。----
                {
                    uint32_t padByte = (0x7FFFFFFFu >> shift) & 0xFFu;
                    bool padIn;
                    if (pass == 0u) {
                        padIn = true;       // pass 0 无前缀; (x>>32) 文本 UB, 分支规避
                    } else {
                        padIn = ((0x7FFFFFFFu >> (shift + 8u)) == (prefix >> (shift + 8u)));
                    }
                    for (uint32_t b = t; b < 256; b += blockDim.x) {
                        int32_t s = 0;
                        for (uint32_t th = 0; th < G; ++th) {
                            int32_t* slot = &hist[th * 256u + b];
                            s += *slot;
                            *slot = 0;
                        }
                        if (padIn && b == padByte) s += (int32_t)padN;
                        globalCnt[b] = s;
                    }
                }
                asc_syncthreads();          // B2: merge 写 globalCnt → 决策读
                // ---- (c) dbg dump (t0, 首迭代) + 全员本地决策 (无 t0 串行、无 scalar
                //      广播、无第 3 栅栏; 决策是 globalCnt 的确定性函数 → 各线程结果
                //      一致, 下一趟 (a) 只写自有切片/读稳定 key, 与本段读无竞争;
                //      globalCnt 复用由下一趟 B1 保护)。----
                if (t == 0 && first) {
                    if (dbgKey != nullptr) {
                        for (uint32_t i = 0; i < validN; ++i) dbgKey[i] = key[i];
                    }
                    // startPass>0 时为首趟 (带 prefix=0 过滤) 直方图, 语义随起跑趟变化,
                    //   仅供诊断; 全趟路径 (startPass==0) 与旧版 pass0 全量语义一致。
                    if (dbgCount != nullptr) {
                        for (uint32_t b = 0; b < 256; ++b) dbgCount[b] = (uint32_t)globalCnt[b];
                    }
                }
                {
                    uint32_t cum = 0u, b = 0u;
                    for (; b < 256u; ++b) {
                        uint32_t c = (uint32_t)globalCnt[b];
                        if (cum + c >= remaining) break;
                        cum += c;
                    }
                    if (b >= 256u) break;   // 快路径放不下 → 外层 attempt 回退 (见下)
                    prefix |= (b << shift);
                    remaining -= cum;
                    T = prefix;             // remaining' ≥ 1 恒成立 (b-loop break 语义)
                    quota = remaining;
                }
            }
            if (pass >= 4u) break;          // pass 3 正常跑完, 选择完成
            // [方向1] 快路径放不下 (剩余配额超出 prefix=0 候选总数): 回退 pass 0
            //   完整重跑一次。key 已打包 (firstIter 已为 false, 不重 pack); 全趟
            //   候选 ≥ fsClamp (fsClamp ≤ topkDim ≤ Npad = 全部键数), 必然收敛。
            pass = 0u;
            remaining = fsClamp;
            prefix = 0u;
        }
    }

    // [P1+B] emission 并行化 + 只扫有效区: 全员连续切片统计 (nLt | nEq<<16) 打包块
    //   扫描得各线程写入基址后并行写 freeIdxOut 与补零。
    //   ⚠ 线程切片必须连续 (lo = t*chunk) 而非 grid-stride: 保证跨线程输出仍按
    //   全局槽号升序 —— 先全部 key<T (槽号升序), 再前 quota 个 key==T (槽号升序),
    //   与原语义逐位一致 (grid-stride 切片会得到"线程主序", 顺序就错了)。
    //   选择越过有效区 (T == I32_MAX 且配额耗尽有效 ==T 键) 时, 剩余 m 个按槽号
    //   升序补 padding 索引 [B+1, B+1+m) —— padding 键恒为 I32_MAX 且位于全部有效
    //   索引之后, 与旧满扫描的 ==T 组输出顺序逐位一致。
    {
        int32_t* warpTotals = hist;        // hist 在各趟选择后已死, 复用作扫描 scratch
        uint32_t chunk = (validN + blockDim.x - 1u) / blockDim.x;
        uint32_t lo = t * chunk;
        uint32_t hi = (lo >= validN || lo + chunk > validN) ? validN : (lo + chunk);
        uint32_t nLt = 0u, nEq = 0u;
        for (uint32_t i = lo; i < hi; ++i) {
            uint32_t v = (uint32_t)key[i];
            if (v < T) ++nLt;
            else if (v == T) ++nEq;
        }
        // 字段上界: nLt/nEq 块内总和 ≤ validN < 2^16, 16bit 字段无跨字段进位。
        int32_t excl = PackedBlockScan((int32_t)(nLt | (nEq << 16)), warpTotals);
        uint32_t wLt = (uint32_t)excl & 0xFFFFu;            // 本线程之前所有切片的 key<T 总数
        uint32_t wEq = (uint32_t)excl >> 16;                // 本线程之前所有切片的 key==T 总数
        // [Scan-v2] grand total 本地 Σ warpTotals (行间有趟栅栏, 单缓冲即安全)。
        uint32_t warpCountE = blockDim.x >> 5;
        int32_t grandPacked = 0;
        for (uint32_t w = 0; w < warpCountE; ++w) grandPacked += warpTotals[w];
        uint32_t totalLt = (uint32_t)grandPacked & 0xFFFFu;  // 有效区 key<T 总数
        uint32_t totalEq = (uint32_t)grandPacked >> 16;      // 有效区 key==T 总数
        for (uint32_t i = lo; i < hi; ++i) {
            uint32_t v = (uint32_t)key[i];
            if (v < T) {
                freeIdxOut[row * topkDim + wLt] = (int32_t)i;
                ++wLt;
            } else if (v == T && wEq < quota) {
                freeIdxOut[row * topkDim + totalLt + wEq] = (int32_t)i;
                ++wEq;
            }
        }
        // padding 补齐: m = fsClamp - totalLt - min(quota, totalEq) (不变量保证 ≥0 且 ≤ padN)。
        uint32_t validEq = (quota < totalEq) ? quota : totalEq;
        uint32_t used = totalLt + validEq;
        uint32_t m = (fsClamp > used) ? (fsClamp - used) : 0u;
        for (uint32_t k = t; k < m; k += blockDim.x) {
            freeIdxOut[row * topkDim + used + k] = (int32_t)(validN + k);
        }
        // 尾部补零 [fsClamp, topkDim) (选择不变量: totalLt + quota == fsClamp, 计数含 padding)。
        for (uint32_t i = fsClamp + t; i < topkDim; i += blockDim.x) {
            freeIdxOut[row * topkDim + i] = 0;
        }
        asc_syncthreads();
        // [P-C] freeIdx 在 UB 时块内 syncthreads 即可见, 免 dcci; GM 路径由函数尾
        //   栅栏保证 dcci 完成后才返回。
        if (freeIdxIsGm && t == 0) asc_dcci_entire((void*)freeIdxOut);
    }
    asc_syncthreads();
    (void)segLocal;  // 预留分段 scatter 缓冲, 基数选择路径未使用
    (void)segBase;   // 同上
    (void)scalar;    // [方向2] 决策本地化后无读者, 槽位保留防 UB 布局漂移
}



__aicore__ inline void OpFreeRelease(
    uint32_t R, uint32_t topkDim, uint32_t B, uint32_t maxModelLenP1,
    uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* freeIdxGm, int32_t* freeSizeGm,
    int32_t* priGm, int32_t* dthGm, int32_t* htdGm) {
    uint32_t priCols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t fs = freeSizeGm[row];
        uint32_t rowOff = row * topkDim;
        uint32_t priRowOff = row * priCols;
        uint32_t htdRowOff = row * maxModelLenP1;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if ((int32_t)j >= fs) continue;
            int32_t slot = freeIdxGm[rowOff + j];
            int32_t s = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
            uint32_t off = priRowOff + s;
            int32_t tok = dthGm[off];
            priGm[off] = -1;
            dthGm[off] = I32_MAX;
            if (tok != I32_MAX) {
                int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
                htdGm[htdRowOff + tokSafe] = I32_MAX;
            }
        }
    }
}

__aicore__ inline void OpFreePush(
    uint32_t R, uint32_t topkDim, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* freeIdxGm, int32_t* freeSizeGm,
    int32_t* availGm, int32_t* freeGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t fs = freeSizeGm[row];
        int32_t avail = availGm[row];
        uint32_t rowOff = row * topkDim;
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if ((int32_t)j >= fs) continue;
            int32_t writeIdx = avail + (int32_t)j;
            if (writeIdx < 0) writeIdx = 0;
            if (writeIdx >= (int32_t)freeStackSize) writeIdx = (int32_t)freeStackSize - 1;
            freeGm[freeRowOff + writeIdx] = freeIdxGm[rowOff + j];
        }
    }
}

// ---- Step 4: alloc pop ----
// [P4 2026-10-13] 直读 freeIdx: 旧版 avail 先 +=fs 再 -=nm 后从 free 栈弹出
//   free[availOld+fs-nm+j]; 而新压入段满足 free[availOld+k] == freeIdx[k] (k<fs),
//   故弹出等价拆分为: readIdx ≥ availOld → freeIdx[readIdx-availOld] (不依赖
//   OpFreePush 对 free 栈新段的写入), readIdx < availOld → 老栈 free[readIdx]
//   (本 kernel 从未写该区段) —— 与旧版逐值一致, 且 FreePush 的写入在本 kernel
//   内不再有读者 (仅持久化给下一次调用), FreePush→AllocPop 间的 SyncGrid 可删。
//   avail 的两步更新 (原 OpFreeAvailAdd +=fs / OpAllocAvailSub -=nm) 合并为
//   Step5 尾 OpFifoAvailUpdate 的一次写; 本函数读到的 avail 恒为初值 availOld。
__aicore__ inline void OpAllocPop(
    uint32_t R, uint32_t topkDim, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* numMissesGm, int32_t* freeSizeGm, int32_t* availGm,
    int32_t* freeIdxGm, int32_t* freeGm, int32_t* rawSlotsGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t nm = numMissesGm[row];
        int32_t avail = availGm[row];      // 初值 availOld (更新推迟到 Step5 尾)
        int32_t fs = freeSizeGm[row];
        uint32_t rowOff = row * topkDim;
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            int32_t slot;
            if (j < (uint32_t)nm) {
                int32_t readIdx = avail + fs - (int32_t)nm + (int32_t)j;
                if (readIdx >= avail) {
                    int32_t fi = readIdx - avail;                 // ∈ [0, fs) ⊂ [0, topkDim)
                    if (fi < 0) fi = 0;
                    if (fi >= (int32_t)topkDim) fi = (int32_t)topkDim - 1;
                    slot = freeIdxGm[rowOff + fi];
                } else {
                    if (readIdx < 0) readIdx = 0;                 // 老栈区段, 本 kernel 未写
                    if (readIdx >= (int32_t)freeStackSize) readIdx = (int32_t)freeStackSize - 1;
                    slot = freeGm[freeRowOff + readIdx];
                }
            } else {
                slot = 0;
            }
            rawSlotsGm[rowOff + j] = slot;
        }
    }
}

// ---- Step 5: alloc priority ----
__aicore__ inline void OpAllocPriority(
    uint32_t R, uint32_t topkDim, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* rawSlotsGm, int32_t* numMissesGm,
    int32_t* priGm, int32_t* fifoGm) {
    uint32_t priCols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t nm = numMissesGm[row];
        int32_t fifo = fifoGm[row];
        uint32_t rowOff = row * topkDim;
        uint32_t priRowOff = row * priCols;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if (j >= (uint32_t)nm) continue;
            int32_t slot = rawSlotsGm[rowOff + j];
            int32_t s = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
            priGm[priRowOff + s] = fifo;
        }
    }
}

// ---- Step 6: set_update ----
// [P3 2026-10-13] miss 前缀和改两级 warp 扫描 (对照基线 BlockExclusiveScan), 替代
//   Hillis-Steele: 原 11 轮 × 1024 线程块栅栏 (~20K 周期, cold/warm 耗时相同 →
//   barrier bound 而非 work bound), 现 warp shuffle 一级 + t0 扫 warp total 二级,
//   3 次栅栏/行。每线程 2 元素打包进一个 int32: 低 12bit = p0 miss, 高 12bit 起
//   = p1 miss。p0 位 = 线程号, p1 位 = 线程号 + Bm, 均为连续映射 → 排名可由
//   打包前缀直接拼出: rank(p0=t) = E0(t); rank(p1=t+Bm) = TotalMiss0 + E1(t)。
//   字段打包安全由 "每 tile 计数 ≤ Bm ≤ 1024 < 2^12" 保证 (与 topkDim 无关);
//   扫描 scratch 固定 66 槽 (warpTotals/warpBases/ctrlA/ctrlB), 亦与 topkDim
//   无关 —— 旧版的 "2*topkDim/4*topkDim ≤ 8192" 门是 Hillis-Steele 时代
//   bufA/bufB 缓冲 (大小 2*topkDim) 的遗留约束, warp 扫描版已不需要, 已删
//   (否则 topk=4096 时 4 元素门被 UB 侧误拦, 掉 t0 串行)。
// 路径选择 (线程侧条件唯一): topkDim ≤ 2*blockDim → 2 元素; ≤ 4*blockDim →
//   4 元素 ([消融 2026-10-15] 512 线程 × 4 = 2048 生产尺寸, 见分支内注释);
//   否则回退 t0 串行累加 (plugin 自适应 blockDim 下, topk ≤ 4096 恒并行)。
// 语义与串行版逐字段一致: wsMissRank = miss 的 exclusive 前缀秩,
// missPos = miss ? pos : 0。
__aicore__ inline void OpMissCumsum(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* curSlotsGm, int32_t* posGm,
    int32_t* wsMissRankGm, int32_t* missPosGm,
    int32_t* scanUb) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    uint32_t t = threadIdx.x;
    const uint32_t Bm = blockDim.x;
    const uint32_t warpCount = Bm >> 5;   // [Scan-v2] 本地 Σ warpTotals 用

    if (topkDim <= 2u * Bm) {
        // [Scan-v2] warpTotals 双缓冲按行奇偶 (行 r+2 复用 buf(r&1), 中间隔行 r+1
        //   的 B1, 读先于写); total0 由各线程本地 Σ 得到, 无 control 槽。
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            uint32_t rowOff = row * topkDim;
            uint32_t p0 = t;
            uint32_t p1 = t + Bm;

            int32_t ind0 = 0;
            if (p0 < topkDim) {
                bool miss0 = (p0 < effTopk) && (curSlotsGm[rowOff + p0] == I32_MAX);
                missPosGm[rowOff + p0] = miss0 ? posGm[rowOff + p0] : 0;
                ind0 = miss0 ? 1 : 0;
            }
            int32_t ind1 = 0;
            if (p1 < topkDim) {
                bool miss1 = (p1 < effTopk) && (curSlotsGm[rowOff + p1] == I32_MAX);
                missPosGm[rowOff + p1] = miss1 ? posGm[rowOff + p1] : 0;
                ind1 = miss1 ? 1 : 0;
            }
            int32_t* buf = scanUb + (r & 1u) * 32u;
            int32_t excl = PackedBlockScan(ind0 | (ind1 << 12), buf);
            int32_t e0 = excl & 0xFFF;
            int32_t e1 = (excl >> 12) & 0xFFF;
            int32_t grandPacked = 0;
            for (uint32_t w = 0; w < warpCount; ++w) grandPacked += buf[w];
            int32_t total0 = grandPacked & 0xFFF;
            if (p0 < topkDim) {
                wsMissRankGm[rowOff + p0] = ind0 ? e0 : 0;
            }
            if (p1 < topkDim) {
                wsMissRankGm[rowOff + p1] = ind1 ? (total0 + e1) : 0;
            }
        }
        return;
    } else if (topkDim <= 4u * Bm) {
        // [消融 2026-10-15] 每线程 4 元素路径, 服务于 512 线程消融 (512×4=2048,
        //   topk=2048 时 2 元素并行门 512×2=1024 不够, P3 收益会归零)。每线程认领
        //   4 个连续映射位置 p0=t, p1=t+Bm, p2=t+2Bm, p3=t+3Bm; 位置序
        //   [0,Bm)=p0s, [Bm,2Bm)=p1s, [2Bm,3Bm)=p2s, [3Bm,4Bm)=p3s。
        //   两次链式扫描 (12bit 字段安全: 各 tile 的 miss 总数 ≤ topkDim ≤ 2048
        //   < 4096, 打包加法无跨字段进位): excl1 = scan(ind0 | ind1<<12) 覆盖
        //   [0,2Bm), excl2 = scan(ind2 | ind3<<12) 覆盖 [2Bm,4Bm)。
        //   [Scan-v2] 四缓冲按 (行奇偶 × scan 序) 轮转: 行 r 的 scan1 用
        //   scanUb+(r&1)*64, scan2 用 +32。背靠背安全: scan2 的写与 scan1 的读
        //   被 scan1 的 B1 隔开; 行 r+2 复用本行缓冲, 中间隔行 r+1 的两次 B1。
        //   t0/t01/t2 由各线程返回后本地 Σ 缓冲算出, 无 control 槽。
        //   rank 拼装 (p1 在 tile1 段内, 前缀 = tile0 总数 + 段内偏移; p2/p3 在
        //   后续 tile 段内, 前缀 = 前面所有 tile 的完整总数 + 段内偏移, 即
        //   p3 的前缀 = 前两个 tile 的总数 + 第三个 tile 的 p2 总数 + 段内偏移):
        //     rank(p0=t)     = e0
        //     rank(p1=t+Bm)  = t0 + e1
        //     rank(p2=t+2Bm) = t01 + e2
        //     rank(p3=t+3Bm) = t01 + t2 + e3
        //   其中 t0 = [0,Bm) 的 miss 总数 (buf1 低 12bit), t01 = [0,2Bm) 的 miss
        //   总数 (buf1 两字段和), t2 = [2Bm,3Bm) 的 miss 总数 (buf2 低 12bit)。
        //   ⚠ rank(p1) 不能写 t01+e1: p1 位于 tile1 段内而非段后, 那样会把
        //   tile1 自己的总数也计入 (多加 total1), rank 偏大。
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            uint32_t rowOff = row * topkDim;
            uint32_t p0 = t;
            uint32_t p1 = t + Bm;
            uint32_t p2 = t + 2u * Bm;
            uint32_t p3 = t + 3u * Bm;

            int32_t ind0 = 0;
            if (p0 < topkDim) {
                bool miss0 = (p0 < effTopk) && (curSlotsGm[rowOff + p0] == I32_MAX);
                missPosGm[rowOff + p0] = miss0 ? posGm[rowOff + p0] : 0;
                ind0 = miss0 ? 1 : 0;
            }
            int32_t ind1 = 0;
            if (p1 < topkDim) {
                bool miss1 = (p1 < effTopk) && (curSlotsGm[rowOff + p1] == I32_MAX);
                missPosGm[rowOff + p1] = miss1 ? posGm[rowOff + p1] : 0;
                ind1 = miss1 ? 1 : 0;
            }
            int32_t ind2 = 0;
            if (p2 < topkDim) {
                bool miss2 = (p2 < effTopk) && (curSlotsGm[rowOff + p2] == I32_MAX);
                missPosGm[rowOff + p2] = miss2 ? posGm[rowOff + p2] : 0;
                ind2 = miss2 ? 1 : 0;
            }
            int32_t ind3 = 0;
            if (p3 < topkDim) {
                bool miss3 = (p3 < effTopk) && (curSlotsGm[rowOff + p3] == I32_MAX);
                missPosGm[rowOff + p3] = miss3 ? posGm[rowOff + p3] : 0;
                ind3 = miss3 ? 1 : 0;
            }
            // 两次链式扫描: scan1 打包 [0,2Bm) 的 miss 标志, scan2 打包 [2Bm,4Bm)。
            int32_t* buf1 = scanUb + (r & 1u) * 64u;
            int32_t* buf2 = buf1 + 32;
            int32_t excl1 = PackedBlockScan(ind0 | (ind1 << 12), buf1);
            int32_t excl2 = PackedBlockScan(ind2 | (ind3 << 12), buf2);
            uint32_t e0 = (uint32_t)excl1 & 0xFFFu,  e1 = (uint32_t)excl1 >> 12;
            uint32_t e2 = (uint32_t)excl2 & 0xFFFu,  e3 = (uint32_t)excl2 >> 12;
            int32_t grand1 = 0, grand2 = 0;
            for (uint32_t w = 0; w < warpCount; ++w) { grand1 += buf1[w]; grand2 += buf2[w]; }
            uint32_t t0  = (uint32_t)grand1 & 0xFFFu;                                    // [0,Bm) 的 miss 总数
            uint32_t t01 = t0 + ((uint32_t)grand1 >> 12);                                // [0,2Bm) 的 miss 总数
            uint32_t t2  = (uint32_t)grand2 & 0xFFFu;                                    // [2Bm,3Bm) 的 miss 总数
            // rank(p0)=e0;  rank(p1)=t0+e1;  rank(p2)=t01+e2;  rank(p3)=t01+t2+e3
            if (p0 < topkDim) {
                wsMissRankGm[rowOff + p0] = ind0 ? (int32_t)e0 : 0;
            }
            if (p1 < topkDim) {
                wsMissRankGm[rowOff + p1] = ind1 ? (int32_t)(t0 + e1) : 0;
            }
            if (p2 < topkDim) {
                wsMissRankGm[rowOff + p2] = ind2 ? (int32_t)(t01 + e2) : 0;
            }
            if (p3 < topkDim) {
                wsMissRankGm[rowOff + p3] = ind3 ? (int32_t)(t01 + t2 + e3) : 0;
            }
        }
        return;
    }

    if (t == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            uint32_t rowOff = row * topkDim;
            int32_t rank = 0;
            for (uint32_t j = 0; j < topkDim; ++j) {
                bool miss = (j < effTopk) && (curSlotsGm[rowOff + j] == I32_MAX);
                wsMissRankGm[rowOff + j] = miss ? rank : 0;
                missPosGm[rowOff + j] = miss ? posGm[rowOff + j] : 0;
                if (miss) rank += 1;
            }
        }
    }
}

__aicore__ inline void OpSetUpdate(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t B, uint32_t maxModelLenP1,
    uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* curSlotsGm, int32_t* wsMissRankGm,
    int32_t* missHostPosGm, int32_t* missAllocFlatGm, uint8_t* missMaskGm,
    int32_t* rawSlotsGm, int32_t* htdGm, int32_t* dthGm) {
    uint32_t priCols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    // 输入契约: 同 row 内 pos (即 missHostPos) 不重复. 稀疏大模型 KV offload 场景下
    // 一行=一个请求, topk 选出的 token id 必不重复; 不同 row 对应独立 buffer, 行间重复无害.
    // 故 miss 时各 j 写的 htd[row, mpos_j] / dth[row, aslot_j] 地址行内互不冲突 → 可并行.
    // (若违反契约喂入行内重复 pos, 并发写同地址将不确定 —— gen_data.py 保证不发生.)
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        uint32_t rowOff = row * topkDim;
        uint32_t rawRowOff = row * topkDim;   // rawSlots row stride = topkDim (kEff 已删)
        uint32_t htdRowOff = row * maxModelLenP1;
        uint32_t dthRowOff = row * priCols;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if (j >= effTopk) { missAllocFlatGm[rowOff + j] = 0; missMaskGm[rowOff + j] = 0; continue; }
            int32_t slot = curSlotsGm[rowOff + j];
            if (slot == I32_MAX) {
                int32_t rank = wsMissRankGm[rowOff + j];
                // rk=rank 不再 clamp 到 kEff-1. rank < numMisses <= topkDim, rawRowOff+rk < R*topkDim 安全.
                int32_t rk = rank < 0 ? 0 : rank;
                int32_t aslot = rawSlotsGm[rawRowOff + rk];
                int32_t mpos = missHostPosGm[rowOff + j];
                int32_t mposSafe = mpos < 0 ? 0 : (mpos > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : mpos);
                htdGm[htdRowOff + mposSafe] = aslot;
                int32_t s = aslot < 0 ? 0 : (aslot > (int32_t)B ? (int32_t)B : aslot);
                dthGm[dthRowOff + s] = mpos;
                missAllocFlatGm[rowOff + j] = (int32_t)row * (int32_t)B + aslot;  // row*B+aslot, 32 位足够
                missMaskGm[rowOff + j] = 1;
                // [P2 2026-10-13] miss 的最终槽位就是本轮分配的 aslot, 此处顺手写回 curSlots。
                //   原 OpFinalSlots 对全部 topk 重新 gather htd 属冗余 (hit 位 curSlots 在
                //   Step1 已正确), 该步已删除; 行内 pos 不重复契约下 htd[pos_j] 恰为 aslot_j。
                curSlotsGm[rowOff + j] = aslot;
            } else {
                missAllocFlatGm[rowOff + j] = 0;
                missMaskGm[rowOff + j] = 0;
            }
        }
    }
}

// ---- Step 7: final slots ----
// [P2 2026-10-13] 已并入 OpSetUpdate (miss 位直接写回 aslot, hit 位 Step1 已正确),
//   独立的 OpFinalSlots (全量 topk 重新 gather htd) 已删除。

__aicore__ inline void OpFillZero(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows, int32_t* outGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            outGm[row] = 0;
        }
    }
}

// ===========================================================================
// EchoLruKernel: 纯 SIMT 主体 (Path A). 全部 block 跑同一 Process(), 每核只处理
// 自己的连续行段. 步间 SyncGrid 用块内 asc_syncthreads + asc_dcci_entire 解决行内跨线程 RAW
// (不同 flat 维度映射到同一地址) 与单核内写后读 stale. 行间零依赖 → 无需跨核栅栏.
//
// 跨核 grid.sync 攻关已暂停 (2026-09-16): probe_grid_sync/ 极简验证 grid.sync 经任意包装全
//   PASS, 但 echo_lru Process 内哪怕一个 grid.sync 即 trap (根因为编译单元整体属性, 不可复现).
//   详见 GRID_GROUP_PITFALLS.md "未解 trap" 段. Stage 2 grid-stride 跨核协同待新线索.
// ===========================================================================
class EchoLruKernel {
public:
    __aicore__ inline EchoLruKernel() {}

    __aicore__ inline void Init(int32_t* htd, int32_t* dth, int32_t* pri, int32_t* free,
                     int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                     uint8_t* resetMask, int32_t* curSlots,
                     int32_t* missHostPos, int32_t* missAllocFlat, uint8_t* missMask,
                     void* workspace, const EchoLruTiling& tiling,
                     int32_t* argtopkUb) {
        tiling_ = tiling;
        htdGm_ = htd;
        dthGm_ = dth;
        priGm_ = pri;
        freeGm_ = free;
        availGm_ = avail;
        fifoGm_ = fifo;
        posGm_ = pos;
        splGm_ = spl;
        resetMaskGm_ = resetMask;
        curSlotsGm_ = curSlots;
        missHostPosGm_   = missHostPos;
        missAllocFlatGm_ = missAllocFlat;
        missMaskGm_      = missMask;
        wsBase_ = workspace;

        int32_t* ws = (int32_t*)workspace;
        uint32_t R = tiling_.numTokens;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t o = 0;
        wsCumsumGm_   = ws + o; o += R * (B + 1);
        pushCountGm_  = ws + o; o += R;
        numHitsGm_    = ws + o; o += R;   // [P0] 已无读写, 槽位保留防 workspace 布局漂移
        numMissesGm_  = ws + o; o += R;
        freeIdxGm_    = ws + o; o += R * topkDim;   // row stride = topkDim (kEff 已删)
        freeSizeGm_   = ws + o; o += R;
        rawSlotsGm_   = ws + o; o += R * topkDim;   // row stride = topkDim
        wsMissRankGm_ = ws + o; o += R * topkDim;

        uint32_t npad = NpadOf(B);
        argtopkNpad_ = npad;

        uint32_t G = ARGTOPK_GROUP_THREADS;
        argtopkGroupThreads_ = G;
        // [UB 布局] 单大 __ubuf__ 数组 argtopkUb 分段 (每核自己的 UB, 无 myBlock 偏移;
        //   probe_ub_size 实测多 __ubuf__ 数组会别名, 单大数组 32768 元素 PASS):
        //   [0, npad) = key 打包区; [npad, npad+G*256) = per-thread hist 切片;
        //   +256 = globalCnt; +G*2 = segLocal; +G*2 = segBase; +8 = scalar.
        //   峰值 npad+G*256+256+2*G*2+8 = 8192+16384+256+128+128+8 = 25096 < ARGTOPK_UB_ELEMS.
        //   ArgtopkAllRows 完成后整块 UB 即死, OpMissCumsum 复用 [0, 2*topkDim) 作 scan 缓冲.
        argtopkKeyUb_      = argtopkUb;
        uint32_t uo = npad;
        argtopkHistUb_     = argtopkUb + uo; uo += G * 256u;
        argtopkGlobalCntUb_ = argtopkUb + uo; uo += 256u;
        argtopkSegLocalUb_ = argtopkUb + uo; uo += G * 2u;
        argtopkSegBaseUb_  = argtopkUb + uo; uo += G * 2u;
        argtopkScalarUb_   = argtopkUb + uo; uo += 8u;
        // GM workspace 尾部 (8 字节对齐后): [诊断3] 每核 pass0 归并直方图 dump [bn*256]
        //   + 打包 srcKey dump [bn*ARGTOPK_GM_NPAD_MAX] + [ECHO_PERF_STEP] 每核分步周期
        //   [bn*ECHO_PERF_NUM_STEPS]. 与 host EchoWorkspaceElements 公式逐项一致.
        uint32_t bn = tiling_.blockNum;
        uint32_t elemOff = (o + 1u) & ~1u;  // 8 字节对齐 (2×int32)

        argtopkDebugCountGm_ = (uint32_t*)(ws + elemOff);  elemOff += bn * 256u;
        argtopkDebugKeyGm_   = ws + elemOff;               elemOff += bn * ARGTOPK_GM_NPAD_MAX;

        stepCyclesGm_ = nullptr;
#ifdef ECHO_PERF_STEP
        stepCyclesGm_ = ws + elemOff;
        elemOff += bn * ECHO_PERF_NUM_STEPS;
#endif
        perfStepIdx_ = 0u;

        // [P-C] freeIdx 迁 UB: 仅本 kernel 内读者 (FreeRelease/FreePush/AllocPop),
        //   且步骤按行批处理需容纳 rowsPerCore*topkDim; 放得下走 UB (省 GM 写 +
        //   emission 尾 dcci + 下游 GM 读), 放不下 (大 batch 形态) 回退 GM。
        freeIdxInUb_ = false;
        freeIdxPtr_ = freeIdxGm_;
        uint32_t fiElems = tiling_.rowsPerCore * topkDim;
        if (uo + fiElems <= ARGTOPK_UB_ELEMS) {
            freeIdxUb_ = argtopkUb + uo; uo += fiElems;
            freeIdxPtr_ = freeIdxUb_;
            freeIdxInUb_ = true;
        }
    }

    // Process: 各步顺序执行, 步间 SyncGrid() (块内 syncthreads + dcci) 解决行内跨线程 RAW.
    //   ECHO_PERF_STEP 构建下 PerfBegin/PerfEnd 把每步 t0 的 clock() 周期写入 workspace
    //   尾部 (见 tiling.h ECHO_PERF_NUM_STEPS); 普通构建为空操作, 不改变行为.
    __aicore__ inline void Process() {
        uint32_t R = tiling_.numTokens;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t maxModelLenP1 = tiling_.maxModelLenP1;
        uint32_t freeStackSize = tiling_.freeStackSize;
        uint32_t effTopk = tiling_.effTopk;
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;

        // [P0] numHitsGm 已无读者, 不再清零 (槽位保留防布局漂移); pushCount 供 spec 路径。
        OpFillZero(R, rowsPerCore, tailRows, pushCountGm_);
        SyncGrid();

        // ---- Step 0: reset ----
        PerfBegin(); OpResetRows(R, maxModelLenP1, rowsPerCore, tailRows, resetMaskGm_, htdGm_); PerfEnd();
        PerfBegin(); OpResetDthPri(R, B, rowsPerCore, tailRows, resetMaskGm_, dthGm_, priGm_); PerfEnd();
        PerfBegin(); OpResetFree(R, B, freeStackSize, rowsPerCore, tailRows, resetMaskGm_, freeGm_); PerfEnd();
        PerfBegin(); OpResetAvailFifo(R, B, rowsPerCore, tailRows, resetMaskGm_, availGm_, fifoGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();

        // ---- Step 0.5: invalidate speculative suffix (spec 路径; 非 spec 跳过) ----
        // specEnabled=0 (非 spec, 前缀全长 spl==maxModelLenP1) 时 Step0.5 整段 no-op, 跳过 4 个 Op 与 4 个栅栏.
        // specEnabled=1 时驱逐投机后缀 slot 并回收到本行 free 栈 (动态 free/alloc, 图捕获禁区).
        if (tiling_.specEnabled) {
            PerfBegin(); OpInvalidateSpecSuffix(R, B, maxModelLenP1, rowsPerCore, tailRows, splGm_, dthGm_, priGm_, htdGm_, wsCumsumGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
            PerfBegin(); OpSpecCumsum(R, B, rowsPerCore, tailRows, wsCumsumGm_, pushCountGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
            PerfBegin(); OpSpecPushFree(R, B, freeStackSize, rowsPerCore, tailRows, wsCumsumGm_, availGm_, freeGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
            PerfBegin(); OpSpecAvailAdd(R, rowsPerCore, tailRows, availGm_, pushCountGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
        }

        // ---- Step 1: hit/miss + miss 计数 + freeSize (warp 归约, 无 atomic) ----
        PerfBegin(); OpHitMiss(R, topkDim, effTopk, maxModelLenP1, rowsPerCore, tailRows,
                     htdGm_, posGm_, curSlotsGm_, numMissesGm_, availGm_, freeSizeGm_,
                     argtopkKeyUb_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();  // curSlots 行内跨线程 RAW → Step2 OpProtect;
                                            // numMisses/freeSize (t0 写) 经本 dcci 对 Step3/4 全员可见。

        // ---- Step 2: protect + fifo++ ----
        PerfBegin(); OpProtect(R, topkDim, effTopk, B, rowsPerCore, tailRows, curSlotsGm_, priGm_, fifoGm_); PerfEnd();
        PerfBegin(); asc_syncthreads(); PerfEnd();  // ⚠ Bug2 fix: 所有线程完成全部行 OpProtect 的 fifo 读+pri 写后,
                            //   t0 才能 OpFifoInc 做 fifo+=1. 否则 t0 串行 +=1 追上并超过其他线程
                            //   散在 OpProtect 各行的进度 → 某行后到的线程读到 +1 后的 fifo →
                            //   pri 多 1 (诊断4: row143 slot3706 kernel=12 golden=11 diff+1) →
                            //   argtopk pack srcKey 错 → freeIdx 顺序错 → 全链错. 排序方案不变.
        PerfBegin(); OpFifoInc(R, rowsPerCore, tailRows, fifoGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd(); // pri 行内写 → step3 argtopk 行内读 (行内跨线程 RAW)

        // ---- Step 3: free (无淘汰行时整段跳过) ----
        // [C1 2026-10-13] t0 在 Step1 写入"本核任一行 fs>0"标志 (UB 槽 64, 经 Step1
        //   尾 SyncGrid 可见)。全 0 时 Argtopk/FreeRelease/FreePush 对状态零影响:
        //   freeIdx 无任何读者 (fs=0 时 OpAllocPop 恒走老栈路径 readIdx<avail),
        //   FreeRelease/FreePush 的 j<fs 循环体不执行, avail += 0。四步保持打点
        //   (记录 ≈0 周期, 步数/步名不变), 条件对全块统一, 栅栏无发散。
        bool needFree = (argtopkKeyUb_[64] != 0);
        PerfBegin(); if (needFree) ArgtopkAllRows(); PerfEnd();   // radix select
        // [P-C] freeIdx RAW → OpFreeRelease: UB 路径普通栅栏即可 (免 dcci)。
        PerfBegin(); if (needFree) { if (freeIdxInUb_) asc_syncthreads(); else SyncGrid(); } PerfEnd();
        PerfBegin(); if (needFree) OpFreeRelease(R, topkDim, B, maxModelLenP1, rowsPerCore, tailRows, freeIdxPtr_, freeSizeGm_, priGm_, dthGm_, htdGm_); PerfEnd();
        PerfBegin(); if (needFree) OpFreePush(R, topkDim, freeStackSize, rowsPerCore, tailRows, freeIdxPtr_, freeSizeGm_, availGm_, freeGm_); PerfEnd();

        // ---- Step 4: alloc pop ----
        PerfBegin(); OpAllocPop(R, topkDim, freeStackSize, rowsPerCore, tailRows,
                     numMissesGm_, freeSizeGm_, availGm_, freeIdxPtr_, freeGm_, rawSlotsGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();  // rawSlots 跨线程 RAW → OpAllocPriority

        // ---- Step 5: alloc priority + fifo/avail 折叠更新 ----
        PerfBegin(); OpAllocPriority(R, topkDim, B, rowsPerCore, tailRows, rawSlotsGm_, numMissesGm_, priGm_, fifoGm_); PerfEnd();
        PerfBegin(); asc_syncthreads(); PerfEnd();  // ⚠ Bug2 fix (同 Step2): OpAllocPriority 全线程写 pri=fifo 完成后,
                            //   t0 才能 fifo+=1, 防 pri 多 1 race; 同时排空 avail 的全员读者。
        PerfBegin(); OpFifoAvailUpdate(R, rowsPerCore, tailRows, fifoGm_, availGm_, freeSizeGm_, numMissesGm_); PerfEnd();
        // [P4] 尾 SyncGrid 已删: pri/fifo/avail 此后无本 kernel 内读者。

        // ---- Step 6: set_update (含最终槽位回写, 原 OpFinalSlots 已并入) ----
        PerfBegin(); OpMissCumsum(R, topkDim, effTopk, rowsPerCore, tailRows,
                     curSlotsGm_, posGm_, wsMissRankGm_, missHostPosGm_, argtopkKeyUb_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();  // wsMissRank/missPos 跨线程 RAW → OpSetUpdate
        PerfBegin(); OpSetUpdate(R, topkDim, effTopk, B, maxModelLenP1, rowsPerCore, tailRows,
                    curSlotsGm_, wsMissRankGm_, missHostPosGm_, missAllocFlatGm_, missMaskGm_,
                    rawSlotsGm_, htdGm_, dthGm_); PerfEnd();
        // [P2] 之后无本 kernel 内读者, 不再需要尾步/尾栅栏 (kernel 结束即对 host 可见)。
    }

private:
    // 块内栅栏 + 刷本核 Data Cache. argtopk bitonic 等块内协作用此 (UB 是块内共享, 不需跨核).
    __aicore__ inline void SyncBlock() {
        asc_syncthreads();
        if (threadIdx.x == 0) asc_dcci_entire(wsBase_);
        asc_syncthreads();
    }

    // 步间块内栅栏 (Path A: 每核只处理自己连续行段, 行间零数据依赖, 无需跨核 sync).
    // asc_syncthreads 解决行内跨线程 RAW; asc_dcci_entire 刷本核 D-cache 防单核内写后读 stale.
    //
    // ⚠ 跨核 grid.sync 已知未解问题 (2026-09-16, 见 GRID_GROUP_PITFALLS.md "未解 trap"):
    //   probe_grid_sync/ 极简验证 grid.sync 经任意包装 (inline/template/类成员/dcci/ubuf/launch)
    //   全 PASS, 但 echo_lru Process 内哪怕一个 grid.sync (点0) 即 trap (输出全 0, 不 hang).
    //   根因是 echo_lru 编译单元整体属性 (Process 巨大/19 Op/Init 多 GM 指针), probe 无法复现.
    //   Stage 2 grid-stride 跨核协同因此暂停, 回退 Path A 块内栅栏 (4/4 pass 稳定).
    //   待后续 CANN 版本或新线索 (cooperative launch 前提 / __launch_bounds__ / 编译选项) 再攻关.
    __aicore__ inline void SyncGrid() {
        asc_syncthreads();
        if (threadIdx.x == 0) asc_dcci_entire(wsBase_);
        asc_syncthreads();
    }

    __aicore__ inline void PerfBegin() {
#ifdef ECHO_PERF_STEP
        if (threadIdx.x == 0) perfLast_ = clock();
#else
        (void)0;
#endif
    }

    __aicore__ inline void PerfEnd() {
#ifdef ECHO_PERF_STEP
        if (threadIdx.x == 0) {
            uint64_t now = clock();
            uint64_t delta = now - perfLast_;
            uint32_t d = (delta > 0x7FFFFFFFull) ? 0x7FFFFFFFu : (uint32_t)delta;
            if (stepCyclesGm_ != nullptr && perfStepIdx_ < ECHO_PERF_NUM_STEPS) {
                stepCyclesGm_[blockIdx.x * ECHO_PERF_NUM_STEPS + perfStepIdx_] = (int32_t)d;
            }
            ++perfStepIdx_;
        }
#else
        (void)0;
#endif
    }

    // [UB 布局 2026-10-13] key/hist/globalCnt/seg/scalar 全在顶层单大 __ubuf__ 数组分段
    //   (每核自己的 UB, 无 myBlock 偏移); 仅 [诊断3] debug dump 留 GM (每核独占段).
    //   行串行处理, 每核复用自己的 UB 段.
    __aicore__ inline void ArgtopkAllRows() {
        uint32_t myBlock = blockIdx.x;
        uint32_t blockNum = gridDim.x;
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;
        uint32_t startRow = myBlock * rowsPerCore;
        uint32_t rowCount = (myBlock == blockNum - 1) ? tailRows : rowsPerCore;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t Npad = argtopkNpad_;
        uint32_t G = argtopkGroupThreads_;
        uint32_t R = tiling_.numTokens;
        uint32_t* dbgCount = argtopkDebugCountGm_ + myBlock * 256u;
        int32_t*  dbgKey   = argtopkDebugKeyGm_   + myBlock * ARGTOPK_GM_NPAD_MAX;
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t freeSize = freeSizeGm_[row];   // 取前 freeSize 个 (不再固定 kEff)
            uint32_t fifoVal = (uint32_t)fifoGm_[row];   // [方向1] 跳趟启发式 (同地址 GM 广播读)
            bool dbg = (tiling_.argtopkDebugRow == row);
            OpArgtopkRow(B, Npad, topkDim, G, fifoVal, priGm_, row, freeIdxPtr_, freeSize,
                         !freeIdxInUb_,
                         argtopkKeyUb_, argtopkHistUb_, argtopkGlobalCntUb_,
                         argtopkSegLocalUb_, argtopkSegBaseUb_, argtopkScalarUb_,
                         dbg ? dbgCount : nullptr,
                         dbg ? dbgKey   : nullptr);
        }
    }

    static __aicore__ inline uint32_t NpadOf(uint32_t B) {
        uint32_t n = B + 1;
        uint32_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    EchoLruTiling tiling_;
    int32_t* htdGm_;
    int32_t* dthGm_;
    int32_t* priGm_;
    int32_t* freeGm_;
    int32_t* availGm_;
    int32_t* fifoGm_;
    int32_t* posGm_;
    int32_t* splGm_;
    uint8_t* resetMaskGm_;
    int32_t* curSlotsGm_;
    int32_t* wsCumsumGm_;
    int32_t* pushCountGm_;
    int32_t* numHitsGm_;
    int32_t* numMissesGm_;
    int32_t* freeIdxGm_;
    int32_t* freeSizeGm_;
    int32_t* rawSlotsGm_;
    int32_t* wsMissRankGm_;
    int32_t* missHostPosGm_;
    int32_t* missAllocFlatGm_;
    uint8_t* missMaskGm_;
    int32_t* argtopkKeyUb_;
    int32_t* argtopkHistUb_;
    int32_t* argtopkGlobalCntUb_; 
    int32_t* argtopkSegLocalUb_; 
    int32_t* argtopkSegBaseUb_; 
    int32_t* argtopkScalarUb_; 
    uint32_t* argtopkDebugCountGm_; // [诊断3] 每核 pass0 归并后直方图 dump [bn*256] uint32
    int32_t* argtopkDebugKeyGm_;    // [诊断3] 每核打包 srcKey dump [bn*ARGTOPK_NPAD_MAX] int32
    void* wsBase_;
    uint32_t argtopkNpad_;
    uint32_t argtopkGroupThreads_;
    int32_t* stepCyclesGm_;
    uint32_t perfStepIdx_;
    uint64_t perfLast_;
    int32_t* freeIdxUb_;    // [P-C] freeIdx 的 UB 落点 (放得下时启用)
    int32_t* freeIdxPtr_;   // freeIdx 实际指针: UB 或 GM
    bool freeIdxInUb_;
};

// 纯 SIMT 入口: __global__ (无 __vector__), 裸指针参数, host 四参 <<<blockNum, 512, 0, stream>>>.
// extern "C" 保持 C 链接, host main 通过 ascend_kernel::echo_lru_kernel 调用.
extern "C" __global__ __launch_bounds__(ECHO_LRU_THREAD_NUM) void echo_lru_kernel(
    int32_t* htd, int32_t* dth, int32_t* pri, int32_t* free,
    int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
    uint8_t* resetMask, int32_t* curSlots,
    int32_t* missHostPos, int32_t* missAllocFlat, uint8_t* missMask,
    void* workspace, EchoLruTiling* tilingBuf) {
    EchoLruTiling tiling;
    tiling.blockNum       = tilingBuf->blockNum;
    tiling.rowsPerCore    = tilingBuf->rowsPerCore;
    tiling.tailRows       = tilingBuf->tailRows;
    tiling.numTokens      = tilingBuf->numTokens;
    tiling.topkBufferB    = tilingBuf->topkBufferB;
    tiling.freeStackSize  = tilingBuf->freeStackSize;
    tiling.maxModelLenP1  = tilingBuf->maxModelLenP1;
    tiling.topkDim        = tilingBuf->topkDim;
    tiling.effTopk        = tilingBuf->effTopk;
    tiling.blockSize      = tilingBuf->blockSize;
    tiling.blockTableCols = tilingBuf->blockTableCols;
    tiling.kDim           = tilingBuf->kDim;
    tiling.vDim           = tilingBuf->vDim;
    tiling.kvHeadNum      = tilingBuf->kvHeadNum;
    tiling.specEnabled    = tilingBuf->specEnabled;
    tiling.argtopkDebugRow = tilingBuf->argtopkDebugRow;
    tiling.argtopkGroupThreads = tilingBuf->argtopkGroupThreads;
    // [UB 布局 2026-10-13] argtopk 全部状态放顶层单大 __ubuf__ 数组 (手动分段, 见 Init).
    //   probe_ub_size 实测: 单 __ubuf__ 数组可到 192KB (256KB 才 trap); 多个独立 __ubuf__ 数组会
    //   别名互相覆盖 (读 A 得 D). 故用单大数组 argtopkUb[ARGTOPK_UB_ELEMS=32768] 分段作
    //   key/hist/globalCnt/segLocal/segBase/scalar. debug dump 留 GM (每核独占段).
    //   每核 block 内共享自己的 UB (无需 myBlock 偏移). ARGTOPK_UB_ELEMS 见 tiling.h.
    __ubuf__ int32_t argtopkUb[ARGTOPK_UB_ELEMS];
    EchoLruKernel op;
    op.Init(htd, dth, pri, free, avail, fifo, pos, spl, resetMask, curSlots,
            missHostPos, missAllocFlat, missMask, workspace, tiling, argtopkUb);
    // Path A: 不使用 grid_group (跨核 grid.sync 在 echo_lru 内 trap, 见 SyncGrid 注释, 暂停攻关).
    //   行间零依赖, 步间块内 SyncGrid() (syncthreads+dcci) 即可.
    op.Process();
}

}  // namespace ascend_kernel
