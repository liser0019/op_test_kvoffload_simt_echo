// Adapted from peer_code/echo_lru_kernel.asc; computational stages retained.
// echo_lru 纯 SIMT 版本 (Path A: 按行分块) — LRU 元数据多步 kernel
//   (Step0 reset → Step5 set_update, protect/final-slots/计数/avail 子步均已被
//    折叠, 见各 [P*] 标记; 原 Step1
//    已分别并入 warp 归约与 OpSetUpdate, 见 [P0]/[P2] 标记)。
//
// [v2 维护序 2026-10-19] 淘汰路径的持久状态从 pri (slot→时间戳的"派生序",
//   淘汰时 O(B) 基数选择现算最旧 fs 个) 换成显式维护的顺序数组
//   lru[rows, B+1] (秩→槽位: lru[p] = 第 p 旧的占用槽, 有效区 [0, occ),
//   occ = B − avail, 尾部 [occ, B] 恒 0)。每步末重建:
//     新 lru = [幸存者(保持旧序)] ++ [本步 hit 槽按槽号升序] ++ [新分配槽按槽号升序]
//   与 v1 的 (pri, slot) 稳定排序逐位等价 → 全部输出与 v1 逐位一致。
//   淘汰 = 扫 lru 跳过本步 hit (v1 protect 语义: hit 槽 pri 已抬新不可淘汰)
//   取前 fs 个非 hit 槽 —— O(occ) 平扫取代 7 栅栏基数选择。
//   htd/dth/free_slots/avail/fifo 全部不变 (fifo 无 kernel 内读者, 仍按 +2/步
//   维护以保持 golden 语义); 查询路径 (htd 直查表) 原样保留 —— v2 = SIMT 的
//   维护序 + Echo 的直查表。
//
// 纯 SIMT 范式 (对照 examples/03_simt_api gather_2d / grid_config):
//   - 不 include kernel_operator.h (那是 SIMD 头树, 纯 SIMT 后端找不到 kernel_tpipe.h).
//   - kernel 参数与设备指针都用裸指针 (float*/int32_t*), 不用 GM_ADDR / __gm__.
//     __gm__ 是混合编程才需要的地址空间限定符, 纯 SIMT 默认指针即 GM.
//   - 设备函数 __aicore__ inline (被 __global__ 调用, 见 reduce_sum.asc/stack_overflow.asc); 入口 __global__ __launch_bounds__(N).
//   - host 四参 <<<blockNum, 512, 0, stream>>>; 子 Op 直接调用 (无 VF_CALL).
//
// 排布 (Path A): 算法每行完全独立 (golden.py 全是 for row in range(R), 行间零依赖).
//   每核只处理自己的连续行段 [blockIdx.x*rowsPerCore, +rowCount), 行内 blockDim
//   线程 grid-stride 扫列. 每核读/写的都是自己负责的行 → 无跨核写后读.
// ⚠ [栈上限 2026-10-19] 设备 VEC SIMT 每线程栈有硬上限 (errcode 354 "VEC SIMT
//   stack overflow", 全核同 PC trap): Process 的全部内联代码共用一个栈帧, 帧大小
//   = 所有内联函数溢出槽之和; 重型 Op (EvictScan/LruRebuild 等) 必须 noinline,
//   新增内联大函数前先想栈。v1 帧恰好在上限下, v2 +2 Op 即越界。
// 同步: 步间用块内 asc_syncthreads (解决行内跨线程 RAW, 如 OpHitMiss 以 topkDim
//   步长写 flags[row,slot]、OpEvictScanRow/OpLruRebuildRow 以 B+1 步长读写
//   同行状态, 同地址被不同线程访问);
//   asc_dcci_entire 刷本核 D-cache 防单核内写后读 stale. 行间零依赖故无需跨核栅栏.

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

// 打包块内独占扫描 [Scan-v2]: value = 字段A | (字段B << shift), 两字段各自的
//   块内总和都必须小于 (1 << shift) (打包加法无跨字段进位, 由调用方保证)。
//   仅 1 次栅栏: warp shuffle 一级 + lane31 写 warpTotals → B1 → 每线程本地读
//   warpTotals[0..warp) 算自己 warp 的 base (确定性函数, 无需 t0 广播)。
//   返回 exclusive (同打包格式); grand total 由调用方返回后本地 Σ warpTotals
//   得到 (缓冲在下次被同槽位复用前有效 —— 背靠背扫描用双/四缓冲错开,
//   顺序安全性: 下一次同槽写发生在此后的 B1 之后, 而本方读在到达该 B1 之前)。
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

// ---- UB 缓冲池 (定义见 echo_lru_tiling.h, host/device 共享) ----
// [v2] 单大 __ubuf__ 数组分段作 scanScratch/flags/aAt/lru/hExcl/aExcl/freeIdx
//   (见 EchoLruKernel::Init)。GM workspace 另有 flags[R][B+1] 分类位数组
//   + [ECHO_PERF_STEP] 分步周期区。

// ===========================================================================
// 行段宏: 每核负责 [startRow, startRow+rowCount). R 以内才有效 (尾核 tailRows).
// ===========================================================================
// 元素级 step: 外层按核遍历自己行段, 内层 ECHO_LRU_THREAD_NUM(=1024) 线程 grid-stride 扫该行 cols.
// 行内串行子步: threadIdx.x==0 串行扫自己行段 (cumsum / freeSize / fifo++ 等).
// ===========================================================================

// ---- Step 0: reset ----
__attribute__((noinline)) __aicore__ void OpResetRows(
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

__attribute__((noinline)) __aicore__ void OpResetDthLru(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* dthGm, int32_t* lruGm) {
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
            // [v2] reset 后 occ=0, lru 有效区为空 → 全行写 0 (尾部约定恒 0,
            //   与 golden 的 lru 状态逐字段全等一致; 无 v1 pri 的 slot0 特例)。
            lruGm[rowOff + col] = 0;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpResetFree(
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

__attribute__((noinline)) __aicore__ void OpResetAvailFifo(
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
// [v2] pri 写改为 flags 位 1 (被淘汰/移除标记): LruRebuild 的分类据此把
//   spec 驱逐槽从 lru 丢弃 (v1 写 pri=-1 的等价物)。spec 路径无基准覆盖
//   (compare.py 的 spl 恒为 max_token), 语义按此维护。
__attribute__((noinline)) __aicore__ void OpInvalidateSpecSuffix(
    uint32_t R, uint32_t B, uint32_t maxModelLenP1, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* splGm, int32_t* dthGm, int32_t* flagsGm,
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
                flagsGm[rowOff + slot] = 2;   // [v2] bit1: LruRebuild 分类丢弃
                int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
                htdGm[htdRowOff + tokSafe] = I32_MAX;
            }
        }
    }
}

__attribute__((noinline)) __aicore__ void OpSpecCumsum(
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

__attribute__((noinline)) __aicore__ void OpSpecPushFree(
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

__attribute__((noinline)) __aicore__ void OpSpecAvailAdd(
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
//   (写入点仅 Step0/0.5 reset 与 Step4 尾折叠更新), 经 Step1 尾 SyncGrid 的
//   dcci 后对后续全员读者可见。计数语义不变: nm 直接数 effTopk 内
//   curSlots==I32_MAX 的个数, 与原公式 effTopk - numHits 恒等。
//   numHitsGm 不再读写 (槽位保留防 workspace 布局漂移)。
// [C1/C2 2026-10-13] 栅栏 2 次/行 → 末行省 1 次 (复用保护); t0 另在 warpScratch[64]
//   写淘汰标志, 供 Process 整段跳过 fs 全 0 的淘汰流水线。
// [v2 2026-10-19] hit 分支从写 pri[slot]=fifo (protect 时间戳) 改为写
//   flags[row][slot] = 1 (bit0 hit 标记): LruRebuild 据此把 hit 槽挪到 lru 尾部
//   (等效 v1 的 pri 抬新 —— EvictScan 跳过 bit0 槽 = v1 "hit 不可被本步淘汰")。
//   distinct token → distinct slot, 线程间无写冲突; 纯 store (prologue 已清零,
//   bit1 只写非 hit 槽 —— EvictScan 选中的必非 hit)。
__attribute__((noinline)) __aicore__ void OpHitMiss(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t maxModelLenP1,
    uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* htdGm, int32_t* posGm, int32_t* curSlotsGm,
    int32_t* numMissesGm, int32_t* availGm, int32_t* freeSizeGm,
    int32_t* flagsGm,
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
        uint32_t flagsRowOff = row * (B + 1u);
        for (uint32_t j = t; j < topkDim; j += blockDim.x) {
            if (j >= effTopk) { curSlotsGm[rowOff + j] = 0; continue; }
            int32_t tok = posGm[rowOff + j];
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            int32_t slot = htdGm[htdRowOff + tokSafe];
            curSlotsGm[rowOff + j] = slot;
            if (slot == I32_MAX) {
                localMiss++;
            } else {
                // [v2] hit 标记: flags bit0 (v1 protect 的等价物)。distinct token →
                //   distinct slot, 线程间无写冲突; flags 写 → EvictScan/Rebuild 读
                //   的 RAW 由 Step1 尾 SyncGrid (dcci) 保证。clamp 语义与原版一致。
                int32_t sp = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
                flagsGm[flagsRowOff + sp] = 1;
            }
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

// ---- Step 2 (v1 时代): protect + fifo++ ----
// [protect 折叠 2026-10-17] OpProtect 已并入 OpHitMiss; [v2] protect 的载体从
//   pri 时间戳换成 flags bit0 (见 OpHitMiss 注释)。
// [v2 2026-10-19] OpFifoInc 删除: fifo 在 v2 无任何 kernel 内读者 (v1 的读者
//   OpAllocPriority/Argtopk startPass 均已不存在), 两次自增 (protect 盖 f /
//   alloc 盖 f+1) 合并为 Step4 尾 OpFifoAvailUpdate 的一次 fifo += 2 ——
//   t0 单线程自写自读无竞争, golden 的 fifo 终值语义不变。

// Step4 尾: fifo 两次自增合并 (+2) + avail 折叠更新 (替代原 OpFreeAvailAdd/
//   OpAllocAvailSub 两个独立 t0 步及其前后 SyncGrid)。
// [P4 2026-10-13] avail 终值 = availOld + fs - nm, 与旧版两步序列 (+fs 后 -nm) 恒等,
//   且恒 ≥ 0: fs = max(nm - availOld, 0) → nm > availOld 时终值 0, 否则 availOld - nm ≥ 0。
//   t1 读的 fs/nm 由 t0 在 Step1 写入、经 Step1 尾 SyncGrid dcci 可见;
//   avail 的全员读者 (OpFreePush/OpAllocPop) 已被 Step3 尾 SyncGrid 排空。
// [v2] fifo 无 kernel 内读者 (OpAllocPriority 已删, 其读 fifo 写 pri 的竞争与
//   Bug2 栅栏随之消失) → 两次 +1 合并为一次 +2, t0 串行 GM 往返再减半。
__attribute__((noinline)) __aicore__ void OpFifoAvailUpdate(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* fifoGm, int32_t* availGm, int32_t* freeSizeGm, int32_t* numMissesGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    // [FifoAvail 拆分] fifo 与 avail 无数据依赖, t0/t1 各管一个。t1 读的 fs/nm
    //   由 t0 在 Step1 写入、经 Step1 尾 SyncGrid dcci 可见; 两写地址不同无冲突。
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            fifoGm[row] = fifoGm[row] + 2;   // [v2] protect(f) 与 alloc(f+1) 两次自增合并
        }
    } else if (threadIdx.x == 1) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            availGm[row] = availGm[row] + freeSizeGm[row] - numMissesGm[row];
        }
    }
}

// ---- prologue (v2): flags 清零 ----
// [v2 2026-10-19] flags 是 EvictScan/Rebuild 的分类输入 (bit0=本步 hit, bit1=
//   本步被淘汰/spec 移除), 每次 invocation 必须从全零起 —— audit 用 I32_MAX
//   污染 workspace, 残留位会被误判 hit/evicted。清零后的 SyncGrid (dcci) 同时
//   排空写缓冲, 保证同核 WAW (清零写 → Step1 scatter 写同地址) 的顺序 ——
//   写滞留线程 store buffer 而他人写已落地会反超覆盖 (fifo 事件同型教训)。
__attribute__((noinline)) __aicore__ void OpZeroFlags(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* flagsGm) {
    uint32_t cols = B + 1u;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        uint32_t rowOff = row * cols;
        for (uint32_t s = threadIdx.x; s < cols; s += blockDim.x) {
            flagsGm[rowOff + s] = 0;
        }
    }
}

// ---- Step 2 (v2): evict scan (淘汰扫描, 取代 v1 Argtopk 基数选择) ----
// 淘汰 = 顺序扫 lru[row][0..occ), 跳过本步 hit (flags bit0; v1 protect 语义的
//   等价物 —— hit 槽的 pri 已被抬到最新, 不可被本步淘汰), 取前 fsClamp 个非 hit
//   槽写入 freeIdx (输出顺序 = 位置序 = (pri,slot) 字典序, 与 v1 emission 逐位
//   一致)。O(occ) 平扫 (1 次 PackedBlockScan + 重扫) 取代 v1 的 7 栅栏多趟选择;
//   每行栅栏 3: staging→扫描 1、scan 内部 1、尾 1 (freeIdx 写 → AllocPop 读 +
//   flagsUb 行间复用)。
// padding 边界 (仅 topk > B 的接口一般性触发): 非 hit 总数 < fsClamp 时按 v1
//   语义补索引 [B+1, B+1+m)。
// ⚠ 切片必须连续 ([P1] 连续切片不变量): 扫描给出的是切片序前缀, grid-stride
//   会得到"线程主序", 淘汰输出顺序就错了。
// [noinline 2026-10-19] 设备 errcode 354 "VEC SIMT stack overflow": Process 若把
//   全部 Op 内联为单一巨型栈帧 (帧 = 所有内联函数溢出槽之和, kernel 入口一次性
//   分配, 与执行路径无关), 会越过设备每线程栈硬上限 —— v2 普通构建擦线通过后,
//   打点构建 (ECHO_PERF_STEP 多几行代码) 再次越界即为实证。规则: **所有 Op 一律
//   noinline** (帧峰值 = 单 Op 最大值, 余量充足); warp 协作辅助 PackedBlockScan
//   保持 inline (并入宿主帧, 宿主帧已很小)。
__attribute__((noinline)) __aicore__ void OpEvictScanRow(
    uint32_t B, uint32_t topkDim, uint32_t fsClamp, uint32_t occ,
    int32_t* lruGm, uint32_t lruRowOff,
    int32_t* flagsGm, uint32_t flagsRowOff,
    int32_t* freeIdxOut,          // 本行切片基址 ([P-C]: UB 本地行 / GM 绝对行, 调用方算好)
    int32_t* flagsUb, int32_t* scanScratch) {
    uint32_t t = threadIdx.x;
    uint32_t warpCount = blockDim.x >> 5;
    uint32_t cols = B + 1u;
    // phase A: flags GM → UB staging (供两遍扫描的随机读; 位置序读 lruGm 保持 GM 连续读)
    for (uint32_t s = t; s < cols; s += blockDim.x) {
        flagsUb[s] = flagsGm[flagsRowOff + s];
    }
    asc_syncthreads();            // staging 写 → 扫描读 (跨线程 RAW, UB 块内栅栏即可)
    // phase B: 切片非 hit 计数 → PackedBlockScan 得切片基址 (单字段, 和 ≤ occ ≤ 4096 < 2^16)
    uint32_t chunk = (occ + blockDim.x - 1u) / blockDim.x;
    uint32_t lo = t * chunk;
    uint32_t hi = (lo >= occ || lo + chunk > occ) ? occ : (lo + chunk);
    int32_t cnt = 0;
    for (uint32_t p = lo; p < hi; ++p) {
        int32_t slot = lruGm[lruRowOff + p];
        int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
        if ((flagsUb[sv] & 1) == 0) ++cnt;
    }
    int32_t base = PackedBlockScan(cnt, scanScratch);
    int32_t grand = 0;
    for (uint32_t w = 0; w < warpCount; ++w) grand += scanScratch[w];
    // phase C: 重扫发射被淘汰槽 (位置序) + flags bit1
    int32_t run = base;
    for (uint32_t p = lo; p < hi; ++p) {
        int32_t slot = lruGm[lruRowOff + p];
        int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
        if ((flagsUb[sv] & 1) == 0) {
            if (run < (int32_t)fsClamp) {
                freeIdxOut[run] = slot;
                // bit1 纯 store: 该槽必非 hit (bit0==0) 且为该槽本步唯一写, 无 RMW 必要;
                //   GM 写 → Rebuild staging 读的可见性由 Step3 尾 SyncGrid (dcci) 保证。
                flagsGm[flagsRowOff + sv] = 2;
            }
            ++run;
        }
    }
    // padding 补齐 (v1 语义: fsClamp 超出非 hit 总数时按序补 [B+1, ...) 索引)
    if (grand < (int32_t)fsClamp) {
        uint32_t m = fsClamp - (uint32_t)grand;
        for (uint32_t k = t; k < m; k += blockDim.x) {
            freeIdxOut[grand + k] = (int32_t)(B + 1u + k);
        }
    }
    // 尾部补零 [fsClamp, topkDim) (v1 逐位一致约定)
    for (uint32_t i = fsClamp + t; i < topkDim; i += blockDim.x) {
        freeIdxOut[i] = 0;
    }
    asc_syncthreads();            // 尾栅栏: freeIdx (UB) 写 → OpAllocPop 读; flagsUb 下一行 staging 复用
}

// ---- Step 4 (v2): lru rebuild (顺序数组重建, 取代 v1 OpAllocPriority) ----
// 每步必执行 (fs=0 时 hit 也要挪尾 —— "每种情况都需要维护序")。产出:
//   新 lru = [幸存者(保持旧序)] ++ [本步 hit 槽按槽号升序] ++ [新分配槽按槽号升序],
//   与 v1 步末 (pri, slot) 稳定排序逐位等价 (幸存者旧 pri < f < f+1 严格分层,
//   同层平局按槽号)。分配列表来自 rawSlots (AllocPop 输出, 语义/顺序与 v1 全同)。
// 每行栅栏 5: B1 staging→扫描; B2 位置域 classify 扫描内部; B3 槽域 hit|alloc
//   联合扫描内部 (scan2 用 scanScratch+32 独立缓冲, [Scan-v2] 链式扫描不相交
//   缓冲规则 —— scan1 的 grand Σ 读 [0,32) 与 scan2 的写 [32,64) 无冲突);
//   B3.5 phase C 前缀存储 → phase D 散射读 (跨线程, 见函数内注释);
//   B4 尾 (UB 五段行间复用保护)。
// lru staging 进 UB 是 phase D 散射写 lruGm 的前置 (H 目标位置可 < p, 同数组
//   跨线程读写竞争, 必须先全体读入 UB 再写回 GM)。
__attribute__((noinline)) __aicore__ void OpLruRebuildRow(
    uint32_t B, uint32_t topkDim, uint32_t occ, uint32_t nm,
    int32_t* lruGm, uint32_t lruRowOff,
    int32_t* flagsGm, uint32_t flagsRowOff,
    int32_t* rawSlotsGm, uint32_t rawRowOff,
    int32_t* flagsUb, int32_t* aAtUb, int32_t* lruUb,
    int32_t* hExclUb, int32_t* aExclUb, int32_t* scanScratch,
    uint32_t specOn) {
    uint32_t t = threadIdx.x;
    uint32_t warpCount = blockDim.x >> 5;
    uint32_t cols = B + 1u;
    // ---- phase A: staging + 清零 (纯写): flags GM→UB; lru GM→UB; aAt 清零 ----
    for (uint32_t s = t; s < cols; s += blockDim.x) {
        flagsUb[s] = flagsGm[flagsRowOff + s];
        aAtUb[s] = 0;
    }
    for (uint32_t p = t; p < occ; p += blockDim.x) {
        lruUb[p] = lruGm[lruRowOff + p];
    }
    asc_syncthreads();                       // B1
    // ---- phase B: aAt 散射 (rawSlots, Step3 尾 SyncGrid 后已可见) + 位置域分类扫描。
    //      打包 (isK | isH<<16): 各切片和 ≤ occ ≤ 4096 < 2^16, 字段无跨位进位。 ----
    for (uint32_t j = t; j < nm; j += blockDim.x) {
        int32_t a = rawSlotsGm[rawRowOff + j];
        int32_t ac = a < 0 ? 0 : (a > (int32_t)B ? (int32_t)B : a);
        aAtUb[ac] = 1;
    }
    uint32_t chunk = (occ + blockDim.x - 1u) / blockDim.x;
    uint32_t lo = t * chunk;
    uint32_t hi = (lo >= occ || lo + chunk > occ) ? occ : (lo + chunk);
    int32_t kCnt = 0, hCnt = 0;
    for (uint32_t p = lo; p < hi; ++p) {
        int32_t slot = lruUb[p];
        int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
        int32_t v = flagsUb[sv];
        if (v & 1) ++hCnt;
        else if ((v & 2) == 0) ++kCnt;       // bit1 (淘汰/spec 移除) → 丢弃不计
    }
#ifdef ECHO_V2_RB_SERIAL
    // [debug] 串行替代 PackedBlockScan (判别 codegen 嫌疑): 切片计数写缓冲 → 栅栏 →
    //   t0 串行 exclusive 前缀 + grand → 栅栏 → 读回。语义与 PackedBlockScan 逐位一致。
    scanScratch[t] = kCnt | (hCnt << 16);
    asc_syncthreads();
    if (t == 0u) {
        int32_t run = 0;
        for (uint32_t i = 0; i < blockDim.x; ++i) {
            int32_t v = scanScratch[i];
            scanScratch[i] = run;
            run += v;
        }
        scanScratch[blockDim.x] = run;   // grand (打包)
    }
    asc_syncthreads();
    int32_t excl = scanScratch[t];
    int32_t grandP = scanScratch[blockDim.x];
#else
    int32_t excl = PackedBlockScan(kCnt | (hCnt << 16), scanScratch);   // B2
    int32_t grandP = 0;
    for (uint32_t w = 0; w < warpCount; ++w) grandP += scanScratch[w];
#endif
    int32_t baseK = excl & 0xFFFF;
    uint32_t totalK = (uint32_t)grandP & 0xFFFFu;
    uint32_t totalH = (uint32_t)grandP >> 16;
    // ---- phase C: 槽域联合扫描 (hitBit | allocBit<<16) 产出 hExcl[s]/aExcl[s]
    //      (exclusive 前缀: 槽 s 之前的 hit/alloc 数) —— H/A 段内槽号升序的秩。 ----
    uint32_t slotChunk = (cols + blockDim.x - 1u) / blockDim.x;
    uint32_t slo = t * slotChunk;
    uint32_t shi = (slo >= cols || slo + slotChunk > cols) ? cols : (slo + slotChunk);
    int32_t haCnt = 0;
    for (uint32_t s = slo; s < shi; ++s) {
        // ⚠ 计数必须用加法: 切片可含多个 hit/alloc (B=4096/blockDim=128 时 33 槽/线程),
        //   首版误用 |= (塌缩成至多 1) → hExcl/aExcl 秩错 → H/A 段错位
        //   (CPU 回归 group2 B=4096 用例抓获, 2026-10-19)。
        if (flagsUb[s] & 1) haCnt += 1;
        if (aAtUb[s]) haCnt += (1 << 16);
    }
#ifdef ECHO_V2_RB_SERIAL
    int32_t* buf2s = scanScratch + 1024;   // 串行模式独立半区 (rbScan 段已扩容)
    buf2s[t] = haCnt;
    asc_syncthreads();
    if (t == 0u) {
        int32_t run = 0;
        for (uint32_t i = 0; i < blockDim.x; ++i) {
            int32_t v = buf2s[i];
            buf2s[i] = run;
            run += v;
        }
        buf2s[blockDim.x] = run;
    }
    asc_syncthreads();
    int32_t excl2 = buf2s[t];
#else
    int32_t excl2 = PackedBlockScan(haCnt, scanScratch + 32);           // B3
#endif
    int32_t runH = excl2 & 0xFFFF;
    int32_t runA = excl2 >> 16;
    for (uint32_t s = slo; s < shi; ++s) {
        hExclUb[s] = runH;
        aExclUb[s] = runA;
        if (flagsUb[s] & 1) ++runH;
        if (aAtUb[s]) ++runA;
    }
    // B3.5: phase C 的 hExcl/aExcl 存储发生在 B3 (scan2 内部栅栏) 之后, 而 phase D
    //   要读任意槽的前缀 (写者与读者跨线程) —— 没有这道栅栏会读到未写的哨兵
    //   (-777 → 转 uint32 野索引), CPU 回归实测 SEGV (2026-10-19, checks 组 2 抓获)。
    //   对照: OpMissCumsum 的链式扫描无此问题, 其跨线程读 (warpTotals Σ) 的数据
    //   在 PackedBlockScan 内部栅栏前已写好。
    asc_syncthreads();
    // ---- phase D: 散射写回 GM lru。K: 切片基址+片内运行前缀 (位置序); H: totalK +
    //      hExcl[slot] (槽号升序); A: totalK+totalH + aExcl[aslot]; 尾 [occNew, B] 清零。 ----
#if defined(ECHO_V2_RB_NOD)
    // [debug] RB_NOD: 跳过 phase D 全部 GM 写 (栅栏配平保留)。
    (void)baseK;
#elif defined(ECHO_V2_RB_ZEROONLY)
    // [debug] RB_ZEROONLY: phase D 换全行清零 (与 OpResetDthLru 的 lru 写同型 ——
    //   该模式已在 Step0 被设备验证安全; 用于判别触发源是 "写容量/模式" 还是
    //   "数据依赖散射")。
    (void)baseK;
    for (uint32_t p = t; p <= B; p += blockDim.x) {
        lruGm[lruRowOff + p] = 0;
    }
#elif defined(ECHO_V2_RB_ONEONLY)
    // [debug] RB_ONEONLY: 全行写常量 1 (升序非零) —— 判别触发器是 "零值"
    //   (memset/MTE fill 惯用语) 还是 "升序模式" 本身。golden 必挂, 只看 trap。
    (void)baseK;
    for (uint32_t p = t; p <= B; p += blockDim.x) {
        lruGm[lruRowOff + p] = 1;
    }
#else
    int32_t runK = baseK;
    for (uint32_t p = lo; p < hi; ++p) {
        int32_t slot = lruUb[p];
        int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
        int32_t v = flagsUb[sv];
        if (v & 1) {
            lruGm[lruRowOff + totalK + (uint32_t)hExclUb[sv]] = slot;
        } else if (v & 2) {
            // 被淘汰 / spec 移除: 移出顺序 (v1 语义中其 pri 已无效)
        } else {
            lruGm[lruRowOff + (uint32_t)runK] = slot;
            ++runK;
        }
    }
    for (uint32_t j = t; j < nm; j += blockDim.x) {
        int32_t a = rawSlotsGm[rawRowOff + j];
        int32_t ac = a < 0 ? 0 : (a > (int32_t)B ? (int32_t)B : a);
        lruGm[lruRowOff + totalK + totalH + (uint32_t)aExclUb[ac]] = ac;
    }
    // [尾零删除 2026-10-19] 非 spec 路径的尾零循环数学上冗余, 已删除:
    //   ① K/H/A 散射目标恒 < occNew (K < totalK; H < totalK+totalH; A < occNew);
    //   ② 尾区 [occNew, B] ⊆ [occOld, B] (fs>0 ⟹ nm>availOld ⟹ availNew=0 ⟹
    //      occNew=B 尾空; fs=0 ⟹ occNew=occOld+nm);
    //   ③ [occOld, B] 调用前已为零 (restore 的 initial 尾零 / reset 的全行清零 /
    //      上一 invocation 的尾零 —— 不变量归纳; host 侧需保证首次分配后 lru 清零,
    //      compare.py 的 make_echo_initial 已满足)。
    //   删除同时消除设备 trap 触发源: "升序可合并 GM 写循环"(值无关、-O0 亦然)
    //   在重 UB 写之后触发 507035 (§9.18 调试: NOTAIL clean / ZEROONLY+ONEONLY
    //   trap / ZLOAD trap / 散射 A-append clean)。
    //   spec 路径例外: avail 动态增加 → [occNew, occOld) 可能残留旧序值 → 保留
    //   尾零 (spec 无基准覆盖, 语义按此维护; 若上板触发同类 trap 需另行处理)。
    if (specOn != 0u) {
        uint32_t occNew = totalK + totalH + nm;
        for (uint32_t p = occNew + t; p <= B; p += blockDim.x) {
            lruGm[lruRowOff + p] = 0;
        }
    }
#endif
    asc_syncthreads();                       // B4: flagsUb/aAtUb/lruUb/hExclUb/aExclUb 下一行 phase A 复用保护
}



__attribute__((noinline)) __aicore__ void OpFreeRelease(
    uint32_t R, uint32_t topkDim, uint32_t B, uint32_t maxModelLenP1,
    uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* freeIdxGm, bool freeIdxLocal, int32_t* freeSizeGm,
    int32_t* dthGm, int32_t* htdGm) {
    uint32_t cols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t fs = freeSizeGm[row];
        uint32_t fiOff = (freeIdxLocal ? (row - startRow) : row) * topkDim;  // [P-C] UB 本地 / GM 绝对
        uint32_t dthRowOff = row * cols;
        uint32_t htdRowOff = row * maxModelLenP1;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if ((int32_t)j >= fs) continue;
            int32_t slot = freeIdxGm[fiOff + j];
            int32_t s = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
            uint32_t off = dthRowOff + s;
            int32_t tok = dthGm[off];
            // [v2] pri=-1 写已删 (顺序维护移交 OpLruRebuild 的 bit1 丢弃路径)
            dthGm[off] = I32_MAX;
            if (tok != I32_MAX) {
                int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
                htdGm[htdRowOff + tokSafe] = I32_MAX;
            }
        }
    }
}

__attribute__((noinline)) __aicore__ void OpFreePush(
    uint32_t R, uint32_t topkDim, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* freeIdxGm, bool freeIdxLocal, int32_t* freeSizeGm,
    int32_t* availGm, int32_t* freeGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t fs = freeSizeGm[row];
        int32_t avail = availGm[row];
        uint32_t fiOff = (freeIdxLocal ? (row - startRow) : row) * topkDim;  // [P-C] UB 本地 / GM 绝对
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if ((int32_t)j >= fs) continue;
            int32_t writeIdx = avail + (int32_t)j;
            if (writeIdx < 0) writeIdx = 0;
            if (writeIdx >= (int32_t)freeStackSize) writeIdx = (int32_t)freeStackSize - 1;
            freeGm[freeRowOff + writeIdx] = freeIdxGm[fiOff + j];
        }
    }
}

// ---- Step 3: alloc pop ----
// [P4 2026-10-13] 直读 freeIdx: 旧版 avail 先 +=fs 再 -=nm 后从 free 栈弹出
//   free[availOld+fs-nm+j]; 而新压入段满足 free[availOld+k] == freeIdx[k] (k<fs),
//   故弹出等价拆分为: readIdx ≥ availOld → freeIdx[readIdx-availOld] (不依赖
//   OpFreePush 对 free 栈新段的写入), readIdx < availOld → 老栈 free[readIdx]
//   (本 kernel 从未写该区段) —— 与旧版逐值一致, 且 FreePush 的写入在本 kernel
//   内不再有读者 (仅持久化给下一次调用), FreePush→AllocPop 间的 SyncGrid 可删。
//   avail 的两步更新 (原 OpFreeAvailAdd +=fs / OpAllocAvailSub -=nm) 合并为
//   Step5 尾 OpFifoAvailUpdate 的一次写; 本函数读到的 avail 恒为初值 availOld。
__attribute__((noinline)) __aicore__ void OpAllocPop(
    uint32_t R, uint32_t topkDim, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* numMissesGm, int32_t* freeSizeGm, int32_t* availGm,
    int32_t* freeIdxGm, bool freeIdxLocal, int32_t* freeGm, int32_t* rawSlotsGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t nm = numMissesGm[row];
        int32_t avail = availGm[row];      // 初值 availOld (更新推迟到 Step5 尾)
        int32_t fs = freeSizeGm[row];
        uint32_t rowOff = row * topkDim;                          // rawSlots (GM) 恒绝对行
        uint32_t fiOff = (freeIdxLocal ? (row - startRow) : row) * topkDim;  // [P-C] UB 本地 / GM 绝对
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            int32_t slot;
            if (j < (uint32_t)nm) {
                int32_t readIdx = avail + fs - (int32_t)nm + (int32_t)j;
                if (readIdx >= avail) {
                    int32_t fi = readIdx - avail;                 // ∈ [0, fs) ⊂ [0, topkDim)
                    if (fi < 0) fi = 0;
                    if (fi >= (int32_t)topkDim) fi = (int32_t)topkDim - 1;
                    slot = freeIdxGm[fiOff + fi];
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

// ---- Step 4: alloc priority ----
// [v2 2026-10-19] OpAllocPriority 已删除: v1 写 pri[aslot]=fifo+1 的等价物是
//   OpLruRebuild 的 A 段追加 (新分配槽按槽号升序接到 lru 尾部)。

// ---- Step 5: set_update ----
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
__attribute__((noinline)) __aicore__ void OpMissCumsum(
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

__attribute__((noinline)) __aicore__ void OpSetUpdate(
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

__attribute__((noinline)) __aicore__ void OpFillZero(
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

    __aicore__ inline void Init(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* free,
                      int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                      uint8_t* resetMask, int32_t* curSlots,
                      int32_t* missHostPos, int32_t* missAllocFlat, uint8_t* missMask,
                      void* workspace, const EchoLruTiling& tiling,
                      int32_t* ubPool) {
        tiling_ = tiling;
        htdGm_ = htd;
        dthGm_ = dth;
        lruGm_ = lru;
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
        wsCumsumGm_   = ws + o; o += R * (B + 1);       // spec 路径前缀和暂存
        pushCountGm_  = ws + o; o += R;                 // spec 路径释放计数
        numHitsGm_    = ws + o; o += R;                 // [P0] 已无读写, 槽位保留防布局漂移
        numMissesGm_  = ws + o; o += R;
        freeIdxGm_    = ws + o; o += R * topkDim;       // row stride = topkDim ([P-C] GM 回退落点)
        freeSizeGm_   = ws + o; o += R;
        rawSlotsGm_   = ws + o; o += R * topkDim;       // row stride = topkDim
        wsMissRankGm_ = ws + o; o += R * topkDim;
        flagsGm_      = ws + o; o += R * (B + 1);       // [v2] 分类位: bit0=hit, bit1=淘汰/spec 移除

        // GM workspace 尾部: [ECHO_PERF_STEP] 每核分步周期 [bn*ECHO_PERF_NUM_STEPS]。
        //   与 host 侧 EchoWorkspaceElements 公式逐项一致 (v1 的 argtopk 诊断 dump 区已随基数选择删除)。
        uint32_t bn = tiling_.blockNum;
        uint32_t elemOff = (o + 1u) & ~1u;  // 8 字节对齐 (2×int32)

        stepCyclesGm_ = nullptr;
#ifdef ECHO_PERF_STEP
        stepCyclesGm_ = ws + elemOff;
        elemOff += bn * ECHO_PERF_NUM_STEPS;
#endif
        perfStepIdx_ = 0u;

        // [UB 布局 v2] 单大 __ubuf__ 数组分段 (probe_ub_size 实测多 __ubuf__ 数组会
        //   别名互覆, 单大数组 32768 元素 PASS):
        //   [0, 128)          scanScratch —— 分时复用: Step1 warp 归约+C1 槽 64
        //                      → Step2 EvictScan 扫描 → Step4 Rebuild scan1[0,32)+scan2[32,64)
        //                      → Step5 MissCumsum 扫描缓冲 [0,128) (各生命周期由其间
        //                      SyncGrid/行内尾栅栏隔开, 详见各 Op 注释);
        //   [128, +B+1)       flagsUb  (EvictScan/Rebuild 的 flags staging);
        //   +B+1              aAtUb    (Rebuild 分配指示, 槽域);
        //   +B+1              lruUb    (Rebuild 的旧序 staging, 防 lruGm 同数组读写竞争);
        //   +B+1              hExclUb  (槽域 hit exclusive 前缀);
        //   +B+1              aExclUb  (槽域 alloc exclusive 前缀);
        //   基准形态合计 128 + 5*4097 = 20613 ≤ ARGTOPK_UB_ELEMS=32768 (余 12155)。
        scanScratchUb_ = ubPool;
        // [对齐 2026-10-19] 每段基址向上取整到 256 元素 (1KB) 边界: 对齐 v1 已验证
        //   的 UB 布局形态 (v1 全部分段偏移均为 256 元素倍数, 内部子缓冲 128B 对齐);
        //   v2 首版奇数偏移 (4225/8322/..., 仅 4B 对齐) 是 cold_reset 上板 trap
        //   (507035) 的头号嫌疑 —— CPU mock 无法覆盖硬件对齐行为。基准形态峰值
        //   22016 + 2048 = 24064 ≤ 32768。
        uint32_t uo = 128u;
        uo = (uo + 255u) & ~255u;
        flagsUb_  = ubPool + uo; uo += B + 1u;
        uo = (uo + 255u) & ~255u;
        aAtUb_    = ubPool + uo; uo += B + 1u;
        uo = (uo + 255u) & ~255u;
        lruUb_    = ubPool + uo; uo += B + 1u;
        uo = (uo + 255u) & ~255u;
        hExclUb_  = ubPool + uo; uo += B + 1u;
        uo = (uo + 255u) & ~255u;
        aExclUb_  = ubPool + uo; uo += B + 1u;

        // [P-C] freeIdx 迁 UB (语义沿用 v1): 仅本 kernel 内读者, 容纳 rowsPerCore*topkDim;
        //   放得下走 UB, 放不下 (大 batch 形态) 回退 GM。
        freeIdxInUb_ = false;
        freeIdxPtr_ = freeIdxGm_;
        uint32_t fiElems = tiling_.rowsPerCore * topkDim;
        if (uo + fiElems <= ARGTOPK_UB_ELEMS) {
            freeIdxUb_ = ubPool + uo; uo += fiElems;
            freeIdxPtr_ = freeIdxUb_;
            freeIdxInUb_ = true;
        }
        // [scan 分离 2026-10-19] Step2/Step4 的扫描缓冲独立成段 (freeIdx 之后, 256 对齐):
        //   不再与 scanScratch [0,128) 分时复用。v1 里 [0,128) 的写者序列是
        //   OpHitMiss warpScratch → MissCumsum (已验证); v2 首版让 rebuild 的
        //   PackedBlockScan warpTotals 也写 [0,wc)/[32,32+wc) —— 与 MissCumsum 的
        //   缓冲同址重写, 该序列在设备上触发 trap (507035, skip-rebuild 实验隔离)。
        uo = (uo + 255u) & ~255u;
        // 段容量 2*ECHO_LRU_THREAD_NUM+64: 正常路径只用 [0,64); ECHO_V2_RB_SERIAL
        //   串行判别模式需 2*(blockDim+1) ≤ 2050 槽。
        if (uo + 2u * ECHO_LRU_THREAD_NUM + 64u <= ARGTOPK_UB_ELEMS) {
            rbScanUb_ = ubPool + uo; uo += 2u * ECHO_LRU_THREAD_NUM + 64u;
        } else {
            rbScanUb_ = ubPool;   // 超预算时回退共享 (大 batch 回退形态, 不在基准覆盖)
        }
    }

    // Process: 各步顺序执行, 步间 SyncGrid() (块内 syncthreads + dcci) 解决行内跨线程 RAW.
    //   ECHO_PERF_STEP 构建下 PerfBegin/PerfEnd 把每步 t0 的 clock() 周期写入 workspace
    //   尾部 (见 tiling.h ECHO_PERF_NUM_STEPS); 普通构建为空操作, 不改变行为.
    // [v2 步表 (非 spec 18 步, v1 为 20)]:
    //   prologue(无打点): OpZeroFlags(+spec OpFillZero) + SyncGrid
    //   Step0 (anyReset 门控): OpResetRows / OpResetDthLru / OpResetFree /
    //     OpResetAvailFifo + SyncGrid —— 5 槽
    //   Step1: OpHitMiss (hit→flags bit0) + SyncGrid —— 2 槽
    //   Step2 ([C1] needFree 门控): EvictScanAllRows / SyncGrid(仅 GM freeIdx
    //     回退) / OpFreeRelease / OpFreePush —— 4 槽
    //   Step3: OpAllocPop + SyncGrid (OpFifoInc 已并入 Step4) —— 2 槽
    //   Step4: LruRebuildAllRows + OpFifoAvailUpdate (fifo+=2, avail+=fs-nm) —— 2 槽
    //   Step5: OpMissCumsum + SyncGrid + OpSetUpdate —— 3 槽
    __aicore__ inline void Process() {
        uint32_t R = tiling_.numTokens;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t maxModelLenP1 = tiling_.maxModelLenP1;
        uint32_t freeStackSize = tiling_.freeStackSize;
        uint32_t effTopk = tiling_.effTopk;
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;

        // ---- prologue (无打点): flags 清零 (+ spec pushCount) + SyncGrid ----
        // [v2] flags 每次 invocation 从全零起 (audit 污染防御); SyncGrid 的 dcci
        //   排空写缓冲, 保证同核 WAW (清零 → Step1 scatter 同地址) 顺序 (fifo 事件
        //   同型教训: 写滞留 store buffer 可能反超覆盖他人已落地写)。
        OpZeroFlags(R, B, rowsPerCore, tailRows, flagsGm_);
        if (tiling_.specEnabled) {
            OpFillZero(R, rowsPerCore, tailRows, pushCountGm_);
        }
        SyncGrid();
        if (ECHO_V2_DEBUG_CUTOFF_V == 1u) return;   // [v2 debug] 阶段 1 截断

        // ---- Step 0: reset (anyReset 门控, 沿用 round-4 白付清理) ----
        bool anyReset = false;
        {
            uint32_t startRow = blockIdx.x * rowsPerCore;
            uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
            for (uint32_t r = 0; r < rowCount; ++r) {
                anyReset = anyReset || (resetMaskGm_[startRow + r] != 0);
            }
        }
        PerfBegin(); if (anyReset) OpResetRows(R, maxModelLenP1, rowsPerCore, tailRows, resetMaskGm_, htdGm_); PerfEnd();
        PerfBegin(); if (anyReset) OpResetDthLru(R, B, rowsPerCore, tailRows, resetMaskGm_, dthGm_, lruGm_); PerfEnd();
        PerfBegin(); if (anyReset) OpResetFree(R, B, freeStackSize, rowsPerCore, tailRows, resetMaskGm_, freeGm_); PerfEnd();
        PerfBegin(); if (anyReset) OpResetAvailFifo(R, B, rowsPerCore, tailRows, resetMaskGm_, availGm_, fifoGm_); PerfEnd();
        PerfBegin(); if (anyReset) SyncGrid(); PerfEnd();
        if (ECHO_V2_DEBUG_CUTOFF_V == 2u) return;   // [v2 debug] 阶段 2 截断

        // ---- Step 0.5: invalidate speculative suffix (spec 路径; 非 spec 跳过) ----
        if (tiling_.specEnabled) {
            PerfBegin(); OpInvalidateSpecSuffix(R, B, maxModelLenP1, rowsPerCore, tailRows, splGm_, dthGm_, flagsGm_, htdGm_, wsCumsumGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
            PerfBegin(); OpSpecCumsum(R, B, rowsPerCore, tailRows, wsCumsumGm_, pushCountGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
            PerfBegin(); OpSpecPushFree(R, B, freeStackSize, rowsPerCore, tailRows, wsCumsumGm_, availGm_, freeGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
            PerfBegin(); OpSpecAvailAdd(R, rowsPerCore, tailRows, availGm_, pushCountGm_); PerfEnd();
            PerfBegin(); SyncGrid(); PerfEnd();
        }

        // ---- Step 1: hit/miss + hit 标记 + miss 计数 + freeSize ----
        PerfBegin(); OpHitMiss(R, topkDim, effTopk, maxModelLenP1, B, rowsPerCore, tailRows,
                     htdGm_, posGm_, curSlotsGm_, numMissesGm_, availGm_, freeSizeGm_,
                     flagsGm_, scanScratchUb_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();  // flags/curSlots (全员写) 经 dcci 对后续全员读可见;
                                            // numMisses/freeSize (t0 写) 同点可见。
        if (ECHO_V2_DEBUG_CUTOFF_V == 3u) return;   // [v2 debug] 阶段 3 截断

        // ---- Step 2: free ([C1] needFree 门控; v2 用 EvictScan 取代基数选择) ----
        // [C1 2026-10-13] t0 在 Step1 写入"本核任一行 fs>0"标志 (UB 槽 64, 经 Step1
        //   尾 SyncGrid 可见)。全 0 时 EvictScan/FreeRelease/FreePush 对状态零影响
        //   (fs=0 行 OpEvictScanRow 整行跳过、freeIdx 无读者 —— OpAllocPop 恒走老栈
        //   路径, FreeRelease/FreePush 的 j<fs 循环体不执行)。打点槽保留 (≈0 周期),
        //   条件块统一, 栅栏无发散。
        bool needFree = (scanScratchUb_[64] != 0);
        PerfBegin(); if (needFree) EvictScanAllRows(); PerfEnd();
        // [P-C+尾栅栏去重] freeIdx 写 (末行 EvictScan 尾栅栏) → OpFreeRelease 读:
        //   UB 路径已排序, 本槽跳过; GM 回退路径保守保留完整 SyncGrid (大 batch
        //   形态不在基准覆盖, 不冒险)。
        PerfBegin(); if (needFree && !freeIdxInUb_) SyncGrid(); PerfEnd();
        PerfBegin(); if (needFree) OpFreeRelease(R, topkDim, B, maxModelLenP1, rowsPerCore, tailRows, freeIdxPtr_, freeIdxInUb_, freeSizeGm_, dthGm_, htdGm_); PerfEnd();
        PerfBegin(); if (needFree) OpFreePush(R, topkDim, freeStackSize, rowsPerCore, tailRows, freeIdxPtr_, freeIdxInUb_, freeSizeGm_, availGm_, freeGm_); PerfEnd();
        if (ECHO_V2_DEBUG_CUTOFF_V == 4u) return;   // [v2 debug] 阶段 4 截断

        // ---- Step 3: alloc pop (OpFifoInc 已并入 Step4 的 fifo+=2) ----
        PerfBegin(); OpAllocPop(R, topkDim, freeStackSize, rowsPerCore, tailRows,
                     numMissesGm_, freeSizeGm_, availGm_, freeIdxPtr_, freeIdxInUb_, freeGm_, rawSlotsGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();  // rawSlots 跨线程 RAW → Rebuild/SetUpdate 读;
                                             // flags bit1 (EvictScan 写) → Rebuild staging 读 (dcci)
        if (ECHO_V2_DEBUG_CUTOFF_V == 5u) return;   // [v2 debug] 阶段 5 截断

        // ---- Step 4: lru 重建 (每步必执行) + fifo/avail 折叠更新 ----
        PerfBegin(); LruRebuildAllRows(); PerfEnd();
        PerfBegin(); OpFifoAvailUpdate(R, rowsPerCore, tailRows, fifoGm_, availGm_, freeSizeGm_, numMissesGm_); PerfEnd();
#if defined(ECHO_V2_RB_DCCI)
        // [debug/候选修复] Step4→Step5 之间补回栅栏: 重 UB 写 (rebuild A-C) 之后的
        //   可合并递增 GM 写 (phase D 尾零) 会毒化紧随的访存 —— MissCumsum trap。
        //   v1 在此位置有 SyncBlock (v2 曾以竞争源删除为由去掉)。RB_DCCI = 完整
        //   SyncGrid (dcci 三明治); RB_SYNC = 裸 syncthreads (最小修复对照)。
        SyncGrid();
#elif defined(ECHO_V2_RB_SYNC)
        asc_syncthreads();
#endif
        if (ECHO_V2_DEBUG_CUTOFF_V == 6u) return;   // [v2 debug] 阶段 6 截断 (Step4 后)
        // [P4/v2] 尾 SyncGrid 已删: lru/fifo/avail 此后无本 kernel 内读者 (kernel
        //   结束即对 host/下一次调用可见); Bug2 栅栏随 OpAllocPriority 删除 (fifo
        //   竞争源消失, avail 读者已被 Step3 SyncGrid 排空)。

        // ---- Step 5: set_update (含最终槽位回写, 原 OpFinalSlots 已并入) ----
        PerfBegin(); OpMissCumsum(R, topkDim, effTopk, rowsPerCore, tailRows,
                     curSlotsGm_, posGm_, wsMissRankGm_, missHostPosGm_, scanScratchUb_); PerfEnd();
        if (ECHO_V2_DEBUG_CUTOFF_V == 7u) return;   // [v2 debug] 阶段 7 截断 (MissCumsum 后)
        PerfBegin(); SyncGrid(); PerfEnd();  // wsMissRank/missPos 跨线程 RAW → OpSetUpdate
        if (ECHO_V2_DEBUG_CUTOFF_V == 8u) return;   // [v2 debug] 阶段 8 截断 (Step5 SyncGrid 后)
        PerfBegin(); OpSetUpdate(R, topkDim, effTopk, B, maxModelLenP1, rowsPerCore, tailRows,
                    curSlotsGm_, wsMissRankGm_, missHostPosGm_, missAllocFlatGm_, missMaskGm_,
                    rawSlotsGm_, htdGm_, dthGm_); PerfEnd();
        // [P2] 之后无本 kernel 内读者, 不再需要尾步/尾栅栏 (kernel 结束即对 host 可见)。
    }

private:
    // [v2] SyncBlock 已删: 唯一调用者 Bug2 栅栏随 OpAllocPriority/fifo 竞争源删除。

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

    // [v2] EvictScan 的行循环包装 (v1 ArgtopkAllRows 的位置)。fs=0 行整行跳过
    //   (无栅栏无写; fs 为行级一致条件 —— 全员广播读同一 freeSizeGm[row])。
    //   freeIdx 行切片基址沿用 [P-C] 语义: UB 路径按核内本地行索引 (缓冲仅容纳
    //   rowsPerCore*topkDim), GM 路径按绝对行索引 (§9.13 教训)。
    __aicore__ inline void EvictScanAllRows() {
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;
        uint32_t startRow = blockIdx.x * rowsPerCore;
        uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t R = tiling_.numTokens;
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t fs = freeSizeGm_[row];
            if (fs <= 0) continue;          // 无淘汰行: 零写零栅栏 (v1 fsClamp==0 早退等价)
            uint32_t fsClamp = (uint32_t)((fs > (int32_t)topkDim) ? topkDim : fs);
            int32_t avail = availGm_[row];  // 恒为初值 availOld (更新在 Step4 尾)
            uint32_t occ = (avail >= (int32_t)B) ? 0u : ((uint32_t)B - (uint32_t)avail);
            int32_t* fiRow = freeIdxPtr_ + ((freeIdxInUb_ ? (row - startRow) : row) * topkDim);
            OpEvictScanRow(B, topkDim, fsClamp, occ,
                           lruGm_, row * (B + 1u),
                           flagsGm_, row * (B + 1u),
                           fiRow,
                           flagsUb_, rbScanUb_);
        }
    }

    // [v2] LruRebuild 的行循环包装。每行必执行 (fs=0 时 hit 也要挪尾)。
    //   occ 用 availOld (avail 更新在紧随其后的 OpFifoAvailUpdate); nm/avail
    //   均为 Step1/Step3 后的稳定值, Step3 尾 SyncGrid 已保证可见。
    __aicore__ inline void LruRebuildAllRows() {
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;
        uint32_t startRow = blockIdx.x * rowsPerCore;
        uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t R = tiling_.numTokens;
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
#ifdef ECHO_V2_SKIP_REBUILD
            // [v2 debug] 跳过重建体 (零写零栅栏): 定位 MissCumsum trap 是否由
            //   rebuild 的 UB/GM 写触发。golden 必挂 (lru 不更新) —— 只看 trap 有无。
            continue;
#endif
            int32_t avail = availGm_[row];
            uint32_t occ = (avail >= (int32_t)B) ? 0u : ((uint32_t)B - (uint32_t)avail);
            int32_t nm = numMissesGm_[row];
            uint32_t nmU = (nm < 0) ? 0u : (uint32_t)nm;
            OpLruRebuildRow(B, topkDim, occ, nmU,
                            lruGm_, row * (B + 1u),
                            flagsGm_, row * (B + 1u),
                            rawSlotsGm_, row * topkDim,
                            flagsUb_, aAtUb_, lruUb_,
                            hExclUb_, aExclUb_, rbScanUb_,
                            tiling_.specEnabled);
        }
    }

    // [v2] NpadOf 已删 (基数选择移除后无调用者)。

    EchoLruTiling tiling_;
    int32_t* htdGm_;
    int32_t* dthGm_;
    int32_t* lruGm_;      // [v2] 秩→槽位顺序数组 (替代 v1 pri 时间戳)
    int32_t* freeGm_;
    int32_t* availGm_;
    int32_t* fifoGm_;     // [v2] 死状态 (无 kernel 内读者), 按 golden 语义 +2/步维护
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
    int32_t* flagsGm_;    // [v2] workspace 分类位: bit0=hit, bit1=淘汰/spec 移除
    int32_t* scanScratchUb_;   // [UB v2] [0,128) Step1 warp 归约/C1 与 Step5 MissCumsum 专用 (v1 已验证序列)
    int32_t* rbScanUb_;        // [scan 分离] Step2 EvictScan / Step4 Rebuild 的扫描缓冲 (独立段, 不与 [0,128) 复用)
    int32_t* flagsUb_;
    int32_t* aAtUb_;
    int32_t* lruUb_;
    int32_t* hExclUb_;
    int32_t* aExclUb_;
    void* wsBase_;
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
    int32_t* htd, int32_t* dth, int32_t* lru, int32_t* free,
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
    tiling.argtopkDebugRow = tilingBuf->argtopkDebugRow;          // [v2] 已无 kernel 内读者, 保结构兼容
    tiling.argtopkGroupThreads = tilingBuf->argtopkGroupThreads; // [v2] 同上
    // [UB 布局 v2] 单大 __ubuf__ 数组分段作 scanScratch/flags/aAt/lru/hExcl/aExcl/
    //   freeIdx (见 Init 注释)。probe_ub_size 实测: 单 __ubuf__ 数组可到 192KB
    //   (256KB 才 trap); 多个独立 __ubuf__ 数组会别名互相覆盖 (读 A 得 D)。
    //   基准形态峰值 128+5*4097+2048 = 22661 ≤ ARGTOPK_UB_ELEMS=32768。
    __ubuf__ int32_t argtopkUb[ARGTOPK_UB_ELEMS];
    EchoLruKernel op;
    op.Init(htd, dth, lru, free, avail, fifo, pos, spl, resetMask, curSlots,
            missHostPos, missAllocFlat, missMask, workspace, tiling, argtopkUb);
    // Path A: 不使用 grid_group (跨核 grid.sync 在 echo_lru 内 trap, 见 SyncGrid 注释, 暂停攻关).
    //   行间零依赖, 步间块内 SyncGrid() (syncthreads+dcci) 即可.
    op.Process();
}

}  // namespace ascend_kernel
