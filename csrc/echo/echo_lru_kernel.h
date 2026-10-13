// Adapted from peer_code/echo_lru_kernel.asc; computational stages retained.
// echo_lru 纯 SIMT 版本 (Path A: 按行分块) — LRU 8 步元数据 kernel.
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
// clock() (PerfBegin/PerfEnd 分步打点) 由 common_functions.h → utils/debug/asc_time.h 提供
// (3510 纯 SIMT 下返回 uint64_t 周期计数).
// Path A does not use cooperative_groups; avoid an unnecessary CANN 9.2 dependency.

namespace ascend_kernel {

constexpr int32_t I32_MAX = 2147483647;

// ---- argtopk 缓冲 (定义见 echo_lru_tiling.h, host/device 共享) ----
// [radix select] OpArgtopkRow 用 4 趟 MSB-first 基数选择: key/hist/globalCnt/scalar
//   全在单大 __ubuf__ 数组 argtopkUb 分段 (见 EchoLruKernel::Init), 无 GM ping-pong.
//   GM workspace 尾部仅 [诊断3] debug dump 区 (+ ECHO_PERF_STEP 分步周期区).
//   host/torch 侧 CHECK(B <= ARGTOPK_B_MAX).

// ===========================================================================
// 行段宏: 每核负责 [startRow, startRow+rowCount). R 以内才有效 (尾核 tailRows).
// ===========================================================================
// 元素级 step: 外层按核遍历自己行段, 内层 512 线程 grid-stride 扫该行 cols.
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

// ---- Step 1: hit/miss ----
__aicore__ inline void OpHitMiss(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t maxModelLenP1,
    uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* htdGm, int32_t* posGm, int32_t* curSlotsGm, int32_t* numHitsGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        uint32_t rowOff = row * topkDim;
        uint32_t htdRowOff = row * maxModelLenP1;
        // [Perf 2026-10-09] hit 计数改每线程本地累加, 行末一次 atomic (原逐 hit 对同一地址
        //   numHitsGm[row] 原子加, 每行最多 effTopk 次争用; 现降为 ≤ 每行活跃线程数 次).
        //   atomic 可加任意增量 (见 atomic_intrinsics 示例), 正确性不变 (加法可交换).
        uint32_t localHits = 0;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if (j >= effTopk) { curSlotsGm[rowOff + j] = 0; continue; }
            int32_t tok = posGm[rowOff + j];
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            int32_t slot = htdGm[htdRowOff + tokSafe];
            curSlotsGm[rowOff + j] = slot;
            if (slot != I32_MAX) localHits++;
        }
        if (localHits != 0) asc_atomic_add(&numHitsGm[row], (int32_t)localHits);
    }
}

__aicore__ inline void OpNumMisses(
    uint32_t R, uint32_t effTopk, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* numHitsGm, int32_t* numMissesGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t h = numHitsGm[row];
            int32_t nm = (int32_t)effTopk - h;
            numMissesGm[row] = nm < 0 ? 0 : nm;
        }
    }
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

// ---- Step 3: free (argtopk bitonic + release + push) ----

__aicore__ inline void OpFreeSize(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* numMissesGm, int32_t* availGm, int32_t* freeSizeGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t nm = numMissesGm[row];
            int32_t av = availGm[row];
            int32_t fs = nm - av;          // 真实需弹栈量, 不再被 kEff 截断 (kEff 已删)
            if (fs < 0) fs = 0;
            freeSizeGm[row] = fs;
        }
    }
}

// per-row 基数选择 (radix select, 4 趟 MSB-first, 全定长, 无递归, 无动态分配).
// 目标: 从打包 key 中选出 pri 最小的 freeSize 个槽 (平局按槽号升序), 写 freeIdxOut.
//   key[i] = (i==0 || i>B || pri<0) ? I32_MAX : pri  → 全非负, 无效/填充沉底.
// 每趟 (pass 0..3, shift 从高字节到低字节):
//   (1) G=ARGTOPK_GROUP_THREADS 个线程各清零自己独占的 hist[t*256] UB 切片 (纯写无争用);
//   (2) 并行直方图: pass 0 统计全部 key, pass>0 只统计 (v >> (shift+8)) == prefix 的候选;
//   (3) 按桶并行归并到 globalCnt[256];
//   (4) t0 找到第一个 cum+c >= remaining 的桶 b, 更新 scalar[0]=阈值前缀 T,
//       scalar[1]=remaining-cum (桶内还需的配额 quota), scalar[2]=新 prefix, scalar[3]=完成标志.
//   不变量: |{key < T}| + quota == fsClamp; 全程 UB + asc_syncthreads, 无 dcci/无 atomic.
// 输出 (t0): 先按槽号升序写全部 key < T 的槽, 再按槽号升序补 quota 个 key == T 的槽,
//   尾部补 0 (下游按 freeSize 裁剪). key==T 平局取最小槽号, 与稳定排序的 (pri, slot)
//   字典序一致; key<T 组内部按槽号序输出 —— 本基准的初始 pri 满足 pri[slot]=slot
//   (见 compare.py make_echo_initial), 与 (pri, slot) 排序结果一致; 若未来输入存在
//   非单调 pri, 被淘汰"集合"仍正确, 仅 free 栈内的压栈顺序与全排序版不同.
// [回退保留] 旧 4-pass LSD 全排序与 GM bitonic 实现已删除, 见 git 历史 (9cb776e 及更早).
__aicore__ inline void OpArgtopkRow(
    uint32_t B, uint32_t Npad, uint32_t topkDim, uint32_t G,
    int32_t* priGm, uint32_t row, int32_t* freeIdxOut, int32_t freeSize,
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

    for (uint32_t i = t; i < Npad; i += blockDim.x) {
        if (i <= B) {
            int32_t p = priGm[rowOff + i];
            key[i] = (p < 0 || i == 0) ? I32_MAX : p;
        } else {
            key[i] = I32_MAX;
        }
    }
    asc_syncthreads();
    if (t == 0 && dbgKey != nullptr) {
        for (uint32_t i = 0; i < Npad; ++i) dbgKey[i] = key[i];
    }
    asc_syncthreads();
    
    if (t == 0) { scalar[0] = 0; scalar[1] = (int32_t)fsClamp; scalar[2] = 0; scalar[3] = 0; }
    asc_syncthreads();

    for (uint32_t pass = 0; pass < 4; ++pass) {
        uint32_t shift = (3u - pass) * 8u;
        if (active) {
            int32_t* myHist = hist + t * 256u;
            for (uint32_t b = 0; b < 256; ++b) myHist[b] = 0;
        }
        asc_syncthreads();

        if (active) {
            int32_t* myHist = hist + t * 256u;
            uint32_t prefix = (uint32_t)scalar[2];
            for (uint32_t i = t; i < Npad; i += ARGTOPK_GROUP_THREADS) {
                uint32_t v = (uint32_t)key[i];
                if (pass == 0u) {
                    myHist[(v >> shift) & 0xFFu] += 1;
                } else if ((v >> (shift + 8u)) == (prefix >> (shift + 8u))) {
                    // 候选过滤: v 的已收敛高字节 == prefix 的同高字节. prefix 是完整 32 位
                    //   路径 (b0<<24|b1<<16|...), 两侧都右移到 bit0 对齐后再比较.
                    myHist[(v >> shift) & 0xFFu] += 1;
                }
            }
        }

        asc_syncthreads();
        if (active) {
            for (uint32_t b = t; b < 256; b += ARGTOPK_GROUP_THREADS) {
                int32_t s = 0;
                for (uint32_t th = 0; th < G; ++th) s += hist[th * 256u + b];
                globalCnt[b] = s;
            }
        }
        asc_syncthreads();

        if (t == 0) {
            if (pass == 0 && dbgCount != nullptr) {
                for (uint32_t b = 0; b < 256; ++b) dbgCount[b] = (uint32_t)globalCnt[b];
            }
            uint32_t remaining = (uint32_t)scalar[1];
            uint32_t cum = 0u;
            uint32_t b = 0u;
            for (; b < 256; ++b) {
                uint32_t c = (uint32_t)globalCnt[b];
                if (cum + c >= remaining) break;
                cum += c;
            }

            uint32_t prefix = (uint32_t)scalar[2];
            scalar[0] = (int32_t)(prefix | (b << shift));
            scalar[1] = (int32_t)(remaining - cum);
            scalar[2] = (int32_t)(prefix | (b << shift));
            scalar[3] = (remaining - cum == 0u) ? 1 : 0;
        }
        asc_syncthreads();
        if (scalar[3] != 0) break;
        
    }
    
    uint32_t T = (uint32_t)scalar[0];
    uint32_t quota = (uint32_t)scalar[1];

    if (t == 0) {
        uint32_t w = 0u;
        for (uint32_t i = 0; i < Npad; ++i) {
            uint32_t v = (uint32_t)key[i];
            if (v < T) {
                freeIdxOut[row * topkDim + w] = (int32_t)i;
                ++w;
            }
        }
        uint32_t q = 0u;
        for (uint32_t i = 0; i < Npad && q < quota; ++i) {
            uint32_t v = (uint32_t)key[i];

            if (v == T) {
                freeIdxOut[row * topkDim + w] = (int32_t)i;
                ++w;
                ++q;
            }
        }
        for (uint32_t i = w; i < topkDim; ++i) freeIdxOut[row * topkDim + i] = 0;
        asc_dcci_entire((void*)freeIdxOut);
    }
    asc_syncthreads();
    (void)segLocal;  // 预留分段 scatter 缓冲, 基数选择路径未使用
    (void)segBase;   // 同上
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

__aicore__ inline void OpFreeAvailAdd(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* availGm, int32_t* freeSizeGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            availGm[row] = availGm[row] + freeSizeGm[row];
        }
    }
}

// ---- Step 4: alloc pop ----
__aicore__ inline void OpAllocPop(
    uint32_t R, uint32_t topkDim, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* numMissesGm, int32_t* availGm,
    int32_t* freeGm, int32_t* rawSlotsGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t nm = numMissesGm[row];
        int32_t avail = availGm[row];
        uint32_t rowOff = row * topkDim;
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            int32_t slot;
            if (j < (uint32_t)nm) {
                int32_t readIdx = avail - nm + (int32_t)j;
                if (readIdx < 0) readIdx = 0;
                if (readIdx >= (int32_t)freeStackSize) readIdx = (int32_t)freeStackSize - 1;
                slot = freeGm[freeRowOff + readIdx];
            } else {
                slot = 0;
            }
            rawSlotsGm[rowOff + j] = slot;
        }
    }
}

__aicore__ inline void OpAllocAvailSub(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* availGm, int32_t* numMissesGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            availGm[row] = availGm[row] - numMissesGm[row];
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
// [Perf 2026-10-13] miss 前缀和改并行 Hillis-Steele 扫描: 每线程处理 p0=t 与 p1=t+Bm
//   (Bm=blockDim.x, 两元素/线程), bufA/bufB 在 argtopk UB 内 ping-pong (argtopk 已完成,
//   UB 可复用). 条件 topkDim <= 2*Bm && 2*topkDim <= ARGTOPK_UB_NPAD_MAX 不满足时回退
//   t0 串行累加 (大 topkDim 或 UB 不足). 语义与串行版逐字段一致: wsMissRank = miss 的
//   exclusive 前缀秩, missPos = miss ? pos : 0.
__aicore__ inline void OpMissCumsum(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* curSlotsGm, int32_t* posGm,
    int32_t* wsMissRankGm, int32_t* missPosGm,
    int32_t* scanUb) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    uint32_t t = threadIdx.x;
    const uint32_t Bm = blockDim.x;

    if (topkDim <= 2u * Bm && 2u * topkDim <= ARGTOPK_UB_NPAD_MAX) {
        int32_t* bufA = scanUb;
        int32_t* bufB = scanUb + topkDim;
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
                bufA[p0] = ind0;
            }
            int32_t ind1 = 0;
            if (p1 < topkDim) {
                bool miss1 = (p1 < effTopk) && (curSlotsGm[rowOff + p1] == I32_MAX);
                missPosGm[rowOff + p1] = miss1 ? posGm[rowOff + p1] : 0;
                ind1 = miss1 ? 1 : 0;
                bufA[p1] = ind1;
            }
            asc_syncthreads();

            int32_t* src = bufA;
            int32_t* dst  = bufB;
            for (uint32_t offset = 1u; offset < topkDim; offset <<= 1) {
                if (p0 < topkDim) {
                    int32_t v0 = (p0 >= offset) ? src[p0 - offset] : 0;
                    dst[p0] = src[p0] + v0;
                }
                if (p1 < topkDim) {
                    int32_t v1 = (p1 >= offset) ? src[p1 - offset] : 0;
                    dst[p1] = src[p1] + v1;
                }
                asc_syncthreads();
                int32_t* tmp = src;
                src = dst;
                dst = tmp;
            }

            if (p0 < topkDim) {
                int32_t exc0 = src[p0] - ind0;
                wsMissRankGm[rowOff + p0] = (ind0 != 0) ? exc0 : 0;
            }
            if (p1 < topkDim) {
                int32_t exc1 = src[p1] - ind1;
                wsMissRankGm[rowOff + p1] = (ind1 != 0) ? exc1 : 0;
            }
            asc_syncthreads();
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
            } else {
                missAllocFlatGm[rowOff + j] = 0;
                missMaskGm[rowOff + j] = 0;
            }
        }
    }
}

// ---- Step 7: final slots ----
__aicore__ inline void OpFinalSlots(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t maxModelLenP1,
    uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* htdGm, int32_t* posGm, int32_t* curSlotsGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        uint32_t rowOff = row * topkDim;
        uint32_t htdRowOff = row * maxModelLenP1;
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if (j >= effTopk) { curSlotsGm[rowOff + j] = 0; continue; }
            int32_t tok = posGm[rowOff + j];
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            curSlotsGm[rowOff + j] = htdGm[htdRowOff + tokSafe];
        }
    }
}

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
        numHitsGm_    = ws + o; o += R;
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

        OpFillZero(R, rowsPerCore, tailRows, numHitsGm_);
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

        // ---- Step 1: hit/miss ----
        PerfBegin(); OpHitMiss(R, topkDim, effTopk, maxModelLenP1, rowsPerCore, tailRows, htdGm_, posGm_, curSlotsGm_, numHitsGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();
        PerfBegin(); OpNumMisses(R, effTopk, rowsPerCore, tailRows, numHitsGm_, numMissesGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();

        // ---- Step 2: protect + fifo++ ----
        PerfBegin(); OpProtect(R, topkDim, effTopk, B, rowsPerCore, tailRows, curSlotsGm_, priGm_, fifoGm_); PerfEnd();
        PerfBegin(); asc_syncthreads(); PerfEnd();  // ⚠ Bug2 fix: 所有线程完成全部行 OpProtect 的 fifo 读+pri 写后,
                            //   t0 才能 OpFifoInc 做 fifo+=1. 否则 t0 串行 +=1 追上并超过其他线程
                            //   散在 OpProtect 各行的进度 → 某行后到的线程读到 +1 后的 fifo →
                            //   pri 多 1 (诊断4: row143 slot3706 kernel=12 golden=11 diff+1) →
                            //   argtopk pack srcKey 错 → freeIdx 顺序错 → 全链错. 排序方案不变.
        PerfBegin(); OpFifoInc(R, rowsPerCore, tailRows, fifoGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd(); // pri 行内写 → step3 argtopk 行内读 (行内跨线程 RAW)

        // ---- Step 3: free ----
        PerfBegin(); OpFreeSize(R, rowsPerCore, tailRows, numMissesGm_, availGm_, freeSizeGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();
        PerfBegin(); ArgtopkAllRows(); PerfEnd();   // [Step2] GM count 并行 histogram (UB atomic 非真原子, 已回 GM)
        PerfBegin(); SyncGrid(); PerfEnd();
        PerfBegin(); OpFreeRelease(R, topkDim, B, maxModelLenP1, rowsPerCore, tailRows, freeIdxGm_, freeSizeGm_, priGm_, dthGm_, htdGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();
        PerfBegin(); OpFreePush(R, topkDim, freeStackSize, rowsPerCore, tailRows, freeIdxGm_, freeSizeGm_, availGm_, freeGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();
        PerfBegin(); OpFreeAvailAdd(R, rowsPerCore, tailRows, availGm_, freeSizeGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();

        // ---- Step 4: alloc pop ----
        PerfBegin(); OpAllocPop(R, topkDim, freeStackSize, rowsPerCore, tailRows, numMissesGm_, availGm_, freeGm_, rawSlotsGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();
        PerfBegin(); OpAllocAvailSub(R, rowsPerCore, tailRows, availGm_, numMissesGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();

        // ---- Step 5: alloc priority + fifo++ ----
        PerfBegin(); OpAllocPriority(R, topkDim, B, rowsPerCore, tailRows, rawSlotsGm_, numMissesGm_, priGm_, fifoGm_); PerfEnd();
        PerfBegin(); asc_syncthreads(); PerfEnd();  // ⚠ Bug2 fix (同 Step2): OpAllocPriority 全线程写 pri=fifo 完成后,
                            //   t0 才能 OpFifoInc fifo+=1, 防 pri 多 1 race.
        PerfBegin(); OpFifoInc(R, rowsPerCore, tailRows, fifoGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();

        // ---- Step 6: set_update ----
        PerfBegin(); OpMissCumsum(R, topkDim, effTopk, rowsPerCore, tailRows,
                     curSlotsGm_, posGm_, wsMissRankGm_, missHostPosGm_, argtopkKeyUb_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();
        PerfBegin(); OpSetUpdate(R, topkDim, effTopk, B, maxModelLenP1, rowsPerCore, tailRows,
                    curSlotsGm_, wsMissRankGm_, missHostPosGm_, missAllocFlatGm_, missMaskGm_,
                    rawSlotsGm_, htdGm_, dthGm_); PerfEnd();
        PerfBegin(); SyncGrid(); PerfEnd();

        // ---- Step 7: final slots ----
        PerfBegin(); OpFinalSlots(R, topkDim, effTopk, maxModelLenP1, rowsPerCore, tailRows, htdGm_, posGm_, curSlotsGm_); PerfEnd();
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
            bool dbg = (tiling_.argtopkDebugRow == row);
            OpArgtopkRow(B, Npad, topkDim, G, priGm_, row, freeIdxGm_, freeSize,
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
