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
//   每核只处理自己的连续行段 [blockIdx.x*rowsPerCore, +rowCount), 行内 512 线程
//   grid-stride 扫列. 每核读/写的都是自己负责的行 → 无跨核写后读.
// 同步: 步间用块内 asc_syncthreads (解决行内跨线程 RAW, 如 OpProtect 以 topkDim 步长写
//   pri[row,s]、OpArgtopkRow 以 B+1 步长读 pri[row,i], 同地址被不同线程访问);
//   asc_dcci_entire 刷本核 D-cache 防单核内写后读 stale. 行间零依赖故无需跨核栅栏
//   (本 CANN 的 simt_api 头树亦无 grid_group / this_grid, 连官方样例都编不过).
// argtopk 还原真正 bitonic. 算法语义与 golden.py 1:1.

#include "echo_lru_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"     // asc_syncthreads
#include "simt_api/device_atomic_functions.h"   // asc_atomic_add
#include "simt_api/device_functions.h"          // asc_dcci_entire
// Path A does not use cooperative_groups; avoid an unnecessary CANN 9.2 dependency.

namespace ascend_kernel {

constexpr int32_t I32_MAX = 2147483647;

// ---- argtopk 缓冲 (定义见 echo_lru_tiling.h, host/device 共享) ----
// [GM radix] OpArgtopkRow 用 4-pass 8-bit LSD 基数排序 (稳定). 状态全在 GM workspace:
//   ping-pong 两端 keyPriA/SlotA + keyPriB/SlotB (各 [blockNum*Npad] int32) + 直方图
//   count[256]/核 ([blockNum*256] uint32), 每核独占段. 不放 UB (16KB + 1024 线程 trap).
//   Npad=next_pow2(B+1) 可达 4096, 不受 UB 布局约束. 稳定排序 + keySlot 初始化升序 → pri
//   平局自动保 slot 升序, 匹配 golden lexsort((slot,pri)), 无需 uint64 复合 key.
// __global__ 顶层 __ubuf__ argtopkKeyPri/Slot[ARGTOPK_NPAD_MAX=128] 仅占位 (防 UB 布局回退),
//   GM radix 不读它们. host/torch 侧 CHECK(B <= ARGTOPK_B_MAX).

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
        for (uint32_t j = threadIdx.x; j < topkDim; j += blockDim.x) {
            if (j >= effTopk) { curSlotsGm[rowOff + j] = 0; continue; }
            int32_t tok = posGm[rowOff + j];
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            int32_t slot = htdGm[htdRowOff + tokSafe];
            curSlotsGm[rowOff + j] = slot;
            if (slot != I32_MAX) asc_atomic_add(&numHitsGm[row], 1);
        }
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

// per-row 4-pass 8-bit LSD 基数排序 (全定长, 无递归, 无动态分配).
// 字典序 key = (pri asc, slot asc). 实现技巧: keySlot 初始化为 0..Npad-1 (升序), LSD 稳定
//   → pri 平局自动保 slot 升序, 故只需对 pri 排 4 趟 (8-bit/趟 × 4 = 32 位), 无需另排 slot.
//   这正好匹配 golden 的 np.lexsort((slot, pri)) (pri 主键, slot 次键升序).
//   dav-3510 不支持 uint64 值运算, 故不用 uint64 复合 key; 稳定排序 + 升序 payload 替代.
//   Npad = next_pow2(B+1) <= ARGTOPK_B_MAX+1. 排序后取前 freeSize 个 slot 写 freeIdxOut.
//
// [UB count Step1] ping-pong 缓冲 (keyPriA/SlotA, keyPriB/SlotB) 留 GM workspace (每核独占段),
//   直方图 count[256] 改放 UB 静态数组 (1KB, 256×uint32). keyPri/keySlot 仍留 GM: 16KB 静态
//   数组 + 1024 线程会 trap (布局冲突, 见 echo-lru-ub-large-array-1024thread-trap 记忆).
//   count 1KB 是否安全正是 Step1 验证目标 (128 int32=512B 已知安全, 256 uint32=1KB 待测;
//   若 trap 即回退 .bak_step0_serial). 真正的收益 vs bitonic: 仅 4 趟 × 三明治栅栏.
//   排序逻辑仍由 threadIdx.x==0 串行 (无竞态, 易调试), 但 GM 写后读仍用
//   "syncthreads + dcci_entire + syncthreads" 三明治 (单 dcci 在 dav-3510 不足, 见
//   GRID_GROUP_PITFALLS MODE 3; 串行版曾只用 dcci 仍 49 错, 疑即此因).
//   count 在 UB: 清零/prefix/scatter 仍 t0 单线程内访问 (无竞态);
//   **直方图阶段 [Step2] 已并行化** — 全员 grid-stride 扫 Npad, asc_atomic_add 到 UB count.
//   pack/clear/prefix/scatter 仍 t0 串行 (保 LSD 稳定性, 易调试; 并行 scatter 难保稳定, Step3 再攻).
//   freeIdxOut row stride = topkDim.
//
// [回退保留] 旧 GM bitonic 全排序实现已注释保留在下方, 便于回退对比, 勿直接复用.
//   Step1 回退点: op_kernel/echo_lru_kernel.asc.bak_step0_serial (count 在 GM 版).
__aicore__ inline void OpArgtopkRow(
    uint32_t B, uint32_t Npad, uint32_t topkDim,
    int32_t* priGm, uint32_t row, int32_t* freeIdxOut, int32_t freeSize,
    int32_t* keyPriA, int32_t* keySlotA,
    int32_t* keyPriB, int32_t* keySlotB,
    uint32_t* countBase,    // [Step2] per-thread 局部直方图基址 [HIST_THREADS*256] uint32 (GM)
    uint32_t* scatLocal,    // [Step3] per-thread scatter localCnt 基址 [SCAT_THREADS*256] uint32 (GM)
    uint32_t* dbgCount,     // [诊断3] 本核 pass0 归并后直方图 dump 目标 [256] (可为 nullptr)
    int32_t* dbgKey) {      // [诊断3] 本核打包 srcKey dump 目标 [Npad] (可为 nullptr)
    // [UB 攻关 2026-09-30] keyPriA/SlotA/keyPriB/SlotB 改传顶层单大 __ubuf__ 数组分段指针 (块内共享 UB).
    //   probe_ub_size 实测: 单 __ubuf__ 数组可 192KB, 4 独立 __ubuf__ 数组会别名. 故 ping-pong 用
    //   单大数组分段. UB 块内共享: syncthreads 即保证线程间可见, **key/slot 读写无需 dcci** (dcci 刷
    //   GM D-cache, UB 无关). count/scatLocal 仍在 GM, 其 dcci 三明治保留.
    // 排序: pack/clear/reduce+prefix/scatter 由 threadIdx.x==0 串行 (保 LSD 稳定性, 无竞态);
    //   **直方图阶段 [Step2] HIST_THREADS 线程并行**, 每线程写自己独占的 count[256] GM 切片
    //   (countBase + t*256) —— **纯写无争用, 无 atomic**. dcci 三明治后 t0 串行归并各切片 →
    //   全局 count (复用 countBase+0*256 即 t0 自己的切片当全局累加目标) → prefix → scatter.
    //   ⚠ atomic 路线已弃 (见旧注释). GM 写后读须 "syncthreads + dcci_entire + syncthreads" 三明治.
    //   pack/clear/reduce/prefix/scatter 仅 t0 干活; 前 HIST_THREADS 线程参与直方图.
    uint32_t t = threadIdx.x;
    uint32_t rowOff = row * (B + 1);
    // 打包: keyPriA=(pri 升序优先, 无效 slot 沉底 I32_MAX), keySlotA=slot id (升序).
    //   打包后 keyPriA ∈ [0, I32_MAX] 全非负 (pri<0 与 slot0 → I32_MAX) → 无需符号翻转,
    //   无符号 LSD 即正确 (I32_MAX=0x7FFFFFFF 最大, 沉底). fifo 值远小于 I32_MAX, 有效 slot 靠前.
    if (t == 0) {
        for (uint32_t i = 0; i < Npad; ++i) {
            if (i <= B) {
                int32_t p = priGm[rowOff + i];
                keyPriA[i]  = (p < 0 || i == 0) ? I32_MAX : p;   // slot 0 / pri<0 无效, 沉底
                keySlotA[i] = (int32_t)i;
            } else {
                keyPriA[i]  = I32_MAX;                            // Npad 填充位沉底
                keySlotA[i] = (int32_t)i;
            }
        }
    }
    asc_syncthreads();
    // [UB 攻关] keyPriA 在 UB (块内共享), syncthreads 已保证线程间可见, 无需 dcci (dcci 刷 GM D-cache).
    //   旧 GM 版此处 dcci_entire(keyPriA) 已去.
    // [诊断3] 打包后 keyPriA[0..Npad-1] dump (仅匹配调试行). = 直方图输入, host 据此算期望 pass0 直方图.
    if (t == 0 && dbgKey != nullptr) {
        for (uint32_t i = 0; i < Npad; ++i) dbgKey[i] = keyPriA[i];
    }
    // 4-pass 8-bit LSD 基数排序 (稳定). src/dst ping-pong (A<->B), 每趟:
    //   (1) HIST_THREADS 线程各清零自己独占切片 countBase[t*256..+255] (纯写无争用)
    //   (2) **并行直方图**: HIST_THREADS 线程各 grid-stride 扫 Npad, 写自己切片 countBase[t*256+byt]
    //       (纯写无争用, 无 atomic)  (3) t0 串行归并各切片 → 全局 count (countBase[0..255]) +
    //       exclusive prefix sum 得各桶起始偏移  (4) t0 稳定 scatter (slot 跟随 key, 同桶按出现序
    //       写入, 保上一趟相对序; 串行保 LSD 稳定性, 并行 scatter 难保稳定, Step3 再攻)
    //   (5) 三明治刷 D-cache + ping-pong. 4 趟后结果在 src (指向最后写入方).
    // [Step2] 仅直方图并行 (pack/clear/reduce/prefix/scatter 仍 t0 串行): Npad=4096 时 histogram
    //   从串行 4096 次 GM 读 → HIST_THREADS(64) 线程各 ~64 次, 加速. 无 atomic → 无丢增量风险.
    //   per-thread 切片纯写 → dcci → t0 读 可靠 (类 OpResetFree). count 复用: t0 切片 [0..255]
    //   归并后即全局 count, scatter 由此读偏移.
    int32_t* srcKey  = keyPriA;  int32_t* srcSlot = keySlotA;
    int32_t* dstKey  = keyPriB;  int32_t* dstSlot = keySlotB;
    uint32_t* count  = countBase;                                          // t0 切片 = 全局 count (归并目标)
    for (uint32_t pass = 0; pass < 4; ++pass) {
        uint32_t shift = pass * 8;
        // (1) 各 hist 线程清零自己切片 (纯写, 无争用). 线程 ≥ HIST_THREADS 不参与直方图.
        if (t < ARGTOPK_HIST_THREADS) {
            uint32_t* myCount = countBase + t * 256u;
            for (uint32_t b = 0; b < 256; ++b) myCount[b] = 0u;
        }
        asc_syncthreads();                                                 // 清零完成后才进入直方图 (同线程序内)
        // (2) 并行直方图: HIST_THREADS 线程各 grid-stride 扫 Npad, 写自己切片 (纯写无争用).
        if (t < ARGTOPK_HIST_THREADS) {
            uint32_t* myCount = countBase + t * 256u;
            for (uint32_t i = t; i < Npad; i += ARGTOPK_HIST_THREADS) {
                uint32_t v = (uint32_t)srcKey[i];
                myCount[(v >> shift) & 0xFFu] += 1u;
            }
        }
        // [Step2] histogram→reduce 三明治: 各切片纯写须全刷可见后 t0 归并才不漏.
        asc_syncthreads();
        if (t == 0) asc_dcci_entire((void*)countBase);
        asc_syncthreads();
        if (t == 0) {
            // (3) t0 归并各切片到 countBase[0..255] (自己的切片当全局), 再 exclusive prefix sum.
            for (uint32_t b = 0; b < 256; ++b) {
                uint32_t s = 0u;
                for (uint32_t th = 0; th < ARGTOPK_HIST_THREADS; ++th) {
                    s += countBase[th * 256u + b];
                }
                count[b] = s;                                              // 归并到 t0 切片 = 全局 count
            }
            // [诊断3] pass0 归并后 (prefix 前) 全局直方图 dump (仅匹配调试行). 期望 = 打包 srcKey
            //   低 8 位直方图; host 比对可定位 错在 histogram/reduce (count 错) 还是 scatter (count 对).
            if (pass == 0 && dbgCount != nullptr) {
                for (uint32_t b = 0; b < 256; ++b) dbgCount[b] = count[b];
            }
            uint32_t sum = 0u;
            for (uint32_t b = 0; b < 256; ++b) {
                uint32_t c = count[b];
                count[b] = sum;                                            // exclusive prefix (全局各桶起始偏移)
                sum += c;
            }
        }
        // (4) [Step3] 并行稳定 scatter. LSD 稳定性靠"同桶按全局出现序写入"; 并行下用分段+段内序实现:
        //   把 Npad 切 SCAT_THREADS 段, 每线程负责一段 [lo,hi). 三步:
        //   (a) 各线程正向扫自己段统计本段各桶计数 localCnt[t*256+byt] (纯写独占切片, 无争用);
        //   (b) t0 归并: segBase[seg][byt] = 全局 prefix[byt] + Σ_{s<seg} localCnt[s][byt], 写回
        //       scatLocal[seg*256+byt] (覆盖 localCnt, 统计完后只需基址);
        //   (c) 各线程再正向扫自己段, 写入位 = segBase[t*256+byt] + 段内该桶已见数 (边扫边自增).
        //   稳定性: 段内正向扫 → 段内同桶按出现序; segBase 含前面段该桶总数 → 段间按段序. 合起来 =
        //   全局按出现序, 与串行 scatter 逐元素 count[byt]++ 等价 (可证). 全程纯写无 atomic.
        // [Step3 调试·二分1] 只加 (a0)+(a) 多线程写 scatLocal (不 dcci/不加 b/不加 c 读).
        //   (a) 与 Step2 直方图写 countBase 同模式 (64 线程各写自己切片). PASS→(a) 无辜, bug 在 b/dcci;
        //   crash→scatLocal 区写入本身 fault (区分配/对齐问题).
        if (t < ARGTOPK_SCAT_THREADS) {
            uint32_t* myLocal = scatLocal + t * 256u;
            for (uint32_t b = 0; b < 256; ++b) myLocal[b] = 0u;            // (a0) 清零本段 localCnt
        }
        asc_syncthreads();
        if (t < ARGTOPK_SCAT_THREADS) {                                    // (a) 统计本段各桶数
            uint32_t* myLocal = scatLocal + t * 256u;
            uint32_t lo = t * Npad / ARGTOPK_SCAT_THREADS;
            uint32_t hi = (t + 1u) * Npad / ARGTOPK_SCAT_THREADS;
            for (uint32_t i = lo; i < hi; ++i) {
                uint32_t v = (uint32_t)srcKey[i];
                myLocal[(v >> shift) & 0xFFu] += 1u;
            }
        }
        // (b) 三明治: (a) 多线程写 scatLocal → t0 归并读, dcci 刷全核 D-cache 可见.
        asc_syncthreads();
        if (t == 0) asc_dcci_entire((void*)scatLocal);
        asc_syncthreads();
        if (t == 0) {                                                       // (b) t0 归并 localCnt → segBase
            // cum 搬 GM (countBase 切片1). ⚠ 不用栈 cum[256]: OpArgtopkRow 内 256 元素栈数组 →
            //   ASC 按函数最大栈帧给所有 1024 线程预留 → 1KB×1024=1MB 栈超限 → 溢出 device fault
            //   (实锤: 二分3 (b)+cum栈 crash, cum搬GM PASS; 与 localPos[256] 同源, dav-3510 栈预算紧).
            uint32_t* cum = countBase + 256u;
            for (uint32_t b = 0; b < 256; ++b) cum[b] = count[b];          // 起始 = 全局 exclusive prefix
            for (uint32_t seg = 0; seg < ARGTOPK_SCAT_THREADS; ++seg) {
                uint32_t* segBase = scatLocal + seg * 256u;
                for (uint32_t b = 0; b < 256; ++b) {
                    uint32_t lc = segBase[b];                              // 先读 localCnt[seg][b]
                    segBase[b] = cum[b];                                   // 覆盖为 segBase[seg][b]
                    cum[b] += lc;                                          // 累加本段计数供后续段
                }
            }
        }
        // (c) 三明治: (b) t0 写 scatLocal segBase → 多线程 scatter 读, dcci 刷全核 D-cache 可见.
        asc_syncthreads();
        if (t == 0) asc_dcci_entire((void*)scatLocal);
        asc_syncthreads();
        if (t < ARGTOPK_SCAT_THREADS) {                                    // (c) 各线程稳定写 dst
            uint32_t* myBase = scatLocal + t * 256u;                        // segBase[t][byt] (本段基址)
            // myPos 搬 GM (countBase 切片 t). ⚠ 不用栈 localPos[256]: 同 cum 栈压力 → device fault.
            //   时序安全: (b) 先用 cum(切片1), (c) 后用 myPos; (c) 各线程清零自己切片再写, 纯写无争用,
            //   同线程 WAW→RAW 不需 dcci. t0 myPos=切片0 会覆盖 count[0..255], 但 (c) 并行不读 count,
            //   下一趟 (1) 清零+(3) 归并会重写切片0, 无残留.
            uint32_t* myPos = countBase + t * 256u;
            for (uint32_t b = 0; b < 256; ++b) myPos[b] = 0u;
            uint32_t lo = t * Npad / ARGTOPK_SCAT_THREADS;
            uint32_t hi = (t + 1u) * Npad / ARGTOPK_SCAT_THREADS;
            for (uint32_t i = lo; i < hi; ++i) {
                uint32_t v = (uint32_t)srcKey[i];
                uint32_t byt = (v >> shift) & 0xFFu;
                uint32_t pos = myBase[byt] + myPos[byt];
                dstKey[pos]  = srcKey[i];
                dstSlot[pos] = srcSlot[i];
                myPos[byt] += 1u;
            }
        }
        // (5) [UB 攻关] dstKey/dstSlot 在 UB (块内共享), syncthreads 已保证下趟读可见, 无需 dcci.
        //   旧 GM 版此处 dcci_entire(dstKey) 已去 (UB 无 D-cache stale 问题).
        asc_syncthreads();
        // ping-pong 换向: 全员换 (非仅 t0). srcKey/dstKey 是每线程各自的局部指针, pass0 全员一致
        //   起点为 keyPriA, 换向确定 → 全员同步换即保持一致. 若只 t0 换, pass1+ 直方图线程 1..63 仍读
        //   keyPriA (初始打包) 而 t0 读 keyPriB (上趟输出) → 归并混搭两数组 → count 错 → 散列错序
        //   (实锤: 诊断3 pass0 count 对, 串行直方图全 PASS, 并行 len=262144 FAIL, len=64/mtp=0 PASS).
        {
            int32_t* tk = srcKey;  srcKey = dstKey;  dstKey = tk;
            int32_t* ts = srcSlot; srcSlot = dstSlot; dstSlot = ts;
        }
    }
    // 取前 freeSize 个 slot (pri 最小 = LRU = 首先驱逐) 写 freeIdxOut. 余位置 0 (下游按 freeSize 裁剪).
    //   freeSize <= topkDim (numMisses<=effTopk<=topkDim, avail>=0).
    if (t == 0) {
        uint32_t fs = freeSize < 0 ? 0 : (freeSize > (int32_t)topkDim ? topkDim : (uint32_t)freeSize);
        for (uint32_t i = 0; i < topkDim; ++i) {
            freeIdxOut[row * topkDim + i] = (i < fs) ? srcSlot[i] : 0;
        }
    }
    asc_syncthreads();
    if (t == 0) asc_dcci_entire((void*)freeIdxOut);  // 下游 OpFreeRelease 行内读 (后续 SyncGrid 亦刷, 双保险)
    asc_syncthreads();
}

// [回退保留] 旧 GM bitonic 全排序 (升序). 实测 B=64/topk=32 仍 73 错 (D-cache 刷新无效,
//   根因疑 bitonic 78 级 swap 顺序/步长难保证). 已被上方 4-pass LSD 基数排序取代. 保留备查.
// __aicore__ inline void OpArgtopkRow_BITONIC(
//     uint32_t B, uint32_t Npad, uint32_t topkDim,
//     int32_t* priGm, uint32_t row, int32_t* freeIdxOut, int32_t freeSize,
//     int32_t* keyPri, int32_t* keySlot) {
//     uint32_t t = threadIdx.x;
//     uint32_t tn = blockDim.x;
//     uint32_t rowOff = row * (B + 1);
//     // 打包: keyPri=(pri 升序优先, 无效 slot 沉底 I32_MAX), keySlot=slot id.
//     for (uint32_t i = t; i < Npad; i += tn) {
//         if (i <= B) {
//             int32_t p = priGm[rowOff + i];
//             keyPri[i] = (p < 0 || i == 0) ? I32_MAX : p;
//             keySlot[i] = (int32_t)i;
//         } else {
//             keyPri[i] = I32_MAX;
//             keySlot[i] = (int32_t)i;
//         }
//     }
//     asc_syncthreads();
//     if (threadIdx.x == 0) asc_dcci_entire((void*)keyPri);
//     asc_syncthreads();
//     // bitonic 全排序 (升序)
//     for (uint32_t s = 2; s <= Npad; s <<= 1) {
//         for (uint32_t sub = s; sub > 1; sub >>= 1) {
//             uint32_t half = sub >> 1;
//             for (uint32_t i = t; i < Npad; i += tn) {
//                 uint32_t chunk = i ^ half;
//                 if (chunk > i) {
//                     int32_t aPri = keyPri[i]; int32_t aSlot = keySlot[i];
//                     int32_t bPri = keyPri[chunk]; int32_t bSlot = keySlot[chunk];
//                     bool ascending = ((i & s) == 0);
//                     bool aGreater = (aPri > bPri) || ((aPri == bPri) && (aSlot > bSlot));
//                     bool swap = ascending ? aGreater : !aGreater;
//                     if (swap) {
//                         keyPri[i] = bPri;  keySlot[i] = bSlot;
//                         keyPri[chunk] = aPri; keySlot[chunk] = aSlot;
//                     }
//                 }
//             }
//             asc_syncthreads();
//             if (threadIdx.x == 0) asc_dcci_entire((void*)keyPri);
//             asc_syncthreads();
//         }
//     }
//     uint32_t fs = freeSize < 0 ? 0 : (freeSize > (int32_t)topkDim ? topkDim : (uint32_t)freeSize);
//     for (uint32_t i = t; i < topkDim; i += tn) {
//         freeIdxOut[row * topkDim + i] = (i < fs) ? keySlot[i] : 0;
//     }
//     asc_syncthreads();
//     if (threadIdx.x == 0) asc_dcci_entire((void*)keyPri);
//     asc_syncthreads();
// }

// [回退保留] partial top-k 插入排序版 (threadIdx.x==0 串行). kEff 删除前的过渡实现,
//   依赖 kEff 截断候选长度. kEff 已删后语义不再适用 (freeSize 可达 topkDim=2048, 插入
//   排序候选数组需 topkDim 长, 单核串行 B×topkDim 比较过慢). 保留备查, 勿直接复用.
// __aicore__ inline void OpArgtopkRow_PARTIAL(
//     uint32_t B, uint32_t kEff, uint32_t Npad,
//     int32_t* priGm, uint32_t row,
//     __ubuf__ int32_t* keyPri, __ubuf__ int32_t* keySlot, int32_t* freeIdxOut) { ... }

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
__aicore__ inline void OpMissCumsum(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* curSlotsGm, int32_t* posGm,
    int32_t* wsMissRankGm, int32_t* missPosGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
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
        // [UB 攻关] ping-pong key/slot 用顶层单大 __ubuf__ 数组分段 (每核自己的 UB, 无 myBlock 偏移).
        //   argtopkUb[0..npad)=keyPriA, [npad..2npad)=keySlotA, [2npad..3npad)=keyPriB, [3npad..4npad)=keySlotB.
        //   npad<=ARGTOPK_GM_NPAD_MAX=4096, 单大数组 4*4096=64KB (probe_ub_size 实测 PASS 无别名).
        argtopkKeyPriUb_  = argtopkUb;
        argtopkKeySlotUb_ = argtopkUb + npad;
        argtopkKeyPriBUb_  = argtopkUb + 2 * npad;
        argtopkKeySlotBUb_ = argtopkUb + 3 * npad;
        // [Step2] 4-pass LSD 基数排序缓冲 (全 GM): ping-pong 两端 (A/B) 各 keyPri+keySlot
        //   (4 个 [blockNum*Npad] int32 段) + per-thread 局部直方图 (blockNum*HIST_THREADS*256 uint32).
        //   dav-3510 不支持 64 位值运算, 全程 32 位寻址. ping-pong 不放 UB (16KB + 1024 线程 trap);
        //   count 用 per-thread 局部表 (纯写无争用) 替代 atomic (atomic 写后设备端读不可靠, 见
        //   OpArgtopkRow 注释). 布局与 host workspace_elems 一致: int32 区后 8B 对齐,
        //   4×[bn*Npad] int32, 再 [bn*HIST_THREADS*256] uint32. uint32 段与 int32 段同 4B 不破坏对齐.
        uint32_t bn = tiling_.blockNum;
        uint32_t elemOff = (o + 1u) & ~1u;  // 8 字节对齐 (2×int32)
        argtopkKeyPriGm_   = ws + elemOff;  elemOff += bn * npad;   // A keyPri
        argtopkKeySlotGm_  = ws + elemOff;  elemOff += bn * npad;   // A keySlot
        argtopkKeyPriBGm_  = ws + elemOff;  elemOff += bn * npad;   // B keyPri (ping-pong)
        argtopkKeySlotBGm_ = ws + elemOff;  elemOff += bn * npad;   // B keySlot (ping-pong)
        argtopkCountGm_    = (uint32_t*)(ws + elemOff);  elemOff += bn * ARGTOPK_HIST_THREADS * 256u;  // per-thread 局部直方图
        // [Step3] 并行 scatter per-thread 局部桶计数 localCnt[256] (各线程独占切片, 纯写无争用).
        //   scatter 两遍: (a) 各线程正向扫自己段统计本段各桶数 → localCnt; (b) t0 归并 localCnt 得各段
        //   基址 segBase; (c) 各线程再扫写 dst, 写入位 = segBase[seg][byt] + 段内该桶已见数. 全程纯写无 atomic.
        argtopkScatLocalGm_ = (uint32_t*)(ws + elemOff);  elemOff += bn * ARGTOPK_SCAT_THREADS * 256u;
        // [诊断3] 调试直方图 dump 区: 每核 [256 uint32 归并后 count] + [ARGTOPK_NPAD_MAX int32 打包 srcKey].
        //   仅当 tiling_.argtopkDebugRow != 0xFFFFFFFF 时 OpArgtopkRow 对匹配行写 pass0 归并后 (prefix 前)
        //   的全局 count + 打包 srcKey, host 读出与 golden 期望直方图比对, 定位 histogram/reduce/scatter 哪段错.
        argtopkDebugCountGm_ = (uint32_t*)(ws + elemOff);  elemOff += bn * 256u;
        argtopkDebugKeyGm_   = ws + elemOff;               elemOff += bn * ARGTOPK_GM_NPAD_MAX;
    }

    // [GM radix] argtopk 缓冲在 GM (ping-pong A/B + count), Process 不再接收
    //   UB 指针. Path A: 行间零依赖, 步间用块内 SyncGrid() (syncthreads+dcci), 不传 grid
    //   (跨核 grid.sync 在 echo_lru 内 trap, 见 SyncGrid 注释, 暂停攻关).
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
        OpResetRows(R, maxModelLenP1, rowsPerCore, tailRows, resetMaskGm_, htdGm_);
        OpResetDthPri(R, B, rowsPerCore, tailRows, resetMaskGm_, dthGm_, priGm_);
        OpResetFree(R, B, freeStackSize, rowsPerCore, tailRows, resetMaskGm_, freeGm_);
        OpResetAvailFifo(R, B, rowsPerCore, tailRows, resetMaskGm_, availGm_, fifoGm_);
        SyncGrid();

        // ---- Step 0.5: invalidate speculative suffix (spec 路径; 非 spec 跳过) ----
        // specEnabled=0 (非 spec, 前缀全长 spl==maxModelLenP1) 时 Step0.5 整段 no-op, 跳过 4 个 Op 与 4 个栅栏.
        // specEnabled=1 时驱逐投机后缀 slot 并回收到本行 free 栈 (动态 free/alloc, 图捕获禁区).
        if (tiling_.specEnabled) {
            OpInvalidateSpecSuffix(R, B, maxModelLenP1, rowsPerCore, tailRows, splGm_, dthGm_, priGm_, htdGm_, wsCumsumGm_);
            SyncGrid();
            OpSpecCumsum(R, B, rowsPerCore, tailRows, wsCumsumGm_, pushCountGm_);
            SyncGrid();
            OpSpecPushFree(R, B, freeStackSize, rowsPerCore, tailRows, wsCumsumGm_, availGm_, freeGm_);
            SyncGrid();
            OpSpecAvailAdd(R, rowsPerCore, tailRows, availGm_, pushCountGm_);
            SyncGrid();
        }

        // ---- Step 1: hit/miss ----
        OpHitMiss(R, topkDim, effTopk, maxModelLenP1, rowsPerCore, tailRows, htdGm_, posGm_, curSlotsGm_, numHitsGm_);
        SyncGrid();
        OpNumMisses(R, effTopk, rowsPerCore, tailRows, numHitsGm_, numMissesGm_);
        SyncGrid();

        // ---- Step 2: protect + fifo++ ----
        OpProtect(R, topkDim, effTopk, B, rowsPerCore, tailRows, curSlotsGm_, priGm_, fifoGm_);
        asc_syncthreads();  // ⚠ Bug2 fix: 所有线程完成全部行 OpProtect 的 fifo 读+pri 写后,
                            //   t0 才能 OpFifoInc 做 fifo+=1. 否则 t0 串行 +=1 追上并超过其他线程
                            //   散在 OpProtect 各行的进度 → 某行后到的线程读到 +1 后的 fifo →
                            //   pri 多 1 (诊断4: row143 slot3706 kernel=12 golden=11 diff+1) →
                            //   argtopk pack srcKey 错 → freeIdx 顺序错 → 全链错. 排序方案不变.
        OpFifoInc(R, rowsPerCore, tailRows, fifoGm_);
        SyncGrid();  // pri 行内写 → step3 argtopk 行内读 (行内跨线程 RAW)

        // ---- Step 3: free ----
        OpFreeSize(R, rowsPerCore, tailRows, numMissesGm_, availGm_, freeSizeGm_);
        SyncGrid();
        ArgtopkAllRows();   // [Step2] GM count 并行 histogram (UB atomic 非真原子, 已回 GM)
        SyncGrid();
        OpFreeRelease(R, topkDim, B, maxModelLenP1, rowsPerCore, tailRows, freeIdxGm_, freeSizeGm_, priGm_, dthGm_, htdGm_);
        SyncGrid();
        OpFreePush(R, topkDim, freeStackSize, rowsPerCore, tailRows, freeIdxGm_, freeSizeGm_, availGm_, freeGm_);
        SyncGrid();
        OpFreeAvailAdd(R, rowsPerCore, tailRows, availGm_, freeSizeGm_);
        SyncGrid();

        // ---- Step 4: alloc pop ----
        OpAllocPop(R, topkDim, freeStackSize, rowsPerCore, tailRows, numMissesGm_, availGm_, freeGm_, rawSlotsGm_);
        SyncGrid();
        OpAllocAvailSub(R, rowsPerCore, tailRows, availGm_, numMissesGm_);
        SyncGrid();

        // ---- Step 5: alloc priority + fifo++ ----
        OpAllocPriority(R, topkDim, B, rowsPerCore, tailRows, rawSlotsGm_, numMissesGm_, priGm_, fifoGm_);
        asc_syncthreads();  // ⚠ Bug2 fix (同 Step2): OpAllocPriority 全线程写 pri=fifo 完成后,
                            //   t0 才能 OpFifoInc fifo+=1, 防 pri 多 1 race.
        OpFifoInc(R, rowsPerCore, tailRows, fifoGm_);
        SyncGrid();

        // ---- Step 6: set_update ----
        OpMissCumsum(R, topkDim, effTopk, rowsPerCore, tailRows,
                     curSlotsGm_, posGm_, wsMissRankGm_, missHostPosGm_);
        SyncGrid();
        OpSetUpdate(R, topkDim, effTopk, B, maxModelLenP1, rowsPerCore, tailRows,
                    curSlotsGm_, wsMissRankGm_, missHostPosGm_, missAllocFlatGm_, missMaskGm_,
                    rawSlotsGm_, htdGm_, dthGm_);
        SyncGrid();

        // ---- Step 7: final slots ----
        OpFinalSlots(R, topkDim, effTopk, maxModelLenP1, rowsPerCore, tailRows, htdGm_, posGm_, curSlotsGm_);
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

    // [UB 攻关 2026-09-30] ping-pong key/slot 搬顶层单大 __ubuf__ 数组 (每核自己的 UB, 无 myBlock
    //   偏移); count/scatLocal/debug 仍留 GM (per-thread 切片, 需 myBlock 偏移每核独占).
    //   行串行处理, 每核复用自己的 UB key/slot 段 + GM count 段.
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
        uint32_t R = tiling_.numTokens;
        int32_t* keyPriA  = argtopkKeyPriUb_;     // [UB] 每核自己 UB, 无偏移
        int32_t* keySlotA = argtopkKeySlotUb_;
        int32_t* keyPriB  = argtopkKeyPriBUb_;
        int32_t* keySlotB = argtopkKeySlotBUb_;
        uint32_t* count   = argtopkCountGm_    + myBlock * (ARGTOPK_HIST_THREADS * 256u);  // [GM] per-thread 局部直方图基址
        uint32_t* scatLcl = argtopkScatLocalGm_ + myBlock * (ARGTOPK_SCAT_THREADS * 256u);  // [GM] scatter localCnt 基址
        // [诊断3] 本核调试 dump 槽 (匹配行才写, 否则 OpArgtopkRow 收 nullptr 跳过).
        uint32_t* dbgCount = argtopkDebugCountGm_ + myBlock * 256u;
        int32_t*  dbgKey   = argtopkDebugKeyGm_   + myBlock * ARGTOPK_GM_NPAD_MAX;
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t freeSize = freeSizeGm_[row];   // 取前 freeSize 个 (不再固定 kEff)
            bool dbg = (tiling_.argtopkDebugRow == row);
            OpArgtopkRow(B, Npad, topkDim, priGm_, row, freeIdxGm_, freeSize,
                         keyPriA, keySlotA, keyPriB, keySlotB, count, scatLcl,
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
    int32_t* argtopkKeyPriUb_;      // [UB 攻关] ping-pong A keyPri (顶层单大 __ubuf__ 数组分段, 块内共享)
    int32_t* argtopkKeySlotUb_;     // ping-pong A keySlot
    int32_t* argtopkKeyPriBUb_;     // ping-pong B keyPri
    int32_t* argtopkKeySlotBUb_;    // ping-pong B keySlot
    int32_t* argtopkKeyPriGm_;      // [保留] 旧 GM ping-pong 段 (UB 攻关后 kernel 不读, 仅留 layout 占位)
    int32_t* argtopkKeySlotGm_;     //   避免动 host workspace_elems 布局; count/scatLocal/debug 仍用 GM.
    int32_t* argtopkKeyPriBGm_;
    int32_t* argtopkKeySlotBGm_;
    uint32_t* argtopkCountGm_;      // [Step2] 4-pass LSD 直方图 count[256]/核 (GM, 并行 atomic_add 用)
    uint32_t* argtopkScatLocalGm_;  // [Step3] 并行 scatter per-thread localCnt[256] 切片 (GM, 纯写无争用)
    uint32_t* argtopkDebugCountGm_; // [诊断3] 每核 pass0 归并后直方图 dump [bn*256] uint32
    int32_t* argtopkDebugKeyGm_;    // [诊断3] 每核打包 srcKey dump [bn*ARGTOPK_NPAD_MAX] int32
    void* wsBase_;
    uint32_t argtopkNpad_;
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
    // [UB 攻关 2026-09-30] argtopk ping-pong key/slot 搬顶层单大 __ubuf__ 数组 (手动分段).
    //   probe_ub_size 实测: 单 __ubuf__ 数组可到 192KB (256KB 才 trap); 4 个独立 __ubuf__ 数组会
    //   别名互相覆盖 (读 A 得 D). 故用单大数组 argtopkUb[4*NpadMax] 分 4 段 (偏移 0/Npad/2Npad/3Npad)
    //   作 keyPriA/SlotA/keyPriB/SlotB. count/scatLocal/debug 仍留 GM (per-thread 切片, 跨核需 GM).
    //   每核 block 内共享自己的 UB (无需 myBlock 偏移). NpadMax=8192 (B_MAX=4096→next_pow2(4097)),
    //   4*8192=32768 元素=128KB (probe 实测 32768 PASS). ARGTOPK_UB_ELEMS 见 tiling.h.
    __ubuf__ int32_t argtopkUb[ARGTOPK_UB_ELEMS];
    EchoLruKernel op;
    op.Init(htd, dth, pri, free, avail, fifo, pos, spl, resetMask, curSlots,
            missHostPos, missAllocFlat, missMask, workspace, tiling, argtopkUb);
    // Path A: 不使用 grid_group (跨核 grid.sync 在 echo_lru 内 trap, 见 SyncGrid 注释, 暂停攻关).
    //   行间零依赖, 步间块内 SyncGrid() (syncthreads+dcci) 即可.
    op.Process();
}

}  // namespace ascend_kernel
