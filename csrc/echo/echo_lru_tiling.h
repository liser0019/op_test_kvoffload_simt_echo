// echo_lru 纯 SIMT 版本 — tiling 结构 (Plan B, 纯 SIMT, 全元素分配 + grid_group::sync).
//
// 与 hybrid 版差异:
//   - 排布改为全元素 grid-stride (blockNum ≤ AIV 物理核, 见 grid_group::sync 约束),
//     元素级 step 由全核协作扫 R*cols; 行内串行子步 (cumsum / argtopk) 仍按行做,
//     行间由各 block 分摊 (blockNum≥R 时一行一核, 否则一核多行).
//   - step 间用 this_grid()->sync() + asc_dcci_entire 保证跨核写后读一致
//     (hybrid 无此能力, 是 stale pri 的根因; 纯 SIMT 解决之).
//   - tiling 仍保留 rowsPerCore/tailRows 供按行串行子步划行; 元素级 step 不用它们.

#ifndef ECHO_LRU_TILING_H
#define ECHO_LRU_TILING_H

#include <cstdint>

namespace ascend_kernel {

// 纯 SIMT 每核启动线程数. __launch_bounds__ 与 host <<<grid, block, dynUB, stream>>>
// 的 block 维同源. kernel 内部全部 blockDim.x 自适应, 改此值即改每核线程数.
// [性能对比] 设 1024 (与对照算子同规模). 旧值 512 注释保留便于回退:
// constexpr uint32_t ECHO_LRU_THREAD_NUM = 512;
constexpr uint32_t ECHO_LRU_THREAD_NUM = 1024;

// ---- argtopk 缓冲上界 (host/device 共享) ----
// [Step2] OpArgtopkRow 用 4-pass 8-bit LSD 基数排序: ping-pong keyPriA/SlotA + keyPriB/SlotB
//   + **per-thread 局部直方图** 全放 GM workspace. histogram 并行: ARGTOPK_HIST_THREADS 个线程各
//   写自己独占的 count[256] GM 切片 (纯写无争用), dcci 后 t0 串行归并+prefix (无 atomic).
//   pack/clear/prefix/scatter 仍 t0 串行保 LSD 稳定性. Npad=next_pow2(B+1) 决定 GM 段大小,
//   可达 4096 不受 UB 布局约束, 不 trap. host/torch 侧 CHECK(B<=ARGTOPK_B_MAX).
// ⚠ atomic 路线已弃: asc_atomic_add 写后设备端立即读不可靠 (dcci 不排空 atomic 写缓冲, 见
//   OpArgtopkRow 注释); UB/GM atomic 均在 len=262144 spec 路径非确定丢增量 (87/98/115 错运行间变).
//   per-thread 局部直方图纯写 → dcci → t0 读, 已验证可靠 (类 OpResetFree 多线程纯写 GM 后 dcci 读).
// ARGTOPK_NPAD_MAX 仅 __ubuf__ 占位数组大小 (GM radix 不读, 保持小值防 1024 线程 UB trap).
// 旧值 (B<=64): ARGTOPK_B_MAX=64, ARGTOPK_NPAD_MAX=128, 注释保留便于回退.
constexpr uint32_t ARGTOPK_B_MAX = 4096;          // B 上界 (per-request KV block 数 / buffer 容量)
constexpr uint32_t ARGTOPK_NPAD_MAX = 128;        // UB 占位数组大小 (GM radix 不读, 保持小值防 trap)
constexpr uint32_t ARGTOPK_GM_NPAD_MAX = 8192;    // [诊断3] GM 调试 dump 区每核打包 srcKey 容量 (=next_pow2(B_MAX+1))
// [UB 攻关 2026-09-30] argtopk ping-pong key/slot 搬顶层单大 __ubuf__ 数组的编译期容量.
//   NpadMax = next_pow2(ARGTOPK_B_MAX+1) = next_pow2(4097) = 8192. 4 段 (keyPriA/SlotA/keyPriB/SlotB)
//   → 4*NpadMax = 32768 int32 = 128KB. probe_ub_size 实测 32768 元素 (128KB) + 1024 线程 PASS.
constexpr uint32_t ARGTOPK_UB_NPAD_MAX = 8192;    // = next_pow2(ARGTOPK_B_MAX+1), argtopk 单趟排序元素上界
constexpr uint32_t ARGTOPK_UB_ELEMS = 4u * ARGTOPK_UB_NPAD_MAX;  // 单大 __ubuf__ 数组总元素 (4 段 ping-pong)
constexpr uint32_t ARGTOPK_HIST_THREADS = 64;     // 并行直方图线程数 (各独占 count[256] GM 切片, 无 atomic)
constexpr uint32_t ARGTOPK_SCAT_THREADS = 64;     // 并行 scatter 线程数 (各独占 localCnt[256] GM 切片, 纯写无争用)

struct EchoLruTiling {
    // ---- 排布 ----
    uint32_t blockNum;        // ≤ AIV 物理核数 (grid_group::sync 硬约束, 否则卡死)
    uint32_t rowsPerCore;     // 按行子步: 每 core 处理的 row 数
    uint32_t tailRows;        // 按行子步: 最后一个 core 的 row 数
    uint32_t numTokens;       // 本步活跃 row 数 R

    // ---- 形状参数 ----
    uint32_t topkBufferB;     // B
    uint32_t freeStackSize;   // B + min(topkDim,B) (free 栈余量 ≥ 最大 release 量 numMisses≤effTopk≤B)
    uint32_t maxModelLenP1;   // max_model_len + 1
    uint32_t topkDim;
    uint32_t effTopk;         // = min(topkDim, B) (kEff 的正确语义: 防 buffer 超订, 非硬编码 16)
    uint32_t blockSize;
    uint32_t blockTableCols;
    uint32_t kDim;
    uint32_t vDim;
    uint32_t kvHeadNum;
    uint32_t specEnabled;     // 1=投机解码 (存在 spl<maxModelLenP1 的行, 跑 Step0.5 驱逐后缀);
                              // 0=非 spec (前缀全长 spl==maxModelLenP1, Step0.5 整段跳过).
                              // host/torch 侧按 min(spl)<maxModelLenP1 判定. 默认 0 (非 spec).
    uint32_t argtopkDebugRow; // [诊断3] 需 dump pass0 归并后直方图的行号; 0xFFFFFFFF=关闭.
};

}  // namespace ascend_kernel

#endif  // ECHO_LRU_TILING_H
