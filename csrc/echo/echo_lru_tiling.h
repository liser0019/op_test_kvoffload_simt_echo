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
// [线程数消融] 可用 CMake 缓存变量 ECHO_LRU_THREADS 覆盖 (build.sh --threads=N 或
//   环境变量 ECHO_LRU_THREADS), CMake 经 -DECHO_LRU_THREAD_NUM_OVERRIDE=N 同时注入
//   kernel 与 plugin 两处 (必须同值: plugin 用该常量做 <<<>>> 的 blockDim).
//   约束: 32 的倍数 (warp 大小, 见下方 static_assert)、>= 64 (覆盖
//   ARGTOPK_GROUP_THREADS 的并行直方图切片)、<= 1024 (AIV launch 上限).
//   用法: bash build.sh --threads=512 → 产物在 build_t512/ (默认 1024 在 build/).
//   [性能对比] 默认设 1024 (与对照算子同规模); 消融矩阵见 README "线程数 / blockNum / batch 消融".
#ifdef ECHO_LRU_THREAD_NUM_OVERRIDE
constexpr uint32_t ECHO_LRU_THREAD_NUM = ECHO_LRU_THREAD_NUM_OVERRIDE;
#else
constexpr uint32_t ECHO_LRU_THREAD_NUM = 1024;
#endif
static_assert(ECHO_LRU_THREAD_NUM % 32u == 0u, "ECHO_LRU_THREAD_NUM must be a multiple of warp size 32");
static_assert(ECHO_LRU_THREAD_NUM >= 64u, "ECHO_LRU_THREAD_NUM must cover ARGTOPK_GROUP_THREADS");
static_assert(ECHO_LRU_THREAD_NUM <= 1024u, "ECHO_LRU_THREAD_NUM exceeds AIV launch bounds");

// ---- argtopk 缓冲上界 (host/device 共享) ----
// [Perf 2026-10-13] OpArgtopkRow 改为 4 趟 MSB-first 基数选择 (radix select):
//   key 打包 + 前缀桶直方图在 UB (单大 __ubuf__ 数组分段, 见 EchoLruKernel::Init),
//   不再做全量 scatter 排序, GM 中不再需要 ping-pong key/slot 段与 per-thread 直方图段.
//   ARGTOPK_GROUP_THREADS 个线程各写自己独占的 hist[256] UB 切片 (纯写无争用, 无 atomic),
//   归并到 globalCnt[256] 后由 t0 逐字节收敛阈值 T 与配额 quota, 最后按阈值筛出 freeSize 个槽.
//   host/torch 侧 CHECK(B<=ARGTOPK_B_MAX).
constexpr uint32_t ARGTOPK_B_MAX = 4096;          // B 上界 (per-request KV block 数 / buffer 容量)
constexpr uint32_t ARGTOPK_NPAD_MAX = 128;        // UB 占位数组大小 (GM radix 不读, 保持小值防 trap)
constexpr uint32_t ARGTOPK_GM_NPAD_MAX = 8192;    // [诊断3] GM 调试 dump 区每核打包 srcKey 容量 (=next_pow2(B_MAX+1))
//   → 4*NpadMax = 32768 int32 = 128KB. probe_ub_size 实测 32768 元素 (128KB) + 1024 线程 PASS.
constexpr uint32_t ARGTOPK_UB_NPAD_MAX = 8192;    // = next_pow2(ARGTOPK_B_MAX+1), argtopk key 打包元素上界
constexpr uint32_t ARGTOPK_UB_ELEMS = 4u * ARGTOPK_UB_NPAD_MAX;  // 单大 __ubuf__ 数组总元素 (key/hist/globalCnt/seg/scalar 分段)
constexpr uint32_t ARGTOPK_GROUP_THREADS = 64;    // 并行直方图线程数 (各独占 hist[256] UB 切片, 无 atomic)

// ---- [Perf 分步打点] ECHO_PERF_STEP 编译宏打开时, 每核把 Process 各步的 clock() 周期数
//   写入 workspace 尾部 stepCycles[blockIdx.x * ECHO_PERF_NUM_STEPS + stepIdx].
//   非 spec 路径共 23 步 (索引 0..22, P0 删 OpNumMisses 子步、P2 删 OpFinalSlots、
//   P4 再删 OpFreeSize/OpFreeAvailAdd/OpAllocAvailSub 并省 2 个无 RAW 的 SyncGrid 后);
//   specEnabled=1 时 Step0.5 在索引 4 之后额外插入 8 步, 共 31 步.
//   步名对照见 echo_perf_step.py 的 ECHO_PERF_STEP_NAMES.
constexpr uint32_t ECHO_PERF_NUM_STEPS = 31;
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
    uint32_t argtopkGroupThreads;
};

}  // namespace ascend_kernel

#endif  // ECHO_LRU_TILING_H
