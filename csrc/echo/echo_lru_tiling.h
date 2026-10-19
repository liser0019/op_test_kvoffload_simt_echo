// echo_lru 纯 SIMT 版本 — tiling 结构 (Path A: 每核连续行段, 无跨核同步).
//
// [v2 维护序 2026-10-19] 淘汰路径状态从 pri 时间戳换成 lru 顺序数组
//   (秩→槽位); kernel 详见 echo_lru_kernel.h 头注释与 deep_dive §9.17.
//   排布不变: Path A 每核处理自己连续行段, 步间块内栅栏 (无 grid_group).
//
// 历史包袱: 下方头注释曾描述 "Plan B / grid_group::sync" 时代排布, 已废止 ——
//   以 kernel 实际代码 (Path A) 为准.

#ifndef ECHO_LRU_TILING_H
#define ECHO_LRU_TILING_H

#include <cstdint>

namespace ascend_kernel {

// 纯 SIMT 每核启动线程数上限. __launch_bounds__ 与 host <<<grid, block, dynUB, stream>>>
// 的 block 维同源. kernel 内部全部 blockDim.x 自适应, 改此值即改每核线程数上限.
// [线程数消融] 可用 CMake 缓存变量 ECHO_LRU_THREADS 覆盖 (build.sh --threads=N 或
//   环境变量 ECHO_LRU_THREADS), CMake 经 -DECHO_LRU_THREAD_NUM_OVERRIDE=N 同时注入
//   kernel 与 plugin 两处 (必须同值: plugin 用该常量做 <<<>>> 的 blockDim clamp).
//   约束: 32 的倍数 (warp 大小, 见下方 static_assert)、>= 64 (plugin 自适应 blockDim
//   的下界)、<= 1024 (AIV launch 上限).
//   用法: bash build.sh --threads=512 → 产物在 build_t512/ (默认 1024 在 build/).
//   [性能对比] 默认设 1024; 消融矩阵见 README "线程数 / blockNum / batch 消融".
#ifdef ECHO_LRU_THREAD_NUM_OVERRIDE
constexpr uint32_t ECHO_LRU_THREAD_NUM = ECHO_LRU_THREAD_NUM_OVERRIDE;
#else
constexpr uint32_t ECHO_LRU_THREAD_NUM = 1024;
#endif
static_assert(ECHO_LRU_THREAD_NUM % 32u == 0u, "ECHO_LRU_THREAD_NUM must be a multiple of warp size 32");
static_assert(ECHO_LRU_THREAD_NUM >= 64u, "ECHO_LRU_THREAD_NUM must cover plugin blockDim floor");
static_assert(ECHO_LRU_THREAD_NUM <= 1024u, "ECHO_LRU_THREAD_NUM exceeds AIV launch bounds");

// ---- 缓冲上界 (host/device 共享) ----
// [v2] 基数选择已删除: ARGTOPK_UB_ELEMS 转为通用 UB 池容量 (scanScratch/flags/
//   aAt/lru/hExcl/aExcl/freeIdx 分段, 见 EchoLruKernel::Init; 基准形态峰值
//   128 + 5*(B+1) + rowsPerCore*topkDim ≈ 22.6K)。ARGTOPK_B_MAX 仍为 plugin
//   侧 B 上界 CHECK; 其余 ARGTOPK_* 常量为遗留无引用 (保留防外部引用断裂)。
constexpr uint32_t ARGTOPK_B_MAX = 4096;          // B 上界 (per-request KV block 数 / buffer 容量; plugin CHECK)
constexpr uint32_t ARGTOPK_NPAD_MAX = 128;        // [v2 遗留] 基数选择时代的占位常量, 无引用
constexpr uint32_t ARGTOPK_GM_NPAD_MAX = 8192;    // [v2 遗留] GM 调试 dump 区容量 (dump 区已随基数选择删除), 无引用
constexpr uint32_t ARGTOPK_UB_NPAD_MAX = 8192;    // = next_pow2(ARGTOPK_B_MAX+1), UB 池分段基准
constexpr uint32_t ARGTOPK_UB_ELEMS = 4u * ARGTOPK_UB_NPAD_MAX;  // 单大 __ubuf__ 数组总元素 (v2: 通用 UB 池)
constexpr uint32_t ARGTOPK_GROUP_THREADS = 64;    // [v2 遗留] 基数选择直方图线程数, 无引用

// ---- [Perf 分步打点] ECHO_PERF_STEP 编译宏打开时, 每核把 Process 各步的 clock() 周期数
//   写入 workspace 尾部 stepCycles[blockIdx.x * ECHO_PERF_NUM_STEPS + stepIdx].
//   [v2] 非 spec 路径共 18 步 (索引 0..17: Step0×5, Step1×2, Step2×4, Step3×2,
//   Step4×2, Step5×3 —— 相对 v1 的 20 步: 删 OpFifoInc/OpAllocPriority/Bug2
//   SyncBlock, 增 EvictScan/LruRebuild, 详见 deep_dive §9.17);
//   specEnabled=1 时 Step0.5 在索引 4 之后额外插入 8 步, 共 26 步.
//   步名对照见 echo_perf_step.py 的 ECHO_PERF_STEP_NAMES.
constexpr uint32_t ECHO_PERF_NUM_STEPS = 26;
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
