# SIMT 变体接入与性能对比

本分支 `feat/simt-variant-comparison` 从独立测试工程的 `9cb776e` 建立，保留原 SIMT 和作者新版 Echo，接入用户提供的 11 个 SIMT 版本。**默认共有 13 路，四种场景，生产尺寸不变。**

## 1. 各个名字代表什么

| CLI / JSON 名称 | 计算内容 | capacity 分类数组数 |
|---|---|---:|
| `simt` | 原测试工程的 SIMT，来源基线 f51dc07 | 2 |
| `echo` | 作者 2026-10-09 更新的 Echo，含小尺寸暂存修复 | 不适用 |
| `00_baseline` | var_simt_ops 的 clean_simt 基线 cd84b83 | 2 |
| `01_a_hit_short_circuit` | A：hit 跳过第二次源检查 | 2 |
| `02_b_parallel_scan` | B：并行二级 scan，复用调用点的尾部同步 | 2 |
| `03_c_compact_partition` | C：evict/hit 从同一分类数组的两端写入 | 1 |
| `04_d_cold_start` | D：仅 request ID 更换时走冷路径 | 2 |
| `05_ab` | A+B | 2 |
| `06_abc` | A+B+C | 1 |
| `07_abcd` | A+B+C+D | 1 |
| `08_e_validity_markers` | E：用 currentSlots 临时负值缓存合法性 | 2 |
| `09_bce` | B+C+E | 1 |
| `10_bcde` | B+C+D+E | 1 |

保留 `simt` 和 `00_baseline` 两个对照，因为它们来自不同源码快照，不把它们自动当成同一个编译产物。各个优化的具体说明见对应目录的 README；这些 README 是原交付文档，所说的“同名注册”限制在本工程中已由独立符号解决。

## 2. 接入时改了什么

源码位于 `csrc/variants/<版本名>/`：

- `sparse_kv_plan_compute.h`：从原 `_apt.cpp` 开头到注册入口之前逐字提取，末尾只增加结束标记注释；包含全部行算法、scan、hash、VF，没有重写算法。
- `sparse_kv_plan_config.h`：原配置头逐字复制，设备线程仍为 2048。
- `sparse_kv_plan_launch.cpp`：kernel 入口保留原有执行流程；改名、去掉未使用的框架 workspace 参数、将 tiling 宏读取替换成逐字段 GM 读取，并增加直调 launch。VF 名通过宏机械改名，避免链接冲突。
- `sparse_kv_plan_plugin.cpp`：只设置算子名、launch 名、分类数组数和下发标签，然后编译共享的原 SIMT 注册层。

例如 `00_baseline` 的注册名为 `torch.ops.sparse_kv_plan_op.sparse_kv_plan_v00`，对应 `launch_sparse_kv_plan_v00` 和 `sparse_kv_plan_v00_kernel`。直到 `v10` 都有独立名字，因此可以同时加载并计时，不需要切换 OPP 安装。

全部 SIMT 共用相同的参数校验、POD tiling 布局、tiling 上传、按值捕获和 `RunOpApi` 下发方式。没有在计时内增加一个根据算法编号切换 kernel 的 C++ 分支。

`variants_manifest.json` 记录原基线提交、原 kernel 哈希、实际提取的计算头哈希和配置头哈希。`source_manifest.json` 则保留原工程来源及后续接入修改说明。

### 接入层类型名修复

`e0b4084` 的计算头保留了匿名命名空间中的 `SparseKvPlanTilingData`，直调 POD 又用了同名全局类型，入口会发生 C++ 名称歧义。现在直调 POD 和 11 个入口统一使用 `SparseKvPlanVariantTilingData`。算法头、配置头及其原有哈希保持不变。

`checks/check_variant_layouts.py` 已增加将实际上游结构体与入口声明一起编译的回归检查，覆盖全部 11 路；同时核对 host、上游及直调 POD 的大小、对齐和每个字段偏移。原有检查只对照字段和 host 布局，没有覆盖这次名称冲突。修复后这些检查及 832 次框架 CPU 模拟调用通过；仍未执行 CANN/NPU 编译。

## 3. 为什么 C 系列不能沿用旧 workspace

令 H 为 `max(32, next_pow2(2*topk))`：

```text
普通版本：row_elements = 3*H + 2*capacity + topk
C 系列： row_elements = 3*H +   capacity + topk
```

行临时区按 32 字节对齐，不超过 112 KiB 时放 UB；否则放 GM。动态 UB 还包括 544 字节 scan 区，启动语句真实传入 `tiling.localMemoryBytes`。

| topk / capacity | 普通版本行临时区 | C 系列行临时区 | 实际路径 |
|---|---:|---:|---|
| 2048 / 4096 | 88 KiB | 72 KiB | 都是 UB |
| 2048 / 8192 | 120 KiB | 88 KiB | 普通为 GM，C 为 UB |

host tiling、注册层 workspace 长度检查、Python 分配三个位置都按每版的真实数组数计算。GM 需要 `rows*row_elements` 个 int32，UB 路径只需要一个外部占位元素。

Echo 当前 capacity 上限为 4096，因此 8192 容量实验必须排除 Echo。CLI 会显式拒绝不支持的组合。

## 4. 性能输入与正确性如何保持一致

四个场景仍是 cold_reset、warm_75pct、warm_all_hit、full_cache_75pct。

- 所有实现共享同一 query 和只读输入；所有 SIMT 使用同一初始历史快照。
- 每路持久工作状态、输出、workspace 独立分配。
- 每次调用前恢复初始持久状态并等待恢复完成，恢复在计时区间外。
- 审计阶段污染输出和 scratch，执行后校验。
- 每个 SIMT 候选的 3 个持久字段及 4 个输出，逐字段精确对照同一个 SIMT 参考；Echo 仍对照自己的参考。
- 如果包含 Echo，测量后再逐个检查 SIMT 与 Echo 的共同查询映射、miss 数和淘汰 token 集合。
- 所有检查成功才写结果 JSON；失败不会发布本次性能结果。

`--cpu-check` 仅检查参考输入与布局，不执行候选 kernel。候选源码的 CPU 语义模型是下面单独的检查入口。

## 5. 计时口径与执行顺序

仍用同一 `measure_one()`：恢复、start Event、算子调用、end Event、等待结束。记录 Event 和 host 微秒样本，统计 P50/P95/min/max。

13 路按轮次轮转位置：第 0 轮从第 0 路开始，第 1 轮从第 1 路开始……每满 13 轮，每个实现都在每个位置出现一次。结果保存完整顺序。默认 50 轮，各位置次数最多相差一次；想让位置次数完全相等，可用 52 轮。只选两路时，轮转仍是原先的交替顺序。

加速比定义：

```text
speedup_vs_reference_event_p50 = 基准 Event P50 / 当前实现 Event P50
```

大于 1 表示当前实现的中位延迟更低。默认基准优先 `simt`，其次 `00_baseline`，否则第一路；可用 `--reference` 指定。

这些仍是公共接口 Event 延迟，包含 tiling 上传和下发间隙，不能称作纯 kernel 耗时。纯 kernel 比较需另行 profiler。接口计时期间不运行 CPU golden 或 D2H 校验。

## 6. 上机步骤

在之前已配好的 CANN / torch / torch_npu 环境中执行。新目录下载可以用：

```bash
git clone --branch feat/simt-variant-comparison --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git
cd op_test_kvoffload_simt_echo
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --soc=ascend950
```

CANN 路径与逻辑卡号按实际环境调整。每次换源码后重新编译，旧共享库不含新增符号。

### 全部 13 路：先审计、后正式计时

```bash
unset ASCEND_LAUNCH_BLOCKING
python3 -X faulthandler compare.py --device 1 \
  --warmup 2 --repeat 3 --verify-each-sample \
  --json results/variants_audit.json

python3 -X faulthandler compare.py --device 1 \
  --warmup 10 --repeat 50 --json results/variants_benchmark.json
```

默认尺寸仍为 `rows=32, topk=2048, capacity=4096, max_token=262144`。

### 只选 A/B/C/E、基线和 Echo

```bash
python3 compare.py --device 1 --implementations \
  simt echo 00_baseline 01_a_hit_short_circuit 02_b_parallel_scan \
  03_c_compact_partition 08_e_validity_markers \
  --warmup 10 --repeat 50 --json results/single_points.json
```

### 只看 C 导致的 GM → UB 路径变化

```bash
python3 compare.py --device 1 --capacity 8192 --implementations \
  00_baseline 03_c_compact_partition 06_abc 07_abcd 09_bce 10_bcde \
  --reference 00_baseline --warmup 2 --repeat 3 --verify-each-sample \
  --json results/c8192_audit.json
```

审计通过后，去掉 `--verify-each-sample` 并增加 warmup/repeat，再正式测量。

### 保留原来的两路比较

```bash
python3 compare.py --device 1 --implementations simt echo \
  --warmup 10 --repeat 50 --json results/simt_echo_only.json
```

## 7. 本地验证范围

本机没有 CANN/NPU。以下检查不产生 NPU 性能结论：

```bash
python3 compare.py --cpu-check
python3 checks/check_variant_layouts.py
python3 checks/simt_variants/run_cpu_checks.py
python3 checks/simt_variants/run_cpu_checks.py \
  --variants 03_c_compact_partition 07_abcd 10_bcde --large
# 最后一项需要可用的 CPU torch 和 NumPy：
python3 checks/check_variant_framework.py
```

已完成：

- 11 个版本的算法提取哈希、配置哈希、直调 ABI、2048 线程配置检查。
- 16 组实际 C++ host tiling 与 Python 的公式对照，包含两种布局的 UB/GM 边界。
- 11 个版本各 96 项独立 oracle/持久状态检查；含 B 的 6 个版本各 128 项生产配置 warp-total scan 模型检查。模型行线程为 64，生产源码仍为 2048。
- C、ABCD、BCDE 各 108 项含大尺寸的 oracle 检查，含 B 的两个组合另做 scan 检查。
- 13 个真实 CPU schema，四场景 832 次框架模拟调用；同输入指针、独立工作状态、重复恢复、轮转位置、样本数量、8192 容量下 GM/UB 分配与末样本故障拦截均通过。临时模拟计时不作为结果发布。
- 使用真实 CPU Torch 头文件和临时 torch_npu API 占位头检查基线、v00、v03 plugin 的 C++ 语法。
- 使用临时依赖占位目录完成 CMake 配置与生成，确认 13 条 kernel 构建命令（12 条 mixed SIMT、1 条 pure Echo）。未编译 NPU kernel，也未链接可运行共享库。

仍需服务器验证：CANN/bisheng 编译、NPU 地址空间、DCache 与同步、完整设备结果及正式性能。不要用本地 CPU 检查代替这些结论。
