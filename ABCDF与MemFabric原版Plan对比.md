# ABCDF 与 MemFabric 原版 Plan：以 kernel profile 为准

## 对比范围与来源

本分支 `feat/abcdf-memfabric-plan-profile` 基于用户指定的 `feat/simt-plan-micro-variants`，提交 `9cc99d5`。

- **ABCDF**：直接调用现有 `plan_abcdf`，计算代码、1024 线程配置、tiling 和编译选项均保留，原 catalog 的全部哈希仍校验通过。
- **MemFabric**：从用户提供的 `memfabric-hybrid_kvoffload-release_kv_miss-src-acc_offload-csrc-operators.zip` 提取 `acc_offload_lru_compact.cpp/.h`，两个文件逐字保存于 `csrc/memfabric_plan/upstream/`。压缩包和成员文件 SHA256 记录在 `memfabric_plan_manifest.json`。
- MemFabric 调用原 `OffloadOpsLruCompact` → `OffloadLruCompactOps` → `LruResidentCompactKernel`；保留原固定 **8 个 AIV block**，32 行时每核依次处理 4 行。ABCDF 沿用其原有分核方式。
- 不编译或调用 `ComputeLruResidentAddrs`、地址列表生成、`SparseCopy` 或 KV payload 搬运。Plan 内部用于读写元数据的 UB/GM DataCopy 保留，这是原算法的一部分。

原版 kernel/launch 零修改。新增的 `csrc/memfabric_plan/plugin.cpp` 仅提供 PyTorch 校验、持有 Tensor 的按值捕获和与现有 SIMT 相同的 `RunOpApi` 入队方式。没有把原版算法改写成 Python 或另一种 kernel。

Release 构建两路均为 `-O3`。原版 AIV 使用 SDK 默认的同步/缓存编译策略，原框架的 SIMT 专有编译选项仅用于 SIMT。

## 共同支持的输入域

两路公共输出是：

```text
持久状态：last_req_ids, slot_to_token, lru_slots
输出：current_slots, miss_count, miss_tokens, miss_slots
```

原版 Plan 没有 SIMT 的 `visible_seq_lens`、`block_table`、`token_to_req` 和部分活跃行参数，因此本次性能对比限定：

1. 每行一个 request，行与 request 恒等映射，全部行活跃。
2. query 合法且互不重复，resident token 不重复，LRU 是合法槽排列。
3. query 在可见长度内，源页合法，`topk<=capacity`。
4. stablePrefix、request reset、miss 分配和 LRU 更新仍按原语义执行；连续正确性审计覆盖后缀失效与 reset。

`compare_memfabric_plan.validate_case()` 会拒绝不满足条件的用例。不能用本次结果宣称两路对非法来源、请求重排、重复 resident、部分行或槽不足等扩展场景完全等价。

默认性能尺寸为 `rows=32, topk=2048, capacity=4096, max_token=262144`，场景与原框架相同：冷状态、75% 命中、全命中、满缓存且 75% 命中。默认随机打乱初始 LRU，种子固定为 2026；两路用同一输入数组和初始历史。

## 状态恢复与 MemFabric 的 epoch

每次测量前，两路都恢复同一公共持久状态，恢复完成后再下发目标 kernel。输出和工作状态各自独立，不互相复用。

MemFabric 的标记工作区是原实现所需的：

```text
token_mark: int32 [8,max_token]
token_pos:  int32 [8,max_token]
epochs:     int32 [8]
```

在每个场景开始时全部置零，然后按原算法跨调用推进 epoch，不把这些工作区当作公共缓存状态回滚。**不能只恢复 epochs 而留下旧 mark，否则旧 token 可能被误认成当前命中。**

审计阶段会污染输出与 token_pos；计时前预热和记录期间保持原工作区续用。每次读回正确性结果时还校验各核 epoch 推进次数，防止 scratch 被错误重置。本次短跑/正式采集远低于 epoch 清零阈值；很长的采集若触发阈值，检查会拒绝静默混入周期性全表清空的结果。

生产尺寸下原版 UB 元数据缓冲总量为 `6*8192 + 4*16384 + 4096 + 64 = 118848` 字节。接入层按原 Init 的对齐公式检查 120 KiB 的保守测试预算，超出预算直接拒绝。

## 上机操作

### 1. 下载与构建

在匹配的 torch/torch_npu 环境中：

```bash
git clone --branch feat/abcdf-memfabric-plan-profile --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git
cd op_test_kvoffload_simt_echo
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --soc=ascend950
```

CANN 路径按实际安装位置调整。共享库仍为 `build/libsimt_plan_variants_ops.so`，新增原版 Plan 和源码身份接口。每次改源码都要重新编译；运行时同时检查 Plan catalog 和 MemFabric 的原文件/接入层身份。

### 2. 一条命令：先连续正确性审计，再抓两路 profile

```bash
unset ASCEND_LAUNCH_BLOCKING
MEMFABRIC_PROFILE_OUTPUT=/dev/shm/abcdf_memfabric_profile \
  bash run_abcdf_memfabric.sh 1 2>&1 | tee abcdf_memfabric_profile.log
```

`1` 是逻辑卡号，按实际环境调整。若当前 shell 没有启用 `pipefail`，`tee` 后的退出码不能直接代表测试脚本成功；要可靠检查可先执行 `set -o pipefail`。

审计覆盖小尺寸、尾核分行以及生产尺寸，共 12 组初始状态，每组连续 4 步：初始输入、同 request 后缀失效、query 改写、request reset。后续步骤直接使用前一步 NPU 写出的公共状态和原版 scratch，CPU golden 只用于比较，不写回设备状态。

### 3. 分开执行

```bash
python3 -X faulthandler compare_memfabric_plan.py --device 1

python3 -X faulthandler profile_plan_variants.py \
  --comparison-module compare_memfabric_plan.py \
  --device 1 --scenario all --implementations abcdf memfabric \
  --reference memfabric --shuffle-lru --warmup 10 --repeat 60 \
  --output /dev/shm/abcdf_memfabric_profile
```

只采 75% 命中场景时，将 `--scenario all` 改为 `--scenario warm_75pct`。

## 结果看哪里

复用现有 `torch_npu.profiler` 脚本，采集器 warmup 为 1 step，正式 active 为 repeat 个 step；每个 step 包含两路各一次调用，逐轮交换顺序。采集器预热不计入正式样本数。

成功采集会生成独立的 `capture_*`：

```text
capture_*/capture_metadata.json
capture_*/cold_reset/kernel_summary.csv
capture_*/warm_75pct/kernel_summary.csv
capture_*/warm_all_hit/kernel_summary.csv
capture_*/full_cache_75pct/kernel_summary.csv
capture_*/<场景>/**/kernel_details.csv
```

目标 kernel 名：

- ABCDF：`sparse_kv_plan_kernel_plan_abcdf`
- MemFabric：`OffloadLruCompactOps`

**以 `kernel_details.csv` 的设备 Duration(us) 为数据源，以 `kernel_summary.csv` 的 kernel P50/P95 为主要比较指标。** 每路每场景必须恰好 60 条；同名混合任务组、缺失 kernel、样本数不符、非正数或非有限耗时都会报错，不生成本场景成功摘要。

`TensorMove`、状态恢复、H2D 下发和其他任务不加入目标 kernel 的时间。它们可能仍在原始 trace 中，因为状态恢复发生在 profiler 进程内。采集进程还会执行计时外的前后结果检查。

加速比为：

```text
speedup_vs_reference_p50 = MemFabric kernel P50 / ABCDF kernel P50
```

大于 1 代表 ABCDF 更快。一次调用处理全部 32 行，不是单行耗时。不得把 Event 延迟或 CPU 模型时间混入该排名。

同样的公共输入不等于相同的内部资源配置：8 核与 SIMT 原核数、UB 数组与 hash/mark 工作区均保留各自原设计。该结果比较两种 Plan 实现，不代表单一编程模式的固有倍数。

## 本地验证与边界

本机没有 CANN、torch_npu 或 NPU，未执行真机编译、设备正确性或 profile 采集，也没有性能排名。

已完成的本地检查：

- 上游两个文件逐字哈希核对；原 ABCDF 及所有 micro-variant 的 catalog 哈希保持不变。
- 原版完整计算类通过同步 CPU UB/DMA shim 执行，AddressSanitizer/UBSan 下通过 64 项状态/oracle 检查，包括生产尺寸、1/7/9 行尾核、连续调用、后缀失效、query 改写、request reset 与 epoch 推进。
- 13 个张量与属性的实际 C++ schema 对接检查；真实 CPU Tensor 的 profiler 协议模拟，共 64 次调用，覆盖公共状态恢复、共享输入、epoch 续用、采集器预热排除、两路样本数、拷贝过滤、相对基准，以及故障时不得完成摘要。模拟计时全部在临时目录删除，不作为性能结果。
- MemFabric plugin 使用真实 Torch 头文件和 NPU API 占位头通过 C++ 语法检查。
- 临时 CANN/torch_npu 占位目录下完成 CMake 配置/生成，包含 18 路原 SIMT 和 1 路原版 Plan，共 19 条 kernel 构建命令；未实际编译或链接 NPU 共享库。
- 原 profile 的 8 项离线检查通过。

可复跑：

```bash
python3 compare_memfabric_plan.py --cpu-check
python3 checks/check_memfabric_plan.py
python3 checks/check_memfabric_profiling.py
python3 checks/check_plan_profiling.py
```

CPU shim 不验证原版的 DataCopy/DCache/流水同步和 NPU 地址空间。首次上机应先构建、运行连续正确性审计，成功后再取 profile。原版 kernel 若在目标 CANN 出错，应根据实际错误修复接入或明确建立修订版本，不能把改写后的算法当作压缩包原版。
