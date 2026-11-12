# MTP7：中间去重与融合消费统计三路对照

本轮基于 `947e35e` 中冻结的 `joint_mtp7_reuse_parallel_512`。新增两条独立候选，不组合上轮安全前缀或 16 位映射。当前仅完成 CPU 模型与框架检查，**未执行 CANN/Bisheng 编译、NPU 正确性或性能测试**。

## 1. 本轮测哪三路

| 实现 | 编号 | 改动 |
|---|---:|---|
| `joint_mtp7_reuse_parallel_512` | 196 | 冻结参考：两组四路归并 → 最终两路归并 → 并行消费统计 → SIMD 去重 |
| `joint_mtp7_mid_unique_512` | 203 | 两个四路中间前缀先分别 SIMD 去重，再进行最终两路硬件归并；恢复全部原 query 的映射与消费量 |
| `joint_mtp7_fused_count_512` | 204 | 保留参考版的归并输入与输出，将八行消费统计合入已有 SIMD 去重 VF |

均保留 512 SIMT 线程、32 位 query→UID 映射、每路最多 612 条 packet 记录，以及生产尺寸动态 UB **181024 字节**。只测 Plan，包含其全部归并、去重、规划和写回，不包含 Transfer/SFA。

改动仅在八行组、当前有效源数大于四的两级归并路径启用。MTP0/1/3、八行中只剩四条或更少有效源时，仍执行参考版的对应算法；仍须复测，因为代码组织可能改变编译资源。冻结版预处理后的计算体与冷路径指纹继续接受回归检查；重新构建的设备二进制不能预先假定时延不变。

## 2. MID：为什么不能直接丢掉重复记录

例如两个四路中间结果：

```text
前四行：token [10,10,10,20,20]，payload 是五个不同的原 query ID
后四行：token [10,10,30]

先分别去重： [10,20] + [10,30]
最终归并：   [10,10,20,30]
最终 unique：[10,20,30]
```

最终两路归并处理的记录数可以减少，但原来八个 query 位置仍要得到各自的 `current_slots`。因此本实现维护：

```text
原 query → 四路中间 UID → 最终 union UID → resident/miss 槽位
```

中间 UID 使用 query 映射中的临时负值 `-(midUID+2)`，不会写入持久状态或对外输出。最终归并未消费的中间 UID 对应的 query 恢复为 `-1`，下一包重新装入。消费游标按**已消费的原 query 数量**推进，绝不能按中间 unique 数量推进。

跨包重复仍通过原来的 `lastToken` 与累计 unique 数量处理。容量超限在持久状态写回前进入原有逐行回退；没有修改回退语义、线程 1 发布规避或 miss 尾部约定。

### UB 生命周期

`C=chunk<=612`，以下偏移单位为 int32 元素，均相对原 packet 工作区：

| 阶段 | 有效缓冲 |
|---|---|
| 两次四路归并后 | rawA `[16C,24C)`；rawB `[0,8C)` |
| SIMD 分别去重 | compactA `[24C,32C)`；compactB `[32C,40C)` |
| rawA 已读完后 | 原 rawA 区清为 `-1`，作为 midUID→unionUID 表 |
| 最终两路 MrgSort | 读两段 compact；输出到 `[0,16C)` |
| SIMD 最终去重 | 写 union unique 与 midUID→unionUID |
| SIMT 恢复映射 | 读取原始 query 的临时中间编号，恢复最终编号并汇总八行消费量；warp 部分和复用 `[8C,16C)` |

没有新增全局缓存、设备分配或额外 UB 申请。输入区变成输出区、rawA 变成映射表之前，分别保留原阶段交接和必要的 VF 内存屏障。

这条候选增加了中间压缩、临时映射恢复和一次清零。减少第二层归并数据量不保证减少 MrgSort 调用次数，也不保证整体更快。

## 3. FUSED：一次读取兼做去重与统计

参考版两步都读取最终归并记录中的 query ID：

```text
SIMT：遍历 query ID → 统计每行消费量 → 更新游标
SIMD：读取 score/query ID → 去重编号 → 写 query→UID
```

204 在 SIMD 去重的每批 64 条记录中，同时累计 query ID 落入八行区间的数量。用八个累计区间计数避免额外的动态行号除法；包末归约并做相邻差分，得到每行原 query 消费量，更新原游标与 `more` 标志。

每条有效原始记录都计数，包括重复 token；无效尾 lane 不计数。`Reg::Reduce<SUM>` 只读取 lane0 的有效结果，再广播，未假定其他 lane 的内容。此路径不再调用独立 `EightParallelProgressVf`。

代价是多了 SIMD 比较、累加和包末归约，可能增加寄存器压力。因此它与 MID 分开编译、独立比较，没有把收益当作已知事实。

## 4. 代码阅读位置

1. `csrc/joint_variants/mtp7_merge_unique.h`：本轮新增的两个 SIMD 算法及 MID 映射恢复。
2. `csrc/joint_variants/mtp8_plan_impl.h`：硬件归并控制与上述 helper 的接入位置。
3. `joint_variants.py`、`checks/generate_joint_variants.py`：三路选择器、线程配置与生成入口。
4. `run_mtp8.py` 中 `merge_unique` suite：构建、一次门禁、一次采集与报告。
5. `checks/check_mtp7_merge_unique.py`：实际 C++ 控制流的 CPU 模型检查，包含局部原语与跨 warp 检查。

没有重写硬件 MrgSort；仍通过公开 AscendC API 使用它。新增 SIMD 归约签名依据 [CANN 9.1 Reduce 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reduction_compute/Reduce.md)。CPU API 模型仅检查源码语义和边界，不能验证设备指令、寄存器 spill 或真实流水。

## 5. 服务器一轮短测

选择空闲卡。下面明确使用逻辑 NPU2；如生产机 CANN 路径不同，请调整 `CANN_HOME`。不要在旧构建目录覆盖原失败/成功结果。

```bash
set -eo pipefail
BRANCH=feat/mtp7-mid-unique-fused-count
DEVICE=2
RUN=$(mktemp -d /workspace/single_op_test/mtp7_merge_unique_XXXXXX)
git clone --single-branch -b "$BRANCH" \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"

CANN_HOME=/usr/local/Ascend/cann-9.1.0
source "$CANN_HOME/set_env.sh"
export ASCEND_CANN_PACKAGE_HOME="$CANN_HOME"
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

bash scripts/test_mtp7_merge_unique.sh \
  --device "$DEVICE" --requests 28 \
  --mtp 0 1 3 7 --scenarios cold_reset full_cache_75pct \
  --warmup 2 --repeat 5 --regression-repeat 5 \
  --output "$RUN/results" --temp-dir "$RUN/tmp"

cat "$RUN/results/report.md"
echo "结果目录：$RUN/results"
```

此入口自动构建，仅选择三条联合变体，保持现有 `-O3`，执行一次专项门禁与一次采集。正式目标样本为 `3实现×2工况×4种MTP×5次=120`，不叠加初筛、50 次复测或历史候选矩阵。门禁还包含少量重复 token、可见性、容量边界、5/6 有效源及实际连续状态用例，这些不是性能样本。

输出/临时目录在 NPU 初始化前进行空间与 inode 检查；空间不足会失败，不静默切换位置。可显式把目录设到 `/dev/shm`，但该位置是临时存储，需自行转存。源码可以留在其他分区。

单独构建可执行：

```bash
bash build_joint_plan.sh --soc=ascend950 --merge-unique-only
# 构建完成后，在上面的测试命令加 --skip-build。
```

如短测通过且卡空闲，可复用同一库与本轮门禁进行 20 样本稳定性复测：

```bash
bash scripts/test_mtp7_merge_unique.sh \
  --skip-build --device "$DEVICE" --requests 28 \
  --mtp 0 1 3 7 --scenarios cold_reset full_cache_75pct \
  --warmup 10 --repeat 20 --regression-repeat 5 \
  --correctness-report "$RUN/results/gate.json" \
  --output "$RUN/confirm20" --temp-dir "$RUN/tmp"
```

仅在源码、库、设备及精确输入身份匹配时允许复用门禁；每个正式样本仍做完整输入、输出、持久状态和保护区检查。两轮数据分别报告，不混合统计。

## 6. 如何判断结果

- `report.md`：三路 P50/P95、样本数、相对冻结版的耗时比、共享/回退数、实际 descriptor 数、CSV Block Num。
- `capture_metadata.json`、子采集目录的 `data_index.json`：精确 kernel 名称、单位、样本顺序、原始 CSV 与源码/库/设备身份；也保留各实现相对自身 MTP0 的比值。
- `RUN_STATUS.json`：构建、门禁、采集及解析墙钟时间。`COMPLETE.json`、报告和退出码 0 齐全才算整轮成功。
- 失败立即停止并保留输入/设备现场及日志，不输出成功性能结论。

三路轮换执行，按精确目标 kernel 的设备 Duration 统计，状态恢复与校验不计入目标。保留 ACTIVE 开始后 1 秒宿主等待的采集侧规避；它不属于 kernel 时间，也不能解释正确性。

MID 是否减少时间，应查看实际 MTP7 数据；FUSED 是否有收益，同样不能仅凭“少一次 VF”判断。MTP0/1/3 用于检查接入/编译退化。本轮没有 ABCDF、同事版本或 Transfer 的正式采样，不能把本轮结果说成这些路径的同场比较。

## 7. 本地验证与局限

本地实际执行：两路各 139 项完整控制器/状态对照；MID 新增 18 项、FUSED 新增 100 项局部与跨 warp 检查；两路各复跑 45 项消费统计模型；69 项框架单元回归；17 个构建子集的 C++ dispatch 编译/链接检查；源码身份、Python/Bash 语法与 diff 检查。

使用真实候选 helper 和控制器源码、CPU SIMD/SIMT 语义替身及 UBSan。独立 golden、逐位置 currentSlots、LRU、miss、只读输入、保护区、连续状态及回退检查保留。局部归约模型特意污染非 lane0 输出，以检查错误的广播假设。

CPU 工作量观察：某个单请求、长重复 run 输入中，两路归并累计提交记录数由参考的 16393 降至 MID 的 1572；FUSED 仍为 16393。该观察仅在 CPU MrgSort 语义模型中计数，没有给设备增加打点或计数；不能当作生产吞吐、设备实际调用次数或性能结论。另一常规 overlap40 模型中三路的 MrgSort 调用次数仍均为 12。

同步扫描包含新增 helper、八行控制器、cold_finish、turbo_plan、sparse_kv_joint_plan 和入口 kernel。两份正则扫描均无候选；其图提取没有覆盖 Reg Gather/Scatter、模板展开和真实 VF 跨流水依赖，不构成同步正确性证明。首次命令的旧头文件路径不存在，已更正为实际 `cold_finish.h` 并重跑；未用失败扫描作为通过依据。

人工核对的主要交接：四路 MrgSort→SIMD 中间压缩；rawA 读完→复用映射表；最终 MrgSort→SIMD 编号；SIMD MID 映射→SIMT 恢复；SIMD 游标写回→控制器读 `more`。必须以服务器编译、原有严格正确性门禁和 profiler 对照继续验证。
