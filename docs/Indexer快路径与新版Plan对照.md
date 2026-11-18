# Indexer 快路径与新版 Plan 对照

默认测试 32 个请求、TopK 2048 和 256K 上下文。MTP3 为每请求 4 行和 8192 个共享缓存槽位，MTP7 为每请求 8 行和 16384 个槽位。

| 实现 | Indexer | Plan |
|---|---|---|
| `final4` | 原始打分、TopK 选择和自动 tiling。对完整的 2048 个选中位置在 UB 内执行一次无符号整数升序排序 | 独立 `joint_rankpipe_ac_512` |
| `split4` | 同事有序 TopK，固定 `S1Base=4` | 同一个 `joint_rankpipe_ac_512` |

`final4` 沿用原始 Indexer，在 32 请求的两档 MTP 下均选择 `S1Base=4`。排序复用已经结束使用的打分和 TopK scratch，直接从排序结果缓冲写出。短序列沿用 Indexer 原生的升序有效前缀和 `-1` 尾部。候选支持的是 Ascend 950 的 TND／PA_BSND、TopK 2048 链路。

Plan 直接消费 Indexer 的输出缓冲。两条 kernel 在同一个 current stream 连续下发，Plan 使用采集前创建的 prepared 上下文。CPU 比较原始 Indexer 的 token 集合和候选的有序输出，并逐样本校验 Plan 状态及保护区。校验在计时区间外执行。目标性能指标为 Indexer 首个设备组件开始到 Plan 结束的 span，报告同时保留 Indexer 和 Plan 的分段时间。两档都要求所有请求走联合规划。

## 上机命令

在当前仓库目录执行，使用带有 CANN、torch 和 torch_npu 的 Python 环境。

```bash
export QLI_FUSION_BUILD_DIR=/workspace/qli_fast_split_build
bash scripts/build_qli_fusion.sh --fast-split-only
bash build_joint_plan.sh --rankpipe-only
source "$QLI_FUSION_BUILD_DIR/env.sh"

bash scripts/test_qli_fast_split.sh --device 4 \
  --output /workspace/qli_fast_split_results \
  --temp-dir /workspace/qli_fast_split_tmp
```

若已有身份匹配的 Plan 库，可给采集命令传入 `--library`。追加 `--warmup 10 --repeat 20` 可正式复测，追加 `--implementations final4` 可单独验证主候选，原始 Indexer 仍参与采集前的集合校验。

## 真机复测

Ascend950DT、CANN 9.2.1、NPU1。固定 32 请求、256K、TopK 2048，预热 10 次，每路每工况 20 个正式样本。设备链路 span 单位为 μs。

| 工况 | split4 P50／P95 | final4 P50／P95 | P50 降幅 |
|---|---:|---:|---:|
| 冷 MTP3／8K | 693.861／698.837 | 621.428／622.734 | 10.4% |
| 热 MTP3／8K | 714.188／719.055 | 644.582／646.764 | 9.7% |
| 冷 MTP7／16K | 1386.131／1414.360 | 1287.010／1306.185 | 7.2% |
| 热 MTP7／16K | 1426.399／1450.158 | 1330.899／1419.161 | 6.7% |

两路均通过 TopK 和 Plan 校验，四个工况均为 32 组联合执行、零回退。两路独立 Plan 的 P50 相近，链路收益主要来自 Indexer。原始报告留在服务器 `/workspace/qli_fast_split_QhDp5h/confirm/report.md`。
