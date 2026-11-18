# QLI 融合 Batch 对齐实验

固定 MTP3、TopK 2048、TopK 选择后的 4 个 query 行，比较 batch8 与满核 batch32。两个值为上限时分别测试 100K 和 256K 上下文。

| 实现 | QLI 分块 | 目的 |
|---|---|---|
| `peer` | 总 query 行不超过 64 时使用两行 | 同事固定提交的基线 |
| `peer4` | Plan 资源允许时固定使用四行，空间不足仍回退两行 | 验证 query 行与 request 对齐能否修复小 batch |

batch8、MTP3 共 32 条 query 行，原基线因此选择两行分块。四行候选可以让每个 request 的 4 行处于同一 QLI row block。batch32、MTP3 共 128 行，两路都选择四行，用于检查候选是否引入退化。

每路每工况默认预热 2 次、正式采样 5 次。正确性验证比较 TopK、Plan 状态、输出和保护区。正式报告使用设备 kernel Duration。初筛通过后应把 `--warmup 10 --repeat 20` 传入脚本复测。

```bash
export QLI_FUSION_BUILD_DIR=/workspace/qli_peer_aligned_build
bash scripts/build_qli_fusion.sh --peer-aligned-only
source "$QLI_FUSION_BUILD_DIR/env.sh"
bash scripts/test_qli_peer_aligned.sh \
  --device 2 \
  --output /workspace/qli_peer_aligned_results \
  --temp-dir /tmp/qli_peer_aligned_tmp \
  --warmup 2 --repeat 5 --regression-repeat 2
```

## Plan 替换候选

首轮固定四行的 `peer4` 在 batch8 上反而变慢，因此保留结果作反例。第二阶段保持同事原 QLI 自动 tiling 不变，只把内部 ABD512 Plan 替换为 rankpipe AC。该分支中的 `c` 因此代表“自动 QLI tiling + rankpipe AC”。

```bash
bash scripts/test_qli_auto_rankpipe.sh \
  --device 2 \
  --output /workspace/qli_auto_rankpipe_results \
  --temp-dir /tmp/qli_auto_rankpipe_tmp \
  --warmup 2 --repeat 5 --regression-repeat 2
```

## 通用 batch 测试

实际生产 batch 可能是 6、8、10 等小规模，候选不能在 tiling 或调度中写死 28/32。通用入口测试 batch 6、8、10，并保留 batch32 作为满 AI Core 参考：

```bash
bash scripts/test_qli_auto_rankpipe_general.sh \
  --device 2 \
  --output /workspace/qli_auto_rankpipe_general \
  --temp-dir /tmp/qli_auto_rankpipe_general \
  --warmup 2 --repeat 5 --regression-repeat 2
```

## 8K 四行分块的 32 字节修复

旧 `c` 在 batch32、MTP3 下会从 `S1Base=4` 回退到 2，原因是 8K Plan 为 123,680 B，与 QLI 四行资源合计 221,216 B，比 216 KiB 上限 221,184 B 多 32 B。

修复把 private 完成标志打包进已有 `ctx[47]` 的高 2 位，C 布局的 context 从 128 个字缩为 120 个字。8192 容量下 Plan 从 123,680 B 降为 123,648 B，正好的四行总预算为 221,184 B。A/B 布局继续使用完整 128 字 context。

`checks/test_qli_context_pack.py` 检查 C 和非 C 两种 context 大小，并验证生成头没有过期。

## 修复后正式复测

NPU4、CANN 9.2.1，预热 10 次、正式 20 样本。`c` 保持上游自动 QLI tiling，仅使用压缩后的 rankpipe AC context。

| Batch | 上下文 | 工况 | peer P50/P95 μs | c P50/P95 μs | c P50 变化 |
|---:|---:|---|---:|---:|---:|
| 8 | 100K | 冷 | 290.480/291.967 | 267.742/269.165 | -22.738 μs，-7.83% |
| 8 | 100K | 热 | 344.988/345.800 | 304.123/305.070 | -40.865 μs，-11.85% |
| 8 | 256K | 冷 | 627.256/628.161 | 607.812/609.218 | -19.444 μs，-3.10% |
| 8 | 256K | 热 | 684.558/685.166 | 640.978/642.209 | -43.580 μs，-6.37% |
| 32 | 100K | 冷 | 323.409/324.348 | 303.185/304.165 | -20.224 μs，-6.25% |
| 32 | 100K | 热 | 375.639/377.109 | 334.962/336.051 | -40.677 μs，-10.83% |
| 32 | 256K | 冷 | 706.458/708.255 | 688.050/689.637 | -18.408 μs，-2.61% |
| 32 | 256K | 热 | 764.371/766.160 | 723.578/724.572 | -40.793 μs，-5.34% |

两档均为完整联合执行和零回退。修复前的 batch32 退化已消除。
