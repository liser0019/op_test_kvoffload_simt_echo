# MTP3 保序 Indexer 与 Plan 的同组交接实验

同一个 AI Core 的两个 AIV 各产出两条 TopK。现有 `native_group` 用 GM 原子计数选择最后完成的一侧运行 Plan。`native_pair4` 在两侧完成 TopK 写出后用 mode 1 硬件信号交接，固定由奇数 AIV 运行 Plan。它继续直接读取自己最后一行的 UB 数据，其他三行仍从原 TopK GM 输出读取。AIC/AIV 的配对、S2 分核、保序 TopK、Plan 算法、工作区和外部接口沿用现有实现。

| 实现 | S1Base | 四行组的交接方式 |
|---|---|---|
| `ordered_full` | 独立保序 Indexer 的原生规则 | 后续独立四路整包 Plan，同一 stream |
| `native_group` | 总 query 行数 ≤64 时 2，否则 4 | 现有原子计数 |
| `native_group4` | 固定 4，Host 和 AIC/AIV 读取同一 tiling 值 | 现有原子计数 |
| `native_pair4` | 同 `native_group4` | 同一 AI Core 的双 AIV 硬件同步，固定 Plan 执行者 |

以 `native_pair4 / native_group4` 判断交接改动的净收益，以 `native_group4 / native_group` 判断固定四行的净收益。拆分方案用于判断最终链路收益。小 batch 强制四行有可能减少参与 Indexer 的核心数，需用数据判断。上下文 100K/256K，MTP3，TopK2048，容量8192，首行75%命中和请求重置分开测。

硬件交接只用于未分割 S2 的完整四行 tile，且这四行正好构成一个活跃的独立请求组。两侧根据同一 tile 边界和只读 Plan 元数据作出相同判断。两边的两个 MTE3 写出都完成后才能发出完成信号，各侧都消费一次信号。奇数 AIV 在 MTE3→Scalar 之后调用 Plan，Plan 的 Vector 写入完成后才复用 TopK arena。其他输入仍由现有发布逻辑处理。公开 TopK 输出保留原接口。

## 复现

使用 CANN 9.2.1 和独立虚拟环境。`PLAN_LIBRARY` 指向此分支通过现有 `build_joint_plan.sh --soc=ascend950 --mtp3-only` 生成的库。确认 `DEVICE` 空闲，`RUN` 为新建的持久化目录。

```bash
export QLI_FUSION_BUILD_DIR="$RUN/build"
bash scripts/build_qli_fusion.sh --native-affinity-only

MTP3_BATCHES='8 17 32' bash scripts/test_native_ordered_mtp3.sh \
  "$DEVICE" "$RUN/build" "$PLAN_LIBRARY" "$RUN/gate" \
  --implementations native_group4 native_pair4 --contexts 8 2048 8192 \
  --scenarios cold_reset --regression-repeat 2 --gate-only --prove-pair

MTP3_BATCHES='8 17 32' bash scripts/test_native_ordered_mtp3.sh \
  "$DEVICE" "$RUN/build" "$PLAN_LIBRARY" "$RUN/results" \
  --implementations ordered_full native_group native_group4 native_pair4 \
  --contexts 100000 262144 --warmup 2 --repeat 5 --regression-repeat 1
```

`--prove-pair` 仅供完整四行组的门禁使用。它把 `native_pair4` 的 GM 就绪计数预填为 16。所有组的 Plan 输出正确且计数维持 16，才能证明这些组实际走了硬件交接。正式采样保持正常的零初态，并按设备 span 比较 P50/P95。
