# MTP3 保序 Indexer 与 Plan 的同组交接实验

同一个 AI Core 的两个 AIV 各产出两条 TopK。现有 `native_group` 用 GM 原子计数选择最后完成的一侧运行 Plan。`native_pair4` 在两侧完成 TopK 写出后用 mode 1 硬件信号交接，固定由奇数 AIV 运行 Plan。它继续直接读取自己最后一行的 UB 数据，其他三行仍从原 TopK GM 输出读取。AIC/AIV 的配对、S2 分核、保序 TopK、Plan 算法、工作区和外部接口沿用现有实现。

| 实现 | S1Base | 四行组的交接方式 |
|---|---|---|
| `ordered_full` / `ordered_ac` | 独立保序 Indexer 的原生规则 | 256K 用四路整包 Plan，100K 用通用 rankpipe AC，同一 stream |
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
  "$DEVICE" "$RUN/build" "$PLAN_LIBRARY" "$RUN/results_256k" \
  --implementations ordered_full native_group native_group4 native_pair4 \
  --contexts 262144 --warmup 2 --repeat 5 --regression-repeat 1

MTP3_BATCHES='8 17 32' bash scripts/test_native_ordered_mtp3.sh \
  "$DEVICE" "$RUN/build" "$PLAN_LIBRARY" "$RUN/results_100k" \
  --implementations ordered_ac native_group native_group4 native_pair4 \
  --contexts 100000 --warmup 2 --repeat 5 --regression-repeat 1
```

`--prove-pair` 仅供完整四行组的门禁使用。它把 `native_pair4` 的 GM 就绪计数预填为 16。所有组的 Plan 输出正确且计数维持 16，才能证明这些组实际走了硬件交接。正式采样保持正常的零初态，并按设备 span 比较 P50/P95。

## NPU5 实测

Ascend950DT_9582、CANN 9.2.1。设备构建通过。短序列门禁覆盖 batch 8/17/32 和上下文 8/2048/8192，每工况重复两次。`native_pair4` 的每个请求组都通过 GM 计数污染验证。正式矩阵包含两种上下文、三档 batch、冷/热两种初态，每路每工况 5 个样本。所有样本通过 TopK、Plan、状态和保护区校验，各实现 descriptor 数一致，联合组数等于 batch，零回退。

以下是热缓存设备 span P50，单位 μs。

| 上下文 | Batch | 拆分 | 自动分块融合 | 四行 GM 发布 | 四行硬件交接 |
|---|---:|---:|---:|---:|---:|
| 100000 | 8 | 227.803 | 297.626 | 320.794 | 303.038 |
| 100000 | 17 | 281.769 | 316.420 | 321.743 | 304.585 |
| 100000 | 32 | 285.622 | 322.933 | 324.556 | 308.092 |
| 262144 | 8 | 491.827 | 633.838 | 700.597 | 663.427 |
| 262144 | 17 | 623.809 | 689.967 | 709.922 | 667.043 |
| 262144 | 32 | 633.795 | 696.879 | 712.810 | 673.970 |

硬件交接相对四行 GM 发布在 12 个工况的 P50 都下降，范围为 5.1%～6.6%。batch8 的自动两行 tiling 仍比四行硬件交接更快。热缓存六个工况都由拆分获得最低 span。

100K、batch17 冷启动中，拆分链路 busy P50 为 259.265 μs，span P50 为 391.030 μs，逐样本间隙的 P50 为 130.544 μs。硬件交接的 busy/span P50 均为 273.419 μs。这个工况的融合 span 优势来自拆分采样之间的空档。20 样本复测启动前因 NPU5 有其他任务退出。

静态同步扫描记录了 `TryPublishPair` 的 SYNC-08 候选，指向配对前的提前退出。代码以两侧相同的整 tile 边界、活跃行数和只读请求映射决定是否进配对路径，进入配对路径的两侧各有一次 Set 和 Wait。门禁覆盖完整四行组，尚未专项验证动态边界和多流行为。

服务器结果在 `/workspace/qli_affinity_cPem3T`。短序列门禁见 `gate/bs*/capture_metadata.json`，原始采集路径见 `results_100k/bs*/data_index.json` 和 `results_256k/bs*/data_index.json`。初次混合上下文的采集停在独立整包 Plan 的 256K 形状门禁，原始现场保存在 `results/`。两套正式采集分别运行，均完成并通过。

## 自适应交接 `native_pair_auto`

沿用原生 Indexer 的自动分块。当总 query 行数 ≤64，S1Base 为 2，同一个 AI Core 的两个 AIV 各产出一条 TopK。两条都写出后由奇数 AIV 批量发布两条，标准四行请求的 GM 原子增加次数由四次变为两次。当 S1Base 为 4 时，完整请求直接复用四行硬件交接。未分割 S2 的完整 tile 才能执行同组信号。两行 tile 的请求边界和活跃行数继续由 `PublishRows` 检查，发布不完整时沿用现有逐行入口。

生成源码去除算子命名差异后，`native_group` 和 `native_pair_auto` 的 Host、Cube、TopK 与 Plan 内容一致，差异集中在 Vector 发布与交接方法。`native_pair4` 用于对照固定四行的效果，拆分链路用于对比目标设备 span。

复用上面的构建环境。把构建选项换成 `--native-adaptive-only`，256K 和 100K 的采集实现分别换成 `ordered_full native_group native_pair4 native_pair_auto` 和 `ordered_ac native_group native_pair4 native_pair_auto`。本轮性能采集使用 `MTP3_BATCHES='8 16 17 32'`。

两行门禁用 batch 1/8/16 和 `native_group native_pair_auto`，四行门禁用 batch 17/32 和 `native_group native_pair4 native_pair_auto`。两者都验证 8/2048/8192 上下文，每组重复两次。`--prove-pair` 仅用于四行门禁。两行路径需要 GM 计数协调两个 AI Core，门禁使用正常的零初态。

### 自适应版实测

NPU5、Ascend950DT_9582、CANN 9.2.1。设备构建、两档门禁和 16 个冷/热工况均完成，每路每工况 5 个正式样本。四行门禁确认 `native_pair_auto` 与 `native_pair4` 的全部请求组实际走硬件交接。正式样本的 TopK 与 Plan 状态均正确，联合组数等于 batch，回退为零，各实现 descriptor 数一致。

下表为 256K 的热缓存设备 span P50，单位 μs。

| Batch | 拆分 `ordered_full` | 原自动融合 | 固定四行硬件交接 | 自适应硬件交接 |
|---:|---:|---:|---:|---:|
| 8 | 490.947 | 634.604 | 662.573 | 633.926 |
| 16 | 500.197 | 636.733 | 665.894 | 637.119 |
| 17 | 625.345 | 689.124 | 669.652 | 697.051 |
| 32 | 631.679 | 697.641 | 673.915 | 707.272 |

batch 8/16 的自适应版与原自动融合版相比，P50 变化在 0.6% 以内。所有 batch 17/32 工况中，固定四行版仍为最快的融合实现。256K 下自适应版比固定四行版慢 27.399–33.785 μs。两者均通过四行路径证明，且实际报告均为四行 tiling。FP8 E4M3 设备对象的整份 `.text` 从 `0x2170c0` 增至 `0x259cc0`。这可作为生成代码变化的线索，整份对象还包含其他 tiling key，具体运行阶段成本尚未测出。

拆分链路在全部 16 个工况中取得最低设备 span P50。部分 100K 工况出现较高的拆分 P95，五样本足以观察到尖刺，稳定性需单独复测。原始报告和 CSV 索引保存在 `/workspace/qli_adaptive_4LowU3`。设备包与采集对应提交 `1914063`。

### Key scale L2 对照 `native_pair_scale_l2`

四行 tiling 下，两侧 AIV 分别处理两条 query，会读取相同的 `keyScale` 页。自适应对照沿用上游的 `CACHE_MODE_DISABLE` 提示。本候选仅移除 `kScaleGm` 的该提示，使其沿用默认 L2 行为。大 Key Tensor 的缓存策略、TopK、交接和 Plan 保持原有实现。生成的 Host 与设备源码按算子名归一化后，只有设备入口的这段提示存在差异。

使用 `--native-scale-l2-only` 构建。短门禁在 batch17 上使用 `native_pair_auto native_pair_scale_l2 --contexts 8 2048 8192 --scenarios cold_reset --regression-repeat 2 --gate-only --prove-pair`。性能比较使用 `MTP3_BATCHES='17 32'`，分别选择 `ordered_full` 或 `ordered_ac` 与上述两路融合实现进行同场采集。
