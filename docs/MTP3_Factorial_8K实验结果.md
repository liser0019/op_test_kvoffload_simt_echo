# MTP3 Factorial 8K 实验结果

## 范围与身份

- 基线：`dc598b7`，分支 `exp/qli-mtp3-factorial`。
- 形状：MTP3、4 query/请求、TopK2048、Plan8K、gSize32、FP8→BF16。
- c0 保留原名 `quant_lightning_indexer_plan_c`：rankpipe AC streaming unique + packed ctx。
- 新组合使用独立注册身份：

| 角色 | 因子 | Operator |
|---|---|---|
| c0 | 无 | `quant_lightning_indexer_plan_c` |
| A | 原生 score TopK + 最终位置升序 | `quant_lightning_indexer_plan_cfa` |
| B | Vector-only 分阶段复用 | `quant_lightning_indexer_plan_cfb` |
| C | 8K query→UID map 留 UB | `quant_lightning_indexer_plan_cfc` |
| AB | A+B | `quant_lightning_indexer_plan_cfab` |
| AC | A+C | `quant_lightning_indexer_plan_cfac` |
| BC | B+C | `quant_lightning_indexer_plan_cfbc` |
| ABC | A+B+C | `quant_lightning_indexer_plan_cfabc` |

构建 source_id：`7b56c7cc93215aea770c6da2e93e5e395d4cb6dae7f6a15abd5ae120da3bbd6f`。最终包和参考包的 manifest、torch extension `_fusion_identity`、Plan 库身份均与本分支源码一致。

## 实现要点

### A

- 仅在原始 score TopK 已选出 2048 位置后排序，不改变 TopK tie 集合。
- 排序 scratch 使用 QLI TopK/Plan arena，但结果先 `DataCopy` 回独立 `indicesOutLocal_`，Plan publish 和 GM 发布只读该独立 buffer。
- 短行/空行继续使用原升序位置前缀和 `-1` 尾。
- A 系列保留 pinned native `quant_lightning_indexer_topk.h`，结构检查逐字节核对。

### B

- 新增独立开关，将原仅用于 storage mode 1/2 的 Vector-only 复用扩展到 c Plan。
- `resMm` Cube 双缓冲和 `indicesOutBuf_` 保持独立。
- QLI weights、scale、vecout 与 Plan arena 分阶段共用；正常/非法行 publish 前均等待 MTE2_V drain；下一轮 K scale padding 重置后使用 `PIPE_ALL` 与 MTE2 交接。
- S1Base4/c8192：总 UB 从 `221184B` 降到 `197632B`；S1Base2 从 `184832B` 降到 `164864B`。216KiB 门槛保留。

### C

- 只对 `capacity=8192, TopK=2048, 4 rows` 生效。
- 16KiB query→UID map 保持在 `data[0,16KiB)`；Phase6 不再写 `workspace[qb/2+i]`，Phase9 直接读 UB map。
- 相对 owner 的布局：
  - owner `[0,32KiB)`
  - partition `[32,48KiB)`
  - flags `[48,64KiB)`
  - directRank firstUid `[32,64KiB)`
  - directRank slotLookup `[64,80KiB)`
- `firstUid` 在 slotLookup 完成后、flags 写入前死亡；bitmap/firstUid/flags 重叠区有显式生命周期屏障。
- binary 的 `slotLookup=part` 保持 Phase6→压缩阶段同步。
- `LayoutHotCompactVf` 的 flags `LoadAlign` 使用新偏移；`RankPipeCopyOutputs` 在 Phase9 消费 map 后才用 `data[0]` 做 bounded tail staging。

## 初筛

命令形状：batch8/17/32，context 100K/256K，cold/75% warm，warmup2/repeat5，8 个 factorial 角色。全部 case `joint_groups=batch`、`overflow_groups=0`，TopK 有序输出逐元素一致。

| S1Base | 角色 | 相对 c0 mean span ratio | 范围 |
|---:|---|---:|---:|
| 2 | A | 1.0223 | 1.0168–1.0264 |
| 2 | B | 1.0387 | 1.0327–1.0429 |
| 2 | C | 1.0138 | 1.0106–1.0166 |
| 2 | AB | 1.0315 | 1.0249–1.0383 |
| 2 | AC | 1.0373 | 1.0308–1.0433 |
| 2 | BC | 1.0144 | 1.0101–1.0183 |
| 2 | ABC | 1.0203 | 1.0155–1.0293 |
| 4 | A | 1.0164 | 1.0035–1.0306 |
| 4 | B | 1.0196 | 1.0154–1.0227 |
| 4 | C | 1.0097 | 1.0038–1.0131 |
| 4 | AB | 1.0332 | 1.0236–1.0448 |
| 4 | AC | 1.0322 | 1.0222–1.0431 |
| 4 | BC | **0.9875** | **0.9842–0.9890** |
| 4 | ABC | 1.0267 | 1.0158–1.0463 |

结论：A、B、C 单独均不赚；BC 只在 S1Base4 大 batch 出现稳定收益，在 S1Base2 会伤害小 batch，因此不能全局启用。

## 复测

候选 `c0/A/C/BC`，batch6/8/10/16/17/32，100K/256K，cold/75%，warmup10/repeat20。表中 P50/P95 为 span μs。

| batch | context | scenario | S1Base | 最佳 | 最佳 P50/P95 | c0 P50/P95 | 最佳/c0 |
|---:|---:|---|---:|---|---:|---:|---:|
| 6 | 100000 | cold | 2 | c0 | 262.476/263.364 | 262.476/263.364 | 1.0000 |
| 6 | 100000 | warm | 2 | c0 | 298.211/299.241 | 298.211/299.241 | 1.0000 |
| 6 | 262144 | cold | 2 | c0 | 606.123/606.815 | 606.123/606.815 | 1.0000 |
| 6 | 262144 | warm | 2 | c0 | 640.784/642.365 | 640.784/642.365 | 1.0000 |
| 8 | 100000 | cold | 2 | c0 | 263.028/265.874 | 263.028/265.874 | 1.0000 |
| 8 | 100000 | warm | 2 | c0 | 297.688/300.299 | 297.688/300.299 | 1.0000 |
| 8 | 262144 | cold | 2 | c0 | 606.910/608.139 | 606.910/608.139 | 1.0000 |
| 8 | 262144 | warm | 2 | c0 | 640.699/642.546 | 640.699/642.546 | 1.0000 |
| 10 | 100000 | cold | 2 | c0 | 262.511/263.756 | 262.511/263.756 | 1.0000 |
| 10 | 100000 | warm | 2 | c0 | 298.436/299.066 | 298.436/299.066 | 1.0000 |
| 10 | 262144 | cold | 2 | c0 | 606.833/607.899 | 606.833/607.899 | 1.0000 |
| 10 | 262144 | warm | 2 | c0 | 640.589/642.471 | 640.589/642.471 | 1.0000 |
| 16 | 100000 | cold | 2 | c0 | 264.195/265.735 | 264.195/265.735 | 1.0000 |
| 16 | 100000 | warm | 2 | c0 | 301.637/303.096 | 301.637/303.096 | 1.0000 |
| 16 | 262144 | cold | 2 | c0 | 609.107/609.907 | 609.107/609.907 | 1.0000 |
| 16 | 262144 | warm | 2 | c0 | 643.460/644.726 | 643.460/644.726 | 1.0000 |
| 17 | 100000 | cold | 4 | BC | 292.038/293.073 | 295.645/297.667 | 0.9878 |
| 17 | 100000 | warm | 4 | BC | 323.642/325.182 | 326.582/327.944 | 0.9910 |
| 17 | 262144 | cold | 4 | BC | 677.240/677.934 | 685.469/686.188 | 0.9880 |
| 17 | 262144 | warm | 4 | BC | 707.170/709.235 | 716.069/717.348 | 0.9876 |
| 32 | 100000 | cold | 4 | BC | 294.774/295.605 | 297.411/298.597 | 0.9911 |
| 32 | 100000 | warm | 4 | BC | 328.428/330.251 | 331.699/334.856 | 0.9901 |
| 32 | 262144 | cold | 4 | BC | 679.578/680.469 | 687.610/688.457 | 0.9883 |
| 32 | 262144 | warm | 4 | BC | 713.033/717.001 | 723.717/724.844 | 0.9852 |

本轮的融合候选选型与分块阈值一致。batch 6/8/10/16 选择 c0，batch 17/32 选择 BC。两个候选保留独立入口，实际采集由测试框架显式选择。

## 同卡参考

参考包 source_id 相同，设备同为 NPU4 Ascend950DT。参考采集由任务内独立 compare 进程顺序执行；本开发进程核对了 build manifest、source_id、设备、所有 case 的联合/回退与校验状态。

P50/P95 span μs：

| batch | context | scenario | c0 | BC | peer | final4 |
|---:|---:|---|---:|---:|---:|---:|
| 8 | 100000 | cold | 263.095/264.439 | 266.613/268.762 | 286.667/287.652 | 208.097/209.015 |
| 8 | 100000 | warm | 298.461/299.675 | 301.094/302.033 | 340.399/342.274 | 232.733/263.806 |
| 8 | 262144 | cold | 606.438/608.213 | 616.724/617.829 | 626.987/628.056 | 475.252/476.306 |
| 8 | 262144 | warm | 640.845/642.455 | 648.990/651.874 | 682.682/684.073 | 499.976/501.476 |
| 17 | 100000 | cold | 295.316/296.438 | 291.873/292.930 | 317.154/318.196 | 293.930/389.245 |
| 17 | 100000 | warm | 326.231/327.708 | 322.660/324.334 | 370.113/371.387 | 289.524/382.081 |
| 17 | 262144 | cold | 685.599/687.462 | 677.626/678.287 | 704.327/705.813 | 615.644/645.453 |
| 17 | 262144 | warm | 717.235/718.175 | 707.341/708.158 | 759.602/760.914 | 636.206/638.738 |
| 32 | 100000 | cold | 297.440/298.216 | 294.293/295.377 | 319.116/319.881 | 272.550/281.513 |
| 32 | 100000 | warm | 328.758/329.586 | 325.355/326.428 | 370.259/372.017 | 293.554/300.208 |
| 32 | 262144 | cold | 688.003/689.972 | 679.121/679.750 | 706.046/706.911 | 623.455/626.174 |
| 32 | 262144 | warm | 723.178/724.037 | 713.448/714.285 | 761.865/763.014 | 643.655/645.034 |

final4＋独立 rankpipe AC 在 12 个参考工况中拿到 11 个最低 span P50。100K、batch 17 冷启动由 BC 领先 2.057 μs，该工况的拆分 kernel 间隙 P50/P95 为 25.110/120.176 μs。256K 的 6 个工况均由拆分方案领先。这里的 100K 指 100000 个 token。

## 正确性证书

- NPU boundary gate：`results/gate_mtp3_short_v2`，context 8/64/2048/8192/100000，8 个角色全部通过，zero fallback。
- 初筛/复测：每个 case 都保持只读 QLI 输入、host block table、split workspace/TopK guard 不变；Plan 状态按共同初态恢复；完整联合组数等于 batch；无 overflow。
- native TopK 集合认证：`results/a_native_set_gate_v1`，batch32/256K/cold，原始 Indexer 对 ordered canonical 的集合重合率 100%，0 行差异；A 输出与 canonical 逐元素一致。
- final4 native 认证：`results/baseline_final4_gate_v1`，集合重合率 100%，0 行差异。

## 原始结果与复现

- 初筛：`results/factorial_screen_v1/batch{8,17,32}`。
- 复测：`results/factorial_confirm_v2/batch{6,8,10,16,17,32}`。
- 同卡参考：`results/reference_compare/batch{8,17,32}`。
- 资源探针：`results/qli_fusion_resources_final_build.json`。
- 汇总：`results/factorial_summary.json`。
- 构建：`build/factorial_final`、`build/factorial_compare`、`build_joint_plan`。

### 复用服务器现有构建，复测 batch 32

```bash
RUN=/workspace/qli_mtp3_factorial_JiuyXu
cd "$RUN/src"
source /usr/local/Ascend/cann-9.2.1/set_env.sh
source "$RUN/build/factorial_final/env.sh"
export TORCH_EXTENSIONS_DIR="$RUN/build/factorial_final/torch_extensions"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
OUT=$(mktemp -d "$RUN/results/retest_XXXXXX")
mkdir "$OUT/tmp"
"$RUN/venv/cann921/bin/python" -B profile_qli_fusion.py \
  --fusion-build "$RUN/build/factorial_final" --device 4 \
  --library "$RUN/src/build_joint_plan/libmtp_joint_plan_ops.so" \
  --requests 32 --mtp 3 --contexts 100000 262144 \
  --scenarios cold_reset full_cache_75pct \
  --implementations c0 A C BC --warmup 10 --repeat 20 --regression-repeat 1 \
  --output "$OUT/batch32" --temp-dir "$OUT/tmp"
```

## 失败记录与环境局限

- 初次构建 `_c_a` 命名导致生成 proto 为 `_cfa`、CMake 等待 `_c_a_proto.cpp`；改用无歧义 `_cfa/_cfb/...` 身份后通过。证据：`results/build_factorial_full.log`。
- 第一次复测报告固定打印 `ub['B']`，候选子集不含 B 时在报告阶段 KeyError；采集和校验已完成但未形成 COMPLETE，现场保留在 `results/factorial_confirm_v1`。修复后新目录 `factorial_confirm_v2` 全量通过。
- 现有 CPU shim 在 mode3 的新 VF/Cast 模板上触发 GCC internal compiler error，未用于本轮结论；以 CANN9.2.1 编译和 NPU4 真机为准。
- MTP7 未测，不在本次验收范围。
- 未测 Transfer/SFA；性能数值仅覆盖对应 Indexer/Plan 目标 kernel span，不包含宿主框架全链路。
