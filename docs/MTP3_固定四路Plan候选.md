# MTP3 固定四路 Plan 候选

## 范围与出口

固定输入形态为 4 条按 token 位置升序的 query，`TopK=2048`、`capacity=8192`、`max_token=262144`、512 线程。基线保持冻结的 `joint_rankpipe_ac_512`（mask 249）；新候选均在编译期通过独立 macro 关闭，不改变 249 的生成定义，也不把 8 行 MTP7 路径改成 4 行。

| 出口版本 | mask | 功能 |
|---|---:|---|
| `joint_rankpipe_ac_512` | 249 | 冻结基线，mask249 |
| `joint_mtp3_full_512` | 252 | 四路完整装下时，输入/输出各 64KiB packet + 一次 `MrgSort<float,false>` |
| `joint_mtp3_map_512` | 253 | 目标热路径 16KiB query map 留 UB，flags 重定位 |
| `joint_mtp3_load_512` | 254 | 完整 64 条 32B 对齐记录使用 `DIST_DINTLV_B32` 成对加载 |
| `joint_mtp3_full_load_512` | 255 | Full + Load 组合 |

新候选在 `PreparedVariant` 构造时明确拒绝非 `rows=4/TopK2048/capacity8192/maxToken262144/joint=1` 形态。因此其他形状继续走基线，不会静默进入候选快路。基线 249 与 MTP7 的既有形状约束保持不变。

构建入口：

```bash
PYTHON=/usr/local/python3.12.16/bin/python3 \
  bash build_joint_plan.sh --soc=ascend950 --mtp3-only
```

该门只生成 249、252、253、254、255，不展开 32 路 factorial。共享库位于 `build_joint_plan/libmtp_joint_plan_ops.so`，仍由 `JointVariantLaunch`、source-info 和 manifest 指纹校验，未绕过库检查。

## 实现

### Full（252/255）

- 只在固定 4×2048/8192 目标形状把 packet 工作区从原 2×48KiB 扩为 2×64KiB，即增加 32KiB；输入和输出各 64KiB。
- `LayoutHotScratch` 的目标布局为 45184 个 int32，加 `PLAN_SCAN_BYTES=288` 后动态 UB 为 181024B（约 176.8KiB）。
- sources 全部完整时沿用现有 `MrgSort<float,false>` 一次完成四路 8192 记录归并；短行、空行或未完整装下的情况继续走原 chunk/跨包及 exhausted `MrgSort<true>` 逻辑。
- 其他 8 行路径不进入该 packet 扩展；macro 为 0 时源码行为与原 rankpipe AC 相同。
- CPU 专项覆盖并集恰为 8192、40% 重合、短有效前缀、空行、全命中、重复 resident 与 cold reset。

### Map（253）

- Phase6 不再把 16KiB map 写到 GM；目标形状将 map 原址保留到 Phase9，Phase9 用 UB 中的 packed map 生成逐位置 `currentSlots`。
- `flags` 从原 map alias 重定位到 `owner + 12288` 个 int32；它只覆盖 `firstUid` 已消费后的尾段。
- `part` 仍为 `owner + capacity` 对应区间；direct-rank 的 `slotLookup` 仍为 `owner + 16384`，binary fallback 仍为 `part`。
- `LayoutHotCompactVf`/`RankPipeCompactVf` 显式接收 `hotUbMap`，避免 compaction 再读 map alias。
- Phase9 是 UB map 的最后一个 reader；之后既有 bounded tail staging/MTE3 写回仍按原生命周期执行。
- 仅 hot 目标形状启用。cold、reset、状态协议与 miss 描述符 `-1` 尾部不变。
- NPU gate 读取前后 compact workspace，`workspace_no_spill=true`，证明确实未把 16KiB map 落回 GM。

### Load（254/255）

- `RefineUniqueMapVf` 仅在记录块恰为完整 64 条且源地址 32B 对齐时使用 `Reg::LoadAlign<..., DIST_DINTLV_B32>`，一次拆出 score/payload 成对记录。
- 尾块或未对齐地址继续使用原 masked `Gather`，不越界读取。
- score 仍按 float 位型解释，payload/query ID 不做数值 float 转换；其余 union、容量检查、slot lookup 与输出语义不变。

## 验证结果

### CPU64 + UBSan

命令：

```bash
python3 -B checks/check_mtp3_candidates.py --report /tmp/mtp3_cpu_gate.json
```

249、252、253、254、255 各 7 个固定形态用例全部通过；输入只读、guard、LRU、逐位置槽位、miss 有效前缀/`-1` 尾部均由公共语义检查核对。

- 基线/Map/Load 动态 UB：148256B
- Full/Full+Load 动态 UB：181024B
- packet 增量：32768B

### CANN 9.2.1 与 NPU5

构建使用 `/usr/local/Ascend/cann-9.2.1`、`/usr/local/python3.12.16/bin/python3`、`bisheng`。NPU5 在运行前已有其他进程占用；未终止任何进程，只做本路线短测。

NPU 门禁：

```bash
python3 -B checks/check_mtp3_npu.py --device 5 \
  --library build_joint_plan/libmtp_joint_plan_ops.so \
  --report <output>/npu_gate.json
```

五个实现均在 disjoint、overlap40、short、empty 四类目标输入上完成 3 轮独立审计并通过，输入错写回归要求 `input_restore=false`。Map 的 `workspace_no_spill=true`。

### 构建/输出位置

- 构建：`/workspace/ordered_mtp3_parallel_aQVhnU/plan/src/build_joint_plan/`
- 首次失败证据：`/workspace/ordered_mtp3_parallel_aQVhnU/plan/mtp3_round1_20261118/logs/npu_gate.log`
  - Full 的 C++ 动态 UB 为 181024B，但 Python layout 仍按 148256B 检查，触发 `Python/C++ tiling模型不一致`。
  - 修正：宿主 `prepare()` 传递 `mtp3_full`。
- 第二次失败证据：`/workspace/ordered_mtp3_parallel_aQVhnU/plan/mtp3_round2_20261118/logs/npu_gate.log`
  - MTP3 名称尚未纳入 layout16 的 `mtp8` 预算模型。
  - 修正：`supports_eight_rows()` 将 MTP3 纳入同一 layout16 预算分类。
- 通过门禁：`/workspace/ordered_mtp3_parallel_aQVhnU/plan/mtp3_round3_20261118/`
  - `cpu_gate.json`
  - `npu_gate.json`
  - `source_identity.json`
  - `logs/build.log`
  - `logs/npu_gate.log`
- 最终通过门禁/短测：`/workspace/ordered_mtp3_parallel_aQVhnU/plan/mtp3_final4_20261118/`
  - `cpu_gate.json`
  - `npu_gate.json`
  - `source_identity.json`
  - `logs/build.log`
  - `logs/npu_gate.log`
  - `logs/profile.log`
  - `profile/joint_matrix_n12j_x2m/`
- 最终 NPU gate `source_id`：`86b5c33fb8b0ecf13ac24d489881096758e0797008a48caa5931e5757573e825`
- 最终 manifest SHA256：`4a2ecf0d3e80199985a49dfc3d88f89bae4c9af929549f26050b92ce45ba0abf`
- 最终库 SHA256：`baf401522c3e3e02a67cbfa0adb4aeb1c8d71b5a6aafed528200c845e2f060a6`
- 输入指纹：`97c898eaab812d28049c07c8b74cf678ad6ef287648286551b6c78f71a0cae5f`

## 短 profiling

目标输入：`requests=1`（4 行）、MTP3、`full_cache_75pct/overlap40`、TopK2048、capacity8192、maxToken262144。仅采 Plan kernel，比较对象固定为 mask249。主比较留给 NPU4 的全链路复测；本路线结果只用于候选可行性/方向判断。

profile 为同一次采集、同输入、同库的五路轮换 5 样本结果；Plan kernel only，P50/P95 单位 μs。

| 实现 | P50 | P95 | 相对 mask249 P50 | 正确性 |
|---|---:|---:|---:|---|
| `joint_rankpipe_ac_512` | 56.999 | 57.947 | 1.000 | 通过，miss=2290 |
| `joint_mtp3_full_512` | 53.550 | 53.804 | 0.939 | 通过，miss=2290 |
| `joint_mtp3_map_512` | 56.456 | 56.698 | 0.990 | 通过，miss=2290 |
| `joint_mtp3_load_512` | 58.065 | 58.576 | 1.019 | 通过，miss=2290 |
| `joint_mtp3_full_load_512` | 54.786 | 55.885 | 0.961 | 通过，miss=2290 |

该短测只有 5 个 kernel 样本，且 NPU5 同时存在其他进程，不能替代主 Codex 在 NPU4 的同卡 MTP3 全链路比较。当前方向证据是：Full 单独最快（约快 6.1%）；Map 与基线接近（约快 1.0%）；Load 单独回退约 1.9%；Full+Load 比 Full 慢，说明本形状下 pair-load 与归并资源/指令组织存在负交互。后续应优先 Full，并在主 Codex 的同卡全链路测试中复核 Map 与 Load 条件。

该输入 union=3826、4×2048 全部有效、初始逐行命中为 1536/774/761/784；没有 overflow 组。NPU gate 另用 union==8192 的 disjoint 输入验证了 Full 一次装下路径。

## MTP7 回归

- `python3 -B checks/check_mtp7_fix_suite.py`：5/5 通过。
- `python3 -B checks/check_mtp7_pack_suite.py`：4/5 通过；唯一失败是仓库原有 `mtp7_pack_unchanged_sources["csrc/joint_variants/kernel.cpp"]` 冻结哈希与 HEAD 文件不一致。
- 证据：`git diff --exit-code HEAD -- csrc/joint_variants/kernel.cpp` 返回0；HEAD/current SHA256 均为 `bb1b280f9bc65f22a8a04cc7409d462c6af6b8d8bbf584ed711f790c258e6b70`，manifest 旧值为 `33297ca2430b034dc687871009c636456b55af2b20e9485e8c58de2fe6c8f8ce`。本路线没有修改该文件，8 行源码/生成分支与 249 的生成 macro 均未改为 4 行。
