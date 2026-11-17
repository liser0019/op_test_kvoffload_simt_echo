# QLI A/B初始化同步与CSV时间精度修复

基于服务器已推送的 `0a42bee`。本次只修复新增的同步依赖和计时算术，不重写
TopK、MrgSort、LRU、Plan状态协议或C的存储路径。保留精确TopK、逐位置输出、
持久状态和保护区校验。不能把本次补丁称为已经解决A/B设备差异的根因。

## A/B的初始化依赖

A/B把Indexer的Vector参数临时区借给Plan。下一段Indexer开始时，要重新清零
K-scale的padding，再由MTE2加载有效scale。旧的 `PIPE_V` 屏障仅约束V流水；
之前已发布的ready flag不能覆盖后来新增的清零写入。

| 阶段 | 对相同K-scale区域的操作 | 顺序要求 |
|---|---|---|
| Vector | `Duplicate(kScaleUB_, 0, ...)` | 新增初始化写入 |
| 阶段屏障 | `PipeBarrier<PIPE_ALL>()` | 完成初始化后才进入后续流水操作 |
| MTE2 | `GetKeyScale` 中的DataCopyPad | 覆盖有效scale，padding保留为0 |
| Vector | scale参与分数计算 | 保留原有MTE2→V同步 |

屏障只放在 `isFirstS2InnerLoop` 的初始化阶段，不在每次归并或每个token处增加
同步，不分配新event ID，也不修改原ping-pong信号。split/C没有这项Vector临时区
复用，保持原有实现。额外屏障的实际耗时及是否消除TopK差异，都需要服务器验证。
接口语义参考[官方核内同步说明](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/intra_core_sync_overview.html)：
`PIPE_ALL`用于阻塞所有流水。

生成后的A/B源码由 `check_qli_fusion_resources.py` 检查：清零与加载之间必须有
跨流水初始化屏障。这是结构回归检查，不模拟真实NPU流水。

## CSV解析与计时口径

1. 直接从CSV字符串构造Decimal，不先把绝对时间转成double。
2. 所有目标记录减去同一个精确时间原点，再计算busy、span和逐对gap。
3. 结果统计时才转成普通浮点数；组件同时保留 `absolute_start_us` 和
   `csv_duration_us` 原始字符串。组件 `start_us` 为相对时间，原点在每对的
   `timestamp_origin_us`，算术版本为 `decimal_relative_v1`。
4. 保留 `0a42bee` 对上游导出器已量化时间戳的容差；小负gap仍归零，但原始负值、
   归零量、容差分别写入 `reported_between_kernel_gap_us`、`gap_clamped_us`、
   `gap_tolerance_us`。超出容差仍失败，不改样本数或调用顺序校验。

例如原始Duration为533.078μs，大绝对时间浮点相减曾得到533.125μs；新解析保留
533.078μs。busy仍为设备区间并集，span包含两条kernel之间的间隙，Event单独报告。
本次没有重算服务器旧CSV，也不修改之前的上机结果文档。

## 干净重建的资源头修复

`0a42bee`将四路Torch绑定的独立资源头放入公共打包目录；但该目录不在后续
逐算子命名空间重写范围内。新生成的绑定引用 `quant_lightning_indexer_plan_*_resource`，
公共资源头仍声明 `qli_fusion_resource`，导致干净源码的Host编译找不到命名空间。
生成脚本现在显式为这四个公共头使用各自命名空间；Host语法检查也加入真实的
公共include路径。此修复不改变资源计算公式或UB预算。

## 服务器最小复测

在新工作目录克隆 `feat/qli-rankpipe-fusion`，按主教程设置CANN/Python并构建。
必须重建专用融合包及Torch扩展，源码身份已经改变，不能继续加载旧A/B。
构建步骤沿用主教程的 `build_joint_plan.sh --rankpipe-only`、
`scripts/build_qli_fusion.sh` 及构建目录的 `env.sh`。构建和结果放新目录，保留旧库与现场。

以下命令从新仓库根目录执行；`RUN`是本次新建的持久结果目录，已source本次env.sh。
沿用上轮NPU4，若卡不可用再显式修改 `--device`。

```bash
set -euo pipefail

# 先只跑已复现问题的输入；一次生成canonical，A/B/C各20次严格门禁。
# gate-only不采集profiler、不生成性能结论；失败立即停止，保留调用序号及现场。
bash scripts/test_qli_fusion.sh --device 4 --requests 32 --contexts 262144 \
  --mtp 3 --scenarios cold_reset --implementations a b c --gate-only \
  --regression-repeat 20 --warmup 0 --repeat 1 \
  --output "$RUN/topk_gate" --temp-dir "$RUN/tmp"

# 只有上一条退出码0才执行。只采原失败工况，先不展开16工况或20样本矩阵。
bash scripts/test_qli_fusion.sh --device 4 --requests 32 --contexts 262144 \
  --mtp 3 --scenarios cold_reset --implementations split a b c \
  --regression-repeat 1 --warmup 2 --repeat 5 \
  --output "$RUN/focused_profile" --temp-dir "$RUN/tmp"
```

20次通过仅表示本轮重复门禁通过，不能证明所有调度和输入都安全。如果A/B仍出现
不同TopK，停止扩展性能采集，继续检查arena复用与分数差异；不能把“有序且数量正确”
当作放行条件，也不能在没有分数证据时判定为并列分数。

## 本次开发机验证

- 输入/CSV/初始化结构回归共20项通过；包括大时间戳保留Duration、平移不变性、
  mixed区间并集、真实重叠拒绝及旧清零屏障拒绝。
- 干净固定上游生成四路overlay后，四路真实Torch schema注册和Host C++语法通过。
  ACLNN调用使用桩，只验证Host语法，不是CANN编译或链接。
- 实际Host资源函数四模式×三档容量检查通过，A/B新增初始化屏障结构检查通过；
  UB预算未变。融合源码身份已更新，独立冻结Plan身份未变。
- Python语法、设备Plan可重现生成及diff格式检查通过。未修改Plan计算源码，
  没有将先前105次CPU Plan语义调用记为本次重新执行。
- 本机没有CANN/NPU，未验证设备代码、跨流水真实执行或A/B的TopK差异是否消失。

开发机一次PyTorch导入遇到本机ApproximateClock非单调异常；固定Host CPU后
20项回归通过。这是本地检查条件，不涉及服务器NPU编号或生产算子实现。
