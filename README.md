> 当前分支新增位图＋直接索引候选，默认缩减为三路、三个工况。见 [位图实验与上机命令](docs/Joint_Bitmap直接索引实验.md)。

> 本分支增加 Joint A/B/C/D 单点与15种组合，默认候选和重合模式已扩展。最新运行步骤见 [Joint_ABCD优化实验.md](docs/Joint_ABCD优化实验.md)。原三路仍可显式选择 `--implementations abcdf private joint --overlaps shared partial`。

# Joint Plan线程1规避修复：ABCDF/private/joint三方MTP profiling

当前分支：`feat/mtp-joint-plan-profile`。
移植`feat/mtp-joint-plan@bba63d057`的联合去重、共享resident和容量溢出回退实现，
默认比较32请求下MTP=0/1/3：原独立abcdf、联合框架private、线程1规避修复后的joint。
只调整显式modeOutput传递及联合模式发布的写线程；根因未确认，不增加barrier或阶段循环。
保留原计算辅助函数、1024线程和同步；独立构建，不要求安装vLLM。

完整教程：[MTP联合Plan性能测试](docs/MTP联合Plan性能测试.md)。

```bash
bash build_joint_plan.sh --soc=ascend950
python3 -X faulthandler profile_joint_mtp.py --device 2 \
  --implementations abcdf private joint --mtp 0 1 3 --requests 32 --overlaps shared partial \
  --warmup 10 --repeat 50 --temp-dir /workspace/joint_plan_results/tmp \
  --output /workspace/joint_plan_results/formal
```

先按教程小尺寸门禁和短跑，再正式采集。计时来自kernel_details.csv。
本轮只测Plan，报告实际并集、逐行miss和回退情况；不执行Transfer/SFA。
默认100次输入错写回归，逐样本校验不能关闭；可复用同库同设备的完整门禁报告。
开发机仅执行CPU与静态检查，未做设备构建/NPU验证，不宣称根因已确认或新构建已上机通过。

---

# 以下保留Transfer A/B/C/D组合性能实验

原Transfer实验分支：`feat/transfer-factorial-optimizations`，基于修复分支`e610c61`。
保留原ABCDF和原Transfer，新增control及A/B/C/D全部16种组合，共17路Transfer。
完整说明：[Transfer组合优化实验](docs/Transfer组合优化实验.md)。

| 因子 | 改动 |
|---|---|
| A | K/V共用队列记录 |
| B | 每批8个有效token |
| C | block64、1024/128B生产布局特化 |
| D | 全局miss连续区间分核 |

同一工况复用全部输入、输出、payload与恢复快照地址，计时外切换上下文。
默认prepared、Transfer-only、full cache75/90/100%，以kernel_details.csv统计性能。
原版与无优化control单独保留，不预设组合优于单点；当前未在开发机执行编译/CPU/NPU测试。

```bash
bash build.sh --soc=ascend950 --plan-transfer
bash run_transfer_variants.sh 0 --temp-dir /dev/shm/transfer_tmp
```

先按教程小尺寸门禁/短跑，再跑生产全矩阵。支持--variants、--core-limits、--modes chain、
--reverse-order；独立外部msopprof入口为run_transfer_msopprof.py。
90多GB/s参考值需核对payload一次计数与读写累计口径：46GB/s payload可能对应约92GB/s读写累计。

---

# 以下为保留的满缓存 Plan→Transfer 框架说明

框架基线分支：`fix/abcdf-plan-transfer-harness`，基于 `feat/abcdf-transfer-full-cache@fcbaa31`。
保留原ABCDF，接入生产SparseKvTransfer，默认抓串联两条kernel各自延时。
默认模型预设为你提供的GLM MLA BF16：**压缩KV 1024B + RoPE K 128B/token**。

完整教程：[满缓存Plan与Transfer性能测试](docs/满缓存Plan与Transfer性能测试.md)。

在950PR服务器：

```bash
bash build.sh --soc=ascend950 --plan-transfer
bash run_plan_transfer.sh 0 --launch-mode legacy --temp-dir /dev/shm/plan_transfer_tmp
bash run_plan_transfer.sh 0 --launch-mode prepared --temp-dir /dev/shm/plan_transfer_tmp
```

两个采集入口先对所选启动方式做完整正确性门禁，再采集75%、90%、100%三档。
独立checks/check_plan_transfer.py默认同时检查legacy/prepared，便于上机验收。
legacy保留逐次tiling上传；prepared在setup阶段准备两条kernel的tiling与核数，
由调用方持有上下文，通过一次队列回调连续下发原kernel。默认仍为legacy。
尚未上机验证prepared，不预先承诺gap缩小或消失。
默认生产尺寸32请求、TopK2048、capacity4096、max_token262144；
每行miss分别512、205、0。90%报告实际比例89.990234375%；
所有槽初始占满，ID匹配且stable覆盖全部token，因此100%确实零miss。
本实验MTP关闭；Transfer在零miss时仍实际下发。

独立的 `plan`、`transfer`、`chain` 模式、显式K/V字节数、
小尺寸试跑与Host内存预算见教程。默认一层Host池9GiB、resident payload144MiB，
此外保留同等大小的设备恢复快照。
结果包括每条kernel的P50/P95、实际配对busy/span与原始CSV；没有用Event代替设备duration。
本轮补全gap的逐对P50/P95/样本数；运行前检查实际输出/临时目录的字节/inode余量及可写性。
最低余量可配置，不是profiling大小预测；不检查源码分区空间，不自动清理或切换位置。
详细日志为session下profile.log；只有COMPLETE.json和complete元数据同时存在才表示整轮成功。

**本轮仅做源码修改和静态检查，未在本机执行构建、CPU检查或NPU测试。
新Transfer接入及整套采集仍需服务器验证，不预先宣称可编译或有性能收益。**

---

## 以下为保留的MTP / Scan框架说明

该部分来自上一分支；运行其全候选矩阵需不带`--plan-transfer`构建，
使用build/下的完整库。新Plan/Transfer实验请用上面的入口。

# ABCDF 基线：MTP 开关与 Scan 优化实验

分支：`feat/abcdf-mtp-scan`；起点为 `feat/simt-plan-micro-variants@9cc99d5`。
当前参考版本是 **原版 ABCDF**，对应 `csrc/plan_variants/13_abcdf/`。
其四个入口/配置文件及原 `common/` 计算、Host 接入文件保持不变。
新实验写在 `csrc/scan_experiments/`，不包含之前失败的设备侧打点。

**交付状态：仅完成源码修改和静态审阅。本轮未在开发机运行 Python、CPU 模拟、
构建或 NPU 测试；所有新候选的 CANN 编译、正确性及性能均等待服务器验证。**
普通 ABCDF 的此前实机结果不代表本轮候选已经通过验证。

## 1. 从哪里读代码

| 顺序 | 文件 | 阅读目标 |
|---|---|---|
| 1 | `mtp_cases.py` | 请求/行数、visible/stable、初始缓存与精确 miss 分布 |
| 2 | `compare.py::reference_simt` | 独立列表/字典 golden，明确公开语义 |
| 3 | `checks/check_mtp_scan.py` | 边界、污染审计、NPU 自身状态连续续跑 |
| 4 | `profile_plan_variants.py` | 单组采集、轮转顺序、CSV 分组与样本数检查 |
| 5 | `profile_mtp_matrix.py` | MTP 与版本矩阵、正确性门禁及汇总 |
| 6 | `csrc/plan_variants/common/plan_kernel_impl.h` | 原版 ABCDF 算法 |
| 7 | `csrc/scan_experiments/common/experiment_kernel_impl.h` | 本轮 Scan 实现 |
| 8 | `docs/ABCDF_MTP_SCAN设计.md` | 前缀和、同步、scratch 布局及实验解释 |

`compare.py` 的 Event 计时保留为诊断入口；正式排名使用 profiler 的
`kernel_details.csv` 中设备 kernel duration。

## 2. MTP 的明确口径

`--mtp` 是 **draft token 数**，`q_len=1+mtp`。
默认32请求，TopK2048、capacity4096、max_token262144、1024线程、核数上限64。

| 对照 | MTP=0 | MTP=3 | 解释 |
|---|---|---|---|
| 固定32请求 | 32行 | 128行 | 开启MTP后的整次Plan代价，同时增加4倍query行 |
| 固定32query行 | 32请求/32行 | 8请求/32行 | 辅助控制实验，避免把工作量增加都算成算法退化 |

每条 query 行拥有独立的 resident/LRU/last_req_id 状态。同一请求的4行共享
request ID 和页表，初始状态从同一历史快照分叉；不会并发写同一缓存状态。
这是该 Plan 接口的行级测试，不包含 draft 模型、接受判定或 KV Transfer。
连续测试中的接受数只用于推进已提交长度，检查下一轮 Plan 状态。

设已提交长度为 context，位置p取0..mtp：

- `visible=context+p+1`，包含当前 token 以及该位置之前的 draft。
- MTP开启时 `stable=context`；关闭时 `stable=visible`。
- 默认query末尾含1..4个该位置可见的新token；其余为合法、不重复的稳定token。
- warm状态中预埋最多3个历史投机token，ID匹配，确保后缀失效不会被reset掩盖。
- MTP=0/3使用相同context、seed及基础query；固定请求数时第0位置query相同。
- 128行超过核数上限，会测试同一block连续处理多行与scratch复用。

沿用四个场景名：`cold_reset`、`warm_75pct`、`warm_all_hit`、
`full_cache_75pct`。**这些名字描述历史缓存构造，不是所有query的精确命中率。**
例如 `warm_all_hit` 的历史稳定部分全部命中，新token仍miss；
生产尺寸下，MTP=0每行1 miss，MTP=3每请求四行分别1/2/3/4 miss。
报告记录实际 `miss_count_per_row`、visible/stable、reset及失效槽数。

## 3. 本轮版本

**本轮A～F与历史ABCDF的字母相互独立。** 历史掩码始终为47U。
A=Miss Ballot；B=LRU Dual Ballot；C=Warp自主归约；D=整段LRU；
E=整段Miss；F1=冷路径Ballot。

| 命令名 | 本轮因素 | LRU | Miss |
|---|---|---|---|
| abcdf / V000 / F0 | 无 | 原逐tile | 原逐tile |
| v100 | A | 原 | Ballot |
| v010 | B | Dual Ballot | 原 |
| v001 | C | 自主归约 | 自主归约 |
| v110 | A+B | Dual Ballot | Ballot |
| v101 | A+C | 自主归约 | Ballot+自主归约 |
| v011 | B+C | Dual Ballot+自主归约 | 自主归约 |
| v111 | A+B+C | Dual Ballot+自主归约 | Ballot+自主归约 |
| s01 | D | 整段 | 原 |
| s02 | D+A | 整段 | Ballot |
| s03 | D+C | 整段 | 自主归约 |
| s04 | D+A+C | 整段 | Ballot+自主归约 |
| s05 | E | 原 | 整段 |
| s06 | E+B | Dual Ballot | 整段 |
| s07 | E+C | 自主归约 | 整段 |
| s08 | E+B+C | Dual Ballot+自主归约 | 整段 |
| s09 | D+E | 整段 | 整段 |
| cold_ballot / F1 | 仅冷路径Ballot | 热路径语义沿用ABCDF | 热路径语义沿用ABCDF |

共17个热路径实现（含ABCDF）+1个冷路径候选。
D替代B；E替代A；C只作用于还使用逐tile Scan的热阶段。
默认全组矩阵在cold_reset只采集ABCDF/F1，其他三场景采集17个热版本。

每个候选分别编译为独立kernel/VF/launch/torch.ops，策略在编译期选择。
版本、共享计算/测试文件的哈希写入 `plan_variants_manifest.json` 和共享库。
过期库或修改后未更新的源码会被拒绝，避免把旧二进制当新算法。

## 4. 在950服务器构建和试跑

在已有匹配torch/torch_npu/CANN环境中执行，目录须允许加载共享库。
以下所有设备命令明确使用逻辑卡0；不要同时在卡0启动其他采集任务。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh  # 按安装位置调整
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
PYTHON=python3 JOBS=3 bash build.sh --soc=ascend950
```

先跑少量版本的生产正确性，随后试采集：

```bash
python3 -X faulthandler checks/check_mtp_scan.py --device 0 --production \
  --implementations abcdf v001 v111 s01 s05 s09 cold_ballot \
  --report results/smoke_correctness.json

python3 -X faulthandler profile_mtp_matrix.py --device 0 \
  --implementations abcdf v111 --mtp 0 3 --comparison both \
  --scenario warm_75pct --warmup 2 --repeat 4 \
  --correctness-report results/smoke_correctness.json \
  --output /dev/shm/abcdf_mtp_smoke
```

正式全组运行：

```bash
PYTHON=python3 bash run_plan_suite.sh 0
```

该脚本先对全部18版本运行生产门禁，通过后才开始MTP矩阵。
门禁覆盖重复query/resident、非法来源、越界LRU、stable变化、reset、槽不足、
warp/tile尾部、4095/4096/4097全evict、哈希冲突、UB/GM、同核多行及MTP连续步骤。
所有实现比较3个持久状态+4个输出；重复恢复初始状态、污染输出/scratch；
连续步骤直接续用上一轮NPU状态，不从golden重写设备状态。

默认每组预热10次，每版采样数为该组版本数的4倍，执行顺序逐轮轮转。
环境变量可覆盖 `GROUP=local|structural|cold|all`、`WARMUP`、
`REPEAT`、`REQUESTS`、`FIXED_QUERY_ROWS`、`TOPK`、`CAPACITY`、
`MAX_TOKEN`、`SHUFFLE_LRU=1`、`PLAN_PROFILE_OUTPUT`、`PLAN_LIBRARY`。
例如 `GROUP=local SHUFFLE_LRU=1 bash run_plan_suite.sh 0`。
自设REPEAT最好是所选版本数的整数倍，否则执行位置未完全平衡。

复用门禁时必须是同库SHA、同manifest、同逻辑卡且覆盖所选版本的生产报告。
缺卡/缺库/异常/校验失败均会终止，不会用SKIP生成有效性能结论。
源码哈希只是身份核对，不代表正确性；新编译库必须重新过门禁。

### 精确miss数与分布实验

默认四场景包含可见新token，不能强行同时要求MTP=3每行0 miss。
因此另设控制实验：**query只选择稳定token**，精确指定每行miss数，
保留MTP的visible/stable和历史投机槽失效。与默认场景分开解释。

```bash
python3 -X faulthandler profile_mtp_matrix.py --device 0 --group local \
  --mtp 0 3 --comparison both --requests 32 --fixed-query-rows 32 \
  --miss-counts 0 1 16 256 1024 2048 --miss-distributions front uniform \
  --warmup 10 --output /dev/shm/abcdf_miss_sweep
```

`--miss-counts` 替代四场景矩阵。front/back将miss集中在query头/尾，
uniform将miss位置均匀分散。初始化缓存为满缓存；MTP失效可先空出少量槽，
所以miss数与实际淘汰历史token数仍不同。该实验不能冒充真实生产输入回放。

## 5. 为什么使用profile，以及报告怎么看

每次调用前恢复相同初始状态；golden、污染审计和检查在采集循环外。
torch profiler采集中的恢复会产生TensorMove，但CSV汇总只匹配所选kernel，
不会把TensorMove、tiling上传或多个任务类型累加成Plan耗时。

每组先做3轮污染审计，再预热；profiler额外1个warmup step；
每个active step调用各版本一次，每轮改变执行位置。结束后再检查最终状态。
必须精确匹配每版一个kernel分组且count=repeat，否则拒绝排名。
恢复会影响cache温度：结果代表这套固定恢复流程，不能直接推断整网cache行为。

输出保留在唯一的 `session_*/matrix_*/capture_*` 目录，包括：

- `kernel_details.csv`：原始设备任务耗时。
- `kernel_summary.csv`：每版P50/P95/min/max/样本数。
- `capture_metadata.json`：kernel名、输入哈希、库/源码身份、编译选项、shape、工作量与布局。
- `matrix_summary.json`、`report.md`：MTP开/关比值、按行/请求归一化值、局部因素交互项。
- `correctness.json`：必须完整通过的设备门禁。

P50是一次调用处理全部query行的中位耗时。除以行数/请求数是吞吐归一化，
不是独立请求延迟。固定32请求时，MTP=3有4倍query行；
固定行数时请求数不同，页表工作集也可能不同，辅助对照不能消除所有变量。

交互项例如 `AB=t(v110)-t(v100)-t(v010)+t(abcdf)`，单位μs，
负值表示在该输入下组合降低耗时更多；它不是某阶段的独立耗时。
优先看P50/P95、原始样本与重复采集，不能仅凭一次均值宣称优化成功。

## 6. 对胜出版本做msopprof瓶颈分析

独立入口 `run_msopprof.py` **内部不创建torch profiler**。
不要用外部msopprof包住 `profile_mtp_matrix.py` 或 `profile_plan_variants.py`。
先以普通无打点库排名，再对ABCDF与胜出版本分别采硬件指标/PC Sampling。
不再使用clock写GM推算阶段占比。

先检查服务器实际工具能力，按help支持的名字选项运行：

```bash
msprof op --help > /tmp/abcdf_msopprof_help.txt
grep -E 'PcSampling|PCSampling|BasicInfo|Source|replay-mode|mstx|warm-up' /tmp/abcdf_msopprof_help.txt
```

先BasicInfo，warmup0确保首个目标就是随后校验的调用：

```bash
msprof op --aic-metrics=BasicInfo --replay-mode=application --warm-up=0 \
  --kernel-name=sparse_kv_plan_kernel_plan_abcdf --launch-count=1 \
  --output=/dev/shm/abcdf_basic_mtp3 \
  python3 -X faulthandler run_msopprof.py --device 0 --implementation abcdf \
  --requests 32 --mtp 3 --warmup 0 --repeat 1 \
  --scenario full_cache_75pct --report results/basic_mtp3.json
```

正式硬件指标：MSTX筛选已完成预热后的目标调用，恢复在范围外。
MSTX接口不支持时直接报错，不静默采错范围。

```bash
msprof op --aic-metrics=Default --replay-mode=application --warm-up=0 \
  --mstx=on --mstx-include=abcdf_target \
  --kernel-name=sparse_kv_plan_kernel_plan_abcdf --launch-count=1 \
  --output=/dev/shm/abcdf_metrics_mtp3 \
  python3 -X faulthandler run_msopprof.py --device 0 --implementation abcdf \
  --requests 32 --mtp 3 --warmup 10 --repeat 1 --mstx \
  --scenario full_cache_75pct --report results/metrics_mtp3.json
```

以上命令为服务器操作模板，不表示已在本分支执行成功。
切换MTP=0时换输出目录；切换候选必须同时换implementation和精确kernel名。
例如s09是 `sparse_kv_plan_kernel_scan_s09`，列表在manifest中。

源码映射需要独立构建，保持-O3，只增加设备-g：

```bash
PYTHON=python3 JOBS=3 bash build.sh --soc=ascend950 --debug-info
python3 checks/check_mtp_scan.py --device 0 --production \
  --library build_profile/libsimt_plan_variants_ops.so \
  --report results/debug_info_correctness.json
```

对该库重新通过门禁后，上述msprof命令增加
`--library build_profile/libsimt_plan_variants_ops.so`，
分别将指标换为 `PcSampling` 和 `Source`，输出目录分别命名。
PcSampling的大小写、Source的支持程度以安装版本help和实际结果为准。
这里使用application replay，不启用range replay；范围筛选与重放是两件事。
不要开启kill提前结束应用，以免跳过正确性检查。

重点把热点映射到hash初始化/插入、LRU分类、跨warp Scan、Miss压缩和写回，
结合stall类型、访存/冲突指标与各因素消融结果判断瓶颈。
PC样本比例不是精确阶段周期占比；SIMT VF也不能承诺通过InstrTimeline看到完整阶段流水。
可选 `--resource-usage` 使用bisheng的 `--cce-res-usage` 输出资源占用，
先确认本机编译器支持，不将资源统计当作时延结果。

## 7. 旧检查与结果边界

`checks/check_plan_goldens.py`、`check_plan_layouts.py`、
`check_plan_profiling.py` 已适配本轮catalog，可在服务器做辅助检查。
旧 `check_plan_kernels.py` 的CPU shim不支持本轮Ballot/Reduce/整段Scan，
在本分支会明确拒绝执行，避免拿原版源码模拟冒充新候选验证。
设备正确性统一用 `checks/check_mtp_scan.py`；CPU检查不能替代它。

历史 `csrc/plan_variants/` 其他版本保留作来源记录，本次CMake不构建它们。
本轮只比较Plan，不评估真实模型接受率、KV搬运、SFA或端到端吞吐；
未在NPU通过的候选不得用于性能结论。

官方API/工具参考：
[Ballot](https://asc.gitcode.com/api/SIMT-API/Warp_functions/Warp_vote_functions/asc_ballot.html)、
[Popc](https://asc.gitcode.com/api/SIMT-API/math_functions/integer_math_functions/__popc.html)、
[Reduce](https://asc.gitcode.com/api/SIMT-API/Warp_functions/Warp_reduce_functions/asc_reduce_add.html)、
[CANN9.1 msOpProf指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md)、
[官方工具当前源码文档](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)。
