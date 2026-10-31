# Joint Plan A/B/C/D 优化实验

本分支基于 `2556a1f`，保留原独立 ABCDF、private、joint 作为基线。原 `csrc/joint_plan/` 和 `csrc/plan_variants/` 不改动。这里的 A/B/C/D 是 **Joint 实验的名字**，和原 ABCDF 的六个优化字母不是一回事。

目前是待上机的候选实现，不能据此宣称获得加速。你提供的同场设备数据（ABCDF 约51–54 μs、private约75 μs、MTP1 joint约150 μs）是优化起点，不是本机复测结果。

## 先读哪些文件

1. `joint_variants.py`：实验名、位掩码、默认候选。
2. `csrc/joint_variants/sparse_kv_joint_plan.h`：B/C/D 的实际修改；原实现保留在相邻 `csrc/joint_plan/`。
3. `csrc/joint_variants/kernel.cpp`：A 的 VF 隔离；前半部分原 ABCDF helpers 保持逐字一致（只调整 include 路径）。
4. `joint_plan_cases.py`：输入、每行来源/可见性及独立语义检查。
5. `joint_plan_common.py` → `profile_joint_mtp.py`：状态恢复、调用、逐次校验、CSV 匹配与统计。

构建脚本生成的包装只做宏定义和符号改名，算法不藏在生成器中。每个 mask 编译成独立设备入口。`JointDirectBody` 也做符号隔离，避免不同优化对象链接时共用同名 inline 函数。宿主 `PreparedVariant` 持有全部 tensor 和 tiling，mask 不可变，没有全局设备缓存。

## 这次要区分的成本

本算子无跨卡通信、无跨 AIV 数据同步，这两类流水分析不适用。并行粒度仍是每个连续请求组一个 AIV；32请求的联合规划仍最多32个有实际工作的AIV。本轮没有跨核拆组，也没有引入全局同步。

只有 Duration 无法确定是 scalar、访存、SIMT计算还是并行/等待受限：四类都不能排除。源码能证明额外工作存在，但不能证明各占多少微秒，也不能直接断定寄存器 spill。A 是资源隔离实验；B/C/D 是减少具体工作量的实验。

生产 `T=2048,C=4096,threads=1024`：联合哈希8192桶，scratch约80KiB加288B scan。联合路径先建并集，再扫4096个resident槽，最后扫8192桶分配miss；每请求的2/4条query由同一个block处理。

| 点 | 修改 | 主要对照 | 限制与成本 |
|---|---|---|---|
| A | 编译期拆出非联合VF；`private_a`的设备kernel只包含非联合VF | `private_a/private`，尤其MTP=0 | 保留完整元数据发布和原同步；不保证能消除全部约24μs差距 |
| B | 插入时将桶号暂存在本行`currentSlots`，最后直接读`slots[bucket]` | `joint_b/joint` | 增加一次GM写/读，换掉二次来源检查/哈希查询；可能反而变慢 |
| C | 只对reset或失效的resident槽写回 | `joint_c/joint` | 命中/未失效槽不做同值写；miss分配仍真实写入 |
| D | 每线程累加本行新增数，行末做一次全block scan | `joint_d/joint` | 减少scan，但可能增加原子插入重叠/竞争；溢出晚到本行末发现 |

A 的 `joint_a` 仍保留运行时 MTP 开关：MTP=0调用专用非联合VF，MTP开启调用联合VF；同一个joint kernel仍包含两个VF，所以不能假定其资源占用与只含非联合VF的`private_a`一样。重点比较`private_a`，再结合编译资源报告判断，而非只看源码函数大小。

D 只改变 BuildJointUnion 的计数频率，既不删 resident/miss scan，也不直接认定“不可能溢出”。T2048时，W条query的构建scan从2W次变成W次。容量扫描4次、哈希桶分配扫描8次维持原样。T<=1024时D不减少构建scan。

D 的安全条件：宿主保证T<=C，H为不小于2C的2次幂。每行开始已确认U<=C，本行最多新增T，所以本行插入总唯一token<=C+T<=2C<=H。所有插入仍用有界原子CAS，行末所有线程参与scan和barrier，随后才读哈希或整组回退。精确C和C+1边界必须继续测试。

B 不依赖token顺序或跨行相同前缀：每一个位置在构建时按**本行**的visible_seq_lens和页表判断，非法位置写-1；合法位置记稳定桶号。整个调用期间keys不移动/删除。若并集溢出，RunPrivateRow重新计算所有行并覆盖这些暂存桶号。它不复用上次调用答案、不修改只读输入。

## 为什么不照搬附件中的结构性建议

附件中的论文日期、开源方案及性能数字未作为已核实证据使用。强制所有MTP行共用TopK会改变indexer输出语义；假设TopK有序再做归并也缺少输入契约，本轮均不采用。

尾部KV独立缓冲、按请求持有resident、跨核构建/查询，涉及上层物理布局、Transfer和SFA协议，不能只在Plan里改后就称端到端兼容。本轮保留现有容量、回退和stable prefix语义，优先用可归因的实验测量成本。

## indexer 后各行不相同怎么测

所有候选读取相同逻辑输入，但可变状态分别分配。请求数默认32，MTP=0/1/3对应32/64/128行，容量4096不变。

- `shared`：历史集合高度重合，是对照工况，不是算法前提。
- `partial`：历史集合部分重合。
- `independent`：每条follower独立随机抽取历史token，并打乱整行顺序；不共享位置前缀。候选池为2T，控制其接近缓存容量，不代表生产分布。
- `disjoint`：历史集合低重合；生产MTP3通常溢出，验证完整private回退并记录成本。

默认profiling跑partial/independent/disjoint；需要复现上一轮可显式选择shared/partial。报告包含实际并集大小、是否能装入C、每行与首行的集合交集、实际miss/descriptor数和逻辑输入SHA256。

三种MTP的首行query、页表、初始resident和LRU保持一致（同一overlap下）。每行仍有独立因果可见性；旧投机后缀按stable prefix失效。`full_cache_stable_hit`指历史部分命中，不保证含新token后的总miss=0；场景标签的75%/90%不是每个follower的实际命中率，报告的逐行miss才是实际值。

这里只测Plan。并集节省descriptor可能帮助Transfer，但不能将descriptor数量下降或Plan变快直接换算成端到端加速。

## 正确性与源码身份

- 保留原线程1发布rowModes的规避方案；根因未确认。这不是补同步。
- 每个选择的协议候选各跑一轮默认100次故障尺寸回归；可用`--regression-repeat`调整。
- 小尺寸边界、精确C/C+1并集、T>1024多tile、UB/GM、非法来源、重复token、inactive行、一核多组和连续状态都检查。
- 连续调用直接续用上次输出，覆盖joint/private回退和request reset。
- 所有输入不变性、输出保护区、逐位置currentSlots、LRU、miss及owner/page映射检查继续启用。
- prepared校验mask范围、tensor身份、形状/属性、生命周期；复用报告要求同设备、同库、同源码且覆盖全部本次候选。
- 源码指纹覆盖实验源文件、生成器、候选目录和测试脚本。原三个基线仍执行原来的精确身份检查。
- CPU仿真调用真实C++ VF/helper，用64个宿主线程和UBSan，不模拟NPU地址空间、DCache、编译器或1024线程时序。不能替代NPU门禁。

## 服务器运行（设备2，CANN9.1.0，-O3）

在此分支仓库根目录运行。结果和失败现场只写本机`/workspace`，不要上传输入、二进制或profiler原始数据。脚本会预检输出及临时目录空间。

```bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/workspace/joint_abcd_results/tmp
mkdir -p "$TMPDIR" /workspace/joint_abcd_results
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

# 构建原三路基线和15个组合；编译量比上一轮大，-O3不变。
bash build_joint_plan.sh --soc=ascend950 --resource-usage \
  2>&1 | tee /workspace/joint_abcd_results/build.log

"$PYTHON" checks/check_joint_models.py

# 单点+全组合候选，先验证再正式采集。
NAMES=(abcdf private joint private_a joint_a joint_b joint_c joint_d joint_abcd)
"$PYTHON" -X faulthandler checks/check_joint_plan.py \
  --device 2 --implementations "${NAMES[@]}" --regression-repeat 100 \
  --temp-dir "$TMPDIR" --report /workspace/joint_abcd_results/gate.json \
  2>&1 | tee /workspace/joint_abcd_results/gate.log

# 短跑。复用同库门禁，不再次做100次回归。
"$PYTHON" -X faulthandler profile_joint_mtp.py \
  --device 2 --implementations "${NAMES[@]}" --mtp 0 1 3 \
  --requests 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --overlaps partial independent disjoint --scenarios full_cache_75pct \
  --warmup 2 --repeat 3 --correctness-report /workspace/joint_abcd_results/gate.json \
  --temp-dir "$TMPDIR" --output /workspace/joint_abcd_results/smoke \
  2>&1 | tee /workspace/joint_abcd_results/smoke.log

# 正式同场比较；三个满缓存历史命中场景、三种重合、三种MTP。
"$PYTHON" -X faulthandler profile_joint_mtp.py \
  --device 2 --implementations "${NAMES[@]}" --mtp 0 1 3 \
  --requests 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --overlaps partial independent disjoint \
  --scenarios full_cache_75pct full_cache_90pct full_cache_stable_hit \
  --warmup 10 --repeat 50 --correctness-report /workspace/joint_abcd_results/gate.json \
  --temp-dir "$TMPDIR" --output /workspace/joint_abcd_results/formal \
  2>&1 | tee /workspace/joint_abcd_results/formal.log
```

需要完整因子实验时，将NAMES替换成下列列表，并重跑同一套门禁和采集（使用新的输出目录/报告，不能拿9路门禁冒充19路已验证）：

```bash
NAMES=(abcdf private joint private_a \
  joint_a joint_b joint_ab joint_c joint_ac joint_bc joint_abc \
  joint_d joint_ad joint_bd joint_abd joint_cd joint_acd joint_bcd joint_abcd)
```

不必默认把所有组合都在所有输入上长跑。先看单点结果，再从15个组合中选择候选；如需正式交互效应分析，再同场跑完整目录。多候选采用循环轮换，50次各执行位置次数之差至多1；三候选沿用六种排列。状态恢复、校验及tiling上传不混进精确目标kernel Duration。

## 如何读结果和继续定位

`joint_matrix_*/report.md`报告每个实现P50/P95/count、候选/原协议路径、候选/ABCDF、相对自身MTP=0比值。比值大于1表示候选更慢。`data_index.json`索引原始CSV；`capture_metadata.json`保留逐样本Duration、实际顺序、输入/设备/库身份。

只接受退出码0、complete状态、COMPLETE.json和report.md都存在的整轮结果。失败即停，保留原始采集和输入错写现场，不恢复错误输入继续跑。

1. 先看MTP=0的private_a对private。若有明显收益，再结合build.log中的资源使用信息定位寄存器/栈等变化；没有资源证据不能叫“spill已解决”。
2. B/C/D分别对joint，按MTP和overlap分组看。不要只选shared或只报最好数字。
3. D收益应主要出现在T>1024；GM路径和溢出路径必须分别看。
4. 组合收益不默认可加，ABCD也不默认最快。
5. 如果仍有较大差距，再抓PC Sampling/流水指标，决定是否进行跨核联合规划等更大结构变更。

已有`run_joint_msopprof.py`支持所有新名字。例如外部msopprof的application部分可使用：

```bash
"$PYTHON" -X faulthandler run_joint_msopprof.py \
  --device 2 --implementation joint_abcd --mtp 3 --overlap independent \
  --warmup 0 --repeat 1 --temp-dir "$TMPDIR" \
  --report /workspace/joint_abcd_results/msopprof/joint_abcd.json
```

这条命令本身仅运行目标并校验，**不自动采集msopprof**；将它置于服务器已经验证的msopprof命令下，按实际版本选择指标。不要在其中再嵌套torch profiler。先以普通-O3库得到排名；源码映射另用`--debug-info`构建，并通过`--library build_joint_plan_profile/libmtp_joint_plan_ops.so`明确选择对应库，同库重新验证。
