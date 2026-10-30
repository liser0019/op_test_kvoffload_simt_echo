# 新版联合 Plan：MTP=0、1、3 性能测试

## 1. 本分支测什么

分支：`feat/mtp-joint-plan-profile`。
框架基线为 `feat/transfer-factorial-optimizations@a5beb25`，
联合算子来自 `/home/welt/workspace/kvoffload/vllmascend_mtp_joint_plan` 的
`feat/mtp-joint-plan@bba63d05762a467e496c0f8a8ae687dc632a706f`。

默认固定32请求，TopK2048、capacity4096、max_token262144、block64：

| 参数 | 关闭MTP | MTP=1 | MTP=3 |
|---|---:|---:|---:|
| draft token数 | 0 | 1 | 3 |
| 每请求query行数 | 1 | 2 | 4 |
| query总行数 | 32 | 64 | 128 |

同时测两条路径：

- `joint`：MTP开启时按请求组联合去重；并集超过容量则整个组回退逐行规划。
  MTP=0时关闭联合功能，沿用新版的非MTP路径。
- `private`：始终关闭联合功能，逐行ABCDF，作为同输入的对照。

private包含新版的row_modes处理和owner/page映射发布，
不是历史旧库的原始ABCDF入口。两条路径在同一新库内，使用相同计算辅助函数。

本次只执行Plan，不执行真实Transfer、SFA或draft模型。
miss记录总数可体现后续搬运工作量的变化，但不能直接换算Transfer/SFA时延或整网收益。

## 2. 保留了哪些代码

原联合辅助头、配置、ABCDF计算辅助函数和SIMT VF调度逐字保留。
来源文件保存在 `csrc/joint_plan/upstream/`，构建/运行身份检查会核对其哈希及计算区一致性。

接入改动是：

1. 将注册算子入口换为独立kernel直调入口。
2. GM tiling采用逐字段加载，避免跨地址空间整结构体复制。
3. 启动参数显式传动态UB大小，保留1024线程和原同步。
4. private/joint各有唯一kernel名，用于CSV精确过滤。
5. 由调用方持有prepared上下文，预先上传tiling，持有全部tensor。

两个kernel名：

```text
private: sparse_kv_plan_private_kernel
joint:   sparse_kv_plan_joint_kernel
```

没有增加clock打点、阶段写GM或跨核同步。
新库由 `build_joint_plan.sh` 单独构建，不需要安装vLLM，也不编译Transfer或旧Scan候选。

## 3. 输入控制与TopK重合模式

相同seed、请求数和场景下，MTP=0/1/3共用：

- 每个请求的第0行query；
- 原始resident内容及LRU顺序；
- request ID和随机物理页表。

每个位置p的可见长度为 `context+p+1`。
MTP开启时stable=context；关闭时stable=visible。
各行query含自身可见的新token，不能通过未来行的共享缓存绕过因果可见性。
预埋的相同旧投机后缀在三种配置下均会失效。

| overlap | 构造 | 作用 |
|---|---|---|
| shared | 历史TopK高度重合，新token按位置增加 | 联合规划的主要收益场景 |
| partial | 每个后续位置将历史窗口右移topk/4 | 覆盖稳定历史选择也不同的情况 |
| disjoint | 历史窗口右移topk | 压测大并集及溢出回退成本 |

在默认生产尺寸、所有来源有效时，每请求的理论并集大小为：

| MTP | shared | partial | disjoint |
|---|---:|---:|---:|
| 0 | 2048 | 2048 | 2048 |
| 1 | 2049 | 2560 | 4095 |
| 3 | 2051 | 3584 | 8186 |

这些是输入构造的集合大小，不是实测时延。实际报告重新计算并核对各组U。
MTP=1时两行最多2*TopK=4096项，因此默认容量下disjoint也不必溢出；不能只凭模式名宣称回退。

默认场景为full_cache_75pct、full_cache_90pct、full_cache_stable_hit；均从满resident开始。
前两者名称指第0行历史构造的名义命中比例，不保证其他query行也有相同命中率。
第三种是第0行历史稳定部分全命中，新增token仍miss，不能称为整次调用零miss。
每路实际miss_count、去重后搬运记录数均写入报告。
可显式加入cold_reset，核对请求身份变化后的冷路径。

warm联合输入按当前实际布局初始化row_modes：能联合的组为leader/follower；
溢出组保持private。这样重复采集测量稳定布局；布局转换成本在另外的连续状态门禁验证。
两路原始缓存内容相同，编码/owner映射不同，不能要求输入状态tensor逐字相同。

## 4. 正确性门禁

`checks/check_joint_plan.py`覆盖：

- MTP=0/1/3、三种重合模式、冷/热状态。
- U=capacity及U=capacity+1边界。
- 重复query、未来位置、非法token/request/page、非2次幂block。
- active_rows负值/0/截断到组中间/超过上限、非活动尾部保持不变。
- 超过64个请求组、一核顺序处理多组与scratch复用。
- 同一shape下private走UB、joint走GM。
- 共享→溢出回退→共享、请求身份变化、非连续同请求片段。
- prepared上下文关闭、重绑定、属性/形状改变在下发前拒绝。

连续用例直接使用上一轮NPU写出的持久状态，不从golden重写设备状态。
每个矩阵正式工况还会独立做3轮污染审计，并在采集后再检查。
每个可写tensor与workspace都有前后保护区；输出和scratch会填充哨兵，排除旧答案假通过。

private使用独立列表/字典golden核对持久状态、currentSlots、有效miss前缀。
joint由CPU集合模型计算合法并集与应缺失集合，再检查：

1. 每个唯一missing恰好输出一次，分配槽符合可淘汰LRU前缀。
2. 全量slot_to_token、LRU、last_req_id、row_modes正确；follower未被擅自改写。
3. 每行currentSlots通过自己的owner，指向自身选择的合法token；未来token仍为-1。
4. residentRows与供SFA使用的residentBlockTable一致。

联合miss可以有不同的桶遍历顺序，不要求它与private的槽号和LRU逐字相等。
这验证Plan的索引/元数据语义，没有声称验证了真实KV payload或SFA数值结果。

## 5. 服务器构建和小尺寸试跑

以下命令仅在950PR服务器执行。开发机遵守禁跑要求，未执行Python、CPU检查、构建或NPU测试。
上游实现此前的CPU回归结果不等同于本直调移植已通过验证。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh  # 按实际安装位置调整
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
ulimit -c 0

# 独立新库；不要运行旧build.sh后误用旧共享库
PYTHON=python3 JOBS=3 bash build_joint_plan.sh --soc=ascend950

# 服务器CPU辅助检查：输入控制、语义拒错、CSV和统计
python3 checks/check_joint_models.py

# NPU专项门禁，默认逻辑卡0
python3 -X faulthandler checks/check_joint_plan.py --device 0 \
  --temp-dir /dev/shm/joint_tmp --report /dev/shm/joint_gate/report.json

# 小尺寸矩阵：包括真实的并集溢出
python3 -X faulthandler profile_joint_mtp.py --device 0 \
  --requests 2 --topk 7 --capacity 16 --block-size 8 --max-token 512 \
  --mtp 0 1 3 --overlaps shared partial disjoint \
  --warmup 2 --repeat 4 --verify-each-sample \
  --temp-dir /dev/shm/joint_tmp --output /dev/shm/joint_smoke
```

输出/临时目录可写性、剩余字节及inode检查先于NPU初始化；阈值可配置。
Plan-only不分配之前Transfer测试的9GiB Host池。
不检查源码分区是否有空余空间，不自动清理文件或切换采集目录。
/dev/shm是临时存储，结束后转存报告和原始采集。

## 6. 正式MTP对比

```bash
python3 -X faulthandler profile_joint_mtp.py --device 0 \
  --requests 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --mtp 0 1 3 --comparison same_requests \
  --overlaps shared partial disjoint \
  --warmup 10 --repeat 50 --verify-each-sample \
  --temp-dir /dev/shm/joint_tmp --output /dev/shm/joint_formal
```

默认三种满缓存构造，以上显式加入三种重合模式；每例都采private/joint各50个样本。
只看新版可加 `--implementations joint`，但建议第一次保留private对照。
只想先看高度重合可改为 `--overlaps shared`，减小矩阵。

固定行数辅助对照：

```bash
python3 -X faulthandler profile_joint_mtp.py --device 0 \
  --mtp 0 1 3 --comparison both --requests 32 --fixed-query-rows 32 \
  --overlaps shared --scenarios full_cache_75pct \
  --warmup 10 --repeat 50 --temp-dir /dev/shm/joint_tmp \
  --output /dev/shm/joint_row_control
```

same_query_rows时MTP=0/1/3分别是32/16/8请求，各32行。
请求数和页表工作集不同，因此它是辅助控制，不能消除所有输入差异。

## 7. 如何看报告

每次目标调用前恢复对应布局的初始状态；同例两版共用同一份query/页表输入。
tiling及设备buffer在采集前准备完毕；热路径没有结果D2H判断或tiling上传。
每组手动预热10次，profiler额外1个warmup step，再记录50个active step。
private/joint逐轮交换先后，MTP顺序在工况间正反轮换。
开启逐样本检查会增加计时外的D2H和等待；两路均采用相同条件，报告仍只统计目标kernel。

目录内容：

- report.md：各例P50/P95/样本数、miss总数、共享/溢出组数、两路比值、MTP开关比值。
- capture_metadata.json：源码/编译/库身份、实际U、shape、逐行miss与原始样本。
- COMPLETE.json：整轮成功标记。
- 各例kernel_summary.csv、summary.json、原始kernel_details.csv及原始采集文件。
- profile.log：采集与解析日志；失败会报告具体目录、日志路径和首个可定位异常。

统计严格核对精确kernel名、μs单位、分组、样本数、同流时序和实际下发顺序。
不把TensorMove、状态恢复或tiling拷贝计入Plan duration。
一次调用时间包含设备内的联合去重、容量判断、回退与owner/page映射发布。
接口端到端时间与Event时间均不是本轮排名口径。

同一MTP的 `private P50 / joint P50` 衡量联合规划相对逐行路径的变化。
同一实现的 `MTP开启 P50 / MTP关闭 P50` 衡量开启MTP后的整次调用代价。
固定32请求时工作量增加到2/4倍query行，不把这个比值单独称为算法退化率。
除以请求数或行数只做吞吐归一化，不能称作独立请求延迟。

只有退出码0、complete元数据、COMPLETE.json与report.md同时存在才表示整轮完成。
失败保留原始目录，不生成成功报告，不预填预计收益或性能阈值。

## 8. 单版本msopprof深度入口

`run_joint_msopprof.py`内部不创建torch profiler，也不提前执行同名kernel门禁。
先单独运行并确认passed，再套外部工具；不要套在profile_joint_mtp.py外面。

```bash
python3 -X faulthandler run_joint_msopprof.py --device 0 \
  --implementation joint --mtp 3 --requests 32 --overlap shared \
  --warmup 0 --repeat 1 --temp-dir /dev/shm/joint_tmp \
  --report /dev/shm/joint_runner/check.json

mkdir -p /dev/shm/joint_tmp
export TMPDIR=/dev/shm/joint_tmp  # 外部工具的临时目录也须先设置

msprof op --aic-metrics=BasicInfo --replay-mode=application --warm-up=0 \
  --kernel-name=sparse_kv_plan_joint_kernel --launch-count=1 \
  --output=/dev/shm/joint_basic_mtp3 \
  python3 -X faulthandler run_joint_msopprof.py --device 0 \
  --implementation joint --mtp 3 --warmup 0 --repeat 1 \
  --temp-dir /dev/shm/joint_tmp --report /dev/shm/joint_runner/basic_mtp3.json

msprof op --aic-metrics=Default --replay-mode=application --warm-up=0 \
  --mstx=on --mstx-include=joint_plan_target \
  --kernel-name=sparse_kv_plan_joint_kernel --launch-count=1 \
  --output=/dev/shm/joint_metrics_mtp3 \
  python3 -X faulthandler run_joint_msopprof.py --device 0 \
  --implementation joint --mtp 3 --warmup 10 --repeat 1 --mstx \
  --temp-dir /dev/shm/joint_tmp --report /dev/shm/joint_runner/metrics_mtp3.json
```

MTP0/1分别修改--mtp和输出目录。逐行路径需同时换implementation=private和精确kernel名。
工具参数支持以服务器CANN版本的 `msprof op --help` 为准。
PC Sampling/Source可在支持的工具版本中分别采集，不把深度计数采集耗时混入普通排名。
源码映射先 `bash build_joint_plan.sh --debug-info`，保持-O3，
所有运行命令显式加 `--library build_joint_plan_profile/libmtp_joint_plan_ops.so`，重过门禁后再采。
工具若提前终止应用而未完成runner检查，不能把缺失passed当作通过。

## 9. 阅读顺序

1. joint_plan_cases.py：输入构造、U/容量判断与独立语义检查。
2. csrc/joint_plan/sparse_kv_joint_plan.h：BuildJointUnion → PlanJointUnion → ProcessRequestRun。
3. csrc/joint_plan/kernel.cpp：原ABCDF辅助函数、请求组调度、直调入口。
4. csrc/joint_plan/plugin.cpp：prepared绑定、tiling、生命周期和下发。
5. checks/check_joint_plan.py：边界与NPU状态连续续用。
6. profile_joint_mtp.py：目标采集、CSV匹配、MTP对照报告。

本轮交付状态：源码与静态一致性检查；编译、CPU回归、NPU正确性和性能均待服务器执行。
