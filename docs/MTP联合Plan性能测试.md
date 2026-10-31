# Joint Plan 输入错写修复与三方 MTP profiling

## 1. 本轮修改与边界

基线：`de09a0d20b25c7dc5b7c98b3c0f571dbdb97fefd`。
当前分支：`feat/mtp-joint-plan-profile`，保留原分支上的工作。

### 线程1发布的规避修复

`rowModes`已经是VF参数；本次不增加VF入口参数，不增加tensor或workspace。
从这个参数显式传递`__gm__ int32_t* modeOutput`，经过ProcessRequestRun到PlanJointUnion。

仅末尾逐行循环中的模式发布改成：

```cpp
if (thread == 0) {
    int64_t remaining = misses - (row - first) * a.topk;
    a.missCount[row] =
        remaining <= 0 ? 0 : (remaining < a.topk ? remaining : a.topk);
}
if (thread == 1) {
    modeOutput[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;
}
// 原JointTokenValid、哈希查询及currentSlots循环继续位于本逐行循环中。
```

其他rowModes读写、request reset、LRU、查询循环和barrier保持原业务逻辑。
没有Tag、阶段拆分循环、CPU归一化或只读输入恢复。
上游原文件保留；身份检查通过精确补丁模型核对仅允许的修复，没有关闭指纹校验。

用户的服务器实验：线程0版本复现只读visible_seq_lens被模式值覆盖；
线程1候选通过100次原始语义检查与18工况profiling。
这是有实验依据的规避方案，**底层根因仍未确认**，不宣称证明编译器Bug，
也不解释为补足跨线程同步。本开发机的CPU检查不替代新构建产物的设备验证。

## 2. 三个实现的区别

| 名称 | 实际设备路径 | 接口与元数据 |
|---|---|---|
| abcdf | 原`csrc/plan_variants/13_abcdf/kernel.cpp`及公共实现，mask47、1024线程 | 原15个tensor+5个属性；不产生row_modes、residentRows或residentBlockTable |
| private | 联合工程关闭合并，仍调用RunPrivateRow | 保留联合调度、模式处理、映射发布和原同步 |
| joint | 联合去重，容量足够时共享首行resident，溢出回退；线程1发布模式 | 保留每行currentSlots与联合owner/page协议 |

abcdf的原kernel、variant_config、公共实现和原plugin均未改写；
它是独立编译单元和真实独立kernel，不是private别名。
三者在同一共享库内链接，各有唯一kernel名：

```text
abcdf   sparse_kv_plan_kernel_plan_abcdf
private sparse_kv_plan_private_kernel
joint   sparse_kv_plan_joint_kernel
```

abcdf保留原接口的逐次tiling准备/上传，private/joint采用prepared上下文。
本轮比较设备kernel Duration，上传、状态恢复、校验和CPU等待不加入目标耗时。
不能把这些结果解释为三种接口的端到端调用延迟。

独立ABCDF按原接口核对3个持久状态、currentSlots、miss_count和有效miss前缀；
不存在的联合元数据不伪造、不加进独立kernel。private/joint额外核对所有联合协议字段。

## 3. 主矩阵和输入控制

默认requests=32、TopK=2048、capacity=4096、max_token=262144、
block_size=64、seed=2026，MTP表示draft token数：

| MTP | q_len | query_rows |
|---|---:|---:|
| 0 | 1 | 32 |
| 1 | 2 | 64 |
| 3 | 4 | 128 |

MTP=0时joint列同样关闭联合规划。
默认三个场景：full_cache_75pct、full_cache_90pct、full_cache_stable_hit；
两种重合度：shared、partial，共18个输入工况，每工况三路各50个正式样本。

沿用原用例生成器。固定请求数时三种MTP的第0行query、初始resident、LRU和页表相同。
三路共用同一组只读设备输入，各自持有独立的可变状态，调用前恢复相应初始状态。
warm joint按可联合/溢出布局初始化模式，避免把首次布局切换的冷加载冒充稳定命中场景。

visible=context+p+1；MTP开启时stable=context。
每行保留自身TopK顺序、重复位置和因果可见性。
后续行加载了未来token，不能使前面的行读取它。
shared高度重合，partial按位置错开历史选择；disjoint可选，用于大并集和容量回退。
默认生产尺寸下shared的U为2048/2049/2051，partial为2048/2560/3584；
disjoint在MTP=3时可溢出。这些是构造输入的集合大小，不是设备性能结果。

场景名称指第0行历史构造的名义命中率；新token和其他位置的选择影响实际miss。
full_cache_stable_hit的稳定历史命中，新token仍miss，不表示整次零搬运。
报告记录实际每行miss、总descriptor、U、共享/回退组数和逻辑输入SHA256。

## 4. 验证门禁

- 既有重复token、非法来源、因果可见性、LRU、capacity边界、UB/GM及设备保护区保留。
- 既有共享→溢出→共享、请求变化、非连续请求片段和NPU自身状态连续续用保留。
- 每个正式工况先做污染审计，所有预热调用和正式样本均校验只读输入与完整输出。
- 默认100次错写回归，尺寸固定为已复现的32请求/MTP1/2048/4096/262144/shared/75%。
  每次只恢复可变状态；**从不恢复只读输入**。发现一次错误立即停止，不能继续采样。
- 默认不跑1000次；可用--regression-repeat调整正整数次数。
- 可复用同主机、同设备、同库SHA、同源码/编译身份、覆盖三路且回归次数足够的完整门禁报告。
  仅regression-only报告不能代替完整门禁。
- 失败在运行机器保存`*_failure_state.npz`，包括实际/期望输入、可变状态及保护区。
  二次保存失败只警告，保留原异常；不恢复输入掩盖错写。

CPU模型检查只验证标量语义、补丁范围、调用接口及CSV/报告逻辑；
不能复现或证明NPU编译后错写已经消失。

## 5. 服务器一次性操作

先进入包含本次修改的仓库根目录。产物使用持久化/workspace，不上传公司数据或二进制。
源码/共享库所在文件系统需允许加载.so。以下明确使用设备2和指定Python。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/workspace/joint_plan_results/tmp
mkdir -p "$TMPDIR" /workspace/joint_plan_results
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
set -eo pipefail

# 同一-O3和既有SIMT选项，构建三条真实设备路径
bash build_joint_plan.sh --soc=ascend950 2>&1 | tee /workspace/joint_plan_results/build.log

# CPU模型/源码身份/CSV/报告回归，不调用NPU
"$PYTHON" checks/check_joint_models.py 2>&1 | tee /workspace/joint_plan_results/cpu_checks.log

# 默认一轮100次错写回归，再做完整三方门禁
"$PYTHON" -X faulthandler checks/check_joint_plan.py --device 2 \
  --regression-repeat 100 --temp-dir "$TMPDIR" \
  --report /workspace/joint_plan_results/correctness.json \
  2>&1 | tee /workspace/joint_plan_results/correctness.log

# 主矩阵：复用刚才的完整门禁，避免重复跑100次；逐样本检查始终开启
"$PYTHON" -X faulthandler profile_joint_mtp.py --device 2 \
  --implementations abcdf private joint \
  --requests 32 --mtp 0 1 3 --comparison same_requests \
  --topk 2048 --capacity 4096 --max-token 262144 --block-size 64 --seed 2026 \
  --scenarios full_cache_75pct full_cache_90pct full_cache_stable_hit \
  --overlaps shared partial --warmup 10 --repeat 50 \
  --correctness-report /workspace/joint_plan_results/correctness.json \
  --temp-dir "$TMPDIR" --output /workspace/joint_plan_results/formal \
  2>&1 | tee /workspace/joint_plan_results/formal.log
```

每一步确认退出码0再继续；pipeline的退出码由pipefail保留。
100次是回归调用数，不是正式性能样本数；正式保留18*3*50=2700条目标kernel样本。
门禁不能复用时脚本明确拒绝，不静默跳过。
未提供--correctness-report时，profiling入口会自行运行一次完整门禁与100次回归。

如只先定位输入错写，可单独运行：

```bash
"$PYTHON" -X faulthandler checks/check_joint_plan.py --device 2 \
  --regression-only --regression-repeat 100 --temp-dir "$TMPDIR" \
  --report /workspace/joint_plan_results/input_write_only.json
```

这份报告不能拿来跳过完整三方门禁。

## 6. 报告与原始数据索引

唯一session目录位于指定output下，包括：

- report.md：各路P50/P95、样本数、descriptor数量、设备/库/编译身份和原始CSV路径。
- data_index.json：按工况、MTP、实现索引精确kernel名、样本数、原始CSV路径和输入指纹。
- capture_metadata.json：完整原始样本、轮换顺序、只读输入指纹、源码/编译/库身份和门禁。
- 各例summary.json、kernel_summary.csv及原始kernel_details.csv；解析前原始采集保留。
- profile.log与COMPLETE.json。失败时保留现场，不发布成功report.md。

三路按轮换与反序组合，在每6轮覆盖全部顺序；repeat=50时各执行位置次数差至多1。
CSV解析精确匹配三个目标kernel，核对μs单位、一个分组、样本数、同流时序和实际顺序。
TensorMove、tiling上传、辅助kernel、恢复拷贝和CPU等待均不加进目标Duration。

比值方向严格为分子耗时除分母耗时：

- private/abcdf：联合框架逐行路径相对原独立kernel。
- joint/abcdf：联合路径相对原独立kernel。
- private/joint：框架内逐行路径相对联合路径。
- 各实现MTP开启/MTP关闭：同一实现相对自身MTP0。

大于1表示分子耗时更长。没有预填历史58μs或其他性能阈值。
单次Duration处理全部query行；按行或请求除法只是吞吐归一化。
固定行数辅助实验通过--comparison both --fixed-query-rows 32启用，单独标记same_query_rows；
MTP0/1/3对应32/16/8请求，不混入固定32请求的结论。

只有退出码0、complete元数据、COMPLETE.json与report.md同时存在，才是整轮成功。
数据索引本身不是成功标记。没有设备实测时不会生成虚构性能报告。
本轮只测Plan，descriptor减少不直接代表Transfer或整网端到端加速。

## 7. 可选深度profiling

`run_joint_msopprof.py`支持--implementation abcdf/private/joint与--mtp 0/1/3。
它内部不嵌套torch profiler；外部工具应选对应精确kernel名，先通过上述独立门禁。
不要把msprof op套在profile_joint_mtp.py外面。

线程1规避的底层根因、实际生成指令及新的三方时延均需设备侧继续确认。
构建、CPU验证、NPU正确性与性能状态应分别记录，不能互相替代。
