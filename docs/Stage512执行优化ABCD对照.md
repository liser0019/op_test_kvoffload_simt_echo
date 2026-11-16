# Stage512 执行优化：四因素全组合

本轮冻结参考为 `joint_layout_stage_abcd_512`（上一轮热路径最优候选），不是较早的 FUSED 或 stage16。仍是32请求、TopK2048、max_token262144、seed2026、共同升序overlap40输入；MTP0/1/3/7的capacity和历史候选池分别为4096/4096/8192/16384。

本轮只扩展stage布局，512线程；硬件归并、608条/路包长、16位query→UID、GM映射暂存、私有/联合协议、完整miss尾部约定均沿用父版。以下字母属于**本轮**，与上一轮ABCD无关。性能须以新候选与同场冻结参考的目标kernel Duration比较。

## 四个独立因素

| 因素 | 改动 | 直接作用范围 |
|---|---|---|
| A | 连续flags以`LoadAlign<uint16_t, DIST_UNPACK_B16>`加载并零扩展；连续owner用`LoadAlign/StoreAlign`替代Gather/Scatter | 热MTP0/1/3/7 |
| B | 只做hit前缀，用`evictBefore = 全局LRU位置 - hitBefore`得到evict排名和总数 | 热MTP0/1/3/7 |
| C | 已分配miss槽直接覆盖；其余槽的失效清理与LRU写回合并，去掉单独整表清理和其后的barrier | 热MTP0/1/3/7 |
| D | capacity4096使用512段精确目录；capacity8192/16384继续1024段 | 直接作用于热MTP0/1 |

A保留间接partition查询的Gather，因为该地址确实依赖miss排名。所有基址和64项步长满足32B对齐。flags每次读取64个u16；owner区容量为64的倍数且u≤capacity，最后不足64个UID时也可在owner区内安全加载完整64项，但只对有效掩码中的miss写回。高位失效标记按无符号数零扩展，不能做带符号扩展。

B依赖每个LRU位置是合法槽。分类阶段沿用原拒绝语义，在已有同步前发现越界槽时将状态置失败，不进入单前缀VF。未删除失败检查或回退。正常排列中，每个位置必为hit或evict，即使槽内token无效/空闲，仍属于evict。这里不是推导“唯一token数”，重复resident仍逐物理槽分类。

C依赖既有LRU为物理槽排列的接口契约。partition前`misses`个元素对应已分配的新槽，其余对应未分配槽，因此miss写和失效清理的物理目标集合不相交，线程交错不产生互相覆盖。保留末尾状态发布同步、所有map/owner/分类交接；不改thread1发布模式的已验证规避。重复resident token仍由atomic_max选择LRU-last owner，并保护全部命中槽。**不是直接删除清理，也不是把每个barrier都移除。**

D仍使用精确二分，不引入概率过滤。目录占用owner更早的生命周期，保留“目录所有读取完成→同步→初始化owner”的顺序。4K建513个边界；8K/16K仍建1025个。8K/16K或冷分支观察到的差异不能直接算为D的算法收益。

四个因素均不改变reset冷分支；冷工况仍采集以检查合编后的性能变化，不能把冷耗时变化全部归因于上述热计算。没有PMU/指令数据时，不确定是Scalar、Vector、访存或流水衔接受限。

## 实现清单与空间

参考3路：

- `abcdf_pages`：带页表原独立ABCDF，保留原112KiB scratch门槛；16K时走其GM路径。
- `joint_layout_stage_abcd_512`：冻结主推参考。
- `joint_stage_exec_base_512`：复制父版收尾但关闭本轮四因素的control；不以较慢control代替冻结参考计算收益。

新增15路：`joint_stage_exec_{a,b,c,d,ab,ac,ad,bc,bd,cd,abc,abd,acd,bcd,abcd}_512`，ID229..243；control为228。算法和公共入口ID明确分离，不是别名。归并和冷分支复用冻结controller，仅在显式`JOINT_STAGE_EXEC`分支引入新的两份收尾头文件。

| capacity | 动态UB（含Scan） | 映射暂存GM |
|---|---:|---|
| 4096 | 123680 B | rows×TopK/2个int32 |
| 8192 | 148256 B | 同左公式 |
| 16384 | 197408 B | 同左公式；32请求MTP7时1MiB |

本轮未新增UB区或L2/cache策略，不假定映射常驻L2。`design/stage_exec`保存重跑的存活/分配/物理峰值检查；它是缓冲模型，不是编译器spill证明。

阅读顺序：`joint_variants.py`→`stage_exec_plan.h`→`stage_exec_simd.h`→`layout_hot_impl.h`里的Phase6/VF/Phase9编排→`run_stage_exec_sweep.py`。冻结的`layout_hot_plan.h`和`layout_hot_simd.h`保持原样。

## 测试与报告

一次构建17条联合设备入口和独立ABCDF，共18路。先进行重复5次的针对性门禁，包含输入不变、状态/输出/保护区、容量边界、重复query、重复resident、stable prefix失效、63/64/65 UID尾块全命中、4K/8K/16K连续状态及模式切换。连续调用使用上一轮实际状态。

初筛18路×8工况，默认预热2、正式5样本；正式测量顺序逐轮轮换。以四个热工况相对冻结stage ABCD的P50比值几何平均，选出两个**固定组合**；再保留三条参考，同库独立预热10、正式20样本复测。复测复用同源码/库/设备/输入身份匹配的门禁；每样本完整校验仍保留。初筛和复测样本不混合；慢版本也保留初筛数据。

输出：

- `report.md`：独立复测5路的八工况P50/P95、样本、三条参考比值、联合/回退、descriptor。
- `screen_report.md`：18路完整初筛数据。
- `factor_effects.md`：全组合32个条件对照，例如A→AB、ABC→ABCD；包含编译与交互的kernel净差值，不能相加或称为某阶段独立耗时。
- `selection.json`：15候选按四个热工况的固定排名。
- `screen/`、`confirm/`：各自的CSV、逐样本数据、设备/库/源码身份、输入指纹及墙钟时间；入口为`capture_metadata.json/data_index.json`。

只统计精确目标Plan kernel Duration，包含硬件归并及GM映射暂存；恢复、CPU参考、校验不计入。开始采集后1秒宿主等待以及单位/调用顺序/样本数门禁保留。固定地址预热，不代表生产随机轨迹或冷硬件缓存；未测Transfer/SFA。

## 服务器运行

开发机没有CANN/NPU：本地CPU检查不代表Bisheng编译、真实地址空间、同步、缓存一致性或性能通过。

```bash
set -euo pipefail
RUN=$(mktemp -d /workspace/single_op_test/stage_exec_run_XXXXXX)
git clone --single-branch -b feat/stage512-execution-sweep \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
# 使用上一轮已验证的9.2.1环境；若实际安装目录不同，仅修改此行。
CANN_ROOT=/usr/local/Ascend/cann-9.2.1
source "$CANN_ROOT/set_env.sh"
export ASCEND_CANN_PACKAGE_HOME="$CANN_ROOT"
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
# 选择空闲逻辑卡；以下以0为例。此命令会完成构建、门禁、初筛和独立复测。
bash scripts/test_stage_exec_sweep.sh --device 0 --requests 32 \
  --regression-repeat 5 --screen-repeat 5 --confirm-repeat 20 \
  --output "$RUN/results" --temp-dir "$RUN/tmp"
cat "$RUN/results/report.md"
cat "$RUN/results/factor_effects.md"
```

输出目录必须尚不存在。默认向持久化`/workspace`写结果；空间预检在NPU初始化前执行。不自动删除、覆盖或切换路径。若改用`/dev/shm`，机器结束前自行转存。

只初筛：加`--screen-only`；它不等于独立复测。复用现有同身份库：加`--skip-build`。

构建和运行分开时：

```bash
bash build_joint_plan.sh --soc=ascend950 --stage-exec-only
bash scripts/test_stage_exec_sweep.sh --skip-build --device 0 --requests 32 \
  --output /workspace/stage_exec_results --temp-dir /workspace/stage_exec_tmp
```

失败停止并保留现场；先看`build.log`、`screen.log`或`confirm.log`，再看子目录`launcher.log`及失败状态。退出码0、顶层`COMPLETE.json`和报告齐全才算完成，不上传公司输入、失败现场、trace或二进制。

## 本地检查记录

- 16条候选592次真实controller/SIMT/SIMD源码CPU语义调用，UBSan、输入/状态/输出/保护区检查通过。
- 240次全命中/全miss及0/1/63/64/65/2048 UID边界调用通过；包含重复resident。
- control/ABCD双warp模型26次、冻结参考短测13次通过。
- 14项框架/模型回归通过，包含排名、条件效应表、全组合符号、目录边界、无符号解包/掩码写回和2000组清理写集合等价性。
- 冻结221在相同CPU桩和宏配置下，替换回原controller前后的预处理文本完全相同。它证明未改其CPU可见计算正文，不代表设备.text相同。
- 三种容量的动态UB公式、缓冲字节存活、源码身份、Python/shell语法和`git diff --check`通过。

这些次数均为开发机检查，不是服务器实测；新增候选的CANN编译和NPU性能待验证。

API依据：[LoadAlign连续对齐搬入](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/LoadAlign_continuous.md)、[StoreAlign连续对齐搬出](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reg_data_store/StoreAlign_continuous.md)。使用9.1公开接口约定编写，实际9.2.1头文件兼容性仍由服务器编译确认。
