# 动态容量热路径 A/B/C/D：15路对照

基于端侧提交408174f；本轮字母与历史优化因素无关。默认32请求、TopK2048、max_token262144，MTP0/1/3/7分别使用capacity4096/4096/8192/16384。每个工况三方/多方使用同一份升序query、初始缓存与LRU；capacity及历史候选池同时随MTP变化，因此跨MTP不能称纯开关开销。

## 实现清单

冻结参考：abcdf_pages、joint_mtp7_stage16_512（214）、joint_mtp7_ub16_512（215）。原计算文件保持原样。

| 组合 | stage候选 | 全UB候选 | 新因素 |
|---|---|---|---|
| A | joint_layout_stage_a_512 | joint_layout_ub_a_512 | 容量布局 |
| AB | joint_layout_stage_ab_512 | joint_layout_ub_ab_512 | A＋新热压缩 |
| AC | joint_layout_stage_ac_512 | joint_layout_ub_ac_512 | A＋物理槽顺序查找 |
| ABC | joint_layout_stage_abc_512 | joint_layout_ub_abc_512 | 三者组合 |
| ACD | joint_layout_stage_acd_512 | joint_layout_ub_acd_512 | 加1024区间目录 |
| ABCD | joint_layout_stage_abcd_512 | joint_layout_ub_abcd_512 | 全组合 |

候选编号216..221为stage六路，222..227为全UB六路。主路径512线程；沿用1024线程private兼容路径。D只在C路径中启用，无独立D。

## A：容量布局

映射区域由固定32KiB改为capacity*2字节：4K容量8KiB、8K容量16KiB、16K容量32KiB。保留统一入口，host不读取设备请求分组信息；以capacity作为本候选支持的query位置数上界，因此MTP0虽然只有2048位置，也预留4096位置映射空间。该上界在Phase0写映射前检查；超出走原private，不将过小映射交给后续阶段。参考/golden显式记录这个支持条件。

归并区保持96KiB。八行实际包长608记录/路：与父版相同，原注释的612是上限，`(24576/40)&~7`实际得到608。本轮不叠加扩包，避免引入额外变量。

| capacity | stage全部候选 | 全UB A/AC/ACD | 全UB含B |
|---|---:|---:|---:|
| 4096 | 123680B（120.78KiB） | 123680B | 123680B |
| 8192 | 148256B（144.78KiB） | 148256B | 148256B |
| 16384 | 197408B（192.78KiB） | 203552B（198.78KiB） | 205600B（200.78KiB） |

包括控制区512B与Scan288B；动态scratch上限仍204KiB，加Scan后不超过209184B。实际可用资源、DCache分配、寄存器spill以CANN编译及设备报告为准。Python、host C++公式有独立编译对照测试。`design/layout_hot/byte_liveness.json`逐字节检查所有候选、容量和阶段的区间与重叠；同目录liveness/allocate/physical_p产物是保守区域依赖模型，节点数不直接换算为实际UB字节。

## B：两种新热压缩

stage：SIMT先完成resident分类与LRU-last owner仲裁，再由SIMD读取打包32bit flags，解出16bit标记，通过mask/Unsqueeze计算稳定hit/evict排名，Scatter16输出partition；同一VF继续对owner生成miss排名并分配槽位。miss rank存owner空闲位，省去完整miss-rank flags写回。失效清理、miss输出、currentSlots、LRU写回仍在容量检查之后执行。SIMD载入、掩码和尾部使用完整范围检查；不得以普通Scatter覆盖重复resident的owner。

注意uint16 Scatter索引范围：partition从owner近基址寻址，最大有效索引小于49152；不从整个UB基址寻址，以免16K容量时超过65535。未激活lane使用合法、唯一且与实际partition目标不相交的索引。

全UB：使用已有hit/valid位图字计算`valid & ~hit`，popcount得到每32槽的hit/evict数；对这些段数量做前缀，再以段起点＋段内popcount生成精确排名。16K最多512段，生产512线程一次处理这些段数量，替代原32轮512槽Scan。之后复用已失效的位图保存miss投票，按同样方式分配miss。增加capacity/32个uint32前缀暂存（16K时2KiB），已包含在预算中。CPU使用较少线程模拟时有多个段批次，不能把CPU循环数当作设备成本。

B不会在cold reset执行；冷工况的差异可能涉及编译资源变化。SIMD寄存器临时量不计入UB，但须检查真实编译spill。

## C：按物理槽顺序查询

原流程按随机LRU先读slot，再离散读slot_to_token。C先连续读物理slot_to_token，计算slot→UID，暂存到此时尚未使用的uint16 partition区；再按LRU顺序从UB表读取UID并执行原atomic_max仲裁。全部命中槽受保护，重复resident仍按LRU-last选owner。slot表使用14bit UID、bit14命中、bit15失效，容量最多16384。

这项增加一次UB表写读和必要交接，争取降低离散GM读取成本，不保证更快。分类结束后才覆盖临时slot表来生成最终partition。

## D：精确查询目录

物理槽查询前，在尚未初始化的owner区域建立1025个边界，将token值域均分1024段。每个token只在所属段的unique区间做精确二分；没有概率误判。所有目录读结束并同步后，才把owner初始化为-1并开始原子仲裁。目录不跨调用保存，不需要CPU或上游预计算，不新增大块UB。

## 代码阅读

- `joint_variants.py`：15路清单、因素与父版身份。
- `layout_hot_tiling.h`、`joint_plan_cases.py`：一致的host/Python布局。
- `layout_hot_impl.h`：归并控制与Phase6/SIMD/Phase9编排；从冻结controller派生，避免改动冻结计算文件。
- `layout_hot_plan.h`：C/D分类、全UB的B、状态提交。
- `layout_hot_simd.h`：stage的B。
- `layout_hot_fused.h`：归并后去重/映射/消费计数，使用容量派生映射偏移。
- `checks/check_layout_hot_kernels.py`：真实源码CPU桩；`checks/check_layout_hot_suite.py`：布局、host公式、符号及选择规则检查。
- `scripts/test_layout_hot_sweep.sh`：服务器入口；`run_layout_hot_sweep.py`：一次构建、短测和独立复测。

## 正确性与测量

补回8行/16K重复query专项，报告每行显式打印capacity；动态容量套件不再静默忽略--query-pool-tokens。保留只读输入、输出、LRU、resident、miss、保护区、逐位置currentSlots、重复resident仲裁及连续模式切换检查。4K/8K/16K分别进行连续调用，使用前一轮实际设备状态。失败停止并保留现场。

初筛默认15路×8工况×5正式样本＝600条目标kernel记录。每路预热2次。以四个热工况相对各自冻结父版的P50比值几何平均，为stage和全UB分别固定选择一个候选；保留3条参考，进行5路×8工况、预热10次、独立20样本复测。初筛与复测不混合，不按工况拼接赢家；冷工况完整展示，不参与热工况筛选指标。入围只表示该布局新候选中排名靠前，不代表超过冻结父版。

构建一次、复用同一库。复测复用同源码/库/设备/精确输入门禁，不再额外重复audit；每个正式样本完整校验保留。profiler开始后1秒宿主等待、目标kernel精确匹配、单位/顺序/样本数门禁沿用原框架。恢复和校验不计入设备目标Duration；归并及暂存属于目标kernel，全部计入。

原ABCDF112KiB scratch门槛保持不变，16K容量会使用原GM scratch路径；报告以每种候选自身冻结父版为主要优化参照。

## 服务器执行

开发机无CANN/NPU，本地CPU通过不代表已通过Bisheng编译、真实同步、缓存一致性或性能验证。

```bash
set -euo pipefail
RUN=$(mktemp -d /workspace/single_op_test/layout_hot_XXXXXX)
git clone --single-branch -b feat/mtp-layout-hot-sweep \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
bash scripts/test_layout_hot_sweep.sh --device 0 --requests 32 \
  --screen-repeat 5 --confirm-repeat 20 --regression-repeat 5 \
  --output "$RUN/results" --temp-dir "$RUN/tmp"
cat "$RUN/results/report.md"
```

逻辑卡0为默认示例，请选空闲卡。输出目录必须尚不存在。`--screen-only`仅做15路短测；`--skip-build`仅在现有同身份库下使用。若输出/临时目录改到/dev/shm，结束前将报告、CSV、metadata、门禁与日志转存；不自动清理或切换目录。

构建失败：results/build.log；门禁或采集失败：results/screen.log及screen/launcher.log；复测失败：results/confirm.log及confirm/launcher.log。只有退出码0、顶层COMPLETE.json及report.md齐全才算整轮完成。全部数值现场和profiling产物留在运行服务器，不提交仓库。

## API依据

- ShiftRights（CANN9.1、950、标量类型int16_t）：https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/logical_compute/ShiftRights.md
- Unsqueeze：https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/data_compression/Unsqueeze.md
- B16 Scatter沿用父版已使用的B32 mask/低半字约定，CPU桩检查全部索引唯一；仍须目标SDK编译验证。

L2/cache策略此次保持原样。stage暂存由同一线程写回/读回相同32位字，不引入Scalar/DMA消费者。不能假定映射常驻L2，也不添加未经验证的cache旁路或删除现有同步。

## 本地已执行检查

12条新候选共444次真实controller/VF源码CPU桩调用；stage_ABCD和ub_ABCD的两warp模型26次；两条冻结布局参考42次。使用UBSan和输入/状态/输出/保护区检查。另有14项框架回归、36种布局字节存活检查、host公式独立C++编译对照，以及源码身份、Python语法、shell语法和diff空白检查。没有CANN/NPU硬件验证，以上次数不是设备测试次数。
