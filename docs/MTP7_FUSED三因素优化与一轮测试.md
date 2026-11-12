# MTP7 FUSED：A/B/C与七种组合

本轮以 `8317571` 的 `joint_mtp7_fused_count_512`（ID204）为冻结参考。保留已固化的 DCCI 构建修复：`-O3`，不添加 `-cce-aicore-dcci-before-kernel-end=false`。当前没有本地 CANN/NPU，新增版本的编译、正确性和性能都必须在服务器验证。

## 对比范围

| 实现 | ID | 本轮改动 |
|---|---:|---|
| `joint_mtp7_fused_count_512` | 204 | 冻结参考 |
| `joint_mtp7_fused_a_512` | 205 | A：复用完整中间块的硬件消费数量 |
| `joint_mtp7_fused_b_512` | 206 | B：完整64条记录的连续解交织加载 |
| `joint_mtp7_fused_ab_512` | 207 | A+B |
| `joint_mtp7_fused_c_512` | 208 | C：精简累计计数和批量游标更新 |
| `joint_mtp7_fused_ac_512` | 209 | A+C |
| `joint_mtp7_fused_bc_512` | 210 | B+C |
| `joint_mtp7_fused_abc_512` | 211 | A+B+C |

八路都保留512线程、32位映射、每源最多612条记录、生产动态UB181024字节。没有MID、安全前缀、16位映射或线程数矩阵。新增算法只在八行组且当前有效源数大于四的两级归并阶段执行。MTP0/1/3及其他原有分支仍执行原算法，性能上仍需观察重新编译造成的变化。

不提前指定赢家。报告为每个工况独立列出所有八路，不把不同工况的最快版本拼成一个实现。

## A：哪些硬件计数可以复用

第一层分别将四条源行归并成A、B两个有序前缀，并已返回每个源实际贡献的记录数。最终两路归并又返回A、B各消费多少。

- 最终消费量等于该中间前缀的完整长度：其四行数量可直接继承第一层结果。
- 只消费部分前缀：不能继承整块计数，继续在SIMD中统计该组。
- 两个前缀都完整消费：不再执行逐记录行计数，去重、映射仍照常执行。

只在当前八条源全部有效时走该优化；这时前四源确实对应row0..3，后四源对应row4..7。5/6/7条有效源、源耗尽后不足八条的轮次，保留原八行统计，避免把压缩后的source下标误当row下标。

Scalar控制器使用原128元素context中的空闲位置，不申请额外UB：

| context位置 | 用途 |
|---|---|
| 108 | 已完整接收哪组：-1=通用；0=A；1=B；2=两组 |
| 109、110 | 最终两路归并分别消费A、B的记录数 |
| 112..119 | 第一层八个源的消费数量，仅在八源有效时使用 |

第一层计数在输出前缀长度和耗尽结果确认后记录。最终完整性检查通过后，原有S→V交接将metadata发布给SIMD。未改变数据补料、硬件耗尽暂停、持久状态容量检查、线程1模式发布规避或回退协议。

SIMD采用编译期不同的计数布局，Scalar侧选择对应VF；不在SIMD逐记录循环内增加运行时分支。对于已完整消费A的情况，另一组的累计区间计数会减去A的已知总数，恢复后四行的真实计数，不能直接拿跨八行累计量当后四行数量。

## B：连续加载代替离散读取

归并记录是连续的32位字段对：`score,queryID,score,queryID,...`。完整64记录块采用 `LoadAlign<uint32_t, DIST_DINTLV_B32>`，将偶数、奇数位置分别装入score与queryID寄存器。score只做按位解释，queryID不转换为float。

不足64记录的尾块继续执行原带掩码Gather。没有通过填充有效长度或越界预读来加速；也没有减少原始字节数。这项改动主要观察加载与地址生成指令的收益。生产输出区满足32字节对齐，CPU模型会检查对齐。

参考：[CANN9.1 LoadAlign](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/LoadAlign_continuous.md)。

## C：减少统计收尾

- 通用八行路径的最后累计量就是有效记录数n，省掉第八个计数向量及其归约。
- 与A组合时，未完整消费那组的总量已由最终硬件结果给出，因此该组只需三个累计区间计数。
- 将八行增量放到向量的前八个lane，一次读取旧游标、相加和写回。
- 后续`more`判断直接使用已更新的寄存器，不再把刚写入UB的游标重新读回来。

C之外的版本保留原逐行游标更新及其回读屏障。C只是消除了该读后依赖，不删除其他阶段所需的同步。

## 源码阅读

- `csrc/joint_variants/mtp7_fused_refine.h`：新增模板化SIMD实现，包含A/B/C组合。
- `csrc/joint_variants/mtp8_plan_impl.h`：A的硬件计数记录、完整性判断与VF选择；新增内容均由宏隔离。
- `csrc/joint_variants/mtp7_merge_unique.h`：冻结FUSED，不改原函数。
- `checks/check_mtp7_fused_tune.py`：真实C++ helper和控制流的CPU/UBSan检查。
- `run_mtp8.py`中的`fused_tune`：专项门禁、采集、报告。

冻结204在宏展开后的控制器+SIMD内容有独立SHA256检查。source_info与manifest继续验证源码、编译选项和库身份，旧库不能冒充本轮构建。

## 服务器一次构建、一次门禁、一次正式采集

下面选择逻辑NPU4；请使用空闲设备。默认28请求、TopK2048、capacity4096、max_token262144、seed2026、overlap40共同升序输入。满缓存75%为生成器原有场景标签，不代表后续每条draft均有75%命中；每行实际命中/descriptor见metadata。没有改动生成器。

```bash
set -eo pipefail
BRANCH=feat/mtp7-fused-refine
DEVICE=4
RUN=$(mktemp -d /workspace/single_op_test/mtp7_fused_tune_XXXXXX)
git clone --single-branch -b "$BRANCH" \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"

# 按服务器实际安装版本调整此路径。
CANN_HOME=/usr/local/Ascend/cann-9.1.0
source "$CANN_HOME/set_env.sh"
export ASCEND_CANN_PACKAGE_HOME="$CANN_HOME"
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

bash scripts/test_mtp7_fused_tune.sh \
  --device "$DEVICE" --requests 28 \
  --mtp 0 1 3 7 --scenarios cold_reset full_cache_75pct \
  --warmup 2 --repeat 5 --regression-repeat 5 \
  --output "$RUN/results" --temp-dir "$RUN/tmp"

cat "$RUN/results/report.md"
echo "持久结果目录：$RUN/results"
```

只构建本轮八条联合变体，公共基础入口仍随共享库链接。正式目标样本数为 `8实现×8工况×5次=320`，无初筛或额外确认轮叠加。门禁包含少量可见性、重复token、容量4096/4097、5/6有效源、请求变化、真实连续状态和上下文检查。这些不作为性能样本。

如只需独立构建：`bash build_joint_plan.sh --soc=ascend950 --fused-tune-only`；已构建后给测试命令添加`--skip-build`。通过后可在新输出目录把`--warmup 10 --repeat 20`，并用`--correctness-report "$RUN/results/gate.json"`复用同源码/库/设备及精确输入的门禁。每个正式样本的严格检查仍保留，不混合两轮样本统计。

空间与inode预检在NPU初始化前执行；目录不足时直接失败，不自动切换位置。默认结果持久保存在/workspace；若手动改到/dev/shm，需转存原始CSV、报告、metadata和日志。

## 统计与失败行为

- 精确匹配八个目标kernel；按CSV的设备Duration统计P50/P95、样本数与执行顺序。包括Plan全部归并，不含恢复、CPU校验、Transfer/SFA或接口端到端延时。
- 各实现轮换先后顺序，采集开始后1秒宿主等待的记录保留。
- 同一工况恢复同一初态；golden固定初态认证后缓存；每个样本仍检查只读输入、逐位置currentSlots、缓存、LRU、miss、保护区与回退。
- `report.md`列出全部八路相对同场FUSED的比值、实际descriptor、共享/回退数及Block Num；自身MTP0比值及原始CSV索引保存在metadata。
- 失败立即停止并保留现场，不能生成成功性能结论。退出码0、COMPLETE.json和报告齐全才算成功。

## 本地验证边界

使用真实候选源码、CPU SIMD/SIMT语义替身及UBSan，另做构建子集dispatch编译/链接、源码指纹、Python/Bash语法和框架回归。CPU模型不能验证DAV3510地址空间、硬件流水、寄存器分配或DCCI行为。

`design/fused_tune/`保留一轮抽象寄存器数据流的liveness、分配和WAR复核输入/输出，显式包含八个跨批次计数累加器。该模型用于审阅逻辑生命周期；其节点不是实际Bisheng寄存器，未展开全部掩码、广播、地址临时量及循环回边，不能用其P值声称无spill或给出实际寄存器占用。外层record/map/unique/context属于固定UB上下文，生产UB预算仍由host/layout检查。

段编排沿用已有FUSED：S1～S6去重/映射和行计数共用同一遍记录；S6～S10用VF-safe归约与紧随消费完成游标更新（C1）；硬件MrgSort保留在Scalar/kernel控制层，不搬入SIMD VF（C8）。新增方案没有跨卡通信或跨核同步。

实际本地结果：七条候选各139项完整控制器对照、208项局部/跨warp检查，以及各45项既有消费统计模型均通过；74项框架单元回归通过；18个构建子集的dispatch在G++下编译/链接通过；冻结FUSED预处理指纹、源码身份、Python/Bash语法通过。局部检查覆盖0/1/31/32/33/63/64/65/127/128/129/511/512/513/4896等有效长度（按实际query总数截断）、不同TopK、跨64条边界重复、三种已知半组状态与通用统计状态。

同步与数据流正则扫描了新增helper及相关控制器/冷收尾/逐行回退/入口等七个文件，均未报告候选。其结果不覆盖完整C++模板展开、Reg访问和真实硬件时序，不能替代上机同步检查。尚未执行Bisheng编译、NPU正确性、性能或寄存器资源验证。
