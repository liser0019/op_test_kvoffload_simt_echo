# MTP7：八行联合规划与精简测试

本分支新增 `joint_hot_bd8_512`，以已验证的 `joint_hot_bd_512` 为父版本，标准路径512线程，兼容MTP0/1/3，并扩展到MTP7。**本地没有CANN/NPU，只完成CPU语义模型、UBSan和框架检查；不能据此宣称上机通过或一定更快。**

## 三条对照

| 实现 | 用途 | MTP7设备算法 |
|---|---|---|
| `abcdf_pages` | 冻结带页表ABCDF | 256条query各自独立规划 |
| `joint_hot_bd_512` | 本轮51.462μs的冻结新BD参考，源码不改 | 8行超出旧支持范围，逐行回退；报告明确标记 |
| `joint_hot_bd8_512` | 新候选，public ID193 | 单请求8行共同规划；不把两组四行各自提交 |

MTP1/3/7分别是同一请求2/4/8条query，固定32请求时共64/128/256行。MTP0仍保留。新候选的联合组只支持1/2/4/8行，其他组宽、非法请求、数值编码不支持和并集/可淘汰槽不足，保留原1024线程逐行语义回退。回退仍使用已经申请的UB，不因112KiB门槛改用GM。

## 八路流式归并

核心在 `csrc/joint_variants/mtp8_plan_impl.h`。该文件来自新BD的完整计算链，旧 `hot_plan_impl.h` 字节不改。

1. Phase0逐行计算因果有效区间；保留首尾快速判断、边界二分和单行严格递增直接映射。单行重复仍按原接口回退。
2. 每次从各有效行装入不超过512条记录，记录是 `(-token, 原query位置)`，共8字节。四行以下继续使用原四路控制；五到八个非空队列使用树式归并。
3. 前四个有效队列、后面的有效队列分别做一次硬件MrgSort。某路未全部装入时，使用耗尽暂停，只得到该半组全局安全的有序前缀。
4. 两个中间前缀再做两路MrgSort；中间任一未完整，最终也暂停于某条中间队列耗尽。完整时才允许全部归并。
5. **只有最终实际输出的前缀才推进原始八行游标。** `EightProgressVf`中8个线程分别统计最终payload属于哪一行；中间归并的消费数量绝不能直接推进原行，否则会跳过尚未进入最终结果的token。
6. SIMD相邻去重和query→UID映射跨包携带last token与unique累计数，保留每个重复query的位置。全组统一执行resident查询、保护命中槽、SIMD分类/分配和持久状态提交。
7. unique或可淘汰槽不足时，在持久状态提交之前回退。

只复制一个非空队列时使用SIMT搬入中间缓冲，不向MrgSort传只有一条有效队列。输入、两段中间结果、最终结果互不重叠，起点32B对齐。SIMT复制和游标统计保留末尾block barrier，沿用父版本Scalar/Vector事件交接、thread1模式发布规避；未声称已确认此前错写的底层原因。

首版按正确性优先，游标回计和重新装入未消费记录仍有开销，不承诺MTP7与MTP3延迟相同。

## UB与DCache

新候选单独使用192KiB scratch上限，不改冻结参考的112KiB配置。生产2048/4096形状：

| 区域 | 字节 |
|---|---:|
| 控制区128个int32 | 512 |
| 8T query→UID映射 | 65536 |
| C unique | 16384 |
| work：归并三份缓冲，后续复用owner/flags/partition | 98304 |
| scratch合计 | 180736 |
| Scan | 288 |
| 动态UB实际申请 | **181024（约176.8KiB）** |

GM workspace只保留1个int32占位。Python与C++使用同一公式，Prepared再次核对；新候选布局超过192KiB时明确拒绝形状，不静默退到GM。

**多申请UB有代价：可用DCache会减少。** 按950文档的256KiB、默认8KiB系统预留估算，此形状约剩71.2KiB DCache；编译产物与真实资源仍以上机为准。旧BD继续使用原申请。因此MTP0/1/3必须同场复测，不能预先保证继承旧BD耗时。

官方约束：
- [MrgSort：2–4队列、耗尽暂停、每路0–4095记录](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/sort_operations/MrgSort-93.md)。中间队列最多2048条，未超接口范围；使用MrgSort，不使用950不支持的旧MrgSort4接口。
- [CANN9.1编译约束与SIMT内存划分](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/编程指南/编译与运行/算子编译/约束说明.md)。必须扣除系统预留和至少32KiB DCache。

## 为什么旧流程慢，以及本轮怎么缩短

旧hot流程默认38路：完整门禁、5样本初筛、入围集合50样本复测。每次调用还有恢复、D2H和CPU模型比较。几微秒的kernel不是整轮墙钟耗时；此前没有细分日志，不能声称两小时具体分配比例。

本轮入口 `scripts/test_mtp8.sh`：
- `--mtp8-only`只生成旧BD和新BD8的两个实验包装，不编译32种热因素组合；保留原独立ABCDF构建所需对象和注册。
- 固定三路、cold/full75×MTP0/1/3/7。默认warmup2、正式repeat5，一共120个目标kernel样本，**不再初筛后复测两轮**。`--formal`才使用warmup10/repeat50。
- 正确性专项只跑一次，回归默认5次；包含已知输入错写工况、8行重复/边界/空队列/溢出、连续设备状态、Prepared上下文拒错和本轮每个生产工况。
- 性能阶段复用同库、同设备、同输入指纹的已通过门禁，不额外再跑5次audit。指定`--correctness-report`可在第二次正式采集时复用上一轮匹配门禁；旧门禁、不同库或不同输入会拒绝。
- `fast_fixed_check`为本入口显式启用：首个固定初态结果经原CPU模型认证后，保存确定性期望；后续仍逐样本比较只读输入、逐位置输出、持久状态和全部原有保护区。private描述符只校验原约定的有效前缀，joint保留完整尾部检查，不放宽接口约定。
- 这个缓存属于单个bundle，绑定所有输入/属性/初态，销毁即释放；连续状态调用不使用缓存，仍逐步计算模型。**没有从CPU期望结果重写设备持久状态。**
- 逐样本按dtype合并D2H，减少零碎拷贝；额外拼接/拷贝任务不会混入目标kernel Duration。
- `RUN_STATUS.json`记录build/gate/profile墙钟；metadata分工况记录准备、audit、预热、恢复、校验、首次参考认证/缓存命中、profiler关闭导出解析和CSV核验时间。这些时间有包含关系，不能全部相加，也不是设备耗时。
- 仍记录采集ACTIVE开始后1秒宿主等待，严格核对精确kernel名、μs单位、样本数和轮换顺序，逐样本失败立即停止并保存现场。

短流程仍需要构建和NPU门禁，不能保证一定几分钟完成。报告墙钟能说明下一次真正慢在哪里。不要通过关闭正确性门禁提速。

## 上机命令（逻辑NPU0）

保留旧工作目录与数据，新建目录克隆；下列是正常shell语法，变量用单个`$`，不要把`$$FILE`复制成PID拼接。

```bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
RUN=$(mktemp -d /workspace/single_op_test/mtp8_bd_XXXXXX)
git clone --single-branch -b feat/hot-bd-mtp7 \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
bash scripts/test_mtp8.sh --device 0 \
  --output "$RUN/quick" --temp-dir "$RUN/tmp"
```

先默认短测，退出码0、`COMPLETE.json`、报告均齐全后再决定正式采集。使用同一源码和库，复用短测门禁：

```bash
bash scripts/test_mtp8.sh --device 0 --skip-build --formal \
  --correctness-report "$RUN/quick/gate.json" \
  --output "$RUN/formal" --temp-dir "$RUN/tmp"
```

只构建、查看计划：
```bash
bash build_joint_plan.sh --soc=ascend950 --mtp8-only
"$PYTHON" -B run_mtp8.py --describe-only --output /workspace/unused_plan
```

默认结果持久保存在/workspace。显式改为/dev/shm时仍提示临时存储，需要自行转存；不自动删除、不覆盖旧目录。服务器原始输入、现场、二进制和profiling数据只留运行机器，不上传GitHub。

如果编译、UB资源或设备结果报错，保留build.log/launcher.log与失败现场；本地CPU通过不是上机通过的替代。报告要分别看MTP7共享/回退数量和实际descriptor数，不能把旧BD的逐行回退性能当成8行联合性能。

## 本地检查

```bash
python3 -B checks/check_mtp8.py
python3 -B -m unittest checks.check_mtp8_suite checks.check_hot_suite checks.check_joint_models
python3 -B checks/check_merge_catalog.py
```

CPU模型只验证源码算法与缓冲索引，不验证CANN地址空间、真实MrgSort实现、DCache一致性、真实512线程同步或性能。冻结旧BD计算文件及ABCDF基线保持不变。

本轮开发机实际结果：144次真实控制流程CPU语义检查（包含与冻结BD逐字段对照、随机重复和连续状态），12种生成目录CPU编译/链接分派检查通过。同步脚本在新核心及冷收尾头上未报候选，但正则数据流分析没有提取出MrgSort/SIMT访问图，不能据此证明设备同步正确；SDK内部、地址空间和真实同步仍待NPU。

| 新链路 | 生产者→消费者 | 源码中的交接 |
|---|---|---|
| 输入打包 | SIMT写packet→MrgSort读 | Phase1末尾block barrier及V→S等待 |
| 半组输出 | MrgSort写mid→最终MrgSort读 | 每次半组合并后V→S等待，输入/中间/输出地址不重叠 |
| 单队列半组 | EightCopyVf写mid→最终MrgSort读 | helper末尾block barrier及V→S等待 |
| 最终前缀 | MrgSort写out→EightProgressVf读 | V→S等待，Scalar控制发布后S→V交接 |
| 游标与映射 | EightProgressVf写游标→SIMD去重/下一轮 | block barrier、V→S等待，沿用原阶段交接 |
