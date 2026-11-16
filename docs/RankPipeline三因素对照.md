# Rank-pipeline三因素：精确查询、分块前缀、批量输出

本轮基于服务器已验证的`051a439029d9f33620a10b29656661a4e9d6b0e4`，以`joint_stage_exec_abcd_512`为冻结参考。保留其CANN9.2.1的32位unpack调用；CPU桩和接口拒错测试同步修正，不能再用旧uint16桩证明真实API兼容。

## 版本清单

| 实现 | 本轮因素 | ID |
|---|---|---:|
| abcdf_pages | 原始独立ABCDF | — |
| joint_stage_exec_abcd_512 | 冻结性能参考，设备正文不改 | 243 |
| joint_rankpipe_base_512 | control，复制父版收尾、关闭三因素 | 244 |
| joint_rankpipe_a_512 | A | 245 |
| joint_rankpipe_b_512 | B | 246 |
| joint_rankpipe_ab_512 | AB | 247 |
| joint_rankpipe_c_512 | C | 248 |
| joint_rankpipe_ac_512 | AC | 249 |
| joint_rankpipe_bc_512 | BC | 250 |
| joint_rankpipe_abc_512 | ABC | 251 |

本轮字母与之前的exec/layout字母不同。A/B只作用于热分支，C覆盖支持形状的冷、热输出。保留相同的原1024线程回退，不把回退称独立ABCDF；没有改HBM缓存协议或减少输出要求。

生产矩阵为32请求、TopK2048、max_token262144、block_size64、seed2026，共同升序overlap40输入。MTP0/1/3/7分别为1/2/4/8行，每请求capacity及历史候选池分别4096/4096/8192/16384。前后实现读取相同逻辑输入、各自恢复可变状态。热75%是首行标签，不能当作后续每行的实际命中率。

## A：精确位图与字内排名替代二分

硬件归并得到升序、去重的unique后，按tokenID每32个一组，构建两个UB数组：

- `masks[word]`：该组哪些token在unique中。
- `firstUid[word]`：这个非空字中第一个token在unique中的UID。

查询resident token时：

```text
word = token / 32
bit = 1 << (token % 32)
如果 masks[word] & bit == 0：未选中
否则 UID = firstUid[word] + popcount(masks[word] & (bit - 1))
```

它是精确查找，没有Bloom误判；不负责求并集、不保存slot，不替换硬件MrgSort或query→UID映射。重复resident仍全部受保护，由原atomic_max选择LRU-last owner。

unique有序，一个字的token连续排列，由段首线程独占构造整个32位mask和首UID，每段最多32个token。不使用原子OR，也不增加全表前缀Scan。每次调用清理位图，空字不读首UID。

max_token<=262144时启用；更大值域用父版精确目录，仍可联合规划。原分析建议的16位首UID在实现中选为32位：省半字拆包和写入约束，32KiB mask+32KiB首UID+最大32KiB槽分类表恰好放入既有96KiB复用区。全部查询结束并同步后，再让给owner/partition。空间没有增加。

## B：分块计数→块间前缀→块内排名

父版SIMD压缩每64个位置依赖一次上一批累计值；16K时为256批。B让每个SIMT线程负责32个位置：

```text
构造块内hit掩码
→ 对每块命中数做一次跨warp Scan
→ 块间前缀 + 块内popcount得到稳定hit/evict排名
→ 同理对每块missing数量求前缀并分配miss槽
```

生产512线程覆盖512×32=16384个位置，一轮处理完整capacity。小容量多余线程仍以0参与集体操作；所有屏障和失败检查保留。miss分配不越过evicts，不满足容量条件时在持久写入前拒绝。

B使用已有SIMT Scan和popcount原语，不新增UB。它会重读flags，UB访问布局、寄存器与编译结果也可能改变，因此“Scan次数少”不等于已证明更快。

## C：UB准备完整输出→MTE3连续写回

保留miss有效前缀、resident、currentSlots和元数据计算，改变两类确定连续的输出：

1. 所有owner旧读者完成后，将完整LRU写入复用的owner区。
2. 所有unique旧读者完成后，将miss尾部的-1写入复用的unique区。
3. Kernel侧用DataCopyPad写LRU、miss_tokens尾部、miss_slots尾部。

显式V_MTE3等待producer写完；三份copy全部完成并通过MTE3_V等待后，才允许下一请求组复用UB。尾长0只搬LRU；尾目标非32B对齐也只写有效范围。没有清理保护区、缩短输出或搬运额外输入。

GM映射仍由原SIMT线程读取，不改成Scalar或MTE读者。所有copy及等待都在同一个Plan kernel内，Duration必须全部计入；没有新增辅助kernel，也不把准备时间移到CPU或计时外。

## 空间与正确性

| capacity | 动态UB，含Scan | 相对冻结父版 |
|---|---:|---|
| 4096 | 123680 B | 相同 |
| 8192 | 148256 B | 相同 |
| 16384 | 197408 B | 相同 |

`design/rankpipe/notes.md`有字节区间、producer/consumer/同步表和正则检查局限；liveness→allocate→physical_p产物及原始同步扫描一并保存。它们不证明硬件寄存器分配或缓存一致性。

门禁保留：只读输入、逐位置currentSlots、LRU、resident、miss及完整尾部、保护区、重复query、重复resident、stable prefix、容量边界及4K/8K/16K连续状态。新增密集32-token字/bit31边界、较大值域目录路径、63/64/65 UID尾块、多组复用同一AIV scratch检查。129请求仅是多组复用正确性用例，性能矩阵仍是32请求。

## 一次构建、初筛、独立复测

- 初筛10路×8工况，预热2、正式5样本。
- 按四个热工况相对冻结exec ABCD的P50比值几何平均，固定选择两个组合。
- 保留3条参考，5路同库独立复测，预热10、正式20样本。
- 复用同源码/库/设备/输入门禁，仍逐样本完整校验；CPU固定初态参考缓存沿用，连续状态参考不缓存。
- 初筛和复测样本不合并，不按不同工况拼接赢家，不用较慢control代替冻结参考计算收益。

报告包括`report.md`（复测）、`screen_report.md`（10路完整初筛）、`factor_effects.md`（12个条件对照）、`selection.json`、各采集目录的`capture_metadata.json/data_index.json`与原始CSV。严格核对目标kernel、单位、实际顺序和样本数；开始采集后1秒宿主等待保留。固定地址预热，只测Plan设备Duration，不测Transfer/SFA或接口端到端。

## 服务器命令

本机没有CANN/NPU。以下命令由服务器执行，CANN根目录和Python路径按实际安装位置调整；默认选空闲逻辑卡0。

```bash
set -euo pipefail
RUN=$(mktemp -d /workspace/single_op_test/rankpipe_run_XXXXXX)
git clone --single-branch -b feat/stage512-rank-pipeline \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
CANN_ROOT=/usr/local/Ascend/cann-9.2.1
source "$CANN_ROOT/set_env.sh"
export ASCEND_CANN_PACKAGE_HOME="$CANN_ROOT"
export PYTHON="${PYTHON:-python3}"
"$PYTHON" --version
export JOBS=3 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
bash scripts/test_rankpipe_sweep.sh --device 0 --requests 32 \
  --screen-repeat 5 --confirm-repeat 20 --regression-repeat 5 \
  --output "$RUN/results" --temp-dir "$RUN/tmp"
cat "$RUN/results/report.md"
cat "$RUN/results/factor_effects.md"
```

默认向持久化/workspace保存。输出目录必须尚不存在；初始化NPU前检查输出/临时目录的空间和inode，不自动清理或切换。PYTHON默认使用当前环境的python3；如需固定3.12或3.11，请在执行前将PYTHON设为服务器上匹配torch/torch_npu的真实解释器路径。若改到/dev/shm，结束前自行转存，原trace不要上传仓库。

只短测：加`--screen-only`；已有同身份库：加`--skip-build`。单独构建用`bash build_joint_plan.sh --soc=ascend950 --rankpipe-only`。失败立即停止并保留现场，先看build.log/screen.log/confirm.log及子目录launcher.log。退出码0、顶层COMPLETE.json及报告齐全才算整轮完成。

## 本地验证及边界

真实候选CPU语义模型+UBSan/保护区：8路296次基础与连续状态调用，240次全命中/全miss/UID尾块调用，26次双warp模型，以及20次多组与较大值域调用。冻结exec ABCD另外短测。CPU搬出模型延迟到MTE3_V等待时才读取源数据，用来检测提前复用；不是硬件事件或缓存模拟。

新旧框架回归、CANN9.2.1模板类型拒错、位图精确排名模型、缓冲边界、源码身份、Python/shell语法及diff空白检查一并执行。CPU检查不能证明CANN编译、真实地址空间、DCache/MTE同步或性能。新版本尚未进行硬件验证。

API依据：[SIMT与SIMD/MTE交接示例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/编译与运行/算子编译/约束说明.md)、[DataCopyPad](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/基础API/Memory矢量计算/数据搬运/DataCopy_GMToUB_NDDMA.md)。接口最终以服务器CANN9.2.1头文件编译为准。

VECCALC暂存直接DataCopyPad搬出的开源写法参考：[CANN SubMulConcat 950代码](https://gitcode.com/cann/ops-math/pull/4940/diffs)。仅作为接口/数据通路依据，不代替本候选的CANN编译与硬件验证。
