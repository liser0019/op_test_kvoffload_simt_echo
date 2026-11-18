# 保序Indexer与rankpipe Plan融合实验

本分支提供可构建源码与上机脚本。开发机没有CANN/NPU：CPU检查不等于
Bisheng编译通过、设备同步正确或取得性能收益。所有性能数字由服务器生成。

`0a42bee`之后的A/B初始化同步、CSV计时精度修复，以及只复测失败工况的命令，
见[QLI同步与计时修复](QLI同步与计时修复.md)。A/B的原始TopK差异根因仍待设备对照确认。

## 对照版本

| 名称 | Indexer | Plan | 中间数据 |
|---|---|---|---|
| baseline（可选） | 上游原始 `quant_lightning_indexer`，原始 tiling | — | 把 TopK 写入预分配输出 |
| ordered4（可选） | 与 split 同源的保序 QLI，固定 S1Base=4 | — | 同样的保序 TopK，编译时关闭 Plan hook，Plan UB 为 0 |
| split | 同一份保序QLI，关闭Plan hook | 独立、冻结 `joint_rankpipe_ac_512` | TopK直接写入独立Plan绑定的缓冲；中间没有TensorMove或显式同步 |
| a | 保序QLI＋Plan，同一个mixed kernel | 代表位置16位＋精确位图 | unique保存原始query位置；只复用Vector临时区，Cube结果区独占 |
| b | 同a | 代表位置16位＋目录/二分 | 与a比较resident查询方式，保留归并、LRU、owner和持久协议 |
| c | 保序QLI＋Plan，同一个mixed kernel | 流式unique32＋精确位图 | 每批至多2048条新unique经8KiB UB暂存后写GM，不先构造整张UB unique |
| peer（可选） | 同事原融合实现 | 原Plan | 保留原始源码及其自身tiling，仅作已有实现参考 |

默认只跑split/a/b/c。三条候选互斥，不把它们的收益相加。A/B的unique
位置压缩可能增加间接GM读取，C增加DMA/GM访问，均不能预先保证更快。
单行严格递增时，UID直接对应有效区间位置，三条新候选保留直接规划路径。

来源固定为GitCode `ops-transformer@9ec2110969a9e8502bf6d579c027ac4ea9572016`。
新Plan计算来自本仓 `44ee66c` 的rankpipe AC（mask249），保留硬件MrgSort、
SIMD去重/映射、热分类与miss压缩、LRU-last owner、线程1模式发布规避及逐行回退。
导出器移除了未调用的历史联合Hash/软件Merge代码，冻结算法文件本身未修改。

## 输入及公平性

- 默认BS32，TopK2048，MTP0/1/3/7分别为32/64/128/256条query行。
- capacity为4K/4K/8K/16K，默认上下文262144，可指定102400。
- 使用真实FP8 QLI kernel生成TopK；没有CPU预计算命中信息或免费预排序。
- split/a/b/c都使用同一份公开保序QLI算法，固定S1Base=2；其输出逐位置比较。
- QLI的K是实际随机物理页排列的PA_BSND布局，FP8 K和FP32 scale共享页存储。
- 冷状态与满缓存75%标签场景。75%只约束首行；其他draft行由QLI真实选择，
  不强造跨行40%重合或每行75%命中，实际重合/命中与descriptor写入元数据。
- 首次split-only QLI调用取得canonical TopK，在计时外构造所有实现相同的初始
  resident/LRU。每个样本恢复同一初态及输出污染值，逐样本校验，失败立即停止。
- 指纹包含QLI原始输入及其layout/stride，以及Plan的逻辑输入。readonly QLI
  输入保留设备快照，逐样本按字节比较；初态恢复和检查都在目标调用之外。
- 当前独立冻结Plan的性能对照要求完整有效TopK。真实Indexer还支持短序列
  输出“有效升序前缀＋尾部-1”；新融合候选增加单独的边界门禁，不把它混进性能矩阵。

这里的“有效升序前缀”只针对固定的同事保序实现：
`vf/quant_lightning_indexer_ordered_topk.h::SelectOrderedIndicesVFImpl` 按位置
递增扫描并Squeeze选中索引；`quant_lightning_indexer_topk.h::SelectOrderedTopK`
维持分块历史索引与分值的对应。Vector service中的 `qliNeg=-1` 及
`qliValidS2Len < topkCount_` 分支负责短序列补尾。不是说完整数组全局升序：
例如 `[0,1,2,-1]` 只有前三项递增。100K/256K正常行通常选满2048项，没有补尾。
采集入口严格检查有效项数量、递增、因果范围及尾部全为-1；不排序或修复异常输出。

## UB与同步边界

实际Host资源函数，S1Base2、32 indexer heads、BF16 weights/FP32 scale：

| capacity | a/b Plan字节 | a/b融合总字节 | c Plan字节 | c融合总字节 |
|---|---:|---:|---:|---:|
| 4096 | 115488 | 156704 | 115488 | 176672 |
| 8192 | 131872 | 173088 | 123680 | 184864 |
| 16384 | 164640 | 205856（201.03KiB） | 140064 | 201248（196.53KiB） |

这些是静态资源计算，不是NPU可运行性证明。默认216KiB预算保留，不释放系统
预留或最小DCache。Scan72字、ctx128字和96KiB工作区计入申请。

A/B保留整个32KiB Cube→UB结果区的原地址及ping-pong布局。仅将Vector的
weights/scales/output临时区与TopK/Plan arena按阶段复用；不改变Cube同步协议。
TopK只在 `isLastS2InnerLoop` 后调用，下一块从first S2重新加载参数；K scale
padding重新清零。借用前等待Vector-owned DMA；不是用PIPE_V等待替代Cube确认。
发布计数用的24字节放到arena末尾，在RunGroup开始前读取完毕。

C每批DMA完成后才复用8KiB staging；后续SIMT读取已经发布的GM unique。
热路径原有query→UID暂存GM仍保留，不能称为全UB算法。所有额外搬运均在
融合kernel内，计入设备Duration。输出尾部仍完整写-1，未削弱消费者约定。

## 文件阅读顺序

1. `scripts/prepare_qli_fusion.py`：固定上游、生成四个可共存的注册算子、修改Host资源计算。
2. `csrc/qli_fusion/resources.h`：Host/Kernel/绑定共用的空间公式。
3. `csrc/qli_fusion/storage.h`：UB/GM TopK访问器、代表位置/GM unique访问器。
4. `csrc/qli_fusion/packet.inc`：C的归并中分批落GM；A/B复用原消费计数。
5. `csrc/qli_fusion/plan.h`：可直接阅读的设备实现，结尾RunGroup是融合hook入口。
6. `profile_qli_fusion.py` 与 `fusion/metrics.py`：状态恢复、正确性及计时/配对。

`plan.h`由 `scripts/export_fusion_plan.py` 冻结展开并作定点适配。
修改后运行导出器，再用 `--check` 核验；不得绕过源码/库身份检测。

## 服务器构建

在新的工作目录克隆本分支，保留此前已验证工程。以下路径按实际CANN/Python调整：

```bash
set -euo pipefail
RUN=$(mktemp -d /workspace/single_op_test/qli_fusion_XXXXXX)
git clone --single-branch -b feat/qli-rankpipe-fusion \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"

source /usr/local/Ascend/cann-9.2.1/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.2.1
export PYTHON=python3 JOBS=3
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

# 独立冻结Plan库，复测时无需重新构建。
bash build_joint_plan.sh --rankpipe-only 2>&1 | tee "$RUN/plan_build.log"

# 原生QLI采用上游注册工程：构建专用vendor包及Torch扩展。
# 此步骤需要访问GitCode；不会修改或覆盖其他人的源码目录。
export QLI_FUSION_BUILD_DIR="$RUN/fusion_build"
bash scripts/build_qli_fusion.sh 2>&1 | tee "$RUN/fusion_build.log"
source "$QLI_FUSION_BUILD_DIR/env.sh"
```

上游构建可能还需要其原有依赖；本机未执行该CANN构建。脚本不自动清空已有
构建目录，重建请换新 `QLI_FUSION_BUILD_DIR`。源码身份写入Torch二进制；运行时
检查该身份、独立Plan库身份及本次专用安装目录内CANN二进制/配置的哈希。

## 分阶段测试：失败即停，复用同一构建

```bash
# 1. 小批量完整链路；有真实QLI计算，capacity仍按生产规则。
bash scripts/test_qli_fusion.sh --device 0 --requests 1 --contexts 32768 \
  --mtp 0 7 --warmup 0 --repeat 1 --regression-repeat 1 \
  --output "$RUN/smoke" --temp-dir "$RUN/tmp"

# 2. 短序列尾部-1门禁。只验证新融合候选，无性能结论；不调用冻结独立Plan。
bash scripts/test_qli_fusion.sh --device 0 --requests 1 --contexts 1024 \
  --mtp 0 1 3 7 --scenarios cold_reset --gate-only \
  --warmup 0 --repeat 1 --regression-repeat 1 \
  --output "$RUN/padding_gate" --temp-dir "$RUN/tmp"

# 3. 首轮探索：四路×八工况×五样本，没有历史组合或初筛。
bash scripts/test_qli_fusion.sh --device 0 --requests 32 --contexts 262144 \
  --mtp 0 1 3 7 --warmup 2 --repeat 5 --regression-repeat 1 \
  --output "$RUN/quick" --temp-dir "$RUN/tmp"

# 4. 上述通过后，原库独立复测；结果不与探索样本混合。
bash scripts/test_qli_fusion.sh --device 0 --requests 32 --contexts 262144 \
  --mtp 0 1 3 7 --warmup 10 --repeat 20 --regression-repeat 1 \
  --output "$RUN/confirm" --temp-dir "$RUN/tmp"
```

测两档上下文可改为 `--contexts 102400 262144`。可显式加
`--implementations split a b c peer` 保留原融合版参考；peer采用自身tiling，
其结果不参与“同一S1Base”的算法归因。默认不运行它，减少首轮成本。

### 与原始 Indexer 单独运行对比

拉取最新测试框架后，用新的 `QLI_FUSION_BUILD_DIR` 按上面的步骤重建融合设备包和
Torch 扩展。它现在同时构建 GitCode 固定提交 `9ec2110` 中
`attention/quant_lightning_indexer` 的原始算子，并为该算子安装一个轻量测试绑定。
绑定检查两个 ACLNN 入口从本次专用 vendor 加载。

```bash
bash scripts/test_qli_fusion.sh --device 4 --requests 32 --contexts 262144 \
  --mtp 3 --scenarios cold_reset full_cache_75pct \
  --implementations baseline split a b c \
  --warmup 10 --repeat 20 --regression-repeat 1 \
  --output "$RUN/native_indexer_mtp3" --temp-dir "$RUN/tmp"
```

这五路共享相同的 Q/K、scale、weights 和页表，并交替采集。冷／热标签描述的是
Plan 的缓存初态，`baseline` 两种工况都只运行原始 Indexer。它保留上游 tiling，
所以报告同时提供 `span/原始Indexer`（整段耗时比）和 `span/拆分`（评估融合收益）。
两者均使用设备 span P50。原始 Indexer 采用自身的 TopK 输出顺序，每次检查有效索引
范围、数量、重复项和 padding，并单独记录与保序 Indexer 的 TopK 集合重合度。
即便集合存在差异，仍可查看这条性能对照及其重合率。split 和融合路径之间继续执行
逐位置 TopK、一致的 Plan 状态及保护区校验。

可用 `--contexts 102400 262144 --mtp 0 1 3 7` 将同一对照扩展到此前的 16 个工况。

### 单独比较原始 Indexer 与保序 Indexer 的耗时

在 MTP3、32 请求时，上游 baseline 每块处理 4 条 query。用 `ordered4` 让保序
Indexer 也以 4 行一块单独运行。这个版本从 split 的 Indexer 生成，保留同一
算子绑定所需的占位参数，编译时禁用 Plan hook。测试核对占位状态及 TopK，设备
采集只记录此 Indexer 的 kernel。资源不足会报错，避免自动降为 2 行。

若就做这项对照，在设置好 CANN 环境和新的 `$RUN` 后，用精简包：

```bash
export QLI_FUSION_BUILD_DIR="$RUN/indexer_compare_build"
bash scripts/build_qli_fusion.sh --indexer-compare-only
source "$QLI_FUSION_BUILD_DIR/env.sh"

bash scripts/test_qli_fusion.sh --device 4 --requests 32 --contexts 262144 \
  --mtp 3 --scenarios cold_reset full_cache_75pct \
  --implementations baseline ordered4 split --warmup 10 --repeat 20 \
  --output "$RUN/indexer_breakdown" --temp-dir "$RUN/tmp"
```

split 照常使用独立 rankpipe AC Plan 库，可通过 `--library` 指向同身份的已有产物。
报告把以下三项同时列出：

- **原始 Indexer 和保序 4 行 Indexer 的 span 差值**。同为 4 行分块下的实现差异。
- **split 的保序 2 行 Indexer 段和保序 4 行 Indexer 的 span 差值**。反映切块及随之改变的调度、缓存配置。
- **split 中独立 Plan 的设备耗时**。冷、热缓存只改变 Plan 的初态。

同一行切分下的差值包含保序 Indexer 的实现差异，报告不会把它当作某个
TopK 指令的独立耗时。两条保序 Indexer 输出逐位置比较；原始 Indexer 仍报告 token 集合重合率。

## 计时与失败诊断

- 主报告同时给出设备目标执行区间并集busy、完整链路span、Event接口时间的
  P50/P95、样本数和原始CSV。busy不代表硬件单元实际利用率。
- mixed kernel若分别记录AIC/AIV，按每次真实调用配对，不能把重叠时段相加。
- split额外给出Indexer span、Plan Duration、between-kernel gap。
  gap逐对计算，不能用两个中位数相减。完整span包含间隙。
- 两条独立kernel在同一current stream连续下发；没有TopK TensorMove或D2H。
  Event仍可能包含参数处理和上传，不能称作纯kernel Duration。
- 四路逐轮轮换顺序，采集开始后显式记录1秒宿主等待，仍严格核对样本与顺序。
- CSV匹配目标算子名/构建后缀，并固定实际完整kernel名与task分组。
  混入TensorMove不计目标时间，缺少目标或多采一个均失败。
- 若目标SDK采用不同kernel命名，保留 `kernel_inventory` 后提供
  `--kernel-map` JSON：键为baseline/ordered4/split/a/b/c/peer/plan，值为精确kernel名字数组。
  不通过放宽样本数或拼接CSV绕过检查。
- 输出目录预检先于NPU初始化和大输入分配；支持 `--temp-dir`，不自动换目录。
- 失败留存 `profile.log`、原始trace、`failure_call.json` 和数值现场，失败不写
  COMPLETE或成功性能报告。结果优先放/workspace，/dev/shm需要自行转存。

## 开发机实际可做的验证

```bash
PYTHONPATH=. python3 -B checks/check_qli_fusion_plan.py
PYTHONPATH=. python3 -B -m unittest checks.test_qli_fusion_metrics checks.test_qli_fusion_fixture checks.test_qli_fusion_sync -v
python3 -B scripts/export_fusion_plan.py --check
# 已在独立上游checkout生成overlay时：
PYTHONPATH=. python3 -B checks/check_qli_fusion_resources.py --upstream /path/to/generated/upstream
```

CPU驱动执行实际Plan控制器、SIMT/SIMD函数及MrgSort/DMA语义模型，采用64个模拟
fast-path线程；覆盖连续状态、reset、尾部-1、63/64/65边界及重复resident。
这些检查不能验证真实地址空间、DCache、跨AIV发布、Fixpipe交接、寄存器spill
或真实指令吞吐，必须以服务器小测/门禁/CSV结果决定是否采用。
