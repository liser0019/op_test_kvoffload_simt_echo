# 28请求与同事独立Plan对照

本轮以当前工程78ed607为基础。主MTP入口默认28个请求，保留 `--requests` 以便复测32请求。MTP0/1/3/7对应28/56/112/224条query行。在56个block下，独立ABCDF分别最多处理1/1/2/4行；实际启动数量从目标kernel的CSV `Block Num`读取并记录。

## 来源与历史核对

同事分支为 `https://gitcode.com/sparse/ops-transformer/tree/wzj_qli_order_fused`，本次固定HEAD `9ec2110969a9e8502bf6d579c027ac4ea9572016`。

| 提交 | 内容 | 是否包含Plan |
|---|---|---|
| d896650 | 新增token有序QLI | 有序Indexer，不含Plan |
| 7b434c4 | 简化有序TopK cutoff过滤 | 有序Indexer，不含Plan |
| 25c3fb9 | 融合有序QLI和SparseKV Plan | 引入融合Plan，同时收录独立Plan快照与对照测试 |
| ad55e02、a3464ad、9ec2110 | 增加独立设备验证、分页用例、GLM5.2性能入口 | Plan计算文件相对25c3fb9未变化 |

融合前的分支历史里没有找到独立Plan算子目录，但25c3fb9的 `tests/host_plan/fixtures/sparse_kv_plan_apt.cpp` 收录了来自 `AmourSec/vllmascend_sep_fif@382819673c5b9147d992a67f77dca0786369bc99` 的完整独立Plan源码。此次直接接入这个快照。

融合与独立Plan的主要区别如下。

| 部分 | 独立Plan | 融合Plan |
|---|---|---|
| TopK读取 | 全部从GM读取 | `TopkSource` 优先读取当前producer的UB行，其余从GM读取 |
| 请求组调度 | 独立kernel扫描请求组，按block分派 | QLI完成后通过ready计数选出最后producer，调用有界RunGroup |
| 生命周期 | 自己初始化Plan的TPipe/TBuf | 借用QLI的pipe/storage |
| 归并、去重、resident保护、LRU、miss分配 | 独立实现 | 保留对应规划语义，新增输入和有界调度适配 |

本机实际执行了同事的 `check_plan_diff.py --sanitize`。1725次融合/独立逐字段调用对照通过，包含2329次当前producer的GM TopK污染检查，另有288项SIMD compact检查。这个结果验证CPU模型下的输出与状态语义，并不能证明两者设备耗时相同。融合还能省去生产者/消费者边界，并复用UB输入，其收益需要完整QLI输入单独测量。

原始独立源码、融合参考头、配置及许可证保存在 `csrc/peer_qli_plan/upstream`，`source.json` 固定来源提交和SHA256。`standalone_body.h`保留独立源码在注册kernel入口之前的计算体，仅清理截取处末尾空行，每次构建和加载都会验证它与快照一致。新的kernel外壳只提供直调入口、GM tiling逐字段读取和原RunAbdPlan调用。

## 本轮三路

| 名称 | 设备路径 |
|---|---|
| `abcdf_pages` | 原带页表独立ABCDF |
| `joint_mtp7_reuse_parallel_512` | 当前主候选，512线程并行消费统计＋缓冲复用，支持1/2/4/8行 |
| `peer_qli_plan` | 同事分支收录的独立Plan快照，联合512线程，逐行回退1024线程 |

三者共享同一逻辑query、可见长度、stable prefix、请求身份、初始resident和LRU。两路需要源页表的实现采用相同seed构造的合法随机页表，并校验其输入不变性。每路持有独立可变状态。

同事Plan保留源实现的112KiB scratch门槛，按本轮q_len设置maxJointRows。生产形状下，MTP7的scratch为131328字节，进入原生GM scratch路径，动态UB49696字节；MTP0/1/3使用其原UB布局。当前reuse_parallel的生产动态UB为181024字节。两种布局代表各自实现，报告记录实际路径，不扩大同事的UB以改变其算法。

两路联合算法都保留每query位置映射、重复token、MTP因果可见性、容量检查、整体回退、连续状态和原协议元数据。独立ABCDF按原逐行协议验证。新的CPU检查执行了51次同事真实独立计算体，覆盖小尺寸、生产TopK、MTP0/1/3/7、未活动行、非2次幂尾组、并集溢出、重复token和10步状态续用，输入及缓冲保护区校验通过。

主测试仍使用冷启动与full_cache_75pct标签，后者75%对应首行。draft与首行共同输入采用overlap40，因果可见性保持原样。共同输入在计时前按token ID排序，排序成本不计入设备Plan Duration。此次不采集QLI/TopK生成或Transfer。

## 新服务器编译配置与时延范围

同事独立计算体保留了 `#include "sparse_kv_plan_config.h"`。直调构建的公共 include 路径中也存在同名框架头，曾导致 namespace 别名与 namespace 定义冲突。接入目录现增加同名转接头，明确包含 `upstream/sparse_kv_plan_config.h`；上游快照、独立计算体、线程数和 scratch 布局均不变。回归检查按真实 kernel.cpp 的包含关系预处理，并把框架 include 路径放在前面，确认不会再选中框架同名配置。该检查不替代 CANN 编译。

将服务器已验证的编译修正固化，移除 `-cce-aicore-dcci-before-kernel-end=false`，保留-O3及其余SIMT选项，三路采用同一配置。旧编译库和门禁会被源码身份检查拒绝，需要重新构建。该修改的单独性能贡献尚未测出。

Profile匹配三个精确kernel名，每次轮换执行顺序，CSV核对单位、样本数和下发顺序。记录Block Num、P50/P95、实际descriptor数、共享/回退组、原始样本及源码/库身份。Plan kernel内的所有归并、同步、去重和写回均计入Duration。每次调用前恢复初态，逐样本输入、状态、输出和保护区检查保留，固定参考缓存、批量回读和ACTIVE开始1秒等待继续使用。

本机没有CANN/NPU，未执行Bisheng编译或设备正确性/性能测试。上机若失败，完整日志与数值现场保存在运行机器，失败立即停止，成功报告只在全部阶段通过后生成。

## 一轮短测

先在新服务器安装有匹配Torch/TorchNPU的Python环境中执行。输出放在持久化/workspace。

```bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
RUN=$(mktemp -d /workspace/single_op_test/peer_plan28_XXXXXX)
git clone --single-branch -b feat/peer-plan-28requests \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
bash scripts/test_peer_plan.sh --device 1 \
  --output "$RUN/quick" --temp-dir "$RUN/tmp"
cat "$RUN/quick/report.md"
```

该脚本只选择三路，自动构建、跑专项门禁并采集冷/热×MTP0/1/3/7。默认28请求，warmup2、repeat5、regression-repeat5，共120个正式目标样本，无初筛和二次复测叠加。结果必须同时具备退出码0、`PEER_PLAN_ALL_STAGES_PASSED`、COMPLETE.json和report.md。

可选的小尺寸设备检查仍走同一个脚本与严格门禁。

```bash
bash scripts/test_peer_plan.sh --device 1 --skip-build \
  --requests 2 --topk 65 --capacity 256 \
  --output "$RUN/small" --temp-dir "$RUN/tmp"
```

短测通过后，可复用同源码/库/设备且包含本轮精确输入的门禁，增加正式样本。

```bash
bash scripts/test_peer_plan.sh --device 1 --skip-build --formal \
  --correctness-report "$RUN/quick/gate.json" \
  --output "$RUN/formal" --temp-dir "$RUN/tmp"
```

本机检查入口如下。

```bash
python3 -B checks/check_peer_plan.py
python3 -B -m unittest checks.check_peer_suite
python3 -B checks/check_merge_catalog.py
```

公司输入、失败现场、采集包及二进制留在服务器。Git仓库仅保存源码、来源身份、检查代码和运行说明。
