# MTP7：8K容量的FUSED／AB对照

本分支基于 `bec66c3e2bbbe3e60d03cb0baba521a9d7047aa9`。原FUSED（204）、原AB（207）的计算路径与192KiB门禁继续保留。新增两条符号独立的8K预算适配路径：

| 实现 | ID | 计算来源 | 本次变动 |
|---|---:|---|---|
| `joint_mtp7_fused_8k_ref_512` | 212 | FUSED 204 | 仅放宽指定形状的UB预算 |
| `joint_mtp7_fused_ab_8k_512` | 213 | AB 207 | 同样放宽预算，保留原A/B |

AB是当前4096容量的主推候选；8K下是否更快仍待真机验证。原提交在8K下会拒绝启动，因此本轮不能声称比较了“原提交原样运行8K”的性能。两条适配路径的输入、初始缓存、UB申请与采样流程完全相同。参考路径没有人为强制GM或逐行回退，候选没有额外扩大归并包。

## 空间与边界

`TopK=2048, capacity=8192` 时：

```
Scan                              288 B
ctx                               512 B
query→UID int32[16384]           65536 B
unique int32[8192]               32768 B
复用work int32[24576]            98304 B
---------------------------------------
总动态UB                        197408 B（约192.78 KiB）
```

work在归并时存输入／中间／最终记录，之后复用为owner、flags、partition。scratch为197120B，不包含Scan；新候选scratch门槛为196KiB。只对 `TopK2048/capacity8192` 放宽，其他形状沿用192KiB门槛。Python布局、C++Prepared以及新kernel的UB／GM判断同步维护，workspace仍为1个int32占位，没有静默切到GM。

预算静态检查沿用工程的8KiB编译区域和至少32KiB DCache预留。这不是设备编译资源报告；最终可运行性、实际DCache与性能仍须CANN/NPU验证。动态UB比4096容量时增加16KiB，capacity相关扫描及LRU写回也增加，不能保证扩容更快。

两路仍为512线程、32位映射、最多612记录/源、硬件4+4→2归并。线程1发布mode的经验规避、完整描述符尾部、输入不变性和保护区检查保持。并集大于8192仍逐行回退；其他不支持条件也没有移除。

## 测试输入与比较口径

默认28请求、TopK2048、capacity8192、max_token262144、block_size64、seed2026，MTP0/1/3/7，冷启动和full_cache_75pct。保持每行因果可见性，所有实现共同在计时外按token位置排序，不去掉重复位置。

历史候选池默认从原生成器的4096扩为8192。否则只扩大缓存，所选历史token仍受限于旧4K池，不能充分覆盖本次8K路径。每条draft与首行仍至少40%集合重合。当前seed下，MTP3每组并集约5011～5080，MTP7约6853～6961，超过4096但小于8192。设置 `--query-pool-tokens 4096` 可复用原候选池，结果须与默认8192池分开记录。

full_cache_75pct描述首行名义命中比例，不能理解成每条draft都75%命中。实际每行命中、并集、miss和descriptor见metadata。测试始终恢复相同初态；不能把初始状态恢复和CPU校验计入Plan耗时。

专项门禁还包括：并集8192、8193、16384，跨包重复，空有效查询，以及冷联合→直接续用→溢出private→重新联合→请求变更。性能只在门禁通过后采集；失败保留现场，不输出成功报告。

使用现有torch_npu.profiler的目标kernel Duration。两路轮换顺序，逐样本完整校验，严格检查CSV中的kernel名、单位、样本数和调用顺序。保留启动采集后的1秒宿主等待及其记录。包含全部归并和写回，只测Plan，不含Transfer/SFA。

## 服务器一次短测

以下源码与数据均留在服务器，结果放持久化的`/workspace`。逻辑卡默认0，可将DEVICE改成空闲卡号。

```bash
set -euo pipefail
DEVICE=0
RUN=$(mktemp -d /workspace/single_op_test/capacity8k_XXXXXX)
git clone --single-branch -b feat/mtp7-capacity8k \
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

bash scripts/test_capacity8k.sh \
  --device "$DEVICE" --requests 28 --mtp 0 1 3 7 \
  --warmup 2 --repeat 5 --regression-repeat 5 \
  --output "$RUN/results" --temp-dir "$RUN/tmp" \
  2>&1 | tee "$RUN/console.log"
cat "$RUN/results/report.md"
```

构建失败时，首先查看`$RUN/results/build.log`。成功标志为退出码0、`CAPACITY8K_ALL_STAGES_PASSED`以及`COMPLETE.json`、report.md齐全。这里只编译本轮两条适配路径及工程固定基线，不展开历史组合矩阵，无初筛／复测叠加。

只构建的命令：

```bash
bash build_joint_plan.sh --soc=ascend950 --capacity8k-only
```

短测通过后，复用同一库、同一卡和门禁增加样本：

```bash
bash scripts/test_capacity8k.sh \
  --skip-build --device "$DEVICE" --requests 28 --mtp 0 1 3 7 \
  --warmup 10 --repeat 20 --regression-repeat 5 \
  --correctness-report "$RUN/results/gate.json" \
  --output "$RUN/confirm" --temp-dir "$RUN/tmp-confirm" \
  2>&1 | tee "$RUN/confirm-console.log"
```

原始CSV索引、每轮数据和来源身份见输出目录`data_index.json`、`capture_metadata.json`。这两轮分别统计，不混合5样本与20样本。若需要50样本，可单独将第二轮repeat改为50。

## 本地验证范围

可运行：

```bash
python3 -B -m unittest checks.check_capacity8k_suite
python3 -B checks/check_capacity8k_kernels.py
```

这些是CPU语义模型和UBSan验证，不是CANN编译、NPU地址空间、真实同步或性能验证。冻结计算的预处理指纹继续校验；两个适配版相对各自父版本只允许预算常量变化。原source身份校验没有关闭。

本机已通过53项框架回归，以及99次两路控制流/完整状态对照、108次冻结4K参考对照和4次8K跨warp对照；没有执行CANN编译或NPU验证。
