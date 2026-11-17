# AB8K、rankpipe A、rankpipe AC 两档上下文对照

本入口只跑三路既有算法，不新增优化组合、不进行初筛，三路全部正式采样。

| 名称 | 工程实现名 | mask | 映射布局 |
|---|---|---:|---|
| AB8K | joint_mtp7_fused_ab_8k_512 | 213 | 原32位query→UID映射保留在UB，8K适配 |
| rankpipe A | joint_rankpipe_a_512 | 245 | 16位映射暂存GM，精确UB位图＋word起始UID查询 |
| rankpipe AC | joint_rankpipe_ac_512 | 249 | A＋MTE3批量写回LRU和miss尾部 |

虽然实现名含`mtp7`，本轮三路都只测试 **MTP3**。这是核对此前“AB8K热MTP3约58μs”与新rankpipe版本的同场专项，历史数字不作为阈值。

## 固定和变化的参数

固定32请求、MTP3（3个draft、每请求4条query，共128行）、TopK2048、capacity8192、block_size64、512线程、seed2026、overlap40、历史候选池8192。冷启动和满缓存75%两个工况分别报告。

| 长度标签 | max_token范围上界 | 已提交context | 4条query的visible_seq_lens |
|---|---:|---:|---|
| 256K | 262144 | 262128 | 262129、262130、262131、262132 |
| 100K | 102400 | 102384 | 102385、102386、102387、102388 |

这里K按1024计算；`max_token`并不等于每条query的精确可见长度。延续现有生成器，为因果可见性和边界数据保留16个位置。100K不是100000；所有实际数值都会写入报告。

每个“长度×工况”只生成一次逻辑输入，三路共同排序、保留重复token，读相同TopK、请求身份、初始resident和LRU，持有各自独立的可变状态。不同长度使用相同seed重新生成token位置，因此不是完全相同的token轨迹；各自的输入指纹单独记录。

75%是首行名义缓存命中率；draft与首行至少40%重合，不表示每条draft都命中75%。逐行命中率、跨行交集和并集大小见metadata。

## 运行

在已有匹配torch/torch_npu和CANN的环境中运行；CANN环境脚本按服务器实际安装位置加载。默认设备0，可显式改成空闲卡。

```bash
set -euo pipefail
RUN=$(mktemp -d /workspace/single_op_test/ab8k_context_XXXXXX)
git clone --single-branch -b feat/ab8k-rankpipe-context-compare \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD > "$RUN/commit.txt"
export PYTHON="$(command -v python3)"
export JOBS=3
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

bash scripts/test_ab8k_context.sh \
  --device 0 --max-tokens 262144 102400 \
  --warmup 10 --repeat 20 --regression-repeat 5 \
  --output "$RUN/results" --temp-dir "$RUN/tmp"
```

这一条包含构建、门禁和一次正式采集。构建只选择三条variant编译单元；工程链接所需的公共/独立ABCDF对象仍会构建，但不会下发ABCDF kernel或测量它。

可先短跑，将`--warmup 10 --repeat 20`改成`--warmup 2 --repeat 5`；通过后复用同一源码和库，输出到新目录，省去再次构建：

```bash
bash scripts/test_ab8k_context.sh \
  --skip-build --device 0 --max-tokens 262144 102400 \
  --warmup 10 --repeat 20 --regression-repeat 5 \
  --output "$RUN/confirm" --temp-dir "$RUN/tmp"
```

也可以单独构建：`bash build_joint_plan.sh --ab8k-compare-only`。库指纹校验继续生效，过期库必须重建。默认输出已存在会直接拒绝，避免覆盖历史结果。

只看输入说明、不需要NPU：

```bash
"$PYTHON" -B profile_ab8k_context.py --describe-only
```

## 正确性与计时

每个实际32请求用例先做5次污染审计，检查只读输入、持久状态、逐位置currentSlots、LRU、miss及输出尾部和保护区；每档长度还做三路各5步的小连续链：reset、续用设备状态、更换请求身份、stable失效、继续续用。失败保存调用次数/阶段及数值现场并立即停止。

性能阶段复用本进程内、相同输入指纹的门禁证据，省去额外重复audit；预期结果缓存只用于固定初态采集，连续状态仍逐步计算参考。**每个采样仍完整检查**，不会只查最后一次。

三路在同一个profiler采集中轮换顺序。保留进入ACTIVE后的1秒宿主等待；严格核对CSV中的精确目标kernel名、微秒单位、样本数、顺序、同流非重叠。恢复拷贝和TensorMove不计入目标耗时。

本轮要求每路都是32组联合、0回退，否则停止，不把回退性能当作联合性能比较。

比较的是设备完整Plan kernel Duration，包含归并以及rankpipe的GM映射暂存和批量写回；不是接口Event，也不是纯归并指令耗时。固定地址预热不代表生产访问轨迹或冷硬件缓存。AB8K与rankpipe的UB布局不同，按各自原配置执行，实际layout写入metadata。

## 结果与失败诊断

- `report.md`：P50/P95、数量、相对AB8K比值、联合/回退/descriptor、Block Num；另列每个实现100K/256K比值。
- `capture_metadata.json`：环境、源码/编译/库身份、输入指纹、实际可见长度、原始逐样本数据、执行顺序、墙钟耗时。
- `data_index.json`：按长度和工况索引原始CSV；长度不同不会覆盖采集目录。
- `gate.json`：固定输入与连续状态门禁证据。
- `build.log`、`launcher.log`、`RUN_STATUS.json`：构建/采集诊断及失败状态。
- `COMPLETE.json`：仅整轮成功后写出。失败保留原始trace和现场，不生成成功报告。

使用`/workspace`持久目录默认保留全部数据；若显式选择`/dev/shm`，它是临时存储，需自行转存，不自动清理或迁移。

## 开发机检查范围

```bash
"$PYTHON" -B -m unittest checks.check_ab8k_context -v
```

包括三条variant的生成/dispatch一致性、32请求参数与因果可见性、报告比值方向、无NPU描述入口，以及真实设备源码的CPU桩执行（3实现×2长度×冷/热）。CPU模型可检查语义与保护区，**不能验证CANN编译、真实设备地址空间、同步或性能**。本轮开发机无CANN/NPU；硬件结果需按上述服务器命令获得。
