# MTP7并行消费统计与归并优化

本分支基于 `c956f96252bc10c3dc2e014fd61fa97e68acfa8d`，只优化八行专有路径，不扩大线程/因素矩阵。**开发机没有CANN/NPU，以下新增候选仅通过CPU语义/UBSan与框架检查。**

## 已知事实与适用边界

用户服务器已验证上轮 `joint_hot_bd8_512`：生产TopK2048/capacity4096、32请求、MTP0/1/3/7、逻辑NPU0、CANN9.1.0，正确性门禁与5样本采集完成。动态UB为181024字节（约176.8KiB），GM scratch为1个int32占位。**这一结果支持当前布局可用，不证明继续增大UB一定可行或更快，也不替代新代码的设备验证。**

上轮冷/热MTP7分别约983/1004μs；MTP3为27/48μs。MTP3可利用大缓冲减少补料；MTP7却进入新增树式归并，并执行 `EightProgressVf` 的8线程完整重复扫描。这里是源码确认的高串行度成本；未测出其独立设备耗时，不能把1ms全部归因于这一处。

## 固定四路对照

| 实现 | public ID | 定位 |
|---|---:|---|
| `abcdf_pages` | 独立入口 | 冻结带页表ABCDF基线 |
| `joint_hot_bd8_512` | 193 | 上轮已经上机的8行版本，保留慢路径作对照 |
| `joint_mtp7_parallel_512` | 194 | A：并行消费计数，保持原硬件归并流程 |
| `joint_mtp7_rank_512` | 195 | B：并行两路rank归并，融合消费计数 |

四路保持相同逻辑输入；三个联合版本使用**同一181024字节生产动态UB布局**，标准512线程和原1024线程语义回退。没有新增GM临时区、设备时钟打点、输入恢复或模式归一化。原descriptor尾部、输入/状态/保护区检查均保留。

两个候选共用 `csrc/joint_variants/mtp8_plan_impl.h`，由 `JOINT_MTP7_OPT=0/1/2` 选择。OPT0的计算体经预处理token指纹与c956f96核对一致（忽略新增的编译期选择范围assert）；这不等于证明新旧NPU二进制或耗时完全一致，仍需同场对照。原ABCDF及旧热BD计算文件不改。

## A：每条记录只读取一次，512线程分摊

原实现：
```text
线程0：扫描整个输出，数第0行
线程1：扫描整个输出，数第1行
……
线程7：扫描整个输出，数第7行
其余504线程等待
```

改为：
```text
每线程读取 t, t+512, t+1024……这些记录
→ 线程局部累计8种来源行的数量
→ 每个warp用shuffle汇总8个计数
→ lane0将warp局部结果写入临时区
→ block barrier
→ 8个线程各汇总16个warp的一个计数，推进对应原始行游标
→ block barrier
```

每条记录只加载/解码一次，没有逐记录原子加热点。TopK2048显式使用 `qid >> 11`，其他TopK保留除法。8个局部计数按常量下标展开访问，避免动态索引局部数组；最终寄存器/溢出资源需要CANN编译报告确认。

warp部分结果最多128个int32（512字节），复用最终归并完成后已不再读取的原packet输入区。它和最终记录区不重叠，owner等后续用途尚未开始，所以不增加UB申请。原始8线程版本只在OPT0中执行。

## B：少一次MrgSort，合并最后归并与消费计数

前两次四路硬件归并不变，仍得到两段**各自安全的有序前缀**。用一个512线程VF代替“最后两路MrgSort + 单独消费统计”：

1. 未完整的半组会限制安全输出边界。选择这些半组尾token中的较小值作为cut，完整半组不限制cut。
2. 两段中只输出token≤cut的已知前缀。即使后续还有与cut相等的token，下一批仍合法，并由原有lastToken跨批去重。
3. 每个线程负责若干输入记录，在另一段中二分求排名，再将两字段记录写到最终位置。第一段同值记录排在第二段前：第一段用严格前驱数量，第二段用包含等值的前驱数量，保证写入位置唯一。
4. 同时统计这些**实际输出记录**的原始来源行。沿用A的warp归约，直接推进8行游标。
5. 后续SIMD去重、query→UID映射、resident保护、LRU/miss分配和持久状态提交保持原语义。

**绝不按中间归并消费量直接推进原行，也不把两组四行各自提交。** 未输出记录下轮继续处理，容量检查仍在持久状态写入前。

B减少硬件归并调用和一次独立VF/Scalar交接，但引入并行二分读取。它是结构候选，不保证一定优于A；若收益主要来自消费统计，A可能是更合适的统一版本。

## 本地验证及局限

- 每个候选135次完整控制流程CPU逐字段对照（共270次），与OPT0冻结8行结果一致。包含MTP0/1/3/7、生产形状、随机重复、跨未知尾部的全相同token、5/6个非空队列、空输入、因果可见性、容量恰满/超1、回退及10步真实状态续用。
- 128个CPU模拟线程/4warps下，消费统计各39种长度/TopK检查；B另有112种半组长度/完整标志/相等与交错值边界检查。总190项原语检查，含缓冲保护区、未写区域和逐位置顺序。
- CPU模型对一个生产请求统计：原版15次MrgSort，A仍15次，B为10次。**这是CPU流程计数，不是设备采样或耗时归因。**
- 79项框架回归、13种生成目录的CPU编译/链接分派检查、Python/bash语法、源码身份检查。
- 这些模型不能验证真实512线程、地址空间、DCache一致性、Bisheng资源分配或硬件性能。当前没有新增候选的NPU结果。

相关入口：
```bash
python3 -B checks/check_mtp7_fast.py
python3 -B -m unittest checks.check_mtp7_fix_suite checks.check_mtp8_suite
python3 -B checks/check_merge_catalog.py
```

## 上机短流程（逻辑NPU0）

只构建3个联合实验包装及独立ABCDF所需对象，不编译几十个历史候选。默认四路、冷/热×MTP0/1/3/7，warmup2/repeat5，共160个目标kernel样本，无初筛后再复测叠加。逐样本校验、固定参考缓存、同身份门禁复用、墙钟统计及采集ACTIVE开始1秒等待继续保留。

```bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
RUN=$(mktemp -d /workspace/single_op_test/mtp7_parallel_XXXXXX)
git clone --single-branch -b feat/mtp7-parallel-merge \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
bash scripts/test_mtp7_fix.sh --device 0 \
  --output "$RUN/quick" --temp-dir "$RUN/tmp"
```

先看退出码0、COMPLETE.json和report.md。A是否显著降低MTP7，可检验原串行统计路径的影响；B与A的差值还包含归并方式、交接与编译资源变化，不能作为某条指令的精确阶段时间。MTP0/1/3也要比较，不能只保留MTP7快的结果。

短测通过且确有收益后，再在同一源码/库下复用本轮门禁正式采集：
```bash
bash scripts/test_mtp7_fix.sh --device 0 --skip-build --formal \
  --correctness-report "$RUN/quick/gate.json" \
  --output "$RUN/formal" --temp-dir "$RUN/tmp"
```

报告、CSV索引、原始样本和源码/库身份留在本机。默认使用持久化/workspace；显式使用/dev/shm时必须转存。若失败，保留build.log、launcher.log及failure_state，不输出成功报告、不自动删除现场。不要直接复用c956f96库或旧门禁，身份检查会拒绝。
