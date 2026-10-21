# ABCDF 接入与分阶段计时

## 1. 本次换了什么

本地分支 `feat/abcdf-step-timing` 基于同事仓库 `d89b6d4`。SIMT 计算来自用户当前工程
`simt_vs_echo_variants/csrc/plan_variants/13_abcdf`，对应提交 `9cc99d5`。
其入口包含公共计算头，因此本工程展开保存公共实现，而不是只复制三行入口文件。

- `variant_config.h` 为 `PLAN_OPT_MASK=47`，即 A/B/C/D/F；线程数为 1024。
- scratch 为 `keys[H] + firstPos[H] + owner[H] + partitionSlots[C] + missPositions[T]`，即 `3H+C+T`。
- 保留原 hash、scan、冷路径、LRU 分配、UB/GM 选择、动态 UB 大小及所有计算同步。
- 外部算子名仍为 `sparse_kv_plan`，kernel 名仍为 `sparse_kv_plan_kernel`。原比较脚本无需改调用参数；现在比较的是 ABCDF 与本目录 Echo v2.2。
- Echo kernel/状态/golden 不作算法修改；以前文档中的旧 SIMT 性能数字仍是历史数据。

来源哈希在 `source_manifest.json` 的 `simt_import` 中。`files` 列表仍是历史初始快照，
不能拿它当当前源码清单。运行时会通过 `simt_abcdf_info()` 校验实际 so 冻结的六个
SIMT 文件摘要、1024 线程、mask=47、打点开关，防止误载旧库。

## 2. 打点方法与边界

`SIMT_PERF_STEP=ON` 才启用。`SimtRecordStamp` 标记为 `__simt_callee__`，由 VF
调用链内的线程 0 读取 `clock()` 低 32 位，写入 GM 的 per-row 区域。没有新增
`asc_syncthreads`、DCache flush 或其他强制排序。普通构建预处理后移除这些观察点
及附加参数，计算路径保持来源实现。

每条逻辑 row 独占 12 个 int32 槽。这样同一 block 顺序处理多行时不会覆盖前一行。
offset 按 **maxRows** 的完整 scratch 容量计算，不按 activeRows 计算：

```text
scratchElements = rowElements <= 112KiB/4 ? 1 : maxRows*rowElements
stampOffset = align_up(scratchElements,128)    # int32 元素；保守按 512B 隔离
stamp[row,slot] = compactWorkspace[stampOffset + row*128 + slot]
```

生产形状下 `H=4096, rowElements=18432`，走 UB，普通 GM workspace 为 1 个占位
元素，打点构建为 `128+32*128=4224` 个 int32。其中每 row 只有 12 个有效槽，其余为
隔离填充，避免多个 block 的计时写入与 scratch 共用缓存行；读回同时检查填充未改写。
GM 回退路径同样在完整 scratch 尾部对齐追加，不改变算法临时区本身的布局。

这些值是**线程 0 到达源码边界的周期观测**，包含打点及编译调度的影响。
没有额外同步的阶段可能将其他线程/写缓冲的等待成本计入后续原有同步阶段。
不能把一栏解释成该函数独占、完全排空后的硬件耗时。

### 热路径：8 个戳、7 个区间

| 槽 | 区间结束点 | 本区间包含 |
|---|---|---|
| 0 | row entry | 起点；VF/TPipe/tiling 初始化在其前 |
| 1 | Prepare | current_slots 初始化、request ID 判断与原有广播同步 |
| 2 | HashInit | hash 清理、原有同步、来源上下文准备 |
| 3 | InsertTopK | 合法 query 插入 hash，原有结束同步 |
| 4 | ClassifyLRU | stablePrefix 截断、LRU hit/evict 分类及 tile scan |
| 5 | ResolveHit/CompactMiss | F 的命中发布与 miss 稳定压缩融合循环 |
| 6 | AssignMiss/ClearTail | miss 分配、写输出、清尾；保留 ABCDF 原有同步选择 |
| 7 | WriteLRU/Finalize | LRU 更新、miss_count 写出、原有末尾同步 |

ABCDF 已融合原阶段 5/6，因此不人为拆开它，也不为打点重新引入已删除的栅栏。

### 冷路径：6 个戳、5 个区间

request ID 改变时，ABCDF 提前进入 `ProcessColdRow`，跳过 hash 和历史 resident 分类。
使用槽序 `0,1,8,9,10,11`，依次表示：Prepare、ClearResidents、Validate/CompactMiss、
AssignMiss/ClearTail、WriteLRU/Finalize。热路径槽 2..7 保持哨兵。

脚本按初始 request/last ID 推导每 row 的路径，既检查应写的槽，也检查不应写的槽。
不会把未走到的热阶段报告为冷场景的零耗时。当前默认性能用例不包含 MTP 回滚。

## 3. 报告怎么读

`simt_perf_step.py` 默认 3 次污染审计、5 次预热、20 次采样。每次调用前恢复相同
公共持久状态。正式样本前只污染打点区，不整体重写 hash scratch；每个正式样本均
全量校验 SIMT 状态和输出，再读取时间戳。CPU golden 从不参与设备内部计时。

- 相邻戳以 uint32 模差计算，支持计数回绕和 int32 负数；未写哨兵为 -12345。
- 每条 row 的阶段差之和必须等于该 row 的 entry→end 跨度，跨度要求小于 2^31。
- 分核沿用 `row % gridBlocks`。多行/核时，先把该 block 拥有的 row 阶段周期累加。
- `p50_max_block_cycles`：每轮取该阶段累计周期最多的 block，再跨轮取中位数。
- `p50_block_skew_cycles`：每轮参与该路径的 block 的 max−min，再取跨轮中位数。
- `share_of_all_row_cycles_pct`：该路径全部 row、全部轮的该阶段周期和 / 全阶段周期和。
  这是**观测周期工作量分布**，加总为 100%；不是 kernel 延迟百分比。
- `max_block_span_p50_cycles`：每 block 的首 row entry→末 row end，包含该 block 的行间
  间隙，然后对各 block 取最大值、跨轮取中位数。只在同一 block 内作绝对戳差。
  不包括 kernel 外壳初始化、VF 进入和最后返回等开销，不能代替 profiler 的完整时长。
- 不同阶段的最大值可能来自不同 block，**不能把各阶段 max/P50 相加当 kernel 总时间**。

默认只输出 cycles，不假定 1.585GHz，不用 Event 或外部 P50 拟合频率。确有同机同轮
可靠频率时可以显式给 `--freq-ghz`，CSV/JSON 的 `est_max_block_us` 才是换算值；
HTML 保持 cycles，避免图表与终端使用不同的默认频率。

每次运行创建独立 `capture_*`，包含：

```text
capture.json                         完整配置、so/hash、所有轮原始戳与统计；status=complete 才成功
<场景>/stage_summary.csv             跨轮分阶段统计
<场景>/row_stamps.csv                代表轮每 row 的 12 个原始戳
<场景>/row_stages.csv                同一代表轮的分阶段差值
<场景>/report.html                   跨轮汇总 + 代表轮每 row×阶段热力表
```

代表轮是最大 block 跨度最接近跨轮中位数的一轮。热力表来自这一轮；汇总表来自全部
轮次，二者不能当同一组统计。任何采样正确性或打点检查失败，进程非零退出，capture
保持 `running`；已有的其他场景文件不能当整轮成功结果。

## 4. 服务器运行顺序

本仓 `CLAUDE.md` 要求本开发机不编译、不运行项目程序。本次仅静态检视和来源比对，
下列新增检查尚未执行；没有新增 NPU 正确性、编译或性能结果。

同步本分支文件到服务器后，激活匹配的 torch/torch_npu/CANN 环境，在工程根目录执行：

```bash
# 服务器侧 CPU 检查：布局、回绕、冷路径、哨兵、核内多行、统计口径、来源算法不变。
python3 checks/check_simt_perf.py

# 普通构建：先验证 ABCDF + Echo，再以完整 kernel profile 比较性能。
SIMT_PERF_STEP=OFF ECHO_PERF_STEP=OFF bash build.sh --soc=ascend950
python3 compare.py --device 1 --warmup 2 --repeat 4 --verify-each-sample
bash profile_kernel.sh --device 1 --scenario all --active 50

# ABCDF 打点构建；不覆盖普通 build/。
SIMT_PERF_STEP=ON ECHO_PERF_STEP=OFF bash build.sh --soc=ascend950
python3 simt_perf_step.py --device 1 --scenario all --warmup 5 --repeat 20 \
  --output results/simt_perf
```

逻辑卡号 `1` 按实际设备修改。普通 profiler/compare 会拒绝两路任一打点开启的库；
分步脚本会拒绝未启用 SIMT 打点的库。加载的 so 摘要和源码身份写入结果。

补充验证 row-stride 和 GM 回退，使用同一打点库：

```bash
python3 simt_perf_step.py --device 1 --rows 65 --topk 16 --capacity 32 \
  --max-token 256 --scenario all --warmup 2 --repeat 4
python3 simt_perf_step.py --device 1 --rows 2 --topk 4096 --capacity 4096 \
  --max-token 16384 --scenario warm_75pct --warmup 2 --repeat 4
```

Echo 的分步工具仍是 `ECHO_PERF_STEP=ON bash build.sh` + `echo_perf_step.py`。两路都
要打点可用 `SIMT_PERF_STEP=ON ECHO_PERF_STEP=ON bash build.sh`，产物在
`build_both_perf/`，运行脚本时通过 `--library` 显式指定。Echo 原有微秒标定/报告器
未在本轮重写，不应把其估算微秒与 ABCDF 的原始 cycles 直接作倍数。

## 5. 如何据此找优化空间

先看同一场景中哪些阶段的 max-block 周期和工作量占比较大，再看 skew 与热力表：
高 max、高 skew 优先查输入相关长尾、hash 探测和核内分配；普遍偏高则查扫描、
访存和同步链。以保存的原始周期验证单点改动，再用普通构建的完整 kernel profiler
确认净收益。阶段间等待会转移，某一栏下降并不自动代表整个 kernel 变快。
