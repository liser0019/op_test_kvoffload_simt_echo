# Echo LRU 算子深度解读（kvoffload_simt_echo）

> **本文档定位**：面向算子开发协作者的 Echo（`echo_lru`）实现深度解读。目标 是**只读这一份文档就能读懂当前代码**——每个论断都给出代码出处（`文件:行号` 或函数名），所有"为什么"都附依赖分析、硬件特性或不变量证明。
>
> **本文档随代码迭代同步更新**：每次代码合入后由维护者增量修订对应章节，并在文末"文档更新日志"表追加一行。
>
> 覆盖范围：v2 维护序重构 + 栈溢出修复 + v2-opt1/2 + opt3 负结果 + t1024 消融
> + v2.1 单趟分类（§9.17–§9.22，2026-10-19/20；v2.1 已上板验证，tag `v2.1`）
> + v2.1-prof 打点升级（§9.23，2026-10-20，已上板验证）
> + v2.2 flags 世代号化（§9.24，2026-10-20，commit `284ff71`，NPU 待验证）。
>
> ⚠ **v2 转型说明（2026-10-19）**：淘汰路径已从 pri 时间戳 + Argtopk 基数选择重构为
> lru 顺序数组 + 每步重建（§9.17）。本文 §5（20 步表）、§6（Argtopk 深读）及 §9.6–§9.12
> 中涉及 Argtopk/protect/pri 的内容**保留为 v1 历史**，与现行代码的对应关系以 §9.17 为准；
> 状态字段 `pri`→`lru`、非 spec 步骤 20→18、ECHO_PERF_NUM_STEPS 28→26（v2.1-prof
> 起 29 绝对戳、v2.2 起 27，§9.23/§9.24）、checks 重写为 50→54→56 项（v2.1/v2.2，
> §9.22/§9.24；§8.5 的 172 项为 v1 历史）。§2/§3/§4 的结构分析（Path A/UB/栅栏模型/PackedBlockScan）
> 在 v2 下继续有效。
> v1 终版快照：tag `v1-argtopk-final`。v1 末次验证数字见 §9.14/§9.15。
>
> 历史覆盖：`05046f2`（round-4b）；round-4 `3906f91`（[P-E]
> 反向 P-D + 白付清理，含 round-3 两批 f8b9c69+1121091 与修复 d461f5d，已上板验证
> §9.15）+ `9b63e24`（文档同步）。
> 工程全貌与运行命令见根目录 `README.md`，本文不重复，只做深读。

---

## 目录

1. [算子要解决的问题](#1-算子要解决的问题)
2. [与 SIMT 基线的本质差异](#2-与-simt-基线的本质差异)
3. [并行与内存结构](#3-并行与内存结构)
4. [warp 协作原语](#4-warp-协作原语)
5. [流水线逐步解读（20 个非 spec 步）](#5-流水线逐步解读20-个非-spec-步)
6. [Argtopk 基数选择深读](#6-argtopk-基数选择深读)
7. [硬件特性与避坑清单](#7-硬件特性与避坑清单)
8. [构建与验证工作流](#8-构建与验证工作流)
9. [优化迭代日志](#9-优化迭代日志)
10. [当前状态与后续方向](#10-当前状态与后续方向)
11. [文档更新日志](#11-文档更新日志)

---

## 1. 算子要解决的问题

### 1.1 业务背景

KV offload（KV 缓存卸载）场景下，推理引擎每一步都要为每个 request（行）选出 top-k 个
token，查询这些 token 的 KV 是否驻留在设备侧缓存槽中：命中（hit）则提升其新旧程度，
未命中（miss）则为其分配一个缓存槽——空闲槽不足时必须按 LRU 语义淘汰若干历史槽。
`echo_lru` 就是这个"查询 + 淘汰 + 分配 + 映射更新"的**元数据算子**：它不搬运 KV 数据
本身，只维护 token↔槽位的映射与淘汰优先级。

每行（request）完全独立：golden 参考实现全部是 `for row in range(R)`（`compare.py:169`），
行间零数据依赖——这是第 3 节 Path A 行段划分的结构前提。

### 1.2 输入、持久状态与输出的精确语义

默认尺寸 `rows=32, topk=2048, capacity=4096, max_token=262144`（`compare.py:373-376`）。
接口 schema 见 `csrc/echo/echo_lru_plugin.cpp:124-130`（14 个张量 + blocks + spec_enabled）。

**输入（只读，每步相同语义）**

| 张量 | 形状 | 语义 |
|---|---|---|
| `pos`（topk_indices） | `[rows, topk]` int32 | 本步要查询的 token id。kernel 内叫 `posGm`，语义就是 token 值 |
| `spl`（stable_prefix_lens） | `[rows]` int32 | 稳定前缀长度；投机后缀（`token >= spl`）不可靠需失效。本基准恒为 `max_token`，即 spec 路径关闭 |
| `reset_mask` | `[rows]` uint8 | 请求切换标志；=1 时该行状态整体重置 |

**持久状态（kernel 原地改写，跨步保留）**

| 字段 | 形状 | 语义 | 哨兵/约定 |
|---|---|---|---|
| `htd` | `[rows, max_token]` int32 | token → 槽位（hash-to-device 直查表） | 未驻留 = `I32_MAX`；`htd[row][0]=0`（token 0 保留） |
| `dth` | `[rows, capacity+1]` int32 | 槽位 → token | 空槽 = `I32_MAX`；**槽位 1 基**，slot 0 保留 |
| `pri` | `[rows, capacity+1]` int32 | 历史时间戳，越小越先淘汰；平局按槽号升序 | 无效槽 = -1；`pri[row][0]=I32_MAX`（沉底，永不先被选） |
| `free_slots` | `[rows, capacity+min(topk,capacity)]` int32 | 空闲槽**栈**，有效区 `[0, avail)` | 压栈在 `free[avail+k]`，弹栈从栈尾 `free[avail-1]` 往下 |
| `avail` | `[rows]` int32 | free 栈有效长度 | 终值 = `availOld + fs - nm ≥ 0`（见 §5 Step4） |
| `fifo` | `[rows]` int32 | 本轮时间戳发生器 | 每轮自增**两次**（hit 保护后一次、分配后一次，`compare.py:187,203`） |

**输出（每步覆写）**

| 字段 | 形状 | 语义 |
|---|---|---|
| `current_slots` | `[rows, topk]` int32 | 每个 query 的最终槽位：hit = 原槽位（Step1 写），miss = 本轮分配的 `aslot`（Step5 回写，`echo_lru_kernel.h:950`） |
| `miss_host_pos` | `[rows, topk]` int32 | miss 位置填 token 值（host 由此知道去主机取哪个 token 的 KV），hit 填 0（`compare.py:207`） |
| `miss_alloc_flat` | `[rows, topk]` int32 | miss 填 `row*capacity + aslot`（**保留原接口约定，1 基槽位不减 1**），hit 填 0；**hit 位的 0 不是合法 miss 地址**（`compare.py:208`、`echo_lru_kernel.h:945`） |
| `miss_mask` | `[rows, topk]` uint8 | 1 = miss。输出不做压缩（SIMT 是压缩前缀 + `miss_count`，两者格式不同） |

`effTopk = min(topkDim, B)`（`echo_lru_plugin.cpp:77`）：防 buffer 超订的有效查询宽度。
`j >= effTopk` 的位置是 inactive 尾巴：`curSlots=0`、`flat=0`、`mask=0`
（`echo_lru_kernel.h:279,933`）。本基准 `topk ≤ capacity`，无 inactive 区。

### 1.3 一轮的语义时序

```
reset（若换请求）→ hit/miss 判定与计数 → hit 槽位 pri=fifo（保护）→ fifo++
→（若 nm > avail）按 pri 选出 fs 个最旧槽淘汰并压回 free 栈
→ 从 free 栈尾弹出 nm 个槽分给 miss → 新槽 pri=fifo' → fifo++、avail 折叠更新
→ miss 前缀秩扫描 → htd/dth/输出/curSlots 映射更新
```

其中 `nm` = miss 数，`fs = max(nm - avail, 0)` = 需淘汰数（`echo_lru_kernel.h:305-307`）。
CPU 参考 `reference_echo`（`compare.py:159-211`）按同样时序用排序表达淘汰语义。

### 1.4 与 golden 的关系

- **golden 只做结果验证，不参与计时**：CPU 参考"只有需要淘汰时才排序"，这是参考实现
  省掉无效计算（`compare.py:160-163`）；**设备实现无论是否淘汰都走常数趟数的基数选择
   路径**（`fsClamp=0` 时提前返回，`echo_lru_kernel.h:420-423`），所以两者只在"输出"上等价，
  不在"过程"上等价。
- 验证方式是**逐字段全等**：`check_npu_result`（`compare.py:324-329`）把全部 6 个状态
  字段 + 4 个输出与 `reference_echo` 逐一比对，包括"没有使用的输出尾部、free 栈余量"
  （README §5）。
- `check_common_semantics`（`compare.py:214-236`）再从**原始 LRU** 独立推导"最旧的未命中
  历史 token"集合做交叉核对，避免两份参考互相佐证的假通过。

### 1.5 状态构造：SIMT 历史如何翻译成 Echo 状态

`make_echo_initial`（`compare.py:80-108`）把 SIMT 的初始历史翻译成 Echo 状态，三条关键规则：

1. **槽位平移**：SIMT 槽 s（0 基）↔ Echo 槽 s+1（1 基），token 0 / slot 0 保留不用；
2. **pri 单调**：`lru_slots` 最旧→最新，rank 越小越旧，所以 `pri = rank + 1`（递增），
   **不能写 `capacity - rank`**（`compare.py:83-84` 注释）；`fifo = capacity + 1` 保证当前轮
   时间戳高于全部初始历史值；
3. **free 栈倒序预埋**：空槽升序取出，故 `free[:n] = empty_slots[::-1]`，弹栈时小槽先出
     （`compare.py:105-107`），与 kernel reset 行为 `free[col] = B - col`（`echo_lru_kernel.h:130`，
    弹出顺序 1,2,3,…）语义一致。

基准四场景（`SCENARIOS` 定义见 `compare.py:24-29`；同 seed 同 query，只差预埋与是否
reset，场景语义表见 README §3）：

| 场景 | 每行 miss | 初始 avail | 需淘汰 fs | 特点 |
|---|---:|---:|---:|---|
| `cold_reset` | 2048 | 4096 | 0 | 全量 reset，htd 重写是带宽地板 |
| `warm_75pct` | 512 | 2560 | 0 | 部分命中，无淘汰 → C1 整段跳过 |
| `warm_all_hit` | 0 | 2048 | 0 | 全命中 |
| `full_cache_75pct` | 512 | 0 | 512 | **缓存满，每步 miss 必然淘汰**——解码稳态的代表场景，Argtopk 唯一跑满的场景 |

### 1.6 基准口径与真实 decode 的对应

**基准是单步口径**：每次计时迭代都先 restore 同一初态再计时——`measure_one`
（`compare.py:335-347`）的序列为 restore_state（内含 sync，`compare.py:298-302`）→
start.record → invoke → end.record → sync。四场景数字都是"从各自初态出发走**一步**"
的成本，观测不到跨步状态演化。

**多步推演**（不 restore、同一请求连续解码，假设每步持续 512 miss）：warm_75pct
初态 avail=2560（§1.5 表；`compare.py:105-107`），每步 −512 → **5 步耗尽，第 6 步
起 fs=512 恒淘汰（avail 恒 0）≡ full_cache_75pct 稳态**。因此四场景对应的真实
decode 语义：

| 场景 | 真实 decode 中的角色 |
|---|---|
| `warm_75pct` | 请求生命周期前 5 步的**瞬态**（free 栈尚有余量，C1 整段跳过） |
| `full_cache_75pct` | **长解码稳态**的代表（每步持续 miss>0 时，free 栈耗尽后恒淘汰） |
| `warm_all_hit` | 理想化上界（全部命中、零淘汰） |
| `cold_reset` | 请求切换事件（reset_mask 全 1，`compare.py:63,76`） |

**远期成本提示（单步口径观测不到）**：fifo 每步 +2（`compare.py:187,203`），长期
运行 fifo 超过 0xFFFF（自初值 4097 起约 3 万步）后 Argtopk 跳趟降为 3 趟（§6.5
startPass 规则）；基准每次 restore 初态 fifo=4097（`compare.py:93`）恒 2 趟，该
成本在本基准中不可见。维护序 v2（§10.3②）可顺带消除。

---

## 2. 与 SIMT 基线的本质差异

### 2.1 两种"序"的哲学

这是理解一切性能差异的结构根源：

- **SIMT 的 `lru_slots` 是"维护好的序"**：一个显式的槽位排列数组，materialized 在 GM。
  每步都必须做 O(C) 的分类扫描（阶段 4 按 LRU 顺序稳定分出 evictable/hit，
  `sparse_kv_plan_kernel.cpp:292-334`）+ O(C) 的 LRU 重建（阶段 8："剩余可淘汰 → 新 miss
  → 原 hit"，`sparse_kv_plan_kernel.cpp:384-397`）。**淘汰 = 走链表头**：`evictableSlots[0..]`
  前缀即最旧可淘汰槽，拿前 `assignCount` 个即可（阶段 7，`sparse_kv_plan_kernel.cpp:373-381`）。
  每步成本稳定 O(C + topk)，与是否淘汰无关。

- **Echo 的 `pri` 是"派生的序"**：pri 只是时间戳，平时不维护任何排列。hit 只做 O(topk)
   次散射写 `pri[slot]=fifo`（OpHitMiss 的 hit 分支顺手写，protect 已折叠，§5 Step1）；
   **只有需要淘汰时**才用 O(B) 的 4 趟基数选择
   （radix select）从全量 key 中"现算"出 pri 最小的 fs 个槽（OpArgtopkRow）。无淘汰时
   （fs=0）整条淘汰流水线被 C1 跳过（§5 Step2）。

```
SIMT:  每步 ──> O(C) 分类 ──> O(topk) hash 查/插 ──> O(C) 重建序   (序是持久的)
Echo:  每步 ──> O(topk) 直查表判 hit/miss ──┬─ fs=0 ──> 结束      (序不存在)
                                            └─ fs>0 ──> O(B) 基数选择现算最旧 fs 个
```

### 2.2 结构性后果

| 维度 | SIMT | Echo |
|---|---|---|
| 查询 | open-addressing hash（CAS/min atomic 插入，`sparse_kv_plan_kernel.cpp:160-190`） | **直接寻址表 htd**：`htd[row, token]` 一步出槽位，无 hash 冲突（O(1) 随机 GM 读） |
| 淘汰 | 走维护序的链表头，O(fs) | 全量基数选择，O(B)（趟数×扫描宽度） |
| 无淘汰步成本 | 仍付 O(C) 分类+重建 | 近似只剩 O(topk)（C1 跳过） |
| 线程组织 | 2048 线程/块，行跨块 stride 轮转（`sparse_kv_plan_config.h:12`、`sparse_kv_plan_kernel.cpp:27-28`） | blockDim 按 topk 运行时自适应（生产 topk=2048 → 512，上限 1024；§3.2），**每核连续行段**（Path A，§3.1） |
| miss 输出 | 压缩前缀 + `miss_count` | 非压缩 mask + flat 地址 |

由此得出各场景的胜负手：warm（无淘汰）场景 Echo 少付整个 O(C) 维护成本 → 快；
full_cache（每步必淘汰）场景 Echo 每步多付整个 O(B) 选择成本 → 慢（§10 的 kernel 级
数字：warm 0.67-0.72× vs full 1.29×）。**这不是实现优劣，是状态表示的代价分布不同**。

**warm→full 边际成本（v2 的最强定量动机；全部实测数据，出处 §9.15 上板验证）**：把
同一对实现从"无淘汰"推到"每步必淘汰"，两路的增量完全不对称——

| | Echo | SIMT | 差 |
|---|---:|---:|---|
| warm_75pct 基线（fs=0） | 58.28 μs | 81.20 μs | Echo 赢 22.92 |
| full_cache（每步 fs=512） | 120.12 μs | 93.21 μs | Echo 输 26.89 |
| **warm→full 边际成本** | **+61.84** | **+12.01** | 边际差 49.83 |

① **SIMT 的成本对场景平坦**：2×O(C) 维护（分类 + 重建）每步必付、与 fs 无关——
warm_all_hit（0 miss）与 warm_75pct（512 miss）的 SIMT 耗时几乎相同（82.26 vs
81.20 μs，§9.15）可证；**Echo 的成本是尖峰**：fs=0 时 C1 跳过、O(B) 选择归零，
fs>0 一付就是整个 O(B) select。② Echo 的查询路径领先（22.92 μs，直查表 vs
hash+CAS）**不足以覆盖**每步淘汰的边际成本差（49.83 μs）→ full_cache 稳态净输
26.89 μs。③ 推论：v2 把"每步从零发现"（+61.84）换成"每步增量维护/搬运"（估
+12~15），warm/full 两条曲线都压到 SIMT 之下（§10.3② 的定量依据）。

### 2.3 状态不可逐字段对比

两种实现**不能要求分配相同槽号**（README §4）：SIMT 按已有 LRU 顺序分配、Echo 用栈，
分配顺序与多轮时间戳平局策略都不同。因此验证方法只有两条，缺一不可
（`compare.py:214-236`）：

1. **各自与自己的 golden 全等**（全部状态字段 + 全部输出）；
2. **共同语义核对** `check_common_semantics`：命中率、最终 query→token→槽位映射
   （`slot_to_token[cur_slots]==query` 与 `dth[cur_slots]==query`、`htd[query]==cur_slots`
   双向）、被淘汰的历史 token 集合（从原始 LRU 独立推导）。

---

## 3. 并行与内存结构

### 3.1 Path A：每核独立行段，无跨核同步

排布（`echo_lru_kernel.h:13-18`）：

```
grid: blockNum 默认 = min(rows, vector cores)，可用 --blocks 覆盖（1..rows，compare.py:273-283）
每核处理连续行段 [blockIdx.x * rowsPerCore, +rowCount)，尾核 rowCount=tailRows
行内 blockDim.x 线程 grid-stride 扫列（plugin 按 topk 运行时自适应，生产 topk=2048 → 512，§3.2）
```

**为什么可以完全不跨核同步**：算法每行独立（§1.1），而每行只被一个核处理 → 任何
GM 地址的写者与后续读者要么同核（行内跨线程 RAW，用块内栅栏解决），要么不存在跨核
写后读。`SyncGrid()` 的注释明说"Path A: 每核只处理自己连续行段, 行间零数据依赖,
无需跨核 sync"（`echo_lru_kernel.h:1186-1187`）。

**历史包袱（为什么叫 Path A）**：早期曾攻关 Stage 2 全元素 grid-stride 跨核协同
（需要 `grid_group::sync()`），结论是"极简 probe 全 PASS，但 echo_lru Process 内哪怕
一个 grid.sync 即 trap，根因是编译单元整体属性，不可复现"，2026-09-16 暂停，回退
Path A（`echo_lru_kernel.h:1189-1194`、`:1345-1346`；详见上游 `GRID_GROUP_PITFALLS.md`，
该文件不在本仓）。`csrc/echo/echo_lru_tiling.h:1-9` 头部注释仍残留"Plan B /
grid_group::sync"时代描述，**以 Path A 实际代码为准**（README §9 亦有说明）。
`blockNum ≤ 向量核数` 的约束仍由 plugin 强制（`echo_lru_plugin.cpp:91`）。

### 3.2 tiling 常量表

全部在 `csrc/echo/echo_lru_tiling.h`（host/device 共享）：

| 常量 | 值 | 含义 | 出处 |
|---|---:|---|---|
| `ECHO_LRU_THREAD_NUM` | 1024（默认） | 每核线程数**上限**（= `__launch_bounds__`，`echo_lru_kernel.h:1312`）。实际启动线程数由 plugin 按 topkDim 运行时自适应（公式见下）；仅当 `ECHO_LRU_THREADS` 手动压低上限时才直接约束启动值。static_assert 三条：32 的倍数（warp 大小）、≥64（覆盖 G 路直方图切片）、≤1024（AIV launch 上限） | tiling.h:18-34 |
| `ARGTOPK_B_MAX` | 4096 | B（=capacity）上界，plugin CHECK | tiling.h:43 |
| `ARGTOPK_NPAD_MAX` | 128 | 遗留常量，当前无实际引用（仅出现在一行旧注释里） | tiling.h:44 |
| `ARGTOPK_GM_NPAD_MAX` | 8192 | = next_pow2(B_MAX+1)：GM 调试 dump 每核 srcKey 容量 | tiling.h:45 |
| `ARGTOPK_UB_NPAD_MAX` | 8192 | key 打包元素上界（= next_pow2(B+1)，B=4096 时） | tiling.h:47 |
| `ARGTOPK_UB_ELEMS` | 32768 | 单大 `__ubuf__` 数组总元素（4×8192，128 KiB） | tiling.h:48 |
| `ARGTOPK_GROUP_THREADS` | 64 | 并行直方图线程数 G（各独占 hist[256] 切片，纯写无争用） | tiling.h:49 |
| `ECHO_PERF_NUM_STEPS` | 28（v1 值） | 分步打点步数（非 spec 20 + spec 8）；v2 起 26、v2.1-prof 起 29（绝对戳，§9.23）、**v2.2 起 27**（entry + 非 spec 18 + spec 陈列 8，§9.24） | tiling.h:53-57 |

**G=64 是"乘出来的"设计点**（而非扫描出的甜点）：私有切片免 atomic（各线程独占
hist[256] 切片纯写，§6.2；同址 atomic 规避原则 §7.1）× 256 桶/片（8-bit 基数）×
int32 切片宽（设计选择：每片每桶计数上界 ⌈4097/G⌉=65，int8 本可装下，未窄化）×
128 KiB 单大 `__ubuf__` 预算（需同时容纳 key/hist/globalCnt/seg/scalar/freeIdx，
§3.3）——四个因子相乘，G=64 恰好放得下（25096+2048=27144 ≤ 32768）。窄化切片可把
G 抬至 128/256，但 merge 每桶归并成本随 G 线性变贵，且 [P-E] 后 pass1 hist 已
UB-fed、hist 相位已非主导（§9.15 验证结论 6 的扩 G 候选已撤下）——维持 G=64。

**线程数：运行时自适应为主 + 编译期 knob 为辅**。plugin 在 launch 前按 `tiling.topkDim`
计算实际启动线程数（`echo_lru_plugin.cpp:104-108`）：

```
launchThreads = clamp( round32(ceil(topk / 4)), 64, ECHO_LRU_THREAD_NUM )
```

（`round32` = 向上取整到 32 的倍数，warp 对齐；下界 64 覆盖 G=64 路直方图；上界 =
`ECHO_LRU_THREAD_NUM`，默认 1024 = `__launch_bounds__` 上限，kernel 允许以少于上限的
线程启动。）`launch_echo_lru` 带 blockDim 参数，`<<<blockNum, blockDim>>>`
（`echo_lru_launch.h:5-13`、`echo_lru_launch.cpp:8-9`）；kernel 内部全部 `blockDim.x`
自适应（§4.4）。topk → blockDim 对照：

| topk | ceil(topk/4) | round32 后 | clamp 后 blockDim |
|---:|---:|---:|---:|
| 100 | 25 | 32 | **64**（下界兜住） |
| 1536 | 384 | 384 | **384** |
| 2048 | 512 | 512 | **512**（消融甜点，§9.10） |
| 4096 | 1024 | 1024 | **1024**（上界） |

设计依据有二：① 消融实测 512 线程全场景最优（§9.10），topk=2048 → 512 自动落在
甜点；② `ceil(topk/4)` 保证 OpMissCumsum 4 元素并行门的线程侧条件
`4 × blockDim ≥ topk`（§5 Step5）在 topk ≤ 4096 内恒成立。

编译期 `ECHO_LRU_THREADS`（`build.sh --threads=N` / 环境变量选择；CMake 经
`-DECHO_LRU_THREAD_NUM_OVERRIDE` 同时注入 kernel 与 plugin，**必须同值**，
`CMakeLists.txt:100-113,165-170`）定位为**纯消融 knob**：压低上限即压低自适应结果的
天花板，此时 `topk > 4×上限` 会使 OpMissCumsum 回退 t0 串行（§5 Step5）。正式构建
保持默认 1024。

`EchoLruTiling` 结构（tiling.h:58-81）由 plugin `Check()` 填充：`rowsPerCore/tailRows`
划分行段，`topkBufferB`=capacity，`freeStackSize`=B+min(topk,B)，`effTopk`=min(topk,B)，
`specEnabled`（host 按 min(spl)<maxModelLenP1 判定，本基准恒 0），
`argtopkGroupThreads`=G（`echo_lru_plugin.cpp:68-81`）。kernel 入口从 GM tiling 缓冲逐
字段重载（`echo_lru_kernel.h:1318-1335`）。

### 3.3 单大 `__ubuf__` 数组分段布局

kernel 入口声明 `__ubuf__ int32_t argtopkUb[ARGTOPK_UB_ELEMS]`（128 KiB 静态 UB，
`echo_lru_kernel.h:1341`），`Init` 手动分段（`echo_lru_kernel.h:1036-1042`）：

```
argtopkUb（每核一份, 块内共享, 无 myBlock 偏移）, npad = next_pow2(B+1) = 8192, G = 64
┌───────────────────┬────────────────────────┬──────────┬──────────┬─────────┬───────┬─────────────┐
│ key               │ hist (G 个切片)        │ globalCnt│ segLocal │ segBase │ scalar│ freeIdx     │
│ [0, npad)         │ [npad, npad+G*256)     │  256     │  G*2     │  G*2    │  8    │ [25096, +fi)│
│ 8192              │ 16384                  │          │ (预留)   │ (预留)  │(无读者)│ (条件段,P-C)│
└───────────────────┴────────────────────────┴──────────┴──────────┴─────────┴───────┴─────────────┘
固定段合计 = 8192+16384+256+128+128+8 = 25096
freeIdx 段 fi = rowsPerCore*topkDim, 门: 25096+fi ≤ 32768（基准形态 27144 ✓; 大 batch 回退 GM）
```

**[P-C] freeIdx 迁 UB（带门）**：freeIdx 的读者全在本 kernel 内（OpFreeRelease /
OpFreePush / OpAllocPop），且步骤按行批处理需容纳 `rowsPerCore*topkDim` 个元素——
放得下就走 UB（省 GM 写 + emission 尾 dcci + 下游 GM 读），放不下（大 batch 形态）
回退 GM（`echo_lru_kernel.h:1059-1069`，成员 `freeIdxUb_/freeIdxPtr_/freeIdxInUb_`
`:1305-1307`）。freeIdx 段位于 scalar 之后的固定段峰值之外 [25096, +fi)，与
key/hist/globalCnt 等固定段零重叠；Step5 的 scan 缓冲只复用 key 区 [0,128)（见下），
且彼时 freeIdx 已无读者（末次读者是 Step3 的 OpAllocPop）→ 分时复用无冲突。

**为什么必须单大数组**：`probe_ub_size` 实测（Init 与 kernel 入口注释，
`echo_lru_kernel.h:1030-1031,1336-1339`）：
- 多个独立 `__ubuf__` 数组会**别名互相覆盖**（读 A 得 D）；
- 单 `__ubuf__` 数组可到 192 KiB（256 KiB 才 trap）。

所以 key/hist/globalCnt/seg/scalar/freeIdx 全部分段塞进一个数组。`segLocal/segBase`
是预留的分段 scatter 缓冲（基数选择路径未用，`(void)` 压未用告警，`echo_lru_kernel.h:615-616`）；
`scalar` 在 [方向2] 决策本地化后**无任何读者**，槽位保留只为防 UB 布局漂移
（`echo_lru_kernel.h:617`）。

**UB 的分时复用**（同一块内存三个生命周期，互不重叠）：

```
Step1:  argtopkKeyUb_[0..64) 作 warpScratch（行奇偶双缓冲 warp 归约 miss 计数）, [64] 作 C1 淘汰标志
Step2:  key/hist/globalCnt/scalar 全量投入 radix select；freeIdx 段（若启用）作选择输出
Step5:  ArgtopkAllRows 早已结束 → argtopkKeyUb_[0..128) 复用作 OpMissCumsum 的 scan 缓冲
         （2 元素: 行奇偶双缓冲 [0,64)；4 元素: (行奇偶×scan序) 四缓冲 [0,128)，echo_lru_kernel.h:796,868-869）
```

### 3.4 GM workspace 布局与公式逐项对应

> [v2/v2.2 转注] 下图为 v1 形态：v2 已删 argtopk 诊断 dump 段（dbgCount/dbgKey）
> 并增每行 flags(B+1) 段；v2.1-prof 起 stepCycles 更名 stepAbs（t0 绝对时间戳低
> 32 位，§9.23）、ECHO_PERF_NUM_STEPS 现值 27（§9.24），且现行 host 公式
> （`echo_lru_launch.h` 的 `EchoWorkspaceElements`）按符号引用 perBlock =
> ECHO_PERF_NUM_STEPS（不再写死数值）。现行布局以 launch.h 头注释与 `Init` 为准；
> 本节"两侧公式逐项对应"的核对方法仍有效。

kernel 侧布局在 `Init`（`echo_lru_kernel.h:1011-1056`），host 侧公式在
`EchoWorkspaceElements`（`csrc/echo/echo_lru_launch.h:21-30`），两者逐项一致：

```
int32 元素偏移
[0 ───────────────────────────── 每行工作区, 长度 R*(B+1+4+3*topk) ────────────── )
   wsCumsum   R*(B+1)     spec 路径的前缀和暂存（非 spec 不用）
   pushCount  R           spec 路径释放计数
   numHits    R           [P0] 已无读写, 槽位保留防布局漂移 (kernel.h:1003)
   numMisses  R           每行 miss 数
   freeIdx    R*topk      radix select 的输出（被淘汰槽列表, 行 stride = topkDim）
   freeSize   R           每行需淘汰数
   rawSlots   R*topk      分配给 miss 的槽
   wsMissRank R*topk      miss 的独占前缀秩
[align2(base) ─────────── 每核调试/打点区, 长度 blocks*(256+8192[+28]) ───────── )
   dbgCount   blocks*256            [诊断3] 每核 pass0 归并直方图 dump (uint32)
   dbgKey     blocks*8192            [诊断3] 每核打包 srcKey dump
   stepCycles blocks*ECHO_PERF_NUM_STEPS   [ECHO_PERF_STEP] 每核每步 t0 clock() 周期
```

对应关系：`base = rows*(capacity + 1 + 4 + 3*topk)` 中的 `1` = wsCumsum 的 `B+1`、
`4` = pushCount/numHits/numMisses/freeSize、`3*topk` = freeIdx/rawSlots/wsMissRank
（freeIdx 的 GM 段保留为 [P-C] 大 batch 回退落点，`echo_lru_kernel.h:1059-1069`）；
`(base+1)&~1` 对应 Init 的 `elemOff = (o+1)&~1`（8 字节对齐）；`perBlock = 256 +
ARGTOPK_GM_NPAD_MAX(8192)`（perf 构建再 +28 = ECHO_PERF_NUM_STEPS）。

**注意**：`compare.py` 分配 workspace 用的是**旧版上界公式**（含 GM ping-pong/直方图
占位段，每核 `4*npad + 64*256*2 + 256 + 8192` 个元素，`compare.py:284-292`），严格大于
新版所需，因此同一份测试可同时兼容普通/打点两种构建。`echo_perf_step.py` 的
`echo_step_region_offset`（`echo_perf_step.py:71-75`）则按**新公式**精确定位 stepCycles 区。

### 3.5 栅栏模型：syncthreads + dcci 三明治

两个原语（`echo_lru_kernel.h:1195-1214`）：

- `asc_syncthreads()`：块内栅栏，解决**行内跨线程 RAW**——同一 GM/UB 地址被不同线程
  先写后读（例：OpHitMiss 的 hit 分支以 topkDim 步长写 `pri[row,s]`，OpArgtopkRow 以
  B+1 步长读 `pri[row,i]`，同地址不同线程；头注释举例 `echo_lru_kernel.h:17-18`）。
- `asc_dcci_entire(ptr)`：刷**本核 D-cache**。dav-3510 上单核内 GM 写后读可能命中
  stale D-cache 行，必须显式刷（这是 tiling.h 头注释所述 hybrid 版 "stale pri" 的根因
  之一，纯 SIMT 用 dcci 解决，`echo_lru_tiling.h:7-8`）。

`SyncGrid()` 的三明治结构：

```cpp
asc_syncthreads();                          // 1) 全员写完成且排序在 dcci 之前
if (threadIdx.x == 0) asc_dcci_entire(wsBase_);  // 2) t0 刷本核 D-cache (整核一份, 一线程足够)
asc_syncthreads();                          // 3) 全员的后继读排序在 dcci 之后
```

**为什么 t0 做一次就够**：dcci_entire 作用于整核 cache，块内全部线程共享
同一份 D-cache，多线程重复刷只是浪费（且一次 ~成本不低，见 §7）。

**SyncBlock vs SyncGrid**：代码里两者实现相同（`echo_lru_kernel.h:1195-1214`），语义上
区分"块内协作栅栏"与"步间栅栏"。Process 内裸 `asc_syncthreads()`（不带 dcci）自
round-4b 起只剩一处（`echo_lru_kernel.h:1178`）：Step4 的 **Bug2 栅栏**——要解决的
是"**读先于写**"的排序（Bug2，§5），写在栅栏之后发生，不存在 stale 读问题；后续读者
可见性由更后面的完整 SyncGrid 兜底。（Step2 freeIdx RAW 栅栏的 [P-C] UB 路径裸
syncthreads 已被 round-4b 整段跳过取代——UB 路径的排序由 ArgtopkAllRows 末行尾
栅栏 1 提供，GM 路径仍走完整 SyncGrid，§5.1/§9.16。OpArgtopkRow 行内的
B0/B1/B2/emission 栅栏也是裸 syncthreads，属行内结构，见 §6.2/§6.7——emission 尾部
自 round-4b 起为"尾栅栏 1 + GM-only 尾栅栏 2"两段式（UB 路径止于尾栅栏 1，
`echo_lru_kernel.h:612-624`）。另有两处步间 SyncGrid 自 round-4 起按条件门控：
prologue FillZero 后的 SyncGrid 仅 spec 时执行、Step0 尾 SyncGrid 仅本核行段有
reset 时执行——跳过论证见 §9.15 白付清理 A/B。（[v2/v2.2 转注] 该门控描述为
round-4/v1 形态：v2 起 prologue SyncGrid 因 OpZeroFlags 的 WAW 排空转为无条件
执行，v2.2 起随 flags 世代号化**整体删除**——现行 prologue 仅剩 spec 的
OpFillZero，无后随栅栏，§9.24。）

---

## 4. warp 协作原语

`PackedBlockScan`（`echo_lru_kernel.h:53-69`）是全部 warp 协作的基础（OpArgtopkRow 的
emission、OpMissCumsum 的秩扫描都用它），对照 SIMT 基线的 `BlockExclusiveScan`
（`sparse_kv_plan_kernel.cpp:107-139`）同构。

### 4.1 单栅栏结构（[Scan-v2]）

```cpp
// 输入: packedValue = 字段A | (字段B << shift)
uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
uint32_t warpCount = blockDim.x >> 5;              // ← 运行时推导 warp 数
int32_t inclusive = packedValue;
for (offset = 1; offset < 32; offset <<= 1) {      // 一级: warp 内 shuffle inclusive 扫描
    prior = asc_shfl_up(inclusive, offset, 32u);   //   log2(32)=5 轮, 零栅栏
    if (lane >= offset) inclusive += prior;
}
if (lane == 31u) warpTotals[warp] = inclusive;     // 每 warp 最后一个 lane 写 warp 总和
asc_syncthreads();                                 // ── 唯一栅栏: warpTotals 写 → 全员读
int32_t base = 0;
for (w = 0; w < warp; ++w) base += warpTotals[w];  // [Scan-v2] 每线程本地重算自己 warp 的前缀
return base + inclusive - packedValue;             // exclusive = warp 基址 + warp 内 inclusive − 自身
```

- **一级**（warp shuffle）：warp 内 32 lane 做 inclusive 前缀和，5 轮 shuffle 完成，
  warp 同步由硬件保证，无需栅栏。`lane >= offset` 时才累加，保证 exclusive 语义可由
  `inclusive - packedValue` 得到。
- **二级（本地重算）**：warp 基址 `base` 是 warpTotals 的**确定性函数**——给定栅栏后
  的同一 warpTotals 快照，每线程自己 `Σ warpTotals[0..warp)` 即可（≤32 次 UB 读，
  warp 数 ≤ 32），无需 t0 串行扫描、无需 warpBases 数组、无需第二次栅栏（[Scan-v2]
  删除 B2，`echo_lru_kernel.h:64-68`）。
- **grand total 本地 Σ**：全块总和相关论断（emission 的 totalLt/totalEq、秩扫描的
  total0/t0/t01/t2）也由**调用方**在返回后各自 `Σ warpTotals[0..warpCount)` 得到，
  `control[0]` 广播槽已删除。

### 4.2 字段打包防进位规则

打包值是两个字段拼进一个 int32：`value = 字段A | (字段B << shift)`。**两个字段各自的
块内总和都必须 < (1 << shift)**，这样打包加法在各自字段内进位不会越界污染另一字段
（`echo_lru_kernel.h:46-47` 注释，"由调用方保证"）。两个调用点：

| 调用点 | 打包 | 字段上界论证 |
|---|---|---|
| OpArgtopkRow emission（`echo_lru_kernel.h:579`） | `nLt \| (nEq << 16)` | 块内总和 ≤ validN = B+1 ≤ 4097 < 2^16（`echo_lru_kernel.h:578`） |
| OpMissCumsum 2 元素路径（`:797`）/ 4 元素路径（`:870-871`） | `ind0 \| (ind1 << 12)`；4 元素再链一次 `ind2 \| (ind3 << 12)` | 块内各字段（各 tile）总和 ≤ topkDim ≤ 2048 < 2^12（`:753,816-817`） |

### 4.3 为什么只有 1 次栅栏（B2 的删除论证）

SIMT 基线的 `BlockExclusiveScan` 有第 3 次栅栏："保证所有线程已读取 warpBases，下一次
scan 才能复用它"（`sparse_kv_plan_kernel.cpp:105`）。Echo 先由 P4 删掉这次保护栅栏，
再由 [Scan-v2] 删掉 t0 广播 warpBases/control 的第 2 次栅栏（B2），只保留 warpTotals
写 → 全员读的 B1。安全性依赖一组顺序论证（前提：**所有调用方都是统一控制流的逐行
循环**，每线程经历的栅栏数相同）：

- **base 是确定性函数 → 不需要广播**：每线程 B1 之后读到的 warpTotals 是同一快照，
  本地重算的 base 与 grand total 逐线程一致——这就是删 B2 的全部依据；
- **(a) 线程 X 读 warpTotals**（B1 之后、返回前，含调用方的 grand total 本地 Σ）
  **vs lane31 下一轮写它**（下一轮 B1 之后）：lane31 要写必须先等所有线程（含 X）
  到达下一轮 B1；而 X 的读发生在 X 到达下一轮 B1 **之前** → **读先于写**，安全；
- **(b) 背靠背两次调用免栅栏**（OpMissCumsum 4 元素路径的链式 scan1/scan2）：同一行
  的 scan1 与 scan2 写的是**不相交的缓冲**（四缓冲按 (行奇偶 × scan 序) 轮转，
  `echo_lru_kernel.h:819-822`），scan2 的写与 scan1 的读无同址竞争；跨行复用
  （行 r+2 复用行 r 的缓冲）则隔着行 r+1 的 B1/B1 两次栅栏，同样读先于写。

竞争对被"下一轮的 B1"天然隔开，B2 是纯冗余。这正是"1024 线程栅栏一次数千周期"
（§7）背景下的必要抠法——OpMissCumsum 4 元素路径每行栅栏 4→2、OpHitMiss 归约
2→1、Argtopk emission 2→1（§9.11）。

### 4.4 warp 数按 blockDim.x 运行时推导

`warpCount = blockDim.x >> 5`（`echo_lru_kernel.h:56`）而不是编译期常量，原因
（`echo_lru_kernel.h:43`）：同一份源码要在多处跑——设备 blockDim 由 plugin 自适应
传入（§3.2：生产 topk=2048 → 512 → 16 warp；上限 1024 → 32 warp；消融 knob 压到
768/256 → 24/8 warp）、CPU 回归 `checks/check_echo_kernel_host.py` 模拟
blockDim=128（4 warp）。运行时推导使线程数变化（自适应公式或 knob 调整）或 CPU
模型变化都无需改扫描布局（与 SIMT 基线"warp 数完全
由 PLAN_THREADS 推导，后续调整线程数无需重写 scan 布局"同理，
`sparse_kv_plan_kernel.cpp:102`）。

> 顺带指出：README §9 提到的 warp 协作辅助 `BlockSumToGm` 已不存在于当前 kernel——
> P0 把块求和折叠进了 `OpHitMiss` 的 warp 归约段。README 该句是陈旧描述（本文 §11
> 之外的已知文档债，见文末"通读发现的不一致"）。

---

## 5. 流水线逐步解读（20 个非 spec 步）

### 5.1 总表

`Process()`（`echo_lru_kernel.h:1085-1191`）内非 spec 路径共 20 个打点步，步名与
`echo_perf_step.py` 的 `ECHO_PERF_STEP_NAMES`（`echo_perf_step.py:50-58`）一一对应。
打点由 `PerfBegin/PerfEnd` 包裹（t0 的 `clock()` 差值写入 workspace 尾部，
`echo_lru_kernel.h:1216-1238`）。

| # | 步名 | 函数 | 做什么（读 → 写） | 行内跨线程 RAW / 栅栏 |
|---|---|---|---|---|
| 0 | （无打点） | OpFillZero + SyncGrid | t0 清 pushCount（spec 用）；**非 spec 时整段跳过**（[白付清理 A]，§9.15）；prologue 不在 20 步内 | — |
| 1 | Step0.OpResetRows | OpResetRows | resetMask=1 行：htd 整行 `col==0?0:I32_MAX`；anyReset=0 时跳过（打点槽保留，[白付清理 B]） | 各线程写各自列，无 RAW |
| 2 | Step0.OpResetDthPri | OpResetDthPri | dth=I32_MAX；pri=`col==0?I32_MAX:-1`；同上跳过 | 同上 |
| 3 | Step0.OpResetFree | OpResetFree | free[col<B]=B−col，其余 0；同上跳过 | 同上 |
| 4 | Step0.OpResetAvailFifo | OpResetAvailFifo | t0：avail=B，fifo=1；同上跳过 | t0-only；可见性由步 5 兜底 |
| 5 | Step0.SyncGrid | SyncGrid | —；anyReset=0 时跳过 | reset 写（含 t0 的 avail/fifo）→ Step1 读 htd（全员）/ avail（t0）|
| 6 | Step1.OpHitMiss | OpHitMiss | pos/htd → curSlots；hit 槽顺手写 pri=fifo（protect 折叠）；miss 计数 warp 归约；t0 写 numMisses/freeSize/C1 标志 | 1 栅栏/行（行奇偶双缓冲，[Scan-v2]） |
| 7 | Step1.SyncGrid | SyncGrid | — | curSlots/pri 写 → Step2/3/5 读；numMisses/freeSize（t0 GM 写）经 dcci 对全员可见 |
| 8 | Step2.ArgtopkAllRows | [C1] needFree=0 时**跳过** | radix select：pri/fifo → freeIdx（UB 或 GM，[P-C]） | 每行内部自带栅栏（§6） |
| 9 | Step2.SyncGrid | [C1] 同上跳过；**UB 路径整段跳过**（round-4b：末行尾栅栏 1 已提供排序）；GM 路径保留完整 SyncGrid | — | freeIdx 写 → OpFreeRelease 读（UB：Argtopk 末行尾栅栏 1；GM：本槽 SyncGrid） |
| 10 | Step2.OpFreeRelease | [C1] 同上跳过 | freeIdx/dth → pri=−1、dth=I32_MAX、htd[tok]=I32_MAX | j<fs 才执行 |
| 11 | Step2.OpFreePush | [C1] 同上跳过 | freeIdx → free[avail+j] | 本 kernel 内**无读者**（P4） |
| 12 | Step3.OpAllocPop | OpAllocPop | numMisses/freeSize/avail/freeIdx/free → rawSlots | 读 freeIdx（步 9 后）、numMisses/freeSize（步 7 后）、avail（步 5 后） |
| 13 | Step3.OpFifoInc | OpFifoInc | t0：fifo 第一次自增（f→f+1，protect 折叠后自旧 Step2 挪来） | t0-only；Argtopk（步 8）恒读步 7 dcci 后的旧值 f |
| 14 | Step3.SyncGrid | SyncGrid | — | rawSlots 写 → Step4 读；fifo（t0 写）→ Step4 读 |
| 15 | Step4.OpAllocPriority | OpAllocPriority | rawSlots/fifo → pri[aslot]=fifo（f+1） | rawSlots/fifo 读在步 14 后；pri 写，本 kernel 内此后无读者 |
| 16 | Step4.SyncBlock | 裸 syncthreads | — | **Bug2**：全员 fifo 读完成先于 t0 fifo++；排空 avail 全员读者 |
| 17 | Step4.OpFifoAvailUpdate | OpFifoAvailUpdate | t0：fifo 第二次自增（f+1→f+2）；t1：avail += fs − nm（[FifoAvail 拆分]） | t0/t1 分写不同地址；尾 SyncGrid 已删（P4） |
| 18 | Step5.OpMissCumsum | OpMissCumsum | curSlots/pos → wsMissRank/missHostPos | 2/4 元素路径 PackedBlockScan 各 1/2 栅栏/行（[Scan-v2]） |
| 19 | Step5.SyncGrid | SyncGrid | — | wsMissRank/missPos 写 → OpSetUpdate 读 |
| 20 | Step5.OpSetUpdate | OpSetUpdate | wsMissRank/rawSlots/missHostPos → htd/dth/curSlots/flat/mask | 行内 pos 不重复契约保证写地址不冲突 |

### 5.2 每步要点

**Step0（reset 四连）**：只对 `resetMaskGm[row]!=0` 的行执行；行内 blockDim.x 线程
（运行时自适应，§3.2；生产 topk=2048 → 512）grid-stride 扫列（`echo_lru_kernel.h:85-150`）。htd 行长 `maxModelLenP1`=262144 int32=1 MiB，cold
场景 32 行全 reset → 每核 1 MiB 纯写，是 cold 场景的带宽地板项（§10）。步 5 的 SyncGrid
同时解决"全员写 htd → 全员读 htd"与"t0 写 avail/fifo → t0 读 avail"两类依赖。
**[白付清理 B]** 本核行段无 reset 时整段跳过（每线程各自读本核行段 resetMask 算出
逐线程一致的 anyReset，`echo_lru_kernel.h:1105-1117`）：四个 reset Op 对该核行段本来
就全部 continue 零写，步 5 SyncGrid 无 RAW 可保护；Step1 读的 htd/avail/fifo 是跨调用
持久态，kernel 边界保证可见（§9.15）。打点槽原位保留（跳过记录 ≈0 周期，同 C1 做法）。

**Step1：hit/miss 判定 + protect 折叠 + warp 归约计数 + fs 折叠 + C1 跳过标志**
（`OpHitMiss`，`echo_lru_kernel.h:257-315`）：

```
每线程: for j = t; j < topkDim; j += blockDim.x
            slot = htd[row, clamp(pos[j])]; curSlots[j] = slot
            slot == I32_MAX → localMiss++                     // j≥effTopk: curSlots[j]=0
            否则 (hit): pri[row, clamp(slot,0,B)] = fifo       // [protect 折叠] 本步初值 f
warp 内:  warpSum = asc_reduce_add(localMiss)                 // 零栅栏
lane0:    warpScratch[(r&1)*32 + warp] = warpSum              // UB 行奇偶双缓冲 ([Scan-v2])
── 栅栏（每行 1 次）──────────────────────────────────
t0:       nm = Σ warpScratch[(r&1)*32 + 0..warpCount)
          numMisses[row] = nm
          fs = max(nm - availGm[row], 0);  freeSize[row] = fs   // [P4] 原 OpFreeSize 折进来
          fs != 0 → fsAny = 1
行循环结束后: t0: warpScratch[64] = fsAny                      // [C1] 淘汰标志, UB 槽 64
```

五个设计点：

1. **[P0] 去 atomic**：atomic 版本每线程一次 `asc_atomic_add` 打同一 GM 地址，warm 场景
   ~1024 次/行 × 同地址 atomic 串行 ~9K 周期 → 该步实测 ~9.4M 周期（根因证据：cold 无
   atomic 仅 ~10K，hit 数 1536 vs 2048 耗时几乎相同 → 耗时与 atomic 次数强相关、与 hit
   数无关）。改 warp 归约 + t0 汇总后一次 GM 写搞定（`echo_lru_kernel.h:245-248`）。
     `numHitsGm` 槽位保留但不再读写（防 workspace 布局漂移，`echo_lru_kernel.h:254,1018`）。
2. **[P4] fs 折叠**：`fs = nm - avail`（clamp≥0）在 t0 汇总段顺手写出，省掉独立的
   OpFreeSize 步及其 SyncGrid（`echo_lru_kernel.h:249-253,305-307`）。`nm` 直接数
   `curSlots==I32_MAX` 的个数，与公式 `effTopk - numHits` 恒等。
3. **[protect 折叠]**：hit 分支 slot 就在手边，顺手写 `pri[row, clamp(slot,0,B)] = fifo`
   （本步初值 f，`echo_lru_kernel.h:276,287-292`）——免去独立步骤重读 curSlots 的
   O(topk) GM 往返与其前后 3 个栅栏。写冲突安全性：一行内 pos 不重复（§5 Step5 契约）
   → distinct token 经 htd 直查得 distinct slot（htd/dth 互逆），各线程写的 pri 地址
   互不相同；pri 行内写 → Argtopk 行内读的 RAW 由步 7 的 SyncGrid（dcci）保证。
4. **[Scan-v2] 每行 1 栅栏（行奇偶双缓冲）**：warpScratch 按行奇偶双缓冲，行 r+2 复用
   buf(r&1)，中间隔着行 r+1 的栅栏 → t0 读先于下一轮 lane0 写，无需每行第二个
   "复用保护"栅栏（`echo_lru_kernel.h:294-300`）。t0 的 GM 写可见性由步 7 的 SyncGrid
   兜底。
5. **[C1] 标志**：`warpScratch[64]`（UB）写"本核任一行 fs>0"，经步 7 的 SyncGrid 对全
   块可见；槽 64 与双缓冲占用的 [0,64) 不冲突（`echo_lru_kernel.h:311-314`）。

**fifo 的两次自增与 Argtopk 的一致旧值（[OpFifoInc 挪位]，机理）**：fifo 每轮自增两次，
语义逐值不变——protect 盖 f（Step1 读初值）、alloc 盖 f+1（Step4 OpAllocPriority）、
终值 f+2（Step4 尾第二次自增），与 `reference_echo` 的两次 `fifo += 1`
（`compare.py:187,203`）一致。第一次自增（OpFifoInc）位于 Step3 的 OpAllocPop **之后**
（`echo_lru_kernel.h:1154-1158`，函数头注释 `:318-324`），这是关键的设计修正：

- **若把 fifo++ 留在 protect 之后并删除其后 SyncGrid**：t0 率先执行 OpFifoInc，其 GM 写
  滞留在 t0 自己的 D-cache——t0 后续读 fifo 命中**新值**，其他线程读 GM 拿到**旧值**。
  ArgtopkAllRows 里每线程都要读一次 `fifoGm[row]` 算 startPass（[方向1] 跳趟启发式），
  读到不一致的 fifo → 各线程 startPass 分歧 → 同一行内线程跑不同趟数 → **栅栏发散**
  （部分线程停在 pass 2 的 B1，其余已进入 pass 3），kernel 卡死。
- **挪后的正确性**：Argtopk（步 8）恒读步 7 SyncGrid dcci 之后的**一致旧值 f**——此时
  全部 pri ≤ f 仍成立（protect 盖的就是 f），跳趟启发式的安全前提不变；fifo 的下一个
  读者是 Step4 的 OpAllocPriority（读 f+1），其可见性由步 14 的 SyncGrid 保证。

**Step2：C1 条件执行**（`echo_lru_kernel.h:1150-1166`）：

```cpp
bool needFree = (argtopkKeyUb_[64] != 0);   // 步 7 之后读, 块统一条件
PerfBegin(); if (needFree) ArgtopkAllRows();                       PerfEnd();
PerfBegin(); if (needFree && !freeIdxInUb_) SyncGrid();            PerfEnd();  // [P-C+尾栅栏去重]
PerfBegin(); if (needFree) OpFreeRelease(...);                     PerfEnd();
PerfBegin(); if (needFree) OpFreePush(...);                        PerfEnd();
```

全 0（本核所有行 fs=0）时四步整体跳过，**步数/步名不变（打点记录 ≈0 周期）**。安全性
论证（`echo_lru_kernel.h:1151-1155`）：fs=0 时 OpArgtopkRow 本来就走 fsClamp==0 提前
返回；OpFreeRelease/OpFreePush 的 `j<fs` 循环体不执行；**freeIdx 无任何读者**——fs=0
时 OpAllocPop 的 `readIdx = avail + 0 - nm + j`，由 fs=0 ⟹ nm≤avail 得 readIdx <
avail，恒走老栈路径（§5 Step3），从不读 freeIdx；avail 也不变（+=0）。条件对全块统一
（单一 UB 标志），**栅栏无发散**。warm/cold 场景由此跳过整条淘汰流水线。

第二个打点（Step2.SyncGrid 槽）自 round-4b 起 UB 路径**整段跳过**：末行 freeIdx 写 →
OpFreeRelease 读的排序已由 ArgtopkAllRows 末行函数内尾栅栏 1 提供（全员经过该栅栏才
离开 ArgtopkAllRows，§9.16）；GM 回退路径保守保留完整 SyncGrid（大 batch 形态不在
基准覆盖，不冒险）（`echo_lru_kernel.h:1158-1164`）。

Argtopk 本身见 §6。OpFreeRelease（`echo_lru_kernel.h:632-661`）对 j<fs 的每个淘汰槽：
`pri=-1, dth=I32_MAX, htd[tok]=I32_MAX`（tok≠I32_MAX 时）。OpFreePush（`:663-684`）：
`free[avail+j] = freeIdx[j]`——把新释放槽压回栈；P4 之后**本 kernel 内它没有读者**
（OpAllocPop 直读 freeIdx，见下），其写入只为持久化给下一次调用，因此 FreePush 之后
不再需要 SyncGrid。

**Step3：OpAllocPop 直读 freeIdx 的等价性推导**（`echo_lru_kernel.h:685-720`）：

栈语义序列：FreePush 写 `free[availOld+k] = freeIdx[k]`（k<fs）→ avail += fs →
avail −= nm → 弹出 `free[availOld + fs − nm + j]`（j<nm）。

avail 更新推迟到 Step4 尾，本步读到的 avail 恒为初值 availOld，弹出索引
`readIdx = availOld + fs − nm + j` 拆两段：

- **readIdx ≥ availOld**：落在新压入段。栈语义读 `free[availOld + k]`（k=readIdx−availOld
  ∈ [0, fs)），而 FreePush 恰好写的是 `free[availOld+k] = freeIdx[k]` ⟹
  **`free[availOld+k] == freeIdx[k]`**，直接读 `freeIdx[k]` 逐值等价，且**不依赖
  FreePush 的写入是否完成**（FreePush→AllocPop 间的 SyncGrid 因此可删）；
- **readIdx < availOld**：老栈区段 `free[readIdx]`，本 kernel 从未写该区段（FreePush
  只写 [availOld, availOld+fs)），读老栈原值即可。

两段合起来与栈弹出**逐值一致**（`echo_lru_kernel.h:676-684` 的注释即此推导；
`checks` 第 4 组用旧栈 golden 逐值验证，`check_echo_kernel_host.py:272-310`）。
`j ≥ nm` 的位置填 0（无效）。freeIdx 的落点由 [P-C] 决定（`freeIdxPtr_`，UB 或 GM），
两条路径输出逐位一致（`checks` 第 1b 组）。

步 13 的 OpFifoInc（t0，全部行 f→f+1）紧随其后，其挪位机理见 Step1 末段。

**Step4：OpAllocPriority + Bug2 栅栏 + FifoAvail 拆分**（`echo_lru_kernel.h:723-744,1161-1166`）：

OpAllocPriority 对 j<nm 的每个分配槽写 `pri[row, clamp(aslot)] = fifo`——fifo 此时已是
f+1（步 13 自增、步 14 可见）。**Bug2 栅栏**（步 16，裸 syncthreads，
`echo_lru_kernel.h:1163-1164`）：全线程在 OpAllocPriority 里的 fifo 读必须先于 t0 在
OpFifoAvailUpdate 里的第二次 fifo++，否则后到的线程会把 pri 多写 1（历史实测诊断：
row143 slot3706 kernel=12 golden=11 diff+1）；该栅栏同时排空 avail 的全员读者。裸
syncthreads 即可：竞争是"读（栅栏前）vs 写（栅栏后）"，不存在 stale 读。

**OpFifoAvailUpdate 的 avail 终值恒等式**（`echo_lru_kernel.h:339-367`）：

```
终值 avail' = availOld + fs − nm
栈语义两步序列: (availOld + fs) − nm = availOld + fs − nm        ← 恒等
非负性: fs = max(nm − availOld, 0)
   nm > availOld 时: fs = nm − availOld ⟹ avail' = 0
   nm ≤ availOld 时: fs = 0     ⟹ avail' = availOld − nm ≥ 0
```

**[FifoAvail 拆分]**：fifo++ 与 avail+=fs−nm 无数据依赖，t0/t1 各管一个
（`echo_lru_kernel.h:351-365`）——t0 的串行 GM 依赖访存从 6 次/行减半到 3 次（t1 同）。
依赖分析（`:343-345,351-353`）：t0 读的 fifo 是自己写的（同线程 GM RAW 安全）；
t1 读的 fs/nm 由 t0 在 Step1 写入、经步 7 SyncGrid dcci 可见；两写地址不同无冲突；
avail 的全员读者（OpFreePush/OpAllocPop）已被步 14 与步 16 排空。**尾 SyncGrid 已删**：
此步之后 pri/fifo/avail 在本 kernel 内再无读者，kernel 结束即对 host 可见
（`echo_lru_kernel.h:1166`）。

**Step5：两级扫描 rank 拼装与 SetUpdate 回写**：

`OpMissCumsum`（`echo_lru_kernel.h:763-910`）按门条件选三条路径之一（Bm =
blockDim.x）。每线程认领的都是**连续映射**位置（`p = t + k*Bm`，理由同 §6.6
emission 的连续切片——rank 必须按位置序拼装，grid-stride 会得到"线程主序"）：

| 路径 | 门条件 | 每线程位置 | 扫描结构 | 栅栏/行 |
|---|---|---|---|---:|
| 2 元素 | `topkDim ≤ 2*Bm`（`:774`） | p0=t, p1=t+Bm | 1 次 PackedBlockScan | 1 |
| 4 元素 | `topkDim ≤ 4*Bm`（`:811`） | p0..p3 = t, t+Bm, t+2Bm, t+3Bm | 2 次链式 PackedBlockScan | 2 |
| 串行 | 以上均不满足 | — | t0 逐位累加（`:896-909`） | 0 |

门只剩线程侧条件（UB 侧的 `2*/4*topkDim ≤ 8192` 是 Hillis-Steele 时代 bufA/bufB 缓冲
的遗留约束，warp 扫描版 scratch 与 topkDim 无关，已删，commit 8856e2a；头注释
`echo_lru_kernel.h:754-757`）。自适应 blockDim（§3.2）下的路径分布：生产 topk=2048 →
Bm=512，2 元素门失效（2048 > 2×512），由 **4 元素路径**接住（2048 ≤ 4×512）——4 元素
路径即生产路径；小 topk（≤ 128，Bm 压到 64 下限）命中 2 元素路径（topk ≤ 2×64）；
topk > 4096 时 4×1024（上界）< topk → t0 串行（超出当前生产 topk 形态）——自适应
blockDim 下 **topk ≤ 4096 恒并行**。消融 knob 把上限压到 256 时（4×256=1024 < 2048）
回退串行——t256 消融崩塌的根因（§9.10）。

**2 元素路径的 rank 拼装**（`echo_lru_kernel.h:777-809`）：

```
e0(t) = Σ_{t'<t} ind0(t') = 位置 [0, t) 的 miss 数            → rank(p0=t)   = e0
e1(t) = Σ_{t'<t} ind1(t') = 位置 [Bm, Bm+t) 的 miss 数
total0 = (Σ warpTotals) & 0xFFF = 前半段 miss 总数              // [Scan-v2] 本地 Σ, 无 control 槽
rank(p1 = t+Bm) = total0 + e1(t)                          ← 后半段的排名 = 前半段总数 + 后半段内部前缀
```

**4 元素路径的 rank 拼装**（`echo_lru_kernel.h:811-893`）：位置序切成四个
tile——[0,Bm) 由各线程 p0 覆盖、[Bm,2Bm) 由 p1、[2Bm,3Bm) 由 p2、[3Bm,4Bm) 由 p3。
两次链式扫描（12bit 字段安全：各 tile 的 miss 总数 ≤ topkDim ≤ 2048 < 4096，
`echo_lru_kernel.h:816-817`）：`excl1 = scan(ind0 | ind1<<12)` 覆盖 [0,2Bm)、
`excl2 = scan(ind2 | ind3<<12)` 覆盖 [2Bm,4Bm)（`:870-871`）。**[Scan-v2] 四缓冲**：
warpTotals 按 (行奇偶 × scan 序) 轮转——行 r 的 scan1 用 `scanUb+(r&1)*64`、scan2 用
`+32`（`:868-869`）；背靠背两次扫描间**不需要栅栏**（scan1/scan2 写不相交缓冲；行 r+2
复用本行缓冲时隔着行 r+1 的两次栅栏，论证见 §4.3(b)）；t0/t01/t2 由各线程返回后本地
Σ 缓冲得出（`:874-878`），无 control 槽。rank 拼装（`:880-891`）：

```
t0  = grand1 & 0xFFF             = [0,Bm) 的 miss 总数
t01 = t0 + (grand1 >> 12)        = [0,2Bm) 的 miss 总数
t2  = grand2 & 0xFFF             = [2Bm,3Bm) 的 miss 总数
rank(p0=t)     = e0
rank(p1=t+Bm)  = t0  + e1       ← ⚠ 不能写 t01+e1：p1 在 tile1 段内而非段后，
rank(p2=t+2Bm) = t01 + e2          多加 tile1 自身总数会让 rank 偏大（`:832-833`）
rank(p3=t+3Bm) = t01 + t2 + e3
```

（rank(p1) 的这个陷阱曾在实现规格里以 t01+e1 的形式出现，被 CPU 回归 golden 拦下，
commit d20fc50。）串行回退与两条并行路径语义逐字段一致（`checks` 第 3 组三路同
golden）。scan 缓冲复用 argtopkKeyUb_（radix select 早已结束）：2 元素用行奇偶双缓冲
[0,64)、4 元素用 (行奇偶×scan序) 四缓冲 [0,128)（§3.3）。

`OpSetUpdate`（`echo_lru_kernel.h:912-957`）对每个 miss 位 j：
`rank = wsMissRank[j]` → `aslot = rawSlots[rank]` → `htd[row, mpos] = aslot`、
`dth[row, aslot] = mpos`、`missAllocFlat = row*B + aslot`、`missMask = 1`。**[P2] 顺手
回写 `curSlots[j] = aslot`**：miss 的最终槽位就是本轮分配的 aslot，hit 位 curSlots 在
Step1 已正确，独立的全量 gather 步（OpFinalSlots）是纯冗余，[P2] 已删除
（`echo_lru_kernel.h:947-950`）。并行写 htd/dth 的地址不冲突由**输入契约**保证：同一行
内 pos 不重复（一行 = 一个 request，topk 选出的 token id 必不重复，
`echo_lru_kernel.h:921-924`）。

**spec 路径（Step0.5，本基准不覆盖）**：`specEnabled=1` 时在 Step0.SyncGrid 之后插入
8 步（4 个 Op + 4 个栅栏）：`OpInvalidateSpecSuffix`（驱逐 `tok ≥ spl` 的投机后缀槽，
写 wsCumsum 标记）→ `OpSpecCumsum`（t0 前缀和 + pushCount）→ `OpSpecPushFree`
（差分判定 `cum[slot]−cum[slot−1]==1` 找回真 spec 槽压栈——**不能用 `cum>0`**，否则
末段非 spec 槽会误判并发覆盖，见 `echo_lru_kernel.h:213-217` 的实战记录）→
`OpSpecAvailAdd`。

---

## 6. Argtopk 基数选择深读

本章是 Echo 的性能核心。`OpArgtopkRow`（`echo_lru_kernel.h:403-618`，头注释
`:372-402`）解决：**从 B+1 个打包 key 中选出 pri 最小的 fsClamp 个槽，平局按槽号
升序，写入 freeIdxOut**。

### 6.1 目标与 key 打包

```
key[i] = (i==0 || i>B || pri[i]<0) ? I32_MAX : pri[i]        // i ∈ [0, validN=B+1)
```

全非负、无效/填充沉底为 I32_MAX（`echo_lru_kernel.h:374,438`）。选择目标等价于
稳定排序 `(key, slot)` 字典序取前 fsClamp 个（`checks` 用 stable_sort 交叉验证，
`check_echo_kernel_host.py:154-161`）。

**key 数组只写 [0, B+1)**：[P-E] 后 key 的写入集中在行首独立的全员 pack 相位
（`for i = t; i < validN; i += blockDim.x`，`echo_lru_kernel.h:436-439`），选择循环内
不再写 key。padding
区 [B+1, Npad) 的 Npad−(B+1) 个键（恒为 I32_MAX）**从不物化**——它们只在两处被
"解析式"消费：merge 相位计入直方图（下文）、emission 越界时按序补索引——相比写满
8192 个键的满扫描省掉约一半打包写入。CPU 回归显式断言 padding 区未被触碰
（`check_echo_kernel_host.py:167`）。

### 6.2 每行结构：行首 [pack（全员）] → B0；每趟 [hist] → B1 → [merge（破坏性）] → B2 → [本地决策]

4 趟 MSB-first（pass 0 看 byte3，pass 3 看 byte0，`shift = (3-pass)*8`）。[P-E 反向 P-D]
后的相位图——pack + B0 **每行一次**（位于 fsClamp==0 早退之后、attempt 循环之前，
`echo_lru_kernel.h:429-440`），其余为每趟结构：

```
┌──── (0) [P-E] pack ─────┐   ┌──────────── 相位 (a) hist ─────────┐   ┌─ (b) merge ─┐   ┌─ (c) 本地决策 ─┐
│ 全块 blockDim 线程读    │   │ G 路读 key[] 统计本字节直方图       │   │ 256 桶并行: │   │ 全员各自重算    │
│ priGm 写 key[] (每线程  │ B0│ 首迭代: 先清零各自 hist[256] 切片;   │ B1│ 每桶归并 G  │B2│ b/prefix/      │
│ ≤ceil((B+1)/blockDim.x)│ → │ 后续趟: 不 clear —— 上趟 merge 已    │ → │ 个切片, 读后│ →│ remaining      │
│ 次, 完全协同、高 MLP;   │   │ 破坏性清零; 带 prefix 过滤          │   │ 清零 ([P-B])│   │ (寄存器 T/quota)│
│ 每行一次)               │   │                                    │   │ +padding    │   │                 │
│                         │   │                                    │   │ 解析式计入  │   │                 │
└─────────────────────────┘   └────────────────────────────────────┘   └─────────────┘   └─────────────────┘
```

- **[P-E] 行首全员 pack + B0**（`echo_lru_kernel.h:429-440`）：全块 blockDim 线程读
  priGm 打包写 `key[i]`（每线程 ≤ceil(validN/blockDim.x) 次、完全协同、高 MLP），B0
  栅栏隔开 "pack 写 key（全员）→ hist 读 key（G 路）" 的跨线程 RAW。此后 hist 首迭代
  与后续趟**一律读 key[]（UB）**，pass1 起的 GM 依赖读链整体消除（动机与分步数据见
  §9.15）。可见性：priGm 的上一个写者是 Step1 OpHitMiss（protect），由 Step1 尾
  SyncGrid 保证；上一行 emission 对同一 key UB 区的读已由其尾部两栅栏排空（行间复用，
  §9.13 教训：key 无行索引）。pack 必须放在 fsClamp==0 早退之后（checks 断言早退时
  输出/UB/dbg 全不写）。
- **首迭代：clear 仅首次**（`:473-474`）：用途是覆盖 kernel 启动时 UB 的初始垃圾；首趟若
  `startPass>0`，hist 带 `prefix=0` 候选过滤：`(v >> (shift+8)) == 0`
  （key 的高于本字节的全零）；I32_MAX 键 byte3=0x7F≠0 被天然排除（`:475-476,479-483`）。
- **后续趟：不 clear**（[P-B]）：上一趟 merge 已把全部 G×256 槽破坏性清零（见 (b)），
  hist 直接增量；过滤条件改为高字节匹配当前 `prefix`
  （`(v >> (shift+8)) == (prefix >> (shift+8))`，`echo_lru_kernel.h:490`）。回退重跑
  （§6.5 的 attempt=1）同样不 clear：快路径失败的 break 发生在 merge **之后**的决策段，
  彼时 hist 已被该趟 merge 清零 → 回退起跑时恒为全零。
- **G=64 路纯写切片**：`ARGTOPK_GROUP_THREADS` 个线程各写自己独占的 `hist[t*256..]`
  切片，**无 atomic、无争用**（tiling.h:40-41）。

**B0/B1/B2 保护的读写对**（跨趟/跨行复用的安全性）：

```
hist 切片:  pass p 的 merge 读+清零 (B1~B2 之间) vs  pass p+1 的 hist 增量写 (B2 之后)  → B2 分隔 ✓
globalCnt:  pass p 的决策读 (B2 之后)         vs  pass p+1 的 merge 写 (B1(p+1) 之后)  → B1(p+1) 分隔 ✓
key:        [P-E] pack 写 (B0 之前, 全员)     vs  各趟 hist 读 (G 路, B0 之后)          → B0 分隔 ✓
            上一行 emission 读 key            vs  本行 pack 写 (同一 UB 区复用)         → emission 尾栅栏 1 排空 ✓ (GM 尾栅栏 2 仅服务 dcci, 排序上栅栏 1 已足够)
```

### 6.3 merge：256 桶并行 + 破坏性清零 + padding 解析式计入

```cpp
uint32_t padByte = (0x7FFFFFFFu >> shift) & 0xFFu;       // I32_MAX 在本趟字节的桶号
bool padIn = (pass == 0u) ? true                          // pass 0 无前缀 (x>>32 文本 UB, 分支规避)
                          : ((0x7FFFFFFFu >> (shift+8u)) == (prefix >> (shift+8u)));
for (uint32_t b = t; b < 256; b += blockDim.x) {          // 256 桶摊给全块线程 (blockDim.x 自适应)
    int32_t s = 0;
    for (uint32_t th = 0; th < G; ++th) {
        int32_t* slot = &hist[th*256u + b];
        s += *slot; *slot = 0;                            // [P-B] 破坏性归并: 读后清零
    }
    if (padIn && b == padByte) s += (int32_t)padN;        // padding 键解析式计入
    globalCnt[b] = s;
}
```

（`echo_lru_kernel.h:497-518`。）

- **256 桶并行**：t0 串行扫 G×256=16384 项的版本改为全块线程各领 ≤1 桶（循环步长
  blockDim.x，≥256 线程时每线程至多一桶）、每桶归并 G=64 个切片，**墙钟只有 G 次读**
  （[B] 注释，`echo_lru_kernel.h:381`）。
- **[P-B] 破坏性归并**：merge 读切片后顺手清零（`s += *slot; *slot = 0`）——清零写
  与下一趟 hist 增量之间有 B2 隔开，下一趟 hist 因此不再需要 clear 相位（首迭代除外，
  见 §6.2）。
- **padIn 的语义**：padding 键全是 I32_MAX，它们"在候选集合里"当且仅当选择前缀已经
  走到 I32_MAX 的路径上（I32_MAX = 0x7FFFFFFF，字节路径 0x7F,0xFF,0xFF,0xFF，与 prefix 逐字节相等）。
  pass 0 无前缀，全部候选（padIn=true）。计入的量 padN = Npad − (B+1) 是编译期可算
  的常数——这就是 padding 不需要物化的全部原因。
- **两个编译器坑的规避**就写在这段附近（§7.6）：pass==0 分支不用 `||` 短路（文本 UB
  `v>>32` 会被 bisheng 投机求值），改 if/else。

### 6.4 决策本地化（方向 2）：为什么能去掉第 3 次栅栏

决策不经过任何广播（`echo_lru_kernel.h:534-546`）：**桶决策是 globalCnt 的确定性函数**
——给定相同的 globalCnt 快照与相同的 remaining，任何线程执行同一段循环必然得到相同的
b/prefix/remaining/T/quota。于是让**每个线程各自重算**（本数据 break 出现在前几个桶，
每线程 ≤4 次 UB 读，`echo_lru_kernel.h:391-392`），T/quota 留在**寄存器**里直接给
emission 用。没有跨线程写 ⟹ 没有栅栏（对照：t0 串行决策 → 把 T/quota 写 scalar（UB）
→ 第 3 次栅栏广播 → 全员读 scalar 的结构需要那次栅栏）。`scalar` 从此无读者（槽位保留
防布局漂移）。

决策逻辑与不变量：

```cpp
uint32_t cum = 0, b = 0;
for (; b < 256; ++b) {
    uint32_t c = globalCnt[b];
    if (cum + c >= remaining) break;      // 第 remaining 个候选落在桶 b
    cum += c;
}
if (b >= 256) break;                       // 快路径放不下 → 回退 (§6.5)
prefix |= (b << shift);                    // T 的本字节定为 b
remaining -= cum;                          // 桶内还差 remaining' 个
T = prefix;  quota = remaining;
```

**不变量**（`echo_lru_kernel.h:398-399`）：`|{key < T}| + quota == fsClamp`（计数含
padding）。归纳：每趟结束时 remaining' = fsClamp − |{高字节前缀 < prefix 的键}|，
4 趟走完 T 即 fsClamp-th 小的键，quota = fsClamp − |{key<T}|。又 **remaining' ≥ 1 恒
成立**：b-loop 在 `cum + c ≥ remaining` 时 break 且 `cum < remaining`（`:544` 注释），
这保证下一趟必有候选、决策不会空转。

### 6.5 自适应跳趟（方向 1）：startPass、候选过滤与回退

```cpp
uint32_t startPass = 3u;
if      ((fifoVal >> 24) != 0u) startPass = 0u;   // fifo 高字节非零 → 全趟
else if ((fifoVal >> 16) != 0u) startPass = 1u;   // 前导零字节 z=1 → 跳 pass 0
else if ((fifoVal >> 8 ) != 0u) startPass = 2u;   // z=2 → 跳 pass 0..1
                                                    // z=3 → 只跑 byte0
```

（`echo_lru_kernel.h:445-448`。）

**依据**：pri 的唯一写入来源是 fifo（OpHitMiss 的 hit 分支 / OpAllocPriority 写
`pri=fifo`），fifo 单调递增，故**当前所有有效 pri ≤ fifoVal**（`echo_lru_kernel.h:385`）。
Argtopk 读到的 fifoVal 恒为本轮初值 f（第一次自增在 Step3 OpAllocPop 之后，§5 Step1
末段），故 pri ≤ f 前提稳定成立。若 fifoVal
有 z 个前导零字节，则所有有效 pri 的前 z 个字节全零——**pass 0..z−1 对这些键的直方图
全落在桶 0，prefix 不变、不产生任何信息**，直接从 byte(3−z) 起跑。基准 fifo=4097
(0x1001) → z=2 → 只跑 byte1/byte0 两趟。

具体到 full_cache_75pct：初始 `pri[slot]=slot`（§1.5），hit 槽被 Step1 的 protect 分支
盖为 4097，非 hit 占用槽保持 1537..4096；fs=512 → 选 1537..2048。这些键 byte1 ∈ 0x06..0x08，
两趟内即可收敛（byte1 定桶 → byte0 定位）。

**prefix=0 候选过滤**：起跑趟 >0 时首趟直方图只统计 `(v >> (shift+8)) == 0` 的键
（高字节全零的小键）。I32_MAX 键（无效槽/padding，byte3=0x7F）被天然排除——它们本来
就"沉底"，只有当选择需要深入到 I32_MAX 值域时才相关。

**放不下 → 回退全趟**：若 fsClamp 超出 prefix=0 候选总数（b 扫到 256 仍不满足），说明
存在大量"高字节非零"的键（比如 pri 远大于 fifo 的异常状态），快路径失效。此时
`pass=0, remaining=fsClamp, prefix=0` 重来一次完整 4 趟（key 已打包，不重 pack、不重
clear——失败趟的 merge 已破坏性清零，§6.2）。
全趟候选 = 全部 Npad 个键 ≥ fsClamp（fsClamp ≤ topkDim ≤ Npad），**必然收敛**
（`echo_lru_kernel.h:549-555`）。

**健壮性不依赖 pri ≤ fifo**：跳趟只是启发式；启发式失效（pri>fifo 的键真实存在时）
的表现是"快路径放不下 → 回退"，**任何输入下输出与全趟版逐位一致**——失效只损失
性能，不损失正确性（`echo_lru_kernel.h:388-390`，commit 145d3ce 信息）。CPU 回归用
4 个 fifo 变体（0x40000000 全趟 / 4097 跳2趟 / 65536 跳1趟 / 100 只跑 byte0）×
全部 B/pattern 组合验证跳趟与回退路径输出 golden 与全趟逐位一致，含全 I32_MAX 键等
强制回退用例（`check_echo_kernel_host.py:99-105,171-178`）。

**外层 attempt 循环的写法**：`for (attempt = 0; attempt < 2; ++attempt)` + 迭代边界
break，**刻意不用 `while(true)` + 数据依赖 continue**——后者会跨栅栏结构段错误（§7.6），
   前者与旧版 scalar[3] break 同构、已验证可编译（`echo_lru_kernel.h:456-460`）。

### 6.6 emission：连续切片不变量与 padding 补齐

选出 T/quota 后，全员把选择结果写出去（`echo_lru_kernel.h:559-613`）：

```
1) 每线程统计自己连续切片 [lo, hi) 内:  nLt (key<T 个数) | nEq (key==T 个数) 打包
2) PackedBlockScan(nLt | nEq<<16) → excl=(wLt,wEq) 本线程之前所有切片的累计;
   totalLt/totalEq 由各线程本地 Σ warpTotals 得到                ← 1 次栅栏 ([Scan-v2])
3) 重扫切片写 freeIdxOut:
      key < T                    → freeIdxOut[wLt++]  = i          (槽号升序段)
      key == T 且 wEq < quota    → freeIdxOut[totalLt + wEq++] = i (平局段, 取前 quota 个)
4) padding 补齐: m = fsClamp − totalLt − min(quota, totalEq)
      freeIdxOut[used + k] = validN + k   (k = t; k < m; k += blockDim.x)   → 索引 [B+1, B+1+m)
5) 尾部补零: freeIdxOut[fsClamp .. topkDim) = 0
6) 尾栅栏 1 (syncthreads); GM 路径: t0 dcci(freeIdxOut) + 尾栅栏 2   ← [P-C]: UB 免 dcci; round-4b: UB 再免尾栅栏 2
```

**连续切片不变量（为什么不能 grid-stride）**（`echo_lru_kernel.h:560-563`）：切片必须
是 `lo = t*chunk` 的**连续区间**。若用 grid-stride（线程 t 拿 {t, t+blockDim.x, …}），扫描
给出的写入基址按"线程主序"分配，各线程写出的键顺序会变成线程交错序——freeIdx 的
输出顺序就不再是全局槽号升序，与原语义（稳定排序的 (pri,slot) 序）逐位不一致。连续
切片下：切片内按 i 升序、切片间按线程号升序 = 全局槽号升序。

**输出语义**：先全部 key<T（槽号升序），再前 quota 个 key==T（槽号升序），越过有效区
时补 padding 索引，尾部补 0。**key==T 平局取最小槽号**，与稳定排序 (pri, slot) 字典序
一致（`echo_lru_kernel.h:400-402`）。

**padding 补齐何时触发**：`m > 0` ⟺ quota > totalEq（有效区 ==T 键不够配额）。由选择
不变量（计数含 padding）保证 `0 ≤ m ≤ padN`。基准形状下 `topkDim ≤ B < validN`，有效
区键数总够，m 恒为 0；该路径为接口一般性（topkDim > B 的合法输入）保留，与满扫描
版本逐位一致（`echo_lru_kernel.h:564-566,598-604`）。

### 6.7 栅栏计数演变（基准路径，每行口径）

| 版本 | 每趟栅栏 | 趟数 | emission | 合计 |
|---|---:|---:|---:|---:|
| 手抄版（4-pass LSD 排序） | 4 | 4 | — | 16+ |
| [B] 重构后 | 3 | 4 | 4 | 16 |
| 方向 1+2（145d3ce，fifo=4097 跳 2 趟） | 2 | 2 | 2（扫描）+2（尾） | **8** |
| Scan-v2 + P-C（f8b9c69） | 2 | 2 | 1（扫描）+2（尾） | **7** |
| [P-E] 反向 P-D（round-4） | 2 + 行首 B0×1 | 2 | 1（扫描）+2（尾） | **8**（= B0 1 + 2 趟×2 + 扫描 1 + 尾 2） |
| round-4b 尾栅栏去重（05046f2，UB 路径） | 2 + 行首 B0×1 | 2 | 1（扫描）+1（尾） | **7**（= B0 1 + 2 趟×2 + 扫描 1 + 尾 1；GM 路径尾 2 → 8） |

（[Scan-v2] 后 emission 扫描 2→1，合计 2×2+1+2=7；[P-C] UB 路径再免掉 emission 尾的
dcci。[P-E] 加回行首 B0（pack 写 key → hist 读 key 的 RAW 栅栏），合计回到 8。round-4b
再删 UB 路径的函数尾栅栏 2（GM 路径保留），相对 [P-E] 行构成变化：B0 已有、−尾栅栏 2
→ UB 7 / GM 8；头注释已同步更新为 "B0 1 + 基准 2 趟 × 2 栅栏 + emission 扫描 1 +
尾栅栏 1 = 7 次栅栏/行（UB 路径；函数尾栅栏 2 仅 GM 路径保留 → GM 路径 8 次/行）"
（`echo_lru_kernel.h:395-399`）——文末"通读发现的不一致"⑥ 随 round-4 解决。）

---

## 7. 硬件特性与避坑清单

每条给出代码内证据或实测教训。这些是 dav-3510（Ascend 950，`--npu-arch=dav-3510`，
CMakeLists.txt:131）纯 SIMT 后端的经验总结，改动相关代码前先过一遍本清单。

### 7.1 同地址 GM atomic：~9K 周期/次，串行化严重

- **现象**：P0 前的 OpHitMiss 每线程一次 `asc_atomic_add` 打同一 GM 地址，warm 场景
  ~1024 次/行，该步实测 ~9.4M 周期（占绝对大头）。
- **证据**：cold 场景 0 hit 无 atomic 仅 ~10K 周期；hit 数 1536（warm75）与 2048
（warm_all）两场景耗时几乎相同 → 耗时与 atomic 次数强相关、与有效工作量无关
（`echo_lru_kernel.h:245-248`）。
- **规避**：块内计数一律走 warp 归约/扫描（`asc_reduce_add`/`asc_shfl_up`）+ t0 汇总
  单写。CPU 回归甚至断言 `atomicCalls == 0`（`check_echo_kernel_host.py:244`）。

### 7.2 GM 写后读需要 dcci，否则 stale

- **现象**：单核内线程 A 写 GM、栅栏后线程 B 读同地址，可能命中 stale D-cache 行
  （hybrid 版 "stale pri" 根因，`echo_lru_tiling.h:7-8`）。
- **规避**：步间统一走 `SyncGrid()` 三明治（syncthreads + t0 `asc_dcci_entire` +
  syncthreads，§3.5）。注意 dcci_entire 整核生效，t0 做一次足够。

### 7.3 块内栅栏：数千周期/次（随线程数伸缩），栅栏本身是大成本

- **实测**：`asc_syncthreads`（1024 线程块）约 4~8K 周期（kernel 注释记 "~6-8K 周期，
  fs=0 早退路径实测"，`echo_lru_kernel.h:41`）。
- **教训**：能 warp 内解决的不上块栅栏（P0/P3）；每个栅栏要么给出保留论证（依赖
  分析），要么删（P4 删 3 个无 RAW SyncGrid、PackedBlockScan 逐版削栅栏、[Scan-v2]
  再删 B2）。这也是 Argtopk 每趟 3→2 栅栏、跳趟省整趟的动机（§6.7）。
- **随线程数伸缩**：栅栏成本与块内线程数正相关。线程数消融（§9.10）已量化
  "更便宜的栅栏 vs 更多扫列轮数"这笔账：512 线程全场景优于 1024（−9%~−18%），
  栅栏成本/到达散布的收益大于 gather 轮数翻倍的代价；256 线程因 MissCumsum
  串行回退不可比（混杂因素，§9.10）。

### 7.4 t0 串行大循环无法隐藏延迟

- **现象**：P1 前的 Argtopk emission 是 t0 串行两遍 Npad(8192) 扫描 + 串行补零，
  full_cache 该步 ~2.1M 周期；[B] 前的 merge 是 t0 串行扫 16384 项。
- **根因**：单线程的依赖 load 序列把每次 GM/UB 访问延迟全额暴露，无并行度可摊。
- **改法**：emission 改全员连续切片 + 打包块扫描（P1）；merge 改 256 桶并行（B）；
  决策本地化（方向 2）。t0 只留 ≤32 项的 warpTotals 扫描这类小循环。

### 7.5 UB 多数组别名：必须单大 `__ubuf__` 数组

- **实测**：多个独立 `__ubuf__` 数组会别名互相覆盖（读 A 得 D）；单数组可到 192 KiB
  （256 KiB 才 trap）（`echo_lru_kernel.h:1030-1031,1336-1339`）。
- **规避**：argtopk 全部状态（key/hist/globalCnt/seg/scalar/freeIdx）分段塞进单个
  `argtopkUb[32768]`，分段偏移集中在 `Init` 一处管理（§3.3）。

### 7.6 bisheng `-xasc` 编译坑三个（前端两个 145d3ce 轮 + 后端一个 v2.2 轮）

1. **短路式里的文本 UB 被投机求值**：`pass == 0 || (v >> (shift+8)) == 0` 这种写法，
   pass 0 时 `shift+8 == 32`，`v >> 32` 是文本可见的 UB——即使短路逻辑上不会执行到，
   bisheng SIMT 后端可能**投机求值**导致段错误。规避：写成 if/else 分支，让 UB 表达式
    不出现在 pass 0 的路径上（`echo_lru_kernel.h:479-483,488-492,503-507`）。
2. **`while(true)` + 数据依赖 continue 跨栅栏结构段错误**：attempt 循环刻意**不用**
   `while(true)` + continue 重跑——continue 的跳转目标取决于数据、且跨越栅栏结构，
   bisheng 前端收敛分析崩溃。规避：有界 `for (attempt < 2)` + 迭代边界 break（与旧版
    scalar[3] break 同构，已验证可编译）（`echo_lru_kernel.h:456-460`；commit 145d3ce）。
3. **后端指令选择崩溃：i32 加载值与运行时形参的 eq/ne 比较（v2.2 轮踩到，2026-10-21）**：
   `fatal error: error in backend: Copy one register into another with a different
   width`（clang frontend exit code 70），-O1/-O2/-O3 全触发、**-O0 不触发**。实证边界
   （当日十二变体定位，§9.24 落地后记）：与**常量**比较（`v == 1`）不触发；差值后跟
   常量谓词（`d=(uint32_t)v-(uint32_t)gen; d==0/d>1/d==1`）不触发；形参 int32/uint32、
   tiling 结构体尺寸、prologue 栅栏删除均无关；**非 eq/ne 谓词与形参比较不触发**
   （v2.1 的 `tok >= spl` 即此型，长期可编）。推测机理（假说，与错误信息字面吻合）：
   eq/ne 指令需把形参寄存器搬运到谓词比较所需宽度的寄存器，后端生成宽度不匹配的
   COPY。规避 = 差值判别形态（`OpClassifyEvictRow` 两处扫描，d=v−gen 后比常量）。
   **取证教训**：该错误不落 `~/ascend/log/run/plog`（那里只有 python 运行时日志），
   bisheng 也吞掉 clang 后端 fatal 后的 stack dump——变体二分是有效定位手段（每变体
   = 干净树 + 单点 sed + cp 备份还原；服务器无 git）。

### 7.7 dav-3510 无 64 位值运算

- 设备侧热路径全部使用 32 位运算（kernel 内索引/计数均 uint32_t/int32_t）；仅有的
  uint64_t 是打点用的 `perfLast_` 与 `clock()` 差值，且**立即 clamp 到 0x7FFFFFFF 后以
  int32 存储**（`echo_lru_kernel.h:1214`）。手抄修正清单里专门有一条 "perfLast_
  截断为 32 位" 的教训（README）。写设备代码时不要引入 64 位算术路径。

### 7.8 Event 计时含 ~百 μs 固定开销，看 kernel 级要用 profiler

- Event 口径是"公共算子接口的流区间延迟"，包含 tiling H2D（pageable 小缓冲拷贝）与
  host 下发空隙，两路各 ~百 μs（README §6、`profile_kernel.py:4-8`）。比值可用、绝对值
  不可用。
- 规避：kernel 纯耗时用 `bash profile_kernel.sh`（torch_npu.profiler 抓
  kernel_details.csv 的 P50/P95/min/max，§8.4）。

### 7.9 跨核 grid.sync 在本 kernel 内 trap（未解）

极简 probe 全 PASS，但 echo_lru Process 内哪怕一个 grid.sync 即 trap（输出全 0 不
hang），根因是编译单元整体属性（Process 巨大/19 Op/Init 多 GM 指针，代码注释原文），
probe 无法复现。跨核协同方案因此暂停（`echo_lru_kernel.h:1189-1194`）。**在可预见的将来不要
在本 kernel 里尝试 grid.sync**。

### 7.12 VEC SIMT 栈溢出（errcode 354）——单帧内联和 > 每线程栈上限

- **现象**：kernel 启动即 trap（上层 507035），全核同 PC、确定性、`-O0` 不救；
  **trap 位置随任意源码增删漂移**（帧在入口一次性分配，与执行路径无关）。
- **根因**：bisheng 把 `__global__` 入口的全部内联设备代码编成一个巨型栈帧
  （= 所有内联函数局部/溢出槽之和），超过向量核每线程栈硬上限。v1 帧恰在限下，
  v2 +2 Op 越界；普通构建擦线通过后打点构建再越界（余量以"行"计）。
- **规避**：重型设备函数一律 `__attribute__((noinline))`（帧峰值从"求和"变
  "取最大"）；本仓现行规则为**全部 Op noinline**。新增内联大函数前先想栈。
- **诊断铁律**：507035 第一步 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 重跑读 errcode
  字符串（上层异常不透出 354）；"位置漂移 + 与执行无关 + 全核同 PC"是本坑指纹，
  出现即停止语义排查（阶段二分对它完全失效——帧与执行路径无关）。
- **全记录**：`docs/ascend_vec_simt_stack_overflow.md`（含完整误判链复盘）。
  §7.9 的 grid.sync trap 高度疑似同根因（加代码涨帧越界），待按本条规则收窄帧后重测。

### 7.10 设备被占/残留进程：新进程第一次 sync 即 507034 超时

- **现象**：round-3 上板复核（§9.14）途中一轮 compare.py 报"cold_reset 挂死"——
  实为 507034 超时，出现在进程**第一次** `torch.npu.synchronize()`（`compare.py:294`，
  `prepare_npu` 尾部，早于任何 op 调用）。
- **根因**：device 1 已被他人占满；换卡后同一二进制四场景 golden 全过
  （§9.14 事故闭环）。
- **教训（测量 ops）**：挂死/超时**先 `npu-smi` 查卡占用与残留进程**、换卡或复位，
  再归因代码——第一次 sync 就超时说明问题在设备/环境侧，与 kernel 代码无关。

### 7.11 静态延迟链模型会高估依赖读成本（round-4 教训）

- **现象**：round-4 [P-E] 的静态审计把 pass1 hist 的 GM 读建成 "每线程 64 次 ×
  ~750cyc 串行依赖链"（§9.15），预期 Argtopk −30K 周期；上板实测仅 −9.1K
  （105.5K→96.4K）。
- **根因**：load 的**地址**（`i += G` 步进）与**加载到的数据无关**，硬件 scoreboard
  可并行挂起多个 load——真实成本是**吞吐受限**而非延迟链，"延迟 × 次数"的建模把
  本可重叠的访存串行化，系统性高估。
- **教训**：对地址与数据无关的访存序列，不要按串行延迟链估算优化收益；分步实测
  （echo_perf_step 前后对照，§8.3）是唯一可信的校准。

---

## 8. 构建与验证工作流

### 8.1 build.sh 与多套产物目录

`bash build.sh --soc=ascend950`（仅支持 ascend950）→ CMake 配置 + 编译到 `build/`；
`ECHO_PERF_STEP=ON bash build.sh`（或 `--perf-step`）→ 定义 `ECHO_PERF_STEP` 宏，产物
在 `build_perf/`；`bash build.sh --threads=N`（或环境变量 `ECHO_LRU_THREADS`，默认
1024）→ 覆盖 Echo 每核线程数**上限**（32 的倍数、[64,1024]，`build.sh:40-43` 校验；
实际启动线程数仍由 plugin 按 topk 运行时自适应，§3.2——该开关是**纯消融 knob**，
压低上限时 `topk > 4N` 会回退串行 MissCumsum，§5 Step5），N≠1024
时产物在 `build_t{N}` / `build_perf_t{N}`——三套产物目录互不覆盖（目录选择逻辑
`build.sh:46-57`；`.gitignore` 同步忽略 `build_t*/`、`build_perf_t*/`）。

CMakeLists 要点（`CMakeLists.txt:94-171`）：
- bisheng 单独编两个 kernel object：SIMT 用 mixed/VF 选项组（`--cce-auto-sync=off` 等
  四项），Echo 用 `--enable-simt`；两者均 `-O3 --npu-arch=dav-3510 -xasc`。
- `ECHO_PERF_STEP=ON` 时给 Echo kernel **和** plugin 同时定义宏（plugin 侧
  `EchoWorkspaceElements` 公式随之扩容 +28，保证 workspace 校验通过）。
- **线程数注入必须两侧同值**：CMake 缓存变量 `ECHO_LRU_THREADS`（默认 1024）经
  `-DECHO_LRU_THREAD_NUM_OVERRIDE=N` 同时加进 Echo kernel 的编译选项与 plugin 的
  `target_compile_definitions`（`CMakeLists.txt:100-113,165-170`）——plugin 侧它是
  自适应 blockDim 的 clamp 上限（`echo_lru_plugin.cpp:105-108`），kernel 侧它是
  `__launch_bounds__`（`echo_lru_kernel.h:1312`），实际 `<<<blockNum, blockDim>>>`
  传运行时计算值（`echo_lru_launch.cpp:8-9`）；上限必须 ≤ launch bounds，两侧同值
  注入即保证这一点，错位会静默 mismatch。
- 系统 C++ 编两个 plugin，链接 torch/torch_npu/ascendcl/tiling_api/platform/
  libascendc_runtime.a，产出 `libsimt_vs_echo_ops.so`，由 `torch.ops.load_library`
  触发 TORCH_LIBRARY 注册。

**本机 / 服务器分工（2026-10-16 起，owner 指示）**：本机计算资源需申请、未装编译
环境——**代码轮次的本机验证只跑 `python3 checks/check_echo_kernel_host.py`**（纯
CPU 逻辑检查，§8.5），不再本机 `bash build.sh` / bisheng 编译；编译与 NPU 验证由
owner 在服务器侧完成（pull → `build.sh` → compare.py audit → profile/perf_step）。
§9.15 及之前各轮记录的"本机编译 exit 0"是旧工作流口径；round-4b（§9.16）起本机
验证状态即"checks 通过、编译与上板归服务器"。

### 8.2 compare.py：审计 / 正式测量 / A-B-C-D 检查阶段

`compare.py` 把输入构造、golden、校验、计时集中在一个文件，主流程四阶段
（`compare.py:460-511`，README §7）：

- **A 计时前审计**（默认 3 次）：恢复状态 → 用 255/−777 污染全部输出、I32_MAX 污染
  workspace → 调用 → 与参考逐字段全等。排除"读到旧答案"的假通过。
- **B 预热**：仍每次恢复状态；预热后检查两路最后完整结果。
- **C 正式采样**：两路**交替先后**（偶数 repeat 各半先执行），记录原始 Event/Host 延迟；
  状态恢复在 Event 区间外（`measure_one`，`compare.py:335-347`：restore → sync →
  start.record → invoke → end.record → sync）。
- **D 计时后检查**：两路全部状态/输出 + 共同语义，成功才写 JSON（含
  `echo_over_simt_event_p50`、原始样本、执行顺序、库与源码哈希）。

消融入口（§8.6）：`--blocks N` 覆盖 Echo blockNum——默认 `min(rows, cores)`；限
1..rows，实际 `blocks = ceil(rows / ceil(rows/N))`（向上取整可能小于请求值，如 rows=32
请求 9 实际得 8），尾核非空由公式自动保证，超出向量核数由 plugin CHECK 拦截
（`compare.py:273-283,380-381`）；`--rows` 变体 batch 形态。results JSON 在
`blocks_override` 字段记录请求值便于归因（`compare.py:438`）。

`--verify-each-sample` 是短跑审计模式（每样本读回校验），会改变缓存条件，正式对比
必须关闭。`--cpu-check` 不加载 torch，只验证输入与两份参考。

### 8.3 echo_perf_step.py：分步周期、哨兵检测、频率标定

> [v2.1-prof 2026-10-20] 本节按 v2 差值口径撰写，**打点语义已升级**（§9.23）：每核
> 槽位存 t0 绝对 clock 低 32 位（29 槽 = entry + prologue×2 + 非 spec 18 + spec 陈列
> 8；[v2.2] prologue 两槽已删、27 槽 = entry + 非 spec 18 + spec 8，§9.24），host 侧
> uint32 回绕差分；频率默认 1.585 GHz 常数（msprof 实测）+ `--p50-us`
> 外部锚点两档对照（[2026-10-20 二轮] 锚点支持逗号分隔 `场景=值` 逐场景取值，纯
> 浮点 = 全场景统一旧行为；自带跨 run 漂移探测——两 run 频率可漂移 ~4-5%，§9.23
> 验证记录），**Event 标定行已废弃**（§7.8）；哨兵只认等值 -12345（绝对戳
> 高位可负，"负值即异常"防御废止）；统计新增 skew（每步 32 核 max−min 中位）与
> `--matrix` CSV 导出（供 prof_report.py）。下文未逐条重写，以 §9.23 为准。

- 直接 import compare.py 的输入构造/参考/恢复/调用（同尺寸/seed/场景），分步之和可与
  Event 总耗时互相印证（`echo_perf_step.py:11-16`）。
- **哨兵机制**：每次调用前用 `-12345` 填充 workspace；读回 stepCycles 区若仍有哨兵或
  负值 → 报错。这同时兼任"加载的是不是打点构建"的检测（普通构建不写该区，
  `echo_perf_step.py:157-194`）。
- **统计口径**：每步取"最慢核"（max over blocks）的跨轮中位数；用每轮 Σ步周期 / Event
  微秒标定频率（`echo_perf_step.py:200-211`）。
- **步名防漂移断言**：`ECHO_PERF_NUM_STEPS == 20 + 8`（`echo_perf_step.py:59-62`），
  tiling 常量与步名表任一方改动都会触发。
- 注意：打点构建改变代码形态，其总耗时不用于正式结论；每步周期含 PerfBegin/End 两次
  clock() 开销（~2-3K/步）。

### 8.4 profile_kernel.sh / profile_kernel.py：kernel 级 P50

- 采集目录放 `/dev/shm`（trace 文本量大，避免写盘），清掉 `ASCEND_LAUNCH_BLOCKING` /
  `ASCEND_RT_VISIBLE_DEVICES` 等干扰变量（`profile_kernel.sh:11-18`）。
- `torch_npu.profiler` 按 `schedule(wait=0, warmup, active, repeat=1)` 采样，导出
  kernel_details.csv；**只统计** `sparse_kv_plan_kernel` / `echo_lru_kernel` 两个名字
  （restore 的 copy_ 等按名过滤），并核对 kernel 次数与预期一致
  （`profile_kernel.py:53-56,59-79`）。
- 采集前后均与 golden 全量比对；汇总 `summary.json` 含两路 P50/P95/min/max、原始样本
  与 `echo_over_simt_kernel_p50`。
- 消融透传：`--blocks` / `--rows` 与 compare.py 同语义（summary JSON 同样记录
  `blocks_override`，`profile_kernel.py:140-141,176,190`）；线程数消融用
  `--library build_t{N}/libsimt_vs_echo_ops.so` 指向对应构建产物。

### 8.5 checks/check_echo_kernel_host.py：172 项 CPU 回归

用 Python 标准库 + 支持 C++20 的 g++ 运行：**直接从 `echo_lru_kernel.h` 文本提取**
`PackedBlockScan` / `OpHitMiss` / `OpArgtopkRow` / `OpMissCumsum` / `OpAllocPop` /
`OpFifoAvailUpdate` 六个函数源码，配上 mock SIMT 头（128 个 std::thread 模拟一个
block、`std::barrier` 模拟 syncthreads、共享缓冲模拟 shfl_up/reduce_add）编译执行，
**开 ASan/UBSan**（`check_echo_kernel_host.py:33-43,55-92`）。

覆盖面（合计 172 项 = 160 + 1 + 3 + 7 + 1）：

| 组 | 数量 | 覆盖 |
|---|---:|---|
| 1. OpArgtopkRow | 160 = 10 个 B（1..4096）× 4 种 pri pattern（含无效/平局/全无效）× **4 个 fifo 变体**（全趟/跳2趟/跳1趟/只跑 byte0） | fsClamp 钳位与提前返回（输出/UB/dbg 全不写的哨兵断言）；选择 golden（T=第 fsClamp 小键、quota 公式）与 stable_sort (key,slot) 交叉验证；输出逐位（含 padding 补齐与尾部补零）；key 只写有效区、padding 区/UB 越界哨兵；dbgKey 恒验证、dbgCount 仅全趟路径验证（跳趟首趟是过滤直方图，语义随起跑趟变化） |
| 1b. [P-C] UB 路径 | 1 | freeIdxIsGm=false（freeIdx 走 UB）时输出与 GM 路径 golden **逐位一致**（`check_echo_kernel_host.py:183-210`） |
| 2. OpHitMiss | 3（eff=4/1536/2048） | hit 槽位、[protect 折叠] pri 时间戳 golden（distinct token→distinct slot 直接覆盖）、miss 计数、[P4] 折叠 freeSize、[C1] 淘汰标志；**断言零 atomic**（`check_echo_kernel_host.py:212-246`） |
| 3. OpMissCumsum | 7（topk=4..2048） | **三条路径同 golden**（`check_echo_kernel_host.py:247-271`）：blockDim=128 时 2 元素路径（topk≤256）、4 元素两次链式扫描路径（300/512：topk≤4×128=512）、t0 串行回退（2048）的 rank/missPos 逐位一致；scan 缓冲按 [Scan-v2] 四缓冲扩至 160 槽（`:258`） |
| 4. OpAllocPop + OpFifoAvailUpdate | 1 | nm/fs 四种组合下与"旧栈先 push 再弹"golden **逐值等价**；avail 折叠终值恒等且 ≥0、fifo 恰好 +1（`check_echo_kernel_host.py:272-310`） |

模型常量断言（ARGTOPK_GROUP_THREADS=64 等）防止 tiling 改动后误用旧模型
（`check_echo_kernel_host.py:28-32`）。它是 CPU 算法和索引检查，**不能**替代 NPU 编译、
设备缓存一致性或性能验证。

### 8.6 线程数 / blockNum / batch 消融（ablation.sh）

一键脚本 `ablation.sh`：逐配置构建（`build.sh --threads=N` → `build_t{N}/`）→ 每配置 4 场景 kernel 级 profile（复用 profile_kernel.sh，采集前后过 golden）→ 汇总 `results/ablation-<时间戳>/report.md`（`results/` 不入库，数据在采集机）。参数：`--device/--active/--threads/--blocks/--rows/--phase(threads|blocks|rows|all)/--skip-build`。矩阵默认 threads {1024,768,512,256}、blocks {16,8}、rows {64,128}；基准为 t1024（消融对照；自适应版生产 launch 见 §3.2，topk=2048 → 512）。解读注意：t256 的 MissCumsum 走 t0 串行（4×256=1024<topk 2048，混杂因素）；rows64/128 的跨配置对比需注意 rows 口径。三相结论见 §9.10：t512 全场景最优（甜点不随场景漂移）、blocks 16/8 单调变差、rows 64/128 暴露行间串行软肋。


## 9. 优化迭代日志

先给总表（数据来源：ECHO_PERF_STEP 分步周期 = Σ步 max 跨轮中位，Event = compare.py
口径；标注"kernel 级"者为 profiler P50）。**优化前基线**（手抄修正后、P0 之前，warm 三
场景分步周期）：

| 步 | 基线周期 | 根因 |
|---|---:|---|
| OpHitMiss | **9.3-9.4M** | 每线程一次 GM 同地址 atomic，~1024 次/行 × ~9K 周期；cold 无 atomic 仅 10K、hit 数 1536 vs 2048 耗时几乎相同，证实根因 |
| ArgtopkAllRows | **2.1M** | t0 串行 emission 两遍 8192 扫描（无法隐藏延迟） |
| OpMissCumsum | 20K | Hillis-Steele 11 轮块栅栏，barrier bound（cold/warm 耗时相同） |
| OpFinalSlots | 10K | 全量 topk 重新 gather htd（冗余） |

基线 Echo/SIMT Event 比 2-10×。

| 轮次 | 改动 | warm | cold | full | Event（Echo/SIMT） |
|---|---|---:|---:|---:|---|
| 基线 | — | 9.4M | — | — | 2-10× |
| P0-P3 | 去 atomic / emission 并行 / 删 FinalSlots / 两级扫描 | 125K | — | 514K | 0.81-0.92×（full 1.38×） |
| P4 | 栅栏削减（步骤 40→31） | 106K | 166K | 509K | — |
| B+C1+C2 | Argtopk 重构（该步 410K→199K）+ 无淘汰跳过 + 末行栅栏 | ~95K | 139K | 309K | 审计 0.91-0.97×（full 1.00×） |
| 方向1+2（145d3ce） | 自适应跳趟 + 决策本地化 | — | — | Argtopk ~135→~87 μs（估算），kernel 210.51→162.16 μs | kernel 级 0.95×/0.91×（warm）、1.38×（cold）、1.75×（full） |
| 消融基础设施（d20fc50） | 线程数可配 + 4 元素路径 + `--blocks`/`--rows` 入口 | — | — | —（默认 1024 线程路径无变化；消融结果见 §9.10） | — |
| 自适应 blockDim（e2ca2fd） | plugin 按 topkDim 算 launchThreads，`<<<blockNum, blockDim>>>`（§3.2） | 63.30 / 59.17 μs | 91.44 μs | 137.38 μs | kernel 级 0.78×/0.73×（warm）、1.24×（cold）、1.48×（full）＝消融 t512 数字（§9.10）；自适应构建 NPU 已复核（§9.14） |
| round-3 两批（f8b9c69+1121091） | Scan-v2 / 破坏性归并 / freeIdx 迁 UB / pack 融合 + protect 折叠 / fifo++ 挪位 / FifoAvail 拆分（非 spec 步骤 23→20，§9.11/§9.12） | — | — | 129.29 μs / −5.9% vs t512（动机分步数据见 §9.11；上板复核见 §9.14） | — |
| round-4（3906f91，§9.15） | [P-E] 反向 P-D（行首全员 pack + B0，Argtopk 栅栏 7→8）+ 白付清理 A/B（非 spec 跳过 prologue、无 reset 跳过 Step0） | 58.28 / 55.34 μs | 94.16 μs | 120.12 μs（Argtopk 105.5K→96.4K，实测 −9.1K 远低于静态预期 −30K） | kernel 级 0.72×/0.67×（warm）、1.25×（cold）、1.29×（full）＝上板验证（§9.15） |

**kernel 级 profiler（profile_kernel.sh，P50，SIMT/Echo，方向 1+2 落地后）**：

| 场景 | SIMT (μs) | Echo (μs) | Echo/SIMT |
|---|---:|---:|---:|
| cold_reset | 73.55 | 101.67 | 1.38× |
| warm_75pct | 80.61 | 76.44 | 0.95× |
| warm_all_hit | 81.08 | 73.52 | 0.91× |
| full_cache_75pct | 92.65 | 162.16 | **1.75×** |

（该轮为 1024 线程 launch 口径；blockDim 自适应后的现状数字见 §9.10 / §10.1。SIMT
基线正被同事优化至 ~50 μs，对比结论要随之更新分母。）`clock()` 频率约
1.47 GHz（B+C1+C2 轮打点构建标定：full 场景 Σ步 309K cyc ≈ 同期 kernel 210.51 μs
反推；方向 1+2 的 Argtopk ~87 μs 为按此频率的估算值）。

### 9.1 P0：OpHitMiss 去 GM 同地址 atomic

- **现象**：warm 场景该步 9.3-9.4M 周期，占绝对大头；cold（无 atomic）仅 ~10K。
- **根因**：1024 线程对同一 GM 地址的 atomic 串行化，每次 ~9K 周期（§7.1）。
- **改法**：warp 归约（`asc_reduce_add`）→ lane0 写 UB → 1 栅栏 → t0 汇总直接写
  `numMisses`（`echo_lru_kernel.h:298-309`）；OpNumMisses 子步删除。
- **效果**：warm 该步 9.4M → ~19K（source_manifest 记录），Event 比进入 <1 区间。

### 9.2 P1：Argtopk emission 并行化

- **现象**：full_cache 该步 ~2.1M 周期。
- **根因**：t0 串行两遍 Npad 扫描 + 串行补零，延迟无法隐藏（§7.4）。
- **改法**：全员连续切片统计 `(nLt | nEq<<16)` 打包块扫描得写入基址 → 并行写 + 并行
  补零；连续切片保证输出仍按全局槽号升序、与原语义逐位一致（`echo_lru_kernel.h:560-563`）。
- **效果**：该步 2.1M → 398K（source_manifest）。

### 9.3 P2：OpFinalSlots 删除

- **现象**：miss 的最终槽位在 OpSetUpdate 里已经现成（aslot），之后再全量 gather 一遍
  htd 纯属重复（~10K 周期 + 一个尾步栅栏）。
- **改法**：OpSetUpdate 顺手写回 `curSlots[j]=aslot`（hit 位 Step1 已正确），
  OpFinalSlots 删除（`echo_lru_kernel.h:947-950,959-961`）。

### 9.4 P3：OpMissCumsum 去 Hillis-Steele

- **现象**：该步 ~20K 周期且 cold/warm 相同 → barrier bound 而非 work bound。
- **根因**：11 轮 × 1024 线程块栅栏（Hillis-Steele 扫描每轮一次栅栏）。
- **改法**：warp shuffle 一级 + t0 扫 warp total 二级（3 栅栏/行），p0/p0+blockDim 连续
  映射使 rank 可由打包前缀直接拼装（§5 Step5）。
- **效果**：20K → ~9K（source_manifest）。

### 9.5 P4：栅栏削减（步骤 40→31）

- **改法**（每项都有依赖论证，见 §5）：OpFreeSize 折进 Step1 t0 汇总段；OpAllocPop 直读
  freeIdx + 老栈（等价性推导见 §5 Step3）；`avail += fs - nm` 折叠为 Step4 尾一次写
  （终值恒等式见 §5 Step4）；删 FreeRelease→FreePush、FreePush→AllocPop 两个无 RAW
  SyncGrid 与 Step4 尾无读者 SyncGrid；PackedBlockScan 去保护性第 3 栅栏（顺序论证见
  §4.3）。
- **效果**：warm 106K、cold 166K、full 509K（相对 P0-P3 的 125K/?/514K，full 微降、
  warm 因打点步减少略升属噪声范围）。

### 9.6 B：Argtopk 重构（410K → 199K）

- **现象**：P4 后 full_cache 的 ArgtopkAllRows ~410K 周期，占该场景 ~80%。
- **改法**（§6.2/6.3）：pass 0 的 pack 与 clear+hist0 合并为单相位（hist0 直读 priGm，
  无交叉 RAW）；scalar 初始化与 dbg dump 折进 t0 决策段；每趟栅栏 4→3；merge 改 256
  桶并行；hist/pack/emission 只扫有效区 `[0, B+1)`，padding 键解析式计入 + emission
  按需补齐——与旧满扫描输出逐位一致（`echo_lru_kernel.h:381-384`）。
- **效果**：该步 410K → 199K。

### 9.7 C1+C2：无淘汰跳过与末行栅栏

- **C1**：Step1 的 t0 在 UB 写"本核任一行 fs>0"标志，全 0 时 Argtopk/SyncGrid/
  FreeRelease/FreePush 四步整体跳过（安全性论证见 §5 Step2）。warm/cold 场景四步 ≈0
  周期。
- **C2**：OpHitMiss 的 warpScratch 复用保护栅栏仅在还有下一行时执行——该末行特判
  后被 [Scan-v2] 的行奇偶双缓冲整体取代（§9.11），现每行恒 1 栅栏（§5 Step1）。
- **效果**：warm ~95K、cold 139K、full 309K；Event 审计 0.91-0.97× / full 1.00×。

### 9.8 方向 1+2（commit 145d3ce）

- **改法**：自适应跳趟（fifo 前导零字节 → startPass，§6.5）+ 决策本地化（去 t0 串行
  决策与 scalar 广播，每趟 3→2 栅栏，§6.4）。基准路径栅栏 16→8。
- **验证状态**：checks（现 172 项）ASan/UBSan 通过（4 个 fifo 变体全组合，跳趟/回退
  输出与全趟逐位一致）、g++ mock-SIMT 全头语法检查（开/关 ECHO_PERF_STEP）通过、
  bisheng 编译普通/打点 kernel 通过。**kernel 级实测（profiler P50）**：full_cache
  210.51 → 162.16 μs（2.27×→1.75×），warm_75 0.96×→0.95×、warm_all 0.93×→0.91×，
  cold 1.39×→1.38×；ArgtopkAllRows 估算 ~135 μs → ~87 μs（降幅 ~48 μs ≈ kernel
  总降幅，该步吃掉了几乎全部改进空间）。
- **附带教训**：踩到 bisheng 前端两个段错误坑（§7.6），规避写法已固化在代码注释。

### 9.9 消融基础设施（commit d20fc50）

- **内容**：`ECHO_LRU_THREAD_NUM` 编译期可配（§3.2：CMake `ECHO_LRU_THREADS` +
  `build.sh --threads=N`，产物目录 `build_t{N}` 隔离）；OpMissCumsum 每线程 4 元素
   并行路径（§5 Step5——512 线程消融的前置，2 元素门在 512 线程下装不下生产 topk，
  否则掉回 t0 串行）；compare.py / profile_kernel.py `--blocks` 覆盖与
  `blocks_override` 归因字段（§8.2/8.4）；README 消融实验章节（线程
  {1024,768,512,256}×4 场景 + blocks {32,16,8} + rows {32,64,128} 矩阵，§8.6）。
- **性质**：默认 1024 线程路径行为不变（4 元素门不命中），本轮无性能主张；CPU 回归
  171 项（新增 topk=300/512 两个 4 元素用例）、bisheng 三配置（默认 / 512 覆盖 /
  打点）编译通过。NPU 消融数据见 §9.10。

### 9.10 消融实验（2026-10-16）

ablation.sh（§8.6）在 NPU 服务器跑完线程 / blocks / rows 三相矩阵。口径：kernel 级
P50，Echo μs / Echo÷SIMT；基准形态 rows=32、topk=2048、capacity=4096。

**线程数**（其余默认）：

| blockDim | cold_reset | full_cache_75pct | warm_75pct | warm_all_hit |
|---:|---|---|---|---|
| 1024（消融基准） | 100.05 μs / 1.36× | 159.44 μs / 1.73× | 74.42 μs / 0.93× | 71.82 μs / 0.89× |
| **512（全场景最优）** | **91.44 μs / 1.24×** | **137.38 μs / 1.48×** | **63.30 μs / 0.78×** | **59.17 μs / 0.73×** |

- **t512 全场景最优，较 t1024 −9%（cold）~−18%（warm_all）**（warm75 −15%、full
  −14%，warm 获益最大）。结论："**线程多不一定快**"在本 kernel 成立——栅栏成本/
  到达散布的收益 > gather 扫列轮数翻倍的代价（512 线程扫 2048 列要 4 轮、1024 线程
  2 轮；§7.3）。
- t768：全场景较 t1024 小幅改善，不及 t512。
- **t256 崩塌（7–9×，587–673 μs）是混杂因素而非线程数本身**：topk=2048 > 4×256，
  OpMissCumsum 掉 t0 串行 GM 循环（≈550 μs ≈ 2048 元素 × 4 次 GM 访问 × 串行无法
  隐藏延迟，§7.4），与 miss 数无关（warm_all 也 587 μs）。

**blocks 16/8**：单调变差（每核串行 2/4 行），符合预期——`min(rows, cores)` 默认
保持。

**rows 64/128**（每核 2/4 行，生产 batch > 核数形态）：**Echo 的 warm 优势消失**——
Echo/SIMT 比 rows128 warm 1.07–1.13×、rows64 1.17–1.25×（对比 rows32 的
0.89–0.93×）；SIMT 行扩展更优（rows=128 时 warm 耗时相对 rows=32 的扩展比：SIMT
仅 1.84× vs Echo 2.25×）。根因：**每核行间完全串行**（各 Op 内 r 循环逐行执行、
无跨行重叠）是结构性软肋（§10.3②）。

**落地（e2ca2fd）**：plugin 按 topkDim 自适应 blockDim（§3.2），topk=2048 → 512
自动落在甜点。本机验证：checks 171 项、自适应公式 9 组边界用例（含 2047/2049/4095）、
bisheng kernel 编译（launch.cpp）exit 0、plugin 头语法检查；NPU 复核已完成
（§9.14：cold/warm 与 t512 数字噪声内一致，预期兑现；full −5.9% 归因 round-3）。

### 9.11 round-3 batch 1（commit f8b9c69）：Scan-v2 / 破坏性归并 / freeIdx 迁 UB / pack 融合

- **动机数据（512 线程自适应口径分步，full_cache Σ216K 周期）**：ArgtopkAllRows
  111.8K（51.9%）、MissCumsum 12.6K（4 元素路径 4 栅栏/行）、HitMiss 21.4K——热点
  集中在 kernel 内部的栅栏次数与 GM 往返，四项优化全部不改变任何输出。
- **[Scan-v2] PackedBlockScan 去 B2**（§4）：签名改单缓冲 `(packedValue, warpTotals)`，
  B1 后每线程本地读 `warpTotals[0..warp)` 算自己 warp 的 base；调用方 grand total 本地
  Σ。调用点配套：OpMissCumsum 2 元素路径 warpTotals 行奇偶双缓冲、4 元素路径
  (行奇偶×scan序) 四缓冲（ctrlA/ctrlB 控制槽删除）——背靠背链式扫描间不再需要
  栅栏，4→2 栅栏/行；OpHitMiss 归约 2→1 栅栏/行（双缓冲取代 C2 末行特判）；
  Argtopk emission 2→1 栅栏。
- **[P-B] 破坏性归并**（§6.3）：merge 读切片后清零（`s += *slot; *slot = 0`），下一趟
  hist 不 clear；clear 仅保留首迭代（覆盖 kernel 启动时 UB 初始垃圾），回退重跑靠
  失败趟 merge 清零（趟间 B2 隔开）。
- **[P-C] freeIdx 迁 UB（带门）**（§3.3）：Init 计算 `uo + rowsPerCore*topkDim ≤
  ARGTOPK_UB_ELEMS`，满足则 `freeIdxPtr_`=UB（基准形态 27144 ≤ 32768 ✓），否则回退
  GM（大 batch）；emission 尾 dcci 与 Process 中 Argtopk 后的 SyncGrid 均条件化（UB
  路径普通 syncthreads 即可）；OpFreeRelease/OpFreePush/OpAllocPop 改读 `freeIdxPtr_`。
- **[P-D] pack 融合**（§6.2）：首趟 active 线程的 hist 循环（本来就读 priGm）顺手写
  `key[i]`（stride=G 覆盖 [0,validN)），独立的全员 pack 循环删除。（round-4 [P-E] 已
  回退该融合，见 §9.15——融合删掉的 pack 循环本便宜，真正的 GM 依赖链未动。）
- **验证（本机）**：checks 172 项 ASan/UBSan 通过（新增 UB 路径 vs GM 路径输出逐位
  一致用例；scan 缓冲扩至 160）；bisheng 三配置（默认/512 覆盖/打点）编译 exit 0；
  g++ mock 全头语法通过。步骤数/步名不变（23 步）。NPU 复核已完成（§9.14）。

### 9.12 round-3 batch 2（commit 1121091）：protect 折叠 / fifo++ 挪位 / FifoAvail 拆分 / 步骤重排

- **[protect 折叠]**：OpProtect 并入 OpHitMiss 的 hit 分支（§5 Step1）——slot 就在手边
  顺手写 `pri=fifo`，clamp 语义同前；distinct token → distinct slot 无写冲突；pri→
  Argtopk 的 RAW 由 Step1 尾 SyncGrid 保证。OpHitMiss 签名增加 B/priGm/fifoGm。原
  Step2 整段删除（OpProtect 步、Bug2 SyncBlock、OpFifoInc、尾 SyncGrid 共 −4 步）。
- **[OpFifoInc 挪位——设计修正]**：直接删 Step2 尾 SyncGrid 会让 Argtopk 读到不一致
  fifo（t0 的 OpFifoInc 写滞留其 D-cache、其他线程读 GM 旧值 → startPass 分歧 →
  栅栏发散，完整机理见 §5 Step1 末段）。修法：OpFifoInc 挪至 OpAllocPop 之后（Step3
  中段）——Argtopk 恒读 Step1 尾 dcci 后的一致旧值（pri ≤ 旧 fifo，跳趟安全），
  AllocPriority 读 +1 值由 Step3 尾 SyncGrid 保证；fifo 两次自增语义（protect 盖 f、
  alloc 盖 f+1、终值 f+2）逐值不变。
- **[FifoAvail 拆分]**：OpFifoAvailUpdate 的 fifo++（t0）与 avail+=fs−nm（t1）无数据
  依赖，各管一个 → t0 串行 GM 往返延迟减半（各 ~3 次依赖访存，§5 Step4）。
- **步骤重排**：非 spec 23→20（Step2 整删；新序：Step0 reset×5、Step1
  OpHitMiss(含 protect)+SyncGrid、Step2 Argtopk+SyncGrid(条件)+FreeRelease+FreePush、
  Step3 AllocPop+OpFifoInc+SyncGrid、Step4 AllocPriority+SyncBlock+FifoAvailUpdate、
  Step5 MissCumsum+SyncGrid+SetUpdate）；`ECHO_PERF_NUM_STEPS` 31→28，步名表与打点
  顺序逐一核对一致（§5.1）。
- **验证（本机）**：checks 172 项 ASan/UBSan 通过（OpHitMiss 用例新增 protect pri
  时间戳 golden）；bisheng 三配置编译 exit 0；g++ mock 全头语法通过；步骤顺序 20+8
  核对一致。NPU 复核已完成（§9.14）。

---

### 9.13 事故记录：UB freeIdx 绝对行索引越界（2026-10-16，d461f5d 修复）

**现象**：round-3 后 compare.py 仅在 full_cache_75pct 挂死（其余三场景正常）。
**根因**：[P-C] freeIdx 迁 UB 时沿用 GM 的绝对行索引 `freeIdxOut[row*topkDim+w]`；GM 布局按绝对行分区合法，但 UB 缓冲是每核私有的 `rowsPerCore*topkDim`（基准 1 行=2048 元素）。block≥1 写到 `uo+row*2048`，最大 32×2048=65544 ≫ ARGTOPK_UB_ELEMS=32768 → UB 越界写 → 核挂死。cold/warm 的 fs=0 被 C1 跳过不写 freeIdx，故仅 full_cache 触发。
**盲区**：CPU checks 单 row=0（startRow=0）时绝对==本地，无法覆盖。
**修复**：UB 路径全部改核内本地行索引 `row-startRow`（OpArgtopkRow 的 freeIdxOut 参数语义改为"本行切片基址"，由 ArgtopkAllRows 计算传入；OpFreeRelease/OpFreePush/OpAllocPop 增加 freeIdxLocal 参数）；GM 回退路径保持绝对行。rawSlots 等其余 GM 量恒绝对行。
**教训**：per-core 私有缓冲的行索引基准与全局共享缓冲不同，迁移数据结构时索引语义必须同步审计；单 block 测试无法暴露多 block 索引问题。

### 9.14 round-3 上板复核（2026-10-16，第二台服务器）

round-3 两批（f8b9c69+1121091，§9.11/§9.12）+ UB freeIdx 修复（d461f5d，§9.13）
在第二台 NPU 服务器完成上板复核（自适应 blockDim 构建，topk=2048 → 512 线程，
§3.2）。**事故闭环**：d461f5d 设备复测通过——full_cache_75pct 不再挂死，四场景
golden 全过；此前一轮"cold_reset 挂死"实为 device 1 被他人占满（507034 出现在
进程第一次 sync，`compare.py:294`，早于任何 op 调用，§7.10）。

**kernel 级 P50**（profile_kernel.sh 口径，blocks=32、rows=32、topk=2048、
capacity=4096）：

| 场景 | SIMT μs | Echo μs | Echo/SIMT |
|---|---:|---:|---:|
| cold_reset | 140.86 | 92.94 | 0.66× |
| warm_75pct | 150.28 | 64.39 | 0.43× |
| warm_all_hit | 151.47 | 61.92 | 0.41× |
| full_cache_75pct | 169.35 | 129.29 | 0.76× |

**与 §9.10 消融 t512（前一台服务器、round-3 之前）的 Echo 对比**：

| 场景 | §9.10 t512 | 本轮 | Δ | 判读 |
|---|---:|---:|---:|---|
| cold_reset | 91.44 | 92.94 | +1.6% | 噪声内复现 |
| warm_75pct | 63.30 | 64.39 | +1.7% | 噪声内复现 |
| warm_all_hit | 59.17 | 61.92 | +4.6% | 噪声内复现 |
| full_cache_75pct | 137.38 | 129.29 | **−5.9%** | 归因 round-3 |

cold/warm 跨机噪声内复现 → **Echo 数字跨机可信**，"自适应构建 = t512 消融数字"
的预期兑现（§9.10）；full −5.9% 归因 round-3 的 Argtopk 定向优化——ArgtopkAllRows
111.8K → 105.5K 周期（§9.11 动机数据 → 本轮分步，见下）。

**分步数据**（echo_perf_step 口径，每步 = 最慢核跨轮中位，单位周期，% 为占该场景
Σ步周期比。**本轮分步 μs 换算不可信**——见 open question 2，只看周期）：

| 场景（Σ步周期） | Top1 | Top2 | Top3 | Top4 | Top5 |
|---|---|---|---|---|---|
| cold_reset（Σ138468） | OpResetRows 31788 / 23.1% | OpSetUpdate 23424 / 17.0% | OpHitMiss 20420 / 14.8% | OpMissCumsum 15542 / 11.3% | OpAllocPop 7317 / 5.3% |
| warm_75pct（Σ96418） | OpHitMiss 26812 / 28.3% | OpMissCumsum 14250 / 15.0% | OpSetUpdate 13964 / 14.7% | OpAllocPop 5860 / 6.1% | OpAllocPriority 5610 / 5.8% |
| warm_all_hit（Σ88094） | OpHitMiss 26635 / 30.4% | OpMissCumsum 14036 / 16.0% | OpSetUpdate 11084 / 12.6% | OpAllocPriority 5351 / 6.1% | OpAllocPop 5278 / 6.0% |
| full_cache_75pct（Σ210956） | ArgtopkAllRows 105459 / 50.1% | OpHitMiss 28154 / 13.4% | OpMissCumsum 14658 / 7.0% | OpSetUpdate 14241 / 6.8% | OpFreeRelease 7712 / 3.7% |

- **OpHitMiss 21.4K→28.2K（full 口径）不是回退**：protect 折叠（§9.12）把原独立
  OpProtect 步（约 7K 周期）并入 OpHitMiss 的 hit 分支，属记账转移，两步合计相当。
- **频率交叉验证**：Σ步周期 ÷ 本轮 kernel P50 ≈ 1.42–1.63 GHz（cold 138468 ÷
  92.94 μs ≈ 1.49 GHz、full 210956 ÷ 129.29 μs ≈ 1.63 GHz），与历史 ~1.47 GHz
  标定（§9 开篇注）一致 → **分步周期数可信**（尽管脚本自身的频率标定失真，见
  open question 2）。

**两个测量疑点（open question，暂不下结论）**：

1. **SIMT 分母疑似受扰——已关闭（round-4 空闲设备重测，§9.15 验证结论 3）**：本轮
   SIMT P50 比前一台服务器慢约 2×（cold 73.55→140.86、warm_75 80.61→150.28、
   warm_all 81.08→151.47、full 92.65→169.35 μs；旧值见 §9 开篇方向 1+2 表，
   §9.10 反推一致）而 Echo 跨机复现 → 本轮 0.66×/0.43×/0.41×/0.76× 比值**不可作
   正式结论**。疑似共卡干扰或设备状态差异：本轮 device 1 曾被他人占满（§7.10），
   换卡后继续。round-4 上板（空闲设备）SIMT 回到 75-93 μs（§9.10 水平）→ 确认为
   设备占用干扰，本轮比值作废；正式比值以 §9.15 为准（warm 0.72×/0.67×、cold
   1.25×、full 1.29×）。
2. **Event 口径失真**：Event P50 与 kernel P50 的差距远超历史 ~200μs 量级开销
   （§7.8；cold 641.67 vs 92.94 μs），echo_perf_step 频率标定随之失真（214/168/
   180/386 MHz vs 真实 ~1.5 GHz）→ 本轮 Event 口径与分步 μs 换算**不可用**，只看
   周期；host 侧疑似受扰。

### 9.15 round-4：[P-E] 反向 P-D + prologue/Step0 白付清理（2026-10-16，commit 3906f91，已上板验证）

三处改动全部不改变任何输出（checks 逐位一致）；行号按当前 HEAD 代码。主会话已逐
hunk 审查通过；改动已提交（3906f91，文档同步 9b63e24）并完成上板验证（见本节末
"上板验证"）。

- **[P-E 反向 P-D]（OpArgtopkRow，`echo_lru_kernel.h:429-440`）**：在 fsClamp==0 早退
  之后、attempt 循环之前新增**全员 pack 相位 + B0 栅栏**——全块 blockDim 线程读
  priGm 写 key[]（每线程 ≤ceil(validN/blockDim.x) 次、完全协同、高 MLP），此后 hist
  首迭代与后续趟一律读 key[]（UB）。**动机**：分步审计证实 pass1 hist 的 G=64 路 GM
  依赖读链（每线程 64 次串行依赖读、MLP=64）占 Argtopk 105.5K 周期（§9.14）中的
  40-51K，单项主导；round-3 [P-D] pack 融合删掉的全员 pack 循环本来就便宜（纯协同
  写、无 GM 链），GM 读链一根没动——用 1 个 B0 把 priGm 读从 G=64 路换成全块
  blockDim 路并行。**栅栏计数 7→8**（B0 1 + 2 趟×2 + emission 扫描 1 + 尾 2，§6.7；
  头注释 `:395-397` 已同步）。可见性：priGm 的上一写者是 Step1 OpHitMiss（protect），
  由 Step1 尾 SyncGrid 保证；上一行 emission 对同一 key UB 区的读已由其尾两栅栏排空
  （§9.13 教训：key 无行索引）。
- **[白付清理 A]（Process prologue，`echo_lru_kernel.h:1085-1095`）**：非 spec 时
  OpFillZero + 其 SyncGrid 整段跳过——pushCount 的唯一读者 OpSpecAvailAdd 在
  Step0.5 的 spec 门内，非 spec 时该 SyncGrid 无 RAW 可保护；prologue 本就不占
  打点槽，无打点顺序影响。
- **[白付清理 B]（Process Step0，`echo_lru_kernel.h:1097-1117`）**：每线程各自读本核
  行段 resetMask 算同一 anyReset（全员读同样字节 → 逐线程一致 → 栅栏无发散），
  anyReset=0 时四个 reset Op + Step0 SyncGrid 全跳过；打点槽原位保留（跳过记录 ≈0
  周期，同 C1 做法，步数/步名/顺序不变）。**安全性论证**：四个 reset Op 对该核行段
  本来就全部 continue 零写；Step1 读的 htd/avail/fifo 是跨调用持久态，写者在上一
  invocation 或 host，kernel 边界保证可见（同 P4 删尾 SyncGrid 的论证：kernel 结束
  即对 host/下一次调用可见），本 kernel 内无需 dcci。

**本机验证**：checks 172 项全过（输出逐位不变）；`bash build.sh` 与
`ECHO_PERF_STEP=ON bash build.sh` 均 exit 0；28 个打点槽顺序逐一核对一致。

**NPU 预期（估算口径，上板前）**：full_cache ArgtopkAllRows 105.5K → ~75K 周期、
kernel 129.29 → ~107-110 μs；warm 64.39/61.92 → ~61-63 μs；cold 92.94 → ~90-92 μs。
（实测对照见下节：warm −9.5%/−10.6% **超出**白付清理预期、cold 噪声内持平（reset
全 1 无白付可省）；Argtopk 实际 −9.1K 远低于静态预期 −30K，full 120.12 μs 未达
~107-110 μs 预期即源于此——验证结论 2。）

**残余风险（来自代码子代理报告、主会话认可；上板已验证为安全，见下）**：跳过的两个 dcci 点的 invalidate
侧——若 runtime 不在 kernel launch 时 invalidate 本核 D-cache 且跨核不连贯，
warm/full 三场景可能读到 restore 前的 stale 行；compare.py 的 audit 第一轮（restore →
哨兵污染 → invoke → 全量比对，§8.2）即可抓住。**NPU 若挂的二分顺序**：① 无条件恢复
Step0 SyncGrid（保留其余）→ 过则锁定白付清理 B；② 再恢复 FillZero+SyncGrid → 过则
锁定 A；③ 都恢复仍挂才查 P-E（B0 是加栅栏，priGm 读可见性与旧 pass1 直读同构，
风险极低）。辅助判据：cold_reset 过而 warm/full 挂 → 强指跨 invocation 可见性问题。

**上板验证（2026-10-16，空闲设备）**：3906f91+9b63e24 构建上板，audit/golden 四
场景全过——上述残余风险（跳过两个 dcci 的跨 invocation 可见性）在设备上兑现为
**安全**，挂起二分顺序未启用。口径：kernel 级 P50（profile_kernel.sh，空闲设备，
blocks=32、rows=32、topk=2048、capacity=4096，自适应 512 线程，§8.4）：

| 场景 | SIMT μs | Echo μs | Echo/SIMT | vs round-3（§9.14） |
|---|---:|---:|---:|---|
| cold_reset | 75.24 | 94.16 | 1.25× | 92.94 → 94.16（+1.3%，噪声内持平；cold 无白付可省） |
| warm_75pct | 81.20 | 58.28 | **0.72×** | 64.39 → 58.28（**−9.5%**） |
| warm_all_hit | 82.26 | 55.34 | **0.67×** | 61.92 → 55.34（**−10.6%**） |
| full_cache_75pct | 93.21 | 120.12 | 1.29× | 129.29 → 120.12（−7.1%） |

**分步数据**（echo_perf_step 口径，Σ步 max 跨轮中位，单位周期；括号内为 round-3
同项（§9.14），未入 round-3 Top5 的项无括号）：

| 场景（Σ步周期；round-3） | Top1 | Top2 | Top3 | Top4 | Top5 |
|---|---|---|---|---|---|
| cold_reset（Σ141787；138468） | OpResetRows 34362 / 24.2%（31788） | OpSetUpdate 21892（23424） | OpHitMiss 19717（20420） | OpMissCumsum 17010（15542） | OpAllocPriority 7487 |
| warm_75pct（Σ90018；96418） | OpHitMiss 24356 / 27.5%（26812） | OpMissCumsum 15836（14250） | OpSetUpdate 13602（13964） | OpAllocPop 5954（5860） | OpAllocPriority 5827（5610） |
| warm_all_hit（Σ83277；88094） | OpHitMiss 23756 / 28.8%（26635） | OpMissCumsum 15622（14036） | OpSetUpdate 10850（11084） | OpAllocPop 5382（5278） | OpAllocPriority 5146（5351） |
| full_cache_75pct（Σ197964；210956） | ArgtopkAllRows 96356 / 48.9%（105459，**−8.6%**） | OpHitMiss 26060（28154） | OpMissCumsum 16162（14658，+10% 噪声） | OpSetUpdate 14132（14241） | OpFreeRelease 8038（7712） |

**验证结论**：

1. **round-4 验证通过，残余风险兑现为安全**：audit/golden 全过。warm 两场景
   −9.5%/−10.6%，**超出白付清理 −3~6% 的预期**；其中 OpHitMiss 自身降
   −2.5~2.9K（26812→24356 / 26635→23756），假说（**待验证**）：prologue dcci
   移除后，Step1 的 htd 读命中了 restore 拷贝留在 D-cache 的行。
2. **P-E 收益重校准（重要教训）**：Argtopk 105.5K→96.4K，实际 −9.1K，**远低于
   静态审计预期 −30K**。模型错误：把 pass1 hist 的 GM 读建成 64 次 × ~750cyc 的
   串行依赖链，但 load 地址（`i += G` 步进）与加载数据无关，硬件 scoreboard 可
   并行挂起多个 load——真实成本是**吞吐受限**而非延迟链（方法论教训录入 §7.11）。
   **Argtopk 现为"栅栏地板（8 栅栏/行）+ pack GM 吞吐 + merge + emission"的弥散
   结构，无单一主导相位**。边际规律：round-3 batch1 微观优化 −6K、[P-E] −9K，
   Argtopk 内部每次微观优化的预期收益只有 ~5-10K。
3. **§9.14 open question 1 关闭**：SIMT 回到 75-93 μs（§9.10 水平）→ 上一轮
   SIMT ~2× 确认为设备占用干扰。**本轮比值可信：warm 0.72×/0.67×（Echo 领先，
   warm 已进入 50-60 μs 目标档）、cold 1.25×、full 1.29×**。
4. **Event 失真依旧**：本轮 warm/full 的 Event−内核差 ~600 μs vs cold ~140 μs，
   规律未明（§9.14 open question 2 维持开放）→ 维持"只用 kernel 级 P50 + 分步
   周期"的口径（§7.8）。
5. **距 50-60 μs 的定量账**：full 现 198K 周期（120.12 μs @ ~1.65 GHz，可用
   Σ197964 ÷ 120.12 μs 反推验证），目标 50-60 μs ≈ 80-95K 周期；**即使 Argtopk
   （96.4K）归零余 101K ≈ 65 μs——不动结构到不了目标，v2（维护序，§10.3②）
   必要性坐实**；v2 落点估 ~75 μs，50-60 μs 还需 v2 之上叠加 pipeline 压缩
   （HitMiss 26K / MissCumsum 16K / SetUpdate 14K / FreeRelease 8K / misc 37K）。
6. **int16 hist 扩 G=128 候选撤下**：曾评估把 hist 切片压 int32→int16 以在 UB 内
   将 G=64 翻到 128、缩短 pass1 hist 的 GM 读链——[P-E] 后 pass1 已 UB-fed（读
   key[]），该方案收益收缩而 merge 归并成本随 G 线性变贵，评估由正转负。

### 9.16 round-4b：emission 尾栅栏去重（2026-10-19，commit 05046f2，NPU 待验证）

两处改动均不改变任何输出（checks 逐位一致）；行号按当前 HEAD（05046f2）核对。主会话
已逐 hunk 审查通过、已提交推送；文档同步曾按 owner 指示暂停，本节为恢复后补记（commit
信息里预告的 §9.16/§6.7/§3.5/§5.1/§8/§11 本轮全部补齐）。

- **[改动 1] OpArgtopkRow 函数尾栅栏仅 GM 路径保留**（`echo_lru_kernel.h:612-624`）：
  emission 尾部结构变为"尾栅栏 1（`:612`，emission 块内）+ GM-only dcci（`:615`）+
  `if (freeIdxIsGm) asc_syncthreads();`（`:624`）"。UB 路径下 dcci 被跳过，连续
  第二个 syncthreads 之间无任何排序需求。**安全性**：尾栅栏 1 已完成 emission 的
  key 重扫读 + freeIdx 写 → 之后一切（下一行 pack 写 key / OpFreeRelease 读
  freeIdx）的排序；尾栅栏 2 的唯一作用是 GM 路径 dcci 三明治的后半（t0 dcci 后
  全员再同步，后续读者排在 dcci 之后）；下一行无论如何有新同步点（fsClamp==0
  早退 barrier `:422-425` 或 pack 后 B0 `:443`），行循环栅栏配平不变；
  `freeIdxIsGm` 是全块一致的函数参数（ArgtopkAllRows 按 `!freeIdxInUb_` 逐行传入，
  `:1267-1269`）→ 分支统一，无栅栏发散。
- **[改动 2] Process Step2.SyncGrid 槽 UB 路径整段跳过**（`echo_lru_kernel.h:1158-1164`）：
  `PerfBegin(); if (needFree && !freeIdxInUb_) SyncGrid(); PerfEnd();`——"末行
  freeIdx 写 → OpFreeRelease 读"已由 ArgtopkAllRows 末行函数内尾栅栏 1 提供（全员
  经过该栅栏才离开 ArgtopkAllRows）；GM 回退路径保守保留完整 SyncGrid（大 batch
  形态不在基准覆盖，不冒险）；打点槽原位（UB 路径记录 ≈0 周期，同 C1 惯例）。
  条件块统一：needFree 为 C1 块统一标志、freeIdxInUb_ 为 Init 后不变成员
  （`:1069-1079`）→ 无栅栏发散。
- **栅栏计数**：UB 路径 7/行（B0 1 + 2 趟×2 + emission 扫描 1 + 尾 1），GM 路径
  8/行（尾 2）——代码头注释（`:395-399`）、§6.7 表、§3.5/§5.1 描述已同步。
- **验证**：checks 172 项通过（输出逐位不变）；**本机未编译**——2026-10-16 起新
  工作流本机只跑 checks（§8.1 末"本机/服务器分工"），bisheng 编译与 NPU 验证在
  服务器侧由 owner 执行（pull → `build.sh` → compare.py audit →
  profile/perf_step）。
- **预期（待验证）**：full_cache 每次调用省 2 个 512 线程栅栏（每行尾栅栏 2 一个 +
  Process 槽一个；基准 rowsPerCore=1 即 1 行/核），按 §7.3 栅栏成本量级估 −4~8K
  周期 → full_cache 120.12 → **~115-118 μs**；warm 不受影响（needFree=0，C1
  整段跳过）。

### 9.17 v2：维护序重构（2026-10-19，fe10f91/692c1d7 上板全过）

**动机**（§2.2 边际成本账）：v1 full_cache 的 ArgtopkAllRows 96.4K 周期（48.9%）
是结构性亏损——O(B) 基数选择每步重付；即使 Argtopk 归零也余 ~101K ≈ 65 μs，
50-60 μs 目标不动结构不可达。

**设计**（"SIMT 做对的那半 + Echo 做对的那半"）：
- 状态 `pri` 删除 → `lru[rows, B+1]`（秩→槽位，有效区 [0, occ)、occ = B−avail、
  尾部恒 0）；htd/dth/free/avail 全不变；fifo 成死状态（+2/步合并进
  OpFifoAvailUpdate，golden 语义不变）。
- flags workspace `[R][B+1]`：bit0=本步 hit（OpHitMiss 散射纯 store）、bit1=淘汰
  /spec 移除（OpEvictScanRow）；prologue 全量清零 + SyncGrid（audit 污染防御 +
  同核 WAW 写缓冲排空）。
- **OpEvictScanRow**（替代 7 栅栏 Argtopk）：位置序扫 lru 跳过本步 hit（v1
  protect 等价物）取前 fsClamp 个非 hit 槽 → freeIdx（输出序 = (pri,slot) 字典序，
  与 v1 逐位一致）；O(occ) 平扫 3 栅栏/行；padding/尾零语义同 v1。
- **OpLruRebuildRow**（替代 OpAllocPriority）：每步必执行（"每种情况都维护"），
  产出 新 lru = [幸存者旧序] ++ [hit 槽号升序] ++ [分配槽号升序]，与 v1 步末
  (pri,slot) 稳定排序逐位等价（幸存者 pri 严格分层 + 同层槽号平局）。五相位：
  A staging（flags/lru GM→UB）/ B aAt 散射+位置域分类扫描 / C 槽域联合扫描
  （hit|alloc 前缀 → hExcl/aExcl 秩）/ D 散射写回 GM / 尾部处理；5 栅栏/行
  （B1 staging、B2/B3 扫描内部、B3.5 前缀存储→散射读、B4 行间复用）。
- **尾零删除**：非 spec 路径尾零数学冗余——K/H/A 散射目标恒 < occNew；真实管线
  fs = max(nm−avail,0) ≤ nm ⟹ occNew ≥ occOld ⟹ 尾区 ⊆ 旧零尾（restore/reset/
  上一 invocation 不变量归纳）；spec 路径 avail 动态变化保留尾零（specOn 门控）。
- 步骤 20→18（ECHO_PERF_NUM_STEPS 28→26）；schema `pri`→`lru`；workspace
  +R×(B+1) flags、删 argtopk 诊断区；UB 布局：[0,128) scanScratch（Step1 归约+C1
  与 Step5 MissCumsum 的 v1 已验证序列）+ 256 对齐的 flags/aAt/lru/hExcl/aExcl
  （各 B+1）+ freeIdx + rbScan 段（Step2/Step4 扫描独立缓冲）。
- checks 整体重写为 **50 项**（v1 pri 语义独立 golden vs v2 链逐位跨验证、
  尾零 spec 门控、EvictScan/Rebuild 各 15/11 用例）。

**上板结果**（kernel 级 P50，空闲设备；v1 对照为 05046f2 数字）：

| 场景 | SIMT | Echo v2 | Echo/SIMT | vs v1 |
|---|---:|---:|---:|---:|
| cold_reset | 75.32 | 111.73 | 1.48× | 94.16 → +19% |
| warm_75pct | 80.88 | 79.18 | 0.98× | 58.28 → +36% |
| warm_all_hit | 82.03 | 77.25 | 0.94× | 55.34 → +40% |
| warm 两场景 71/73/78 → 失去 v1 领先、与 SIMT 持平 | | | | |
| full_cache_75pct | 93.68 | **104.06** | **1.11×** | **120.12 → −13%** |

分步（周期，最慢核中位；频率交叉验证 Σ/μs ≈ 1.6 GHz，Event 换算照旧不可用）：

| 步 | cold | warm75 | warm_all | full |
|---|---:|---:|---:|---:|
| OpLruRebuild | 34.8K | 36.9K | 37.2K | 45.8K |
| OpEvictScan | — | — | — | 21.7K |
| OpHitMiss | 21.0K | 24.8K | 25.1K | 26.2K |
| OpMissCumsum | 17.5K | 16.9K | 16.2K | 16.8K |
| OpSetUpdate | 23.4K | 14.8K | 11.3K | 14.7K |

**warm 回退的账**：v1 warm Σ90K → v2 Σ118K，差额 +31K ≈ Rebuild 的 37K（减去
删除的 AllocPriority 5.6K）——回退全部来自每步重建的固定开销（flags staging
4097 GM 读 + 槽域 O(B) 扫描 + 5 栅栏），与场景工作量弱相关（三 warm 场景
Rebuild 几乎等值）。这是 v2 的"维护税"，方向上与 SIMT 的 O(C) 维护同价；
常数可再砍（§9.19/§10.3），O(B) 地板是设计的一部分。

### 9.18 事故：VEC SIMT 栈溢出（2026-10-19，fe10f91 修复）

v2 首次上板 cold_reset 即 507035（device error type 3）trap。根因：**bisheng
把 Process 全部内联 Op 编成单一巨型栈帧（= 内联函数溢出槽之和，kernel 入口
一次性分配），超过设备每线程栈硬上限**（设备日志 errcode 354 "The VEC SIMT
stack overflows"；v1 帧恰在限下，v2 +2 Op 越界）。修复：全部 Op 函数
`__attribute__((noinline))`（帧峰值 从"求和"变"取最大"）；普通构建擦线通过后
打点构建（ECHO_PERF_STEP 多几行代码）再次越界，最终统一 noinline 后余量充足。
教训与完整排查记录见 §7.12 与 `docs/ascend_vec_simt_stack_overflow.md`。

### 9.19 v2-opt 第 1 轮：nm=0 快路径 + 末行免 B4（2026-10-19，NPU 待验证）

分步动机：Rebuild 固定开销主导（§9.17），两项最低风险削减：
- **opt-1 nm=0 快路径**：hasAlloc 为假时跳过 aAt 清零/散射/槽域扫描 alloc 字段
  /aExcl 存储（aExcl 届时无读者——A-append 的 j<nm 不执行，语义不变）；
  warm_all_hit 等零分配步省 ~4097 UB 写 + 2×4097 读。
- **opt-2 末行免 B4**：B4 仅保护 UB 段行间复用；末行后本 invocation 无这些段的
  读者（Step5 用 scanScratch [0,128) 不同段；lru 写由 kernel 边界保证）。
  rowsPerCore=1 的基准形态每次调用直接省一道栅栏。lastRow 块一致参数无发散。
预期 warm Rebuild 37K → ~30K（−5~8K）；EvictScan 的尾栅栏**保留**（它排序
freeIdx 写 → OpAllocPop 读，跨 Op 依赖，与 B4 性质不同）。checks 50 项同步更新
（组 2 首例 lastRow=0 覆盖 B4 执行路径）。

**上板验证（同日）**：Rebuild cold 34.8→31.8K / warm75 36.9→33.2K / warm_all
37.2→**30.5K**（opt-1+2 双收益，降幅最大，符合预测）/ full 45.8→44.2K。
kernel P50：warm_all 77.25→74.02（0.91×）、warm75 79.18→77.59（0.96×）、
full 104.06→100.68（1.08×）、cold 111.73→108.19。快照 tag `v2.0`。

### 9.20 opt3 FifoAvail 预置摊销：实测净回退与回退记录（2026-10-19，48610c0 → 266ed64 revert）

**尝试**（commit `48610c0`，已 revert）：基于 "6.2K = t0 单次冷 GM RMW 链延迟、
store 火 后即忘" 的假设，把 fifo/avail 终值早算进 Step1 OpHitMiss 的 t0 段
（fifo 冷读与既有 avail 冷读并行在途）存 UB stash，Step4 尾改全员摊开纯存储。
设计前提、语义等价证明与 checks 双链验证详见该 commit message（git 历史保留）。

**实测**（上板，v2.0 对照；分步为 20 repeat 最慢核中位）：

| 指标 | v2.0 | opt3 | Δ |
|---|---:|---:|---:|
| HitMiss cold / warm75 / warm_all / full | 21.0 / 24.8 / 25.1 / 26.2 K | 25.2 / 27.7 / 28.9 / 28.9 K | **+4.2 / +2.9 / +3.8 / +2.7 K** |
| FifoAvail warm75 / warm_all | ~6.2 K | 6.5 / 6.9 K | 持平略差 |
| P50 cold / warm75 / warm_all / full（干净跑） | 108.19 / 77.59 / 74.02 / 100.68 | 109.8~111.8 / 79.6~79.7 / 75.1~76.0 / 102.8~103.6 | **+1~3.6 μs 全场景同号** |

净回退 ~+3K 周期，`git revert 266ed64` 回到 v2.0 基线（代码与 checks 均还原）。

**机理修正（负结果的核心价值，三条实测微架构结论）**：
- **H1 同 warp 冷 GM 访存近似串行，~3-4K/次**：opt3 恰好往 t0 链上加了 1 次
  独立冷读（fifo），HitMiss 四场景恰好 +~1 次冷读的量级（若独立冷读可重叠
  在途，应 ≈0）。交叉验证：opt2 删单道 B4 栅栏实测 −3.7K（同量级）。
- **H2 冷行 store 非 fire-and-forget**：对本 invocation 首触的冷行，store 同样
  支付 ~3K 行填充往返——opt3 前（1 冷读 + 1 冷写）与后（2 冷写）的 FifoAvail
  都是 ~6K ≈ 2 次冷触。冷读被搬走后冷写补位，成本被移动而非消除。
- **块内推论：栅栏同步一切**——把冷触"搬"给其他线程/其他步骤只是换打点槽位
  记账，不减总时间（Step1 尾/Step3 尾 SyncGrid 会等最慢线程）。真正减时间的
  只有三件事：①减少冷触总数 ②增加并行 warp（t1024）③跨 Op 融合（消栅栏 +
  共享 staging）。
- **歧义注记**：HitMiss 的 +Δ 无法完全区分 "fifo 冷读串行" 与 "stash UB store
  成本" 两种成因（都拟合实测）；t1024 消融（`ab4fb5b`）与后续微探针可分辨。

**判定**：FifoAvail ~6K = 2 次冷触（fifo/avail 各一次行填充）≈ **该步地板**；
除非 fifo/avail 状态合并为单数组（host 布局 + golden 改动，省 1 次冷触 ~3K，
记远期）。CLAUDE.md 原 "6.2K→~1K" 估算的 "store 火 后即忘" 前提在设备上证伪。

**后续影响**：t1024 重测动机成立（H1 ⇒ HitMiss 在 t512 下每 warp 4 轮 gather
已近地板 ~22-25K，t1024 减半轮数是唯一大杠杆，§10.3①）；Rebuild 方向按
H1/H2 重排（§10.3②）；OpFifoAvailUpdate 移入地板项清单（§10.2）。

### 9.21 t1024 消融：v1 结论在 v2 复现，t512 保持（2026-10-20，ab4fb5b → f2acb7c 回退）

**动机**（§9.20/§10.3①）：v1 "t512 全场景甜点" 结论产生于 Argtopk（7-8 栅栏/行）
主导的构成；v2 换维护序后 Rebuild（4 栅栏/行）+ HitMiss 主导，且 H1 预测
t1024 可把 HitMiss 每 warp gather 轮数减半。一行改动重测（plugin 公式
ceil(topk/4)→ceil(topk/2)，topk=2048 → 1024 线程）。

**实测**（P50 两跑一致；分步 20 repeat 最慢核中位；v2.0=t512 对照）：

| 指标 | v2.0 (t512) | t1024 | Δ |
|---|---:|---:|---:|
| P50 cold / warm75 / warm_all / full (μs) | 108.2 / 77.6 / 74.0 / 100.7 | 136.7~140.3 / 102.6~105.9 / 99.4~102.9 / 136.1~137.6 | **+25~37 全场景惨败** |
| Rebuild (cold/warm75/warm_all/full) K | 31.8 / 33.2 / 30.5 / 44.2 | 47.2 / 48.9 / 46.7 / 55.6 | +11~16 |
| HitMiss K | 21.0 / 24.8 / 25.1 / 26.2 | 29.5 / 31.7 / 32.0 / 31.8 | **+6~8.5（未降反升）** |
| EvictScan (full) K | 21.7 | 29.8 | +8.1 |
| MissCumsum (warm75/warm_all) K | 16.9 / 16.2 | 13.0 / 13.7 | **−2.5~3.9（2 元素门兑现）** |
| SetUpdate / FifoAvail K | 14.8 / 6.2 (warm75) | 16.2 / 7.7 | +1.4 / +1.5 |
| ResetRows (cold) K | 33.1 | 34.0 | ~持平 |

**三条结论**（微架构模型进一步收敛）：
1. **栅栏随线程数翻倍**：t1024 栅栏 ≈ 6-8K/道（§7.3/kernel.h:57 的 v1 实测
   复现）vs t512 ~3.7K。Rebuild 回归量 (+11~16K) ≈ 4 道 × Δ~3.3K 闭合；
   EvictScan (+8.1K) 同型。全 kernel ~11-15 道栅栏 ⇒ **t512 下栅栏已占
   总时间 ~1/3（warm Σ116K 中 ~41K）**——栅栏削减（融合/消解）是最大杠杆。
2. **gather 是核级吞吐受限，非每 warp 延迟串行**：t1024 把每 warp gather
   轮数 4→2，HitMiss 未降反升 (+6~8K = 栅栏增量+调度开销)——2048 个随机
   sector 的总吞吐在 512 线程已打满，加线程只加栅栏与调度成本。§9.20 H1
   修正为：**单发冷触（少量在途）串行 ~3-4K/次；批量 gather 达核吞吐上限，
   加 warp 无益**。
3. **ResetRows 持平佐证带宽地板**（§10.2 分类正确，与线程数无关）。

**处置**：`f2acb7c` 回退公式，t512 保持为正式形态（v1/v2 双证实）。
MissCumsum 的 −2.5~3.9K 是唯一正项但远不抵损失，无 salvage 价值。
HitMiss 在 t512 的优化空间确认为收尾级（~4-7K，§10.2）。

### 9.22 v2.1：单趟分类 + 平局松弛 + 尾段追加（2026-10-20，NPU 待验证）

**动机**（§9.21 结论的直接推论）：v2.0 的 EvictScan（Step2）与 Rebuild 的位置域
相位（Step4）是**对同一条 lru 位置序链的两遍同型扫描**——EvictScan 为取受害者
（计数扫 + 发射扫），Rebuild 为重建分类，各自 staging flags、各带各的栅栏。拆成
两遍是 Step 的顺序限制（中间隔着 Step3 AllocPop），不是元数据限制。§9.21 实测
栅栏占总时间 ~1/3（warm ~41K/116K），跨 Op 融合（消栅栏 + 共享 staging）是最大
杠杆；cold 的 Rebuild 31.8K 几乎全是空位置域上的槽号域扫描 + staging + 栅栏
白付。叠加 §10.3 原决策点②获批（owner 2026-10-20）：hit/alloc 组内部排列属同
年龄组，LRU 语义无差别——Rebuild 的槽号域秩扫描（aAt/hExcl/aExcl 相位）整体
失去存在理由。

**设计**：
- **OpClassifyEvictRow**（替代 OpEvictScanRow，Step2，**每步每行必跑**——fs=0
  也要维护序，v2.0 的"fs=0 跳行"删除）：单趟位置序扫描，每位置按 flags 分类
  （bit0=hit / bit1=spec 移除 / 其余=候选），PackedBlockScan 打包
  (kCnt|hCnt<<16) 前缀，一次发射三产出：
  ① freeIdx：候选全局序 < fsClamp 者为受害者（位置序，与 v2.0 逐位一致）+
    padding [B+1,..) + 尾零（三条写路径全部 fsClamp>0 门控，fs=0 行零写）；
  ② K 段（幸存者，散射目标 = 候选全局序 − fsClamp，旧序保持）+ H 段（hit，
    基址 totalK + hit 全局序 = **旧位置序**，平局松弛）直接进 lruGm；
  ③ rbBase = totalK + totalH，t0 写 numHitsGm[row]（v2.0 起的死状态复活为
    Step2→Step4 传递槽，workspace 布局零漂移）。
  每行 3 道栅栏（staging→scan、scan 内部、尾——末行也不可省，freeIdx 有跨 Op
  读者）；staging 吸收旧 Rebuild 的 lru GM→UB（散射写 lruGm 前全体读入，防同
  数组跨线程读写竞争）；**occ==0 快路径**（cold）：零栅栏零 staging 零散射，
  仅防御性保留 fsClamp>0 的 padding/尾零（复刻旧 EvictScan；管线内 occ=0 ⟹
  fs=0 不可达）。
- **OpAllocAppendRow**（替代 OpLruRebuildRow，Step4）：退化为一层循环——分配槽
  按 rawSlots j 序（**分配序**，平局松弛）追加到 lru[rbBase, rbBase+nm)。零栅栏、
  零 staging、零扫描：A 段与 K/H 段区间不重叠（[base,base+nm) vs [0,base)）；
  rawSlots/rbBase 可见性由 Step3 尾 SyncGrid（dcci）保证（Op 内不另设栅栏）；
  非 spec 尾零数学冗余论证沿用 v2.0（occNew ≥ occOld ⟹ 尾区 ⊆ 旧零尾），
  spec 尾零保留。
- Process：Step2 = ClassifyEvict（无条件）+ [needFree] SyncGrid(GM 回退)/
  FreeRelease/FreePush；Step4 = AllocAppend + FifoAvailUpdate。needFree 语义
  收窄为"free 流水线跳过"（fs=0 行不发 freeIdx，AllocPop 恒走老栈）——v2.0
  时代它门控整个 EvictScan。新重建序（golden 契约）：
  **新 lru = [幸存者(保持旧序)] ++ [hit(旧位置序)] ++ [新分配(分配序)]**；
  ECHO_PERF_NUM_STEPS=26 不变（非 spec 18 步，步名换）。

**布局变更**：UB 删 aAt/hExcl/aExcl 三段（槽号域秩扫描的载体）。新序
scanScratch[0,128) → flags@256 → lru@256 → freeIdx（不额外对齐）→ rbScan@256，
基准形态（B=4096 ⟹ B+1=4097；rowsPerCore=1, topkDim=2048 ⟹ freeIdx 2048；
rbScan 2112）逐段实算：flags [256,4353) → lru [4608,8705) → freeIdx [8705,10753)
→ rbScan [11008,13120)，**峰值 13120 ≤ 32768**（v2.0 同径实算 26176，省 13056）。

**语义论证**：
- 淘汰集合与 v1/v2.0 恒等：幸存者 pri = 旧位置序（互异无平局）且严格小于 hit
  的 f / alloc 的 f+1（严格分层）⟹ v1 (pri,slot) 字典序前 fs 个非 hit ≡ 旧序
  前 fs 个候选（候选 = 非 hit 非 bit1）；fs ≤ 候选数由 nm+hits=topk ≤ capacity
  保证。check_common_semantics 的淘汰集合从初始历史独立推导，一字未改。
- 全部输出（current_slots/miss_host_pos/miss_alloc_flat/miss_mask 与 htd/dth/
  free/avail/fifo 终态）不依赖 lru 组内排列：hit/miss 由 htd 判定、分配由 free
  栈决定，lru 只在"下一周期的年龄序"意义上被消费——组内排列自由度正是平局
  松弛的安全边界。token-0 保留语义不变（reference_echo 的 htd 法判 miss 原样，
  仅步末 lru 拼接规则换）。
- bit1 语义收窄：受害者不再标 bit1（融合后无读者——旧读者是 Rebuild 的分类）；
  spec 移除标记（Step0.5 写）照读照跳。非 spec 下 [0,occ) 无 bit1 条目，新旧行为
  恒等；spec 下 bit1 槽已释放，不当候选反而正确（spec 无基准覆盖，按此维护）。

**旋钮清理**：ECHO_V2_SKIP_REBUILD + 9 个 ECHO_V2_RB_* 随旧 Rebuild 删除
（build.sh 不再传参；CMakeLists 的 option 默认关，不影响构建）；旋钮结论本就不
跨二进制（§9.18）。ECHO_V2_DEBUG_CUTOFF 保留（阶段 6=Step4 后仍成立）。
checks 重写为 **54 项**（v2.2 增至 56，§9.24；ClassifyEvict 21 / AllocAppend 8 / 跨验证 14 / HitMiss
3 / MissCumsum 7 / AllocPop+FifoAvail 1）；compare.py 的 reference_echo 改旧序
直接法（删 pri 仿真，miss 判定仍 htd 法）。

**预期收益**（口径：op 单价 ~300 cyc 由 v2.0 分步反推、栅栏 3.7K/道 t512 实测；
ClassifyEvict ≈ 旧 EvictScan + 旧 Rebuild 的位置域相位 − 1 道栅栏 − 槽号域相位，
AllocAppend ≈ 1-2K）：

| 场景 | v2.0 P50 | v2.1 预期 | 依据 |
|---|---:|---:|---|
| warm_all_hit | 74.02 | ~67-69 | Rebuild 30.5K → ClassifyEvict ~20-23K + Append ~1K |
| warm_75pct | 77.59 | ~68-71 | Rebuild 33.2K → 同上（warm 的 EvictScan 本被 needFree 跳过） |
| full_cache_75pct | 100.68 | ~79-83 | EvictScan 21.7K + Rebuild 44.2K = 65.9K → ClassifyEvict ~25-30K + Append ~2K |
| cold_reset | 108.19 | ~87-90 | Rebuild 31.8K → occ==0 快路径 ≈0 |

**风险与预案**：① Step2 的 lru 散射紧邻 OpFreeRelease 读（UB freeIdx 路径无
SyncGrid）——同型散射（v2.0 Rebuild phase D 后接 MissCumsum 读）已上板验证
安全，但"重 UB 写后紧随访存"曾是 §9.18 的 trap 模式；若上板 507035（先 grep
errcode 确认 354），预案 = Step2.SyncGrid 槽改无条件（一行）。② ClassifyEvict
的单价估算若偏乐观（lru staging 的 GM 读为主），warm 收益缩水——分步对照
（ECHO_V2_DEBUG_CUTOFF 或 perf_step）可定位。③ ~~occ==0 快路径的 freeIdx 防御性
写无尾栅栏~~（审查修正，2026-10-20）：快路径收窄为 `occ==0 && fsClamp==0`（管线
内唯一可达组合）；fsClamp>0 的 occ==0 组合落入正常路径（空扫描循环 + padding +
尾零 + 全套栅栏，逐位复刻旧 EvictScan 边界行为）——原实现把该组合放在无栅栏的
快路径内联，freeIdx (UB) 写 → 后续 Op 读缺 B3 排序，属 out-of-invariant 隐患，
主会话审查时修正。

**上板验证（2026-10-20，commit `6188d3b`，同日）**：checks 54 项 PASS、四场景
golden 逐字段全过、无 trap（风险①未触发）。

| 场景 | SIMT | Echo v2.0 | Echo v2.1 | v2.1/SIMT | vs v2.0 |
|---|---:|---:|---:|---:|---:|
| cold_reset | 74.54 | 108.19 | 99.59 | 1.34× | −8.0% |
| warm_75pct | 80.72 | 77.59 | **71.36** | **0.88×** | −8.0% |
| warm_all_hit | 81.54 | 74.02 | **69.33** | **0.85×** | −6.3% |
| full_cache_75pct | 93.23 | 100.68 | **82.12** | **0.88×** | **−18.5%** |

**首次 warm×2 + full 三场景全部 ≤ SIMT**；full 首破 100。预测兑现度：warm_all
69.33（预测 67-69）✓、warm75 71.36（68-71）✓、full 82.12（79-83）✓；cold
99.59（预测 87-90）偏高——快路径成立但未计入 wrapper 的 avail 冷读 + rbBase
冷写（occ=0 时 ClassifyEvict 槽实测 10.7K ≈ 冷触成本，非分类工作）。

分步（周期，v2.1；对照 v2.0 同口径）：

| 步 | cold | warm75 | warm_all | full | 对照 v2.0 |
|---|---:|---:|---:|---:|---|
| Step2.OpClassifyEvict | 10.7K | 19.4K | 21.4K | 29.0K | full: EvictScan+Rebuild 65.9K → **−56%**；warm: Rebuild 30.5~33.2K → −35~40%；cold: Rebuild 31.8K → 10.7K |
| Step4.OpAllocAppend | （Top5 外，≤6.4K） | | | | 预测 ~0.5-1K ✓ |
| Step1.OpHitMiss | 22.4K | 23.7K | 23.2K | 23.9K | −1~2.5K（代码未动，伴生流量减少） |
| Step5.OpMissCumsum | 17.3K | 16.5K | 16.5K | 16.4K | 持平 |
| Step5.OpSetUpdate | 21.6K | 14.3K | 10.8K | 14.2K | 持平 |
| Step4.OpFifoAvailUpdate | — | 6.4K | 6.6K | — | 持平（地板确认，§9.20） |
| Step2.OpFreeRelease (full) | — | — | — | 7.1K | — |

快照 tag `v2.1`。微架构口径注记：本轮频率标定行（Event P50 基准）依旧不可信
（§7.8），Σ 步 max 中位与 kernel P50 交叉验证 ≈ 1.4-1.5 GHz 量级自洽；分步
绝对值与 P50 的对账问题见 §10.3①（msprof + 时间戳升级计划）。

### 9.23 v2.1-prof：打点升级（entry + 绝对时间戳 + 标定修正 + HTML 报告）（2026-10-20，NPU 待验证）

一轮把 §10.3① 精确打点的三个悬案收口：频率假数、对账缺口、核间偏差不可见。
算法代码零改动（全部打点相关改动在 `#ifdef ECHO_PERF_STEP` 内，正式构建零行为
变化），核心是**语义改造**而非加打点。

**动机**：

1. **频率标定关闭**：msprof 首轮实测主频 **1.585 GHz**（aiv_total_cycles 是全核和：
   4.22M cyc / 32 核 / 83.5 μs；DVFS 无降频，§10.3①）。echo_perf_step 旧"频率标定"
   行用每轮 Σ步周期 / Event 微秒反推 → 输出 141-245 MHz 假数（Event 含 ~600 μs 级
   固定开销，§7.8）。本轮废弃该行：默认常数 1.585 GHz；`--p50-us` 给外部锚点
   （profile_kernel.sh 的 kernel 级 P50）时改用 ΣΔ/P50 标定并打印两档对照；Event P50
   仍测量打印但注明 §7.8 不可信、不参与任何换算。
2. **对账缺口闭合**：v2.1 分步 Σ 121.7K 只占单核 ~131.5K cyc 的 94%（msprof，
   full_cache），缺口 ~10K = 未打点 prologue（OpZeroFlags 全量清零 4097×32 int32 +
   prologue SyncGrid）+ 进出/槽间隙 ~6 μs。本轮 prologue 纳入打点（2 槽）+ 间隙
   归属下一步，分步覆盖预期 94% → ~100%（Init/kernel 进出仍在 slot 0 之外，残余
   缺口即该项，直接可见）。
3. **核间偏差首见**：旧差值口径每核只写自己的步差，host 无法看核间不均衡；绝对戳
   + host 差分后，每步 32 核 max−min（skew）成为一等统计量（"每条横线一个核"的
   专家参考形态 → prof_report 热力图）。

**设计（槽位语义）**：`stepCyclesGm_`（步差值）→ `stepAbsGm_`（t0 绝对 clock 低
32 位，int32），`ECHO_PERF_NUM_STEPS` 26→29：

| 槽 | 内容 |
|---|---|
| 0 | entry：Process 进入（PerfEntry 写；Init/kernel 进出开销在其外） |
| 1 | prologue.OpZeroFlags 末（spec 的 OpFillZero 不打点 → 耗时并入槽 2 差分） |
| 2 | prologue.SyncGrid 末 |
| 3..20 | 非 spec 18 步（Step0×5 / Step1×2 / Step2×4 / Step3×2 / Step4×2 / Step5×3，顺序照旧） |
| 21..28 | spec 8 步名（**陈列位**，见下） |

- **PerfBegin/perfLast_ 删除**，PerfEnd 只在步末由 t0 读一次 clock：`++perfStepIdx_;
  stepAbsGm_[bn*N + idx] = (int32)(uint32_t)clock()`；slot 0 由 PerfEntry 先写（不
  推进计数器，首个 PerfEnd 恰落 slot 1）。步耗时 = t[i+1]−t[i]（host 侧 numpy
  uint32 视图回绕差分），**步间间隙自然归属到下一步**——anyReset 扫描（全线程冗余
  读 resetMask）等此前落在打点外的开销，现在计入下一步差分。
- **回绕安全**：单次 invocation 跨度 ~160K cyc @1.585 GHz ≪ 2^31。哨兵 -12345 只认
  等值——绝对戳 int32 高位可负，v2 差值语义的"<0 即异常"防御在新语义下会把一半
  合法戳误判为异常，已废止。
- **spec 槽位约定（规格核对结果）**：spec 8 步在 kernel 内的执行位置沿用 v2 起的
  现行约定——插在 Step0.SyncGrid 之后、Step1 之前；顺序计数器语义下实际落位
  slot 8..15，后续非 spec 步顺延 16..28。名表 21..28 只是 8 个 spec 步名的陈列位
  （非 spec 构建恒为哨兵）。echo_perf_step 读回时校验"槽 0..20 全写且 21..28 全
  哨兵"，检测到 spec 构建直接报错（spec 分步报告属 §10.3⑥ 未来工作）。
- **ECHO_PERF_DRAIN 旋钮**（build.sh + CMakeLists，默认 OFF，需配 ECHO_PERF_STEP=ON，
  产物进 build_v2dbg/）：定义后 PerfEnd 在读 clock 前先 SyncGrid()（syncthreads+dcci
  排空写缓冲——专家 AddTimePoint 方案 PipeBarrier<PIPE_ALL> 在纯 SIMT 头树中的等价
  物），量"真实边界 vs 记账边界"的偏差。强扰动（每标记 +一道栅栏价），结论仅在本
  二进制内有效（旋钮=不同二进制的既有教训）。
- **host 侧（echo_perf_step.py 重写）**：统计 = 最慢核中位（max over blocks →
  median over repeats）+ **skew**（每步 32 核 max−min 的中位）；对账行 = 最慢核跨度
  中位（entry→末戳；逐块望远镜恒等 = 该核各步之和，对账以跨度为准，跨核逐峰求和
  仅参考）+ Event P50（仅打印）；`--matrix <prefix>` 每场景导出
  `<prefix>_<scenario>_abs.csv`（行=块，列=29 槽原始绝对戳）与
  `<prefix>_<scenario>_steps.csv`（行=块，列=28 步耗时 + entry→末戳总跨度；代表轮 =
  最慢核跨度最接近中位的样本，两 CSV 同轮可互相对照）。

**prof_report.py（新增）**：单文件自包含 HTML 报告器（owner 服务器生成、浏览器打开；
纯 stdlib + 手写 SVG/JS，无外部依赖、无网络请求）。页面：§1 总览（统计瓦片 + 对账
条：打点跨度 vs msprof echo Duration vs 外部 P50 锚点，缺口 = Init/kernel 进出等
打点外开销）→ §2 逐核×逐步热力图（行=核、列=28 步，全局线性顺序色标，cyc/μs 可
切换，悬停数值）→ §3 逐步聚合（P50 条形 + max 同色浅轨道 + min 刻度 + skew 列，
μs 轴）→ §4 msprof op 级表（--op-summary 容错列嗅探，echo 行高亮，不猜单位不换算）
→ §5 原始 CSV 折叠表（= 各图的无悬停/键盘可达等价物）。配色沿用 dataviz skill 的
已验证参考调色板（light：顺序色阶蓝 100..700 单 hue；本机禁跑 node 校验脚本，故
原样引用其验证结论而非自配色）。

**实现落点**：`echo_lru_kernel.h`（PerfEntry/PerfEnd 重写 + Process 打点重排 +
stepAbsGm_ 改名，算法一行未动）、`echo_lru_tiling.h`（N=29 + 槽位注释）、
`checks`（常量断言 26→29）、`build.sh`+`CMakeLists.txt`（ECHO_PERF_DRAIN）、
`echo_perf_step.py`（重写）、`prof_report.py`（新增）。

**遗留登记（主会话审查时已顺手同步，随本 commit 一并合入）**：

- ~~`debug_v2_stepcycles.py` 的 `>= 0` 已写判定在新语义下失效~~（已修，2026-10-20
  审查轮）：哨兵改等值判定（`!= -12345`），docstring/打印文案同步为绝对戳语义；
  其依赖的"前缀性质"（trap 于步 N → 槽 0..N 已写、其后哨兵）在新语义下仍成立。
- ~~`compare.py:299` 注释 "stepCycles（26）" 与 `echo_lru_launch.h` 注释 "每核分步
  周期" 未随本轮更新~~（已修，同上：分别改为 "stepAbs 绝对时间戳区（29）" 与
  "分步打点区（t0 绝对时间戳低 32 位）"；行为无影响——compare.py 的 workspace
  上界每核 73984 元素 ≫ bn×29，launch.h 按符号引用自动扩容）。
- CMakeLists.txt 本轮必须同步（ECHO_PERF_DRAIN 的编译宏管道——bisheng 编译命令在
  CMake 内显式拼装，无通用宏透传；不改则旋钮静默失效）。该文件在主会话下发的改动
  白名单之外，已在实现报告中置顶标注，由主会话审查定夺。

**预期与验证计划**：分步覆盖 94% → ~100%（跨度口径）；prologue ~10K 首次可见且
OpZeroFlags 与 prologue SyncGrid 分离；skew 首次可见；频率悬案关闭（1.585 GHz 常数
两档对照）。NPU 验证待做：四场景 echo_perf_step（--warmup 5 --repeat 20）+ 一轮
--matrix + prof_report 生成 + （可选）ECHO_PERF_DRAIN=ON 对照一轮（结论仅该二进制
内有效）。

**上板验证（2026-10-20，owner 服务器执行，同日）**：四场景 echo_perf_step
（--warmup 5 --repeat 20）+ --matrix/prof_report 轮通过。预期兑现情况：对账闭合
（见下）、prologue 两槽分离且实测**偏大**（预期 ~10K → 实测 13.7-13.9K，直接触发
v2.2）、skew 首见；频率悬案未按预期关闭——翻出更深一层的**跨 run 频率漂移**
（见下）。ECHO_PERF_DRAIN 对照未做（可选项，强扰动结论仅本二进制内有效）。

**P50 复现**（kernel 级 profile_kernel.sh，v2.1-prof 二进制；对照 §9.22 的 v2.1
上板轮——算法零改动，差值 = run 间噪声 + 打点构建形态差，顺带验证"打点构建不改
行为"）：

| 场景 | SIMT | Echo(v2.1-prof) | v2.1 上板轮 |
|---|---:|---:|---:|
| cold_reset | 74.85 | 96.19 | 99.59 |
| warm_75pct | 82.78 | 71.30 | 71.36 |
| warm_all_hit | 81.75 | 68.96 | 69.33 |
| full_cache_75pct | 94.47 | 82.24 | 82.12 |

Echo 四场景差 −3.4~+0.1 μs（cold 最大），同两轮 SIMT 亦漂 +0.2~+2.1 μs——run
间噪声量级，无系统性偏移。

**分步**（cyc，最慢核中位；本轮口径 = 绝对戳差分 + 间隙归属下一步，与 §9.22 旧表
**同物异账**——旧表是 PerfBegin/End 差值、不含步间间隙，故偏小；如 ClassifyEvict
旧 29.0K → 本轮 full 31.1K，非回退。栅栏步数字的读法注意见下文注记）：

| 步 | cold | warm75 | warm_all | full |
|---|---:|---:|---:|---:|
| prologue.OpZeroFlags | 4171 | 4238 | 4034 | 4352 |
| prologue.SyncGrid | 9494 | 9626 | 9345 | 9432 |
| Step0.OpResetRows | 34036 | 1062 | 1069 | 1072 |
| Step0.OpResetDthLru | 7071 | 283 | 282 | 284 |
| Step0.OpResetFree | 3714 | 496 | 500 | 482 |
| Step0.OpResetAvailFifo | 2676 | 336 | 376 | 428 |
| Step0.SyncGrid | 5863 | 286 | 288 | 286 |
| Step1.OpHitMiss | 24340 | 26620 | 26199 | 27761 |
| Step1.SyncGrid | 2786 | 4318 | 4036 | 2894 |
| Step2.OpClassifyEvict | 11668 | 21598 | 23296 | 31128 |
| Step2.SyncGrid | 1872 | 864 | 868 | 956 |
| Step2.OpFreeRelease | 652 | 286 | 294 | 8226 |
| Step2.OpFreePush | 530 | 352 | 342 | 5044 |
| Step3.OpAllocPop | 7945 | 6626 | 6084 | 5977 |
| Step3.SyncGrid | 3466 | 3548 | 3549 | 3536 |
| Step4.OpAllocAppend | 6211 | 5320 | 4836 | 5315 |
| Step4.OpFifoAvailUpdate | 5338 | 5859 | 5326 | 4883 |
| Step5.OpMissCumsum | 18488 | 17880 | 17641 | 17772 |
| Step5.SyncGrid | 5494 | 5419 | 5312 | 5430 |
| Step5.OpSetUpdate | 23205 | 15660 | 12220 | 15491 |

**skew 与对账**（skew = 每步 32 核 max−min 的中位）：HitMiss 3595/5888/5341/6815
（cold/warm75/warm_all/full，下同）——数据相关 gather 长尾（§10.2 地板项的核间
视角，hit 越多 gather 越散）；prologue.SyncGrid ~5.9K ≈ 入口斜坡（核进入 Process
的时间差）+ 清零排空被栅栏吸收；cold SetUpdate 4452。**跨度中位**（entry→末戳，
对账基准）：160240/116156/113432/136046；Σ步max（跨核逐峰求和）：
179019/130677/125896/150748，比跨度虚高 +10.8~12.5%（full +10.8%）——望远镜
恒等式（逐块步耗时之和 == 该块跨度）只对单核成立，跨核逐峰求和把不同核的峰加
在一起，仅参考。**对账闭合**：msprof 单核均值 ~131.5K（§10.3① 首轮）vs 本轮
最慢核跨度 136.0K（full）——94% 缺口消失，残余 = Init/kernel 进出（slot 0 外）+
最慢核 vs 均值差，量级吻合。

**头条结论：prologue 实测 13.7-13.9K，直接触发 v2.2**。OpZeroFlags + prologue
SyncGrid 两槽合计 13665/13864/13379/13784 cyc，占跨度 8.5%/11.9%/11.8%/10.1%
（cold/warm75/warm_all/full）——四场景几乎恒定（与场景工作量无关的固定税），远超
§10.3① 候选登记时按对账缺口估的 ~10K/−5~8K 假设基础。当天定刀 v2.2 flags 世代号
化（§9.24）。

**频率漂移发现（标定口径再修正，重要）**：

- **事故经过**：owner 把 full 的 P50=82.24 传给了 echo_perf_step 全部四场景 →
  只有 full 的 μs 列换算正确（cycles 列不受锚点影响，全部正确）。
- **按各自正确 P50 反推隐含频率**（跨度/P50）：cold 1.666 / warm75 1.629 /
  warm_all 1.645 / full 1.654 GHz——一致高于 msprof 首轮的 1.585（+2.8~5.1%）。
- **逻辑实锤（不可能不等式）**：full 跨度 136046 cyc @1.585 GHz = 85.86 μs >
  P50 82.24 μs——跨度是 kernel 内部区间（entry→末戳），不可能超过 kernel 总
  时长；该不等式代数上等价于"打点 run 频率 > 1.585 GHz"。结论：profile_kernel
  与 echo_perf_step **两 run 频率确差 ~4-5%**，msprof 首轮的"无降频 1.585"只是
  单时点采样，跨 run 不可搬运。
- **系统性偏差方向**：锚点频率 = 跨度/P50，而跨度 < kernel 总时长（Init/kernel
  进出在 slot 0 之外）⟹ 隐含频率天然**低估**真值——真漂移只会比上述 4-5% 更大。
- **操作协议（后续遵循）**：决策一律用 **cycles**（场景内占比/排序不受频率影响）；
  profile_kernel 与 echo_perf_step **背靠背连跑** + **每场景各自锚点**；要根治
  用 msprof 同 run 锚定（aiv_total_cycles / Duration）；Event P50 依旧不可信
  （§7.8）。
- **工具项收口（同批落地，commit `50128d3`，echo_perf_step.py）**：`--p50-us` 支持逗号分隔
  `场景=值` 逐场景锚点（纯浮点 = 全场景统一，旧行为向后兼容；未列出场景回落
  1.585 GHz 档）；漂移探测器三信号——①有锚点场景"跨度@1.585GHz > P50"（不可能
  不等式）打醒目 WARNING（本场景 μs 列仅参考）；②隐含频率偏离 1.585 GHz >2% 而
  未触发①打 NOTE（跨 run 常见量级，μs 等比例缩放，占比/排序不受影响）；③收尾
  汇总各锚点场景隐含频率，离散 (max−min)/min >3% 再提示（run 内频率波动或锚点
  口径问题的信号）。

**打点读法注记（栅栏步的 max 统计天然隐藏等待）**：max 挑的是**最晚到达者**——
它到达时其余核已等完，量到的是它自己那段短穿越；等待成本落在早到的快核上，被
max 丢弃。例：Step2.SyncGrid full 956 ≠ "栅栏便宜"（§9.22 实测同型栅栏 3.7K+）；
prologue.SyncGrid 9432 ≈ 入口斜坡吸收 + 清零排空 + 栅栏地板的打包（skew ~5.9K
即斜坡成分）。解读栅栏步数字必须带着这条；同理，**删一道栅栏的临界路径收益 ≠
槽位数字**（快核的等待不消失、只转移到下一道栅栏——v2.2 的收益估算即按此折算，
§9.24）。

（[v2.2 注] 本节槽位表/名表所述 29 槽为 v2.1-prof 形态；v2.2 删 prologue 两槽后
为 27 槽，见 §9.24。）

### 9.24 v2.2：flags 世代号化 — 删 OpZeroFlags 与 prologue SyncGrid（2026-10-20，commit `284ff71`，NPU 待验证）

**动机**（§9.23 验证轮头条结论）：prologue 两槽（OpZeroFlags 全量清零
4097×32 int32 + 其后 SyncGrid）上板实测 13.7-13.9K cyc、占跨度 8.5-11.9%、四场景
几乎恒定——与场景工作量无关的纯固定税，且远超 §10.3① 候选登记时按对账缺口估的
~10K/−5~8K 假设基础。同日定刀、同日落地。

**设计（值域：bit 语义 → 世代号等值链）**：flags 语义从"每 invocation 全量清零 +
bit0/bit1 位判定"改为**世代号**——hit 槽写 gen、spec 移除槽写 genSpec（=gen+1），
分类改等值链：`v==gen` → hit；`v==genSpec` → spec 移除跳过；其余（窗外）→ 候选。

- **gen 发放**：plugin 进程级静态原子计数器（`g_echoFlagsGen{1}`，每次调用
  `fetch_add(4, relaxed)`）——从 1 起、stride 4，恒 ≡1 (mod 4)；经 tiling 新字段
  `flagsGen` 随 H2D 传入（launch 接口与 GM workspace 布局零改动）。genSpec 在
  Init 用**无符号加法**派生（int32 有符号溢出是 UB，uint32 回绕保 mod-4 类）。
  gen/genSpec 经**参数**传入三个 Op（OpHitMiss / OpInvalidateSpecSuffix /
  OpClassifyEvictRow）而非读成员——checks 从 kernel.h 文本提取函数做 g++ mock，
  成员访问会破坏提取。relaxed 足够：取号与 kernel 读 tiling 之间隔着 launch 边界
  （H2D 拷贝 + 流序），launch 边界即同步点。Meta 注册路径（EchoMeta）同样经
  Check 取号——只消耗号段不发射 kernel，无害（real 调用仍严格递增）。
- **窗 {4k+1, 4k+2} 与免疫论证**：0（分配清零基线，≡0）/ I32_MAX（audit 样本间
  污染，≡3）/ SENTINEL(-12345，echo_perf_step 每轮全 workspace 填充，≡3) /
  上一 invocation 陈旧值 {g−4, g−3, …} 全部落窗外、一律按候选。旧 bit 语义下
  I32_MAX 的 bit0=1 会被误判 hit、靠 OpZeroFlags 兜底——**新值域严格更安全**。
  基准污染口径不变（compare.py 仍填 I32_MAX、perf-step 仍填 SENTINEL，均窗外
  免疫）。
- **stride=4 论证**：stride=1 撞"本届 genSpec (g+1) = 下届 gen"（spec 标记残留到
  下届被误判 hit）；stride=2 虽躲开它，但 gen 取遍全体奇数、I32_MAX 亦奇数会撞
  gen；4 才把 0/I32_MAX 的余类（0/3）永久留在窗外。uint32 回绕保持 mod-4 类
  不变，碰撞需 2^30 次调用（理论界，登记）。
- **全局单调即可，无需 per-workspace 计数**：任何陈旧值（上一 invocation 的、或
  内存回收复用前的旧 workspace 留下的）恒旧于当前窗，天然不撞；进程重启则配
  "新 workspace 分配时清零一次"契约（0 窗外）。
- **删除与 WAW 论证**：OpZeroFlags + 其后 prologue SyncGrid 整体删除。旧 SyncGrid
  的独占职责 = 排空"清零写 → Step1 scatter 同址写"的写缓冲乱序（写滞留 store
  buffer 可能反超覆盖他人已落地写——fifo 事件同型教训，§3.5/§7.2）；世代号方案
  下**每槽每 invocation 至多一个写者**（hit 写 gen / spec 写 genSpec），无清零写
  可被反超覆盖。hit 槽与 spec 槽不相交由管线顺序保证：Step0.5 先把 spec 槽的
  dth/htd 清成 I32_MAX → Step1 查询必 miss（假想同址双写由 Step0.5 尾栅栏排序，
  终值 gen=hit，与旧 bit0 优先同效）；写→读可见性由 Step0.5 尾栅栏与 Step1 尾
  栅栏各自负责，不依赖被删的 prologue 栅栏。spec 的 OpFillZero（pushCount 清零）
  原地保留为 prologue 最后动作：全链 t0 单线程自写自读，后续读者（OpSpecCumsum/
  OpSpecAvailAdd，亦 t0）前有 Step0.5 自带栅栏，无后随栅栏必要。
- **语义等价论证**（golden 逐字段不变的依据）：给定同样的 hit/spec 集合，新旧对
  槽的划分恒等——hit 槽写 gen（旧写 1）、spec 槽写 genSpec（旧写 2，且只写非
  hit 槽）、其余槽 ∉ 窗（旧为 0，per-invocation 清零的产物）；全部输出（lru
  拼接/淘汰集合/free/avail/fifo）不依赖组内排列之外的 flags 信息。

**⚠ 生产契约变化（醒目登记）**：workspace **分配时清零一次**成为硬契约（生产侧
义务；compare.py 的 empty→zeros 已落实基准侧）。旧方案每轮全量清零、对任意初始
垃圾免疫；新方案对 0/I32_MAX/陈旧 gen 免疫，但**不对任意垃圾免疫**——随机垃圾
撞窗概率 2/2^32/槽，靠契约兜底（分配侧清零 / 复用侧值域自洽）。

**打点槽位级联（29→27）**：名表删 prologue 两项；`ECHO_PERF_NUM_STEPS` 29→27、
`ECHO_PERF_MAIN_STAMPS` 21→19（entry + 非 spec 18 步末 = 19 绝对戳 → 18 步
耗时），断言/文档字符串/compare.py 注释同步；spec 构建下 8 戳实际落位 slot
6..13（原 8..15 顺移）。非 spec 构建 prologue 为空、不占槽（spec 的 OpFillZero
从不打点，耗时并入 entry→Step0 首 Op 差分）。ECHO_V2_DEBUG_CUTOFF 阶段 1 语义
同步收窄（"prologue 后"——非 spec 构建该段为空）。

**checks 54→56**：gen=1/genSpec=2 编码下旧 golden 数值不变（fixtures 编码值
{1,2} 恰为该窗口，等值链与旧 bit 判定在此编码下重合——原 54 项原样通过）；新增
两项：①ClassifyEvict 陈旧/audit 免疫（预填 {gen, genSpec, gen−4, gen−3, 0,
I32_MAX} 混合，断言仅窗口内两值生效、窗外一律候选）；②OpHitMiss 覆写（陈旧
hit 值 5 → gen=9、非 hit 槽原样保留陈旧/audit 值——设备侧已无 prologue 清零可
依赖，**覆写语义即正确性**）。

**预期收益（诚实版）**：账面 = 两槽合计 13.8K；**临界路径估 −7~10K/场景**。
论证（§9.23 打点读法注记的直接应用）：SyncGrid 槽的 max 统计含快核等待成分——
临界路径是最慢核，**慢核不等人**，删栅栏省的是它自己的冷 store（OpZeroFlags
~4.3K；冷行 store 同价，§9.20）+ 一道栅栏地板与排空；快核的等待成分不消失、
转移到后续首道栅栏（warm/full 为 Step1.SyncGrid、cold 为 Step0.SyncGrid）。

| 场景 | v2.1-prof 基线（μs） | v2.2 预期（μs） |
|---|---:|---:|
| full_cache_75pct | 82.24 | ~76-78 |
| warm_75pct | 71.30 | ~65-67 |
| warm_all_hit | 68.96 | ~63-65 |
| cold_reset | 96.19 | ~90-92 |

（基线 = §9.23 验证轮的 kernel 级 P50；−7~10K cyc @ ~1.65 GHz ≈ −4.2~6.1 μs
折算。）上板 A/B 见真章；**实测 <6K 则复盘账目**（等待转移假设是否成立，用
分步对照定位）。

**验证计划**：checks 56 项 → 双构建（正式 + ECHO_PERF_STEP）→ 四场景 golden
（语义等价论证 ⟹ 逐字段不变）→ profile A/B（基线：full 82.24 / warm75 71.30 /
warm_all 68.96 / cold 96.19）→ 分步对照（prologue 两槽应消失；跨度基线：full
136046 / warm75 116156 / warm_all 113432 / cold 160240）。锚点用当轮
profile_kernel 各场景 P50 逐场景传入（§9.23 操作协议）。

**落地后记（2026-10-21，编译坑插曲）**：首次服务器构建触发 bisheng 后端坑
§7.6.3——`v == gen` 等值链在 -O1+ 指令选择崩溃。当日以变体二分定位：V0 复现 /
V0b 对照（因服务器无 git 作废，改 Windows `git show` 提取的备用流程）、V1-V3
（Init 派生/入口拷贝/参数类型，中和后仍挂）、V6（结构体，排除）、V7（比较回退
位测试，**定位命中**）、V8（栅栏回补，排除）、W0-W2（修复形态三连测，W2 差值
判别中选）；其中 V5/V9 两轮因诊断脚本 sed 漏分号产生语法错、结论作废——**教训：
变体判定必须显式 grep 错误类型（后端 fatal / 其他 error / 编译通过三分）而非只看
make 退出码**。最终落码 W2 形态（与等值链逐位等价，见 OpClassifyEvictRow 头注释
[后端坑规避]）；checks/golden/语义论证均不受影响（分类输入相同、划分恒等，
genSpec 形参保留无读者以稳签名）。定位全程无 plog 可读（错误不落 run/plog、
stack dump 被吞，§7.6.3 取证教训）——若后续找到编译期日志路径可补证机理假说。

## 10. 当前状态与后续方向

### 10.1 kernel 级数字（现状）

> [v2.2 2026-10-20] flags 世代号化已落地（§9.24，commit `284ff71`），NPU 待验证
> （预期 full ~76-78 / warm75 ~65-67 / warm_all ~63-65 / cold ~90-92 μs，临界
> 路径 −7~10K/场景）；下列 v2.1 数字仍为**最近已验证基线**。

**v2.1 单趟分类 + 平局松弛 + 尾段追加（§9.22，2026-10-20 上板全过，tag
`v2.1`）：full_cache 82.12 μs（0.88×，v2.0 −18.5%）——首次破 100 且
warm×2 + full 三场景全部 ≤ SIMT；warm 71.36/69.33 μs（0.88×/0.85×）；
cold 99.59（1.34×，快路径兑现 −8%）**。距 50-60 μs 目标：full 还差
~25-30%。精确打点已收口（§9.23 验证轮），剩余大头按新口径（绝对戳差分含
间隙）为 **Step5 簇 38.7K / ClassifyEvict 31.1K / HitMiss 27.8K**（full，
§10.3 重排）。SIMT 本轮 75-93 μs。快照链：v2.2 = `284ff71`（未打 tag，验证
后打）→ tag `v2.1`（最近已验证）→ `v2.0`（§9.17-§9.21）→ `v1-argtopk-final`。

### 10.2 地板项清单（不再可优化或收益有限的部分）

| 项 | 场景 | 量级估算 | 依据 |
|---|---|---|---|
| OpResetRows 带宽 | cold | 每行 htd 1 MiB（262144 int32）纯写，32 行 = 32 MiB；cold 整 kernel 91.44 μs（t512 口径，§9.10），reset 是其中的主导步（若按整 kernel 摊算已 ≈360 GB/s，步内瞬时带宽更高，接近写带宽地板） | `echo_lru_kernel.h:95-96`，`compare.py:376` |
| OpHitMiss gather | 全部 | 每行 2048 次随机 GM 读（htd[row, tok]，跨 1 MiB 区间），cache miss 为主；**核级吞吐受限（t1024 轮数减半实测无效，§9.21）+ 单发冷触串行 ~3-4K（§9.20）⇒ t512 地板 ~22-25K，收尾空间 ~4-7K**（v2.1-prof 新口径实测 full 27.8K、skew 6.8K 数据相关长尾，§9.23） | `echo_lru_kernel.h:281-286`，§9.20/§9.21 |
| OpFifoAvailUpdate 冷触 | 全部 | fifo/avail 各 1 次冷行触（store 同价，H2，§9.20）≈ 6K；早算 stash 已实测无效（§9.20）；唯一削减路径 = 两数组合并（远期） | §9.20 |
| 必要 scatter | 全部 | htd/dth/curSlots/flat/mask 的逐 miss 写 + pri 的逐 hit 写 | OpHitMiss（hit 分支）/OpSetUpdate |

### 10.3 已识别方向（按当前优先级排序；v2.2 已落地待验证，§9.24）

**原列 1（精确打点，专家意见 2026-10-20"先测准再动刀"）已收口**：三件套全部
落地并上板验证（§9.23）——①kernel 内绝对时间戳 + 间隙归属下一步（对账
94% → ~100%，prologue/skew 首见）；②标定修正（1.585 GHz 常数 + `--p50-us`
锚点两档对照，Event 标定行废弃 §7.8），验证轮翻出**跨 run 频率漂移**（两 run
差 ~4-5%，msprof"无降频"是单时点采样）——操作协议 = cycles 决策 + 背靠背连跑 +
每场景锚点，工具项已收口（逐场景锚点语法 + 三信号漂移探测器）；③prof_report.py
单文件 HTML（+ MindStudio Insight 可开 PROF_* 目录）。msprof 首轮结论存档：主频
1.58-1.59 GHz（该 run）、vec 管线主导（aiv_vec_ratio 0.97-0.99、MTE 判据不适用
于纯 SIMT 访存）、单核均值 ~131.5K cyc 与分步对账闭合；**勿用 msprof op**
（还原-重放工具，对 torch 负载不适配），采集时去掉 --verify-each-sample。
**仍挂起**：L2Cache / ResourceConflictRatio / Memory 三组 task-based 计数器 +
Run A 的 aicore.db 逐核 task_cyc 抽取（核间均衡，与 §9.23 的 skew/热力图交叉
验证）；msprof 不含 kernel 内函数级拆解，函数级仍靠时间戳方案。原 [新候选]
**flags 世代号化已落地**（§9.24，commit `284ff71`，NPU 待验证——prologue 实测
13.7-13.9K 远超登记时 −5~8K 的假设基础；验证通过后移入下"已完成"）。

1. **ClassifyEvict 再削**（full 31.1K 新口径；v2.1 旧口径 29.0K = 3 栅栏 ~11K +
   staging/两遍走查/散射 ~18K）：候选 = 尾栅栏与后续 Op 的依赖重审（freeIdx 读者
   是 FreeRelease 而非 AllocPop，B3 能否并入 FreeRelease 前提）、staging 读的
   合并（flags 与 lru 相邻段一次读？）、发射扫与计数扫的切片复用杂项。
   目标 −3-6K。
2. **Step5 簇（full 最大簇，38.7K = MissCumsum 17.8K + SyncGrid 5.4K +
   SetUpdate 15.5K；自旧"收尾清单"提级）**：SetUpdate 的 htd 冷 store 簇
   （miss 路径，§9.20 冷 store 同价）与 FreeRelease（full 8.2K + FreePush
   5.0K）的 dth 读可否合并一趟；MissCumsum 杂项（45% 是 2 道扫描栅栏——
   [Scan-v2] 已是最优结构，剩载入/写回 ~2K 空间）；簇间栅栏依赖重审
   （Step5.SyncGrid 5.4K 的读者是谁）。
3. **HitMiss gather 地板**（full 27.8K、skew 6.8K 数据相关长尾）：核级吞吐
   受限（t1024 证伪，§9.21）+ 单发冷触串行（§9.20）⇒ t512 地板（§10.2）；
   批量发射实验（多在途 load 重叠检验，§9.20 歧义注记）是剩余手段，收尾
   ~4-7K。
4. **cold 专项**：reset 簇 53.4K（ResetRows 34.0K 带宽地板——32 MiB 纯写
   ≈ 1.6 TB/s 已高利用率 + DthLru 7.1K + Free 3.7K + AvailFifo 2.7K +
   SyncGrid 5.9K）+ SetUpdate 23.2K + HitMiss 24.3K + ClassifyEvict 11.7K
   （快路径的 wrapper avail 冷读 + rbBase 冷写，已是地板形态）。
5. **投机解码（spec）场景测试**：compare.py 的 spl 恒为 max_token，spec
   golden 仍未覆盖，需先扩 reference。
6. **混合范式预研（专家意见，parked）**：纯 SIMT 尚未到极限（上列 1-4 估计
   还有 −10-15K ≈ full ~72-77 μs 空间），到 50-60 大概率需要范式级手段
   （如 htd 查询/散射走 vector 侧；ResetRows 的 MTE 写路径候选——restore 的
   TensorMove 32 MiB 拷贝 MTE2 ~13 μs ≈ 4.7 TB/s（r+w 合并）vs 纯写 1.6 TB/s，
   kernel 内不可用 MTE），等纯 SIMT 收尾后再评估。
7. **远期**：fifo/avail 合并为单状态数组（FifoAvail 2 次冷触并 1 次 ~−3K）。
8. 已完成/证伪：~~精确打点三件套~~（§9.23 落地并上板验证）、~~EvictScan×Rebuild
   融合 + golden 平局松弛~~（v2.1 落地并验证，§9.22）、~~行间流水~~、
   ~~维护序 v2~~（§9.17）、~~FifoAvailUpdate 并行化~~（opt3 证伪，§9.20）、
   ~~t1024~~（§9.21 证伪）移除。

---

## 11. 文档更新日志

| 日期 | 版本 | 覆盖 commit | 变更摘要 |
|---|---|---|---|
| 2026-10-15 | 初版 | `145d3ce` | 首次成文：全量深读 compare.py / csrc/echo / csrc/simt（大结构）/ checks / 构建与打点脚本；含优化迭代日志（P0-P4/B/C1/C2/方向1+2）与 kernel 级现状数字 |
| 2026-10-15 | 第 2 版 | `d20fc50` | 同步消融基础设施（ECHO_LRU_THREADS 编译期线程数、OpMissCumsum 4 元素路径、`--blocks`/`--rows` 消融入口、README 消融章节）与方向 1+2 kernel 级实测（full_cache 162.16 μs=1.75×、warm 0.95/0.91×、cold 1.38×）；协作机制：本仓代码由 developer-code 子代理按主会话规格实现、文档由 developer-doc 维护、均经主会话审查 |
| 2026-10-16 | 第 3 版 | `e2ca2fd` | 同步运行时自适应 blockDim（plugin 按 topkDim 计算 launchThreads、`launch_echo_lru` 增加 blockDim 参数、`ECHO_LRU_THREADS` 定位为纯消融 knob）与消融实验结论（§9.10：t512 全场景最优 −9%~−18%、t256 崩塌为串行 MissCumsum 混杂因素、blocks 16/8 单调变差、rows 64/128 行间串行软肋）；§10 现状与后续方向按当前优先级重排 |
| 2026-10-16 | 第 4 版 | `1121091` | 同步 round-3 两批 kernel 内优化（f8b9c69：[Scan-v2] 单栅栏扫描/破坏性归并/[P-C] freeIdx 迁 UB/[P-D] pack 融合，§9.11；1121091：protect 折叠进 OpHitMiss/OpFifoInc 挪至 AllocPop 后/[FifoAvail 拆分]/非 spec 步骤 23→20 与 ECHO_PERF_NUM_STEPS 31→28，§9.12）；§4 单栅栏论证、§5 步表按 20 步重写（含 fifo 挪位机理）、§6 相位图与栅栏计数刷新；补全 8856e2a 扫描门 UB 侧条件删除的表格与正文同步；checks 172 项 |
| 2026-10-16 | 第 5 版 | `d461f5d` | 修复 P-C UB freeIdx 绝对行索引越界（full_cache 挂死），改核内本地行索引（详见 §9.13） |
| 2026-10-16 | 第 6 版 | `d461f5d`（上板复核） | round-3 两批 + UB freeIdx 修复上板复核（第二台服务器）：新增 §9.14（kernel 级 P50 / 分步周期 / full −5.9% 归因 / 两个测量 open question / 事故闭环）与 §1.6（基准单步口径 vs 真实 decode 多步推演）、§7.10（设备占用测量教训）；§9.11/§9.12/§10 复核状态与现状数字刷新；§11 日志表修复孤立行并核对日期 |
| 2026-10-16 | 第 7 版 | 待提交（round-4：[P-E]+白付清理） | 同步工作区未提交的 round-4：[P-E] 反向 P-D（OpArgtopkRow 行首全员 pack + B0，hist 一律读 key，Argtopk 栅栏 7→8，§6.2/§6.7/§9.15）与白付清理 A/B（非 spec 跳过 prologue FillZero+SyncGrid、anyReset=0 跳过 Step0 四 Op+SyncGrid，打点槽保留，§5.1/§9.15）；§9.11 [P-D] 条目标注已被回退；全文 kernel.h 行号按当前工作区核对修正；附录⑥标记已解决；NPU 验证待做（预期数字与挂起二分顺序见 §9.15） |
| 2026-10-16 | 第 8 版 | `3906f91`+`9b63e24`（上板验证） | round-4（[P-E]+白付清理）上板验证（空闲设备）结果写入：§9.15 新增上板验证小节（warm 58.28/55.34 μs＝0.72×/0.67× 超出白付清理预期、cold 94.16 μs、full 120.12 μs；P-E 收益重校准——Argtopk −9.1K 远低于静态预期 −30K，Argtopk 转弥散结构无单一主导相位；距 50-60 μs 定量账 → v2 必要性坐实；int16 hist 扩 G=128 候选撤下）；§9.14 open question 1 关闭（SIMT ~2× 确认为设备占用干扰，比值恢复可信）；§7.11 新增"静态延迟链模型高估依赖读成本"方法论教训；§9 总表 round-4 行、§10.1 现状、§2.2 比值引用、头部覆盖范围同步刷新 |
| 2026-10-20 | 第 13 版 | `ab4fb5b`→`f2acb7c`（t1024 消融实测回退） | §9.21 t1024 消融负结果：P50 四场景 +25~37 μs（Rebuild +11~16K=栅栏翻倍、HitMiss +6~8K 未降反升、MissCumsum −2.5~3.9K 不抵、ResetRows 持平）；三条结论（栅栏随线程数翻倍且占总时间 ~1/3、gather 核吞吐受限非每 warp 串行（H1 修正）、ResetRows 带宽地板佐证）；§10.2 HitMiss 行改"收尾 ~4-7K"；§10.3 重排（EvictScan×Rebuild 融合+栅栏削减列 1、新增 golden 平局规则松弛决策点、t1024 证伪移除）；CLAUDE.md 同步 |
| 2026-10-20 | 第 17 版 | v2.2（`284ff71`）+ 工具项（`50128d3`）+ v2.1-prof 验证轮 | §9.23 上板验证记录：P50 复现（v2.1-prof 二进制四场景 −3.4~+0.1 μs 于 v2.1 轮，打点构建不改行为）+ 分步新口径全表（绝对戳差分含间隙，与旧表同物异账——ClassifyEvict full 29.0→31.1K 非回退）+ skew/跨度/Σ步max 对账（跨核逐峰求和虚高 +10.8~12.5%，对账以跨度为准）+ 头条结论 prologue 两槽 13.7-13.9K（占跨度 8.5-11.9%，直接触发 v2.2）+ 频率漂移发现（隐含频率 1.63-1.67 GHz 一致高于 msprof 1.585；不可能不等式实锤两 run 差 ~4-5%；锚点=跨度/P50 天然低估；操作协议 = cycles 决策/背靠背连跑/每场景锚点）+ 栅栏步 max 隐藏等待的读法注记；新增 §9.24 v2.2 flags 世代号化全记录（gen 从 1 起 stride 4、窗 {4k+1,4k+2}、stride=4 与 WAW 消失论证、生产契约 = workspace 分配时清零一次（compare.py empty→zeros）、槽位 29→27、checks 54→56、预期临界路径 −7~10K 与验证计划）；§10.1 加 v2.2 待验证注记 + 快照链、§10.2 HitMiss 行新口径转注、§10.3 重排（精确打点收口为前言、ClassifyEvict 31.1K / Step5 簇 38.7K / HitMiss 27.8K / cold reset 簇 53.4K 列 1-4）；头部 banner 覆盖 →§9.24、§3.2/§3.4/§3.5/§8.3 槽位与锚点转注；CLAUDE.md 同步（checks 56、v2.2 状态段、优先级、逐场景锚点语法示例、快照链 v2.2=284ff71 未打 tag）；README checks 项数最小修正 |
| 2026-10-21 | 第 18 版 | v2.2 编译坑修复（W2 差值判别） | bisheng 后端坑 §7.6.3 登记（节标题"前端两个"→"编译坑三个"）：i32 加载值与运行时形参 eq/ne 比较在 -O1+ 触发 "Copy one register into another with a different width"（-O0/常量比较不触发；参数类型/结构体尺寸/栅栏无关；非 eq/ne 谓词 vs 形参不触发——v2.1 `tok >= spl` 长期可编为证；机理假说 = 形参寄存器宽度搬运 COPY，待编译期日志补证）；OpClassifyEvictRow 两处扫描改差值判别 d=v−gen（d==0 hit / d==1 spec / d>1 候选，与等值链逐位等价——当日 W2 变体实测可编形态原样落码），genSpec 形参保留无读者（签名稳定，13/15 参均实证可编）；§9.24 落地后记（变体定位矩阵 V0-V9/W0-W2、V5/V9 作废教训 = 变体判定必须显式 grep 错误类型、取证教训 = 错误不落 run/plog 且 stack dump 被吞）；CLAUDE.md：协作规则 3 更正同步链（Windows PC git pull + sftp，服务器无 git、服务器侧命令禁 git）+ 设备坑速查加 bisheng 后端坑条 + v2.2 状态段补注 |
| 2026-10-20 | 第 16 版 | v2.1-prof（NPU 待验证） | §9.23 打点升级全记录：差值→绝对时间戳（29 槽 = entry + prologue×2 + 非 spec 18 + spec 陈列 8；PerfBegin/perfLast_ 删除；host uint32 回绕差分；步间间隙归属下一步）；标定修正（默认 1.585 GHz 常数 + --p50-us 锚点两档对照，Event 标定行废弃 §7.8）；prologue 纳入打点（对账 94%→~100% 预期）+ skew 首见；ECHO_PERF_DRAIN 排空旋钮（build.sh + CMakeLists，产物 build_v2dbg/，需配 ECHO_PERF_STEP=ON）；echo_perf_step.py 重写 + --matrix CSV 导出 + 新增 prof_report.py（单文件 HTML：对账条/逐核×逐步热力图/逐步聚合/op 表/原始折叠表，dataviz skill 已验证调色板原样引用）；§8.3 加 v2.1-prof 指针、§10.3① 刷新（时间戳+标定已落地待验证、可视化结论：自建 prof_report / MindStudio Insight 可开 PROF_* 目录）；checks 常量断言 26→29；遗留三项（debug_v2_stepcycles 写判定、compare.py:299/launch.h 注释）已由主会话审查轮顺手同步、CMakeLists 为白名单外必要管道（旋钮不进 ECHO_FLAGS 即静默失效）一并合入；CLAUDE.md 同步 |
| 2026-10-20 | 第 15 版 | v2.1（`6188d3b`+验证） | §9.22 上板验证：四场景 golden 全过、P50 warm 71.36/69.33（0.88×/0.85×）/full 82.12（0.88×，−18.5%）/cold 99.59（1.34×）——首次三场景 ≤SIMT、full 破百；分步对照表（ClassifyEvict full 29.0K = 旧两 Op 之和 −56%；cold Rebuild 31.8K→10.7K）；§10.1 刷新；§10.3 重排（精确打点列 1：msprof op 硬件侧视角 + kernel 内绝对时间戳升级 + 标定修正；ClassifyEvict 再削/收尾清单/混合范式 parked）；快照 tag v2.1 |
| 2026-10-20 | 第 14 版 | v2.1（单趟分类，NPU 待验证） | §9.22 v2.1 全记录：单趟分类 OpClassifyEvictRow（三产出 freeIdx/K 段/H 段+rbBase，融合 EvictScan+Rebuild 位置域扫描，每步必跑，occ==0 快路径）+ 平局松弛（hit 旧位置序/alloc 分配序，owner 2026-10-20 批准）+ 尾段追加 OpAllocAppendRow（零栅栏零扫描）；UB 删 aAt/hExcl/aExcl 三段（峰值 26176→13120 实算链）、numHitsGm 复活为 rbBase（布局零漂移）；reference_echo 改旧序直接法（htd 法判 miss 原样）；checks 重写 54 项（跨验证组改新规则+成员集合不变量）；旋钮清理（SKIP_REBUILD+9 个 RB_*，CUTOFF 保留）；预期收益表（warm_all ~67-69/warm75 ~68-71/full ~79-83/cold ~87-90）与风险预案（trap 预案=Step2.SyncGrid 无条件化）；§10.1 加 v2.1 待验证注记、§10.3 重排（v2.1 验证列 1，融合/松弛移入已完成）；banner 覆盖范围 →§9.17–§9.22；CLAUDE.md 同步 |
| 2026-10-19 | 第 12 版 | 48610c0→266ed64（opt3 实测回退）+ `ab4fb5b`（t1024 消融待测） | §9.20 负结果记录：opt3 预置摊销实测净回退 ~+3K（HitMiss 四场景 +2.7~4.2K / FifoAvail 持平 / P50 +1~3.6 μs），机理 = 同 warp 冷访存串行（H1）+ 冷行 store 同价非火后即忘（H2）+ 块内栅栏同步推论，已 revert 回 v2.0 基线；§10.2 补 OpFifoAvailUpdate 冷触地板行、OpHitMiss 行按 H1 重估（t512 地板 22-25K）；§10.3 按 H1/H2 重排（t1024 消融验证列 1、Rebuild 栅栏/融合细化、FifoAvail 并行化证伪移除、fifo/avail 合并记远期）；CLAUDE.md 设备坑速查 + 优先级同步 |
| 2026-10-19 | 第 11 版 | v2.0（6f4ad92+验证） | §9.19 opt1/2 上板验证（Rebuild warm_all 37.2→30.5K，P50 四场景 −2~−4%）；§10.1 刷新至 v2.0 数字；快照 tag v2.0。交接文档 CLAUDE.md 新增（新 AI 接手用） |
| 2026-10-19 | 第 10 版 | v2 轮（fe10f91..本轮） | 恢复记录（按 owner 指示跳过中间误判过程）：新增 §9.17 v2 维护序重构全记录（设计/接口变更/上板 P50/分步/warm 回退账目）、§9.18 栈溢出事故（根因+修复，详见 §7.12 与独立文档）、§9.19 v2-opt1/2（待验证）；§7.12 新增 VEC SIMT 栈溢出条目（含 §7.9 grid.sync 关联修订）；头部 v2 转型 banner（§5/§6 Argtopk 内容保留为 v1 历史，checks 50 项）；§10.1/§10.3 按 v2 现状重写 |
| 2026-10-19 | 第 9 版 | `05046f2`（NPU 待验证） | 恢复按 owner 指示暂停的文档同步：新增 §9.16 round-4b（emission 尾栅栏去重——OpArgtopkRow 函数尾栅栏 GM-only、Process Step2.SyncGrid 槽 UB 路径跳过；Argtopk 栅栏 UB 8→7/行）；§6.7 表追加 round-4b 行并核对头注释（:395-399），§3.5/§5.1（含 §5.2 Step2、§6.2/§6.6 emission 尾部描述）同步更新、行号按 05046f2 核对；§8.1 新增本机/服务器验证分工（本机只跑 checks，编译与 NPU 验证归服务器）；§2.2 补 warm→full 边际成本定量论证（Echo +61.84 vs SIMT +12.01、边际差 49.83 μs，v2 定量动机）；§10.3 附注已评估并搁置的微优化候选（rawSlots/wsMissRank 迁 UB、UB 扩容 192 KiB）+ §3.2 G=64 成因注记 |

---

## 附：通读中发现的不一致（文档 vs 代码 / 代码内陈旧注释）

撰写与历次修订时逐条核对发现以下不一致，**均未改动任何代码**，仅在此登记，供后续修订
README 或代码注释时参考（编号约定：新条目按时间顺序**追加在列表尾部**——历史条目
编号被 §11 日志行引用过（如"附录⑥"），不插入中部、不整体重编号）：

1. **README §"本次 Echo 更新"（`README.md:152`）说 checks 为 "171 项"**：当前 `checks/check_echo_kernel_host.py`
   实际产出 **172 项**（160+1+3+7+1，§8.5）——d20fc50 补 4 元素用例至 171 项后，
   f8b9c69 再加 [P-C] UB/GM 一致性用例至 172 项，README 数字未同步。
   （2026-10-20 注：README 的 checks 项数本轮已最小修正为现行 **56 项**（v2.2，
   §9.24）；§8.5 的 172 项分项表为 v1 历史，见头部 banner。）
2. **echo_lru_tiling.h:1-9 头注释**仍描述 "Plan B / grid_group::sync" 时代的排布；
    **echo_lru_kernel.h:12,1310** 的 "<<<blockNum, 512, ...>>>" 是 512 线程旧值，且
   kernel 头部 `:15`、行段宏注释 `:83` 仍写 "行内 ECHO_LRU_THREAD_NUM(=1024) 线程"
   （`echo_lru_launch.cpp:8-9` 实际传 plugin 自适应的运行时 blockDim）。当前有效
   事实：Path A、blockDim 运行时自适应（上限 `ECHO_LRU_THREAD_NUM=1024`，§3.2）。
   README §9 已声明"以有效代码为准"，但逐条修订更好。
3. **栅栏成本口径**：kernel 注释（echo_lru_kernel.h:41）记 "~6-8K 周期"，本仓库外部
   材料有 "~4-8K" 的说法，量级一致、区间略有出入；本文 §7.3 按 "4~8K（注释实测 6-8K）"
   并列表述。
4. **代码注释引用的 `GRID_GROUP_PITFALLS.md`、`probe_grid_sync/`、`probe_ub_size`、
   `golden.py`、`gen_data.py`、`echo-lru-argtopk-debug-status` 等上游工程产物不在本仓**，
   相关结论只能以注释转述为准（§3.1、§7.5、§5 spec 段）。
5. **PackedBlockScan 头注释陈旧**——**已解决（d461f5d 轮）**：f8b9c69/1121091 时该
   注释（曾位于 echo_lru_kernel.h:46-55）确仍描述双栅栏 + `control[0]` grand total +
   warpBases 的旧结构；d461f5d 已改写为 [Scan-v2] 单栅栏/本地重算描述（现
   `echo_lru_kernel.h:46-52`），与函数体（`:53-69`）一致；§4 按函数体描述。
6. **OpArgtopkRow 头注释的栅栏计数陈旧**——**已解决（2026-10-16，round-4 [P-E]）**：
   原注释（曾位于 echo_lru_kernel.h:391-392，旧条目误记 :394-395）的 "emission 扫描 2 +
   尾 2 = 8" 未随 [Scan-v2] 更新；本轮 [P-E] 加 B0 时已一并改写为 "B0 1 + 基准 2 趟 × 2 栅栏 +
   emission 扫描 1 + 尾 2 = 8"（现 `echo_lru_kernel.h:395-397`），与 §6.7 表一致。
7. **OpMissCumsum 头注释（echo_lru_kernel.h:754）**仍写 "扫描 scratch 固定 66 槽
   （warpTotals/warpBases/ctrlA/ctrlB）"——[Scan-v2] 后为 (行奇偶×scan序) 四缓冲
   128 槽、无 warpBases/ctrl（§4.1/§5 Step5）。
8. **kernel 头注释（echo_lru_kernel.h:17-18）**的行内跨线程 RAW 举例仍引用 OpProtect
   ——protect 已折叠进 OpHitMiss 的 hit 分支（§5 Step1），该 Op 函数已删除。
9. **[2026-10-20 本轮登记] README §6/§9 的 Echo 描述停在 v1/v2.0 形态**（本轮仅
   最小修正 checks 项数与 ECHO_PERF_NUM_STEPS 数值，未重写）：§6 的"每核只用
   调试 dump 段 `256+8192`"与"key/hist 等在每核 128 KiB 静态 UB"（v2 已删
   argtopk dump 段与 key/hist UB 段，§9.17）；§9.2"线程数由 tiling.h 定义，
   目前为 1024"（实为 plugin 运行时自适应，topk=2048 → 512，§3.2）；§9.4
   "每步由 PerfBegin/PerfEnd 打点"（v2.1-prof 起为 PerfEntry + 步末绝对戳，
   §9.23）；§9.5 的 OpEvictScanRow/OpLruRebuildRow（v2.1 起为
   OpClassifyEvictRow/OpAllocAppendRow，§9.22）；§9 末尾"以有效的
   OpArgtopkRow（radix select）…为准"的锚点本身已失效（Argtopk 于 v2 删除）。
10. **[2026-10-20 本轮登记] 本文 §3.2 常量表的 ARGTOPK_* 行与 §3.4 布局图的
    dbgCount/dbgKey/stepCycles 段名仍为 v1 形态**：v2 已删 argtopk 诊断 dump 段、
    v2.1-prof 起 stepCycles 更名 stepAbs——本轮仅加 [v2/v2.2 转注] 未重写正文；
    现行布局以 `echo_lru_launch.h` 注释与 `Init` 为准（新公式 perBlock 按
    ECHO_PERF_NUM_STEPS 符号引用，本轮已核对）。
