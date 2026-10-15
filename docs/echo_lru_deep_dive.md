# Echo LRU 算子深度解读（kvoffload_simt_echo）

> **本文档定位**：面向算子开发协作者的 Echo（`echo_lru`）实现深度解读。目标 是**只读这一份文档就能读懂当前代码**——每个论断都给出代码出处（`文件:行号` 或函数名），所有"为什么"都附依赖分析、硬件特性或不变量证明。
>
> **本文档随代码迭代同步更新**：每次代码合入后由维护者增量修订对应章节，并在文末"文档更新日志"表追加一行。
>
> 覆盖范围：commit `e2ca2fd`（perf(echo): runtime-adaptive blockDim from ablation sweet spot）。
> 工程全貌与运行命令见根目录 `README.md`，本文不重复，只做深读。

---

## 目录

1. [算子要解决的问题](#1-算子要解决的问题)
2. [与 SIMT 基线的本质差异](#2-与-simt-基线的本质差异)
3. [并行与内存结构](#3-并行与内存结构)
4. [warp 协作原语](#4-warp-协作原语)
5. [流水线逐步解读（23 个非 spec 步）](#5-流水线逐步解读23-个非-spec-步)
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
接口 schema 见 `csrc/echo/echo_lru_plugin.cpp:126-130`（14 个张量 + blocks + spec_enabled）。

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
| `avail` | `[rows]` int32 | free 栈有效长度 | 终值 = `availOld + fs - nm ≥ 0`（见 §5 Step5） |
| `fifo` | `[rows]` int32 | 本轮时间戳发生器 | 每轮自增**两次**（hit 保护后一次、分配后一次，`compare.py:187,203`） |

**输出（每步覆写）**

| 字段 | 形状 | 语义 |
|---|---|---|
| `current_slots` | `[rows, topk]` int32 | 每个 query 的最终槽位：hit = 原槽位（Step1 写），miss = 本轮分配的 `aslot`（Step6 回写，`echo_lru_kernel.h:926`） |
| `miss_host_pos` | `[rows, topk]` int32 | miss 位置填 token 值（host 由此知道去主机取哪个 token 的 KV），hit 填 0（`compare.py:207`） |
| `miss_alloc_flat` | `[rows, topk]` int32 | miss 填 `row*capacity + aslot`（**保留原接口约定，1 基槽位不减 1**），hit 填 0；**hit 位的 0 不是合法 miss 地址**（`compare.py:217`、`echo_lru_kernel.h:921`） |
| `miss_mask` | `[rows, topk]` uint8 | 1 = miss。输出不做压缩（SIMT 是压缩前缀 + `miss_count`，两者格式不同） |

`effTopk = min(topkDim, B)`（`echo_lru_plugin.cpp:77`）：防 buffer 超订的有效查询宽度。
`j >= effTopk` 的位置是 inactive 尾巴：`curSlots=0`、`flat=0`、`mask=0`
（`echo_lru_kernel.h:284,909`）。本基准 `topk ≤ capacity`，无 inactive 区。

### 1.3 一轮的语义时序

```
reset（若换请求）→ hit/miss 判定与计数 → hit 槽位 pri=fifo（保护）→ fifo++
→（若 nm > avail）按 pri 选出 fs 个最旧槽淘汰并压回 free 栈
→ 从 free 栈尾弹出 nm 个槽分给 miss → 新槽 pri=fifo' → fifo++、avail 折叠更新
→ miss 前缀秩扫描 → htd/dth/输出/curSlots 映射更新
```

其中 `nm` = miss 数，`fs = max(nm - avail, 0)` = 需淘汰数（`echo_lru_kernel.h:299-301`）。
CPU 参考 `reference_echo`（`compare.py:159-211`）按同样时序用排序表达淘汰语义。

### 1.4 与 golden 的关系

- **golden 只做结果验证，不参与计时**：CPU 参考"只有需要淘汰时才排序"，这是参考实现
  省掉无效计算（`compare.py:160-163`）；**设备实现无论是否淘汰都走常数趟数的基数选择
  路径**（`fsClamp=0` 时提前返回，`echo_lru_kernel.h:419-422`），所以两者只在"输出"上等价，
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
   （`compare.py:105-107`），与 kernel reset 行为 `free[col] = B - col`（`echo_lru_kernel.h:138`，
   弹出顺序 1,2,3,…）语义一致。

基准四场景（`SCENARIOS` 定义见 `compare.py:24-29`；同 seed 同 query，只差预埋与是否
reset，场景语义表见 README §3）：

| 场景 | 每行 miss | 初始 avail | 需淘汰 fs | 特点 |
|---|---:|---:|---:|---|
| `cold_reset` | 2048 | 4096 | 0 | 全量 reset，htd 重写是带宽地板 |
| `warm_75pct` | 512 | 2560 | 0 | 部分命中，无淘汰 → C1 整段跳过 |
| `warm_all_hit` | 0 | 2048 | 0 | 全命中 |
| `full_cache_75pct` | 512 | 0 | 512 | **缓存满，每步 miss 必然淘汰**——解码稳态的代表场景，Argtopk 唯一跑满的场景 |

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
  次散射写 `pri[slot]=fifo`（OpProtect）；**只有需要淘汰时**才用 O(B) 的 4 趟基数选择
  （radix select）从全量 key 中"现算"出 pri 最小的 fs 个槽（OpArgtopkRow）。无淘汰时
  （fs=0）整条淘汰流水线被 C1 跳过（§5 Step3）。

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
数字：warm 0.73-0.78× vs full 1.48×）。**这不是实现优劣，是状态表示的代价分布不同**。

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
无需跨核 sync"（`echo_lru_kernel.h:1133-1134`）。

**历史包袱（为什么叫 Path A）**：早期曾攻关 Stage 2 全元素 grid-stride 跨核协同
（需要 `grid_group::sync()`），结论是"极简 probe 全 PASS，但 echo_lru Process 内哪怕
一个 grid.sync 即 trap，根因是编译单元整体属性，不可复现"，2026-09-16 暂停，回退
Path A（`echo_lru_kernel.h:1136-1141`、`:957-959`；详见上游 `GRID_GROUP_PITFALLS.md`，
该文件不在本仓）。`csrc/echo/echo_lru_tiling.h:1-9` 头部注释仍残留"Plan B /
grid_group::sync"时代描述，**以 Path A 实际代码为准**（README §9 亦有说明）。
`blockNum ≤ 向量核数` 的约束仍由 plugin 强制（`echo_lru_plugin.cpp:91`）。

### 3.2 tiling 常量表

全部在 `csrc/echo/echo_lru_tiling.h`（host/device 共享）：

| 常量 | 值 | 含义 | 出处 |
|---|---:|---|---|
| `ECHO_LRU_THREAD_NUM` | 1024（默认） | 每核线程数**上限**（= `__launch_bounds__`，`echo_lru_kernel.h:1250`）。实际启动线程数由 plugin 按 topkDim 运行时自适应（公式见下）；仅当 `ECHO_LRU_THREADS` 手动压低上限时才直接约束启动值。static_assert 三条：32 的倍数（warp 大小）、≥64（覆盖 G 路直方图切片）、≤1024（AIV launch 上限） | tiling.h:18-34 |
| `ARGTOPK_B_MAX` | 4096 | B（=capacity）上界，plugin CHECK | tiling.h:43 |
| `ARGTOPK_NPAD_MAX` | 128 | 遗留常量，当前无实际引用（仅出现在一行旧注释里） | tiling.h:44 |
| `ARGTOPK_GM_NPAD_MAX` | 8192 | = next_pow2(B_MAX+1)：GM 调试 dump 每核 srcKey 容量 | tiling.h:45 |
| `ARGTOPK_UB_NPAD_MAX` | 8192 | key 打包元素上界（= next_pow2(B+1)，B=4096 时） | tiling.h:47 |
| `ARGTOPK_UB_ELEMS` | 32768 | 单大 `__ubuf__` 数组总元素（4×8192，128 KiB） | tiling.h:48 |
| `ARGTOPK_GROUP_THREADS` | 64 | 并行直方图线程数 G（各独占 hist[256] 切片，纯写无争用） | tiling.h:49 |
| `ECHO_PERF_NUM_STEPS` | 31 | 分步打点步数上界（非 spec 23 + spec 8） | tiling.h:57 |

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
`4 × blockDim ≥ topk`（§5 Step6）在 topk ≤ 4096 内恒成立。

编译期 `ECHO_LRU_THREADS`（`build.sh --threads=N` / 环境变量选择；CMake 经
`-DECHO_LRU_THREAD_NUM_OVERRIDE` 同时注入 kernel 与 plugin，**必须同值**，
`CMakeLists.txt:100-113,165-170`）定位为**纯消融 knob**：压低上限即压低自适应结果的
天花板，此时 `topk > 4×上限` 会使 OpMissCumsum 回退 t0 串行（§5 Step6）。正式构建
保持默认 1024。

`EchoLruTiling` 结构（tiling.h:58-81）由 plugin `Check()` 填充：`rowsPerCore/tailRows`
划分行段，`topkBufferB`=capacity，`freeStackSize`=B+min(topk,B)，`effTopk`=min(topk,B)，
`specEnabled`（host 按 min(spl)<maxModelLenP1 判定，本基准恒 0），
`argtopkGroupThreads`=G（`echo_lru_plugin.cpp:68-81`）。kernel 入口从 GM tiling 缓冲逐
字段重载（`echo_lru_kernel.h:1256-1273`）。

### 3.3 单大 `__ubuf__` 数组分段布局

kernel 入口声明 `__ubuf__ int32_t argtopkUb[ARGTOPK_UB_ELEMS]`（128 KiB 静态 UB，
`echo_lru_kernel.h:1279`），`Init` 手动分段（`echo_lru_kernel.h:1006-1018`）：

```
argtopkUb（每核一份, 块内共享, 无 myBlock 偏移）, npad = next_pow2(B+1) = 8192, G = 64
┌───────────────────┬────────────────────────┬──────────┬──────────┬─────────┬───────┐
│ key               │ hist (G 个切片)        │ globalCnt│ segLocal │ segBase │ scalar│
│ [0, npad)         │ [npad, npad+G*256)     │  256     │  G*2     │  G*2    │  8    │
│ 8192              │ 16384                  │          │ (预留)   │ (预留)  │(无读者)│
└───────────────────┴────────────────────────┴──────────┴──────────┴─────────┴───────┘
峰值 = 8192+16384+256+128+128+8 = 25096 < 32768 = ARGTOPK_UB_ELEMS
```

**为什么必须单大数组**：`probe_ub_size` 实测（Init 与 kernel 入口注释，
`echo_lru_kernel.h:1006-1007,1274-1278`）：
- 多个独立 `__ubuf__` 数组会**别名互相覆盖**（读 A 得 D）；
- 单 `__ubuf__` 数组可到 192 KiB（256 KiB 才 trap）。

所以 key/hist/globalCnt/seg/scalar 全部分段塞进一个数组。`segLocal/segBase` 是预留的
分段 scatter 缓冲（基数选择路径未用，`(void)` 压未用告警，`echo_lru_kernel.h:595-596`）；
`scalar` 在 [方向2] 决策本地化后**无任何读者**，槽位保留只为防 UB 布局漂移
（`echo_lru_kernel.h:597`）。

**UB 的分时复用**（同一块内存三个生命周期，互不重叠）：

```
Step1:  argtopkKeyUb_[0..32) 作 warpScratch（warp 归约 miss 计数）, [64] 作 C1 淘汰标志
Step3:  key/hist/globalCnt/scalar 全量投入 radix select
Step6:  ArgtopkAllRows 结束后整块 UB 已死 → argtopkKeyUb_[0..66) 复用作 OpMissCumsum 的 scan 缓冲
        （warpTotals[32] + warpBases[32] + 双控制槽 ctrlA/ctrlB，echo_lru_kernel.h:1011,1116-1117）
```

### 3.4 GM workspace 布局与公式逐项对应

kernel 侧布局在 `Init`（`echo_lru_kernel.h:987-1033`），host 侧公式在
`EchoWorkspaceElements`（`csrc/echo/echo_lru_launch.h:21-30`），两者逐项一致：

```
int32 元素偏移
[0 ───────────────────────────── 每行工作区, 长度 R*(B+1+4+3*topk) ────────────── )
   wsCumsum   R*(B+1)     spec 路径的前缀和暂存（非 spec 不用）
   pushCount  R           spec 路径释放计数
   numHits    R           [P0] 已无读写, 槽位保留防布局漂移 (kernel.h:994)
   numMisses  R           每行 miss 数
   freeIdx    R*topk      radix select 的输出（被淘汰槽列表, 行 stride = topkDim）
   freeSize   R           每行需淘汰数
   rawSlots   R*topk      分配给 miss 的槽
   wsMissRank R*topk      miss 的独占前缀秩
[align2(base) ─────────── 每核调试/打点区, 长度 blocks*(256+8192[+31]) ───────── )
   dbgCount   blocks*256            [诊断3] 每核 pass0 归并直方图 dump (uint32)
   dbgKey     blocks*8192            [诊断3] 每核打包 srcKey dump
   stepCycles blocks*ECHO_PERF_NUM_STEPS   [ECHO_PERF_STEP] 每核每步 t0 clock() 周期
```

对应关系：`base = rows*(capacity + 1 + 4 + 3*topk)` 中的 `1` = wsCumsum 的 `B+1`、
`4` = pushCount/numHits/numMisses/freeSize、`3*topk` = freeIdx/rawSlots/wsMissRank；
`(base+1)&~1` 对应 Init 的 `elemOff = (o+1)&~1`（8 字节对齐）；`perBlock = 256 +
ARGTOPK_GM_NPAD_MAX(8192)`（perf 构建再 +31）。

**注意**：`compare.py` 分配 workspace 用的是**旧版上界公式**（含 GM ping-pong/直方图
占位段，每核 `4*npad + 64*256*2 + 256 + 8192` 个元素，`compare.py:284-292`），严格大于
新版所需，因此同一份测试可同时兼容普通/打点两种构建。`echo_perf_step.py` 的
`echo_step_region_offset`（`echo_perf_step.py:71-75`）则按**新公式**精确定位 stepCycles 区。

### 3.5 栅栏模型：syncthreads + dcci 三明治

两个原语（`echo_lru_kernel.h:1127-1146`）：

- `asc_syncthreads()`：块内栅栏，解决**行内跨线程 RAW**——同一 GM/UB 地址被不同线程
  先写后读（例：OpProtect 以 topkDim 步长写 `pri[row,s]`，OpArgtopkRow 以 B+1 步长读
  `pri[row,i]`，同地址不同线程，`echo_lru_kernel.h:16-18`）。
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

**SyncBlock vs SyncGrid**：代码里两者实现相同（`echo_lru_kernel.h:1127-1146`），语义上
区分"块内协作栅栏"与"步间栅栏"。Step2/Step5 里实际用的是**裸 `asc_syncthreads()`**
（不带 dcci，`echo_lru_kernel.h:1083,1110`）——因为那里要解决的是"**读先于写**"的排序
（Bug2，§5），写在栅栏之后发生，不存在 stale 读问题；后续读者可见性由更后面的完整
SyncGrid 兜底。

---

## 4. warp 协作原语

`PackedBlockScan`（`echo_lru_kernel.h:55-77`）是全部 warp 协作的基础（OpArgtopkRow 的
emission、OpMissCumsum 的秩扫描都用它），对照 SIMT 基线的 `BlockExclusiveScan`
（`sparse_kv_plan_kernel.cpp:107-139`）同构。

### 4.1 两级结构

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
asc_syncthreads();                                 // ── 栅栏 1: warpTotals 写 → t0 读
if (threadIdx.x == 0u) {                           // 二级: t0 串行扫 warp 总和
    base = 0;
    for (w = 0; w < warpCount; ++w) { warpBases[w] = base; base += warpTotals[w]; }
    control[0] = base;                             //   全块 grand total
}
asc_syncthreads();                                 // ── 栅栏 2: warpBases/control 写 → 全员读
return warpBases[warp] + inclusive - packedValue;  // exclusive = warp 基址 + warp 内 inclusive − 自身
```

- **一级**（warp shuffle）：warp 内 32 lane 做 inclusive 前缀和，5 轮 shuffle 完成，
  warp 同步由硬件保证，无需栅栏。`lane >= offset` 时才累加，保证 exclusive 语义可由
  `inclusive - packedValue` 得到。
- **二级**（t0 扫 warpTotals）：warp 数 ≤ 32（线程数 ≤1024），t0 串行扫一遍得到每个
  warp 的基址与全块总和。t0 串行 32 次UB 读远比再做一层树形扫描便宜（且避免第二层栅栏）。

### 4.2 字段打包防进位规则

打包值是两个字段拼进一个 int32：`value = 字段A | (字段B << shift)`。**两个字段各自的
块内总和都必须 < (1 << shift)**，这样打包加法在各自字段内进位不会越界污染另一字段
（`echo_lru_kernel.h:45-47` 注释，"由调用方保证"）。两个调用点：

| 调用点 | 打包 | 字段上界论证 |
|---|---|---|
| OpArgtopkRow emission（`echo_lru_kernel.h:565`） | `nLt | (nEq << 16)` | 块内总和 ≤ validN = B+1 ≤ 4097 < 2^16（`echo_lru_kernel.h:564`） |
| OpMissCumsum 2 元素路径（`:772`）/ 4 元素路径（`:848-849`） | `ind0 \| (ind1 << 12)`；4 元素再链一次 `ind2 \| (ind3 << 12)` | 块内各字段（各 tile）总和 ≤ topkDim ≤ 2048 < 2^12（`:729-732,789-791`） |

### 4.3 为什么只有 2 次栅栏（第 3 次的删除论证）

SIMT 基线的 `BlockExclusiveScan` 有第 3 次栅栏："保证所有线程已读取 warpBases，下一次
scan 才能复用它"（`sparse_kv_plan_kernel.cpp:105`）。Echo 的 P4 把它删了，
`echo_lru_kernel.h:49-54` 给出顺序论证（前提：**所有调用方都是统一控制流的逐行循环**，
每线程经历的栅栏数相同）：

- **(a) 线程 X 读 warpBases/control**（栅栏 2 之后、返回前）**vs t0 下一轮写它们**
  （下一轮栅栏 1 之后）：t0 要写必须先等所有线程（含 X）到达下一轮栅栏 1；而 X 的读
  发生在 X 到达下一轮栅栏 1 **之前** → **读先于写**，安全；
- **(b) t0 读 warpTotals**（栅栏 1~2 之间）**vs lane31 下一轮写它**（栅栏 2 之后）：
  t0 的读在 t0 到达栅栏 2 之前完成；lane31 的写在栅栏 2 释放**之后**才发生 →
  **读先于写**，安全。

两对竞争均被"下一轮的栅栏 1 / 栅栏 2"天然隔开，故第 3 次栅栏是纯冗余。这正是
"1024 线程栅栏一次数千周期"（§7）背景下的必要抠法。

### 4.4 warp 数按 blockDim.x 运行时推导

`warpCount = blockDim.x >> 5`（`echo_lru_kernel.h:59`）而不是编译期常量，原因
（`echo_lru_kernel.h:42`）：同一份源码要在多处跑——设备 blockDim 由 plugin 自适应
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

## 5. 流水线逐步解读（23 个非 spec 步）

### 5.1 总表

`Process()`（`echo_lru_kernel.h:1039-1123`）内非 spec 路径共 23 个打点步，步名与
`echo_perf_step.py` 的 `ECHO_PERF_STEP_NAMES`（`echo_perf_step.py:53-62`）一一对应。
打点由 `PerfBegin/PerfEnd` 包裹（t0 的 `clock()` 差值写入 workspace 尾部，
`echo_lru_kernel.h:1148-1170`）。

| # | 步名 | 函数 | 做什么（读 → 写） | 行内跨线程 RAW / 栅栏 |
|---|---|---|---|---|
| 0 | （无打点） | OpFillZero + SyncGrid | t0 清 pushCount（spec 用）；prologue 不在 23 步内 | — |
| 1 | Step0.OpResetRows | OpResetRows | resetMask=1 行：htd 整行 `col==0?0:I32_MAX` | 各线程写各自列，无 RAW |
| 2 | Step0.OpResetDthPri | OpResetDthPri | dth=I32_MAX；pri=`col==0?I32_MAX:-1` | 同上 |
| 3 | Step0.OpResetFree | OpResetFree | free[col<B]=B−col，其余 0 | 同上 |
| 4 | Step0.OpResetAvailFifo | OpResetAvailFifo | t0：avail=B，fifo=1 | t0-only；可见性由步 5 兜底 |
| 5 | Step0.SyncGrid | SyncGrid | — | reset 写（含 t0 的 avail/fifo）→ Step1 读 htd（全员）/ avail（t0）|
| 6 | Step1.OpHitMiss | OpHitMiss | pos/htd → curSlots；miss 计数 warp 归约；t0 写 numMisses/freeSize/C1 标志 | 2 栅栏/行（末行 1，C2） |
| 7 | Step1.SyncGrid | SyncGrid | — | curSlots 写 → Step2 读；numMisses/freeSize（t0 GM 写）经 dcci 对全员可见 |
| 8 | Step2.OpProtect | OpProtect | curSlots/fifo → pri[hit 槽]=fifo | pri 写 → Step3 读（由步 11 解决） |
| 9 | Step2.SyncBlock | 裸 syncthreads | — | **Bug2**：全员 fifo 读完成先于 t0 fifo++ |
| 10 | Step2.OpFifoInc | OpFifoInc | t0：fifo++（全部行） | t0-only |
| 11 | Step2.SyncGrid | SyncGrid | — | pri 写 → Step3 argtopk 读；fifo 新值对 Step3/5 可见 |
| 12 | Step3.ArgtopkAllRows | [C1] needFree=0 时**跳过** | radix select：pri/fifo → freeIdx | 每行内部自带栅栏（§6） |
| 13 | Step3.SyncGrid | [C1] 同上跳过 | — | freeIdx 写 → OpFreeRelease 读 |
| 14 | Step3.OpFreeRelease | [C1] 同上跳过 | freeIdx/dth → pri=−1、dth=I32_MAX、htd[tok]=I32_MAX | j<fs 才执行 |
| 15 | Step3.OpFreePush | [C1] 同上跳过 | freeIdx → free[avail+j] | 本 kernel 内**无读者**（P4） |
| 16 | Step4.OpAllocPop | OpAllocPop | numMisses/freeSize/avail/freeIdx/free → rawSlots | 读 freeIdx（步 13 后）、avail（步 7 后） |
| 17 | Step4.SyncGrid | SyncGrid | — | rawSlots 写 → Step5 读 |
| 18 | Step5.OpAllocPriority | OpAllocPriority | rawSlots/fifo → pri[aslot]=fifo | pri 写，本 kernel 内此后无读者 |
| 19 | Step5.SyncBlock | 裸 syncthreads | — | **Bug2**（同步 9）+ 排空 avail 全员读者 |
| 20 | Step5.OpFifoAvailUpdate | OpFifoAvailUpdate | t0：fifo++；avail += fs − nm | t0-only；尾 SyncGrid 已删（P4） |
| 21 | Step6.OpMissCumsum | OpMissCumsum | curSlots/pos → wsMissRank/missHostPos | 2/4 元素路径 PackedBlockScan 各 2/4 栅栏/行 |
| 22 | Step6.SyncGrid | SyncGrid | — | wsMissRank/missPos 写 → OpSetUpdate 读 |
| 23 | Step6.OpSetUpdate | OpSetUpdate | wsMissRank/rawSlots/missHostPos → htd/dth/curSlots/flat/mask | 行内 pos 不重复契约保证写地址不冲突 |

### 5.2 每步要点

**Step0（reset 四连）**：只对 `resetMaskGm[row]!=0` 的行执行；行内 blockDim.x 线程
（运行时自适应，§3.2；生产 topk=2048 → 512）grid-stride 扫列（`echo_lru_kernel.h:93-158`）。htd 行长 `maxModelLenP1`=262144 int32=1 MiB，cold
场景 32 行全 reset → 每核 1 MiB 纯写，是 cold 场景的带宽地板项（§10）。步 5 的 SyncGrid
同时解决"全员写 htd → 全员读 htd"与"t0 写 avail/fifo → t0 读 avail"两类依赖。

**Step1：warp 归约计数 + fs 折叠 + C1 跳过标志**（`OpHitMiss`，`echo_lru_kernel.h:265-312`）：

```
每线程: for j = t; j < topkDim; j += blockDim.x
            slot = htd[row, clamp(pos[j])]; curSlots[j] = slot
            slot == I32_MAX → localMiss++                     // j≥effTopk: curSlots[j]=0
warp 内:  warpSum = asc_reduce_add(localMiss)                 // 零栅栏
lane0:    warpScratch[warp] = warpSum                          // UB (argtopkKeyUb_)
── 栅栏 1 ──────────────────────────────────────────
t0:       nm = Σ warpScratch[0..warpCount)
          numMisses[row] = nm
          fs = max(nm - availGm[row], 0);  freeSize[row] = fs   // [P4] 原 OpFreeSize 折进来
          fs != 0 → fsAny = 1
── 栅栏 2（仅当还有下一行, [C2] 末行省略）──────────
行循环结束后: t0: warpScratch[64] = fsAny                      // [C1] 淘汰标志, UB 槽 64
```

四个设计点：

1. **[P0] 去 atomic**：原实现每线程一次 `asc_atomic_add` 打同一 GM 地址，warm 场景
   ~1024 次/行 × 同地址 atomic 串行 ~9K 周期 → 该步实测 ~9.4M 周期（根因证据：cold 无
   atomic 仅 ~10K，hit 数 1536 vs 2048 耗时几乎相同 → 耗时与 atomic 次数强相关、与 hit
   数无关）。改 warp 归约 + t0 汇总后一次 GM 写搞定（`echo_lru_kernel.h:253-256`）。
    `numHitsGm` 槽位保留但不再读写（防 workspace 布局漂移，`echo_lru_kernel.h:262,994`）。
2. **[P4] fs 折叠**：`fs = nm - avail`（clamp≥0）在 t0 汇总段顺手写出，省掉原独立
   OpFreeSize 步及其 SyncGrid（`echo_lru_kernel.h:257-262`）。`nm` 直接数
   `curSlots==I32_MAX` 的个数，与原公式 `effTopk - numHits` 恒等。
3. **[C2] 末行栅栏**：栅栏 2 的唯一作用是保护 warpScratch 的下一行复用（t0 读 vs 下一
   行 lane0 写）；末行无后续写入，省一次（`echo_lru_kernel.h:304-306`）。t0 的 GM 写可见
   性由步 7 的 SyncGrid 兜底。
4. **[C1] 标志**：`warpScratch[64]`（UB）写"本核任一行 fs>0"，经步 7 的 SyncGrid 对全
   块可见；槽 64 与 warp 归约占用的 [0,32) 不冲突（`echo_lru_kernel.h:308-311`）。

**Step2：protect 与 Bug2 栅栏**（`OpProtect` + 裸 syncthreads + `OpFifoInc`）：

OpProtect 每行每线程都读一次 `fifoGm[row]`（`echo_lru_kernel.h:325`），再写
`pri[row, hit 槽]=fifo`。**Bug2**（`echo_lru_kernel.h:1083-1087` 的 fix 注释）：若没有步 9
的栅栏，t0 会率先跑完自己在各行的列片、提前进入 OpFifoInc 把**所有行**的 fifo+1；而其
他线程还散在 OpProtect 的各行里，后到的线程读到 +1 后的 fifo → `pri` 多 1（实测诊断：
row143 slot3706 kernel=12 golden=11 diff+1）→ argtopk 选错 → 全链错。栅栏保证"**所有
线程完成全部行的 fifo 读+pri 写后，t0 才能自增**"。这里用裸 syncthreads 即可（无 dcci）：
竞争是"读（栅栏前）vs 写（栅栏后）"，写后于栅栏发生，不存在 stale 读；fifo 新值对
Step3/Step5 读者的可见性由步 11 的完整 SyncGrid 负责。

**Step3：C1 条件执行**（`echo_lru_kernel.h:1091-1101`）：

```cpp
bool needFree = (argtopkKeyUb_[64] != 0);   // 步 7 之后读, 块统一条件
PerfBegin(); if (needFree) ArgtopkAllRows();                       PerfEnd();
PerfBegin(); if (needFree) SyncGrid();                             PerfEnd();
PerfBegin(); if (needFree) OpFreeRelease(...);                     PerfEnd();
PerfBegin(); if (needFree) OpFreePush(...);                        PerfEnd();
```

全 0（本核所有行 fs=0）时四步整体跳过，**步数/步名不变（打点记录 ≈0 周期）**。安全性
论证（`echo_lru_kernel.h:1092-1096`）：fs=0 时 OpArgtopkRow 本来就走 fsClamp==0 提前
返回；OpFreeRelease/OpFreePush 的 `j<fs` 循环体不执行；**freeIdx 无任何读者**——fs=0
时 OpAllocPop 的 `readIdx = avail + 0 - nm + j`，由 fs=0 ⟹ nm≤avail 得 readIdx <
avail，恒走老栈路径（§5 Step4），从不读 freeIdx；avail 也不变（+=0）。条件对全块统一
（单一 UB 标志），**栅栏无发散**。warm/cold 场景由此跳过整条淘汰流水线。

Argtopk 本身见 §6。OpFreeRelease（`echo_lru_kernel.h:602-631`）对 j<fs 的每个淘汰槽：
`pri=-1, dth=I32_MAX, htd[tok]=I32_MAX`（tok≠I32_MAX 时）。OpFreePush（`:633-654`）：
`free[avail+j] = freeIdx[j]`——把新释放槽压回栈；P4 之后**本 kernel 内它没有读者**
（OpAllocPop 直读 freeIdx，见下），其写入只为持久化给下一次调用，因此 FreePush 之后
不再需要 SyncGrid。

**Step4：OpAllocPop 直读 freeIdx 的等价性推导**（`echo_lru_kernel.h:656-699`）：

旧版序列：FreePush 写 `free[availOld+k] = freeIdx[k]`（k<fs）→ avail += fs →
avail −= nm → 弹出 `free[availOld + fs − nm + j]`（j<nm）。

新版 avail 更新推迟到 Step5 尾，本步读到的 avail 恒为初值 availOld，弹出索引
`readIdx = availOld + fs − nm + j` 拆两段：

- **readIdx ≥ availOld**：落在新压入段。旧版读 `free[availOld + k]`（k=readIdx−availOld
  ∈ [0, fs)），而 FreePush 恰好写的是 `free[availOld+k] = freeIdx[k]` ⟹
  **`free[availOld+k] == freeIdx[k]`**，直接读 `freeIdx[k]` 逐值等价，且**不依赖
  FreePush 的写入是否完成**（FreePush→AllocPop 间的 SyncGrid 因此可删）；
- **readIdx < availOld**：老栈区段 `free[readIdx]`，本 kernel 从未写该区段（FreePush
  只写 [availOld, availOld+fs)），读老栈原值即可。

两段合起来与旧版栈弹出**逐值一致**（`echo_lru_kernel.h:657-664` 的注释即此推导；
`checks` 第 4 组用旧栈 golden 逐值验证，`check_echo_kernel_host.py:237-276`）。
`j ≥ nm` 的位置填 0（无效）。

**Step5：OpFifoAvailUpdate 的 avail 终值恒等式**（`echo_lru_kernel.h:352-372`）：

```
终值 avail' = availOld + fs − nm
旧版两步序列: (availOld + fs) − nm = availOld + fs − nm        ← 恒等
非负性: fs = max(nm − availOld, 0)
   nm > availOld 时: fs = nm − availOld ⟹ avail' = 0
   nm ≤ availOld 时: fs = 0     ⟹ avail' = availOld − nm ≥ 0
```

依赖分析（`echo_lru_kernel.h:356-358`）：t0 读的 fifo/freeSize/numMisses 都是 t0 自己
在 Step0~1 写的（同线程 GM RAW 安全）；avail 的全员读者（OpFreePush/OpAllocPop）已被
步 17 与步 19 排空，t0 写不与他们竞争。步 19 的裸 syncthreads 同样是 Bug2 结构：全员
在 OpAllocPriority 里的 fifo 读必须先于 t0 在本步的 fifo++。**尾 SyncGrid 已删**：此步
之后 pri/fifo/avail 在本 kernel 内再无读者，kernel 结束即对 host 可见
（`echo_lru_kernel.h:1113`）。

**Step6：两级扫描 rank 拼装与 SetUpdate 回写**：

`OpMissCumsum`（`echo_lru_kernel.h:739-886`）按门条件选三条路径之一（Bm =
blockDim.x）。每线程认领的都是**连续映射**位置（`p = t + k*Bm`，理由同 §6.6
emission 的连续切片——rank 必须按位置序拼装，grid-stride 会得到"线程主序"）：

| 路径 | 门条件 | 每线程位置 | 扫描结构 | 栅栏/行 |
|---|---|---|---|---:|
| 2 元素 | `topkDim ≤ 2*Bm && 2*topkDim ≤ 8192`（`:749`） | p0=t, p1=t+Bm | 1 次 PackedBlockScan | 2 |
| 4 元素 | `topkDim ≤ 4*Bm && 4*topkDim ≤ 8192`（`:784`） | p0..p3 = t, t+Bm, t+2Bm, t+3Bm | 2 次链式 PackedBlockScan | 4 |
| 串行 | 以上均不满足 | — | t0 逐位累加（`:872-885`） | 0 |

自适应 blockDim（§3.2）下的路径分布：生产 topk=2048 → Bm=512，2 元素门失效
（2048 > 2×512），由 **4 元素路径**接住（2048 ≤ 4×512 且 4×2048=8192 恰好过
UB 门）——4 元素路径即生产路径；小 topk（≤ 128，Bm 压到 64 下限）命中 2 元素
路径（topk ≤ 2×64）；topk > 2048 时 4 元素 UB 门（4×topk ≤ 8192）失效、2 元素门
所需 Bm=topk/2 又超过 1024 launch 上限 → t0 串行（超出当前生产 topk 形态）。
消融 knob 把上限压到 256 时（4×256=1024 < 2048）同样回退串行——t256 消融崩塌的
根因（§9.10）。

**2 元素路径的 rank 拼装**（`echo_lru_kernel.h:729-732,772-781`）：

```
e0(t) = Σ_{t'<t} ind0(t') = 位置 [0, t) 的 miss 数            → rank(p0=t)   = e0
e1(t) = Σ_{t'<t} ind1(t') = 位置 [Bm, Bm+t) 的 miss 数
total0 = control[0] & 0xFFF = 前半段 miss 总数
rank(p1 = t+Bm) = total0 + e1(t)                          ← 后半段的排名 = 前半段总数 + 后半段内部前缀
```

**4 元素路径的 rank 拼装**（`echo_lru_kernel.h:785-809,847-867`）：位置序切成四个
tile——[0,Bm) 由各线程 p0 覆盖、[Bm,2Bm) 由 p1、[2Bm,3Bm) 由 p2、[3Bm,4Bm) 由 p3。
两次链式扫描（12bit 字段安全：各 tile 的 miss 总数 ≤ topkDim ≤ 2048 < 4096，
`echo_lru_kernel.h:789-791`）：`excl1 = scan(ind0 | ind1<<12)` 覆盖 [0,2Bm)、
`excl2 = scan(ind2 | ind3<<12)` 覆盖 [2Bm,4Bm)（`:848-849`）。**双控制槽**：
PackedBlockScan 会把全块 grand total 写入传入 control 的 `control[0]`，第二次扫描
会覆盖第一次的总数，故 ctrlA=`scanUb+64`（scan1 用）、ctrlB=`scanUb+65`（scan2 用）
（`:812-813`）；warpTotals/warpBases 两次共享，顺序安全由 scan1 的栅栏 2 / scan2 的
栅栏 1 隔开（论证同 §4.3 的 (a)/(b)，`:792-798`）。rank 拼装（`:852-867`）：

```
t0  = ctrlA[0] & 0xFFF          = [0,Bm) 的 miss 总数
t01 = t0 + (ctrlA[0] >> 12)     = [0,2Bm) 的 miss 总数
t2  = ctrlB[0] & 0xFFF          = [2Bm,3Bm) 的 miss 总数
rank(p0=t)     = e0
rank(p1=t+Bm)  = t0  + e1       ← ⚠ 不能写 t01+e1：p1 在 tile1 段内而非段后，
rank(p2=t+2Bm) = t01 + e2          多加 tile1 自身总数会让 rank 偏大（`:799-809`）
rank(p3=t+3Bm) = t01 + t2 + e3
```

（rank(p1) 的这个陷阱曾在实现规格里以 t01+e1 的形式出现，被 CPU 回归 golden 拦下，
commit d20fc50。）串行回退与两条并行路径语义逐字段一致（`checks` 第 3 组三路同
golden）。scan 缓冲复用 argtopkKeyUb_（此刻 radix select 已结束；[0,66) =
warpTotals[32]+warpBases[32]+ctrlA/ctrlB）。

`OpSetUpdate`（`echo_lru_kernel.h:888-933`）对每个 miss 位 j：
`rank = wsMissRank[j]` → `aslot = rawSlots[rank]` → `htd[row, mpos] = aslot`、
`dth[row, aslot] = mpos`、`missAllocFlat = row*B + aslot`、`missMask = 1`。**[P2] 顺手
回写 `curSlots[j] = aslot`**：miss 的最终槽位就是本轮分配的 aslot，hit 位 curSlots 在
Step1 已正确，独立的全量 gather 步（OpFinalSlots）是纯冗余，[P2] 已删除
（`echo_lru_kernel.h:923-926`）。并行写 htd/dth 的地址不冲突由**输入契约**保证：同一行
内 pos 不重复（一行 = 一个 request，topk 选出的 token id 必不重复，
`echo_lru_kernel.h:897-900`）。

**spec 路径（Step0.5，本基准不覆盖）**：`specEnabled=1` 时在 Step0.SyncGrid 之后插入
8 步（4 个 Op + 4 个栅栏）：`OpInvalidateSpecSuffix`（驱逐 `tok ≥ spl` 的投机后缀槽，
写 wsCumsum 标记）→ `OpSpecCumsum`（t0 前缀和 + pushCount）→ `OpSpecPushFree`
（差分判定 `cum[slot]−cum[slot−1]==1` 找回真 spec 槽压栈——**不能用 `cum>0`**，否则
末段非 spec 槽会误判并发覆盖，见 `echo_lru_kernel.h:221-231` 的实战记录）→
`OpSpecAvailAdd`。

---

## 6. Argtopk 基数选择深读

本章是 Echo 的性能核心。`OpArgtopkRow`（`echo_lru_kernel.h:403-598`，头注释
`:377-402`）解决：**从 B+1 个打包 key 中选出 pri 最小的 fsClamp 个槽，平局按槽号
升序，写入 freeIdxOut**。

### 6.1 目标与 key 打包

```
key[i] = (i==0 || i>B || pri[i]<0) ? I32_MAX : pri[i]        // i ∈ [0, validN=B+1)
```

全非负、无效/填充沉底为 I32_MAX（`echo_lru_kernel.h:379,456-459`）。选择目标等价于
稳定排序 `(key, slot)` 字典序取前 fsClamp 个（`checks` 用 stable_sort 交叉验证，
`check_echo_kernel_host.py:154-161`）。

**key 数组只写 [0, B+1)**：pack 循环上界是 validN（`echo_lru_kernel.h:456`）。padding
区 [B+1, Npad) 的 Npad−(B+1) 个键（恒为 I32_MAX）**从不物化**——它们只在两处被
"解析式"消费：merge 相位计入直方图（下文）、emission 越界时按序补索引。旧版满扫描
要写满 8192 个键，[B] 重构省掉约一半打包写入。CPU 回归显式断言 padding 区未被触碰
（`check_echo_kernel_host.py:167`）。

### 6.2 每趟结构：[clear+hist] → B1 → [merge] → B2 → [本地决策]

4 趟 MSB-first（pass 0 看 byte3，pass 3 看 byte0，`shift = (3-pass)*8`）。一趟的相位图：

```
┌──────────────── 相位 (a) clear+hist ────────────────┐   ┌─ (b) merge ─┐   ┌─ (c) 本地决策 ─┐
│ 首迭代: 全员 pack key[0,validN) (写 UB)             │   │ 256 桶并行: │   │ 全员各自重算    │
│         G 路清零各自 hist[256] 切片                  │ B1│ 每桶归并 G  │B2│ b/prefix/      │
│         G 路直读 priGm 统计本字节直方图              │ → │ 个切片      │ →│ remaining      │
│ 后续趟: G 路清零 + 读 key[] 统计(带 prefix 过滤)     │   │ +padding    │   │ (寄存器 T/quota)│
└─────────────────────────────────────────────────────┘   └─────────────┘   └────────────────┘
```

- **首迭代单相位**（[B] 重构，`echo_lru_kernel.h:451-475`）：pack 写 key（UB）、hist0
  **直读 priGm**（GM）——两者操作不同数组，无交叉 RAW，可合并在同一相位、共用一次
  栅栏。首趟若 `startPass>0`，hist 带 `prefix=0` 候选过滤：`(v >> (shift+8)) == 0`
  （key 的高于本字节的全零）；I32_MAX 键 byte3=0x7F≠0 被天然排除（`echo_lru_kernel.h:465-475`）。
- **后续趟**：hist 读 key[]，过滤条件改为高字节匹配当前 `prefix`
  （`(v >> (shift+8)) == (prefix >> (shift+8))`，`echo_lru_kernel.h:481`）。
- **G=64 路纯写切片**：`ARGTOPK_GROUP_THREADS` 个线程各写自己独占的 `hist[t*256..]`
  切片，**无 atomic、无争用**（tiling.h:40-41）。

**B1/B2 保护的读写对**（跨趟复用的安全性）：

```
hist 切片:  pass p 的 merge 读 (B1~B2 之间)  vs  pass p+1 的 clear+hist 写 (B2 之后)  → B2 分隔 ✓
globalCnt:  pass p 的决策读 (B2 之后)         vs  pass p+1 的 merge 写 (B1(p+1) 之后)  → B1(p+1) 分隔 ✓
key:        pass p 的 pack 写 (B1(p) 之前)    vs  pass p+1 的 hist 读                 → B1(p)+B2(p) 分隔 ✓
```

### 6.3 merge：256 桶并行 + padding 解析式计入

```cpp
uint32_t padByte = (0x7FFFFFFFu >> shift) & 0xFFu;       // I32_MAX 在本趟字节的桶号
bool padIn = (pass == 0u) ? true                          // pass 0 无前缀, 全部候选
                          : ((0x7FFFFFFFu >> (shift+8u)) == (prefix >> (shift+8u)));
    for (uint32_t b = t; b < 256; b += blockDim.x) {          // 256 桶摊给全块线程 (blockDim.x 自适应)
    int32_t s = 0;
    for (uint32_t th = 0; th < G; ++th) s += hist[th*256u + b];   // 每桶归并 G 个切片
    if (padIn && b == padByte) s += (int32_t)padN;        // padding 键解析式计入
    globalCnt[b] = s;
}
```

（`echo_lru_kernel.h:488-503`。）

- **256 桶并行**：旧版是 t0 串行扫 G×256=16384 项；现在全块线程各领 ≤1 桶（循环步长
  blockDim.x，≥256 线程时每线程至多一桶）、每桶归并 G=64 个切片，**墙钟只有 G 次读**（[B] 注释，
  `echo_lru_kernel.h:382-383`）。
- **padIn 的语义**：padding 键全是 I32_MAX，它们"在候选集合里"当且仅当选择前缀已经
  走到 I32_MAX 的路径上（I32_MAX = 0x7FFFFFFF，字节路径 0x7F,0xFF,0xFF,0xFF，与 prefix 逐字节相等）。
  pass 0 无前缀，全部候选（padIn=true）。计入的量 padN = Npad − (B+1) 是编译期可算
  的常数——这就是 padding 不需要物化的全部原因。
- **两个编译器坑的规避**就写在这段附近（§7.6）：pass==0 分支不用 `||` 短路（文本 UB
  `v>>32` 会被 bisheng 投机求值），改 if/else。

### 6.4 决策本地化（方向 2）：为什么能去掉第 3 次栅栏

决策不经过任何广播（`echo_lru_kernel.h:505-531`）：**桶决策是 globalCnt 的确定性函数**
——给定相同的 globalCnt 快照与相同的 remaining，任何线程执行同一段循环必然得到相同的
b/prefix/remaining/T/quota。于是让**每个线程各自重算**（本数据 break 出现在前几个桶，
每线程 ≤4 次 UB 读，`echo_lru_kernel.h:392-394`），T/quota 留在**寄存器**里直接给
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
成立**：b-loop 在 `cum + c ≥ remaining` 时 break 且 `cum < remaining`（`:529` 注释），
这保证下一趟必有候选、决策不会空转。

### 6.5 自适应跳趟（方向 1）：startPass、候选过滤与回退

```cpp
uint32_t startPass = 3u;
if      ((fifoVal >> 24) != 0u) startPass = 0u;   // fifo 高字节非零 → 全趟
else if ((fifoVal >> 16) != 0u) startPass = 1u;   // 前导零字节 z=1 → 跳 pass 0
else if ((fifoVal >> 8 ) != 0u) startPass = 2u;   // z=2 → 跳 pass 0..1
                                                   // z=3 → 只跑 byte0
```

（`echo_lru_kernel.h:428-434`。）

**依据**：pri 的唯一写入来源是 fifo（OpProtect/OpAllocPriority 写 `pri=fifo`），fifo
单调递增，故**当前所有有效 pri ≤ fifoVal**（`echo_lru_kernel.h:386-387`）。若 fifoVal
有 z 个前导零字节，则所有有效 pri 的前 z 个字节全零——**pass 0..z−1 对这些键的直方图
全落在桶 0，prefix 不变、不产生任何信息**，直接从 byte(3−z) 起跑。基准 fifo=4097
(0x1001) → z=2 → 只跑 byte1/byte0 两趟。

具体到 full_cache_75pct：初始 `pri[slot]=slot`（§1.5），hit 槽被 Step2 保护为 4097，
非 hit 占用槽保持 1537..4096；fs=512 → 选 1537..2048。这些键 byte1 ∈ 0x06..0x08，
两趟内即可收敛（byte1 定桶 → byte0 定位）。

**prefix=0 候选过滤**：起跑趟 >0 时首趟直方图只统计 `(v >> (shift+8)) == 0` 的键
（高字节全零的小键）。I32_MAX 键（无效槽/padding，byte3=0x7F）被天然排除——它们本来
就"沉底"，只有当选择需要深入到 I32_MAX 值域时才相关。

**放不下 → 回退全趟**：若 fsClamp 超出 prefix=0 候选总数（b 扫到 256 仍不满足），说明
存在大量"高字节非零"的键（比如 pri 远大于 fifo 的异常状态），快路径失效。此时
`pass=0, remaining=fsClamp, prefix=0` 重来一次完整 4 趟（key 已打包，不重 pack）。
全趟候选 = 全部 Npad 个键 ≥ fsClamp（fsClamp ≤ topkDim ≤ Npad），**必然收敛**
（`echo_lru_kernel.h:534-540`）。

**健壮性不依赖 pri ≤ fifo**：跳趟只是启发式；启发式失效（pri>fifo 的键真实存在时）
的表现是"快路径放不下 → 回退"，**任何输入下输出与全趟版逐位一致**——失效只损失
性能，不损失正确性（`echo_lru_kernel.h:389-391`，commit 145d3ce 信息）。CPU 回归用
4 个 fifo 变体（0x40000000 全趟 / 4097 跳2趟 / 65536 跳1趟 / 100 只跑 byte0）×
全部 B/pattern 组合验证跳趟与回退路径输出 golden 与全趟逐位一致，含全 I32_MAX 键等
强制回退用例（`check_echo_kernel_host.py:98-101,171-179`）。

**外层 attempt 循环的写法**：`for (attempt = 0; attempt < 2; ++attempt)` + 迭代边界
break，**刻意不用 `while(true)` + 数据依赖 continue**——后者会跨栅栏结构段错误（§7.6），
前者与旧版 scalar[3] break 同构、已验证可编译（`echo_lru_kernel.h:442-446`）。

### 6.6 emission：连续切片不变量与 padding 补齐

选出 T/quota 后，全员把选择结果写出去（`echo_lru_kernel.h:543-593`）：

```
1) 每线程统计自己连续切片 [lo, hi) 内:  nLt (key<T 个数) | nEq (key==T 个数) 打包
2) PackedBlockScan(nLt | nEq<<16) → excl=(wLt,wEq) 本线程之前所有切片的累计,
   control[0]=(totalLt,totalEq) 全块总计                          ← 2 次栅栏
3) 重扫切片写 freeIdxOut:
      key < T                    → freeIdxOut[wLt++]  = i          (槽号升序段)
      key == T 且 wEq < quota    → freeIdxOut[totalLt + wEq++] = i (平局段, 取前 quota 个)
4) padding 补齐: m = fsClamp − totalLt − min(quota, totalEq)
      freeIdxOut[used + k] = validN + k   (k = t; k < m; k += blockDim.x)   → 索引 [B+1, B+1+m)
5) 尾部补零: freeIdxOut[fsClamp .. topkDim) = 0
6) syncthreads; t0 dcci(freeIdxOut)                               ← 2 次"尾"栅栏
```

**连续切片不变量（为什么不能 grid-stride）**（`echo_lru_kernel.h:544-547`）：切片必须
是 `lo = t*chunk` 的**连续区间**。若用 grid-stride（线程 t 拿 {t, t+blockDim.x, …}），扫描
给出的写入基址按"线程主序"分配，各线程写出的键顺序会变成线程交错序——freeIdx 的
输出顺序就不再是全局槽号升序，与原语义（稳定排序的 (pri,slot) 序）逐位不一致。连续
切片下：切片内按 i 升序、切片间按线程号升序 = 全局槽号升序。

**输出语义**：先全部 key<T（槽号升序），再前 quota 个 key==T（槽号升序），越过有效区
时补 padding 索引，尾部补 0。**key==T 平局取最小槽号**，与稳定排序 (pri, slot) 字典序
一致（`echo_lru_kernel.h:400-402`）。

**padding 补齐何时触发**：`m > 0` ⟺ quota > totalEq（有效区 ==T 键不够配额）。由选择
不变量（计数含 padding）保证 `0 ≤ m ≤ padN`。基准形状下 `topkDim ≤ B < validN`，有效
区键数总够，m 恒为 0；该路径为接口一般性（topkDim > B 的合法输入）保留，与旧满扫描
版本逐位一致（`echo_lru_kernel.h:548-550,580-586`）。

### 6.7 栅栏计数演变（基准路径）

| 版本 | 每趟栅栏 | 趟数 | emission | 合计 |
|---|---:|---:|---:|---:|
| 手抄版（4-pass LSD 排序） | 4 | 4 | — | 16+ |
| [B] 重构后 | 3 | 4 | 4 | 16 |
| 方向 1+2（145d3ce，fifo=4097 跳 2 趟） | 2 | 2 | 2（扫描）+2（尾） | **8** |

（`echo_lru_kernel.h:395-397`："快路径+全趟合计: 基准只跑 2 趟 × 2 栅栏 + emission
扫描 2 + 尾 2 = 8 次栅栏 (B 前为 16)"。）

---

## 7. 硬件特性与避坑清单

每条给出代码内证据或实测教训。这些是 dav-3510（Ascend 950，`--npu-arch=dav-3510`，
CMakeLists.txt:131）纯 SIMT 后端的经验总结，改动相关代码前先过一遍本清单。

### 7.1 同地址 GM atomic：~9K 周期/次，串行化严重

- **现象**：P0 前的 OpHitMiss 每线程一次 `asc_atomic_add` 打同一 GM 地址，warm 场景
  ~1024 次/行，该步实测 ~9.4M 周期（占绝对大头）。
- **证据**：cold 场景 0 hit 无 atomic 仅 ~10K 周期；hit 数 1536（warm75）与 2048
  （warm_all）两场景耗时几乎相同 → 耗时与 atomic 次数强相关、与有效工作量无关
  （`echo_lru_kernel.h:253-256`）。
- **规避**：块内计数一律走 warp 归约/扫描（`asc_reduce_add`/`asc_shfl_up`）+ t0 汇总
  单写。CPU 回归甚至断言 `atomicCalls == 0`（`check_echo_kernel_host.py:209`）。

### 7.2 GM 写后读需要 dcci，否则 stale

- **现象**：单核内线程 A 写 GM、栅栏后线程 B 读同地址，可能命中 stale D-cache 行
  （hybrid 版 "stale pri" 根因，`echo_lru_tiling.h:7-8`）。
- **规避**：步间统一走 `SyncGrid()` 三明治（syncthreads + t0 `asc_dcci_entire` +
  syncthreads，§3.5）。注意 dcci_entire 整核生效，t0 做一次足够。

### 7.3 块内栅栏：数千周期/次（随线程数伸缩），栅栏本身是大成本

- **实测**：`asc_syncthreads`（1024 线程块）约 4~8K 周期（kernel 注释记 "~6-8K 周期，
  fs=0 早退路径实测"，`echo_lru_kernel.h:40-41`）。
- **教训**：能 warp 内解决的不上块栅栏（P0/P3）；每个栅栏要么给出保留论证（依赖
  分析），要么删（P4 删 3 个无 RAW SyncGrid、PackedBlockScan 删第 3 栅栏、C2 末行
  省栅栏）。这也是 Argtopk 每趟 3→2 栅栏、跳趟省整趟的动机（§6.7）。
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
  （256 KiB 才 trap）（`echo_lru_kernel.h:1006-1007,1274-1278`）。
- **规避**：argtopk 全部状态（key/hist/globalCnt/seg/scalar）分段塞进单个
  `argtopkUb[32768]`，分段偏移集中在 `Init` 一处管理（§3.3）。

### 7.6 bisheng `-xasc` 前端两个崩溃坑（145d3ce 轮踩到）

1. **短路式里的文本 UB 被投机求值**：`pass == 0 || (v >> (shift+8)) == 0` 这种写法，
   pass 0 时 `shift+8 == 32`，`v >> 32` 是文本可见的 UB——即使短路逻辑上不会执行到，
   bisheng SIMT 后端可能**投机求值**导致段错误。规避：写成 if/else 分支，让 UB 表达式
   不出现在 pass 0 的路径上（`echo_lru_kernel.h:453-454,470-474,490-496`）。
2. **`while(true)` + 数据依赖 continue 跨栅栏结构段错误**：attempt 循环刻意**不用**
   `while(true)` + continue 重跑——continue 的跳转目标取决于数据、且跨越栅栏结构，
   bisheng 前端收敛分析崩溃。规避：有界 `for (attempt < 2)` + 迭代边界 break（与旧版
   scalar[3] break 同构，已验证可编译）（`echo_lru_kernel.h:442-446`；commit 145d3ce）。

### 7.7 dav-3510 无 64 位值运算

- 设备侧热路径全部使用 32 位运算（kernel 内索引/计数均 uint32_t/int32_t）；仅有的
  uint64_t 是打点用的 `perfLast_` 与 `clock()` 差值，且**立即 clamp 到 0x7FFFFFFF 后以
  int32 存储**（`echo_lru_kernel.h:1159-1161`）。手抄修正清单里专门有一条 "perfLast_
  截断为 32 位" 的教训（README）。写设备代码时不要引入 64 位算术路径。

### 7.8 Event 计时含 ~百 μs 固定开销，看 kernel 级要用 profiler

- Event 口径是"公共算子接口的流区间延迟"，包含 tiling H2D（pageable 小缓冲拷贝）与
  host 下发空隙，两路各 ~百 μs（README §6、`profile_kernel.py:4-8`）。比值可用、绝对值
  不可用。
- 规避：kernel 纯耗时用 `bash profile_kernel.sh`（torch_npu.profiler 抓
  kernel_details.csv 的 P50/P95/min/max，§8.4）。

### 7.9 跨核 grid.sync 在本 kernel 内 trap（未解）

极简 probe 全 PASS，但 echo_lru Process 内哪怕一个 grid.sync 即 trap（输出全 0 不
hang），根因是编译单元整体属性（Process 巨大/19 Op/Init 多 GM 指针），probe 无法
复现。跨核协同方案因此暂停（`echo_lru_kernel.h:1136-1141`）。**在可预见的将来不要
在本 kernel 里尝试 grid.sync**。

---

## 8. 构建与验证工作流

### 8.1 build.sh 与多套产物目录

`bash build.sh --soc=ascend950`（仅支持 ascend950）→ CMake 配置 + 编译到 `build/`；
`ECHO_PERF_STEP=ON bash build.sh`（或 `--perf-step`）→ 定义 `ECHO_PERF_STEP` 宏，产物
在 `build_perf/`；`bash build.sh --threads=N`（或环境变量 `ECHO_LRU_THREADS`，默认
1024）→ 覆盖 Echo 每核线程数**上限**（32 的倍数、[64,1024]，`build.sh:40-43` 校验；
实际启动线程数仍由 plugin 按 topk 运行时自适应，§3.2——该开关是**纯消融 knob**，
压低上限时 `topk > 4N` 会回退串行 MissCumsum，§5 Step6），N≠1024
时产物在 `build_t{N}` / `build_perf_t{N}`——三套产物目录互不覆盖（目录选择逻辑
`build.sh:46-57`；`.gitignore` 同步忽略 `build_t*/`、`build_perf_t*/`）。

CMakeLists 要点（`CMakeLists.txt:94-171`）：
- bisheng 单独编两个 kernel object：SIMT 用 mixed/VF 选项组（`--cce-auto-sync=off` 等
  四项），Echo 用 `--enable-simt`；两者均 `-O3 --npu-arch=dav-3510 -xasc`。
- `ECHO_PERF_STEP=ON` 时给 Echo kernel **和** plugin 同时定义宏（plugin 侧
  `EchoWorkspaceElements` 公式随之扩容 +31，保证 workspace 校验通过）。
- **线程数注入必须两侧同值**：CMake 缓存变量 `ECHO_LRU_THREADS`（默认 1024）经
  `-DECHO_LRU_THREAD_NUM_OVERRIDE=N` 同时加进 Echo kernel 的编译选项与 plugin 的
  `target_compile_definitions`（`CMakeLists.txt:100-113,165-170`）——plugin 侧它是
  自适应 blockDim 的 clamp 上限（`echo_lru_plugin.cpp:104-108`），kernel 侧它是
  `__launch_bounds__`（`echo_lru_kernel.h:1250`），实际 `<<<blockNum, blockDim>>>`
  传运行时计算值（`echo_lru_launch.cpp:8-9`）；上限必须 ≤ launch bounds，两侧同值
  注入即保证这一点，错位会静默 mismatch。
- 系统 C++ 编两个 plugin，链接 torch/torch_npu/ascendcl/tiling_api/platform/
  libascendc_runtime.a，产出 `libsimt_vs_echo_ops.so`，由 `torch.ops.load_library`
  触发 TORCH_LIBRARY 注册。

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

- 直接 import compare.py 的输入构造/参考/恢复/调用（同尺寸/seed/场景），分步之和可与
  Event 总耗时互相印证（`echo_perf_step.py:11-16`）。
- **哨兵机制**：每次调用前用 `-12345` 填充 workspace；读回 stepCycles 区若仍有哨兵或
  负值 → 报错。这同时兼任"加载的是不是打点构建"的检测（普通构建不写该区，
  `echo_perf_step.py:157-171`）。
- **统计口径**：每步取"最慢核"（max over blocks）的跨轮中位数；用每轮 Σ步周期 / Event
  微秒标定频率（`echo_perf_step.py:204-211`）。
- **步名防漂移断言**：`ECHO_PERF_NUM_STEPS == 23 + 8`（`echo_perf_step.py:64-66`），
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

### 8.5 checks/check_echo_kernel_host.py：171 项 CPU 回归

用 Python 标准库 + 支持 C++20 的 g++ 运行：**直接从 `echo_lru_kernel.h` 文本提取**
`PackedBlockScan` / `OpHitMiss` / `OpArgtopkRow` / `OpMissCumsum` / `OpAllocPop` /
`OpFifoAvailUpdate` 六个函数源码，配上 mock SIMT 头（128 个 std::thread 模拟一个
block、`std::barrier` 模拟 syncthreads、共享缓冲模拟 shfl_up/reduce_add）编译执行，
**开 ASan/UBSan**（`check_echo_kernel_host.py:33-43,55-92`）。

覆盖面（合计 171 项 = 160 + 3 + 7 + 1）：

| 组 | 数量 | 覆盖 |
|---|---:|---|
| 1. OpArgtopkRow | 160 = 10 个 B（1..4096）× 4 种 pri pattern（含无效/平局/全无效）× **4 个 fifo 变体**（全趟/跳2趟/跳1趟/只跑 byte0） | fsClamp 钳位与提前返回（输出/UB/dbg 全不写的哨兵断言）；选择 golden（T=第 fsClamp 小键、quota 公式）与 stable_sort (key,slot) 交叉验证；输出逐位（含 padding 补齐与尾部补零）；key 只写有效区、padding 区/UB 越界哨兵；dbgKey 恒验证、dbgCount 仅全趟路径验证（跳趟首趟是过滤直方图，语义随起跑趟变化） |
| 2. OpHitMiss | 3（eff=4/1536/2048） | hit 槽位、miss 计数、[P4] 折叠 freeSize、[C1] 淘汰标志；**断言零 atomic** |
| 3. OpMissCumsum | 7（topk=4..2048） | **三条路径同 golden**（`check_echo_kernel_host.py:212-236`）：blockDim=128 时 2 元素路径（topk≤256）、4 元素两次链式扫描路径（300/512：topk≤4×128=512 且 4×topk≤8192）、t0 串行回退（2048）的 rank/missPos 逐位一致 |
| 4. OpAllocPop + OpFifoAvailUpdate | 1 | nm/fs 四种组合下与"旧栈先 push 再弹"golden **逐值等价**；avail 折叠终值恒等且 ≥0、fifo 恰好 +1（`check_echo_kernel_host.py:237-276`） |

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
| 自适应 blockDim（e2ca2fd） | plugin 按 topkDim 算 launchThreads，`<<<blockNum, blockDim>>>`（§3.2） | 63.30 / 59.17 μs | 91.44 μs | 137.38 μs | kernel 级 0.78×/0.73×（warm）、1.24×（cold）、1.48×（full）＝消融 t512 数字（§9.10）；自适应构建 NPU 复核待服务器 |

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
  `numMisses`（`echo_lru_kernel.h:291-303`）；OpNumMisses 子步删除。
- **效果**：warm 该步 9.4M → ~19K（source_manifest 记录），Event 比进入 <1 区间。

### 9.2 P1：Argtopk emission 并行化

- **现象**：full_cache 该步 ~2.1M 周期。
- **根因**：t0 串行两遍 Npad 扫描 + 串行补零，延迟无法隐藏（§7.4）。
- **改法**：全员连续切片统计 `(nLt | nEq<<16)` 打包块扫描得写入基址 → 并行写 + 并行
  补零；连续切片保证输出仍按全局槽号升序、与原语义逐位一致（`echo_lru_kernel.h:543-547`）。
- **效果**：该步 2.1M → 398K（source_manifest）。

### 9.3 P2：OpFinalSlots 删除

- **现象**：miss 的最终槽位在 OpSetUpdate 里已经现成（aslot），之后再全量 gather 一遍
  htd 纯属重复（~10K 周期 + 一个尾步栅栏）。
- **改法**：OpSetUpdate 顺手写回 `curSlots[j]=aslot`（hit 位 Step1 已正确），
  OpFinalSlots 删除（`echo_lru_kernel.h:923-926,935-937`）。

### 9.4 P3：OpMissCumsum 去 Hillis-Steele

- **现象**：该步 ~20K 周期且 cold/warm 相同 → barrier bound 而非 work bound。
- **根因**：11 轮 × 1024 线程块栅栏（Hillis-Steele 扫描每轮一次栅栏）。
- **改法**：warp shuffle 一级 + t0 扫 warp total 二级（3 栅栏/行），p0/p0+blockDim 连续
  映射使 rank 可由打包前缀直接拼装（§5 Step6）。
- **效果**：20K → ~9K（source_manifest）。

### 9.5 P4：栅栏削减（步骤 40→31）

- **改法**（每项都有依赖论证，见 §5）：OpFreeSize 折进 Step1 t0 汇总段；OpAllocPop 直读
  freeIdx + 老栈（等价性推导见 §5 Step4）；`avail += fs - nm` 折叠为 Step5 尾一次写
  （终值恒等式见 §5 Step5）；删 FreeRelease→FreePush、FreePush→AllocPop 两个无 RAW
  SyncGrid 与 Step5 尾无读者 SyncGrid；PackedBlockScan 去保护性第 3 栅栏（顺序论证见
  §4.3）。
- **效果**：warm 106K、cold 166K、full 509K（相对 P0-P3 的 125K/?/514K，full 微降、
  warm 因打点步减少略升属噪声范围）。

### 9.6 B：Argtopk 重构（410K → 199K）

- **现象**：P4 后 full_cache 的 ArgtopkAllRows ~410K 周期，占该场景 ~80%。
- **改法**（§6.2/6.3）：pass 0 的 pack 与 clear+hist0 合并为单相位（hist0 直读 priGm，
  无交叉 RAW）；scalar 初始化与 dbg dump 折进 t0 决策段；每趟栅栏 4→3；merge 改 256
  桶并行；hist/pack/emission 只扫有效区 `[0, B+1)`，padding 键解析式计入 + emission
  按需补齐——与旧满扫描输出逐位一致（`echo_lru_kernel.h:380-385`）。
- **效果**：该步 410K → 199K。

### 9.7 C1+C2：无淘汰跳过与末行栅栏

- **C1**：Step1 的 t0 在 UB 写"本核任一行 fs>0"标志，全 0 时 Argtopk/SyncGrid/
  FreeRelease/FreePush 四步整体跳过（安全性论证见 §5 Step3）。warm/cold 场景四步 ≈0
  周期。
- **C2**：OpHitMiss 的 warpScratch 复用保护栅栏仅在还有下一行时执行（§5 Step1）。
- **效果**：warm ~95K、cold 139K、full 309K；Event 审计 0.91-0.97× / full 1.00×。

### 9.8 方向 1+2（commit 145d3ce）

- **改法**：自适应跳趟（fifo 前导零字节 → startPass，§6.5）+ 决策本地化（去 t0 串行
  决策与 scalar 广播，每趟 3→2 栅栏，§6.4）。基准路径栅栏 16→8。
- **验证状态**：checks（现 171 项）ASan/UBSan 通过（4 个 fifo 变体全组合，跳趟/回退
  输出与全趟逐位一致）、g++ mock-SIMT 全头语法检查（开/关 ECHO_PERF_STEP）通过、
  bisheng 编译普通/打点 kernel 通过。**kernel 级实测（profiler P50）**：full_cache
  210.51 → 162.16 μs（2.27×→1.75×），warm_75 0.96×→0.95×、warm_all 0.93×→0.91×，
  cold 1.39×→1.38×；ArgtopkAllRows 估算 ~135 μs → ~87 μs（降幅 ~48 μs ≈ kernel
  总降幅，该步吃掉了几乎全部改进空间）。
- **附带教训**：踩到 bisheng 前端两个段错误坑（§7.6），规避写法已固化在代码注释。

### 9.9 消融基础设施（commit d20fc50）

- **内容**：`ECHO_LRU_THREAD_NUM` 编译期可配（§3.2：CMake `ECHO_LRU_THREADS` +
  `build.sh --threads=N`，产物目录 `build_t{N}` 隔离）；OpMissCumsum 每线程 4 元素
  并行路径（§5 Step6——512 线程消融的前置，2 元素门在 512 线程下装不下生产 topk，
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
bisheng kernel 编译（launch.cpp）exit 0、plugin 头语法检查；NPU 复核待服务器
（预期 = t512 消融数字）。

---

## 10. 当前状态与后续方向

### 10.1 kernel 级数字（现状）

见 §9.10 消融表。一句话：**blockDim 已按 topk 运行时自适应（§3.2），topk=2048 →
512 落在消融甜点——warm 场景 Echo 领先（0.78×/0.73×），cold 1.24×，full_cache
（解码稳态代表：缓存满 + 每步 miss 必然淘汰）仍落后 1.48×（137.38 vs ~92.9 μs）**。
（数字为消融 t512 配置实测；自适应构建的 NPU 复核待服务器，预期一致。）full_cache
的大头仍是淘汰流水线——方向 1+2 轮估算 ArgtopkAllRows ~87 μs（~128K 周期
@1.47 GHz，1024 线程口径），512 线程下的分步重采是 §10.3①；进一步压缩要看
行间软件流水与状态表示重构（§10.3②）。

### 10.2 地板项清单（不再可优化或收益有限的部分）

| 项 | 场景 | 量级估算 | 依据 |
|---|---|---|---|
| OpResetRows 带宽 | cold | 每行 htd 1 MiB（262144 int32）纯写，32 行 = 32 MiB；cold 整 kernel 91.44 μs（t512 口径，§9.10），reset 是其中的主导步（若按整 kernel 摊算已 ≈360 GB/s，步内瞬时带宽更高，接近写带宽地板） | `echo_lru_kernel.h:103-105`，`compare.py:376` |
| OpHitMiss gather | 全部 | 每行 2048 次随机 GM 读（htd[row, tok]，跨 1 MiB 区间），cache miss 为主 | `echo_lru_kernel.h:283-289` |
| 必要 scatter | 全部 | htd/dth/curSlots/flat/mask 的逐 miss 写 + pri 的逐 hit 写 | OpProtect/OpSetUpdate |

### 10.3 已识别方向（按当前优先级排序）

1. **自适应版上板复核 + 512 线程分步重采**：自适应 blockDim（e2ca2fd）的 NPU 复核
   待服务器（预期 = t512 消融数字，§9.10）；随后用 echo_perf_step（§8.3）在
   512 线程构建（`--threads=512`，§8.1）重采分步数据，定位 512 线程下的新热点
   （Argtopk 的 ~87 μs 估算为 1024 线程口径，需更新）。
2. **行间软件流水（rows 扩展性）或"维护序"v2 接口**：消融证实每核行间完全串行是
   结构性软肋（rows 64/128 时 warm 优势消失，§9.10）——在 Op 内 r 循环引入跨行
   重叠（软件流水）可改善 rows 扩展性；追平 SIMT ~50 μs 量级的根本路径则是把持久
   状态改成维护好的序（类似 SIMT 的 lru_slots，淘汰 O(fs)，§2——Echo 每步 O(B)
   全量基数选择的根源是 pri 只是"派生的序"），属于**接口级 v2 变更**（状态字段、
   golden、host 侧全部要动），不是 kernel 内部优化。
3. **投机解码（spec）场景测试**：kernel Step0.5 路径已实现（§5），但 compare.py 四
   场景 `spl` 恒为 `max_token`（`make_case` 里 stable/visible 都取 max_token，
   `compare.py:70-71`），spec golden 未覆盖——需先扩 reference_echo 的投机后缀参考，
   才能谈该路径的正确性与性能。

---

## 11. 文档更新日志

| 日期 | 版本 | 覆盖 commit | 变更摘要 |
|---|---|---|---|
| 2026-10-15 | 初版 | `145d3ce` | 首次成文：全量深读 compare.py / csrc/echo / csrc/simt（大结构）/ checks / 构建与打点脚本；含优化迭代日志（P0-P4/B/C1/C2/方向1+2）与 kernel 级现状数字 |
| 2026-10-15 | 第 2 版 | `d20fc50` | 同步消融基础设施（ECHO_LRU_THREADS 编译期线程数、OpMissCumsum 4 元素路径、`--blocks`/`--rows` 消融入口、README 消融章节）与方向 1+2 kernel 级实测（full_cache 162.16 μs=1.75×、warm 0.95/0.91×、cold 1.38×）；协作机制：本仓代码由 developer-code 子代理按主会话规格实现、文档由 developer-doc 维护、均经主会话审查 |
| 2026-10-16 | 第 3 版 | `e2ca2fd` | 同步运行时自适应 blockDim（plugin 按 topkDim 计算 launchThreads、`launch_echo_lru` 增加 blockDim 参数、`ECHO_LRU_THREADS` 定位为纯消融 knob）与消融实验结论（§9.10：t512 全场景最优 −9%~−18%、t256 崩塌为串行 MissCumsum 混杂因素、blocks 16/8 单调变差、rows 64/128 行间串行软肋）；§10 现状与后续方向按当前优先级重排 |

---

## 附：通读中发现的不一致（文档 vs 代码 / 代码内陈旧注释）

撰写与历次修订时逐条核对发现以下不一致，**均未改动任何代码**，仅在此登记，供后续修订
README 或代码注释时参考：

1. **README §"本次 Echo 更新"/§8 说 checks 为 "169 项"**：当前 `checks/check_echo_kernel_host.py`
   实际产出 **171 项**（160+3+7+1，§8.5）——d20fc50 为 4 元素路径补了 topk=300/512
   两个用例后，README 数字未同步。
2. **echo_lru_tiling.h:1-9 头注释**仍描述 "Plan B / grid_group::sync" 时代的排布；
   **echo_lru_kernel.h:11,1248** 的 "<<<blockNum, 512, ...>>>" 是 512 线程旧值
   （`echo_lru_launch.cpp:8-9` 实际传 plugin 自适应的运行时 blockDim）。当前有效
   事实：Path A、blockDim 运行时自适应（上限 `ECHO_LRU_THREAD_NUM=1024`，§3.2）。
   README §9 已声明"以有效代码为准"，但逐条修订更好。
3. **栅栏成本口径**：kernel 注释（echo_lru_kernel.h:40）记 "~6-8K 周期"，本仓库外部
   材料有 "~4-8K" 的说法，量级一致、区间略有出入；本文 §7.3 按 "4~8K（注释实测 6-8K）"
   并列表述。
4. **代码注释引用的 `GRID_GROUP_PITFALLS.md`、`probe_grid_sync/`、`probe_ub_size`、
   `golden.py`、`gen_data.py`、`echo-lru-argtopk-debug-status` 等上游工程产物不在本仓**，
   相关结论只能以注释转述为准（§3.1、§7.5、§5 spec 段）。
| 2026-10-16 | 8856e2a | 删除扫描门遗留 UB 条件（topk=4096 恒并行），Step6/§3.2 同步 |
