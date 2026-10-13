# Sparse KV Plan：BCDE + fused + 1024 优化基线说明

用户实机测试反馈：`combo_bcde_fused_1024` 当前表现更好，因此从 2026-10-13 起，
将它作为后续 Plan 优化的性能参考基线。它继续使用哈希表和前缀和，优化的是原算法
内部的扫描、临时空间、重复检查和线程组织，没有替换为 ECHO 算法。

本文没有收到各尺寸的原始 CSV 和具体加速倍数，不把“当前更好”扩大为所有输入、
所有命中率或整网场景下必然最快。后续候选都应在相同输入和初始状态下与该版比较。

| 项目 | 内容 |
|---|---|
| 工程 | `op_test_kvoffload_simt_echo` |
| 分支 | `feat/peer-simt-comparison` |
| 后续基线 / Torch op 名 | `combo_bcde_fused_1024` |
| 目标平台 | Ascend 950 / A5，SIMT |
| 每个执行行的 SIMT 配置 | 1024 线程，32 个 warp，每 warp 32 线程 |
| 对照对象 | 原 `00_baseline`、`10_bcde`、`combo_bcde_fused` 等仍保留 |
| 当前计算源码版本 | 提交 `68c10218eec41e915a0995e0ec7abdc5d3726121` 中的该版本 |
| kernel 文件 SHA256 | `1114df2886bcd89986ec3efb57e94e638ddbc4975939c394b5bc44a18b199542` |

## 1. Plan 解决什么问题

Indexer 给出本行想读取的 TopK token。Plan 需要判断哪些 token 已在 NPU resident
buffer 中，哪些缺失，并为缺失项选择槽位。它输出槽位编号及搬运安排；KV 数据内容
仍由 Transfer 搬运。

```text
Indexer：需要哪些 token
    ↓
Plan：哪些已经在 resident；缺失的 token 放到哪些 slot
    ↓
Transfer：把缺失 KV 从 Host 池搬到 resident buffer
    ↓
SFA：根据 currentSlots 读取 resident KV
```

这里的 `fused` 指 Plan 内部“命中结果回填与 miss 判断融合”。它没有把 Indexer、
Transfer 或 SFA 融合成一个 kernel，也没有增加搬运与 Indexer 的重叠。

| 名称 | 含义 |
|---|---|
| `T / topk` | 本行 TopK 项数 |
| `C / capacity` | 本行 resident 槽位数 |
| `H / hashCapacity` | 临时哈希桶数：`max(32, next_power_of_two(2*T))` |
| `slotToToken` | 每个 resident 槽位当前记录的 token 位置 |
| `lruSlots` | resident 槽位从旧到新的顺序 |
| `currentSlots` | 本行每个 TopK 项最终对应的 resident 槽位 |
| `missTokens / missSlots` | Transfer 要加载的 token 及其目标槽位 |
| `missPositions` | 临时数组；本版本先存 bucket，随后原地变成 miss 位置列表 |

## 2. 这一版包含哪些优化

| 优化 | 实际改动 | 主要减少的成本 |
|---|---|---|
| B | warp 0 并行计算二级 Scan，复用调用者的同步 | 线程 0 串行累加、重复 barrier |
| C | hit/evict 两类槽位共用一个 capacity 数组 | 一份 C 元素的临时空间 |
| D | 请求 ID 改变时进入专门的冷路径 | 新请求上的建表、命中查询和 resident 分类 |
| E | 保留首次有效性检查的中间结果和 `-2` 标记 | 后续重复检查；在融合版中与 bucket 判据重合 |
| F / fused | 保存插入 bucket，合并命中回填与 miss 压缩 | 单独的 hashOwner 全桶发布、重复查询与部分同步 |
| 1024 线程 | 64 warp 改为 32 warp，相关 Scan 空间同步缩小 | 二级 Scan 组数及可能的线程调度压力 |

相对原 SIMT，是这组累计优化；相对已有 `10_bcde`，新增的是 F 与 1024 线程配置。
因此不能把 B/C/D/E 的收益再次计入“F 相对 BCDE”的收益。

这一版没有加入同事的 pair Scan，没有 tiling 缓存，也没有实现独立的 scratch LRU
读取优化。hit 槽号仍从 GM `lruSlots` 读取，LRU 重建时的新 miss 槽号仍读取 GM
`missSlots`。

## 3. B：让二级前缀和并行计算，并合并同步

前缀和的作用是给被选中的项分配连续输出位置。例如 flag 为 `[0,1,0,1]` 时，两个
有效项分别取得输出排名 0、1，压缩结果就是原位置 `[1,3]`。

原版先在每个 warp 内计算前缀和，再由线程 0 顺序扫描全部 warp 的总数。其他线程
必须等待这段串行累加。

B 用 `PublishWarpBases()` 让 warp 0 的 32 个 lane 协作处理 warp 总数，组内通过
shuffle 计算前缀和。超过 32 个 warp 时按组处理，并携带前一组的总数。

```text
各 warp 内做 Scan，写出各自总数
→ barrier
→ warp 0 并行生成 warp 起点和 block 总数
→ barrier
→ 各线程取得最终排名，写分类结果
→ 调用者的 barrier
```

1024 线程恰好对应 32 个 warp，二级 Scan 只处理一组；2048 线程时要处理两组。
使用的仍是 B 的分组 Scan，不是另外的 pair Scan 实现。

原 Scan 内部末尾另有一次 barrier，B 删除这次同步，由调用者写出分类结果后的
barrier 保证 scratch 安全复用。前两次发布依赖的 barrier 保留；不能简单理解成
“把同步都删了”。

## 4. C：hit 和 evict 共用一份临时列表

原版分别分配 `evictableSlots[C]` 和 `hitSlots[C]`。但合法的旧 LRU 槽位每次只能
属于其中一类，两类数量之和不超过 C，因此不必预留两份完整 C 空间。

C 将可淘汰槽位从数组头部向后写，将命中槽位从尾部向前写：

```text
低地址                                           高地址
[evict →]              [可能的空余区]              [← hit]
```

例如 C=8，evict 的原顺序为 `[0,2,4,6,7]`，hit 为 `[1,3,5]`，临时数组是：

```text
[0,2,4,6,7,5,3,1]
```

最终 LRU 重建时反向读取 hit 区，仍得到 `[1,3,5]`，保留了原顺序。

临时区从 `4*(3H+2C+T)` 字节减少为 `4*(3H+C+T)` 字节，节省 `4*C` 字节。
resident 分类中把 evict/hit 计数打包进同一次 Scan 的策略原本已有，不是 C 新增。

| T / C | H | 原版行临时区 | 当前行临时区 | 路径变化 |
|---|---:|---:|---:|---|
| 2048 / 4096 | 4096 | 88 KiB | 72 KiB | 两者均走 UB |
| 2048 / 8192 | 4096 | 120 KiB | 88 KiB | 原版 GM，当前 UB |
| 2049 / 4096 | 8192 | 139268 字节 | 122884 字节 | 两者均走 GM |

行临时区阈值仍为 112 KiB。减少临时区可能让某些尺寸从 GM 切到 UB，但不是所有
尺寸都会切换。T 从 2048 增加到 2049 时 H 翻倍，仍会触发 GM 回退。

## 5. D：新请求不做无意义的命中查询

触发条件是本 query 行记录的旧请求 ID 与本轮请求 ID 不同。新的请求不能沿用旧
resident 内容，所以检查“哪些旧槽位命中”没有意义。

`ProcessColdRow()` 直接完成：

```text
清空 slotToToken
→ 检查 TopK 源并稳定压缩合法项
→ 为能容纳的 miss 分配槽位
→ 生成新的 LRU 和请求 ID
```

这条路径跳过哈希初始化与插入、resident 命中查询、hit/evict 分类和命中发布。
但它仍做源有效性检查、前缀和和必要同步。连续处理同一请求的暖路径不会每步享受 D
的收益；仅仅使 MTP 后缀失效也不会触发这个冷路径。

## 6. E 与 F：先判断一次，再复用结果并合并遍历

### 6.1 E 在 BCDE 中解决的问题

原版在建表时检查 TopK 的 token 范围、query 可见长度、request 映射和 block table，
后面压缩 miss 时又重新检查。

E 在第一次检查通过后将 `currentSlots` 临时标记为 `-2`，无效项保持 `-1`。普通
BCDE 发布命中后，用剩下的 `-2` 判断 miss，避免第二次源检查。

### 6.2 F 保存 Insert 的返回 bucket

融合版进一步利用 `InsertTopkToken()` 已经得到的 bucket：

```cpp
currentSlots[topkBase + pos] = PENDING_SLOT;
missPositions[pos] = static_cast<int32_t>(
    InsertTopkToken(hashKeys, hashFirstPos, token, pos, hashCapacity));
```

无效来源对应的位置保存 `HASH_BUCKET_INVALID`。这张逐 TopK 的 bucket 映射既提供
直接访问 hashOwner 的索引，也保留了先前检查的结果。

resident 分类仍然需要 `LookupTopkToken()` 查找旧缓存 token；F 没有取消这项工作。
之后却不必再遍历 H 个 hashOwner 桶单独发布命中，而是在压缩 miss 的 T 项遍历中：

```text
读取之前保存的 bucket
├ bucket 无效：跳过
├ 有 owner，且是该 token 最早 TopK 位置：直接回填已有 slot
└ 否则：记为 miss，执行前缀和压缩
```

旧 BCDE 是“遍历 H 个桶发布命中 → barrier → 遍历 T 项压缩 miss”；融合版把两件事
合进一次 T 项遍历，省掉独立的 H 遍历及其阶段间同步，也不重新计算 TopK 的 hash
或重复读取 source 元数据。

E 的 `-2` 写入与最终清理仍在代码中，但融合阶段判断有效性主要依据已保存的 bucket。
所以 E 与 F 在消除重复检查方面有重合，不能宣传成“两次独立收益”。未分配成功的
`-2` 仍在返回前恢复为 `-1`，不改变公开输出约定。

### 6.3 同一临时数组的安全复用

`missPositions[T]` 没有扩容，而是先保存 bucket，随后原地变成稳定压缩的 miss 位置。
每个 tile 的线程先读自己的 bucket/owner，进入 Scan 的 barrier 后才写压缩结果。

压缩输出位置不大于原 query 位置；已处理 tile 的写入也不会覆盖未来 tile 的输入。
scatter 后还保留 barrier，保证下一 tile 的读取和 scratch 复用正确。这些同步是
融合优化的依赖条件，不能继续随意删除。

## 7. 1024 线程：减少 warp 数，同时改变每线程工作量

```cpp
constexpr uint32_t PLAN_THREADS = 1024U;
constexpr uint32_t PLAN_WARP_COUNT = PLAN_THREADS / 32U;
```

VF launch bound、调用线程数、循环步长和 Scan 空间都由同一配置推导。1024 是每个
执行行的 SIMT 线程配置，不是 AIV 核数；blockDim 仍取 `min(64, 平台 AIV 数)`。
多余的 block 在读取 activeRows 后返回，有任务的 block 按 stride 处理后续 query 行。

| 项目 | BCDE / fused 的 2048 配置 | 当前 1024 配置 |
|---|---:|---:|
| warp 数 | 64 | 32 |
| B 的二级 Scan 组数 | 2 | 1 |
| Scan scratch | 544 字节 | 288 字节 |
| T=2048 的逐 TopK tile 数 | 1 | 2 |
| C=4096 的 resident 分类 tile 数 | 2 | 4 |

较少 warp 可能减轻线程调度和资源压力，二级 Scan 也更短；同时更多 tile 会增加
循环和 Scan 调用次数。因此“线程减半”不等于“计算量减半”，也不保证所有尺寸更快。
本次选择它来自用户实测反馈，具体原因仍应结合寄存器、SIMT stall 等 profiler 指标。

在 T=2048、C=4096 时，本版行临时区为 73728 字节，动态 UB 合计为：

```text
73728 + 288 = 74016 字节
```

UB 路径的 GM compactWorkspace 只需占位 1 个 int32。GM 回退时动态 UB 只保留
288 字节 Scan 区，GM 临时区按 `maxRows * (3H+C+T)` 个 int32 分配。

## 8. 保留的算法语义与 MTP 处理

暖路径仍按以下顺序生成 LRU：

```text
未使用的可淘汰槽位 + 新 miss 的槽位 + 原命中槽位
```

重复 TopK 并没有改成全部共用一个 slot：最早 TopK 位置可以取得已有命中，后续重复
位置仍按原约定处理为 miss。同 token 记录在多个 resident 槽位时，仍用旧 LRU 中
最后一个 owner 提供已有命中，hashOwner 的 atomic max 和 firstPos 的 atomic min
规则保持不变。

MTP 相关处理也保留：

- `visible_seq_lens` 限制每个 query 能读取到哪里。
- `stable_prefix_lens` 使可能被改写的旧 resident 后缀失效。
- 同 request 的不同 query 行仍拥有各自的 resident/LRU 状态。
- 请求重排后的行所有权变化仍通过请求 ID 判断。

这说明本版保留了原 Plan 处理这些输入的算法路径，不等于已经完成真实 MTP 整网
正确性或图 replay 验证。当前四场景性能用例是合法、无重复、全激活输入；复杂输入
已有 CPU oracle 检查，仍应在需要部署时进行对应的 NPU 业务验证。

## 9. 代码阅读位置

目录：`csrc/peer_variants/combo_bcde_fused_1024/`。

| 文件 / 函数 | 内容 |
|---|---|
| `upstream/sparse_kv_plan_config.h` | 1024 线程、warp 数、Scan 空间和压缩行布局 |
| `upstream/sparse_kv_plan_launch.h` | Hash 容量、行空间、动态 UB 和 80 字节 tiling |
| `upstream/sparse_kv_plan_kernel.cpp:PublishWarpBases` | B 的并行二级 Scan |
| 同文件 `BlockExclusiveScan` | warp 内 Scan 与同步依赖 |
| 同文件 `ProcessColdRow` | D 的新请求冷路径 |
| 同文件 `ProcessRuntimeRow` | C 的槽位分类、E/F 的有效性复用与融合 |
| 同文件 `SparseKvPlanRuntimeVf` | UB/GM 分派和逐行处理 |
| 同文件 `sparse_kv_plan_kernel` | activeRows、动态 UB、VF 调用 |
| `upstream/plugin_registered.h` | 真正参与编译的 Torch 注册和 Host 调用 |
| `kernel.cpp / plugin.cpp / symbol_aliases.h` | 多版本符号隔离包装 |

只替换 kernel 不够：生产移植还要同步压缩布局公式、动态 UB 和 1024 线程配置。
这份直调工程的 tiling 字段数量和顺序没有改变，workspace 大小与线程参数必须匹配。
本次确立单算子基线，没有修改 vLLM manager、Transfer、SFA 或 MemFabric。

## 10. 后续如何使用这个基线

`profile_peer_variants.py` 和 `run_peer_suite.sh` 默认参照改为
`combo_bcde_fused_1024`。CSV 的加速比含义是：

```text
加速比 = 本基线的 kernel P50 / 被测版本的 kernel P50
```

大于 1 表示被测版本比本基线更快。原 `10_bcde` 继续参与，方便观察相对上一轮基线
的变化。使用 `compare.py` 时显式填写 `--reference`：

```bash
python3 profile_peer_variants.py --device 1 --scenario all \
  --implementations combo_bcde_fused_1024 10_bcde combo_bcde_fused combo_bcde_1024 \
  --reference combo_bcde_fused_1024 --warmup 13 --repeat 60 \
  --output /dev/shm/bcde_fused_1024_baseline

python3 compare.py --device 1 \
  --implementations combo_bcde_fused_1024 10_bcde combo_bcde_fused combo_bcde_1024 \
  --reference combo_bcde_fused_1024
```

性能结论应区分暖路径、冷路径、全命中、不同 miss 比例，以及 UB/GM 路径。
本次默认参照切换没有改变任何 kernel 计算代码。

## 修订记录

| 日期 | 编写人 | 内容 |
|---|---|---|
| 2026-10-13 | 项目维护助手 | 根据用户实机反馈确立基线，对照源码整理 B/C/D/E/F 与 1024 配置 |
