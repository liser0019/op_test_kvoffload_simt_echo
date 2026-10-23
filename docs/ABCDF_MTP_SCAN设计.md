# ABCDF / MTP / Scan 设计说明

## 一、任务范围与固定项

基线为9cc99d5中的13_abcdf，沿用1024线程、32 lane/warp、32 warp/block、
相同hash容量与hash函数、64核上限、80字节tiling、112KiB行scratch阈值。
输入张量顺序和3个持久状态、4个输出的公开接口不变。
本轮不叠加历史E/G/H，不改变历史ABCDF掩码47U。

历史ABCDF指删除pending、scratch重建LRU、尾部清理、安全owner打包、warp-total特化；
本文的因素A～F是新的Scan实验名，两者不能按字母直接混用。

此前分阶段clock打点没有通过设备正确性，因此本轮没有perf指针、时间戳区、
额外VF参数或阶段打点写入。以无打点kernel的profiling和消融实验为证据。

## 二、完整算法里替换了哪两块

热路径仍是：

1. 初始化currentSlots等输出，判断request ID是否更换。
2. 更换则reset并进入cold；未更换则初始化hash。
3. 将合法TopK token插入hash，用atomic_min记录重复token的首次query位置。
4. 按旧LRU顺序分类hit/evict，失效投机后缀；atomic_max决定重复resident的最后owner。
5. 消费每个query的bucket，确定首次hit以及其他miss。
6. 稳定压缩miss位置，按evict顺序分配；miss超过槽数时截断。
7. 写公开输出；未分配项保持-1。
8. LRU重建为“未使用evict + 新分配miss + 旧hit”，保持各段原相对顺序。

新代码只替换4和5/6中的Scan组织，cold只单独比较F1。
golden保持列表/字典实现，不使用hash、Ballot或同款Scan，避免把相同实现错误带入参考。

HiSparse代码提供了warp位图、chunk计数/前缀的思路；
其最新token额外槽、短序列快路及Transfer行为没有移植。
本接口对非法来源、重复TopK、resident owner和stablePrefix有自己的语义，继续以ABCDF为准。

## 三、局部A/B/C

### 3.1 原版分层Scan

每个tile覆盖1024项，所有线程参与，尾部无效线程输入0：

1. warp内5轮shuffle inclusive scan。
2. lane31写warp total。
3. 第一道block barrier，确保所有warp total可读。
4. warp0计算32个warp的exclusive base及全tile total。
5. 第二道block barrier，发布bases/total。
6. 各线程读base，得到rank并scatter。
7. scatter后block barrier，保护当前消费完成后复用scratch。

LRU输入为 `evict + (hit<<12)`；tile最多1024项，低12位不会向高位进位。
Miss输入是0/1。

### 3.2 A：Miss Ballot

全warp执行ballot，将32个flag编码为32位mask。
线程lane的局部rank为 `popc(mask & ((1u<<lane)-1))`，
warp total为 `popc(mask)`。lane范围0..31，不移位32位。
跨warp组织仍为原发布路径，所以A单独不会减少block barrier。

### 3.3 B：LRU Dual Ballot

分别对evict/hit执行ballot，计算两个局部rank和两个warp计数，
再打包为低12位evict、高位hit交给同一跨warp步骤。
无效LRU与尾部线程两项均0，不能将“不是hit”直接视为evict。

### 3.4 C：每个warp自主求base/total

仍先由lane31写warpTotals，并保留生产者→消费者的第一道barrier。
每个warp的32个lane读取32个warp total：

- base = reduce_add(lane < 当前warp编号 ? warpTotal[lane] : 0)。
- total = reduce_add(warpTotal[lane])。

各warp自持结果，无需等待warp0发布base，省去第二道barrier。
scatter后的barrier完整保留，防止快warp覆盖下轮totals，而慢warp仍在读本轮。
32个warp的重复读取和归约增加开销，收益必须实测，不承诺C一定快。

生产尺寸只数LRU/Miss这两段的源码barrier：
原版6tiles×3=18，C版6tiles×2=12。
这不是整个kernel的barrier数，也不是设备计时数据。

## 四、结构D：整个LRU分三步

不能把每tile的低12位计数直接用于整个capacity：4096个evict会进位。
因此D使用独立evict/hit前缀，且保存旧槽号快照。

1. 所有线程遍历完整capacity：保留原分类与hashOwner原子规则；
   每32项保存双位图，保存每项旧slot，完成后barrier。
2. 仅warp0按每组32个chunk计算独占前缀，组间carry留在寄存器；
   前缀和计数完全分开。组间不做block barrier；全部完成后统一barrier。
3. 所有线程读位图、base与slot快照：evict向数组头scatter，hit向尾反向scatter。
   两段总数不超过capacity。scatter后barrier，才进入下一阶段。

warp0外的warp可以较早进入第二道barrier等待；warp0内部所有32 lane执行全部shuffle，
缺失chunk补0，没有部分lane提前返回的warp操作。
该实现源码为3道block barrier，实际性能还包含多一遍读写与warp0前缀计算。

## 五、结构E：整个Miss分三步

missPositions起初保存每个query的hash bucket，之后复用为压缩的原query位置。

1. 遍历所有TopK，读旧bucket并决定hit/miss，将首次hit写currentSlots；
   每32项保存miss位图，全部完成后barrier。
2. warp0对所有chunk计数作前缀，发布完整miss总数与bases，再barrier。
3. 各线程只读位图，将原query位置稳定压缩写回missPositions，再barrier。

关键在第一道barrier：它证明所有旧bucket已经消费完。
不能只去掉逐tile同步后边读bucket边写压缩结果，否则会覆盖慢线程尚未读取的数据。
额外位图/前缀存储在独立scratch区，不与bucket输入重叠。

D替代LRU tile Scan，所以不再单独叠加B；E替代Miss tile Scan，所以不叠加A。
C只影响仍存在tile Scan的那一段，S09不再叠加C。

## 六、UB与GM布局逐项核对

单位int32，H=max(32,nextpow2(2T))，C=capacity，T=topk，
NC=ceil(C/32)，NT=ceil(T/32)。

基础区：

```text
keys[H] | firstPos[H] | owner[H] | partitionSlots[C] | missPositions[T]
```

附加区固定放在基础区之后：

```text
D: savedSlots[C] | evictMasks[NC] | hitMasks[NC] | evictBases[NC] | hitBases[NC]
E: missMasks[NT] | missBases[NT]     # D同时开启时，E接在D后面
```

Host experiment_launch.h与Python plan_layout使用相同公式：

```text
row_elements = 3H+C+T + (D ? C+4NC : 0) + (E ? 2NT : 0)
row_bytes = align32(4*row_elements)
Scan UB = 288B，control_total=64，control_reset=65，D的hit total=66
动态UB = 288B + (row_bytes <=112KiB ? row_bytes : 0)
GM workspace = UB路径1个int32占位，否则 rows*row_elements 个int32
```

| 生产T2048/C4096 | 行scratch | 另加Scan | 路径 |
|---|---:|---:|---|
| 原版/局部A/B/C/F1 | 72KiB | 288B | UB |
| 仅D | 90KiB | 288B | UB |
| 仅E | 72.5KiB | 288B | UB |
| D+E | 90.5KiB | 288B | UB |

T2048/C8192时，原版88KiB仍走UB，D版124KiB转GM；
T4096/C8192基础即144KiB，全组走GM。
报告必须显示scratch_path和动态UB，不能把发生UB/GM切换的比值全归因于Scan。

ScratchPtr分别实例化为__ubuf__和__gm__，保留原地址空间；
没有通过强转让GM地址冒充UB。VF参数、1024线程配置与mixed外壳原调用方式保持不变。
新Host头先包含，再包含原plugin共享实现，确保扩展scratch公式用于校验/tiling。
专属类型/函数别名防止多版本链接混用同名helper或tiling。

## 七、MTP测试与性能对照为什么分开

普通四场景：每个位置加入可见新token，warm状态保留历史投机槽，
用于观察真实可见范围与stablePrefix失效成本。此时“历史全命中”不等于query零miss。

精确miss矩阵：只用稳定query，以0/1/16/256/1024/2048等精确miss数隔离Scan flag分布；
MTP历史失效仍在，但不会把这些结果称为新token全部被查询的生产场景。

所有版本共享同一输入和初始状态；每次测量前恢复，正式排名只统计kernel duration。
固定请求数回答整次Plan开关MTP的代价；固定行数回答同等query工作量下的大致差异。
两种条件分别记录，不能用128行/32行的比值直接证明算法变慢多少。

## 八、验证和解释边界

检查布局/别名、scalar golden和CSV解析的辅助脚本已适配本轮；本机没有执行。
设备门禁需逐项核对所有输出/状态，包含每次从自身NPU状态续用的MTP轨迹。
短跑通过后再正式采集；任何一版错误都先定位并修复，再比较性能。

本轮没有添加独立Scan微基准：优化首先在完整Plan语境中验证。
此前Scan故障与上下文有关，因此独立微基准即使通过，也不能替代这里的整算子门禁。
Top3的深入诊断由run_msopprof.py逐版本/场景单独采集，候选由真实结果选择。

理论barrier数量只是源码模型；PC Sampling是采样概率；
硬件指标、实际kernel duration与消融差值结合后才能判断瓶颈。
尚未获得本轮profiling，不对加速幅度或胜出版本下结论。
