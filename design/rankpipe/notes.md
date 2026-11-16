# Rank-pipeline资源与同步复核

目标DAV_3510 / Ascend950PR、950DT；基于051a439的冻结exec ABCD。没有本地CANN/NPU。

## A：96KiB merge work按阶段复用

从owner/已消费的merge packet起点计字节：

| 阶段 | 活跃区域 | 范围 |
|---|---|---|
| 索引构建/查询 | 精确mask | [0,32768) |
| 索引构建/查询 | word首UID，uint32 | [32768,65536) |
| 索引构建/查询 | 物理槽→UID/标记，uint16 | [65536,65536+2×capacity) |
| owner/partition | owner | [0,4×capacity) |
| owner/partition | partition | [4×capacity,6×capacity) |

capacity最大16384，两阶段都不超过98304B。索引所有读完成并同步后才初始化owner；物理槽UID表在64KiB边界开始，不被最大64KiB的owner初始化覆盖。LRU分类全部完成后才允许partition覆盖原物理槽UID表。map先按父版暂存GM，unique位于工作区之前，不重叠。

word首UID选择uint32而非分析阶段建议的uint16：32位独立写和读无拆包成本，避免半字写入约束，和32KiB物理槽表一起仍能放入96KiB。不增加动态UB。位图每个字由unique有序段的一个首线程独占，寄存器内累积最多32个bit后普通写；不用原子OR，不用全域prefix Scan。空字的首UID不必初始化，只有mask对应位为1时才读取。

stage.dag/flow→liveness→allocate→physical_p结果随本目录保存。P为逻辑缓冲模型，不是编译器RegTensor/spill证明；字节边界由check_rankpipe_suite单独检查。

## B：word分块两级前缀

生产512线程，每线程处理32个位置，capacity16384时512个块一次完成。跨warp Scan只对块计数进行；块内使用hit/miss掩码的popcount计算精确排名。小容量不活动线程仍参加Scan，值为0。减少线程数的CPU模型通过外层循环覆盖全部块，不改变生产512线程配置。增加一次flags重读，不保证一定比SIMD carry链快；保持失败检查和全部同步。

## C：producer → consumer → sync

| 输出 | Producer | Consumer实际源 | 同步和生命周期 |
|---|---|---|---|
| LRU | hot Phase9/cold finish，读完owner后把完整LRU写owner区域 | local[200+capacity/2+capacity] | owner所有旧读者先syncthreads；VF返回后V_MTE3 Set/Wait；MTE3_V Set/Wait后才进入下一组 |
| miss_tokens尾部 | hot Phase9/cold finish，读完unique后用unique存-1 | local[200+capacity/2] | unique所有旧读者先syncthreads；同上 |
| miss_slots尾部 | 同上，同一只读-1缓冲 | 同上 | 与前一份copy共享源，保持完整直到两份copy结束 |
| 控制值 | ctx[46]=尾长、ctx[47]=misses、ctx[120]=ready | Scalar copy helper | 原V_S发布控制；每组Phase0把ready清0，回退不触发copy |

200=Scan72+ctx128个int32。源偏移和步长32B对齐；尾目标可不对齐，DataCopyPad仅写有效字节。miss有效前缀、currentSlots、resident及协议元数据仍由原SIMT计算写回。GM map不改为Scalar/MTE读取，保持相同线程写/读原32位字。UB→GM三次copy（尾长0时仅LRU一次）和等完成均在原Plan kernel内部，不新增kernel或CPU等待。

两份同步扫描报告findings均为空。脚本为正则前端：flow analyzer未解析GlobalTensor别名，甚至把该重载默认归为MTE2；此结果不能证明跨函数/模板/宏的数据流正确。上表为人工追踪。CPU模型把MTE copy延迟到MTE3_V等待时执行，TPipe析构检查无未完成copy；仍不能验证真实DCache/地址空间/硬件事件。
