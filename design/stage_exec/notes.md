# Stage执行因素的资源约束

以冻结stage ABCD的UB分阶段存活关系为共同基础，四因素均不新增UB区。
512线程、每路608记录、容量映射偏移及owner/partition复用时序保持父版。

A只改变相同地址的搬入/搬出原语；B只改变寄存器内排名公式；
C合并GM提交的两次遍历，不改变UB存活；D在4K时只使用目录前513项，
8K/16K仍使用1025项，目录与owner之间的原有同步不变。

stage.dag/flow与上一轮资源图相同；重新执行liveness→allocate→physical_p。
这是缓冲存活模型，不是编译器寄存器分配或spill证明。字节区间检查沿用
checks/check_layout_hot_memory.py；本轮所有组合使用其中stage_ABCD的三个容量布局。
SIMD声明的临时RegTensor不增加，实际寄存器/栈、DCache与指令调度仍待CANN资源报告及硬件采集。

C的GM写集合：miss写part[0..misses)对应槽；清理写带失效位的part[misses..capacity)。
在原LRU为槽排列的契约下两者互斥，允许线程交错，不需要清理→分配的独立barrier。
保留末尾状态发布同步；不修改融合归并、map暂存同步或thread1模式发布规避。
