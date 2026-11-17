# 融合阶段生命周期证据

本目录只验证阶段级复用的抽象槽位/WAR约束；不是逐指令同步、寄存器压力或硬件时序证明。
Cube结果和TopK输出独占；QLI Vector参数与TopK scratch在S4最后异步读完成后死亡，Plan方可覆盖。
allocate.py的槽位大小不按字节建模，真实字节预算另由check_qli_fusion_resources.py编译实际Host函数核对。
C保留原QLI独立分配，只复用原TopK/Plan区；其unique staging在MTE3完成后再覆盖。
