# 无 resident Hash + 流式硬件归并实验

这是新设备实现，保留原带页表 ABCDF、原 Merge 和 BD 文件。5路对照：abcdf_pages、joint_merge、joint_merge_bd、joint_sorted_stream（软件流式归并）、joint_sorted_mrgsort（真实 AscendC::MrgSort<float,true>）。两条新路径具有同样的无Hash规划主体；不能把它们描述为BD的小修补，也没有融合上游TopK。

## 新数据流

1. 每个请求组2或4行的TopK按token升序，逐行二分确定[0,visible)合法区间；MTP0及不支持形状走独立的private VF，不实例化联合规划大VF。
2. 将各行队首的一段转换为8B记录：float32 score=-token、uint32 payload=原组内query位置。max_token<=2^24保证整数到float精确；仍检查原32bit索引范围。未排序输入由框架拒绝；不会排序后隐瞒开销。
3. 软件对照计算整包rank再截取耗尽前缀，硬件只输出消费前缀；因此两者差异也包含避免未消费尾部计算的收益。软件对照按rank散写归并一包；硬件路径对2~4条有效队列调用MrgSort耗尽模式，只消费到首条队列耗尽。单队列直接复制。只更新实际消费的队列位置，未消费元素下次重装，不丢弃尾部；空队列不传给硬件。下一包延续lastToken，相邻去重并建立query→union编号。超过capacity时在持久状态写入前回退。
4. resident每槽在有序并集内二分查找。省去resident Hash清空、CAS建表、全量query查表和hit原子OR。命中resident用atomic_max保留LRU-last owner；每个重复resident槽都单独记录保护标记。
5. 完整扫描LRU分类、给unique miss分配最旧victim，容量不足仍回退；之后写回cache/LRU/descriptor、逐位置currentSlots和联合映射。共享模式发布继续使用thread1规避，不宣称根因已查清。

本版归并的是全部有效query，不是仅miss；这是为了完全省去resident Hash并让resident只在最终有序集合中查询。并集形成后才判断哪些token已驻留。仍采用严格原LRU语义，不通过近似淘汰或假设resident唯一减少工作。

## UB布局及阶段边界

每个AIV：Scan + 64个int32控制字 + queryToUid[4T] + unique[C] + 32B对齐填充 + work[max(3C,64)]。work在归并期间放输入/输出packet，规划期间复用为owner、protected flags、LRU partition。生产T2048/C4096时，scratch=24640个int32=96.25KiB，另加288B Scan，合计98848B。

packet每路最多768条，输入加输出最大48KiB，位于已死的后续规划工作区，不需要128KiB全量双缓冲。任意TopK/capacity都对齐队列首址；长度由合法区间决定，不把padding当token。较大scratch走GM，但packet和控制字仍留UB；库与Python对动态UB字节数逐项核对。

VF参数沿用原标量/指针ABI，阶段用不同模板VF实例，未把大struct作为VF入参。事件通过AllocEventID/ReleaseEventID成对管理；SIMT→scalar通过V_S事件交接，scalar→SIMT通过S_V交接。Vector MrgSort前后的结果消费也有显式交接。CPU模型不能证明这些设备同步和部署SDK原语已正确工作，必须真机门禁。

本轮没有Squeeze或TopK融合。压缩/去重仍由SIMT Scan完成；硬件收益、重装队首的额外流量及多VF切换开销均需实测。全量有序查询替代resident Hash并不保证快，软件/硬件两路用于区分结构变化与归并引擎的影响。

## 上机命令（CANN9.1 / NPU0）

```bash
cd /workspace
git clone --single-branch -b feat/joint-sorted-hardware-merge https://github.com/liser0019/op_test_kvoffload_simt_echo.git joint_sorted_hardware
cd joint_sorted_hardware
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
bash scripts/test_sorted_hardware.sh
```

脚本先用--sorted-only构建新两路与原Merge/BD，跳过其他旧候选；保持-O3。专项门禁覆盖MTP0/1/3、跨tile、输入不变性、保护区、重复resident、溢出、UB/GM、连续状态和prepared绑定，并做5次原故障尺寸回归。通过后才对32请求/TopK2048/C4096/max_token262144/overlap40/seed2026采集MTP0/1/3，预热2、repeat5，全部样本检查。原来的1秒ACTIVE启动等待和严格目标kernel计数/顺序检查保留。

结果放新的/workspace/sorted_hardware_*/profile/joint_matrix_*/report.md。看“无resident Hash：流式有序并集”表、共享/回退组数、各行命中率与重合率、原始CSV索引和库身份。默认生产MTP1/3各有32个共享组；hardware实现这些输入有多条有效队列，真实调用MrgSort。如果候选回退会被语义门禁发现，不能把private耗时归成硬件归并性能。

先跑5样本再决定入围路径。可用以下命令增加样本（路径名仅示例，不代表已获胜）：

```bash
CANDIDATES='abcdf_pages joint_merge_bd joint_sorted_mrgsort' \
SCENARIOS='cold_reset full_cache_75pct' WARMUP=10 REPEAT=50 \
bash scripts/test_sorted_hardware.sh
```

原始输入、现场、profile与二进制只留服务器。缺编译支持、正确性失败或CSV不完整就停止，不生成成功结论；不要修改成静默跳过硬件路径。没有NPU的开发机无法确认MrgSort实际编译、float/index硬件布局、事件交接或性能。

## 本地验证范围与API依据

checks/check_sorted_plan.py实际执行C++控制器和各阶段VF，使用CPU64/UBSan。checks/simt_variants/sorted_cann_cpu.h只模型化MrgSort“消费首条耗尽前缀”的公开语义、LocalTensor和事件，不是CANN SDK或设备指令仿真。检查覆盖跨refill重复、空/单队列、不同长度、容量溢出、最后resident owner、完整LRU、输入及保护区、连续状态、可见性与float精度边界；不能将这些结果写为CANN编译通过。

API依据：CANN9.1 MrgSort说明（最多4路、8B score/index、float、耗尽模式及sortedNum）；本地asc-devkit官方资料中的SIMD/SIMT混合编程、asc_mrgsort4的32B地址约束；现有RegBase参考算子的MrgSortSrcList和V_S事件用法。最终以服务器CANN9.1.0头文件为准。

- https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/sort_operations/MrgSort-93.md
- https://gitcode.com/cann/asc-devkit
