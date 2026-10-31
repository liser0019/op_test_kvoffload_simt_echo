# Joint Bitmap 结构性实验

基于 `8f12ef0`，新增 `joint_bitmap`，保留原 ABCDF、joint_abc 及全部旧候选。这是新候选，尚未在本机执行 CANN 编译或 NPU 验证，没有加速结论。

## 数据结构与原语

XOR 不是并集：同一个 token 出现两次会抵消。采用 `asc_atomic_or(bitmap[token/32], 1 << (token%32))`。返回旧位是0的线程是该token唯一发布者；用现有block scan压缩到unique列表，不通过同一个全局计数器累加。输入仍逐位置检查本行因果可见性和来源，不要求各行TopK相同或有序。

新的工作流程：

1. 清零位图；将整个请求组的query位置展平，1024线程并行校验和原子置位。
2. 每个tile把新增token压入紧凑unique列表，初始化这些token的`map[token]=-1`。一旦并集超过C，终止并整组回退；不清零整个map。
3. 按旧LRU扫描resident，通过位图直接判断并集成员资格。命中项发布`map[token]=slot`；保留原evict/hit稳定分类。
4. 只扫描unique列表，分配尚未命中的token，生成真实miss descriptor并更新map与LRU。
5. 每个query从暂存的合法token直接读map得到currentSlots，保留owner/page发布和线程1模式发布规避。

不再建立联合哈希表、没有联合线性探测、miss阶段不再扫描空哈希桶。unique顺序由原子胜者和tile内扫描决定，可能改变具体token的分配槽号；原集合、LRU分段、逐位置token/slot及descriptor语义校验继续约束结果。

仍然在AIV上使用SIMT；这不是“已经改成SIMD向量指令”的实现。每请求一个AIV的调度仍保留，因此没有解决32请求只能用32个请求组AIV的问题。也没有新增跨核同步。选择先改变数据结构，是为了隔离其收益；如果不足，后续跨核拆分必须连同workspace可见性和launch成本一起设计。

官方接口依据（开发预览与9.1文档分开看）：

- [asc_atomic_or](https://asc.gitcode.com/api/SIMT-API/atomic_operations/asc_atomic_or.html)：950PR/DT支持；int32/uint32可用于UB/GM；返回更新前的值，多个线程的执行先后不保证。
- [asc_atomic_and，CANN9.1](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/atomic_operations/asc_atomic_and.md)：可原子清位，本候选不需要清除命中位。
- [asc_popc，CANN9.1](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/scalar_compute/asc_popc.md)：是PIPE_S标量接口，不能作为SIMT warp归约的直接替代，本版未使用。
- 官方C API目录中的四路归并要求已排序队列；当前indexer输出未保证有序，不能省略排序成本直接套用。

## 空间、风险与实验边界

生产T2048/C4096/maxToken262144：位图32KiB，unique16KiB，partition16KiB；为兼容原private回退，scratch取与原72KiB工作区的最大值，动态UB仍74016B。

另需每行一个maxToken长的int32直接索引表，约1MiB。当前实现按query行隔离GM区，并预留GM scratch；MTP3、128行总workspace约137MiB。只初始化本次唯一token对应的map项，其他项即使污染也不读取。测试恢复/poison全workspace位于目标kernel之外；这会影响cache状态，属于固定输入地址的微基准条件，不代表生产缓存状态。

maxToken更大时位图增长，scratch可能落到GM。因此“直接索引一定更快”不成立；原子位图同字竞争、GM map随机读写、清位图及并行度都可能成为瓶颈。只把本版当独立可证伪实验。

输入状态契约仍要求LRU为合法排列，联合resident中的有效token唯一。没有修复任意损坏LRU的恢复语义。保留原始ABCDF、联合状态转换、精确C/C+1回退和只读输入门禁，不把错误输入修复成正确答案。

## 缩短上机反馈周期

默认只比较 `abcdf joint_abc joint_bitmap`，只跑75%/independent的MTP0/1/3三个工况。不要再默认做27工况×19实现。

在仓库根目录：

```bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/workspace/joint_bitmap_results/tmp
mkdir -p "$TMPDIR" /workspace/joint_bitmap_results
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
bash build_joint_plan.sh --soc=ascend950 \
  2>&1 | tee /workspace/joint_bitmap_results/build.log

# 设备沿用本轮用户已验证的逻辑卡0；明确选择，避免默认卡2。
"$PYTHON" -X faulthandler checks/check_joint_plan.py --device 0 \
  --implementations abcdf joint_abc joint_bitmap --regression-repeat 100 \
  --temp-dir "$TMPDIR" --report /workspace/joint_bitmap_results/gate.json \
  2>&1 | tee /workspace/joint_bitmap_results/gate.log

# 第一轮只收90个目标样本；完整门禁复用，不再次100次回归。
"$PYTHON" -X faulthandler profile_joint_mtp.py --device 0 \
  --implementations abcdf joint_abc joint_bitmap --mtp 0 1 3 \
  --requests 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --scenarios full_cache_75pct --overlaps independent \
  --warmup 3 --repeat 10 --temp-dir "$TMPDIR" \
  --correctness-report /workspace/joint_bitmap_results/gate.json \
  --output /workspace/joint_bitmap_results/pilot \
  2>&1 | tee /workspace/joint_bitmap_results/pilot.log
```

若短跑仍明显慢于ABCDF，先分析这三组数据，不扩大矩阵。只有出现值得保留的改善，再设置repeat50、增加shared/partial/disjoint及90%/stable_hit。至少保留一次disjoint/MTP3回退验证，不能只报告高重合收益。

性能依据仍是精确kernel的CSV Duration，每次调用恢复状态、检查只读输入和全部语义；数据索引、原始CSV、失败现场留在运行机器。新kernel名为`sparse_kv_plan_joint_kernel_bitmap`。只测Plan，不能宣称端到端获益。

核心代码在`csrc/joint_variants/bitmap_plan.h`，CPU仿真入口`checks/check_joint_variant_kernels.py --variants abc bitmap`。CPU用64线程和UBSan运行真实VF/helper，不能验证950地址空间、DCache和设备编译器。
