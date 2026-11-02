# Merge ABCD：减少查询、分段归并与中间写入

本轮以带页表独立 `abcdf_pages` 为性能基线，以已上机的 `joint_merge` 为算法对照。新增 `joint_merge_a/b/.../abcd` 共15个组合；它们的字母与旧 Joint/Turbo 实验无关。代码在 `csrc/joint_variants/merge_experiments.h`，编译常量 `JOINT_OPT_MASK-21` 对应 A/B/C/D 四个位；原 `merge_plan.h` 和独立 ABCDF 算法保持原样。

## 优化点及代价

| 实验 | 改动 | 验证重点 / 代价 |
|---|---|---|
| A | resident 插入时记录 bucket；LRU 分类直接读取 bucket，避免第二次 Hash 探测 | 复用 partition 高16位、低16位写最终槽。分类阶段读和写全部采用原子操作，避免输出覆盖未来输入时的数据竞争。增加原子读取/OR成本，因此可能负收益 |
| B | 两两归并：每8个输出先做一次 co-rank 分区，然后顺序读取两路；4路只需两轮 | 用现有 hash/owner 与 packedMiss 两块空间交替；最后交换指针，UID写入死缓冲。不额外申请UB。仍是SIMT软件归并，不是Vector MrgSort |
| C | 命中只写最终槽；miss延迟到成功提交后写槽，非法token直接写-1 | 去掉 currentSlots 的初始-1写入。联合回退仍由原 private 完整覆盖结果 |
| D | 过滤、LRU分类、去重的中间tile取消屏障，保留各阶段最后tile屏障 | 依赖已有ping-pong Scan，每阶段数组发布屏障保留；仍需真机验证同步与编译行为 |

全部组合用相同1024线程、同一host tiling、相同持久协议和同一当前stream。MTP0仍执行逐行路径，这些优化针对联合阶段，不能将MTP0的编译布局差异解释成归并收益。生产MTP3仍为112KiB scratch +288B Scan；较大shape原样走GM。A没有通过扩大UB来容纳bucket缓存。

B的意图是显著减少逐元素二分；仍可能受连续段处理降低活跃线程数、两轮额外读写影响。A、C、D的组合也不保证可叠加收益。排名必须来自同轮硬件数据。

## 服务器快速筛选：只测热场景、MTP0/1/3

```bash
cd /workspace
git clone --single-branch -b feat/merge-plan-abcd https://github.com/liser0019/op_test_kvoffload_simt_echo.git merge_plan_abcd
cd merge_plan_abcd
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
bash scripts/test_merge_abcd.sh
```

使用逻辑NPU0、32请求、TopK2048/capacity4096/max_token262144/block64/seed2026。输入模式 overlap40；所有实现共用按token升序的查询，至少40%首行重合，排序在计时外。每行实际命中率和首行重合率分别记录。默认预热2、采样5，3个输入工况、51个实现/工况组合（3种MTP ×17条路径），共255个正式目标kernel样本；不是此前27工况展开的矩阵。脚本依次执行：

1. `build_joint_plan.sh --merge-only` 编译原Merge和15个新候选，跳过旧21路Joint候选的设备编译；保留所需基线。未编译的variant在prepared构造时拒绝。构建仍使用-O3，构建子集记入库身份。
2. 专项门禁：每路5次原输入错写尺寸回归；小尺寸MTP0/1/3冷/热、溢出、重复resident、重复query、因果无效项、部分活动行、生产UB和自然GM布局；另有连续状态与prepared绑定校验。每个边界用例做一次完整数值审计，减少重复；不是关闭校验。报告 scope 为 merge_abcd_targeted_v1。
3. 复用本次同设备同库门禁，逐样本检查后采集kernel Duration。保留ACTIVE开始后1秒宿主等待，严格匹配目标kernel、数量和实际下发顺序。候选循环轮换先后顺序；5次短跑无法让17路完全平衡，需复测候选。

结果在新的 `/workspace/merge_abcd_*/profile/joint_matrix_*/report.md`。其中“Merge ABCD 全组合”表列P50及候选/带页表ABCDF、候选/原Merge两种比值，小于1才表示更快。还需检查共享组数、回退组数及描述符数量，不能将回退解释成联合加速。原始CSV、执行顺序和库身份保留在同目录JSON与data_index.json。失败即停并保存现场；不上传服务器数据或二进制。

## 对入围候选增加样本

例如筛选后B、BC入围（这里只是命令示例，不代表已测最快）：

```bash
CANDIDATES='abcdf_pages joint_merge joint_merge_b joint_merge_bc' \
SCENARIOS='cold_reset full_cache_75pct' WARMUP=10 REPEAT=50 \
bash scripts/test_merge_abcd.sh
```

`CANDIDATES`必须保留所需基线才能生成相应耗时比；不自动根据5个样本选择赢家。所有比较仅为Plan设备kernel耗时，不含状态恢复/TopK排序/tiling上传；没有实现TopK融合、也没有Transfer/SFA端到端测量。LIM式TopK payload与Vector MrgSort/Squeeze需要独立重排数据布局与SIMD/SIMT交接，本轮没有把未实现的原语列入优化。

## 本地检查与局限

`python3 -B checks/check_merge_abcd.py` 检查15个候选的真实C++ VF，使用64个CPU线程与UBSan；另对分段归并检查空队列、重复key、uint32高位以及8元素边界。保护区、输入不变性、完整LRU、重复resident最后owner、逐位置槽、miss及连续状态由独立语义检查器验证。CPU结果无法验证1024线程调度、UB/GM地址空间、bisheng编译资源和实际硬件性能。
