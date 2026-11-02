# Merge Joint V1：与原独立带页表 ABCDF 同场对照

脚本 `scripts/test_merge_mtp.sh` 只测 `abcdf_pages` 和 `joint_merge`，不测 Transfer。新实现是独立的 mask 21，旧候选和带页表基线没有切换到 Merge。固定请求数 32；MTP0/1/3 分别是 32/64/128 行，TopK 2048、capacity 4096、max_token 262144。默认测 cold_reset 和 full_cache_75pct、independent 重合模式，先每档预热 2、正式 5 次。75% 是首行构造目标，报告中有各行实际初始命中率。

## 读代码与原型修正

1. `csrc/joint_variants/merge_plan.h` 的 `ProcessRequestRunMerge` 先确定相同请求的连续组、打包位宽及 scratch 边界。MTP0 走原有逐行路径，不能用它衡量归并本身。MTP1/3 对最多 2/4 行联合规划。
2. `BuildAndCommitSortedJoint` 在热路径建 resident Hash。相同 token 选旧 LRU 中最后的 owner，所有已命中的 resident 副本均受保护。逐行按因果上界分流，只有 miss 送去归并。冷路径省去 resident Hash，但所有有效 query 都要归并。
3. 各行 miss 已升序；每个 miss 在其他行二分找排名，用 query 下标打破平局，随后对相邻 token 去重。按 token 升序分配 LRU 中最旧的可淘汰槽；victim 不够就转原逐行算法，不提交半套共享缓存。仅分配序不同不算错：检查 descriptor 顺序、重复 resident、逐位置寻址、完整 LRU、输出尾部和状态。
4. 下载的原型把 token/query 各固定 16 位，256K token 工况会整组回退。这里按组内位置数确定低位并检查 32 位编码是否容纳；8192 个位置占 13 位，262144 种 token 占 18 位。超出编码/形状约束时回退。`residentRows` 是输出，不用它判定热状态；与原 Joint 一样使用 leader 的持久模式和请求身份。只读输入、输出污染和错误现场门禁保留；不要在审计前补写正确结果。
5. 生产 MTP3 的 host 布局为 `2*8192 + 8192 + 4096 = 28672` 个 int32；112 KiB scratch + 288 B Scan。复用现有 mask>=17 的 tiling，无需增大 UB。超过预算的形状使用 GM。该数值是布局计算，不代表实测命中片上路径或实测性能。
6. `--sorted-topk` 在 Host 对所有候选共用的输入按 token 位置升序排序，不去重、不改变可见长度，也不假设各行选中相同历史 token。在 profiler 启动前完成；逻辑输入指纹记录于报告。直接传给候选的未排序用例由测试框架拒绝。底层 VF 自身要求调用方保证排序，不会额外扫描或排序。

没有使用 Vector MrgSort。热路径的 resident 索引仍有 8192 个桶，命中用原子标记；四路时每个 miss 最多在三条流中分别二分。每请求一个 AIV、溢出后该组逐行处理的代价仍在。这些是源码中存在的工作，不能从源码推出加速比。

## 服务器（逻辑 NPU0）

```bash
cd /workspace
git clone --single-branch -b feat/joint-resident-sorted-merge https://github.com/liser0019/op_test_kvoffload_simt_echo.git merge_joint_v1
cd merge_joint_v1
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
bash scripts/test_merge_mtp.sh
```

如服务器的 `set_env.sh` 安装路径不同，沿用之前验证过的 CANN 9.1 环境。脚本会依次构建、运行两路正确性门禁、再用同库同设备的门禁报告采集。失败即停；不会生成成功报告或覆盖旧实验。结果在新的 `/workspace/merge_vs_paged_abcdf_*/`。看 `profile/joint_matrix_*/report.md` 中的两路 P50/P95、样本数、miss 和共享/回退组数；紧邻的 `data_index.json` 索引原始 CSV。输入、采集和二进制只留服务器。

如果短跑有收益，再用 `WARMUP=10 REPEAT=50 bash scripts/test_merge_mtp.sh` 增加样本。只能与同轮带页表 ABCDF 对比；旧结果输入顺序不同。统计精确目标 kernel 的设备 Duration；不含输入排序、状态恢复、TensorMove 与 CPU 等待。减少描述符不等价于端到端提速。

## 本地验证边界

`python3 -B checks/check_merge_plan.py` 使用实际 C++ VF、64 个 CPU 模拟线程与 UBSan，独立检查联合提交、回退、生产 token 编码、排序拒绝、跨 tile 重复、owner 选择、因果过滤、连续状态及 canary。不能代替 1024 线程设备同步、Bisheng 地址空间编译、实际 UB 访问或硬件性能验证。
