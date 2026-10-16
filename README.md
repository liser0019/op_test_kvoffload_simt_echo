# BCDE + fused + 1024：Plan 微优化对照

## 本分支新增：ABCDF 与压缩包原版 MemFabric Plan

分支 `feat/abcdf-memfabric-plan-profile` 基于 `9cc99d5`，保留 ABCDF 计算源码，接入压缩包内的原版 LRU compact Plan；不包含地址生成或 KV payload 搬运。以现有 `torch_npu.profiler` 的 `kernel_details.csv` 为计时依据。

详见 [ABCDF 与 MemFabric 原版 Plan 对比](ABCDF与MemFabric原版Plan对比.md)。

```bash
bash build.sh --soc=ascend950
bash run_abcdf_memfabric.sh 1
```

运行前加载匹配的 CANN/torch_npu 环境；卡号按实际调整。本机仅做 CPU/静态验证，未取得 NPU 性能数据。

分支：`feat/simt-plan-micro-variants`，从 `feat/peer-simt-comparison@a282f5d` 创建。
本分支只保留已选定的 BCDE + fused + 1024 基线和本轮 17 个候选，共 18 路；
原有 Echo、旧 SIMT 和其他历史候选在原分支中保留。

基线是 `baseline`，注册名仍为 `combo_bcde_fused_1024`。计算文件的 SHA256 保持
`1114df2886bcd89986ec3efb57e94e638ddbc4975939c394b5bc44a18b199542`。
候选的实机结果以服务器测试为准；本地 CPU 检查不能替代 CANN 编译和 NPU 验证，
也不能把源码减少的指令数直接当作加速比。

## 本轮优化字母

以下 A～H 是新的实验编号，不是历史 BCDE 基线中的字母。

| 点 | 修改 | 保留的语义与边界 |
|---|---|---|
| A | 删除合法 TopK 的 `PENDING_SLOT=-2` 写入及末尾清理读取 | 输出仍先初始化 `-1`，未分配的 miss 保持 `-1` |
| B | LRU 重建直接读 scratch 中已有的 `evictableSlots` | missTokens/missSlots 的公开输出仍完整写出；单独 B 保留分配后屏障 |
| C | miss 输出改为分配前缀写一次、未分配尾部清理一次 | 冷热路径都处理尾部，避免旧结果残留 |
| D | hashOwner 保存 `(order<<16)|slot`，省掉 Hit 的 LRU 间接读取 | capacity≤32768 才打包；更大容量保留原 order/gather；保持最后 owner 规则 |
| E | 行元数据、可见上界、页表基址和页大小外提；二次幂页大小用移位；精简稳定前缀比较 | 负可见长度截断到零；非二次幂页大小用无符号除法；表偏移仍用 int64；MTP 后缀仍失效 |
| F | ≤32 warp 时特化二级 Scan，由 lane31 写总数，省掉回传 shuffle | 所有 32 lane 都参加 shuffle；更大 warp 数保留通用实现 |
| G | 每线程读行 owner 判断 reset，合并热路径输出/hash 初始化的屏障 | 输出初始化仍在冷热分支之前；冷路径首个 Scan 确保 lastReqIds 更新前所有线程已选择分支 |
| H | 减少中间 tile 的 scatter 后屏障，利用下一次 Scan 的第一道屏障 | 最后 tile、阶段边界及行结束的同步保留；不增加 Scan 双缓冲，也不改变 UB 大小 |

A+B 的组合额外删除一处分配后屏障：A 取消对其他线程 currentSlots 写入的清理读取，
B 取消对其他线程 missSlots 写入的回读，两个依赖都消失后才能删除。只有 A 或只有 B 时仍保留。
不包含普通读预检原子、Host tiling 缓存或 Indexer/Transfer 重叠。

## 版本目录

每个版本在 `csrc/plan_variants/` 下有自己的目录和独立入口。
基线源文件在 `00_baseline/upstream/`；候选共用 `common/plan_kernel_impl.h`，
由每个目录的 `variant_config.h` 在编译期选择优化，便于只维护一份算法。

| 版本 | 编译期点 | 内容 |
|---|---|---|
| `baseline` | `baseline` | 原 BCDE + fused + 1024，计算源码逐字保留 |
| `a_no_pending` | `A` | 删除 PENDING_SLOT 中间写入及清理 |
| `b_scratch_lru` | `B` | LRU 重建读取已有 scratch 槽号，不回读 missSlots |
| `c_tail_clear` | `C` | miss 输出只清理未分配尾部 |
| `d_packed_owner` | `D` | 安全范围内打包 order/slot；capacity>32768 保留原路径 |
| `e_row_context` | `E` | 行元数据外提、非负可见长度、二次幂页号移位与稳定前缀区间 |
| `f_warp_total` | `F` | 单 warp 的二级 Scan 特化，省略总数回传 shuffle |
| `g_reset_merge` | `G` | 每线程判断 reset，合并热路径输出/hash 初始化同步 |
| `h_tile_sync` | `H` | 利用下轮 Scan 同步，保留末 tile 屏障，减少中间 tile 尾屏障 |
| `ab` | `AB` | A+B，并删除失去数据依赖的分配后屏障 |
| `abc` | `ABC` | A+B+C |
| `abcd` | `ABCD` | A+B+C+D |
| `abcde` | `ABCDE` | ABCD + 行上下文优化 |
| `abcdf` | `ABCDF` | ABCD + Scan 总数特化 |
| `abcdg` | `ABCDG` | ABCD + reset 同步合并 |
| `abcdh` | `ABCDH` | ABCD + tile 同步削减 |
| `abcdefgh` | `ABCDEFGH` | 全部八项组合；收益需实测 |
| `abcdfh` | `ABCDFH` | ABCD + 二级 Scan 特化 + tile 同步削减，不启用 E/G |

E 的行上下文由每个 SIMT 线程在本行开始时准备一次，不需要 Host 读设备数据。

### F 与 ABCDFH

Scan 分两级：先在每个 warp 内计算前缀和，再由 warp 0 扫描所有 warp 的计数，
得到各组起点。例如计数 `[3, 2, 4]` 对应起点 `[0, 3, 5]`，总数为 9。
1024 线程只有 32 个 warp，这些计数恰好由一个 warp 处理。
F 让 lane31 直接发布总数，省去通用实现的总数回传 shuffle 和跨组 carry 累加，
并为扫描循环增加展开提示；Scan 内两道必要的 block 屏障保持不变。

F 按编译期线程数选择路径，不按 batch、TopK 或 capacity 特化。
完整 warp 配置下，512 线程的 16 个计数补零到 32 项也可使用；
2048 线程有 64 个 warp，会自动保留通用 Scan。
不同形状的 tile 数和 Scan 次数不同，因此收益需要分别测量。

ABCDFH 的掩码为 `175U`，只叠加 ABCD、F、H，不包含 E/G。
F 精简 Scan 内部指令，H 减少中间 tile 的尾部同步；组合没有增加 UB 或 GM 空间，
也没有修改 MTP 的可见长度与后缀失效规则。新组合仍需服务器独立验证，不能把两版收益相加。

重建后可集中比较基线、已测试较好的两版及新组合：

```bash
python3 compare.py --device 1 --implementations baseline abcdh abcdf abcdfh \
    --audit-edges --warmup 2 --repeat 3 --verify-each-sample
python3 profile_plan_variants.py --device 1 --scenario all \
    --implementations baseline abcdh abcdf abcdfh --reference baseline \
    --warmup 8 --repeat 60 --output /dev/shm/abcdfh_comparison
```

公共文件不是运行时 dispatcher。每个版本单独编译为独立的 object、kernel、VF、launch 和 torch.ops。
共享源码、版本选择、Host 头文件的哈希写入 manifest 和共享库；加载时核对，避免拿旧库测试新源码。
修改候选后需要同步 manifest 中的源码哈希并重建；禁止把修改后的 baseline 当作原始实测基线。

## 固定条件与 UB/GM

所有版本使用 1024 线程、32 warp、64 核上限和相同 80 字节 tiling；
默认 query 行数32、TopK2048、capacity4096、max_token262144、block_size64。

```text
H = max(32, next_power_of_two(2*TopK))
row scratch = 4*(3H+capacity+TopK)，按32B对齐
row scratch≤112KiB：UB；否则：GM
Scan = 288B，另外计入动态UB
```

TopK2048/capacity4096 是72KiB行临时区，capacity8192 是88KiB，均为UB。
TopK4096/capacity8192 是144KiB，使用GM。此分支不提高阈值、不强迫大尺寸走UB。
所有持久状态保存在GM；只调整Plan内部临时数据和管理操作。该工程不执行KV Transfer或SFA。

## 本地 CPU 检查

Python/NumPy 参考和布局检查不执行 kernel：

```bash
python3 compare.py --cpu-check --audit-edges
python3 checks/check_plan_layouts.py
python3 checks/check_plan_profiling.py
python3 checks/check_plan_goldens.py
```

实际提取计算源码，用64个CPU工作线程与 ASAN/UBSAN 检查：

```bash
python3 checks/check_plan_kernels.py --report /tmp/plan_cpu_checks.json
python3 checks/check_plan_kernels.py --variants baseline d_packed_owner abcde abcdh abcdefgh     --large --report /tmp/plan_cpu_large_checks.json
```

该模式检查冷热转换、重复TopK/驻留token、非法页表/请求、非二次幂页大小、
负/零可见长度、槽位不足、MTP多行与后缀重写、请求重排、active rows和跨行scratch复用。
大尺寸覆盖8192容量和真实GM回退；D版额外检查32768/32769 owner边界。
含F的版本还用真实1024线程配置、32个CPU工作线程测试二级warp-total Scan。
CPU shim不模拟NPU地址空间、缓存和时延，不能替代CANN编译或实机测试。

## A5 构建与完整采集

使用已有服务器的 torch/torch_npu、CANN9.1 环境，在本目录执行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh  # 按实际安装目录调整
PYTHON=python3 JOBS=3 bash build.sh
PYTHON=python3 bash run_plan_suite.sh 1
```

末尾1是逻辑卡号。脚本先做MTP等边界审计和四场景逐调用正确性审计，
通过后采集四个场景的真实kernel耗时，默认每版72次，构成18路执行位置的四轮轮转。
库为 `build/libsimt_plan_variants_ops.so`，只包含本轮版本。

```bash
# 同一TopK，比较大容量；当前仍走UB，无需修改源码。
CAPACITY=8192 PYTHON=python3 bash run_plan_suite.sh 1

# 打乱LRU顺序，覆盖owner随机间接读取；原四场景默认仍保持原输入方式。
SHUFFLE_LRU=1 PYTHON=python3 bash run_plan_suite.sh 1

# 真正的GM回退尺寸；用于检查优化对GM路径是否同样正确和有效。
TOPK=4096 CAPACITY=8192 PYTHON=python3 bash run_plan_suite.sh 1

# 先比较部分候选，缩短试跑；需要把baseline列入。
python3 compare.py --device 1 --implementations baseline a_no_pending b_scratch_lru c_tail_clear ab abc     --audit-edges --warmup 2 --repeat 3 --verify-each-sample
python3 profile_plan_variants.py --device 1 --scenario all     --implementations baseline ab abc abcd abcdefgh --reference baseline     --warmup 10 --repeat 60 --output /dev/shm/plan_micro_subset
```

如需更换编译器/runtime位置，build.sh继续支持
ASCEND_CANN_PACKAGE_HOME、BISHENG_CXX、ASCENDC_RUNTIME_LIBRARY。
默认输出 `/dev/shm/simt_plan_micro/session_*`，可用 PLAN_PROFILE_OUTPUT 覆盖。
收集 `profile.log`、`capture_metadata.json` 和四份 `kernel_summary.csv`。

## 如何解释结果

四场景为cold_reset、warm_75pct、warm_all_hit和full_cache_75pct；
每路每次调用前恢复同一初始状态和输入，输出与独立CPU参考逐字段全等核对。
workspace和输出在审计时污染，避免残留结果造成假通过。
正确性失败会停止，不生成有效性能结论。

compare.py 的Event区间含参数准备/tiling上传和Host下发空隙，适合初步诊断；
选优化版本应主要比较profile采集的kernel P50/P95，确认每路采集次数一致。
CSV分析不会把TensorMove等传输任务混入Plan耗时。
这些固定命中率测试衡量Plan算法本身，不评估capacity增大后的长期命中率、Transfer节省或整网吞吐。
组合版不保证比单项更快；先看A～H单项，再对照AB/ABC/ABCD与加E/F/G/H的版本。
