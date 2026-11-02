# Turbo：分阶段查询和每行 Scan 的对照实验

本分支基于 `f5b44bf`。目标是减少 MTP 联合规划的重复 CAS 与 Scan，不承诺已经超过 ABCDF。开发机没有 CANN/NPU；本地 CPU 仿真不能验证设备地址空间、真实同步、寄存器占用或性能。

## 六条对照路径

| 实现 | 内容 |
|---|---|
| `abcdf` | 独立 ABCDF，沿用本轮已去除来源页表检查的版本 |
| `private_turbo` | 已有 Turbo 逐行实现 |
| `joint_turbo` | 已有 Turbo 联合实现，mask17 |
| `joint_turbo_a` | mask18：冻结 Hash 只读查询，然后仅对不存在的 token 做 CAS |
| `joint_turbo_b` | mask19：并集构建每行一次 Scan |
| `joint_turbo_ab` | mask20：组合 A 和 B |

这里的 A/B 与历史 `joint_a/joint_b`、ABCDF 字母无关。新候选仅替换 `BuildJointUnionTurbo`，后续 resident 分类、LRU、miss 分配、逐位置寻址、线程1模式发布均复用原 Turbo。MTP0 不构建并集，仍走逐行回退；它可以检验同一个 VF 包含不同代码后是否有编译资源开销变化。

阅读入口：`csrc/joint_variants/turbo_union_experiments.h` → `turbo_plan.h` → `kernel.cpp`。Python 名称与 mask 在 `joint_variants.py`；prepared 布局在 `joint_plan_common.py` 与 `csrc/joint_variants/plugin.cpp`。

## A：安全地减少重复插入

首行仍全部 CAS 插入。后续每行按以下阶段执行：

1. 上一行结束的 block barrier 保证 Hash 写入完成。
2. 全 block 只读查询前序行的 Hash，查询结果写入现有 `queryBuckets`。已有 token 保存桶号；不存在的有效 token 保存待插入标记；无效/不可见 token 保存 -1。
3. block barrier 等待所有普通读取结束。
4. 只有待插入项调用原 CAS 探测循环。同一行的重复项仍由 CAS 去重。
5. 完成本行新增项的 Scan 和发布，再结束本行。

这与“每个线程先普通读、随后立即 CAS”不同：普通读取与 CAS 不在同一阶段并发，不能删掉两个阶段之间的 barrier。无需 query 有序或共享前缀，仍按每行 visible length 判断因果可见性。

收益候选是跨行重复 token 避免 CAS；代价是额外查表、queryBuckets 往返、同步以及未命中项二次探测。低重合/溢出场景可能更慢。

## B：把多个 tile 的 Scan 合成一行

原 Stage0 对每 1024 个 query 做一次 block Scan。生产 TopK2048 时，每行两次；MTP3 四行共八次。

B 中每线程先处理自己的所有 query，统计新增数量；在现有 queryBuckets 中暂存 `-(bucket+2)` 标记新增桶，-1 表示无效项。全行插入完成后，每线程以新增数量参与一次 Scan，获得自己的唯一编号区间，再解包桶号、散写新增项。不增加 scratch 数组，也不增加 OR 原子操作。

MTP3 的 Stage0 Scan 次数由8变4。这不是整个 kernel 的 Scan 次数，更不代表总 barrier 或耗时减半：行尾同步仍保证 Hash 读取/散写完成，后续 LRU 等阶段原样保留。线程处理多个位置及临时标记读写也有成本。

新增项枚举顺序允许变化，但每个有效 token 只有一个槽/描述符，未命中的槽按原 evict 顺序分配，resident/LRU/逐位置输出仍须通过原语义校验。union>capacity 在写持久缓存前检测，回退原 RunPrivateRow。Hash 容量大于标记安全范围时保留分 tile Scan。

## 固定的配置和输入

- 1024 SIMT 线程、一请求组一个 AIV、-O3 和既有编译选项不变。
- 生产 joint scratch 为112 KiB，另有288B Scan；dynamic UB=114976B。超预算仍走 GM。
- MTP=0/1/3 为0/1/3个 draft，32个请求对应32/64/128行。
- 不改 `make_case` 输入；六路共享逻辑输入，各自恢复初始可变状态。
- `full_cache_75pct/independent` 只有首行约75%命中；后续行独立选词，实际命中率通常更低。原模型还有 stable-prefix 失效和可见的新 token，不能按“每行512 miss”解释。
- 报告新增 `initial_hit_positions_per_row`、`initial_hit_fraction_per_row`，计数遵循 request reset、stable prefix 和每行可见性。这是各行自身初始缓存的命中，不是跨行重合率。实际 descriptor 数仍从正确性审计结果记录。

## 上机：六路小矩阵

保留之前的工程和结果，另克隆目录。下列分支应固定到交付的提交号再构建：

```bash
cd /workspace
git clone --branch feat/joint-turbo-phase-scan --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git turbo_phase_scan
cd turbo_phase_scan
git rev-parse HEAD
# git checkout --detach <交付提交号>
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export PYTHON=/usr/local/python3.11.10/bin/python3
OUTPUT=/workspace/turbo_phase_scan_smoke bash scripts/test_turbo_phase_scan.sh
```

脚本使用 NPU0，先构建，再完整六路门禁（已知输入错写回归5次），最后采集：cold_reset/full_cache_75pct × independent × MTP0/1/3，共6工况。默认 warmup2/repeat5；不运行19路/27工况。已有门禁覆盖 disjoint/MTP3 溢出、UB/GM、连续状态、输入不变性；新增生产双tile重复/未来token检查。任何步骤失败立即停止，保留失败现场。

同一次 profiler 轮换六路顺序，逐样本核对输出和输入。严格匹配 kernel 名、样本数和顺序；开始 ACTIVE 后1秒宿主等待保留并记录。只统计精确目标 kernel 的 Duration(us)，不加入恢复、CPU检查或 TensorMove。

输出位于指定 `/workspace`：build.log、gate.json、profile.log、profile/joint_matrix_*/report.md、capture_metadata.json、data_index.json，以及原始 CSV。报告显式比较候选/ABCDF、候选/private_turbo、候选/joint_turbo；比值小于1才表示候选更快。只减少自身开销不等于超过 ABCDF。

若小矩阵通过且有收益，再使用同库/设备门禁采集更多样本，不重复构建/门禁：

```bash
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
export TMPDIR=/workspace/turbo_phase_scan_smoke/tmp
"$PYTHON" -B -X faulthandler profile_joint_mtp.py --device 0 \
  --implementations abcdf private_turbo joint_turbo joint_turbo_a joint_turbo_b joint_turbo_ab \
  --requests 32 --mtp 0 1 3 --topk 2048 --capacity 4096 \
  --max-token 262144 --block-size 64 --seed 2026 \
  --scenarios cold_reset full_cache_75pct --overlaps independent \
  --warmup 10 --repeat 50 --regression-repeat 5 \
  --correctness-report /workspace/turbo_phase_scan_smoke/gate.json \
  --temp-dir "$TMPDIR" --output /workspace/turbo_phase_scan_confirm
```

没有真实硬件结果之前，不能判定 A、B、AB 哪个最快，也不能将 Plan-only 的 descriptor 减少宣称为端到端收益。此轮不加设备计时写入，不改 Transfer。

## 本地复核入口

```bash
python3 -B checks/check_joint_models.py
python3 -B checks/check_bitmap_launch_structure.py
python3 -B checks/check_joint_variant_kernels.py --variants turbo_a turbo_b turbo_ab
```

最后一项执行真实 VF 源码的64线程 CPU shim + UBSan，检查全部保护区和输入不变性；包括重复、同桶冲突链、union=capacity/capacity+1、空并集、因果可见性、连续状态、自然 GM 和生产 UB 边界。CPU执行时改变了线程数，不能据此声称1024线程 NPU 已验证。
