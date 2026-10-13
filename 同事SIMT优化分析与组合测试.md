# 同事 SIMT 优化与 BCDE 的组合评估

本次使用正确的源码：同事仓库 `simt@95d520858912ef033435782b1cfbf51a930c015d`。
测试框架从已有 `feat/simt-variant-comparison@e9ababc` 派生，新分支为
`feat/peer-simt-comparison`。上一轮读到 echo 分支的结论不适用于这份 SIMT 代码。

## 按同事实测结果收缩候选

下表倍数是用户转述的服务器结果，本机没有原始逐场景数据，也没有复现实测。

| 同事版本 | 转述结果 | 与我们现有代码的关系 | 本次处理 |
|---|---:|---|---|
| 09 cached tiling | 6.6× | Host 层缓存，BCDE 没有这项直调接口优化 | 优先接入，严格区分接口与 kernel 收益 |
| 04 parallel scan | 1.13× | 与 B 同方向；同事是一个 warp 每 lane 扫两个 total，我们 B 支持分组循环 | 保留独立对照，另测试 pair 快路径＋B 分组回退 |
| 06 packed lists | 1.12× | 与 C 相同：evict 从前端写、hit 从后端反向写，共用 C 个元素 | 保留独立对照，BCDE 已有此项 |
| 01 threads_1024 | 1.11× | 修改编译期线程数，原 BCDE 为 2048 | 接入独立版及 BCDE 1024 版本 |
| 02/03/07/08/10 | 0.98/0.93/0.94/0.96/1.02× | 部分与 B、D 已重合；其余这轮没有明确净收益 | 按用户要求不接入这些独立实验，不加入推荐组合 |
| 05 fused lookup | 未提供 | 保存插入所得 bucket，合并 query hit 回填和 miss 判定 | 保留为探索项；不标记为已有实测收益 |

独立版有收益，不意味着把倍数相乘后就是组合收益。线程数会改变 tile 数、barrier 次数和
资源使用；C 的压缩布局可能使另一版本已经切入 UB；Host 缓存则可能让 kernel 成为新的
主导开销。每一对都需要相同输入、相同持久状态和相同计时口径。

## 09 的 6.6× 怎么理解

源码确认：同事 `00_baseline` 和 `09_cached_tiling` 的 kernel、config、launch header
逐字节相同；变化发生在 Python 可调用的 C++ plugin。

基线每次调用都执行：

```cpp
at::Tensor tilingCpu = at::empty({sizeof(SparseKvPlanTilingData)}, byte_options);
std::memcpy(tilingCpu.data_ptr(), &tiling, sizeof(tiling));
at::Tensor tilingDev = tilingCpu.to(reqIds.device(), false);
const uint32_t blockDim = calc_sparse_kv_plan_block_dim();
```

09 将设备核数缓存，并按 device、stream、全部 tiling 字段查询设备 tiling Tensor。
命中时直接复用不可变的 tiling，不再每次分配及上传；callback 仍按值持有 Tensor。
缓存有锁和有限容量，首次遇到新 key 仍会上传。

同事 `scripts/npu_runtime.py:92` 的测量顺序是：

```python
restore_device(...)       # 计时前恢复状态并同步
start.record()
invoke(...)              # 包含 plugin 的准备、tiling H2D、kernel 下发
end.record()
end.synchronize()
event_us = start.elapsed_time(end) * 1000
```

因此 Event 不是纯 kernel 时间。start 已提交后，CPU 分配 Tensor、上传 tiling、等待或
组织下一条任务时，设备 stream 的间隔也会计入。09 能降低这段接口耗时，6.6× 并非
仅凭倍数就能判定为异常。但它不能被解释为“LRU 算法或 NPU kernel 快了 6.6×”。

若相同 shape、相同输入、相同采样的纯 kernel 时间也显示 6.6×，应优先排查：

1. 是否拿相同场景的原始 P50 作分子/分母，而不是混合不同 shape 或统计项。
2. 正式样本前是否恢复了同一持久状态，是否把 cold/miss 变成了 all-hit。
3. 加载 `.so` 与源码指纹是否一致，UB/GM、线程数、实际执行行数是否相同。
4. profiler 是否对相同 kernel 实例计时，有无其他进程负载或缓存状态差异。

本框架新增 `combo_bcde_cache`，只改变 BCDE 的 Host tiling 处理；它与 `10_bcde` 成对
对比。再用 `combo_bcde_1024` / `combo_bcde_1024_cache` 验证更少线程下的缓存收益。
先看 Event 和 Host 数据，再用现有 profiler 比较 kernel duration。

注意：这里的缓存针对直调测试接口手动上传 tiling 的成本。不能把这份 6.6× 直接推广
到生产 vLLM Ascend/aclnn 路径；生产接口的 tiling 管理还需单独核对。

## 新增 BCDE 组合

| 名称 | 算法与线程 | Host 缓存 | 用途 |
|---|---|---|---|
| `10_bcde` | 原 BCDE，2048 线程 | 无 | 原有参照 |
| `combo_bcde_cache` | 原 BCDE，2048 线程 | 有 | 单独验证 09 的接口收益 |
| `combo_bcde_1024` | 原 BCDE，1024 线程 | 无 | 单独验证线程数 |
| `combo_bcde_1024_cache` | 原 BCDE，1024 线程 | 有 | 组合已有收益方向 |
| `combo_bcde_fused` | BCDE＋F，2048 线程 | 无 | 验证 bucket 复用及融合查询 |
| `combo_bcde_fused_cache` | BCDE＋F，2048 线程 | 有 | 分离 F 与接口缓存收益 |
| `combo_bcde_fused_1024` | BCDE＋F，1024 线程 | 无 | 验证 F 与线程数的交互 |
| `combo_bcde_fused_1024_cache` | BCDE＋F，1024 线程 | 有 | 完整探索组合 |

B/C/D/E 主体保留，不再重复引入同事 06/08 的相同策略。默认 BCDE 仍是 2048 线程；
1024 是独立实验，不改生产算子。

F 相比同事原版还利用既有有效性判断结果：插入时已检查 token、visible 长度、request、
block table；保存返回 bucket 后，融合阶段不用再次读取这些源元数据。

`missPositions[T]` 在插入阶段暂存每个 query 的 bucket，随后原地变成有序 miss 列表。
安全条件是：每个 tile 先把 bucket/owner 读到线程局部变量，再执行 block scan 中的
barrier，然后写 compact 结果。输出位置不超过原 query 位置，前一个 tile 不覆盖尚未
读取的后一个 tile。保留 scan 后 scatter barrier，不增加 T 大小数组。

重复 TopK、重复 resident token、未分配 miss 的 `-1` 输出、request-ID 重置和 MTP
后缀失效语义全部沿用原版。稳定前缀失效仍进入 warm 路径，不能当成新 request 的冷启动。

同事 04 的 pair scan 最多覆盖 64 warp；其独立包装增加编译期限制，避免未来改为超过
2048 线程时静默遗漏。我们的 B 仍保留按 32 项分组扫描的写法。

### 04 与 B 的进一步组合

`combo_bcde_pair_scan` 将 64 warp 以内的 total 两两分配给 lane，用一次 shuffle scan
生成基址；2048 线程时省去原 B 第二组 scan。`PLAN_WARP_COUNT > 2*PLAN_WARP_SIZE`
时仍走原 B 的分组算法，保持解耦。没有新增 barrier 或 scratch。

另提供 `combo_bcde_pair_scan_cache`、`combo_bcde_pair_scan_1024_cache`，分别组合 09 和
1024 线程。它们是新组合，尚无 A5 实测收益结论。

## 接入方式与源码位置

- 原有变体仍在 `csrc/variants/`。
- 新增独立版本和组合在 `csrc/peer_variants/<名称>/upstream/`，里面是可直接阅读的四份源码。
- `kernel.cpp`、`plugin.cpp`、`symbol_aliases.h` 做独立名称包装。
- kernel、VF、launch、核数查询、Host inline helper、tiling 类型和 Torch op 均隔离，
  防止把多个版本放进同一 `.so` 后发生符号冲突或错误复用另一版的 tiling helper。
- plugin 注册层仅调整 Torch op 名和 profiler 标签，其他入参、mutation 和 callback 保留。
- `peer_variants_manifest.json` 记录来源、线程数、布局和逐文件指纹。
- `.so` 内固化相同身份信息；源码或加载库不匹配时拒绝计时。

## 服务器测试命令

在当前工程根目录，使用已经能运行 torch/torch_npu 的环境：

```bash
bash build.sh

# 推荐先跑：已有收益方向的组合，参照 BCDE。
python3 compare.py --device 0 --rows 32 --topk 2048 --capacity 4096 \
  --implementations 10_bcde combo_bcde_cache combo_bcde_1024 combo_bcde_1024_cache \
  --reference 10_bcde --warmup 10 --repeat 80 --json results/bcde_cache_threads.json

# 比较我们的原 B 与同事 pair 写法，以及 pair/cache/1024 组合。
python3 compare.py --device 0 --implementations 10_bcde combo_bcde_pair_scan \
  combo_bcde_pair_scan_cache combo_bcde_pair_scan_1024_cache --reference 10_bcde \
  --repeat 80 --json results/pair_scan.json

# 同事有收益的独立版与我们 B/C/D/E/BCDE 对照。
python3 compare.py --device 0 --implementations \
  02_b_parallel_scan 03_c_compact_partition 04_d_cold_start 08_e_validity_markers 10_bcde \
  peer_00_baseline peer_01_threads_1024 peer_04_parallel_scan peer_06_packed_lists peer_09_cached_tiling \
  --reference 10_bcde --repeat 100 --json results/peer_positive.json

# 探索 F；先做短审计，再去掉 verify-each-sample 测正式性能。
python3 compare.py --device 0 --implementations 10_bcde combo_bcde_fused combo_bcde_fused_cache \
  combo_bcde_fused_1024 combo_bcde_fused_1024_cache --reference 10_bcde \
  --warmup 2 --repeat 5 --verify-each-sample --json results/fused_audit.json

# 检查 hash 扩容导致的 GM 路径；同时可测 capacity=8192。
python3 compare.py --device 0 --capacity 4096 --topk 2049 \
  --implementations 10_bcde combo_bcde_cache combo_bcde_1024_cache --reference 10_bcde
```

正式测试每次恢复相同状态，轮转版本执行顺序。cache 保留，以测稳态接口开销；第一次
创建 tiling 的代价不等于稳态代价。频繁 D2H 审计会改变采样间隔，因此短审计结果不作为
正式性能结论。

## 本地验证范围

本机没有 CANN/NPU/torch。可运行 NumPy 参考、C++ Host ABI/layout 检查和 ASAN/UBSan
内核算法模拟。真实 Torch schema 协议脚本保留供服务器运行，不能声称本机运行过。
Host shim 不模拟 NPU 地址空间、cache coherence 或 BiSheng lowering，结果不代表硬件
性能或完整 NPU 编译成功。

本地实际结果：用 64 个 CPU 工作线程执行实际 VF 函数的模拟，16 个新增版本各通过
116 项状态 oracle 检查，共 1856 项；这不等于完整核函数以 1024/2048 线程上机验证。
覆盖 request 重排、同 request 多行 MTP、重复项、非法来源、后缀失效、跨 tile 和
按 Host 公式选择的 UB/GM 路径。B/C/D/E/BCDE 回归各 108 项，B/BCDE 二级扫描
另按原始 2048 线程/64 warp 配置各检查 128 组。Host 头文件 96 组布局、80 字节
ABI、多目标 inline helper 符号隔离和 16 份 schema 静态检查通过。新增 pair/grouped Scan
按 32、96、1024、2048、3072 线程所对应的 warp 数各检查 128 组，共 640 组。3072
只检查大 warp 数的算法回退，不代表平台支持这一启动配置。
这些均是本地 CPU 检查；未执行 Torch 协议脚本、CANN 编译或 NPU 性能测量。
