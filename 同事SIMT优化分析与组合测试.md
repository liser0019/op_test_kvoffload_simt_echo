# 同事 SIMT 优化与 BCDE 的组合评估

2026-10-13 更新：用户实测后选定 `combo_bcde_fused_1024` 为后续基线。
本文件保留先前的候选筛选过程；当前基线见 [优化基线说明](BCDE_fused_1024优化基线说明.md)。

同事源码来自 `simt@95d520858912ef033435782b1cfbf51a930c015d`。
测试框架从 `feat/simt-variant-comparison@e9ababc` 派生，当前分支为
`feat/peer-simt-comparison`。上一轮读到 echo 分支的结论不适用于这份 SIMT 代码。

按用户要求，本轮只比较 kernel 的算法、线程组织和临时布局优化。Host tiling 缓存
独立版及包含它的全部组合已删除；它此前的接口计时收益不纳入本轮结论。

## 候选筛选

下表倍数是用户转述的服务器结果，本机没有原始逐场景数据，也没有复现实测。

| 同事版本 | 转述结果 | 与我们现有代码的关系 | 本次处理 |
|---|---:|---|---|
| 04 parallel scan | 1.13× | 与 B 同方向；首 warp 每 lane 扫两个 total，我们 B 按组扫描 | 独立对照，增加 pair 快路径＋B 分组回退组合 |
| 06 packed lists | 1.12× | 与 C 相同：evict 从前端写、hit 从后端反向写，共用 C 个元素 | 独立对照，BCDE 已有此项 |
| 01 threads_1024 | 1.11× | 改变编译期线程数，原 BCDE 为 2048 | 独立对照及 BCDE 1024 组合 |
| 05 fused lookup | 未提供 | 保存插入所得 bucket，合并 query hit 回填和 miss 判定 | 可选探索项，不标记为已有实测收益 |
| 02/03/07/08/10 | 0.98/0.93/0.94/0.96/1.02× | 本轮没有明确净收益 | 不新增接入 |

独立版有收益，不意味着可以直接把倍数相乘。线程数会改变 tile 数、barrier 次数和
资源使用；压缩布局也会影响 UB/GM 路径。应使用相同输入、持久状态和计时口径。

## 保留的独立版本和组合

保留 5 个同事独立版本：`peer_00_baseline`、`peer_01_threads_1024`、
`peer_04_parallel_scan`、`peer_05_fused_lookup`、`peer_06_packed_lists`。

| 名称 | 算法与线程 | 用途 |
|---|---|---|
| `10_bcde` | 原 BCDE，2048 线程 | 原有参照 |
| `combo_bcde_1024` | BCDE，1024 线程 | 单独验证线程数 |
| `combo_bcde_pair_scan` | BCDE＋pair Scan，2048 线程 | 比较同事 04 与原 B 的 Scan 实现 |
| `combo_bcde_fused` | BCDE＋F，2048 线程 | 验证 bucket 复用及融合查询 |
| `combo_bcde_fused_1024` | BCDE＋F，1024 线程 | 验证融合查询与线程数的交互 |

共 9 个新增候选，与原 SIMT、Echo 和原来的 11 个变体一起构成 22 路。
B/C/D/E 主体保留，默认 BCDE 仍是 2048 线程；1024 是独立实验，不改生产算子。

### 融合查询 F

插入时已检查 token、visible 长度、request 和 block table。F 将返回的 bucket 暂存到
`missPositions[T]`，后续合并 hit 回填与 miss 判定，避免重新查表和部分重复元数据读取。

临时 bucket 数组随后原地变成有序 miss 列表。每个 tile 先把 bucket/owner 读到线程
局部变量，再执行 Scan 的 barrier，随后写 compact 结果。输出位置不超过原 query
位置，前一个 tile 不覆盖尚未读取的后一个 tile；保留 scatter 后的 barrier。

融合版的 hit 槽号仍从 GM `lruSlots` 读取，LRU 重建时的新 miss 槽号仍从 GM
`missSlots` 读取。当前没有独立的 scratch LRU 读取优化，不应将它计入 F 的收益。

重复 TopK、重复 resident token、未分配 miss 的 `-1` 输出、request-ID 重置和 MTP
后缀失效语义沿用原版。稳定前缀失效仍进入 warm 路径，不能当成新 request 的冷启动。

### pair Scan

同事 04 的独立版最多覆盖 64 warp，包装层保留编译期限制，避免未来改为超过 2048
线程时遗漏。`combo_bcde_pair_scan` 在 64 warp 以内将 total 两两分配给 lane，以一次
shuffle Scan 生成基址；2048 线程时省去原 B 的第二组 Scan。
`PLAN_WARP_COUNT > 2*PLAN_WARP_SIZE` 时仍走原 B 的分组算法。没有新增 barrier 或
scratch。这个新组合尚无 A5 实测收益结论。

## 源码与构建

- 原有版本：`csrc/variants/`。
- 新增版本：`csrc/peer_variants/<名称>/upstream/`。
- `kernel.cpp`、`plugin.cpp`、`symbol_aliases.h` 提供独立名称包装。
- kernel、VF、launch、核数查询、Host inline helper、tiling 类型和 Torch op 均隔离，
  避免多个版本合入同一 `.so` 时发生符号冲突或 tiling helper 错误复用。
- `peer_variants_manifest.json` 记录来源、线程数、布局和逐文件指纹。
- `.so` 固化相同身份信息；源码或加载库不匹配时拒绝计时。删除版本后须重新构建。

## 服务器测试命令

在工程根目录，使用已经能运行 torch/torch_npu 的环境：

```bash
bash build.sh

# 推荐先跑线程数和 Scan，参照 BCDE。
python3 compare.py --device 0 --rows 32 --topk 2048 --capacity 4096 \
  --implementations 10_bcde combo_bcde_1024 combo_bcde_pair_scan \
  --reference 10_bcde --warmup 10 --repeat 80 --json results/bcde_threads_scan.json

# 同事独立版本与我们的 B/C/D/E/BCDE 对照。
python3 compare.py --device 0 --implementations \
  02_b_parallel_scan 03_c_compact_partition 04_d_cold_start 08_e_validity_markers 10_bcde \
  peer_00_baseline peer_01_threads_1024 peer_04_parallel_scan peer_06_packed_lists \
  --reference 10_bcde --repeat 100 --json results/peer_positive.json

# F 作为探索项：先短审计，再去掉 verify-each-sample 测正式性能。
python3 compare.py --device 0 --implementations 10_bcde peer_05_fused_lookup \
  combo_bcde_fused combo_bcde_fused_1024 --reference 10_bcde \
  --warmup 2 --repeat 5 --verify-each-sample --json results/fused_audit.json

# hash 扩容后的 GM 路径；也可测试 capacity=8192。
python3 compare.py --device 0 --capacity 4096 --topk 2049 \
  --implementations 10_bcde combo_bcde_1024 combo_bcde_pair_scan --reference 10_bcde
```

每次调用前恢复相同状态，轮转版本执行顺序。Event 仍测公共接口的 stream 区间，包含
Host 下发间隔及原有 tiling 上传，不能直接称为纯 kernel 时间；应结合 profiler 的
kernel duration 判断设备优化。短审计的频繁 D2H 会改变采样条件，不作为正式性能结论。

## 本地验证范围

本机没有 CANN/NPU/torch。保留的 9 个新增版本此前均在 64 个 CPU 工作线程下执行
实际 VF 函数模拟，各通过 116 项状态 oracle 检查，共 1044 项，包括 request 重排、
同 request 多行 MTP、重复项、非法来源、后缀失效、跨 tile 和 UB/GM 路径。
本次删除没有修改保留版本的 kernel。

Host 头文件检查按保留的 9 个版本和 6 组 shape 核对，共 54 组，覆盖 80 字节 ABI、
临时空间、动态 UB 及多目标 inline helper 符号隔离；同时核对注册 schema 和构建列表。
此前 pair/grouped Scan 按 32、96、1024、2048、3072 线程对应的 warp 数各检查 128 组，
共 640 组。3072 只检查算法回退，不代表平台支持这一启动配置。

CPU shim 不模拟 NPU 地址空间、缓存一致性或 BiSheng lowering。未执行真实 Torch
协议脚本、CANN 编译或 NPU 性能测量。
