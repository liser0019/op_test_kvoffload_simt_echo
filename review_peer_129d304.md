> 历史审阅记录：下文针对 `129d304`，不是当前提交的实机验证报告。
> 其中未实现的 scratch LRU read 描述和过时线程注释已在 `18a4b88` 修复；计算代码保持不变。
> 当前脚本、测试入口及验证边界见 `README_peer_profiling.md`。

# Sparse KV Plan 新增 SIMT 版本审阅结论

版本：`feat/peer-simt-comparison@129d304e159ea5680d07b61e572fdb4b11b8dafb`。

当前未发现新增九路中会阻止上机对照测试的已证实算法缺陷，可以继续服务器编译、正确性与 kernel profiling。此次恢复复用上一轮代码概要、API 预研和 CPU 验证记录，定向完成同步/布局、Host 构建接口、采集协议的剩余核对。没有重新做全量规范条例审计，也没有执行真实 CANN/NPU。

## 实验范围

| 候选 | 对照 | 实际变化 |
|---|---|---|
| `peer_00_baseline` | 同事原版 | 新导入的参照 |
| `peer_01_threads_1024` | `peer_00_baseline` | 2048 改为 1024 线程 |
| `peer_04_parallel_scan` | `peer_00_baseline` | warp 0 每 lane 扫两个 warp total，替代线程 0 串行处理 |
| `peer_05_fused_lookup` | `peer_00_baseline` | 暂存 insertion bucket，融合 hit 发布与 miss scan |
| `peer_06_packed_lists` | `peer_00_baseline` | evict/hit 共用一个 capacity 数组，hit 从尾部逆序写 |
| `combo_bcde_1024` | `10_bcde` | 只改变线程数，scan 空间跟随变化 |
| `combo_bcde_pair_scan` | `10_bcde` | 64 warp 内采用 pair scan，更大 warp 数保留 grouped fallback |
| `combo_bcde_fused` | `10_bcde` | bucket 复用与 hit/miss 融合，去掉 hashOwner 全桶发布 |
| `combo_bcde_fused_1024` | `combo_bcde_fused` | 只改变线程数 |

没有导入 peer02/peer03。Host tiling 缓存版已删除，本轮收益只比较当前保留的算法、线程数与临时布局变化。独立优化的加速比不能相乘预测组合收益。

## 已完成的关键核对

1. **融合版原地复用 bucket 数组**：每个 tile 先读取自己的 bucket，然后进入 scan barrier，最后 scatter 压缩 miss。稳定压缩输出位置不大于原位置，当前 tile 的写入不会碰到未来 tile 尚未读取的 bucket。scatter 后仍有 barrier。
2. **scan scratch 生命周期**：BCDE/combo 减少了 scan 函数内部的末尾 barrier，但 caller 在读取总数和 scatter 后均补有同步，下一轮 scan 不会提前覆盖尚未消费的基址/控制信息。
3. **pair scan 与 packed lists**：pair/grouped 两条路径维持完整 warp 的 shuffle 参与；evict 从前、hit 从后写入的两段满足 E+H≤C，后续逆向读 hit 恢复原顺序。
4. **线程和空间配置**：1024/2048 的 launch bound 与 VF 线程数同源；scan 大小按 warp 数重算。Host、Python 和 Kernel 的 `3H+2C+T` / `3H+C+T` 布局匹配；80 字节 tiling ABI 已有编译检查。
5. **构建与调用隔离**：九路命名空间、POD/helper、核数查询、kernel/VF/launch 及 Torch 注册名称独立。真正参与编译的是包装层和 `plugin_registered.h`，原 plugin 文件保留作源码快照。库固化 peer manifest 身份信息，当前源码/旧库不匹配会拒绝计时。
6. **采集协议**：恢复相同初始状态，执行位置轮转；污染输出审计、预热和 D2H 校验不计正式 kernel 样本。交付脚本只统计匹配的 kernel duration，并检查每版恰好规定次数；不会把 `TensorMove`、Event 接口区间计为 Plan kernel。

## 非阻塞观察

- `peer_variants_manifest.json` 的 `combo_bcde_fused.description` 含 “scratch LRU read”，当前融合实现仍从 GM 的 `lruSlots` / `missSlots` 取相关槽号，未实现该描述中的独立优化。分析结果时以实际代码为准。
- `peer_05_fused_lookup` 相对同事 baseline 的融合收益，与 combo F 相对 BCDE 的收益范围不同：BCDE 的 E 已减少 miss 的重复合法性检查。F 的收益不能重复计入旧 E 的收益。
- 部分源码顶部仍写“2048线程/只搬不改”，不适用于 1024 版和组合版；它们属于注释问题。
- 1024 线程会减少每 block 的 warp 数，同时增加 TopK/capacity tile 数及同步次数。pair scan 和融合也有各自的资源开销，排名需看实测，不能只凭源码确定。

## 已有验证记录与实际边界

- 九路各 116 项 CPU 算法/状态 oracle 检查，共 **1044 项**。实际 VF 逻辑使用 64 个 CPU 工作线程和 ASAN/UBSan 模拟；覆盖重排、MTP、重复项、非法来源、后缀失效、跨 tile、UB/GM。它不模拟 NPU 地址空间和缓存。
- **54 组** Host layout/ABI/独立 helper 检查，覆盖九路与六组尺寸。
- **640 组** pair/grouped scan 检查，含 1024/2048 对应的 warp totals；3072 仅测试算法回退，不代表 NPU 支持此 launch 配置。
- 恢复后，采集脚本的离线 stub 检查通过：审计和预热剔除、状态恢复、顺序轮转、末尾结果校验、旧库身份拒绝。CSV 检查覆盖融合版名称后缀区分、复制任务排除、P50 比值，以及缺失/重复/次数/异常时长拒绝。

本环境未执行真实 CANN 编译、torch_npu 注册/设备运行或性能采样。首轮四场景基准使用合法、无重复、全激活的输入，属于公平性能对照；真实连续 decode、MTP 等业务输入和跨调用 DCache 行为仍需后续 NPU 验证。每次状态复制恢复会改变缓存条件，所以本轮 kernel duration 不能直接替代持续业务的端到端延迟。

## 推荐测量顺序

先在 B/T/C/M=`32/2048/4096/262144` 下测原 BCDE 和九路新增候选的四场景 kernel P50/P95；各版此尺寸都走 UB。首轮取得稳定排名后，保留原 BCDE 与获胜组合做 B1/B4 和 GM 路径对照，再使用服务器已跑通的硬件 profiler 查看寄存器资源、SIMT stall 与源码热点。
