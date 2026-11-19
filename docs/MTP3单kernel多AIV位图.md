# MTP3 位图的单 kernel 多 AIV 实验

沿用四路、TopK2048、容量8192、512 SIMT 线程及 187936 字节动态 UB。排序、因果可见性、重复 resident 的 LRU-last owner、miss 描述符和完整写回尾部沿用父版本。475K 的 token 编号范围为 0 至 486399。

| 版本 | 执行方式 | 跨 AIV 临时数据 |
|---|---|---|
| `joint_mtp3_bitmap475_512` | 单 AIV 原位图，保留旧 GM query→UID 暂存 | 无 |
| `joint_mtp3_bitmap_ubmap_512` | 单 AIV 位图。复用已结束使用的归并工作区存放 LRU flags，query→UID 一直留在 UB | 无 |
| `joint_mtp3_bt_share_512` | 两个 AIV。主核硬件归并后发布 unique，辅助核建位图。各查 4096 个物理槽 | 按实际 unique 记录数对齐的 32 位列表、32 字节头、8 KiB 辅助核结果 |
| `joint_mtp3_bt_direct_512` | 两个 AIV。辅助核直接从四行 TopK 构建本地精确位图和 UID 前缀，和主核归并并行。各查 4096 个物理槽 | 8 KiB 辅助核结果 |
| `joint_mtp3_bt_direct_odd_512` | DIRECT 的访问步长对照。辅助核每个 SIMT 线程负责的连续位图 word 数向上取奇数 | 8 KiB 辅助核结果 |

主 AIV 持有完整 query→UID、unique、原 resident 和 LRU 分配所需的临时状态，统一发布新状态。同一个 token 的多个 resident 槽均被保护。辅助 AIV 只查询，结果每两个槽打包成一个 int32。DIRECT 按因果可见性收集 token，位图前缀给出的 UID 和主核升序去重后的 UID 一致。

交接使用 MTE 搬运和硬件全 AIV 同步。SHARE 使用两次同步，DIRECT 使用一次。冷组跳过查槽和结果交接，仍参与同期集合同步。只有满激活且连续四行分组、实际可用 AIV 至少为请求数两倍时才启用双 AIV。其他合法形状使用同 kernel 的单 AIV 流程。56 AIV 设备的双 AIV 上限为 28 个请求。当前测试为单流下发。

## 构建、校验和采集

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.2.1
export PYTHON=/usr/local/python3.12.16/bin/python3 JOBS=6
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
bash build_joint_plan.sh --soc=ascend950 --mtp3-only
RUN=$(mktemp -d /workspace/mtp3_bitmap_team_XXXXXX)
"$PYTHON" -B checks/check_mtp3_bitmap_team_npu.py --device 4 --report "$RUN/gate.json" \
  --implementations joint_mtp3_bitmap475_512 joint_mtp3_bitmap_ubmap_512 \
  joint_mtp3_bt_share_512 joint_mtp3_bt_direct_512
DEVICE=4 OUTPUT="$RUN/perf" bash scripts/test_mtp3_bitmap_team_profile.sh
```

设备编号按空闲卡调整。测试开跑前校验源码身份。计时来源是设备侧 kernel_details.csv 中精确匹配的 Plan Duration。

单独复测 DIRECT 的位图访问步长时使用当前分支的设备库。

```bash
"$PYTHON" -B profile_mtp3_lookup.py --device 4 --output "$RUN/odd" \
  --temp-dir "$RUN/perf/tmp" --contexts 102400 262144 358400 486400 \
  --requests 8 --warmup 5 --repeat 20 --scenarios full_cache_75pct \
  --implementations joint_mtp3_bitmap_ubmap_512 joint_mtp3_bt_direct_512 \
  joint_mtp3_bt_direct_odd_512
```
