# MTP3 四行联合 Plan 查询对照

固定 TopK2048、每请求四行、共享缓存容量 8192、512 个 SIMT 线程。实验使用合成的升序 TopK。100K、256K、350K、475K 分别对应 maxToken 102400、262144、358400、486400。

| 实现 | 请求内 AIV 分工 | 查槽方式 |
|---|---|---|
| `joint_mtp3_full_512` | 单 AIV 完成完整 Plan | 原冻结版。仅用在 256K 参考对照 |
| `joint_mtp3_dir475_512` | 单 AIV 完成完整 Plan | 1024 桶目录＋桶内精确二分 |
| `joint_mtp3_bitmap475_512` | 单 AIV 完成完整 Plan | UB 精确位图＋word 起始 UID |
| `joint_mtp3_lookup1_512` | 单 AIV 查 8192 个物理槽，另外两个阶段各单 AIV | 同一目录＋精确二分 |
| `joint_mtp3_lookup2_512` | 将查槽切成 2 段，每段 4096 槽 | 同上 |
| `joint_mtp3_lookup4_512` | 将查槽切成 4 段，每段 2048 槽 | 同上 |

多 AIV 版本通过同流的三个设备 kernel 交接。准备阶段沿用四路硬件归并，保存有序 unique、query 到 UID 的映射及目录。查槽阶段只写物理槽对应的临时状态。提交阶段按照完整 LRU 选 owner 和 victim，再更新缓存及 miss 输出。冷启动仍下发查槽 kernel，它读取状态后短路。三段式 lookup1 用来检验交接成本。lookup2/4 与它对比可衡量只增加查槽 AIV 的变化。

各段最多下发设备查询到的 64 个 AIV。实际硬件只有 56 个时按 56 划分。准备和提交每请求仍只有一个工作单元，查槽可提供 batch×1/2/4 个任务。任务多于设备 AIV 时由同一个核继续处理后续分段。两种单 kernel 查询对照使用相同的 187936 字节动态 UB。三段式另为每请求预留 13328 个 int32 的 GM 暂存，查槽 kernel 的动态 UB 为 36896 字节。

## 在 CANN 9.2.1 机器运行

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.2.1
export PYTHON=/usr/local/python3.12.16/bin/python3 JOBS=6
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
bash build_joint_plan.sh --soc=ascend950 --mtp3-only
RUN=$(mktemp -d /workspace/mtp3_lookup_XXXXXX)
"$PYTHON" -B profile_mtp3_lookup.py --device 4 \
  --output "$RUN/475k" --temp-dir "$RUN/tmp" --contexts 486400 \
  --requests 1 4 8 9 10 16 28 32 --repeat 15 --warmup 5
"$PYTHON" -B profile_mtp3_lookup.py --device 4 \
  --output "$RUN/other-contexts" --temp-dir "$RUN/tmp" \
  --contexts 102400 262144 358400 --requests 8 16 \
  --scenarios full_cache_75pct --repeat 15 --warmup 5
```

替换为当前空闲的 device。采集严格匹配目标 kernel、样本数和顺序。单 kernel 统计 Duration，三段式按调用逐次统计首段开始到末段结束的 span，含设备侧间隙。各段的 P50 单独报告。采集前恢复初态，每个样本校验输入、设备保护区、持久状态以及完整输出。报告的 `full_cache_75pct` 标签基于首行构造其缓存命中条件。结果限于 Plan，输入恢复、Indexer、Transfer 和接口端到端需要独立计量。
