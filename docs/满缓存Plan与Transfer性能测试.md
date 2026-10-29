# 满缓存下的 ABCDF Plan / SparseKvTransfer 实验

分支：`feat/abcdf-transfer-full-cache`，基于 `feat/abcdf-mtp-scan@85ebbb6`。
Plan固定为原ABCDF（1024线程）；新接入生产工程中的SparseKvTransfer。
本轮仅完成源码和静态检查，未在开发机执行Python、编译、CPU检查或NPU测试。
下面所有运行命令均交给950PR服务器，不代表已验证成功。

## 1. 对比的问题

在缓存所有capacity槽均有效、request ID相同、无投机后缀失效时，
分别构造目标命中率75%、90%、100%，测量一层的一次调用：

| 目标命中率 | T2048每行hit | 每行miss | 实际命中率 | 32行搬运量（1024+128B/token） |
|---|---:|---:|---:|---:|
| 75% | 1536 | 512 | 75% | 18 MiB |
| 90% | 1843 | 205 | 89.990234375% | 7,557,120B，约7.21 MiB |
| 100% | 2048 | 0 | 100% | 0 |

90%采用 `floor(topk*90/100)` 个hit，报告同时记录实际命中率。
100%仍真实下发Transfer，测量零miss时的元数据/调度控制开销，不把未调用写成0μs。

本实验MTP关闭：一行一个请求，`stable=visible=max_token`。
与旧MTP输入分开，避免把新增可见token混入“100%命中”场景。
旧MTP/Scan入口仍保留，不作为本次Plan/Transfer采集入口。

## 2. 你的GLM模型为什么是1024+128字节

用户提供配置：glm_moe_dsa、BF16、kv_lora_rank512、qk_rope_head_dim64、
64个attention heads、78层。代码中的offload分配使用：

- 主缓存头数为1，主缓存必须BF16。
- K缓冲存压缩KV，维度为kv_lora_rank512：512×2=1024B/token。
- 名为V的缓冲实际存RoPE K，维度为qk_rope_head_dim64：64×2=128B/token。
- 不乘64个attention heads，也不使用v_head_dim256来计算这个压缩缓存的搬运量。
- indexer缓存不由本Transfer搬运；这里统计一层，不能直接视为78层端到端耗时。

源码依据在生产工程的
`vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py`：
`reshape_kv_cache_tensors_for_sparse_kv_offload`与`register_kv_caches`；
后者明确校验缓存head数为1且dtype为BF16。
实验默认预设名为 `glm_moe_dsa_bf16`，元数据写出该假设。
如实际服务器换了缓存布局/量化/stride，请显式给出字节参数，不从模型头数盲目推算。

## 3. 控制了哪些变量

`full_cache_cases.py`：

1. 同一seed、shape下，三档共用query、页表、初始LRU以及query对应的槽位置。
2. 所有capacity槽先填满合法、不重复的非query token，再替换指定的hit token。
3. hit集合嵌套；75%已有的hit在90%/100%仍位于同一槽。
4. 每请求有独立物理页，默认将所有物理页随机打乱，不让32请求共享同一Host页池。
5. query合法且不重复；ID不变，无reset；stable覆盖全部token，初始没有空槽。
6. 所有miss均需淘汰历史token。三个工况改变的是命中集合，不改变K/V字节宽度。

每次目标调用前恢复初始Plan持久状态及全部resident payload，
恢复和同步在目标调用外。不能只恢复LRU而让miss目标保留上一轮已搬好的答案。

## 4. 真正的Transfer与Host映射

计算来源：
`vllmascend_bcde_fused_1024/csrc/attention/sparse_kv_transfer/op_kernel/sparse_kv_transfer_apt.cpp`。

原文件保存在 `csrc/transfer/upstream/`。
`csrc/transfer/kernel.cpp` 中的SparseKvTransferRuntimeKernel计算类逐字保留，
只适配直调入口、POD tiling逐字段读取和launch；显式保留两块16KiB UB buffer。
AIV核数沿用min(64,平台AIV数)，未加入SIMT专属编译选项。

实际数据流：

```text
ABCDF
  → miss_count / miss_tokens / miss_slots
  → SparseKvTransfer
      → token_to_req → block_table → Host物理页
      → Host K/V（经注册得到的DVA）
      → resident[row,slot] K/V
```

Host内存由 `aclrtMallocHost` 分配，再按4KiB对齐地址调用
`aclrtHostRegisterV2(..., ACL_HOST_REG_MAPPED)`，
通过 `aclrtHostGetDevicePointer` 获取真正DVA；不假定HVA等于DVA。
注册或映射失败直接失败，不回退到普通CPU指针、HBM源或Python拷贝。
CPU owner一直保留到NPU同步完成，之后解除注册并释放。

当前默认是一层、32请求、每请求262144个token：
Host K池8GiB、Host RoPE池1GiB；resident合计144MiB，
另有同等大小的NPU恢复快照及框架开销。
这属于真实Host池容量；降低byte参数会同时改变实验工作量。
可以先运行describe-only查看预算，再在服务器构建运行。

payload以物理token身份和字节位置生成，K/V各不相同，使用原始uint8存储做逐字节检查；
不使用全零源数据，避免错源地址也得到相同答案。BF16预设指定的是实际stride，
此实验不执行Attention浮点计算。

## 5. 服务器操作

### 构建：只编译ABCDF和Transfer

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0

PYTHON=python3 JOBS=3 bash build.sh --soc=ascend950 --plan-transfer
```

库为 `build_plan_transfer/libsimt_plan_variants_ops.so`。
不用 `--plan-transfer` 时保留原18个Plan版本的构建，并额外编译Transfer；
此时库在build/，新入口需显式传 `--library build/libsimt_plan_variants_ops.so`。
源码、上游计算类、Plan catalog及编译选项记录进库身份；过期库会拒绝运行。

### 先查看用例和内存需求（CPU，仅服务器运行）

```bash
python3 profile_plan_transfer.py --describe-only
```

### 小尺寸试跑

```bash
bash run_plan_transfer.sh 0 \
  --rows 2 --topk 256 --capacity 512 --max-token 8192 \
  --warmup 2 --repeat 3 --verify-each-sample \
  --output /dev/shm/plan_transfer_smoke
```

默认1024/128字节预设不变，只缩小batch和token域。
先跑Host可访问性、边界、chunk和payload检查，再采集三档。
任何校验/解析/样本数错误都会报错，不输出complete报告。

### 生产尺寸正式采集

```bash
bash run_plan_transfer.sh 0 \
  --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --warmup 10 --repeat 50 --output /dev/shm/plan_transfer_formal
```

默认mode=chain，每次按同一current stream先Plan后Transfer，
两次invoke之间不人为调用synchronize。共享库接入层的tiling上传仍可能产生间隙，
它不被算进单kernel duration。

如需补充独立测试：

```bash
bash run_plan_transfer.sh 0 --modes plan transfer chain \
  --warmup 10 --repeat 50 --output /dev/shm/plan_transfer_three_modes
```

Transfer-only的miss输入也由真实ABCDF在计时前生成，不从golden写回设备。
plan-only不执行Transfer；chain验证真实Plan输出被后续Transfer使用。
所有设备命令明确使用逻辑卡0；同卡采集串行运行。

### 显式覆盖缓存stride

```bash
bash run_plan_transfer.sh 0 --payload-preset explicit \
  --token-bytes-k 1024 --token-bytes-v 128 \
  --output /dev/shm/plan_transfer_explicit
```

两项必须同时给出。该例与模型预设字节宽度相同；换其他数值时元数据标记explicit，
不再把它标成GLM生产配置。

## 6. 正确性门禁与计时口径

每次采集前：

- 小尺寸验证Plan/Transfer/串联三种模式，以及1/31/32/33字节、16KiB边界及16KiB+1分块。
- 单独Transfer覆盖负/超大miss_count、非法token/slot/request/physical page、
  request重映射、active_rows边界和不均匀各行miss数。
- 生产用例先与独立列表golden逐字段比较Plan的3个持久状态和4个输出。
- Transfer独立CPU标量地址模型从初始payload出发，只覆盖合法miss目标；
  正常输入再与Plan最终resident交叉核对。
- 校验整个resident payload，包括未写槽以及两端保护区；每组3轮恢复审计。
- 预热后、采集后仍检查；`--verify-each-sample`可对每个采集step再逐字段/逐字节检查。

profile沿用现有torch_npu.profiler：手动预热warmup次，采集器额外warmup1 step，
active=repeat，每个step调用当前mode一次。
CSV只匹配下面的目标kernel，TensorMove、状态恢复及其他任务不混入其duration：

```text
ABCDF:   sparse_kv_plan_kernel_plan_abcdf
Transfer: sparse_kv_transfer_kernel
```

每种目标恰好repeat条、每个目标只有一个kernel/task-type分组才接受。
chain必须有带μs单位的start/duration，按开始时间排序验证Plan→Transfer顺序及不重叠。
如果本机CANN导出的列不同，会明确报错并留下原始CSV；不能猜单位或按相近名字混计。

报告字段：

| 字段 | 含义 |
|---|---|
| plan / transfer | 单个kernel的P50/P95/min/max/原始样本 |
| pair_busy | 每次真实配对的两条duration求和，再统计P50/P95 |
| pair_span | 从Plan开始到Transfer结束的设备时间跨度，包含中间间隙 |
| between_kernel_gap | 两kernel之间的时间间隙 |
| effective_payload_GBps | 搬运字节数/Transfer时间，仅为观测有效payload吞吐 |

两个单kernel P50不能直接相加冒充pair P50。
pair_span也不是完整Python入口端到端延时：它不包含首kernel之前的Host开销。

## 7. 输出和结论边界

输出在每次唯一的session目录：

```text
session_*/capture_metadata.json
session_*/report.md
session_*/full_cache_hit75/chain/summary.json
session_*/full_cache_hit75/chain/stage_summary.csv
session_*/full_cache_hit75/chain/**/kernel_details.csv
# hit90 / hit100、其他mode同样保留
```

元数据含库/源码身份、预设、实际miss与字节数、设备、编译选项、场景顺序和原始任务样本。
失败标记failed，部分结果不能视为整轮通过；成功才生成report.md。
没有实测数据时，不预填延时或预测加速比。

**硬件cache条件**：重复使用固定地址和输入，且有预热、恢复过程。
这测的是该恢复流程下的稳态kernel延时；“逻辑KV缓存满”不等于硬件L2冷。
即使源是Host映射内存，也可能有硬件cache命中，所以有效payload吞吐不等于冷Host DRAM带宽。
若要定位链路瓶颈，后续需结合L2/访存指标与真实生产访问轨迹分析。
本轮不修改Transfer cache policy，也不添加设备侧clock打点。

辅助检查（交给服务器，开发机未执行）：

```bash
python3 checks/check_full_cache_inputs.py
python3 -X faulthandler checks/check_plan_transfer.py --device 0 --production
```

官方Host映射依据：
[aclrtHostRegisterV2，CANN9.1](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2128.html)、
[Host锁页内存使用说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0006.html)。
实际支持以目标950PR驱动/CANN环境及门禁结果为准。
