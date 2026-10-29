# 满缓存下的 ABCDF Plan / SparseKvTransfer 实验

分支：`fix/abcdf-plan-transfer-harness`，基于 `feat/abcdf-transfer-full-cache@fcbaa31`。
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
  --launch-mode legacy --temp-dir /dev/shm/plan_transfer_tmp \
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
  --launch-mode legacy --temp-dir /dev/shm/plan_transfer_tmp \
  --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --warmup 10 --repeat 50 --output /dev/shm/plan_transfer_formal
```

默认mode=chain，每次按同一current stream先Plan后Transfer，
两次invoke之间不人为调用synchronize。共享库接入层的tiling上传仍可能产生间隙，
它不被算进单kernel duration。

如需补充独立测试：

```bash
bash run_plan_transfer.sh 0 --modes plan transfer chain \
  --launch-mode legacy --temp-dir /dev/shm/plan_transfer_tmp \
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
python3 checks/check_transfer_harness.py
python3 -X faulthandler checks/check_plan_transfer.py --device 0 --production \
  --launch-mode prepared --temp-dir /dev/shm/plan_transfer_tmp \
  --report /dev/shm/plan_transfer_correctness/report.json
```

官方Host映射依据：
[aclrtHostRegisterV2，CANN9.1](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2128.html)、
[Host锁页内存使用说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0006.html)。
实际支持以目标950PR驱动/CANN环境及门禁结果为准。

## 7. 本轮接入修复：GM_ADDR

Transfer的10个设备地址参数在launch声明与定义中统一为GM_ADDR。
Host编译时通过ifndef GM_ADDR回退为void*；设备编译先包含kernel_operator.h，
保留其地址空间定义。调用kernel时直接传参，不用reinterpret_cast跨地址空间。
原SparseKvTransferRuntimeKernel计算类及原ABCDF文件保持不变。
用户报告该签名修复已在服务器验证；本次开发机没有编译或运行，不把历史结果当成本次验证。

## 8. legacy / prepared如何对照

### 两种路径的区别

- legacy（默认）：沿用原来两个torch.ops调用。每次Plan、Transfer接入层各自创建、
  上传tiling并查询核数；Transfer的阻塞tiling上传位于两次kernel下发之间。
  这在源码上确认是可能引入同步/Host调度等待的路径，不能凭历史P50计算它的精确贡献。
- prepared（显式选择）：每个用例创建调用方持有的C++ PreparedPlanTransfer对象，
  在采集前上传两份tiling、取得两个blockDim，绑定原Plan的15张量与Transfer的5份资源。
  chain用一次RunOpApi队列回调，在同一current stream连续下发原Plan和原Transfer。
  回调两次launch之间没有tiling申请/上传、设备值CPU读取或显式同步。
- prepared只改变Host启动方式，不改变计算kernel、query、状态恢复、golden、Host来源或cache policy。
  它仍有框架派发/运行时启动成本；实际gap是否缩小须看本次CSV，不能预先声称已消除。

上下文不使用全局缓存，资源由调用方与已排队回调持有。
每次run校验属性、TensorImpl身份、地址、尺寸、步长、dtype和device；
setup验证Host注册映射与DVA匹配，DVA表版本变化会拒绝。
内容允许恢复或更新，但替换/resize绑定缓冲后必须重新prepare。
close后拒绝run；最后释放资源时在目标调用之外等待设备，防止异步缓冲提前释放。
本实验单线程、单current stream串行使用上下文；不要并发修改其绑定数据。
采集入口只对所选启动方式过完整门禁；独立checker默认检查两路。
因此prepared运行时失败不会阻止另一次显式legacy采集，失败路径也不会静默回退。
资源清理遇到设备同步失败会告警并保留资源至进程退出，避免释放仍可能被访问的缓冲。

### 服务器操作顺序

进入新分支的源码目录并激活匹配环境，所有设备命令都使用逻辑卡0：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1

PYTHON=python3 JOBS=3 bash build.sh --soc=ascend950 --plan-transfer

# CPU辅助回归：仅服务器执行
python3 checks/check_transfer_harness.py

# 两种启动路径及prepared失效上下文门禁（还没有正式profiling）
python3 -X faulthandler checks/check_plan_transfer.py --device 0 \
  --temp-dir /dev/shm/plan_transfer_tmp \
  --report /dev/shm/plan_transfer_checks/report.json

# 相同小尺寸、seed与字节宽度，分别试跑
for launch in legacy prepared; do
  bash run_plan_transfer.sh 0 --launch-mode "$launch" \
    --temp-dir /dev/shm/plan_transfer_tmp \
    --rows 2 --topk 256 --capacity 512 --max-token 8192 \
    --seed 2026 --warmup 2 --repeat 3 --verify-each-sample \
    --output "/dev/shm/plan_transfer_smoke_$launch"
done

# 同一生产尺寸/seed/预热次数/采集次数，顺序完成正式对照
for launch in legacy prepared; do
  bash run_plan_transfer.sh 0 --launch-mode "$launch" \
    --temp-dir /dev/shm/plan_transfer_tmp \
    --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
    --seed 2026 --warmup 10 --repeat 50 \
    --output "/dev/shm/plan_transfer_formal_$launch"
done
```

两次采集是独立进程，虚拟地址可能不同；逻辑输入、payload生成规则与预热条件一致。
报告记录launch_mode、case_sha256、库身份及实际字节数。对照前核对两次输入指纹相同。
不要同时在同一卡运行两个采集；有条件时交换legacy/prepared执行先后再做复测。

## 9. 目录门禁、日志与失败判断

运行正式采集前（早于import torch_npu、NPU初始化和Host大池申请）：

1. 检查实际输出目录及临时目录，解析符号链接后输出真实路径。
2. 检查可写性、普通用户可用字节数和可用inode。
3. 同一文件系统上，按两项字节最低余量之和检查；不重复承诺同一份空间。
4. 设置TMPDIR并覆盖tempfile缓存路径，后续解析子进程继承该配置。
5. Host池的9GiB是内存需求，profiling文件的磁盘余量另列。/dev/shm也会消耗主机内存，
   需要同时容纳Host池、框架内存和采集文件。

默认门限：输出1GiB、临时1GiB，各需1024个可用inode。
门限只用于早期拒绝明显不足的环境，不能精确预测本次profiling文件大小，
也不能保证采集期间其他进程不占空间。可以显式覆盖：

```bash
bash run_plan_transfer.sh 0 --launch-mode prepared \
  --output /dev/shm/plan_transfer_capture --temp-dir /dev/shm/plan_transfer_tmp \
  --min-output-free-bytes 1073741824 --min-temp-free-bytes 1073741824 \
  --min-free-inodes 1024
```

只检查输出/临时文件系统，不因源码分区满而拒绝读取源码。
不自动删除用户文件，也不静默切换位置；失败时需手动释放空间或显式改路径。
程序只清理自己创建的空可写性探针。/dev/shm属于临时存储，测试后及时转存完整session。

每轮会打印session路径和profile.log路径。详细日志捕获Python、原生库以及继承stdio的解析子进程输出。
缺CSV或解析失败时，同时输出采集目录、详细日志路径和日志中首个可定位异常；
原始采集文件保留。部分子进程可能另写日志，诊断还会扫描采集目录中的log/err文件。
子进程重定向到其他位置的日志需按其错误提示继续查找。

失败写capture_metadata.json采用best effort；再次遇到ENOSPC只告警并保留原始异常链，
未完成的原子写入可能留下pending文件，不会删除已有采集文件。
只有同时满足以下条件才可当作成功：

- 本轮退出码为0；
- capture_metadata.json的status是complete；
- session根目录存在最后写入的COMPLETE.json；
- report.md、原始CSV和各用例summary.json均在同一session中。

即使满盘导致旧元数据未能从complete改成failed，缺失COMPLETE.json也必须判未完成。
成功报告在所有状态/标记写入完成后才发布为report.md；发布前只会有partial/pending文件。
如果最后发布失败，即使已写标记，缺失report.md也判未完成。
部分用例的passed状态不能代表整轮通过。

report.md和stage_summary.csv都展示Plan、Transfer、pair_busy、pair_span、
between_kernel_gap的P50/P95及样本数。
busy与gap均逐次配对计算后再统计，禁止用P50之和/差代替。
span包含两kernel间隙；单kernel duration、busy、span均不是公共Python接口端到端延时。
CSV继续严格验证目标名、μs单位、样本数、调用顺序和不重叠；
导出stream/device列时也校验其一致性，恢复/TensorMove不混入单kernel duration。
