# Transfer A/B/C/D 组合优化实验

分支：`feat/transfer-factorial-optimizations`，基于 `fix/abcdf-plan-transfer-harness@e610c61`。
本机仅修改源码与静态检查，未执行Python、编译、CPU测试或NPU测试。
所有新版本都必须先在950PR服务器通过门禁，再看性能；当前没有实测胜者。

## 1. 四个因子与组合

| 因子 | 独立改动 | 要验证的假设 |
|---|---|---|
| A | K/V在UB中分区，共用一个queue记录 | 减少Alloc/EnQue/DeQue/Free与同步管理 |
| B | 每批组织8个有效token | 减少小任务的队列批次开销，增大可提前发射的工作量 |
| C | block_size64、K1024B、V128B的常量特化；其他属性保留通用实现 | 减少运行时除法、尺寸与地址运算 |
| D | 全部行的miss拼成全局序列，按连续区间分给各核 | 改善描述符访问局部性；分配的描述符数量差至多1 |

完整目录：`transfer_variants.json`。

- `baseline`：原生产Transfer kernel，源码计算类保持不变。
- `control`：新模板的0000版本，四个因子都关闭。用它区分重构本身的开销。
- 单点：a / b / c / d。
- 双点：ab / ac / ad / bc / bd / cd。
- 三点：abc / abd / acd / bcd。
- 四点：abcd。

合计17路，其中16路是完整2^4因子实验。组合由同一计算模板的编译期位控制，
每路有唯一kernel名，不通过运行时改算法来冒充多个独立测量版本。

### A和B不是同一个优化

A决定K/V是否共用一个队列记录；B决定一个记录组织多少个token。
因此b是“8个K一组、8个V另一组”，ab是“8个token的K/V合为一组”。
a单独存在时只合并一个token的K/V，b单独存在时仍分开管理K/V。

源token和目标slot可任意散布，A/B并不把它们误当成连续GM地址。
通常仍需每token两次搬入、两次搬出，减少的首先是队列/事件管理次数。
批次也可能增加整组等待，A/B或其组合不保证更快。

### D仍保持任务数均衡

旧版以全局miss序号轮转分核；D将同一全局序列分成连续区间。
先对miss_count做与原版相同的负数/上限截断，再分配区间。
D多了一次count扫描，是显式成本，不预先认定有收益。
非法request仍占据其原本的全局描述符位置，随后被跳过，避免改变后续前缀语义。
非法项跳过后，各核真正搬运的有效项数量不保证均衡。
目标槽必须满足Plan正常输出的一项一写约束；重复目标槽跨核写的竞争不被本实验定义为新语义。

## 2. UB预算和回退

仍预留2×16KiB动态UB，与原版一致。K/V各段起点按32B对齐。

生产配置中，ab类每个buffer需：

```text
8 × (1024 + 128) = 9216 bytes ≤ 16384 bytes
```

a：一个token的两份数据共用记录；b：每份数据各自按8个token分组。
不足8项的尾批只搬合法项，不读取未初始化地址。

若请求分组装不进16KiB单buffer，明确回退原计算类的16KiB分块路径。
回退计算类逐字保存在`csrc/transfer_variants/original_class.h`，
构建身份检查会与保留的上游类比较。
这不是隐藏加速路径：报告含requested_batch、effective_path、fallback和C特化是否实际启用。
非生产字节宽度下C可能走通用路径；不能把这种结果当成已测试生产特化的收益。

## 3. 公平对照和采集口径

三档保持原实验语义：

| full cache工况 | 每行hit / TopK2048 | 每行miss |
|---|---:|---:|
| 75% | 1536 | 512 |
| 约90% | 1843 | 205 |
| 100% | 2048 | 0 |

所有capacity4096槽均有效、请求ID匹配、无投机后缀失效、MTP关闭。
相同seed下query、页表、LRU共用，命中集合嵌套。

矩阵入口`profile_transfer_variants.py`：

1. 先通过所选版本的小尺寸、异常描述符、非对齐、大token回退和真满批/尾批检查。
2. 只申请一次真实Host映射池，所有版本共用同一Host源地址。
3. 每个工况只准备一次输入、输出、resident payload和恢复快照。
   切版本只关闭/重建准备上下文，保持这些数据缓冲地址不变。
4. 每次调用恢复Plan初始状态和resident payload；Transfer-only先由真实Plan生成描述符。
5. 初始化、恢复、Plan prime、正确性检查均在Transfer计时外。
6. 默认prepared、transfer-only；可用--modes chain补看真实Plan→Transfer串联。
7. 每配置手动预热10次、采集器额外预热1 step，正式记录50次。
8. 每个配置分块采集；工况间轮换版本顺序，并记录完整顺序。
   --reverse-order支持整轮反序复测，不声称逐样本交错测量。
9. 严格按当前版本的唯一kernel名、单位和样本数解析CSV，原始样本保留。
10. 仅同输入指纹、同数据缓冲地址指纹、同实际核数、同工作量及启动方式之间计算baseline比值。

同一次工况下输入与数据缓冲地址完全复用；不同工况/不同独立进程之间的地址可能不同。
所有统计继续使用kernel duration；chain的busy/span/gap逐对计算。
100%仍下发Transfer，不报告虚构的零kernel时间。
不改变源内存类型、缓存策略，不增加L2 flush、冷缓存或新的访问轨迹。

## 4. 关于“90多GB/s”

报告的effective_payload_GBps定义为：

```text
需搬运的K+V有效字节 / Transfer kernel时间 / 10^9
```

同一份payload只计一次。如果之前的90多GB/s按“读字节+写字节”累计，
它可能对应本口径约45GB/s；当前46GB/s有效payload吞吐按读写累计约92GB/s。
需核对同卡、同Host路径、粒度、NUMA、预热/cache条件和计数字节定义，
再判断距离参考值的差距。90GB/s不是本轮硬编码的通过阈值，也不预填预计加速比。

## 5. 服务器运行

以下命令只交给950PR服务器，使用逻辑卡0。源码开发机没有执行。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
ulimit -c 0

PYTHON=python3 JOBS=3 bash build.sh --soc=ascend950 --plan-transfer

# CPU辅助模型与原框架IO/CSV回归
python3 checks/check_transfer_factorial_models.py
python3 checks/check_transfer_harness.py

# 可先挑少数版本定位编译/运行问题；不带--variants默认全部17路
python3 checks/check_transfer_variants.py --device 0 \
  --variants baseline control a b c d ab abcd \
  --temp-dir /dev/shm/transfer_tmp \
  --report /dev/shm/transfer_gate/report.json
```

### 小尺寸全矩阵

```bash
bash run_transfer_variants.sh 0 \
  --rows 2 --topk 256 --capacity 512 --max-token 8192 \
  --warmup 2 --repeat 3 --verify-each-sample \
  --temp-dir /dev/shm/transfer_tmp --output /dev/shm/transfer_matrix_smoke
```

### 生产尺寸全矩阵

```bash
bash run_transfer_variants.sh 0 \
  --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --warmup 10 --repeat 50 --core-limits 64 \
  --temp-dir /dev/shm/transfer_tmp --output /dev/shm/transfer_matrix_formal
```

默认17版本×3工况，每配置50个Transfer样本，共2550个正式Transfer目标样本。
Host源池仍为9GiB；resident144MiB及其快照按工况复用，不为17个版本分别占用17份。
门禁可能花时间，但不计入kernel duration。

### 择优后扫核数、补串联

```bash
# 这里只是示例版本列表；用你全矩阵实际选出的版本替换
bash run_transfer_variants.sh 0 --variants baseline ab abc bcd abcd \
  --core-limits 16 32 64 --modes transfer chain \
  --warmup 10 --repeat 50 --temp-dir /dev/shm/transfer_tmp \
  --output /dev/shm/transfer_core_sweep
```

实际核数是min(请求上限, 原平台AIV核数上限64)，报告记录请求与实际值；
同一版本被平台裁剪成相同实际核数时会去重。
另一轮可加--reverse-order检查顺序漂移；不要并行占用同一卡。

成功时session目录包含：

- comparison.md：Transfer排名、P50/P95、样本数、payload吞吐和baseline比值。
- report.md：每路各stage的P50/P95/count；chain时包括Plan/busy/span/gap。
- capture_metadata.json、COMPLETE.json：整体状态、源码/编译身份、配置与顺序。
- 每配置summary.json、stage_summary.csv和原始kernel_details.csv。
- profile.log：Python、原生与解析子进程日志。

失败保留原始文件，不发布成功report.md。只有退出码0、complete元数据、
COMPLETE.json与report.md同时存在才算整轮完成。/dev/shm是临时存储，及时转存。

## 6. 对单个Transfer抓msopprof

`run_transfer_msopprof.py`内部不创建torch_npu.profiler。
它先让真实Plan生成miss，目标范围只下发Transfer，恢复和校验在范围外。
--warmup 0时第一个匹配Transfer kernel就是随后进行完整结果校验的目标调用，
没有额外同名Transfer门禁调用抢占BasicInfo的第一条。

先单独运行，确认目标入口正确：

```bash
python3 -X faulthandler run_transfer_msopprof.py --device 0 \
  --transfer-variant abcd --hit-percent 75 --warmup 0 --repeat 1 \
  --temp-dir /dev/shm/transfer_tmp --output /dev/shm/transfer_runner_check
```

外层msprof进程的TMPDIR也要在启动前设置；子进程不能反过来修改父进程环境。
以下沿用CANN9.1的application replay语法，实际支持项以服务器--help为准：

```bash
mkdir -p /dev/shm/transfer_tmp
export TMPDIR=/dev/shm/transfer_tmp
msprof op --help > /dev/shm/transfer_msopprof_help.txt

# BasicInfo单次预检，先确认成功产物再采其他模块
msprof op --aic-metrics=BasicInfo --replay-mode=application --warm-up=0 \
  --kernel-name=sparse_kv_transfer_kernel_opt_abcd --launch-count=1 \
  --output=/dev/shm/transfer_basic_abcd \
  python3 run_transfer_msopprof.py --device 0 --transfer-variant abcd \
  --hit-percent 75 --warmup 0 --repeat 1 \
  --temp-dir /dev/shm/transfer_tmp --output /dev/shm/transfer_runner_basic

# Default + MSTX，仅目标范围；不要套在已有torch_npu.profiler入口外
msprof op --aic-metrics=Default --replay-mode=application --warm-up=0 \
  --mstx=on --mstx-include=transfer_target \
  --kernel-name=sparse_kv_transfer_kernel_opt_abcd --launch-count=1 \
  --output=/dev/shm/transfer_default_abcd \
  python3 run_transfer_msopprof.py --device 0 --transfer-variant abcd \
  --hit-percent 75 --warmup 10 --repeat 1 --mstx \
  --temp-dir /dev/shm/transfer_tmp --output /dev/shm/transfer_runner_default
```

原版kernel名是sparse_kv_transfer_kernel；其他组合是sparse_kv_transfer_kernel_opt_版本名。
工具可能按前缀匹配，因此深度入口每进程只运行一种Transfer版本；结果仍需核对精确目标名。
如需Source定位，额外构建--debug-info并显式指定build_plan_transfer_profile/下的库；
保持-O3，不把调试版本与普通编译计时混在同一排名中。
工具若提前终止应用，runner可能没有passed报告：必须保留外部日志，并以前面的独立门禁确认功能。
Default/MemoryDetail等实际采集量与硬件计数支持以950/CANN9.1工具输出为准，
缺失指标不当作0，不把深度采集时长混入普通50样本排名。

官方参数依据：
[msopprof CANN9.1](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md)。
