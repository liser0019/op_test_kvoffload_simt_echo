# MTP7：缓冲复用与分段归并一轮测试

本轮基于 `25ecfff1c4897a0a72839af7080c50fb9f65b130`，将此前已写的A/B与两项新优化放在同一轮比较。**不修改兼容回退，不扩大线程矩阵，不测试Transfer。新增候选没有在本机执行CANN编译或NPU验证，性能收益待服务器实测。**

## 1. 固定八路，区分每项改动

以下联合路径均为512线程，使用同样的生产动态UB预算181024字节。ABCDF保持原独立设备入口及配置。

| 实现 | ID | 父版本与改动 |
|---|---:|---|
| `abcdf_pages` | 独立入口 | 带页表ABCDF，冻结比较基线 |
| `joint_hot_bd8_512` | 193 | 服务器已测过的BD8：8线程重复扫描消费统计，保留慢路径作对照 |
| `joint_mtp7_parallel_512` | 194 | A：512线程一次遍历统计来源，保持三次硬件归并 |
| `joint_mtp7_rank_512` | 195 | B：并行rank-scatter替换最后两路硬件归并，同时完成来源统计 |
| `joint_mtp7_reuse_parallel_512` | 196 | A＋缓冲复用，仍使用最后两路硬件归并 |
| `joint_mtp7_reuse_rank_512` | 197 | B＋缓冲复用，仍逐记录二分求rank |
| `joint_mtp7_tile_512` | 198 | B＋分段归并，保持原512条packet |
| `joint_mtp7_reuse_tile_512` | 199 | B＋缓冲复用＋分段归并 |

A/B的细节见[此前说明](MTP7并行消费统计与归并优化.md)。它们已经写入源码，但目前同样没有新硬件结果，不应当作已验证的快速基线。ID193的已知硬件结果属于用户在c956f96上的测试，本轮重新构建仍需同场比较。

本轮仅为八行专有流程增加两种新改动。MTP0/1/3保留原算法，但重新编译可能改变生成代码或资源，所以一并复测。兼容回退的请求组调度、1024线程逐行实现、额外协议发布均保持原样；没有改成独立ABCDF启动。

## 2. 缓冲复用：同样UB装更多记录

八行输入先分为两个四行半组，各做一次硬件归并，得到两个中间有序前缀，再合成最终前缀。每条记录包含score/payload两个32位字段。

原布局以 `C=每路packet条数` 为单位，共需 `48*C` 个int32：

```text
[0,16C)    原始八路输入
[16C,24C)  中间结果A
[24C,32C)  中间结果B
[32C,48C)  最终输出
```

第一半组归并完成并等待Vector完成之后，其原输入 `[0,8C)` 已经不用再读取，可以让第二半组输出复用它：

```text
[0,8C)     原输入A → 中间结果B
[8C,16C)   原输入B → 两半组都完成后可作计数临时区
[16C,24C)  中间结果A
[24C,40C)  最终输出
```

因此同一预算只需 `40*C` 个int32，生产形状每路可从512条增加到612条。这里没有增加GM临时区，也没有再申请更大UB。

关键生命周期：

| 交接 | 保留的保护 |
|---|---|
| 中间A完成后复用原输入A | 保留现有Vector→Scalar完成等待；A的输出位于16C，与复用区不重叠 |
| 中间B完成后执行融合归并 | 两个中间结果均已就绪；最终输出24C不覆盖任何仍在读取的中间结果 |
| 归并VF内写warp局部计数 | 融合版本使用已失效的8C区域，避开仍在读取的中间B；保留原block同步 |
| SIMD去重、映射与后续阶段 | 保留原控制交接、容量检查和状态提交顺序 |

生产工作区为24576个int32，40×612=24480。还检查了末尾SIMD的64记录整块地址范围：最大4896条输出补齐到4928条后，`24×612+2×4928=24544≤24576`。这同时约束无效lane可能形成的地址，不能仅靠mask假设安全。各记录起点保持32字节对齐，四路输入长度上限2448，小于MrgSort单路4095限制。

UB被SIMD与SIMT共享并复用时仍需遵守流水依赖；没有因“同一地址”而删掉完成等待。参见[官方SIMD/SIMT存储层次说明](https://asc.gitcode.com/guide/programming_guide/advanced_programming/advanced_ai_core_programming_model/simd_simt_hybrid_programming/memory_hierarchy.html)。CPU检查不能替代真实地址空间和流水验证。

## 3. 分段归并：少做二分，保留高并行

B原来为每个记录在另一半组二分查询rank，再散写。新版本使用merge-path：

```text
每线程领取8个连续输出位置
→ 对这段起点做一次二分，确定A/B各消费多少
→ 从两个起点向前比较，连续输出最多8条
→ 同时统计记录的原始来源行
→ 延用warp归约和block同步，推进八行游标
```

不是让一个线程串行处理整段大数组。每线程以 `512×8` 为步长继续领小段；尾段线程即使没有记录也参与归约。相同token保持A半组在B半组之前，段内稳定顺序与段间边界一致。

两半组只输出安全前缀：未完整半组的尾token限制本轮cut，完整半组不限制。只按最终真正输出的记录推进原始八行，不按中间MrgSort消费数量提前推进。跨packet重复由原lastToken继续去重；每个原query仍写入正确UID，后续逐位置currentSlots、resident保护、LRU、miss、线程1模式发布和溢出回退均保留。

## 4. 本机实际检查

- 新196～199各133次完整控制流程CPU逐字段对照，共532次；另有8次生产流程执行用于归并调用计数。核对冻结193，并通过原独立语义校验；包含生产尺寸、重复token、部分空队列、因果可见性、容量溢出及10步连续状态。
- 750项来源计数/安全前缀/归并原语检查，覆盖8输出分段边界、相同值、不同半组完整标志、保护区及未写区域；使用G++与UBSan。
- 84项框架回归通过；14种生成目录的CPU分派编译/链接检查通过。
- 保留193/194/195预处理计算体的token指纹；原ABCDF、旧BD、RunPrivateRow及联合kernel外壳源码指纹不变。Python/bash语法与工程身份检查通过。
- CPU计数模型中，一个生产请求原193/A/B分别为15/15/10次MrgSort；复用A、复用B、分段B、复用分段B分别为12/8/10/8次。**这是模型流程次数，不是硬件耗时或加速比。**

完整控制流程的CPU模型不是实际512线程设备执行；原语检查使用128模拟线程。以上不能验证Bisheng编译、SIMT地址空间、DCache、寄存器/spill、实际同步或性能。

可复跑本机检查：
```bash
python3 -B checks/check_mtp7_pack.py
python3 -B -m unittest checks.check_mtp7_pack_suite checks.check_mtp7_fix_suite checks.check_mtp8_suite
python3 -B checks/check_merge_catalog.py
```

## 5. 服务器一轮短测

默认32请求，TopK2048、capacity4096、max_token262144、block64、seed2026；冷启动/满缓存75%标签×MTP0/1/3/7。所有实现读取共同排序后的逻辑输入，不去重，不免费预计算hit。draft与首行重合不少于40%；热缓存标签75%指首行，后续因果行的实际命中率由报告给出。

只构建七个联合包装及独立ABCDF必要对象；一次完整专项门禁后一次采集，无初筛/复测叠加。warmup2、repeat5、regression-repeat5，总共320个正式目标kernel样本。复用已认证的固定参考结果及批量回读，但逐样本输入不变性、状态、输出、保护区校验不取消；连续状态仍按真实状态计算参考。Profiler开始ACTIVE后保留1秒Host等待，仍严格核对目标kernel、CSV单位、样本数及顺序。

HDK升级后应使用同场ABCDF和BD8比较，不能直接把不同环境的历史微秒数相减。

```bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
RUN=$(mktemp -d /workspace/single_op_test/mtp7_sweep_XXXXXX)
git clone --single-branch -b feat/mtp7-buffer-tile-sweep \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
bash scripts/test_mtp7_sweep.sh --device 0 \
  --output "$RUN/quick" --temp-dir "$RUN/tmp"
cat "$RUN/quick/report.md"
```

结果保存在持久化/workspace。先确认退出码0、`MTP7_SWEEP_ALL_STAGES_PASSED`、COMPLETE.json和report.md齐全；查看共享/回退与descriptor，不能把回退当成联合收益。报告同时列出相对ABCDF、原193、A194、B195的P50比值，保留P95、样本数、输入/源码/设备/库身份、原CSV路径和阶段墙钟。

短测通过且有明确收益后，才在同源码/库下复用本轮门禁做正式采集：
```bash
bash scripts/test_mtp7_sweep.sh --device 0 --skip-build --formal \
  --correctness-report "$RUN/quick/gate.json" \
  --output "$RUN/formal" --temp-dir "$RUN/tmp"
```

门禁身份或工况指纹不匹配会拒绝复用。失败立即停止，保留原始异常、采集与failure_state，不输出成功性能结论、不自动删除文件。只比较设备目标Plan kernel Duration；恢复、CPU校验、Host等待、Transfer/SFA均不计入。公司输入、设备现场、二进制和采集数据只留运行机器，不推送GitHub。
