# MTP动态容量：映射暂存与全UB两条候选

本分支以FUSED AB为归并基础，针对TopK=2048提供两种独立实现。
默认32个请求，MTP0/1/3/7，对应32/64/128/256条query行。容量按MTP动态选择：MTP0/1=4096，MTP3=8192，MTP7=16384。同一kernel使用最大16K UB布局，低容量只缩小owner/partition的实际范围。

| 实现 | 编号 | 动态UB | GM临时映射 |
|---|---:|---:|---:|
| abcdf_pages | 原独立基线 | 按每档容量tiling | 原实现 |
| joint_mtp7_stage16_512 | 214 | 197408 B（192.78125 KiB） | 每query位置2B |
| joint_mtp7_ub16_512 | 215 | 203552 B（198.78125 KiB） | 无 |

老FUSED/AB的原样版本无法启动16K，不能伪造其16K数据；本轮使用真正的带页表ABCDF同容量对照。

## 数据流与空间

两候选保留硬件MrgSort、AB硬件消费量复用和完整64记录块解交织加载；主路径512线程，八路上限612记录/路，fallback1024线程。
映射区域固定32KiB（即便单行），unique64KiB，归并区96KiB。每阶段均使用同一段UB。

stage16：热规划前按32位打包将query→UID写入独立GM workspace。该映射大小只随query数变化，低容量MTP仍按最大布局预留UB。随后旧映射区作为32KiB uint16 flags；旧归并区分作owner64KiB与partition32KiB。最后由写入该GM word的同一线程读回两个UID，输出两个currentSlots；不涉及Scalar/DMA读取该GM中间值。MTP7、32请求时GM映射分配1MiB，逻辑新增写+读2MiB/调用；实际物理HBM流量取决于cache。冷reset直接使用UB映射，不做这笔暂存。

ub16：query→UID一直留在UB；owner64KiB及partition32KiB复用归并区，后面增加6KiB位图（三份：hit、invalid、valid）。每warp ballot产出一个32bit word，lane0独占写，不做逐bit非原子更新。partition按LRU重读槽号；miss rank写到owner的空闲位（bit30标记新miss，bits16..29存rank），严格在原子owner仲裁结束之后。

unique中的token位置保持int32。owner保留int32原子max以维持重复resident的LRU-last选择，同时所有命中槽都受保护。flags16的hit与invalid互斥，因此0xffff可作为非法槽标记；partition低14位槽号、bit14失效标记。

**执行方式边界：本轮压缩布局的热分类/稳定压缩/miss分配使用SIMT Scan。原BD的HotCompact SIMD假设32位数组，不能直接套在压缩布局上。归并去重与消费统计仍使用SIMD。这是两条可测实现，不应把与旧BD不同容量下的差值解释为纯GM/UB差异，也不保证更快。**

动态UB额外800B=控制区512B+Scan288B。`design/capacity16`记录6个32KiB区域的存活/配槽/物理峰值模型；ub16额外6KiB位图单独计入。模型不包含编译器寄存器spill，最终能否编译及实际资源占用以CANN为准。

## 语义和门禁

限定TopK2048/capacity16384，其他形状由host明确拒绝。1/2/4/8行按原协议，超出支持条件仍走原private回退；不是扩大UB门禁后允许任意形状。正常八行最多16384项，因此合法query的并集不会超过16384，但回退检查保留。

CPU检查执行真实VF/controller，经C++桩模拟硬件API，使用UBSan、输入不变性及缓冲保护区；覆盖四档MTP、动态容量、精确4096/8192/16384并集、重复token、无效区间、重复resident及连续状态。CPU通过不代表CANN编译、缓存一致性或NPU性能通过。

上机先专项门禁，再单轮三路profiler；每个样本恢复相同初态并完整检查。门禁与性能使用预期结果缓存；默认warmup2/repeat5/regression5，无38路初筛。正式重复采集可复用同源码/库/设备/输入的gate.json。失败保留现场并停止，不产生成功结论。

75%是首行标签，draft实际命中率由报告记录；共同输入升序，保留重复和因果可见性。输入历史候选池默认16384；不要与旧4K池直接算加速比。

## 服务器命令

```bash
set -euo pipefail
RUN=$(mktemp -d /workspace/single_op_test/capacity16_XXXXXX)
git clone --single-branch -b feat/mtp7-capacity16k-layouts \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN/src"
cd "$RUN/src"
git rev-parse HEAD | tee "$RUN/commit.txt"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export PYTHON=/usr/local/python3.11.10/bin/python3
export JOBS=3 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset ASCEND_LAUNCH_BLOCKING ASCEND_RT_VISIBLE_DEVICES
ulimit -c 0
# 默认逻辑卡0，按空闲设备修改--device。构建仅两条新增候选及工程基础目标。
bash scripts/test_capacity16k.sh --device 0 --requests 32 \
  --warmup 2 --repeat 5 --regression-repeat 5 \
  --output "$RUN/results" --temp-dir "$RUN/tmp"
cat "$RUN/results/report.md"
```

构建失败先看`results/build.log`；执行失败看`results/launcher.log`及`failure_state`。不在无硬件开发机运行上述服务器命令。

同库复测（上一轮成功后）：

```bash
bash scripts/test_capacity16k.sh --skip-build --device 0 --requests 32 \
  --correctness-report "$RUN/results/gate.json" \
  --warmup 10 --repeat 20 --regression-repeat 5 \
  --output "$RUN/confirm" --temp-dir "$RUN/tmp"
```

## L2说明

L2是硬件管理的缓存，不是能通过指针显式分配的UB区域。官方GlobalTensor提供SetL2CacheHint策略提示；不能因此保证SIMT裸指针访问的指定数组常驻L2。不同访问路径的策略接口不能混用。本轮保持默认策略，不添加“锁L2”假设，也不关闭已有DCCI同步；暂存流量及其等待全部在目标kernel计时内。

参考：https://asc.gitcode.com/api/SIMD-API/basic_api/data_structures/GlobalTensor/SetL2CacheHint.html
缓存一致性：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/基础API/缓存控制/DataCacheCleanAndInvalid.md
