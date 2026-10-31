# Bitmap 单VF启动结构候选规避（根因未确认）

失败基线：`84769ae5d30b9f1ee1cc04ee34e34b27c86c5ee2`。服务器反馈该版本在CANN9.1.0/950编译成功，但MTP0出现缓存、LRU、currentSlots和miss语义错误。只读输入没有差异；没有同库ABCDF对照数据，故不能完全排除状态准备或框架问题。原失败调用序号未知。

设备源码的精确diff见[bitmap_single_vf_kernel.patch](bitmap_single_vf_kernel.patch)。

本候选只改变 `JOINT_OPT_MASK==16 && AllowJoint==true` 的外壳调用点：

```cpp
// before
if (jointHashCapacity == 0) VF<false>(...);
else VF<true>(...);
// candidate
VF<true>(...);
```

MTP0仍不是联合算法：VF内部在jointHashCapacity==0时按rowStart/rowStride逐行调度，ProcessRequestRun的组长为1，进入RunPrivateRow→ProcessRuntimeRow。BuildBitmapUnion不执行。AllowJoint=false、其他旧mask、线程数、同步、线程1发布、位图算法和工作区布局均未修改。不要把本候选称为已修复、已证明编译器Bug或增加同步。

## 分阶段执行

使用逻辑NPU0，所有重复审计设置5次；旧库ABCDF对照例外，只调用1次。连续状态门禁保留原有六步状态转换序列（不是六轮重复审计），覆盖joint/private转换、reset和inactive行。

1. `old-control`：使用失败版本原checkout及现有库，ABCDF在完全相同MTP0参数下执行1次。不修改/重建旧checkout。旧库仍由旧源码manifest检查，不能绕过身份校验。
2. `candidate`：构建新候选；joint_bitmap在完全相同MTP0参数下执行5次，逐次严格校验。
3. 接着两路完整必要门禁（abcdf/joint_bitmap），含MTP0/1/3、disjoint/MTP3溢出、精确容量边界、连续状态、输入不变性、UB/GM与上下文检查。已知故障尺寸回归每路协议候选五次，小用例审计也五次。
4. 仅75pct/independent、MTP0/1/3、warmup2/repeat5采集。复用同候选门禁报告；不跑19路或27个性能工况。

脚本使用set -euo pipefail，任何阶段失败就停止。每次运行创建独立结果目录，旧结果不会覆盖。探测器在每次下发前持久化call_number，失败报告中保留正在执行的序号；common失败现场另有implementation、phase、invocation_count和workload。调用计数以当前bundle为单位，包含审计/预热/正式调用，不把所有失败都叫“第一次失败”。

## 克隆、固定提交与调用

将交付消息中的完整候选提交号赋给CANDIDATE_COMMIT。OLD_SOURCE必须指向保留的失败版本checkout（含原库），不能指向刚克隆的新代码。下面整个块在同一个bash进程执行，任一步失败不继续。

```bash
set -eo pipefail
export CANDIDATE_COMMIT='填入本次交付的完整提交号'
export OLD_SOURCE='/workspace/原失败版本源码目录'
export PYTHON=/usr/local/python3.11.10/bin/python3
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /workspace
git clone --branch fix/bitmap-single-vf-launch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git bitmap_single_vf_candidate
cd bitmap_single_vf_candidate
git checkout --detach "$CANDIDATE_COMMIT"
git rev-parse HEAD
export RESULTS_ROOT=/workspace/bitmap_single_vf_results
bash scripts/test_bitmap_single_vf.sh old-control "$OLD_SOURCE"
bash scripts/test_bitmap_single_vf.sh candidate
```

已有克隆目录时不要覆盖它；选一个新的目录名。若旧库不可用，可以跳过old-control阶段，但必须将“缺少原失败库ABCDF对照”记录为未验证，不得声称已排除框架因素。旧版本源码被诊断脚本修改导致身份失败时，不要通过复制新manifest绕过。

脚本保存source_commit、构建日志、MTP0逐调用报告、gate报告、profile日志及原始采集。数据、失败输入、二进制均留在/workspace，不上传GitHub。

## profiler开始等待

使用已有schedule(wait=0,warmup=1,active=repeat)。warmup步骤的prof.step()切入ACTIVE后，在第一个正式目标调用前显式等待1秒。`capture_start_wait.json`记录原因、开始时间、请求时长、实际等待时长；profile.log也打印该事件，summary.json保留记录。没有把这1秒加入kernel Duration。

这是采集侧规避，不是算子正确性修复。CSV仍精确匹配两个kernel、每路5个样本、实际调用顺序及同流时间关系；丢失样本仍判失败，不丢弃首样本、不补造样本、不放宽计数。

## 本地验证与下一步

本地可执行：CPU语义模型、真实VF/helper的64线程CPU仿真、Python/bash语法、源码身份、C++标准预处理结构对照、git diff --check。这些均不能验证CANN生成代码、UB地址空间或NPU行为。此前服务器“旧版本编译成功”不能算作新候选编译通过。

如果新候选MTP0仍失败：保留本轮现场并停止采集，不继续围绕VF启动结构改动。下一步比较同库ABCDF，定位ProcessRuntimeRow的Hash命中、owner回填和Scan路径；本次不提前加入这些诊断改动。
