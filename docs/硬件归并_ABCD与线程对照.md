# AC 硬件归并基础上的新 ABCD 与线程对照

本分支基于 `b7020a702cedd07fe0ca863fcfaaa83fb6b0189e`。历史带页表 ABCDF 和已上机验证的 `joint_hw_ac_512` 保持原计算代码，作为两条固定参考。本分支新增实现尚未经过 CANN 9.1.0 编译或 NPU 验证；CPU 检查不能证明设备地址空间、同步或性能正确。

## 实验目录与命名

`joint_union_{组合}_{线程数}`，组合为 `base/a/b/ab/c/ac/bc/abc/d/ad/bd/abd/cd/acd/bcd/abcd`，线程数为 `1024/512/256`。共 48 个新候选，加两条参考共 50 路。新 `base` 没有本轮 ABCD，保留 AC 的硬件归并基础并增加单行有序能力、128-word 控制区；不是与历史 AC512 二进制相同的别名。

本轮 ABCD 与历史 `hw_ac_512` 的字母含义不同。

| 因素 | 实际实现 | 风险与验证要点 |
|---|---|---|
| A | SIMD 相邻比较 + 64-lane 寄存器 Hillis–Steele inclusive prefix；payload 高16位保存包内前缀，低16位保留原 query 编号；SIMT 散写 unique/map 不再做 Scan | 每个原 query 都必须得到编号；跨64元素、跨归并包重复，尾mask、浮点token边界；SIMD 多一个 VF 调用，不保证更快 |
| B | UB query→UID map 用 uint16，65535 为无效值；GM 路径仍用 int32。腾出的空间用于自适应归并缓冲 | 单个 query 唯一写者；16位设备访存、奇数长度、对齐及 uint16 上界须上机验证；没有 OR 或 CAS 打包更新 |
| C | 去重/映射消费完成后，在同一个 SIMT VF 内准备下一包；控制器识别已准备标记，省去一次 VF 入口和对应的 V→S 等待 | 原容量检查与读写屏障保留；不删除 MrgSort 消费数量读取所需同步 |
| D | 仅初始化 query 的非法/不可见区间；成功的有序路径不清理 miss 描述符尾部 | 有效前缀严格核对，未写尾部必须逐值等于调用前值；原回退行为不变。这是显式接口候选，不能混称为完全相同的全缓冲输出约定 |

A 没有简单地 Squeeze 后丢弃重复元素：重复 query 的 UID 也保留，后续 current_slots 才能逐位置正确。该版没有融合 TopK，也没有额外建立 resident Hash 或持久 token→slot 索引。

## MTP0：单行有序 private 路径

新候选的 MTP0 保留 `ROW_PRIVATE` 缓存协议：

```
识别单行/请求身份及因果有效区间
→ 严格递增时，直接准备单路记录（不执行 MrgSort）
→ 去重/编号 → resident 查询 → LRU victim 配对 → private 状态与输出
```

输入包含重复 token、值域/容量不受支持或资源不足时，仍调用原 `RunPrivateRow`。原因是原独立算子允许同一个缺失 token 的多个 occurrence 分别产生 miss，不能为了复用联合去重而改变其语义。单行严格递增检查故意保守，包括无效位置的重复也会回退。

MTP1/3 则保持联合协议。连续输入片段只有一行时也按 private 处理；联合/逐行切换仍使用原请求身份及 mode reset 规则。没有把旧的逐行回退改成512/256线程。

因此，1024/512/256 是**新有序规划及新单行路径**的 SIMT 线程数。硬件 MrgSort 不按这个线程数执行；回退统一保留1024线程。报告同时记录请求行数、共享/回退组数与单行有序组数，不能把回退结果解释成对应线程数的有序路径性能。

框架中 `layout.joint_enabled` 为底层准备上下文的控制开关；MTP0新候选也会开启，以到达单行有序控制器。另有 `logical_joint_enabled=false`、`single_sorted_requested=true` 消除歧义。MTP0使用独立 reference_simt 校验，未伪造成联合协议。

## UB、数据生命周期与同步

新布局：Scan 288 B；控制区128个int32（前64个元数据、后64个 SIMD lane 索引）；query map；unique；归并工作区。工作区在归并后复用为 owner/flags/LRU，Host 按未压缩上界保留容量，B 的收益是让实际包缓冲更大。

- Phase0：建立有效区间、初始化必要map项及lane表。
- Phase1：从GM准备包记录，输入输出缓冲不重叠。
- MrgSort：硬件多路归并，或单路直接旁路。
- A SIMD：只改本包 payload，score 保持不变。寄存器前缀处理所有原query，不提取中间标量。
- Phase3：消费包并散写映射。C 在所有读取完成的 block barrier 后复用包区，准备下一包。
- Phase5：完成容量/可淘汰槽检查后，才写持久状态；保留线程1发布 mode 的经验规避，未声称底层错写根因已确认。

| producer → consumer | 缓冲 | 同步/生命周期 |
|---|---|---|
| Phase1 → MrgSort | packet input | 原 V→S Set/Wait，事件配对释放 |
| MrgSort → Scalar消费计数 | packet output + consumed | 保留原 V→S Set/Wait |
| Scalar游标 → 后续VF | ctx | 保留原 S→V Set/Wait |
| A SIMD → Phase3 SIMT | packet payload | V→S Set/Wait完成后才下发消费VF |
| Phase3 → C融合的Phase1 | packet/ctx | block barrier 后覆盖；该轮输出已无后续读者 |

静态同步脚本没有报告候选，但正则数据流工具没有建出 SIMD/SIMT 指针访问图。此结果不能证明真实流水同步正确；NPU门禁仍是必需步骤。

## D 的有效前缀约定

对每行只消费 `[0, miss_count[row])`。成功的 D 有序路径中，尾部严格保持调用前值，不补写 -1；检查器并不把尾部归一化为golden。关闭 D 的新路径仍清理尾部。所有实现的有效 miss、当前槽位、LRU、缓存内容、输入不变性、保护区及非活动行都保持严格检查。

现有 `csrc/transfer/kernel.cpp` 按 miss_count 遍历描述符前缀，但这不代表本轮已经验证 Joint→Transfer 全链路兼容性。新的任何消费者必须遵守前缀约定。D 的性能收益需要同时注明省去了无用尾部写入；不可声称完全相同的全缓冲写入量。

## 本机检查与限制

执行真实新控制器、SIMT helper 以及 A 的 SIMD 函数体；CPU 为 Reg API 提供64元素语义模型，配合 UBSan/保护区检查。不是另写一份A算法来代替执行设备源文件。SIMT主模型线程数缩小到64，512/256配置相应缩小并至少保留一个warp。额外测试覆盖更大的warp数。

检查覆盖：MTP0/1/3、冷/热、连续调用、request变更、stable prefix失效、重复query/resident、非法与不可见输入、零有效输入、局部活动行、并集溢出、超过4行回退、UB/GM、生产尺寸及浮点token边界。SIMD前缀单测覆盖64-lane边界和跨包延续；D校验尾部原值保留。

本轮实际执行结果：48候选共6912次控制器检查和1440次SIMD前缀检查通过；额外较大warp模型通过144次控制器及30次前缀检查；保持原1024配置并实际启动256个CPU worker的8-warp路径通过6个支持工况。27项框架模型、5项新runner检查、4项旧runner回归、9项full-cache输入检查、139组overlap输入契约检查及五种目录的dispatch编译/链接通过。以上计数均为CPU检查，不是设备性能结果。

这些检查不能模拟NPU地址空间、16位UB指令、编译器寄存器分配、真实同步和性能。没有本地硬件结果，不预填任何性能数字。

## 服务器一键运行

推荐保留旧验证目录，用新的克隆目录。逻辑卡0；CANN9.1.0；Python3.11.10。结果默认写持久化 `/workspace`，不上传输入、二进制、失败现场或profiling产物。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export JOBS=3
cd /workspace/single_op_test
git clone --branch feat/hardware-union-abcd-threads --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git union_abcd_src
cd union_abcd_src
git rev-parse HEAD
bash scripts/test_union_next.sh --device 0 \
  --output /workspace/single_op_test/union_abcd_results_01 \
  --temp-dir /workspace/single_op_test/union_abcd_tmp_01
```

一键入口依次执行：空间与inode门禁 → `-O3 --union-only`构建 → 全部选中实现的正确性门禁（错写回归5次）→ warmup2/repeat5 初筛 → 分工况/MTP选择前两名 → 新采集 warmup10/repeat50 独立复测。默认 cold_reset 和 full_cache_75pct、overlap40、固定32请求，MTP0/1/3展开32/64/128行。MTP0有新单行路径，单独参与入围，不与MTP1/3平均排名。

48路编译与检查仍有成本，可用 `--implementations`缩小**运行**范围，构建仍包含全部新候选。首次建议先小范围门禁，命令如下；同库随后一键脚本可以 `--skip-build`。

```bash
bash build_joint_plan.sh --soc=ascend950 --union-only
"$PYTHON" -B -X faulthandler checks/check_union_next_npu.py --device 0 \
  --implementations abcdf_pages joint_hw_ac_512 joint_union_base_512 joint_union_abcd_512 \
  --regression-repeat 5 --report /workspace/single_op_test/union_abcd_smoke/gate.json
```

该专项包含小尺寸边界和必要生产尺寸回归；不只是简单“能启动”。任何失败立即停止，保存实现、工况、调用次数及现场。不能把失败版本继续用于筛选。

只对比线程、不启用新ABCD：

```bash
bash scripts/test_union_next.sh --device 0 --skip-build \
  --implementations abcdf_pages joint_hw_ac_512 \
    joint_union_base_1024 joint_union_base_512 joint_union_base_256 \
  --output /workspace/single_op_test/union_threads_results_01
```

所有计时均来自精确目标kernel的 `kernel_details.csv` Duration；轮换下发顺序，保留采集启动后的1秒宿主等待。恢复、校验、辅助kernel及CPU等待不混入目标Duration。每样本核对，CSV单位/计数/顺序严格匹配；初筛与独立复测不混合统计。失败无成功报告。两轮输入指纹、源码/库/设备身份必须一致。

根目录 report.md 为独立复测摘要；screen/confirm 子目录各保留 data_index.json、capture_metadata.json、原始CSV和样本，selection.json 为入围依据，RUN_STATUS.json 为执行状态。单次50样本复测仍不保证跨运行重现；不测Transfer/SFA，不能据此声称端到端加速。

## 阅读顺序

1. `joint_variants.py`：新因素和mask映射。
2. `csrc/joint_variants/union_next_controller.h`：阶段调度及C融合。
3. `csrc/joint_variants/union_next_simd.h`：A相邻比较与寄存器前缀。
4. `csrc/joint_variants/union_next_stage.h`：B布局、D初始化/尾部及单行private发布。
5. `joint_plan_cases.py` / `joint_plan_common.py`：独立golden、尾部约定与上下文准备。
6. `checks/check_union_next.py` / `checks/check_union_next_npu.py`：CPU模型和NPU门禁。
7. `run_union_next_sweep.py` / `profile_joint_mtp.py`：独立初筛/复测和原始CSV统计。

API核对参考：CANN9.1的 [寄存器Gather](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/scatter_operations/Gather.md)、[UB Gather](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/Gather_discrete.md)、[Scatter](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/reg_data_store/Scatter_discrete.md)。本机同时阅读了本地官方asc-devkit声明与真实RegBase样例；API签名审阅不等同于9.1设备编译通过。

## ShiftLefts 参数类型修正

CANN9.1 的 `Reg::ShiftLefts` 标量移位量要求为 `int16_t`，与向量元素为 `uint32_t` 是两个独立约束。原提交381a3a5两处使用了uint32_t标量；现已分别改成 `int16_t{1}`、`int16_t{16}`，位数和算法不变。CPU API模型新增编译期约束，回归检查确认int16_t可编译、旧uint32_t写法被拒绝。

用户回传的服务器结果身份为“381a3a5＋两处ShiftLefts参数类型修正”，不能表述为原提交未经修改直接通过。这里仅固化明确的API修正；没有在开发机执行CANN9.1或NPU验证，也没有据此确认服务器完整源码及二进制与本次提交逐字相同。
