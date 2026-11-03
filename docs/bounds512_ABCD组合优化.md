# bounds512 后续 A/B/C/D 组合实验

本分支以已经上机验证的 `joint_integrated_bounds_512` 为冻结参考，增加16种固定512线程配置：control、A、B、AB、C、AC、BC、ABC、D、AD、BD、ABD、CD、ACD、BCD、ABCD。另保留带页表 `abcdf_pages`，默认共18路。字母仅指本轮优化，与历史 ABCDF、AC512 及旧 ABCD 的字母无关。

新候选尚未执行 CANN 编译或 NPU 验证。CPU API 模型验证的是算法与控制流程，不能验证真实地址空间、指令支持、同步实现或性能；不能预先承诺组合更快。

## 本轮四个因素

| 因素 | 实际实现 | 适用范围及保留行为 |
|---|---|---|
| A | 单行严格递增 TopK 在 Phase0 直接写 unique 与 query→UID，不再打包 score/payload 或进入 Phase3 | 保留完整重复检测；重复 query 回退原逐行路径。无效前后缀仍写 -1，保持因果可见性。多行保持原归并 |
| B | boolean Scan 在 warp 内用 `asc_ballot` + `__popc` 生成稳定 exclusive rank | 跨 warp prefix、parity 和已有 barrier 保留；hit/evict 打包 Scan 不改。作用于 SIMT Phase3 和 Phase5 miss Scan |
| C | 热 UB 路径先顺序读取 `slotToToken` 到尚未使用的 `partition`，再按 LRU 顺序读取 UB 快照 | 复用同一 capacity 缓冲的前一生命周期，不扩大 UB。分类结束沿用已有 barrier 后才覆盖 partition；reset 及 GM 路径保留原行为 |
| D | 完整 SIMD Phase3：相邻比较 → `Unsqueeze` 编号 → unique 与 query 映射 Scatter | 替换原 Phase3，不是多加一次预处理 VF；保留硬件 MrgSort。GM scratch 使用原 SIMT Phase3。保持 int32 映射和原工作区布局 |

D 的编号处理每个原始 query，而不是只保留 unique：重复 token 的全部 payload 都写回对应 UID。每64 lane之间携带上一 token 与 UID 计数，跨归并批次也携带；即使容量溢出仍先完成 scratch 检查，随后沿用原逐行回退，不提前修改缓存。Scatter 使用32字节对齐的 data/ctx 基址；unique 使用相对 data 的索引，兼容 TopK63/65 等非对齐形状。

D 使用 float32 记录中的负 token 转回 int32，沿用冻结版本对 `max_token <= 2^24` 和整数精确表示的限制；不满足条件仍走原回退。没有新增 GM 时间戳、额外 workspace 或跨核状态。

**组合不是简单重复执行两个实现。** A 命中单行快路径时不执行 Phase3，所以 A+D 的 D 主要作用于多行；B+D 在 UB Phase3 只执行 D，而 B 仍优化 Phase5 miss Scan。GM Phase3 不执行 D，选中 B 时仍使用 ballot Scan。C 与 A/B/D 独立，但不保证组合收益可加。

`joint_refine_base_512` 是新源文件组织下关闭四个优化的 control，用来观察函数组织及编译资源变化。它不能替代冻结 bounds 参考。公开配置ID127..142独立，目标 kernel 名不同；旧基线的计算文件保持不变。

## 阅读顺序

1. `joint_variants.py` 的 REFINE_OPTIONS / REFINE_SWEEP：名字、ID、因子与协议。
2. `csrc/joint_variants/refine_stage.h`：Phase0 单行构造、boolean Scan、热 resident 快照，以及未改动的状态提交。
3. `csrc/joint_variants/refine_simd.h`：D 的完整编号/映射。
4. `csrc/joint_variants/refine_controller.h`：阶段调度、MrgSort、D 的替代调用；`cold_finish.h` 仍为原 reset 收尾。
5. `checks/check_refine_kernels.py`：CPU 全字段、跨向量边界和连续状态验证。
6. `checks/check_refine_npu.py` → `run_refine_sweep.py` → `profile_joint_mtp.py`：设备门禁、初筛/复测、CSV采集与统计。

新文件中的中文注释说明关键生命周期；未将算法工作移动到 CPU 以制造加速。

## 服务器运行

使用新的独立目录，保留以前的源码、库及结果。设备明确为逻辑0。以下脚本在服务器已有匹配 torch/torch_npu 环境后执行，CANN9.1、ascend950、`-O3` 与现有构建方式一致。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
mkdir -p /workspace/single_op_test
cd /workspace/single_op_test
RUN_DIR=$(mktemp -d /workspace/single_op_test/bounds_refine_XXXXXX)
git clone --branch feat/bounds-plan-refine-abcd --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git "$RUN_DIR/src"
cd "$RUN_DIR/src"
git rev-parse HEAD | tee "$RUN_DIR/source_commit.txt"
export PYTHON=/usr/local/python3.11.10/bin/python3
bash scripts/test_refine_sweep.sh --device 0 \
  --output "$RUN_DIR/results" --temp-dir "$RUN_DIR/tmp"
```

运行前请确认 `source` 路径和已激活的环境正确。没有主动选择其他输出文件系统；路径不足会在初始化NPU前失败。最低可用字节及inode阈值可用已有 `--min-output-free-bytes`、`--min-temp-free-bytes`、`--min-free-inodes` 调整，阈值不是文件大小预测。数据和二进制只留在运行机器。

默认一次命令包含：

1. `--refine-only` 精简构建：独立 ABCDF、冻结 bounds 及16个refine配置；保留源码/库指纹。
2. 所选实现的严格门禁：输入不变性回归5次，冷/热、MTP0/1/3、重复 query/resident、有效区间、非均匀组、并集溢出、自然GM路径、保护区与连续状态；另加63/64/65等向量边界。
3. 六工况全部18路初筛，warmup2、repeat5。保留所有初筛结果，不因慢而删除数据。
4. 每个工况选前2个优化候选，取候选并集，并始终保留ABCDF、冻结bounds和control；在同一输入下新采集 warmup10、repeat50。MTP0也独立参与选取，不把不同工况赢家拼成一个实现。

如果希望**全部18路均正式采集**，跳过初筛：

```bash
bash scripts/test_refine_sweep.sh --device 0 --full-confirm \
  --output "$RUN_DIR/all_formal" --temp-dir "$RUN_DIR/tmp"
```

已有本分支构建可加 `--skip-build`，仍检查库与源码身份，不可复用其他分支旧库。要缩小候选范围必须保留三个参考，例如：

```bash
bash scripts/test_refine_sweep.sh --device 0 --skip-build --full-confirm \
  --implementations abcdf_pages joint_integrated_bounds_512 \
    joint_refine_base_512 joint_refine_a_512 joint_refine_b_512 \
    joint_refine_c_512 joint_refine_d_512 joint_refine_abcd_512 \
  --output "$RUN_DIR/subset" --temp-dir "$RUN_DIR/tmp"
```

可先运行 `--describe-only`，打印配置而不初始化NPU或构建。门禁失败立即停止，现场保留在结果目录的 `failure_state`；不生成成功报告或 COMPLETE.json。阶段错误显示详细日志路径和首个可定位异常。不要在未通过门禁时继续性能采集。

## 输入与结果口径

固定32请求、TopK2048、capacity4096、max_token262144、block_size64、seed2026。MTP=0/1/3分别对应32/64/128条query，MTP=3是3个draft。场景为cold_reset和full_cache_75pct；共同输入计时外排序，不去重；overlap40保证draft与首行至少40%重合。

75%指首行的初始命中率；后续行的因果可见性和内容不同，实际miss按报告读取，不声称每行都是75%。三路协议分别验证：ABCDF原逐行协议；有序单行保留private协议；多行联合使用joint协议。原1024线程逐行回退、线程1模式发布规避和其根因未确认的说明均保留。

所有目标调用前恢复相同逻辑初态、污染输出与scratch并校验；逐样本检查只读输入、逐位置currentSlots、LRU、miss有效前缀与原尾部契约、缓存状态和保护区。连续状态门禁复用上一轮真实设备状态，不用golden覆盖掩盖错误。

以 `kernel_details.csv` 中精确目标 kernel 的设备 Duration(us) 为准，严格验证数量和轮换执行顺序，排除TensorMove及状态恢复。沿用采集 ACTIVE 开始后1秒宿主等待，记录在metadata中；它仅是采集侧规避。每轮轮换18路先后顺序，不能把接口Event当成kernel耗时。

最终 `report.md` 只使用独立正式样本，列出P50/P95、样本数、当前实现/ABCDF、当前实现/bounds、当前实现/control及共享/回退/单行组数。比值小于1表示更快。`selection.json`保存初筛排名；采集子目录的 `capture_metadata.json`、`data_index.json` 保留原始CSV索引、样本、输入与库/编译/设备身份、实际miss/descriptor数量、相对自身MTP0耗时。

只有退出码0、报告和 COMPLETE.json 同时存在才算整轮成功。若使用 `/dev/shm`，这些是临时数据，需要自行转存到持久化目录。仅测Plan；未测Transfer/SFA及端到端收益。

## 本地验证边界

16种配置分别与冻结bounds完整控制器逐字段对照，每种324次；CPU API模型+UBSan。另有308项SIMD unique/map模型检查和44项ballot Scan检查，覆盖跨64-lane重复、容量溢出及float整数精确表示上界；Scan原语使用256个CPU模拟线程覆盖8个warp。

这些不是CANN编译或真实SIMT/SIMD执行。还需服务器验证：CANN9.1 API签名/支持、UB地址空间、warp投票、Scatter、真实同步、寄存器/spill与性能。同步脚本未报告候选，但正则数据流分析未识别Reg操作的buffer访问，不能据此证明新SIMD路径同步正确。
