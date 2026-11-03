# 512线程整合 Plan：单行有序 + 多行联合

本分支保留 `joint_integrated_512`，另加一个仅优化 Phase0 的候选 `joint_integrated_bounds_512`。不增加线程/组合矩阵，不接 Transfer。
目标是把已经分别测试过的两条算法放进同一个设备入口，并观察整合后有无编译资源回退；**尚未在本机执行 CANN 编译或 NPU 验证，也没有本版本的性能结果**。

## 五条对照路径

| 测试名称 | 用途 |
|---|---|
| `abcdf_pages` | 冻结的带页表独立 ABCDF，1024线程，原始性能参考 |
| `joint_union_cold_512` | 冻结的单行/联合参考；单行有序 private 路径 + reset 专用收尾 |
| `joint_hw_ac_cold_512` | 冻结的 AC512 联合参考 + reset 专用收尾；MTP0仍为原逐行路径 |
| `joint_integrated_512` | 原整合版：单行采用第一条冷参考的语义，多行采用 AC 冷参考的布局和控制器 |
| `joint_integrated_bounds_512` | 本次新增：相对原整合版仅替换 Phase0 合法区间计算 |

原参考的计算头、控制器、`cold_finish.h` 和原 `integrated_stage.h` 均保持不变。新增 `phase0_bounds_stage.h`；源码身份检查逐字核对它相对整合版仅有下面这一处变化。

## 设备如何分派

Python 不依据性能工况挑选其他算子，不预计算命中结果。`joint_integrated_512` 每次调用使用一个目标 kernel：

```text
按 tokenToReq 中连续的相同 request 分组
  1行：严格递增且满足表示/容量条件
       → 单行有序规划；private缓存协议；不调用多路MrgSort
  2～4行（主要生产形状为2/4；保留3行尾组能力）
       → AC512硬件归并 + 相邻去重 + resident查询
       → joint / follower缓存协议
  超过4行、表示范围不支持、容量检查失败、单行存在重复token等
       → 原1024线程逐行回退

有序路径得到unique与query→UID映射后：
  请求ID或模式不匹配 → reset专用收尾：slot=UID，LRU旋转，发布元数据
  否则              → 原热路径：resident命中，LRU victim/miss配对，写回
```

有序输入是接口前提：Python入口对未升序输入明确报错，不会给一个实现单独排序，也不会静默在设备上把乱序输入当有序输入处理。允许重复的多行查询仍保留重复 query 的逐位置映射；单行重复必须回退，避免改变独立算法的重复 miss 语义。

分组依据实际 `tokenToReq`，不是只看 `mtp` 标签。部分活动行、非均匀组、非连续相同请求分别成组，沿用现有调度语义。大于4行的组不触碰4T映射区，走原回退。

单行 private、联合 leader/follower 的 resident 与 LRU 协议不变；拆组/重新联合时沿用既有模式 reset，不做 CPU 状态归一化。完整保存只读输入、持久缓存、LRU、miss有效前缀、描述符尾部与保护区检查。

## 整合时刻意保留的实现细节

- 使用 AC 的64个 int32 控制字；query→UID仍为 int32，不启用上一轮16位打包因素。
- 共用 `sorted_sweep_controller.h`，没有同时编译两套主控制器来按运行时选择。
- 有序阶段512线程，原逐行回退1024线程；没有加入256/1024新对照。
- 生产尺寸的scratch为24640个int32，含Scan的动态UB为98848字节；上机prepared上下文继续核对Python/C++计算一致。该预算不是整个硬件UB容量。
- 单路直通、组大小映射、自适应分块来自已有 AC；不新增阶段融合、索引、尾部缓存或SIMD编号实验因素。
- 保留现有barrier和V↔S事件，不通过删同步换性能。
- `rowModes`仍由线程1发布。它是已有实验支持的规避，底层根因未确认。
- public ID125映射到算法配置72，另加 `JOINT_INTEGRATED` / `JOINT_COLD_FINISH` 编译宏；源/库身份包含这些包装、分派与文件指纹。ID变更不是把新算子当旧库运行的理由。

新增单行分支、不同的控制区或模板实例都可能影响最终指令和寄存器占用，所以不能把两条参考历史最好耗时拼接成整合版性能。

## Phase0 新增了什么

只在新候选中，对每一行自己的、截断到 `[0,max_token]` 的 `upper` 使用：

```cpp
if (topk > 0 && indices[0] >= 0 && indices[topk - 1] < upper) {
    begin = 0;
    end = topk;
} else {
    begin = lower_bound(indices, 0);
    end = lower_bound(indices, upper);
}
```

因为输入已经升序，两个端点合法即可推出整行合法。`token == upper` 仍不可见；负数前缀、未来 token、全非法及混合情况继续原二分。没有把首行的upper用于其他draft行，没有排序、去重或预计算命中结果。

只对Phase0加判断，通用 `SortedLower` 没变，因此resident→unique查询不增加端点读取。512线程、分组、归并、缓存/LRU、冷收尾、工作区、同步与回退保持原样。此版本不增加并行二分、Unsqueeze或其他因素。

公开ID126与ID125都映射算法配置72；新包装另外定义 `JOINT_PHASE0_BOUNDS`，目标kernel名独立，旧库身份必须失效并重新构建。报告新增“实现/原整合版”一列及原始比值键 `joint_integrated_bounds_512_over_joint_integrated_512_p50`，小于1表示优化候选更快；不按工况挑选赢家来冒充同一版本。

门禁额外为原整合版和新候选各运行15个边界用例，覆盖MTP0/1/3、负数/未来前后缀、等于上界、上界截断、全非法、单元素、重复token和逐行不同可见长度。原有连续状态与全部校验不关闭。正式计时仍是原来的冷/热六个工况，不将边界用例混入性能样本。

CPU检查调用真实Stage0读取区间，与独立searchsorted比较，并运行完整控制器与原整合版逐字段对照；宿主模型不代表设备编译或性能结果。

## 服务器一条命令完成本轮

先进入已有匹配的 torch/torch_npu 环境，保留旧 checkout、旧库、旧结果，在新目录克隆本分支。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /workspace/single_op_test
# 若目录名已有内容，请使用另一个新名字，不覆盖旧结果。
git clone --branch feat/integrated-plan-512 --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git integrated_plan_512
cd integrated_plan_512
git rev-parse HEAD   # 记录并核对交付提交号
export PYTHON=/usr/local/python3.11.10/bin/python3
JOBS=3 bash scripts/test_integrated_plan.sh --device 0 \
  --output /workspace/single_op_test/integrated512_results_$(date +%Y%m%d_%H%M%S)
```

入口会按顺序完成：

1. 输出/临时文件系统空间、inode和写权限检查，设置TMPDIR；不足时停止，不自动删除或切换。
2. `-O3`精简构建，仅四个实验包装及所需基线支持，不展开旧组合。
3. 五路严格门禁：输入错写回归5次；小尺寸MTP0/1/3、重复token、因果边界、部分活动行、3行尾组、超过4行/并集溢出回退、GM scratch、非均匀分组；另外实际续用状态完成8步冷/热/全miss/换请求/拆组/重新联合检查。
4. 32请求、TopK2048、capacity4096、max_token262144、block_size64、seed2026，`cold_reset/full_cache_75pct × MTP0/1/3`，五路共同升序输入、draft与首行重合至少40%。每路warmup10、正式50样本，不做初筛淘汰。
5. 每次调用前恢复初态，调用后完整校验，校验失败立即保留现场并停止。轮换五路先后顺序；仅使用精确kernel名的CSV设备Duration，严格核对单位、顺序、样本数。保留ACTIVE开始后1秒宿主等待。
6. 生成 `report.md`、`RUN_STATUS.json`、`COMPLETE.json`，原始CSV、输入/库/编译身份及每样本值见采集目录的 `data_index.json` / `capture_metadata.json`。失败不能生成成功结论。

若希望先短跑，可增加 `--warmup 2 --repeat 5`。随后用另一个新的 `--output` 目录，带 `--skip-build` 跑正式50样本；各轮均执行门禁，不能将短跑样本合入正式结果。

只构建、只做正确性或手动采集也可：

```bash
PYTHON=/usr/local/python3.11.10/bin/python3 JOBS=3 \
  bash build_joint_plan.sh --soc=ascend950 --integrated-only

$PYTHON -B -X faulthandler checks/check_cold_finish_npu.py --device 0 \
  --implementations abcdf_pages joint_union_cold_512 joint_hw_ac_cold_512 joint_integrated_512 joint_integrated_bounds_512 \
  --regression-repeat 5 --report /workspace/single_op_test/integrated512_manual/gate.json \
  --temp-dir /workspace/single_op_test/integrated512_manual/tmp

$PYTHON -B -X faulthandler profile_joint_mtp.py --device 0 \
  --implementations abcdf_pages joint_union_cold_512 joint_hw_ac_cold_512 joint_integrated_512 joint_integrated_bounds_512 \
  --sorted-topk --requests 32 --mtp 0 1 3 --topk 2048 --capacity 4096 \
  --max-token 262144 --block-size 64 --seed 2026 \
  --scenarios cold_reset full_cache_75pct --overlaps overlap40 \
  --warmup 10 --repeat 50 --regression-repeat 5 \
  --correctness-report /workspace/single_op_test/integrated512_manual/gate.json \
  --output /workspace/single_op_test/integrated512_manual/profile \
  --temp-dir /workspace/single_op_test/integrated512_manual/tmp
```

`full_cache_75pct`是首行缓存构造名称，不表示所有draft行都75%命中。报告实际miss/descriptor数量，不能用名义命中率代替。各MTP展开32/64/128行；逐样本输入相同指五个实现共用相同逻辑输入，不把额外MTP工作量隐藏。这里只测Plan，不推断端到端收益。

## 本地可做的验证

```bash
python3 -B checks/check_phase0_bounds.py
python3 -B checks/check_integrated_plan.py
python3 -B checks/check_integrated_suite.py
python3 -B checks/check_merge_catalog.py
python3 -B -m unittest checks.check_joint_models checks.check_cold_suite \
  checks.check_union_next_runner checks.check_sweep_runner checks.check_simd_api_contract
```

CPU C++模型运行真实控制器和阶段函数，用宿主线程/warp/API桩及UBSan核对语义，不能验证950地址空间、DCache、真实同步、寄存器spill或设备性能。静态事件扫描同样不能证明SIMT指针数据依赖安全。CANN编译、NPU正确性和整合版耗时均以服务器新结果为准。
