# SIMT 候选的正确性审计与 kernel 性能采集

`profile_peer_variants.py` 和 `run_peer_suite.sh` 已包含在仓库中，无需另行下载或解压。
使用 `feat/peer-simt-comparison` 分支。历史算法审阅见 `review_peer_129d304.md`；
其非阻塞描述问题已经修正，真实 CANN 编译、NPU 正确性和性能仍需服务器验证。

## 更新、构建和运行

已有工程且位于该分支时：

```bash
git pull --ff-only
PYTHON=python3 JOBS=3 bash build.sh
PYTHON=python3 bash run_peer_suite.sh 1
```

首次克隆：

```bash
git clone --branch feat/peer-simt-comparison --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git peer_simt
cd peer_simt
PYTHON=python3 JOBS=3 bash build.sh
PYTHON=python3 bash run_peer_suite.sh 1
```

末尾 `1` 是逻辑 NPU 卡号，按服务器分配修改。构建和运行使用已能运行本工程的
CANN、torch、torch_npu 环境。运行脚本不锁定旧提交号；采集脚本仍核对当前源码与
共享库的 peer 身份信息，旧库或源码指纹不匹配时拒绝采集。

## 采集内容

默认参照为 `combo_bcde_fused_1024`，原 `10_bcde` 仍参与对照。优化说明见
`BCDE_fused_1024优化基线说明.md`。

默认尺寸 B/T/C/M=`32/2048/4096/262144`。十路为原 BCDE 和九个新增版本：

```text
10_bcde
combo_bcde_1024
combo_bcde_pair_scan
combo_bcde_fused
combo_bcde_fused_1024
peer_00_baseline
peer_01_threads_1024
peer_04_parallel_scan
peer_05_fused_lookup
peer_06_packed_lists
```

四个场景为冷启动、75% 命中、全命中、缓存已满且 75% 命中。

1. `compare.py --verify-each-sample` 先做逐调用正确性短审计，失败则停止。
2. 正式采集在每次调用前恢复相同初始状态，按轮次改变版本执行顺序。
3. 污染输出审计和预热不计入正式样本。每个场景、每版采集 60 次 kernel。
4. 只汇总匹配的 Plan kernel duration，排除状态复制及 tiling 上传任务；核对样本数
   后输出 P50/P95 和相对 `combo_bcde_fused_1024` 的 P50 加速比。

`compare.py` 的 Event/Host 时间包含接口准备和下发间隔，不能代替这里的纯 kernel
统计。状态恢复虽不计入 kernel duration，但会影响缓存条件；这些结果也不能直接
代替持续 decode 或 MTP 的业务端到端时延。

## 结果位置

默认在 `/dev/shm/peer_simt_profile/session_*`。可指定磁盘目录：

```bash
PYTHON=python3 PEER_PROFILE_OUTPUT=/path/to/peer_profile_results \
  bash run_peer_suite.sh 1
```

保留并发回：

- session 下的 `short_audit.log`、`profile.log`；
- capture 下的 `capture_metadata.json`；
- 各场景的 `kernel_summary.csv`。

metadata 记录实际 Git 提交、分支、工作区状态、源码身份、共享库哈希、环境版本和
输入指纹。若 CSV 汇总失败，应同时保留原始 `kernel_details.csv` 与完整日志。

## 只测五个 BCDE 版本

```bash
python3 -X faulthandler profile_peer_variants.py \
  --device 1 --scenario all \
  --rows 32 --topk 2048 --capacity 4096 --max-token 262144 \
  --implementations 10_bcde combo_bcde_1024 combo_bcde_pair_scan \
                    combo_bcde_fused combo_bcde_fused_1024 \
  --reference combo_bcde_fused_1024 --warmup 13 --repeat 60 \
  --output /dev/shm/peer_bcde_profile
```

先用相同场景比较线程数、pair Scan 和融合查询，再看暖路径和冷路径的差异。
同事独立版本应同时与 `peer_00_baseline` 比较。不要把不同优化的加速比相乘。
这个脚本只采集 kernel 时间，没有核内周期插桩或硬件 stall 指标。

## 本地检查边界

本地可运行 `checks/check_peer_profiling.py` 验证 CSV 解析、版本名称区分、样本校验和
shell 调用顺序。这些检查使用 CPU 数据或桩程序，不代表真实 NPU profiler 采集成功。
