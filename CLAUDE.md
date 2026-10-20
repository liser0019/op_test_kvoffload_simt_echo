# CLAUDE.md — 项目协作指引（接手 AI 必读）

> 本文件是给接手本项目的 AI（你）的操作指引。写于 2026-10-19（v2.0 收口日）。

## 这个项目是什么

KV offload 元数据算子性能攻关：在 Ascend 950（dav-3510，纯 SIMT 后端，bisheng
`-xasc`，CANN 9.1）上优化 **Echo LRU kernel**（本仓 `csrc/echo/`），与同事维护的
SIMT 基线（`csrc/simt/`，同仓对照）竞争。

- **目标：kernel 级 P50 绝对值 50-60 μs**（参照同事改进版 SIMT）
- 生产形状锁定：batch(rows)=32、mtp=8（数据生成暂不支持 mtp，先不管）、
  topk=2048、capacity=4096、maxLength=256k
- 四基准场景：cold_reset / warm_75pct / warm_all_hit / full_cache_75pct。
  **warm_75pct 是请求前 ~5 步瞬态，full_cache 是长解码稳态 = 主战场**
  （deep-dive §1.6）

## 协作规则（必须遵守）

1. **你是主领导**：用 sub-agent 写代码 / 维护文档，主会话出设计规格与审查。
   重大方向先与 owner 对齐再动手。
2. **开发机（本机）不运行任何代码**：不 build、不跑程序（计算资源需申请、
   无环境）。所有验证由 owner 在目标服务器执行——**你写好命令给 owner，
   结果由 owner 贴回**。曾允许本机跑 checks，2026-10-19 起废止。
3. **每次改动必须 commit + push**：owner 的同步链是服务器 `git pull`。
   git 凭据已配置（credential.helper store），直接 push。
4. **文件一律 LF**（.gitattributes 已强制；曾因 Windows 中转 CRLF 出事故）。
5. 注释风格：仓库惯例是"为什么"式详注（每个优化/修复带论证），保持。
6. 代码合入后同步更新 `docs/echo_lru_deep_dive.md`（增量修订 + 文末日志表）。

## 必读文档（按序）

1. 本文件
2. `docs/ascend_vec_simt_stack_overflow.md` — 最阴险的设备坑，先读
3. `docs/echo_lru_deep_dive.md` — 技术深读（注意头部 **v2 转型 banner**；
   §5/§6 的 Argtopk 内容是 v1 历史；§9.17-§9.22 最新（§9.20/§9.21 是负结果
   与微架构实测结论；**§9.22 是 v2.1 单趟分类——已实现待上板验证，必读**）；
   §10.3 是方向清单）
4. `README.md` — 工程全貌

## 当前状态（2026-10-20，tag `v2.1`，上板全过）

v2.1 单趟分类 + 平局松弛 + 尾段追加（§9.22）验证通过——**首次 warm×2 + full
三场景全部 ≤ SIMT，full 首破 100**：

| 场景 | SIMT | Echo v2.1 | Echo/SIMT | vs v2.0 |
|---|---:|---:|---:|---:|
| cold_reset | 74.54 | 99.59 | 1.34× | −8.0% |
| warm_75pct | 80.72 | **71.36** | **0.88×** | −8.0% |
| warm_all_hit | 81.54 | **69.33** | **0.85×** | −6.3% |
| full_cache_75pct | 93.23 | **82.12** | **0.88×** | −18.5% |

- 快照链：tag `v2.1`（现行）→ `v2.0`（§9.17-§9.21，含 opt3/t1024 负结果与
  微架构实测结论）→ `v1-argtopk-final`（Argtopk 版，full 120.92）
- 分步热点（full 口径）：ClassifyEvict 29K（含 3 栅栏 ~11K）、HitMiss 24K
  （gather 吞吐地板）、MissCumsum 16.4K（45% 栅栏）、SetUpdate 14.2K、
  FreeRelease 7.1K、FifoAvail ~6.4K（地板）、SyncGrid 槽 ~15K
- 距 50-60 目标还差 ~25-30%；剩余大头逐项变贵——先精确打点（§10.3①）再动刀

## 下一步优先级（deep-dive §10.3，按序）

1. **精确打点（专家意见 2026-10-20）**：①msprof op 一轮（torch_npu profiler
   只有 kernel 级时长；msprof op 另给 Current/Rated Freq、逐核 aiv_time、
   PipeUtilization、Memory/UB 带宽、L2、Bank Conflict——skill:
   ops-profiling/msprof-op-guide，注意它**不含** kernel 内函数级拆解）；
   ②kernel 内时间戳升级（每核绝对时间戳→核间 skew；可选排空模式开关）；
   ③echo_perf_step.py 标定修正（--p50-us 外部锚点替代 Event 假频率）。
2. **ClassifyEvict 再削**（full 29K → 目标 −3-6K）：B3 尾栅栏依赖重审
   （freeIdx 首读者是 FreeRelease）、staging 合并、切片杂项。
3. **SetUpdate/FreeRelease/HitMiss 收尾**（各 −2-4K）：htd 冷 store 簇与
   dth 读合并、HitMiss 批量发射实验。
4. cold 专项：ResetRows 32.6K ≈ 带宽地板（1.6 TB/s 已高利用率）。
5. spec 场景测试（golden 未覆盖，需先扩 reference）。
6. 混合范式预研（专家意见，parked——纯 SIMT 收尾后评估）；远期：fifo/avail
   合并单数组。

## 设备坑速查（详见 stack-overflow 文档与 deep-dive §7）

- **507035（device error type 3）→ 第一步**：
  `ASCEND_SLOG_PRINT_TO_STDOUT=1` 重跑 + `grep errcode`。
  errcode **354 = 栈溢出**：重型设备函数必须 `__attribute__((noinline))`
  （全部 Op 已 noinline，新 Op 照做）；"位置随源码漂移 + 全核同 PC + -O0 不救"
  是它的指纹，出现即停止语义排查。
- 507034 = 超时：先 `npu-smi info` 查占用/换卡，再归因代码。
- GM 跨线程写后读要 dcci 三明治；同地址 atomic ~9K cyc；UB 必须单大数组
  （多数组别名）；无 64 位运算；Event 计时不可信（用 profiler/分步周期）。
- **冷访存微架构（2026-10-19/20 实测，§9.20/§9.21）**：单发冷 GM 触（少量
  在途）串行 ~3-4K/次，冷行 store 同价（非 fire-and-forget）；批量 gather 是
  核级吞吐受限（加线程无用）；栅栏随线程数翻倍（t512 ~3.7K / t1024 ~6-8K）
  且占总时间 ~1/3——**t512 是正确线程数（v1/v2 双证实），减时间只有：减冷触
  总数 / 跨 Op 融合消栅栏**。

## 验证工作流（owner 在服务器执行，你负责给命令）

```bash
git pull
python3 checks/check_echo_kernel_host.py        # CPU 回归 54 项（纯 stdlib+g++）
bash build.sh && ECHO_PERF_STEP=ON bash build.sh
python compare.py --device 0 --warmup 2 --repeat 4 --verify-each-sample  # 四场景 golden
bash profile_kernel.sh --device 0               # kernel 级 P50
python echo_perf_step.py --device 0 --warmup 5 --repeat 20              # 分步（只看周期，μs 换算失真）
```

调试旋钮（build.sh 环境变量，产物一律进 `build_v2dbg/`，不覆盖正式构建）：
`ECHO_V2_DEBUG_CUTOFF=1..9`（阶段截断）、`ECHO_V2_KOPT=-O0..-O3`。
[v2.1] `ECHO_V2_SKIP_REBUILD` 与 `ECHO_V2_RB_*` 系列已随旧 Rebuild 删除
（旋钮结论本就不跨二进制；旧实验见 git 历史与 deep-dive §9.18/§9.22）。
**警告**：每个旋钮=不同二进制（含栈帧大小变化），旋钮实验结论只在本二进制内
有效、跨 commit 不可组合（2026-10-19 的血泪教训，详见 stack-overflow 文档 §5）。

## 仓库速览

```
csrc/echo/          Echo kernel (echo_lru_kernel.h 主体) + plugin + launch
csrc/simt/          SIMT 基线（同事的，一般不动）
compare.py          基准主控（输入构造/golden/校验/计时）
echo_perf_step.py   分步周期；profile_kernel.sh|py  kernel 级 P50
checks/             54 项 CPU 回归（从 kernel.h 文本提取函数 + g++ mock）
debug_v2_state.py   逐字段 mismatch dump；debug_v2_stepcycles.py  打点取证
docs/               deep-dive（主文档）+ 栈溢出全记录
build_t*/ build_perf*/ build_v2dbg/   各构建产物目录（.gitignore）
```
