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
3. **每次改动必须 commit + push**：owner 的同步链 = Windows PC `git pull` →
   sftp 同步到服务器（**服务器上没有 git**——给 owner 的服务器侧命令一律禁
   git；需要历史版本文件时，给 Windows 侧 `git show <rev>:<path>` 命令由
   owner 提取后传输）。本机（Windows 侧仓库）git 凭据已配置
   （credential.helper store），直接 push。
4. **文件一律 LF**（.gitattributes 已强制；曾因 Windows 中转 CRLF 出事故）。
5. 注释风格：仓库惯例是"为什么"式详注（每个优化/修复带论证），保持。
6. 代码合入后同步更新 `docs/echo_lru_deep_dive.md`（增量修订 + 文末日志表）。

## 必读文档（按序）

1. 本文件
2. `docs/ascend_vec_simt_stack_overflow.md` — 最阴险的设备坑，先读
3. `docs/echo_lru_deep_dive.md` — 技术深读（注意头部 **v2 转型 banner**；
   §5/§6 的 Argtopk 内容是 v1 历史；§9.17-§9.24 最新（§9.20/§9.21 是负结果
   与微架构实测结论；§9.22 是 v2.1 单趟分类，已上板验证；§9.23 是 v2.1-prof
   打点升级，已上板验证——验证记录/频率漂移结论/栅栏读法注记；**§9.24 是
   v2.2 flags 世代号化——已落地待上板验证，必读**）；§10.3 是方向清单）
4. `README.md` — 工程全貌

## 当前状态（2026-10-20，v2.2 落地待验证；最近已验证 = tag `v2.1`）

v2.1 单趟分类 + 平局松弛 + 尾段追加（§9.22）验证通过——**首次 warm×2 + full
三场景全部 ≤ SIMT，full 首破 100**：

| 场景 | SIMT | Echo v2.1 | Echo/SIMT | vs v2.0 |
|---|---:|---:|---:|---:|
| cold_reset | 74.54 | 99.59 | 1.34× | −8.0% |
| warm_75pct | 80.72 | **71.36** | **0.88×** | −8.0% |
| warm_all_hit | 81.54 | **69.33** | **0.85×** | −6.3% |
| full_cache_75pct | 93.23 | **82.12** | **0.88×** | −18.5% |

**v2.2 flags 世代号化（§9.24，commit `284ff71`，NPU 待验证）**：删 OpZeroFlags
+ prologue SyncGrid（§9.23 验证轮实测两槽 13.7-13.9K cyc、占跨度 8.5-11.9%）
——flags 改世代号等值链（hit=gen / spec=gen+1；gen 从 1 起每次调用 +4，窗
{4k+1,4k+2}；0/I32_MAX/陈旧值窗外免疫）。**生产契约变化：workspace 分配时
清零一次**（compare.py 已 empty→zeros）。预期临界路径 −7~10K/场景：full
~76-78 / warm75 ~65-67 / warm_all ~63-65 / cold ~90-92 μs；上板实测 <6K 则
复盘账目。（落地当晚触发 bisheng 后端坑 §7.6.3——等值链在 -O1+ 编译崩溃，
已以 W2 差值判别形态规避，语义不变，见 §9.24 落地后记。）

- 快照链：v2.2 = `284ff71`（未打 tag，验证后打）→ tag `v2.1`（最近已验证）
  → `v2.0`（§9.17-§9.21，含 opt3/t1024 负结果与微架构实测结论）→
  `v1-argtopk-final`（Argtopk 版，full 120.92）
- 分步热点（full，v2.1-prof 新口径 = 绝对戳差分含间隙，§9.23 验证轮）：Step5
  簇 38.7K（MissCumsum 17.8 + SyncGrid 5.4 + SetUpdate 15.5，最大簇）、
  ClassifyEvict 31.1K（含 3 栅栏）、HitMiss 27.8K（gather 吞吐地板，skew 6.8K
  数据相关长尾）、FreeRelease+FreePush 13.3K、prologue 两槽 13.8K（v2.2 删）、
  FifoAvail 4.9K（地板）
- 距 50-60 目标还差 ~25-30%（v2.2 兑现后再算）；精确打点已收口（§9.23），
  剩余大头见 §10.3 新排序

## 下一步优先级（deep-dive §10.3，按序）

**立即待办 = v2.2 上板验证**（§9.24 验证计划）：checks 56 → 双构建 → 四场景
golden（语义等价 ⟹ 逐字段不变）→ profile A/B（基线 full 82.24 / warm75 71.30 /
warm_all 68.96 / cold 96.19）→ 分步对照（prologue 两槽应消失；跨度基线 full
136.0K / warm75 116.2K / warm_all 113.4K / cold 160.2K）。msprof 仍挂起：
L2Cache / ResourceConflictRatio / Memory 三组 task-based 计数器 + 逐核
task_cyc（**勿用 msprof op**，对 torch 负载不适配）。

1. **ClassifyEvict 再削**（full 31.1K 新口径 → 目标 −3-6K）：B3 尾栅栏依赖重审
   （freeIdx 首读者是 FreeRelease）、staging 合并、切片杂项。
2. **Step5 簇**（full 最大簇 38.7K = MissCumsum 17.8 + SyncGrid 5.4 +
   SetUpdate 15.5）：htd 冷 store 簇与 FreeRelease dth 读合并、MissCumsum
   杂项 ~2K、簇间栅栏依赖重审。
3. **HitMiss gather 地板**（27.8K、skew 6.8K 数据相关长尾）：批量发射实验。
4. cold 专项：reset 簇 53.4K（ResetRows 34.0K ≈ 带宽地板，1.6 TB/s 已高
   利用率）。
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
  （多数组别名）；无 64 位运算；Event 计时不可信（用 profiler/分步周期）；
  **两 run 频率可漂移 ~4-5%**（§9.23 验证轮实锤）——决策用 cycles、锚点逐
  场景给、profile_kernel 与 echo_perf_step 背靠背连跑。
- **bisheng 编译坑（§7.6.3，2026-10-21）**：i32 加载值与**运行时形参**直接
  eq/ne 比较在 -O1+ 触发后端 fatal "Copy one register into another with a
  different width"（-O0/与常量比较/非 eq/ne 谓词不触发）——规避 = 差值判别
  （d=v−形参 再比常量）；该错误不落 run/plog、stack dump 被吞，变体二分定位。
- **冷访存微架构（2026-10-19/20 实测，§9.20/§9.21）**：单发冷 GM 触（少量
  在途）串行 ~3-4K/次，冷行 store 同价（非 fire-and-forget）；批量 gather 是
  核级吞吐受限（加线程无用）；栅栏随线程数翻倍（t512 ~3.7K / t1024 ~6-8K）
  且占总时间 ~1/3——**t512 是正确线程数（v1/v2 双证实），减时间只有：减冷触
  总数 / 跨 Op 融合消栅栏**。

## 验证工作流（owner 在服务器执行，你负责给命令）

```bash
git pull
python3 checks/check_echo_kernel_host.py        # CPU 回归 56 项（纯 stdlib+g++）
bash build.sh && ECHO_PERF_STEP=ON bash build.sh
python compare.py --device 0 --warmup 2 --repeat 4 --verify-each-sample  # 四场景 golden
bash profile_kernel.sh --device 0               # kernel 级 P50（与下步背靠背连跑）
python echo_perf_step.py --device 0 --warmup 5 --repeat 20 --matrix results/perf \
    --p50-us cold_reset=96.19,warm_75pct=71.30,warm_all_hit=68.96,full_cache_75pct=82.24
                                                # 分步（v2.1-prof 绝对戳，§9.23）；锚点 = 上一步
                                                # 各场景 P50，逗号分隔"场景=值"逐场景语法（示例值
                                                # 为 v2.1-prof 轮；纯浮点 = 全场景统一旧行为）；
                                                # 自带漂移探测——WARNING 时 μs 仅参考，决策用 cycles
python prof_report.py --matrix results/perf_<场景> --p50-us <该场景P50> \
    --out results/perf_<场景>.html              # 单文件 HTML 报告（可加 --op-summary，浏览器打开）
```

调试旋钮（build.sh 环境变量，产物一律进 `build_v2dbg/`，不覆盖正式构建）：
`ECHO_V2_DEBUG_CUTOFF=1..9`（阶段截断）、`ECHO_V2_KOPT=-O0..-O3`、
`ECHO_PERF_DRAIN=ON`（[v2.1-prof] 打点排空扰动测量：PerfEnd 读 clock 前先
SyncGrid，量"真实边界 vs 记账边界"；**需配 ECHO_PERF_STEP=ON**，每标记 +一道
栅栏价，强扰动结论仅本二进制内有效，§9.23）。
[v2.1] `ECHO_V2_SKIP_REBUILD` 与 `ECHO_V2_RB_*` 系列已随旧 Rebuild 删除
（旋钮结论本就不跨二进制；旧实验见 git 历史与 deep-dive §9.18/§9.22）。
**警告**：每个旋钮=不同二进制（含栈帧大小变化），旋钮实验结论只在本二进制内
有效、跨 commit 不可组合（2026-10-19 的血泪教训，详见 stack-overflow 文档 §5）。

## 仓库速览

```
csrc/echo/          Echo kernel (echo_lru_kernel.h 主体) + plugin + launch
csrc/simt/          SIMT 基线（同事的，一般不动）
compare.py          基准主控（输入构造/golden/校验/计时）
echo_perf_step.py   分步（v2.1-prof 绝对戳 + --matrix CSV）；profile_kernel.sh|py  kernel 级 P50
prof_report.py      分步数据 → 单文件 HTML 报告（对账条/热力图/逐步聚合/op 表，§9.23）
checks/             56 项 CPU 回归（从 kernel.h 文本提取函数 + g++ mock；v2.2 增
                    陈旧/audit 免疫与 HitMiss 覆写用例）
debug_v2_state.py   逐字段 mismatch dump；debug_v2_stepcycles.py  打点取证
                    （[v2.1-prof] 哨兵等值判定，§9.23 遗留项已修）
docs/               deep-dive（主文档）+ 栈溢出全记录
build_t*/ build_perf*/ build_v2dbg/   各构建产物目录（.gitignore）
```
