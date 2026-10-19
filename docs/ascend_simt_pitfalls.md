# Ascend SIMT（dav-3510 / bisheng / CANN 9.1）编程避坑手册

> **定位**：本项目（kvoffload_simt_echo，纯 SIMT 后端 dav-3510、bisheng `-xasc`、
> CANN 9.1.0、torch_npu）踩坑实录的独立汇总。每条坑给出：**现象 / 触发条件 /
> 限制与根因 / 规避方法 / 证据**。算法与流水线的深度解读见
> `docs/echo_lru_deep_dive.md`（其 §7 为本手册的早期子集，本手册是超集与修订）。
>
> 收录原则：只收录有实测证据或代码内注释佐证的坑；推测性内容明确标注。
> 最后更新：2026-10-19（v2 维护序重构攻关日，P1 当天从"完全神秘"到"根因实锤"）。

---

## 目录

- 一、编译单元与栈
  - [P1 VEC SIMT 栈溢出（errcode 354）——最阴险的一条](#p1)
  - [P2 巨型编译单元的整体脆弱性（grid.sync trap）](#p2)
  - [P3 bisheng 前端：短路表达式里的文本 UB 被投机求值](#p3)
  - [P4 bisheng 前端：while(true)+数据依赖 continue 跨栅栏段错误](#p4)
- 二、内存模型
  - [P5 GM 写后读命中 stale D-cache 行（必须 dcci 三明治）](#p5)
  - [P6 同地址 GM atomic 串行化（~9K 周期/次）](#p6)
  - [P7 `__ubuf__` 多数组别名互覆 + 单数组容量上限](#p7)
  - [P8 无 64 位值运算](#p8)
- 三、同步与性能
  - [P9 块内栅栏本身是大成本（4~8K 周期）](#p9)
  - [P10 t0 串行大循环无法隐藏访存延迟](#p10)
  - [P11 Event 计时含 ~百 μs 固定开销](#p11)
- 四、环境与运维
  - [P12 设备被他人占用/残留进程：新进程第一次 sync 即报错](#p12)
  - [P13 507034 vs 507035：先读 errcode 字符串再推理](#p13)
  - [P14 trap 后设备上下文毒化（workspace 读回失败）](#p14)
  - [P15 行尾 CRLF 经 Windows 中转污染脚本](#p15)
- 五、调试方法论（用一下午弯路换来的元教训）
  - [M1 编译期旋钮的混杂性：每个旋钮同时改语义与二进制](#m1)
  - [M2 阶段二分对"入口分配类"问题完全失效](#m2)
  - [M3 对照实验必须同源码树（schema 错位教训）](#m3)
  - [M4 构建系统事故模式（CMake cache / 产物目录路由 / 脚本拼接）](#m4)
  - [M5 CPU mock 的能力边界](#m5)
- 附录：诊断命令速查

---

<a name="p1"></a>
## P1. VEC SIMT 栈溢出（errcode 354）——最阴险的一条

**现象**：kernel 一启动就 trap（`npuSynchronizeDevice` 报 507035 "device error
type 3"），**全核同时、同一 PC、100% 确定性复现**；换卡无效；`-O0` 无效；
任何"语义级"修复（删循环、补栅栏、刷 dcci）全部无效；且 trap 的"位置"随源码
任意增删而**漂移**——同一截断点（如"Step4 后返回"）在 commit A 上 clean、在
commit B 上 trap。

**触发条件**：kernel 入口函数（`__global__`）把大量 `__aicore__ inline` 设备函数
全部内联成**单一巨型栈帧**，帧大小（= 所有内联函数的局部变量与寄存器溢出槽之
和）**超过设备向量核每线程栈的硬上限**。帧在 kernel 入口一次性分配，**与哪段
代码实际执行无关**——这就是"删除一段从不执行的死代码也能让 trap 消失/出现"的
原因（增删代码在微调帧大小跨限）。

**限制与根因**：
- 每线程栈上限是硬件常数（具体值未公开；本项目实测：v1 形态的 Process 帧
  恰好在限下，v2 增加 2 个内联 Op 即越界；普通构建擦线通过后，打点构建
  多几行代码再次越界——余量极小）。
- bisheng 对 `inline` + `-O3` 的内联非常激进；`-O0` 并不能缩小帧（每个局部
  都占栈槽，往往更大）。
- 错误码与文案（来自设备日志，见 P13 的获取方法）：
  `errcode:(354) errorStr: The VEC SIMT stack overflows. Possible cause:
  The local variable is too large or there are too many local variables.`

**为什么它如此迷惑**：
1. 上层只报 507035（不透出 354 与文案），与其它设备错误无法区分；
2. 表现出"位置漂移"——任何对二进制的扰动都可能让它消失或出现，酷似
   codegen bug、时序 race、对齐问题，诱发大量无效假设；
3. 一切运行时语义手段（栅栏/dcci/删代码）都会"偶尔生效"——其实是帧大小
   副作用，且结论不可跨 commit 组合（见 M1/M2）。

**规避方法**：
- **重型设备函数一律 `__attribute__((noinline))`**，让每个 Op 拥有独立栈帧，
  帧峰值从"全部求和"降为"取最大"（本项目规则：所有 Op 函数 noinline；
  warp 协作小函数如 PackedBlockScan 保持 inline，并入已很小的宿主帧）。
- 新增内联大函数/大局部数组前，先估算帧增量。
- **诊断流程**：一旦出现确定性 507035，第一件事开 `ASCEND_SLOG_PRINT_TO_STDOUT=1`
  重跑，grep `errcode`——354 的文案直接点名栈溢出，不要先做任何语义假设。

**证据**：2026-10-19 v2 攻关全程（deep-dive §9.18 待记）；打点构建复现
（fe10f91 前一日）为"余量极小"的直接实证；§7.9 的 grid.sync trap 高度疑似
同根因（加 grid.sync 代码涨帧越界），见 P2。

---

<a name="p2"></a>
## P2. 巨型编译单元的整体脆弱性（grid.sync trap）

**现象**：极简 probe kernel 里 `grid_group::sync()` 全 PASS；但同样的调用放进
echo_lru 的 `Process()`（巨型函数、多 Op、多 GM 指针）立即 trap（输出全 0 不
hang）。换任何包装方式（inline/template/成员函数）无效。

**根因（结合 P1 重新认识）**：当时归因"编译单元整体属性"。现在有强理由怀疑
**就是 P1**——Process 加一个 grid.sync 调用即增大帧/改变栈布局而越限，probe
（小帧）复现不了。原结论"在可预见的将来不要在本 kernel 里尝试 grid.sync"
仍然有效，但攻关方向应改为"先解决帧问题再重试"。

**规避**：保持 Path A（每核独立行段，无跨核同步）设计；若必须 grid.sync，
先按 P1 把帧压到远低于上限。

**证据**：`echo_lru_kernel.h` SyncGrid 注释（2026-09-16 记录）；上游
`GRID_GROUP_PITFALLS.md`（不在本仓）。

---

<a name="p3"></a>
## P3. bisheng 前端：短路表达式里的文本 UB 被投机求值

**现象**：`pass == 0 || (v >> (shift+8)) == 0` 在 pass==0 时逻辑上短路，但
`shift+8 == 32` 使 `v >> 32` 成为**文本可见的 UB**——bisheng SIMT 后端可能对其
投机求值，段错误。

**根因**：C++ 标准允许编译器对"任何执行路径上都 UB"的表达式做任意处理；
`||` 的短路是运行时语义，挡不住前端的文本级 UB 分析/变换。

**规避**：把短路改写成 if/else 分支，让 UB 表达式**根本不出现在该路径的
源代码文本里**。凡是移位量可能到 32 的地方（`1 << 32`、`v >> 32` 同理）都要检查。

**证据**：`echo_lru_kernel.h` Argtopk 时代的规避注释（commit 145d3ce）；
deep-dive §7.6-1。

---

<a name="p4"></a>
## P4. bisheng 前端：while(true)+数据依赖 continue 跨栅栏段错误

**现象**：`while(true) { ...; if (cond) continue; ... }` 其中 continue 跳过
栅栏结构、且跳转目标依赖数据——bisheng 前端收敛分析直接崩溃（编译期段错误）。

**根因**：前端对"跨栅栏结构的数据依赖跳转"无法收敛（具体内部机制未知，
实测复现）。

**规避**：有界 `for (attempt = 0; attempt < N; ++attempt)` + 迭代边界 break，
与已验证可编译的结构同构。

**证据**：commit 145d3ce；deep-dive §7.6-2。

---

<a name="p5"></a>
## P5. GM 写后读命中 stale D-cache 行（必须 dcci 三明治）

**现象**：单核内线程 A 写 GM 地址 X、`asc_syncthreads()` 后线程 B 读 X，可能
读到旧值。变体：t0 的 GM 写滞留在其私有缓冲（写尚未落地 GM），t0 再读自己见
新值、他人读 GM 见旧值——曾导致 fifo 不一致 → 下游 startPass 分歧 → 栅栏
发散挂死（v1 时代 OpFifoInc 挪位事件）。

**根因**：dav-3510 单核 D-cache 写回不及时；`asc_syncthreads` 只排序执行、
**不保证写落地/失效**。

**规避**：跨线程 GM 写后读一律走三明治：
```cpp
asc_syncthreads();
if (threadIdx.x == 0) asc_dcci_entire(wsBase);   // 整核生效, t0 做一次即可
asc_syncthreads();
```
注意：同核 WAW（先清零后写同地址）同样需要 dcci 排空写缓冲，否则后到的
旧写可能反超覆盖新写（v2 prologue flags 清零的设计依据）。**kernel 边界**
（跨调用、host 读回）由 runtime 保证可见，无需 dcci。

**证据**：deep-dive §3.5/§7.2；v1 fifo 事件（§9.12）；v2 prologue 注释。

---

<a name="p6"></a>
## P6. 同地址 GM atomic 串行化（~9K 周期/次）

**现象**：每线程一次 `asc_atomic_add` 打同一 GM 地址，1024 线程时该步实测
~9.4M 周期；耗时与有效工作量无关、与 atomic 次数强相关。

**根因**：同地址 atomic 在硬件上串行执行，单次 ~9K 周期。

**规避**：块内计数一律走 warp 归约（`asc_reduce_add`）/打包扫描
（PackedBlockScan）+ t0 单写。CPU 回归可断言 `atomicCalls == 0` 防回归。

**证据**：deep-dive §7.1（P0 优化）；v2 checks 保留零 atomic 断言。

---

<a name="p7"></a>
## P7. `__ubuf__` 多数组别名互覆 + 单数组容量上限

**现象**：声明多个独立 `__ubuf__` 数组会**别名互相覆盖**（读 A 得 D）；
单一大数组可到 192 KiB（256 KiB 才 trap）。

**规避**：全部 UB 状态塞进**单个** `__ubuf__` 数组手动分段（本项目
`argtopkUb[ARGTOPK_UB_ELEMS]`，128 KiB）；分段偏移集中在 Init 一处管理；
分段基址建议对齐到 256 元素（1 KiB）——防御性实践（曾作为 v2 排查的
候选根因之一，后被 P1 取代解释，保留无害）。

**证据**：`probe_ub_size` 上游探针；deep-dive §7.5；v2 Init 注释。

---

<a name="p8"></a>
## P8. 无 64 位值运算

**现象/限制**：dav-3510 设备侧没有 64 位值运算。热路径全部用 32 位；仅有的
uint64_t（打点 clock() 差值）必须立即 clamp 到 int32 存储。

**规避**：写设备代码不引入 64 位算术路径；索引/计数一律
uint32_t/int32_t，形状乘积在 host 侧校验不超 int32。

**证据**：deep-dive §7.7；compare.py 的形状校验。

---

<a name="p9"></a>
## P9. 块内栅栏本身是大成本（4~8K 周期）

**现象**：`asc_syncthreads`（1024 线程）约 4~8K 周期（512 线程更低），随线程
数伸缩。栅栏次数是首要优化目标之一。

**规避**：能 warp 内解决的不上块栅栏；每个栅栏要么给出保留论证（RAW 依赖
分析），要么删（本项目历轮删栅栏全部附论证，见 deep-dive §4/§5/§9）；
线程数消融显示"更便宜的栅栏 vs 更多扫列轮数"存在甜点（topk=2048 → 512
线程）。

**证据**：deep-dive §7.3/§9.10；v2 rebuild 的 5 栅栏/行设计。

---

<a name="p10"></a>
## P10. t0 串行大循环无法隐藏访存延迟

**现象**：单线程串行扫数千元素（依赖 load 链）时延迟全额暴露（v1 的 Argtopk
emission 曾 ~2.1M 周期）。

**规避**：全员参与 + 连续切片 + 打包扫描（注意：**切片必须连续**——
`lo = t*chunk`，grid-stride 会得到"线程主序"，前缀/输出顺序全错）。

**证据**：deep-dive §7.4/§6.6；v2 EvictScan/Rebuild 的连续切片。

---

<a name="p11"></a>
## P11. Event 计时含 ~百 μs 固定开销

**现象**：Event 口径含 tiling H2D（pageable 小拷贝）与 host 下发空隙，各
~百 μs；且在共享/受扰环境下会放大到数百 μs（实测同机不同时段差异巨大），
甚至把频率标定（Σ步周期/Event）打到 100-400 MHz 的荒谬值。

**规避**：kernel 纯耗时用 `torch_npu.profiler`（profile_kernel.sh，
kernel_details.csv 的 P50）；分步周期用 ECHO_PERF_STEP 打点构建；Event
比值可用、绝对值不可用；打点构建的 μs 换算以 kernel P50 反推频率为准。

**证据**：deep-dive §7.8/§9.14-9.15 两个 open question。

---

<a name="p12"></a>
## P12. 设备被他人占用/残留进程：新进程第一次 sync 即报错

**现象**：设备被他人占满时，新进程**第一次** `torch.npu.synchronize()`
（甚至早于任何 kernel 调用，如张量上传后的 sync）就报 507034/507035。
曾被误判为"代码挂死"浪费一轮。

**规避**：任何超时/挂死，先 `npu-smi info` 查占用与残留进程、换卡或复位，
**再**归因代码。注意 npu-smi 显示 OK 不代表没被占用（利用率/进程列表要看）。

**证据**：deep-dive §7.10；2026-10-19 事件复核。

---

<a name="p13"></a>
## P13. 507034 vs 507035：先读 errcode 字符串再推理

**内容**：
- **507034**：执行超时（挂死/占用）。
- **507035**：设备错误（trap）。**必须进一步读设备日志的错误码字符串**——
  这是唯一能区分"栈溢出(354)/向量单元错/访存错"等类别的入口：
  ```bash
  ASCEND_SLOG_PRINT_TO_STDOUT=1 python xxx.py > /tmp/full.log 2>&1
  grep -n -iE "errcode|aivec|pc start" /tmp/full.log | head
  ```
  plog 文件在 `/root/ascend/log/run/plog/<plog-<PID>_<时间>.log`（文件名含
  python 进程 PID），但 507035 的 errcode 文案**只在 SLOG 直通或特定通道**
  出现，裸跑只给上层 507035。
- 我们付出一下午的代价才拿到这行字符串（P1 全过程）——**任何 507035 先做
  这一步**。

**证据**：2026-10-17（507034 误判）与 2026-10-19（507035 拿到 354）。

---

<a name="p14"></a>
## P14. trap 后设备上下文毒化（workspace 读回失败）

**现象**：kernel trap 后，进程内对该设备的一切后续操作（含把 workspace
张量 `.cpu()` 读回）都报同样的设备错误——**无法在 trap 后读打点数据取证**。

**含义**：指望"trap 后读 stepCycles 定位步数"的取证路径不通（已实测）；
定位必须靠 P13 的错误码/PC，或靠构建变体。

**证据**：`debug_v2_stepcycles.py` 的读回失败记录（2026-10-19）。

---

<a name="p15"></a>
## P15. 行尾 CRLF 经 Windows 中转污染脚本

**现象**：仓库经 Windows 机器 clone/pull 后再 scp/sftp 到服务器，`build.sh`
的 `set -o pipefail`、CMake 解析对 CRLF 敏感而报错。

**规避**：仓库根 `.gitattributes` 强制 `* text=auto eol=lf`（本仓已有）；
写文件一律 LF。

**证据**：仓库 `.gitattributes`；2026-10-16 会话记录。

---

## 五、调试方法论

<a name="m1"></a>
### M1. 编译期旋钮的混杂性

每个 `#ifdef` 调试旋钮**同时改变运行时语义与编译产物（含帧大小）**。用旋钮
做"语义二分"时，若真因是帧大小（P1），结论会系统性误导且**跨 commit 不可
组合**（今天：同一截断点在 commit A clean、commit B trap）。教训：旋钮实验
的结论只在**同一二进制**内有效；跨 commit 推理前先确认"真因是否依赖二进制
形态"。

<a name="m2"></a>
### M2. 阶段二分的适用边界

阶段截断（cutoff）二分的前提是"病灶在执行流里"。对**入口一次性分配类**
问题（栈帧、资源预留）完全失效——截断只改"执行到哪里"，帧早已分配完。
判别特征：病灶表现随任意源码增删漂移、与执行路径无关、确定性复现。

<a name="m3"></a>
### M3. 对照实验必须同源码树

用 v2 的 compare.py 跑 v1 的 .so：schema 张量名已从 pri 改 lru，位置传参
"成功"但语义错位——v1 把 lru 内容当 pri 时间戳用，golden 100% mismatch，
险些误读为"v1 也坏了/设备坏了"。对照实验（尤其跨版本）必须整树 checkout。

<a name="m4"></a>
### M4. 构建系统事故模式

- **CMake cache**：`-D` 覆盖 cache 值，但调试旋钮一旦忘记透传就沿用旧值——
  build.sh 应**每次显式传递全部旋钮**（本仓已如此）。
- **产物目录路由**：`ECHO_V2_DEBUG_CUTOFF=0` 语义是"正式构建"（产物进
  build/），"完整跑但放调试目录"应使用 7——一次误用导致旧二进制复跑、
  结论作废。帮助文本必须把 0 与"其他值"的产物路径差异写明。
- **脚本字符串拼接**：用程序化方式改 shell 脚本后必须 `bash -n` 语法校验
  （一次 `]]; then` 被截成 `]]; t ||` 的事故）。

<a name="m5"></a>
### M5. CPU mock 的能力边界

g++/std::thread 的 CPU 回归能覆盖：算法逻辑、索引边界、栅栏计数、语义
等价性（本仓 50 项 checks）。**不能覆盖**：编译器特有行为（P3/P4）、硬件
限制（P1/P7/P8）、缓存一致性（P5）、时序（P6 之外的竞争）。CPU 全绿 ≠
设备能跑——本仓工作流是 checks → 服务器编译 → 服务器 audit 三级。

---

## 附录：诊断命令速查

```bash
# 507035 类设备错误: 第一时间拿 errcode 字符串 (P13)
ASCEND_SLOG_PRINT_TO_STDOUT=1 python compare.py --device 0 ... > /tmp/full.log 2>&1
grep -n -iE "errcode|aivec|pc start|vec error|mte error" /tmp/full.log | head

# plog (按 python PID 找)
ls -t /root/ascend/log/run/plog/ | head -3

# 设备占用/健康
npu-smi info

# kernel 级性能 (Event 不可靠, P11)
bash profile_kernel.sh --device 0

# 分步周期 (打点构建; trap 后读不回, P14)
ECHO_PERF_STEP=ON bash build.sh && python echo_perf_step.py --device 0

# 逐字段状态/输出比对 (audit 中断时)
python debug_v2_state.py --device 0 --scenario cold_reset --library build/libsimt_vs_echo_ops.so
```

---

## 与 deep-dive 的关系

本手册是 `docs/echo_lru_deep_dive.md` §7 的**超集与修订**：P1/M1/M2/M3 为
2026-10-19 新增（deep-dive 暂停期间独立成文）；P2 补充了 P1 视角的重新解释；
其余条目与 §7.1-7.10 对应并统一编号。deep-dive 恢复更新时将在 §7 引用本手册
并去重。
