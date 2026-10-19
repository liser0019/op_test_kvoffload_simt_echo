# Ascend VEC SIMT 栈溢出（errcode 354 / 上层 507035）问题全记录

> **一句话结论**：bisheng 把 `__global__` 入口函数的全部内联设备代码编成一个
> 巨型栈帧（帧大小 = 所有内联函数的局部变量与寄存器溢出槽之和），**在 kernel
> 入口一次性分配**；一旦超过向量核每线程栈的硬件上限，kernel 启动即 trap。
> 它与哪段代码实际执行**无关**，因此一切"运行时语义"手段（删循环/补栅栏/刷
> dcci/换优化档）全部无效，且 trap 位置随任意源码增删**漂移**——是本项目迄今
> 最具迷惑性的设备问题。
>
> 记录时间：2026-10-19（v2 维护序重构上板攻关日）。历史环境坑（dcci/atomic/
> UB 别名等）见 `docs/echo_lru_deep_dive.md` §7，本文不重复。

---

## 1. 症状特征（怎么在第一时间认出它）

| 特征 | 本案实测 |
|---|---|
| 上层报错 | `npuSynchronizeDevice ... device error type 3, error code is 507035`（trap 类，区别于 507034 超时） |
| 复现性 | 100% 确定性（同二进制连跑 3 次全炸） |
| 范围 | **全部核同时**、同一个 PC（`pc start`/`current` 全核一致） |
| 换卡 | 无效（设备 0/2 都炸，npu-smi 健康） |
| 优化级别 | `-O0` 照样炸（甚至更糟——每个局部都占栈槽） |
| 语义修复 | 删代码、补栅栏、dcci、缓冲分离……全部无效或"偶尔有效" |
| **最独特** | **trap 的"位置"随源码任意增删漂移**：同一截断点在 commit A 上 clean、commit B 上 trap |

## 2. 触发条件（精确描述）

1. 入口函数（`__global__ __launch_bounds__(N)`）直接调用大量 `__aicore__
   inline` 设备函数，bisheng（`-xasc`，`-O3`）把它们**全部内联**进入口帧；
2. 内联后帧大小（局部变量 + 寄存器溢出槽总和）**超过每线程栈硬上限**；
3. 帧在 **kernel 入口一次性分配**（prologue 一次性扣栈），此后不再增长——
   所以 trap 与执行路径无关：哪怕后续代码一步都不执行（调试截断），帧已经
   越界。

本案的边界实证：v1 形态的 Process（约 19 个内联 Op）**恰好**在限下；v2 增加
EvictScan/LruRebuild 两个内联 Op 后越界；把这两个 Op 改 `noinline` 后普通
构建恢复，但 **ECHO_PERF_STEP 打点构建**（只多几行 PerfBegin/End 代码）
**再次越界**——说明修复前余量以"行"计。全部 Op 统一 noinline 后余量充足。

## 3. 限制条件（硬件/工具链性质）

- 每线程栈上限是硬件常数，具体值未公开；本项目只能给出相对刻度
  （v1 帧以内、v1+2 个中等 Op 以外）。
- bisheng 对 `inline` 的内联非常激进；`-O0` 不能缩小帧（见 §1）。
- 错误信息**不在上层异常里**：上层只见 507035；错误码 354 与文案
  （`The VEC SIMT stack overflows. Possible cause: The local variable is
  too large or there are too many local variables.`）只在设备日志里，
  必须开 SLOG 直通才能看到（见 §6 诊断流程）。

## 4. 根因机理（为什么）

- SIMT 后端每个线程有私有栈，函数调用的局部/溢出住在栈上；
- 全内联 ⇒ 只有一个"调用"，所有函数的栈需求**求和**进同一帧；
- noinline ⇒ 每次调用独立帧，峰值 = **max(各帧)** 而非 sum；
- 入口一次性分配 ⇒ 帧大小是**编译期常量**、**执行期不变量**——这是"阶段
  二分失效"的根源（截断只改执行到哪里，帧早已分配完）；
- "增删任意代码影响 trap" ⇒ 任何源码扰动都在微调帧大小，跨过/退回上限即
  改变现象——这使得每个"语义实验"的结论都不可跨 commit 组合。

## 5. 排查复盘：完整的误判链（为什么浪费了一下午）

按时间序，每个假设 → 被什么实验否定 → 真相：

| # | 假设 | 实验 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | 某阶段语义 bug（越界/竞争） | 阶段截断二分（cutoff 1-8） | 边界"锁定"MissCumsum，但**同一边界后来随 commit 漂移** | 误判（截断改变帧大小） |
| 2 | UB 段对齐（v2 用了 4B 对齐的奇数偏移） | 段基址改 256 元素对齐 | "cutoff=6 变 clean" | **巧合**（对齐改动同时微调了帧），后被推翻 |
| 3 | PackedBlockScan 出现在新调用点的 codegen | rebuild 扫描换 t0 串行 | 仍 trap | 误判 |
| 4 | 扫描缓冲跨 Op 同址复用 | 独立扫描段 | 仍 trap | 误判 |
| 5 | 尾零循环（升序可合并 GM 写）触发 | 删尾零（数学冗余，保留删除）/全行写 1/寄存器零 | 删了"变 clean"，但换 commit 后又 trap | 部分巧合 + 误判；"值无关/位置漂移"其实已是否定信号 |
| 6 | 优化器 codegen | 同源码 `-O0` | 仍 trap | 误判（-O0 帧更大） |
| 7 | 设备状态劣化 | v1 对照 | v1 跑通（无 trap） | 排除设备 ✓ |
| 8 | **读设备日志的错误码字符串** | `ASCEND_SLOG_PRINT_TO_STDOUT=1` | **errcode 354 "VEC SIMT stack overflow"** | **实锤** |

**三条血的教训**：
1. **507035 的第一步永远是读 errcode 字符串**（上层异常不透出它；见 §6）。
   我们拿到那行文案只用了一条命令，而它之前的一切推理都是盲猜。
2. **"位置随源码漂移 + 与执行路径无关 + 全核同 PC"是栈溢出的指纹**——出现
   即停止语义排查。
3. **编译期调试旋钮同时改变语义与帧大小**，用它做的二分结论只在同一二进制
   内有效，跨 commit 不可组合。

## 6. 诊断流程（命令级）

```bash
# ① 拿错误码字符串 (507035 的第一步!)
ASCEND_SLOG_PRINT_TO_STDOUT=1 python compare.py --device 0 ... > /tmp/full.log 2>&1
grep -n -iE "errcode|aivec|pc start|vec error|mte error" /tmp/full.log | head

# 预期看到 (栈溢出的指纹):
#   errcode:(354) errorStr: The VEC SIMT stack overflows. ...
#   全部 core id 的 dump: pc start/current 完全一致

# ② plog 归档 (文件名含 python PID)
ls -t /root/ascend/log/run/plog/ | head -3

# ③ trap 后设备上下文已毒化: 进程内任何后续设备操作(含读回 workspace)都报错
#    —— 不要指望 trap 后读数据取证, 定位只能靠 ① 的错误码/PC 或构建变体
```

## 7. 规避方法（本仓固化后的规则）

1. **重型设备函数一律 `__attribute__((noinline))`**，独立栈帧，峰值取 max：
   ```cpp
   __attribute__((noinline)) __aicore__ void OpLruRebuildRow(...) { ... }
   ```
   本仓现行规则：**全部 Op 函数 noinline**；warp 协作小函数（PackedBlockScan）
   保持 inline 并入宿主帧（宿主帧已很小）。
2. **新增设备函数前的检查项**：会不会被内联进入口帧？局部变量/溢出槽多不多？
   大于一个小函数就 noinline。
3. **别用 `-O0` 当"排除 codegen"的手段**——对帧问题它反而更糟；要排除优化器
   影响，用优化级别扫描（-O1/-O2）而非归零。
4. 调试旋钮（`#ifdef` 截断/跳段）的结论标注"仅本二进制有效"；跨 commit 比较
   前先想"真因是否依赖二进制形态"。

## 8. 证据链索引

- 根因实锤：2026-10-19 17:13 SLOG 日志（errcode 354，全核同 PC
  `0x120052e09770/0x120052e09988`）。
- 修复：commit `fe10f91`（EvictScan/LruRebuild noinline，普通构建恢复）→
  commit `692c1d7`（**全部 Op noinline**——打点构建复现同错误证明余量极小，
  顺带成为根因的最后一项实证）。
- 余量实证：v1（19 内联 Op）在限下；v2（+2 Op）越界；+打点再越界。
- 排查过程的时间线与各实验：git log `8efd0a3..fe10f91`（每个 debug 旋钮
  commit 的 message 记录了当时的假设与判定）。
- 误判链复盘：本文 §5。
- 关联历史：`echo_lru_kernel.h` SyncGrid 注释记录的 grid.sync trap
  （2026-09-16，deep-dive §7.9）——"极简 probe 全 PASS、Process 内必 trap、
  根因为编译单元整体属性"——高度疑似同根因（加 grid.sync 代码涨帧越界），
  建议后续按本文 §7 规则收窄帧后重测。

## 9. 一页速查（贴墙版）

```
507035 (device error type 3)
  └→ 第一步: ASCEND_SLOG_PRINT_TO_STDOUT=1 重跑, grep errcode
       └→ 354 "VEC SIMT stack overflows"
            ├─ 指纹: 全核同 PC / 确定性 / 位置随源码漂移 / -O0 不救
            ├─ 机理: 内联帧和 > 每线程栈上限, 入口一次性分配, 与执行无关
            ├─ 修复: 重型 Op 加 __attribute__((noinline)) (帧和→帧max)
            └─ 禁忌: 语义二分 / 删代码试错 / -O0 排查法
```
