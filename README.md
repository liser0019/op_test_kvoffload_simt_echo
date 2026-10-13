# 新增同事 SIMT 对照与 BCDE 组合

本分支基于 `feat/simt-variant-comparison`，使用同事的 `simt` 分支。
分析和推荐测试命令见 [同事SIMT优化分析与组合测试.md](同事SIMT优化分析与组合测试.md)。
新增源码在 `csrc/peer_variants/`；优先对比 `10_bcde`、`combo_bcde_1024`、
`combo_bcde_pair_scan`。当前只比较 kernel 优化，已移除 Host tiling 缓存及所有包含它的组合。
默认编译与比较全部 22 路。下方原有说明中的 13 路指新增前的范围。

# SIMT 变体与作者新版 Echo：可阅读的性能测试

这是一个可独立构建的性能测试工程，**包含原 SIMT、作者新版 Echo 和 11 个 SIMT 候选的 C++/AscendC 源码**。`compare.py` 把输入生成、状态转换、golden、校验和计时集中在一个文件；`build.sh` 编译全部 13 路，Python 直接加载共享库。

本分支为 `feat/simt-variant-comparison`。新增版本、布局差异、源码阅读顺序及完整上机命令见 [SIMT 变体接入与性能对比](SIMT变体接入与性能对比.md)。默认比较全部 13 路；仅运行原两路时加 `--implementations simt echo`。

本文提供阅读顺序、输入与状态语义、正确性检查和构建运行命令。建议从第 1 节的函数表开始，再按第 9 节阅读设备计算源码。

工程基线来自 `kv_plan_single_op_test` 的 `main`，提取提交为 `f51dc079a21c6e9f708e722d245abbccbb73509e`。原 SIMT 计算保留，host 胶水增加变体参数化。**Echo kernel 已在此基线上应用作者 2026-10-09 的性能补丁**，并补充小尺寸 UB 暂存修复。原来源记录在 `source_manifest.json`；11 个候选来自用户的 `var_simt_ops`，基线为 `cd84b83`，具体提取哈希见 `variants_manifest.json`。不包含旧 AIV 或跳过排序的 Echo 实验实现。

### 本次 Echo 更新

- 每线程先累计本行 hit，再执行一次 atomic add，减少同行计数上的原子操作。
- key/slot 打包改为全员并行；histogram 归并和 scatter 段基址计算改为按桶并行。
- 桶前缀仍由线程 0 计算，保留四趟稳定 radix 排序及原有淘汰语义。
- 小尺寸 `Npad<256` 时，256 个桶的暂存区使用现有 UB 四段 ping-pong 后的空闲区域，防止覆盖源数据。生产尺寸 `Npad=8192` 仍复用 `dstKey`，静态 UB 总量仍为 128 KiB。
- 算子接口、tiling、workspace 公式、SIMT 和 benchmark 的输入、参考算法、状态恢复与计时边界均保留。

修改源码后需要重新运行 `bash build.sh --soc=ascend950`，再运行逐样本审计和正式测量。旧共享库不会自动变成新版。此次更新的 NPU 编译、正确性和性能收益仍需服务器验证。

```text
op_test_kvoffload_simt_echo/
├── compare.py                    # 共同输入 → CPU 参考 → 设备调用 → 校验/计时
├── variants_manifest.json        # 11 个候选的名称、配置及来源哈希
├── SIMT变体接入与性能对比.md     # 多版本接入、上机命令与验证边界
├── build.sh                      # CMake 配置与编译，默认 ascend950
├── CMakeLists.txt                # 显式编译两个 kernel + 两个 plugin
├── csrc/
│   ├── simt/
│   │   ├── sparse_kv_plan_kernel.cpp   # SIMT 八阶段计算、kernel 外壳和 launch
│   │   ├── sparse_kv_plan_launch.h     # tiling 构造、workspace 公式、启动声明
│   │   ├── sparse_kv_plan_config.h     # 2048 线程、warp、UB 常量
│   │   └── sparse_kv_plan_plugin.cpp   # torch.ops 校验、tiling 上传、异步下发
│   ├── variants/                      # 11 个候选：计算头、配置、直调与注册胶水
│   └── echo/
│       ├── echo_lru_kernel.h           # Echo 全部设备计算、radix 排序
│       ├── echo_lru_launch.cpp         # pure SIMT 四参数 launch
│       ├── echo_lru_launch.h           # workspace 公式、启动声明
│       ├── echo_lru_tiling.h           # tiling、1024 线程、128 KiB UB 常量
│       └── echo_lru_plugin.cpp         # torch.ops 校验、tiling 上传、异步下发
├── source_manifest.json          # main 来源提交与各文件的哈希
├── checks/check_echo_kernel_host.py  # 可选：直接提取新版函数做 CPU 并行回归
└── README.md
```

## 1. 先从哪里读

建议先读 `main()` 的循环，再回头看函数实现：

| 代码部分 | 函数 | 需要看懂的问题 |
|---|---|---|
| 第 1 节 | `make_case` | 命中率怎样构造？warm 为什么不会 reset？ |
| 第 1 节 | `make_echo_initial` | 同一份历史怎样转换成 Echo 的槽位和时间戳？ |
| 第 2 节 | `reference_simt` / `reference_echo` | 两种算法各自应该输出什么？ |
| 第 2 节 | `check_common_semantics` | 槽号不同的两种实现怎样检查共同语义？ |
| 第 3 节 | `prepare_npu` / `restore_state` | 哪些张量只读？哪些会被改写？如何恢复？ |
| 第 3 节 | `invoke` | 实际传给两个注册算子的参数是什么？ |
| 第 4 节 | `measure_one` | Event 放在哪里？时间单位是什么？ |
| 第 5 节 | `main` | 审计、预热、测量、最终检查如何串起来？ |

没有通用 Runner 类、异步 Python 回调、动态算法列表或其他测试文件依赖。`invoke` 中的参数顺序直接对应两个本地 C++ plugin 的 `TORCH_LIBRARY` schema。完整的调用链是：

```text
compare.py / invoke
  → torch.ops.sparse_kv_plan_op.sparse_kv_plan 或 echo_lru
  → 对应 plugin 的 NPU 实现：检查张量、构造 tiling、上传 tiling
  → RunOpApi 按原 main 的方式下发，lambda 按值捕获张量
  → launch_sparse_kv_plan 或 launch_echo_lru
  → 本目录 kernel 执行并原地更新状态、输出
```

可按第 9 节继续读两版算子源码。

## 2. 运行条件与命令

先下载工程：

```bash
git clone --branch feat/simt-variant-comparison --single-branch \
  https://github.com/liser0019/op_test_kvoffload_simt_echo.git
cd op_test_kvoffload_simt_echo
```

以下命令均在工程根目录运行。

### 无 NPU：阅读和检查输入/参考语义

只需要 Python 和 NumPy：

```bash
python3 compare.py --cpu-check
```

这个模式按生产尺寸构造四个场景，检查两份参考的 miss 数、查询映射和历史淘汰集合。它不加载 torch / torch_npu / 扩展，不输出 NPU 性能数字。

阅读时可把尺寸缩小，便于在调试器中查看数组：

```bash
python3 compare.py --cpu-check --rows 2 --topk 4 --capacity 8 --max-token 64
```

### NPU：直接构建本目录（默认全部 13 路）

在服务器下载本仓库，或复制整个工程目录。在匹配的 torch / torch_npu 环境中，加载实际 CANN 安装的 `set_env.sh`，直接构建本目录；无需原仓库、wheel 或 pip 安装：

```bash
# 路径按服务器 CANN 实际安装位置调整。
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --soc=ascend950
```

构建产物是 `build/libsimt_vs_echo_ops.so`。脚本默认加载这个文件，并直接调用其中注册的两个算子：

```bash
unset ASCEND_LAUNCH_BLOCKING

# 先逐样本审计，验证每次调用的结果。这里的延迟只作诊断。
python3 -X faulthandler compare.py \
  --device 1 \
  --warmup 2 --repeat 4 --verify-each-sample \
  --json results/audit.json

# 审计通过后，同尺寸正式测量。
python3 -X faulthandler compare.py \
  --device 1 \
  --warmup 10 --repeat 50 --json results/benchmark.json
```

`--device` 是逻辑卡号，按服务器实际可见设备选择。默认使用卡 1。

程序通过 `torch.ops.load_library()` 加载本目录共享库，不导入原仓库的 `sparse_kv_plan_op` 包，不回退到环境中已安装的扩展。程序打印实际库路径，JSON 记录共享库 SHA256、源码提取基线提交、`echo_revision` 和当前 Echo kernel 源码 SHA256；源码标记不能证明共享库已重新编译，也不能替代编译产物的验证。确需选择其他构建产物时，可传 `--library /绝对路径/libsimt_vs_echo_ops.so`。

`build.sh` 使用激活环境里的 `python3`，并明确传给 CMake。需要换解释器时设置 `PYTHON=/绝对路径/python3`；并行编译数量用 `JOBS=4`。CANN / 编译器 / runtime 路径可分别用 `ASCEND_CANN_PACKAGE_HOME`、`BISHENG_CXX`、`ASCENDC_RUNTIME_LIBRARY` 覆盖。

只看先前讨论的 75% 命中场景：

```bash
python3 compare.py --device 1 --scenario warm_75pct --warmup 10 --repeat 50
```

## 3. 共同输入到底是什么

默认尺寸：`rows=32, topk=2048, capacity=4096, max_token=262144`。

每行独立抽取 `topk` 个互不重复的正数 token，范围是 `[1,max_token)`。所有 source 都合法：

- `visible_seq_lens=stable_prefix_lens=max_token`。
- 行 i 对应 request i，`token_to_req=[0,1,...,rows-1]`。
- block 大小为 64；block_table 是恒等映射，覆盖整个 token 范围。
- 两个算子拿到同一个 NPU query tensor 和 stable-prefix tensor。
- 不启用投机后缀失效，Echo 的 `spec_enabled=False`。

四个场景使用相同 seed 和相同 query。区别只在预埋的 resident 和是否 reset：

| 场景 | 构造方法 | 每行 miss | 初始空闲槽 | 必须淘汰历史槽 |
|---|---|---:|---:|---:|
| `cold_reset` | `last_req_ids=-1`，新 request 触发 reset | 2048 | reset 后 4096 | 0 |
| `warm_75pct` | 把 query 前 1536 个 token 预埋为 resident | 512 | 2560 | 0 |
| `warm_all_hit` | 把全部 query 预埋为 resident | 0 | 2048 | 0 |
| `full_cache_75pct` | 预埋 1536 个命中 token，再用 query 之外的 token 补满缓存 | 512 | 0 | 512 |

warm 与 full 场景的 `last_req_ids` 和 `req_ids` 相同，所以预埋状态不会被整行 reset 清掉。

这是一套固定初始状态的单步微基准：每次调用处理全部 rows，然后下一次重新恢复初始状态。它不代表多轮 serving 中自然演化的命中率。

## 4. 两种状态怎样对应

### SIMT

- `slot_to_token[rows,capacity]`：0 基槽 → token，空槽是 `-1`。
- `lru_slots[rows,capacity]`：槽的历史顺序，最旧 → 最新。
- `last_req_ids[rows]`：判断是否发生 request reset。
- `current_slots` 是每个 query 的槽；miss 输出是压缩的前缀，由 `miss_count` 指定长度。

### Echo

- 普通槽从 1 开始；slot 0 / token 0 保留。
- `htd[rows,max_token]`：token → 槽，未驻留是 `INT32_MAX`。
- `dth[rows,capacity+1]`：槽 → token，空槽是 `INT32_MAX`。
- `pri`：历史时间戳，越小越先淘汰；相同时间戳按槽号排序。
- `free_slots` / `avail`：空闲栈及有效长度；`fifo` 提供本轮时间戳。
- 输出每个 query 的 miss_mask 与地址，格式没有压缩。

转换时，SIMT 槽 s 对应 Echo 槽 s+1。历史 LRU 中 rank 越小越旧，所以赋 `pri=rank+1`；`fifo=capacity+1`，让当前轮的 hit 时间戳高于所有初始历史值。

**不能要求两种实现分配相同槽号。** Echo 使用栈，SIMT 按已有 LRU 分配；槽的分配顺序与多轮时间戳平局策略不同。我们分别检查各自的完整结果，再核对共同查询、miss 数和历史淘汰 token 集合。

Echo 的 `miss_alloc_flat` 保留原接口约定 `row*capacity+one_based_slot`，不能擅自减 1，也不能把 hit 位置的 0 当成有效 miss 地址。

## 5. 参考结果为什么可以写得更短

原工程的通用 golden 包含非法 token、重复 token、越界 LRU、部分激活、MTP 等分支。本目录仅测试上表的合法性能场景，因此把这些分支从参考代码中省掉。

保留的核心语义是：

- SIMT 稳定分类旧 LRU；按 query 顺序分配 miss；LRU 重建为“剩余可淘汰槽 → 新 miss → 原 hit”。
- Echo 命中更新时间戳；不足的空闲槽从最低历史优先级补充；从 free 栈尾按顺序分配 miss；每轮更新两次 fifo。
- request reset 的状态与输出语义。
- 全部持久状态字段和全部输出，包括没有使用的输出尾部、free 栈余量。

CPU Echo 参考只有需要淘汰时才排序，这是参考实现省掉无效计算。**实际调用的是作者新版 Echo，设备内部仍走无条件四趟完整排序路径。** CPU 参考从不参与 NPU 计时。

`check_common_semantics` 还直接从原始 LRU 推导“最旧的未命中历史 token”集合，检查两版是否淘汰正确，避免仅依赖两份参考互相比较。

## 6. 一次测量精确包含哪些步骤

`measure_one` 的顺序是：

```text
copy_ 持久状态初始快照
torch.npu.synchronize()        ← 等恢复完成

start.record()                ← Event 开始
invoke(...)                   ← 实际 torch.ops 调用
end.record()                  ← Event 结束
end.synchronize()             ← 等这次调用完成

elapsed_time * 1000           ← 毫秒转换为微秒
```

状态分配、数据生成、CPU reference、NPU 状态恢复、结果读回和校验，都在 Event 区间外。工作状态由初始快照 clone 出来，kernel 无法修改恢复操作的源。

临时 workspace 按 main 的布局和复用方式分配：

- SIMT：`hash_capacity=next_pow2(2*topk)`，最小为 32；每行 `3*hash_capacity+2*capacity+topk` 个 int32。单行不超过 112 KiB 时走 kernel 内部 UB，外部只分配占位元素，否则分配完整 GM workspace。
- Echo：保留相同的 GM workspace 公式及默认核数划分，设备仍使用每核 128 KiB 静态 UB。

各 kernel 沿用自己的默认线程数和分核策略。C 系列使用一份 capacity 分类数组，因此其行临时区为 `3*H+capacity+topk`，host 与 Python 同时适配。共同控制的是设备、输入、初始历史、软件环境、计时口径；算法自己的资源配置属于实现的一部分。

**Event 数字是公共算子接口的流区间延迟**，包含该调用的 tiling H2D，以及可能的 Host 下发空隙。它不能直接替代 profiler 的纯 kernel duration。`host_us` 则包括两个 record、算子调用和等待完成。

每个数字都是一次调用处理所有 rows 的耗时，`repeat=50` 产生 50 个独立样本。两路先后次序按轮交替；50 轮时各先执行 25 次。原始样本和执行顺序都写入 JSON。

本目录直接调用 torch.ops，去掉了原来的 Python 状态包装，并复用 Event。Host 包装成本可能与旧脚本略有区别；对比时使用本脚本同一次运行的两路结果。

## 7. 正确性检查发生在哪里

主流程明确分为 A/B/C/D：

1. **A：计时前审计，默认 3 次。** 每次恢复状态，污染所有输出和临时 workspace，再调用算子并全量比较参考，排除读到旧答案的假通过。
2. **B：预热。** 每次仍恢复状态；预热结束后检查两路最后的完整结果。
3. **C：正式采样。** 每次仍恢复状态；两路交换顺序，记录原始延迟。
4. **D：计时后检查。** 检查两路最后的所有持久状态/输出，再检查共同语义，成功后才生成统计。

`--verify-each-sample` 进一步检查每个预热及正式样本的结果。读回与校验发生在 Event 结束后，但它们会改变后续调用的缓存条件和调用间隔，因此这个模式用于短跑审计，正式性能比较应关闭。

默认模式检查计时前审计、预热后和采样后结果；不会宣称已经校验过中间每一个正式样本。

任一检查失败，程序以异常退出，不生成该次运行的性能结论。保留之前的 JSON 时应核对本次退出码和文件时间；程序不会把旧文件当成当前结果。

## 8. 结果怎么看

JSON 包含两路的 Event 和 Host 的 P50/P95/min/max、原始样本、执行顺序、输入指纹、实际共享库路径和哈希。

`echo_over_simt_event_p50 = Echo 的 Event P50 / SIMT 的 Event P50`。

四个场景全部校验和测量完成后才写结果文件。设备 kernel 耗时需要在服务器另外使用 profiler 查看对应完整 kernel；本脚本没有把 Event 标签改成 kernel 标签。

本次整理在无 CANN/NPU 的本机验证输入与参考语义、Python 入口和 CPU 协议模拟。CPU 协议模拟用于核对参数顺序和流程，不代表实际 CANN 编译、NPU 正确性或性能测量通过。

### 本机已完成的核对

- 小尺寸 `2/4/8/64` 与生产尺寸 `32/2048/4096/262144`，各四个场景，共 8 组：共同输入、Echo 初始状态、两版全部参考字段与原 `main` 生成器/golden 逐字段一致。原工程文件只用于整理后的兼容性核对，运行 `compare.py` 不依赖它们。
- 生产尺寸与小尺寸的 `--cpu-check` 四个场景均通过；Python 编译检查和命令行帮助通过。
- CPU 模拟注册了原 C++ 的真实接口 schema，调用原 `main` golden 模拟设备结果；完整主流程共调用两个接口 64 次，验证状态恢复、四个场景、交替顺序、全字段比较，以及恢复操作在 Event 外。
- 故障注入：故意破坏最后一个正式样本的输出，计时后检查正确报错，没有写出结果 JSON。
- 补齐独立工程后，再次核对 9 个源码文件的来源和哈希；用新的共享库装载入口重跑上述 64 次 CPU 模拟及故障注入，均通过。
- `build.sh` 的 shell 语法与参数帮助通过；使用临时依赖占位目录完成 CMake 配置/生成检查，确认仅包含两个 plugin、两个 kernel object、对应编译选项及 runtime 链接。占位目录用于检查构建规则，**没有实际编译 CANN kernel 或生成可运行的 NPU 共享库**。

上述 CPU 模拟不产生可用的性能数字。服务器应先运行第 2 节的逐样本短跑审计，再运行正式测量。

### 作者新版 Echo 的 CPU 回归

可用 Python 标准库和支持 C++20 的 `g++` 运行：

```bash
python3 checks/check_echo_kernel_host.py
```

脚本直接提取当前头文件里的 `OpArgtopkRow` 和 `OpHitMiss`，用 128 个 CPU 线程、同步屏障和原子操作模拟执行，并启用 AddressSanitizer/UBSan。33 项检查覆盖 capacity 1～4096、随机/平局/无效时间戳、完整排序结果、调试直方图、释放输出、缓冲区边界和每线程 hit 累加。它不参与正式性能测试，也不能替代 NPU 编译、设备缓存一致性或性能验证。此次本地运行通过；四个性能场景的 CPU 参考检查也通过。

## 9. 算子源码怎么读

### SIMT

1. `sparse_kv_plan_plugin.cpp` 的 `sparse_kv_plan_npu`：检查所有张量，构造 tiling，上传到 NPU，调用 `RunOpApi`。这是 Event 中 Host 调用的实际内容。
2. `sparse_kv_plan_launch.h` 的 `MakeSparseKvPlanTiling`：运行时形状、hash 容量、行临时区和动态 UB；所有布局常量集中在 `sparse_kv_plan_config.h`。
3. `sparse_kv_plan_kernel.cpp` 尾部的 `launch_sparse_kv_plan`：保留 `<<<blockDim, dynUBufSize, stream>>>`，动态 UB 由 `tiling.localMemoryBytes` 提供。
4. 同文件的 `sparse_kv_plan_kernel`：从 GM 逐字段读取 tiling，创建 UB，启动 2048 线程 VF。
5. `ProcessRuntimeRow`：本行 reset、query hash、历史槽命中/失效、稳定分类、miss 压缩、slot 分配、输出和 LRU 重建。对照 `reference_simt`，可看清业务语义与并行实现的对应。

### Echo

1. `echo_lru_plugin.cpp` 的 `EchoNpu`：校验布局，上传 tiling，按原 main 的 `RunOpApi` 方式调用 launch；`EchoCoreCount` 查询当前设备核数。
2. `echo_lru_launch.cpp`：`<<<blockNum, ECHO_LRU_THREAD_NUM, 0, stream>>>`；线程数由 `echo_lru_tiling.h` 定义，目前为 1024。
3. `echo_lru_kernel.h` 尾部的 `echo_lru_kernel`：分配单块 128 KiB UB，初始化 `EchoLruKernel` 并调用 `Process`。
4. `EchoLruKernel::Process`：按阶段读 reset、hit/miss、protect、free、alloc、priority、映射更新和最终输出；每核处理自己的连续行段。
5. `ArgtopkAllRows` → `OpArgtopkRow`：每行计算 `freeSize` 后仍执行完整四趟 radix 排序，最后只输出需要释放的槽。本目录没有添加 `freeSize=0` 的提前返回。

Echo 文件仍保留部分早期注释和注释掉的旧实现，涉及 512 线程、bitonic、GM ping-pong 或 grid sync；这些文字与当前有效代码不同。阅读时以 `ECHO_LRU_THREAD_NUM=1024`、有效的 `OpArgtopkRow`、顶层 `argtopkUb` 和 `Process` 内实际调用为准。新版排序中的并行打包、桶归并、`bucketPrefix` 暂存及分段 scatter 注释已同步更新。

构建方面，SIMT 保留 mixed/VF 编译选项，Echo 保留 `--enable-simt`，两者均为 `-O3`、`dav-3510`。系统 C++ 编译两个 plugin，链接 main 相同的 torch、torch_npu、ascendcl、tiling_api、platform 和 AscendC runtime。仅去掉了 Python 包/pybind 装载壳，改由 `torch.ops.load_library` 触发相同的算子注册。
