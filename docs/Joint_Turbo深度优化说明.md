# Turbo 当前实现与上机说明

当前开发机仅做 CPU 模型、C++ 语义模拟、host 布局及静态检查；没有执行 CANN 编译或 NPU 验证。历史含来源页表检查版本的耗时不代表当前版本性能。

## 本轮完成的两项改动

1. 活跃 Plan 路径移除来源 `blockTable`、物理页检查及仅供检查的属性。独立 ABCDF 接口是 14 个张量＋3 个属性，联合 Prepared 是 17 个张量＋5 个属性。`blockSize` 在联合接口中仍用于 resident 输出页表，不能删除；`residentBlockTable` 是输出映射，不是已删除的 Host 来源页表。
2. Turbo 使用独立 `queryBuckets` 数组，建表时直接保存桶号，最终通过 `bucketSlots` 得到槽号。冷/热联合路径都不借用 GM currentSlots 暂存，最终输出只写一次。整个 turbo_plan.h 无 asc_atomic_or；Hash 插入保持纯 CAS。

上游负责来源页合法性与生命周期。Plan 保留请求编号、token 范围、逐行因果可见性、stable prefix、重复 token、LRU、reset 和回退语义。Transfer 的来源页表、地址转换和 kernel 算法保留。共享 compare 输入准备仍上传 Transfer 需要的页表，但不再传给 Plan。

## UB 布局与回退

联合 scratch 依次为 keys[J]、bucketSlots[J]、partition[C]、queryBuckets[4*T]。当前支持 MTP 0/1/3；为避免读取设备分组形状，host 对联合入口预留最多四行 query 桶号。

生产 T=2048、C=4096、J=8192：scratch=28672 个 int32=114688 字节，Scan=288 字节，总动态 UB=114976 字节。MTP0 使用 private 布局，总 UB=74016 字节。Python、C++ host、kernel 的偏移必须一致。

超出 112 KiB 行 scratch 预算时使用 GM scratch，queryBuckets 也随之在 GM，不能称所有尺寸都驻留 UB。超过预留 query 容量的请求组在写入前整组回退 private；并集超 capacity 同样回退。最终 currentSlots 始终是 GM 输出。

partition 在热路径仍可用高16位保存唯一桶列表、低16位保存淘汰/命中槽；它不是已经取消的 queryBuckets/bucketSlots 双重打包。

## 其他保留优化与边界

保留冷启动直接分配、热路径紧凑唯一桶列表、private miss 桶号转发、请求组调度提前退出及双缓冲 Scan。仍是一请求组一 AIV，无跨核协作。

双缓冲 Scan 用每个 warp 重算二级前缀和换取一次 block barrier，不能预设必然更快。CPU 模拟默认64线程（2 warp），不能替代1024线程（32 warp）的编译、同步和资源验证。保留 joint MTP0 的 VF<true>、private 的 VF<false>、线程1发布模式的经验规避；根因没有确认。

旧 bitmap 本来就逐 tile 计数并提前溢出回退，最终通过 GM map[token] 直接取槽，不能误写成扫描完整位图计数或重新 Hash 查找。其 1 MiB map 地址范围和位图初始化是源码成本，未测得各自占比。

## 本地检查

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B checks/check_joint_models.py
PYTHONDONTWRITEBYTECODE=1 python3 -B checks/check_joint_variant_kernels.py --variants abcdf turbo a abc abcd bitmap
PYTHONDONTWRITEBYTECODE=1 python3 -B checks/check_plan_layouts.py
PYTHONDONTWRITEBYTECODE=1 python3 -B checks/check_plan_goldens.py
PYTHONDONTWRITEBYTECODE=1 python3 -B checks/check_bitmap_launch_structure.py
```

源码 manifest 已更新，旧库必须重建；不能复用旧门禁报告。独立 ABCDF 的算法配置仍为 mask47/1024线程，但来源合法性契约与 ABI 已改变，报告不会把它冒充历史原样基线。

## 服务器运行

在服务器本工程根目录执行，默认逻辑卡0。产物留在 /workspace，不自动上传。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHON=/usr/local/python3.11.10/bin/python3
bash scripts/test_turbo_no_pages.sh
```

脚本失败立即停止：构建（保持-O3）→三路完整正确性门禁（错写回归5次）→同输入 MTP0/1/3、cold_reset/full_cache_75pct、independent 小矩阵（warmup2/repeat5）。门禁含溢出回退与连续状态。只测 Plan，采用精确 kernel Duration；保留 profiler 启动后1秒等待与严格样本顺序检查，不把 descriptor 减少宣称为端到端收益。

OUTPUT 可指定新的持久目录；脚本拒绝覆盖已有结果。正式复测可提高 repeat，先检查小矩阵是否正确且有收益。
