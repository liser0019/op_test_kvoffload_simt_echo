# 从带页表 ABCDF 出发的 A/B/AB 实验

四路均为独立逐行 Plan，不涉及 MTP 并集或联合缓存协议：

| 名称 | 改动 |
|---|---|
| abcdf_pages | 原带页表 ABCDF，计算头保持历史快照 |
| abcdf_pages_a | warm miss 的 token 从 UB Hash 转发 |
| abcdf_pages_b | ping-pong block Scan，在线程寄存器中返回总数 |
| abcdf_pages_ab | 两项组合 |

带页表基线与三个候选都保留15个tensor、5个属性、80B tiling、mask47、1024线程、核数上限64和相同-O3选项。用户确认生产来源页合法；页表读取及校验代码保留，门禁不新增非法物理页工况。所有候选不改变 query、页表生成、初始缓存、LRU、request reset、因果可见性或重复 token 语义。

## 计算改动

候选公共计算头位于 csrc/abcdf_paged_opt/plan_kernel_impl.h。原 csrc/abcdf_pages 的四个公共头保持不变。

A 在 warm miss 压缩时保存 `(bucket << 16) | position`，最终分配时从 `hashKeys[bucket]` 取得token，避免再次读GM topkIndices。桶号和position均能用16位表示时才启用，否则保持原路径。不扩展UB数组。生产尺寸scratch在UB；较大尺寸走GM时转发读取也来自GM，不保证有收益。每个线程先读原桶号，经过原Scan屏障后才压缩散写，保留原tile结束同步，防止覆盖尚未读取的位置。cold 不建立Hash，A不影响cold token读取方式；其冷场景性能差异不能解释为UB转发收益。

B 保留原Scan函数，新增 PagedScan 包装。B启用时交替使用原 warpTotals/warpBases 空间，各warp用shuffle求跨warp前缀，减少一次内部block barrier。所有调用处通过寄存器 scanTotal 获取总数，避免将去掉的共享control写入继续当作有效值。cold、LRU分类、miss分类三个调用点均适配；原tile后同步保留。没有额外动态UB或全局状态。

B的代价是更多warp内计算和潜在寄存器变化；A也有解码开销。两项均为实验假设，不保证超过带页表基线。B不等同于旧ABCDF字母B，A也不等同于旧字母A。

## 上机

在本分支新克隆目录中，保留之前已验证工程及结果：

```bash
cd /workspace
git clone --branch feat/paged-abcdf-turbo-ab --single-branch \
 https://github.com/liser0019/op_test_kvoffload_simt_echo.git paged_abcdf_ab
cd paged_abcdf_ab
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
export PYTHON=/usr/local/python3.11.10/bin/python3
OUTPUT=/workspace/paged_abcdf_ab_smoke bash scripts/test_paged_abcdf_ab.sh
```

脚本在NPU0构建，再做四路原有门禁（重复、连续状态、request变化、inactive行、生产多tile等），最后采集cold/full_cache_75pct、independent、MTP0/1/3，默认warmup2/repeat5。只测四路，没有Turbo Joint。每次恢复各自初始状态，相同query和合法页表，逐样本检查输入不变性、保护区、逐位置currentSlots、缓存、LRU、miss。失败即停并保存现场。继续保留ACTIVE采集开始后1秒宿主等待及严格CSV样本数/顺序校验。

报告和原始CSV均留在OUTPUT下；设备kernel Duration(us)不包含恢复/校验/tiling上传，不等于接口端到端延迟。ratio键 `abcdf_pages_a_over_abcdf_pages_p50` 等表示候选除以带页表基线，小于1才更快。MTP3为固定32请求、128query行；full_cache_75pct/independent后续query实际命中率不保证75%，原用例不变。

有收益后再选候选复测；也可用新的OUTPUT和 WARMUP=10 REPEAT=50 运行同脚本（会重新检查门禁，已有构建可增量复用）。不要拿去页表版112us作为新候选的性能目标，应与同轮带页表ABCDF比较。

## 本地验证范围

`python3 -B checks/check_paged_abcdf_kernels.py` 编译并执行真实VF的64线程CPU shim + UBSan，覆盖四路、UB/GM、cold/多档满缓存、MTP0/1/3、重复与不可见token、连续状态、reset、部分活动行及生产尺寸。CPU线程数经过替换，不能验证真实NPU地址空间、1024线程同步或性能。

`checks/check_abcdf_pages_adapter.py` 检查参数适配、页表输入不变性、独立kernel名与CSV过滤。开发机没有CANN/NPU，构建和硬件门禁尚待服务器验证。
