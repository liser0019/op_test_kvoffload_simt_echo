# ABCDF 页表检查同场对照

本分支只在性能命令中选择两路：

- abcdf：当前去来源页表检查的独立 ABCDF。
- abcdf_pages：75cf8b2 中的独立 ABCDF，保留来源页表检查。

旧版四个公共计算/接口头逐字保存于 csrc/abcdf_pages，来源为提交75cf8b2。配置仍为mask47、1024线程；符号包装改名以支持同库加载，不改旧版计算。两路使用同一用例生成器及逻辑输入、分别恢复各自缓存状态；旧版额外接收 seed+90001 生成的合法物理页排列，其哈希写入布局报告，每次校验输入不变性。所有来源页均合法，因此独立 golden 与两路语义一致。比较范围是这些合法输入下的设备kernel Duration，不含页表上传/每次tiling上传，不代表接口端到端耗时。

private_turbo 是 Turbo 工程的逐行路径：不构建MTP并集，但包含联合框架的协议输出，如resident owner/page映射。joint_turbo 才会在MTP开启时尝试对同请求多行取并集。此前三路 abcdf/private_turbo/joint_turbo 用于判断逐行实现和联合实现的额外成本。本次实验不需要混入它们。

在新克隆的本分支目录中，配置好 CANN 后执行：

```bash
export PYTHON=/usr/local/python3.11.10/bin/python3
export ASCEND_CANN_PACKAGE_HOME=/usr/local/Ascend/cann-9.1.0
OUTPUT=/workspace/abcdf_pages_compare bash scripts/test_abcdf_page_compare.sh
```

使用NPU0，依次构建、两路门禁、MTP0/1/3各50样本。失败立即停止并保留现场，报告与CSV保存在OUTPUT。每次调用前状态恢复、逐样本校验及采集开始1秒等待沿用现有框架。75pct/independent的后续行并不保证75%命中，报告记录实际初始命中。两路交替先后顺序，严格核对目标kernel名、计数、顺序；abcdf_pages_over_abcdf_p50>1表示带页表版更慢，数值记录于summary/capture_metadata JSON。Markdown的各实现表也直接展示两路P50/P95。

开发机没有CANN/NPU；本地源码/模型检查不能判断哪版快。本实验可以区分当前同环境两版代码的差距，但即使旧版恢复到历史耗时，也不能仅凭一次实验确认具体指令或硬件瓶颈。
