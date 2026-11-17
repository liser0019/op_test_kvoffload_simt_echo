# 开发机验证记录

本分支没有在开发机运行 CANN、Bisheng 或 NPU。以下结果只说明可在此处检查的部分。

| 检查 | 实际结果 | 局限 |
|---|---|---|
| `checks/check_qli_fusion_plan.py` | 三种存储方案合计105次实际Plan控制器语义调用通过；开启CPU UBSan | SIMT线程缩减；地址空间、真实SIMD指令、DMA及跨AIV时序由模型代替 |
| `checks.test_qli_fusion_fixture`、`checks.test_qli_fusion_metrics` | 12项单元检查通过 | 输入布局与合成CSV检查，不是设备性能或Indexer精度结果 |
| `checks/check_qli_fusion_resources.py` | CPU编译实际生成的Host资源函数，四种模式×三档容量均未超过216KiB | 只核对算术与阶段布局，不证明设备运行安全 |
| `checks/check_qli_fusion_bindings.py --syntax` | 四路实际Torch schema及Meta/PrivateUse1注册通过；Host绑定C++语法通过 | ACLNN调用使用桩；没有CANN链接或device编译 |
| `scripts/export_fusion_plan.py --check` | 可重现生成及源码身份检查通过 | 不代替硬件编译 |
| 生命周期/物理复用模型 | 阶段级分配通过，无模型告警 | 不是逐指令同步审计，不验证Cube、Fixpipe、MTE或DCache真实行为 |
| Python AST、`bash -n` | 新增脚本语法检查通过 | 不能发现工具链版本/API或运行环境问题 |

源码核对采用 `ops-transformer@9ec2110969a9e8502bf6d579c027ac4ea9572016`。
保序TopK的有效索引按位置递增输出，短序列分支显式补 `-1`；
测试入口检查此约定，不在CPU上排序或修复Indexer输出。

服务器按中文教程先构建，再进行小批量链路测试、短序列边界门禁，最后采集四路对照。
只有服务器实际通过后才能确认CANN兼容性、跨单元同步和性能收益。
