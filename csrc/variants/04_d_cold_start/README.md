# 04_d_cold_start

请求 ID 更换时跳过哈希与 resident 分类。

- 优化组合：`D`；线程数：2048。
- 接口、算子名、tiling 字段顺序与原版一致；逐版本单独构建。
- 行 workspace：`3H+2C+T` 个 int32。
- 具体接入与验证边界见上一级 README.md。
- changes.patch 是相对 00_baseline 的完整差异。
