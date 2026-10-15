# abcdfh

优化点：`ABCDFH`。ABCD + 二级 Scan 特化 + tile 同步削减。

本轮字母含义与历史 BCDE 的命名不同，详见工程根目录 README。
线程固定为 1024；行临时区阈值固定为 112 KiB；不引入 Host tiling 缓存。

variant_config.h 选择编译期位；kernel.cpp 包含 common/plan_kernel_impl.h。
未启用的优化由 if constexpr 消除，各版拥有独立 kernel/launch/注册名称。

F 精简二级 Scan 内部的总数发布；H 减少中间 tile 的尾部屏障。
不启用 E/G；保持原有 MTP 可见长度、稳定前缀失效和 UB/GM 回退语义。
本组合待 CANN 编译与 NPU 正确性、性能验证。
