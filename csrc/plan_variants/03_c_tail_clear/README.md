# c_tail_clear

优化点：`C`。miss 输出只清理未分配尾部。

本轮字母含义与历史 BCDE 的命名不同，详见工程根目录 README。
线程固定为 1024；行临时区阈值固定为 112 KiB；不引入 Host tiling 缓存。

variant_config.h 选择编译期位；kernel.cpp 包含 common/plan_kernel_impl.h。
未启用的优化由 if constexpr 消除，各版拥有独立 kernel/launch/注册名称。
