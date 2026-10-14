# baseline

优化点：`原基线`。原 BCDE + fused + 1024，计算源码逐字保留。

本轮字母含义与历史 BCDE 的命名不同，详见工程根目录 README。
线程固定为 1024；行临时区阈值固定为 112 KiB；不引入 Host tiling 缓存。

计算与注册源码保存在 upstream/，内容与 a282f5d 中的 combo_bcde_fused_1024 一致。
