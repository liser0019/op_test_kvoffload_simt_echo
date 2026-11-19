# MTP3 475K 单 AIV 两路对照

## 出口与固定形态

本对照从冻结的 `joint_mtp3_full_512`（mask 252）派生，原入口、17 tensor + 5 attr API、mask 249/252/253/254/255 及其行为保持不变。

| 版本 | mask | resident 查询 |
|---|---:|---|
| `joint_mtp3_dir475_512` | 256 | 强制 1024 桶目录 + 精确二分 |
| `joint_mtp3_bitmap475_512` | 257 | 32 位 masks + firstUid + `uint16 slotLookup` |

两者仍按 `tokenToReq` 的连续 run 分组，固定每组 4 行 query；不改变原有 AIV 分核，也不进入 8 行 MTP7 分组。固定 `TopK=2048`、`capacity=8192`、512 线程、joint=1，支持 `maxToken=102400`、`262144`、`358400`、`486400`，硬上限为 `475*1024=486400`。`PreparedVariant` 在启动前检查 shape、4 行分组和实际 maxToken。

## UB 与生命周期

两者动态 UB 均为 `187936B`，即冻结 full 版 `181024B + 6912B`：

- 128 KiB 归并包（输入/输出各 64 KiB）保持不变。
- Phase6 的 resident 查询索引复用归并包死区。
- 486400 时位图索引为 `60800B masks + 60800B firstUid + 16384B slotLookup = 134.75KiB`，比 128 KiB 多 6912B。
- 目录版不额外占用索引字节，但使用相同 UB 预算，避免两路因分配差异影响对照。

位图从 `owner` 开始依次放 masks、firstUid、slotLookup。masks/firstUid 的跨度按实际 `maxToken` 的 32 位 words 计算并向上对齐到 8 words（32B）；slotLookup 起点同时不得早于 owner 的 `capacity` words，防止小上下文时被后续 owner 初始化覆盖。486400 时该保护不改变 `30400 words + 8192 uint16` 的目标偏移。容量检查在 Host 与设备 Phase6 各执行一次，越过索引预算时拒绝并走原回退，而不是只放宽 raw maxToken 限制。

Phase6 末尾沿用既有 block barrier，确认所有 masks/firstUid/slotLookup 读者完成后，owner/partition 才进入下一生命周期。rankpipe 的原始输出与完整 miss 描述符尾部保留，不在本对照中改写。

## 构建与短测

```bash
PYTHON=/usr/local/python3.12.16/bin/python3 JOBS=6 \
  bash build_joint_plan.sh --soc=ascend950 --mtp3-only

/usr/local/python3.12.16/bin/python3 -B \
  checks/check_mtp3_candidates.py --mtp3-475-only \
  --report /workspace/<single-output>/cpu_gate.json
```

CANN 9.2.1 的 NPU5 空闲时：

```bash
bash scripts/test_mtp3_candidates.sh --skip-build --mtp3-475-only \
  --skip-profile --device=5 --output=/workspace/<single-output>
```

主线 NPU4 不参与本路线测试。
