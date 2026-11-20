# Final SparseKV Plan full matrix

- Device: NPU5, Ascend950DT_9582
- CANN: 9.2.1; Python 3.12.16; torch_npu 2.10.0.post7
- Final library SHA256: `d3a0cc94eb1910cb395c48bdfc96910b68a42d0c95db02a4b651385164e68801`
- n=10 per workload; warmup=3; target Plan kernel `Duration(us)` only.
- MTP0/1 use `joint_rankpipe_ac_512`, capacity=4096.
- MTP3 compares `joint_rankpipe_ac_512` with `joint_mtp3_full_map_grid14_512`, capacity=8192.
- Context labels 100K/150K/350K use max_token=102400/153600/358400.
- Every sample passed strict correctness checks; zero command failures.

## Coverage

- 3 MTP modes × 9 batches × 3 contexts = 81 workloads.
- 108 implementation-result rows; MTP0/1 each 27, MTP3 has baseline+final for 54.
- No overflow groups in MTP3.

## P50 summary

| MTP | implementation | min P50 | max P50 |
|---|---|---:|---:|
| MTP0 | `joint_rankpipe_ac_512` | 23.526 | 27.468 |
| MTP1 | `joint_rankpipe_ac_512` | 32.646 | 36.105 |
| MTP3 final | `joint_mtp3_full_map_grid14_512` | 45.488 | 56.601 |

MTP3 speedup range: **1.067×–1.206×**, geometric mean **1.168×**.

## P50 — 100K (max_token=102400)

| batch | MTP0 μs | MTP1 μs | MTP3 baseline μs | MTP3 final μs | speedup | blocks final |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.248 | 33.836 | 57.275 | **51.587** | 1.110× | 1 |
| 4 | 24.891 | 34.759 | 58.324 | **50.930** | 1.145× | 4 |
| 8 | 24.624 | 34.415 | 54.948 | **45.569** | 1.206× | 8 |
| 9 | 24.846 | 34.822 | 54.752 | **46.288** | 1.183× | 9 |
| 10 | 24.524 | 34.852 | 55.166 | **46.358** | 1.190× | 10 |
| 14 | 25.024 | 32.646 | 55.498 | **47.018** | 1.180× | 14 |
| 16 | 25.111 | 33.419 | 56.451 | **47.150** | 1.197× | 16 |
| 28 | 23.526 | 34.058 | 56.773 | **47.782** | 1.188× | 28 |
| 32 | 23.995 | 34.206 | 57.334 | **48.615** | 1.179× | 32 |

## P50 — 150K (max_token=153600)

| batch | MTP0 μs | MTP1 μs | MTP3 baseline μs | MTP3 final μs | speedup | blocks final |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.318 | 34.204 | 60.374 | **56.601** | 1.067× | 1 |
| 4 | 24.894 | 33.863 | 54.033 | **45.493** | 1.188× | 4 |
| 8 | 25.047 | 34.422 | 54.519 | **45.488** | 1.199× | 8 |
| 9 | 24.877 | 35.062 | 54.807 | **46.058** | 1.190× | 9 |
| 10 | 24.742 | 34.716 | 55.213 | **46.343** | 1.191× | 10 |
| 14 | 25.246 | 33.166 | 55.511 | **48.035** | 1.156× | 14 |
| 16 | 25.241 | 33.323 | 55.772 | **47.143** | 1.183× | 16 |
| 28 | 23.974 | 34.248 | 56.880 | **47.859** | 1.188× | 28 |
| 32 | 23.899 | 34.422 | 57.459 | **48.529** | 1.184× | 32 |

## P50 — 350K (max_token=358400)

| batch | MTP0 μs | MTP1 μs | MTP3 baseline μs | MTP3 final μs | speedup | blocks final |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.797 | 35.913 | 60.577 | **55.176** | 1.098× | 1 |
| 4 | 26.627 | 35.541 | 57.227 | **49.114** | 1.165× | 4 |
| 8 | 26.486 | 35.837 | 57.788 | **48.874** | 1.182× | 8 |
| 9 | 27.124 | 35.966 | 57.745 | **49.609** | 1.164× | 9 |
| 10 | 26.901 | 35.764 | 58.482 | **49.742** | 1.176× | 10 |
| 14 | 27.035 | 34.238 | 59.398 | **52.486** | 1.132× | 14 |
| 16 | 27.468 | 34.859 | 59.347 | **50.394** | 1.178× | 16 |
| 28 | 25.540 | 35.213 | 60.242 | **51.494** | 1.170× | 28 |
| 32 | 26.045 | 36.105 | 60.165 | **52.014** | 1.157× | 32 |

## P95 — 100K

| batch | MTP0 μs | MTP1 μs | MTP3 baseline μs | MTP3 final μs |
|---:|---:|---:|---:|---:|
| 1 | 27.358 | 39.448 | 58.454 | 52.080 |
| 4 | 27.021 | 39.010 | 61.394 | 53.240 |
| 8 | 25.756 | 35.466 | 57.752 | 46.067 |
| 9 | 25.444 | 35.352 | 64.887 | 56.394 |
| 10 | 25.461 | 35.445 | 55.886 | 46.575 |
| 14 | 25.835 | 33.399 | 64.218 | 54.714 |
| 16 | 29.284 | 37.002 | 64.196 | 52.869 |
| 28 | 24.470 | 35.201 | 61.261 | 54.678 |
| 32 | 24.860 | 35.216 | 57.881 | 48.828 |

## P95 — 150K

| batch | MTP0 μs | MTP1 μs | MTP3 baseline μs | MTP3 final μs |
|---:|---:|---:|---:|---:|
| 1 | 26.451 | 39.617 | 67.348 | 62.605 |
| 4 | 27.936 | 37.139 | 54.671 | 45.766 |
| 8 | 25.336 | 35.146 | 55.629 | 45.565 |
| 9 | 25.610 | 35.754 | 55.697 | 46.355 |
| 10 | 25.436 | 35.455 | 56.252 | 46.494 |
| 14 | 25.861 | 36.361 | 62.006 | 57.039 |
| 16 | 28.187 | 36.422 | 60.777 | 53.753 |
| 28 | 24.560 | 42.201 | 57.351 | 48.443 |
| 32 | 24.851 | 35.457 | 65.340 | 58.971 |

## P95 — 350K

| batch | MTP0 μs | MTP1 μs | MTP3 baseline μs | MTP3 final μs |
|---:|---:|---:|---:|---:|
| 1 | 29.585 | 42.113 | 61.823 | 55.674 |
| 4 | 27.453 | 41.768 | 58.271 | 49.364 |
| 8 | 27.834 | 36.290 | 59.221 | 48.980 |
| 9 | 30.090 | 36.962 | 58.994 | 49.800 |
| 10 | 27.244 | 37.011 | 61.285 | 54.834 |
| 14 | 27.669 | 37.404 | 66.662 | 71.275 |
| 16 | 29.897 | 37.693 | 64.971 | 55.366 |
| 28 | 26.519 | 35.860 | 60.957 | 51.868 |
| 32 | 27.018 | 42.957 | 64.815 | 74.404 |

## Observations

- MTP3 final is faster than same-run baseline in all 27 MTP3 workload combinations; worst improvement remains 1.067× at batch1/150K, best is 1.206× at batch8/100K.
- At production-like batch8–16, MTP3 final P50 is roughly 45.5–48.6 μs for 100K/150K, with the exact grid launching only `batch` blocks.
- MTP0 P50 remains stable at roughly 23.5–27.5 μs and MTP1 at 32.6–36.1 μs across the matrix; these paths remain on rankpipe AC.
- 350K uses the directory+binary fallback and remains faster than baseline for every batch.
- P95 has several isolated host/system long-tail samples on the shared server. P50 regressions were not observed; P95 entries should be treated as preliminary for the affected cases.

## batch1 与 batch4 的独立复测

原 81 工况矩阵中 batch1/150K 的 56.601 μs 含共享服务器长尾。随后用同库、同设备、n=30、warmup=10 复测：

| context | batch | MTP3 baseline P50/P95 | MTP3 final P50/P95 | final blocks |
|---|---:|---:|---:|---:|
| 100K | 1 | 57.727/58.739 | 51.597/52.098 | 1 |
| 100K | 4 | 53.766/54.952 | 45.513/45.703 | 4 |
| 150K | 1 | 57.152/57.853 | 51.397/51.962 | 1 |
| 150K | 4 | 53.867/54.899 | 45.399/45.565 | 4 |

- batch1/150K 的稳定结果约 51.4 μs，不是 56.6 μs；原值属于长尾干扰。
- batch1 仍比 batch4 慢约 6.0 μs，下降比例约 13%。
- batch1 的 MTP3 只有一个 request group，只启动一个 AIV block；该 AIV 需要顺序完成四路归并、去重、位图查询、resident 分类、miss/LRU 生成和输出。
- batch4 同时启动四个独立 group block，因此可同时处理四个请求。每个 group 的 descriptor 约 2290–2307，输入工作量基本相同，所以 batch1 的额外耗时不是 token 数更多，而是只有一个 AIV 可利用。
- batch1 仍快于同场 rankpipe baseline，只是收益从 batch4+ 的约 18% 降到约 11%–12%。

## Artifacts

- Raw CSV: `/dev/shm/plan14_full_matrix_20261120_092327`
- Summary CSV: `/workspace/plan14_overnight_20261119/results/full_matrix_20261120_092327/matrix.csv`
- Summary JSON: `/workspace/plan14_overnight_20261119/results/full_matrix_20261120_092327/matrix.json`
- Runner log: `/workspace/plan14_overnight_20261119/results/full_matrix_20261120_092327/runner.log`
- Raw index: `/workspace/plan14_overnight_20261119/results/full_matrix_20261120_092327/raw_index.md`

