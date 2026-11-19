# Native ordered Plan fusion variants

## Provenance

- Upstream: `9ec2110969a9e8502bf6d579c027ac4ea9572016`.
- Device QLI/Cube/Vector/SplitCore/ordered TopK source: `attention/quant_lightning_indexer_ordered`, tree `2f3532304e78af4325d77c7160c50d9d24cbf776`.
- Plan ABI, host parameter map, registration scaffold, and reviewed publication hook: `attention/quant_lightning_indexer_ordered_fused`.
- `quant_lightning_indexer_ordered_topk.h` is copied byte-for-byte. SHA256: `14a93886bce85c1c59f2d6592db65993b84e4b7727c475b7bedfc8c95b4c19fd`.

`scripts/native_plan_overlay.py` performs only the audited native-to-Plan overlay. It does not use the old fused QLI/Cube/Vector implementation.

## Roles

- `quant_lightning_indexer_plan_native_shell`
  - Standalone ordered device path and UB allocation.
  - Plan hook is compile-time disabled.
  - Fused ABI tensors are accepted as dummies and are not modified.
  - `compact_workspace` is only required to be non-empty.
- `quant_lightning_indexer_plan_native_fused`
  - Per-row publication after the native MTE3 output write.
  - `MTE3_S`, existing `V_S`, thread fence, and GM atomic counter elect the last
    producer of the request group; there is no cross-AIV polling.
  - Invalid/short rows use the same one-producer publication path.
- `quant_lightning_indexer_plan_native_group`
  - For native 4-row chunks, one AIV normally owns two adjacent rows.
  - The two rows are published once as `count=2` after both MTE3 writes drain.
  - `S1Base2`, `count!=2`, or request-boundary cases fall back to per-row
    publication. Each MTP3 request still contributes exactly four row-count
    increments before group reset.
  - `ready_counts` is returned to zero after each completed group.

## Tiling and layout

- Native rule is retained: total query rows `<=64` use S1Base2, otherwise S1Base4.
- Host tiling fails if the native ordered footprint plus Plan exceeds 216 KiB; it
  does not silently force S1Base2.
- The first fused Plan uses compact rankpipe AC (`QLI_PLAN_STORAGE=3`) from
  `csrc/qli_fusion/plan.h`.
- Compact storage streams unique-token data through GM. For `R` requests this is
  `R*8*capacity` unique words, not an alias of the native ordered storage. The
  total workspace is:
  `R*8*topk/2 + R*8*capacity + R*8*(3*4096 + capacity + topk)` int32 words.
- Native fused reuses the TopK scratch arena only after TopK values are dead;
  Cube `resMm` and `indicesOutLocal_` remain independent.

## Build

Use an isolated checkout/venv. Pass an absolute build directory because the CANN
vendor installer rejects relative `--install-path` values.

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
export PYTHON=/absolute/path/to/venv/bin/python
export QLI_UPSTREAM_SOURCE=/workspace/qli_ordered_standalone_9zCUYA/build_ordered_standalone/upstream
export QLI_THIRD_PARTY_CACHE=/workspace/qli_ordered_standalone_9zCUYA/build_ordered_standalone/upstream/third_party
export QLI_FUSION_BUILD_DIR=/absolute/isolated/build/native_plan
bash scripts/build_qli_fusion.sh --native-plan-only
```

The package contains only:

`quant_lightning_indexer`, `quant_lightning_indexer_ordered`,
`quant_lightning_indexer_plan_native_shell`,
`quant_lightning_indexer_plan_native_fused`,
`quant_lightning_indexer_plan_native_group`.

## Verified artifact

- Build/deploy root: `/workspace/ordered_mtp3_parallel_aQVhnU/fuse/build_native_plan_final_234201`.
- Environment: `source /workspace/ordered_mtp3_parallel_aQVhnU/fuse/build_native_plan_final_234201/env.sh`.
- Package: `upstream/build_out/cann-ops-transformer-qlifusion_linux-aarch64.run`.
- Package SHA256: `edc631c22177002188287f17546a9e8248cc66ea13e1bca65ab7200ac3824eb2`.
- Wheel: `upstream/build_out/cann_ops_transformer_qlifusion-1.0.0-py3-none-any.whl`.
- Wheel SHA256: `47505e2e59f2ed3b4facb547dbde28c01633b7d850f1204f3691d375f723930a`.
- Source identity (`_fusion_identity` verified in all packaged indexer extensions): `f26bb501bf4f46061b1bb62d58393d3fcac8755c9dc25b227350424d9f255a22`.
- Per-operator source fingerprints:
  - shell: `579cf198ceff6a310d522ad2fb1b31b1956492401a77d2805ae4bde7ad3ced24`
  - fused: `c3d896b76a701423cbea1dbeed245588cdd4b8831f61262cfc5f953171460cc6`
  - group: `8711aa6fc96dba29b1153aab6730d494d968f511413aec7f0a02c169fed845ef`

## NPU3 short test

The following passed on an otherwise idle NPU3 with CANN 9.2.1:

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
source /workspace/ordered_mtp3_parallel_aQVhnU/fuse/build_native_plan_final_234201/env.sh
PYTHONPATH=$PWD /workspace/ordered_mtp3_parallel_aQVhnU/fuse/venv_native_plan.5gNYrW/bin/python -B   scripts/test_qli_native_plan.py   --build-dir /workspace/ordered_mtp3_parallel_aQVhnU/fuse/build_native_plan_final_234201   --device 3 --batch 32 --width 4 --seq-len 4096 --capacity 8192 --phases 3
```

Result:

`PASS native_shell/native_fused/native_group B=32 width=4 seq=4096 capacity=8192 phases=3 ordered_topk=exact plan=golden dummy_plan=unchanged ready_counts=zero`

The smoke uses the independent joint Plan golden in `joint_plan_cases.py`; it
does not repair or sort the ordered TopK output. Full 256K timing remains with
the main integration harness.

## Rebuild note

Any change to `prepare_qli_fusion.py`, `native_plan_overlay.py`, `plan.h`, the
Plan config, or the native binding changes the source identity. Rebuild and
redeploy both the vendor package and torch extension wheel; do not reuse stale
`TORCH_EXTENSIONS_DIR` artifacts.
