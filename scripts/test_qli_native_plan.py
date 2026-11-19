#!/usr/bin/env python3
"""Short NPU smoke for the standalone-ordered Plan variants.

This reuses the pinned fused fixtures and the repository's independent joint
Plan golden. It is a correctness helper, not a timing harness, and it does not
modify the shared profile/metrics files.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def _load_build_helpers(build: Path):
    fused_tests = build / 'upstream/attention/quant_lightning_indexer_ordered_fused/tests'
    sys.path.insert(0, str(fused_tests / 'host_plan'))
    sys.path.insert(0, str(fused_tests / 'npu'))
    import fixture  # type: ignore
    import standalone  # type: ignore
    import standalone_state  # type: ignore
    return fixture, standalone_state, standalone


def _op(module: str):
    return getattr(importlib.import_module(module), module.rsplit('.', 1)[-1])


def _case(fixture, topk: "torch.Tensor") -> dict:
    rows = fixture.spec.batch * fixture.spec.width
    return {
        'rows': rows,
        'active_rows': rows,
        'topk': fixture.spec.topk,
        'capacity': fixture.spec.capacity,
        'max_token': fixture.spec.max_token,
        'block_size': 128,
        'req_ids': fixture.req_ids.cpu().numpy().copy(),
        'topk_indices': topk.numpy().reshape(rows, fixture.spec.topk).copy(),
        'stable_prefix_lens': fixture.stable_prefix_lens.cpu().numpy().copy(),
        'visible_seq_lens': fixture.visible_seq_lens.cpu().numpy().copy(),
        'token_to_req': fixture.token_to_req.cpu().numpy().copy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', required=True, type=Path)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=4096)
    parser.add_argument('--capacity', type=int, default=8192)
    parser.add_argument('--phases', type=int, default=3)
    args = parser.parse_args()
    if args.width != 4:
        raise SystemExit('this smoke is pinned to MTP3 = four query rows per request')

    from joint_plan_cases import validate
    fixture, state_mod, standalone = _load_build_helpers(args.build_dir.resolve())
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError('TorchNPU reports no available Ascend device')
    torch.npu.set_device(args.device)
    if '950' not in str(torch.npu.get_device_properties().name):
        raise RuntimeError('standalone ordered Plan variants require Ascend950')

    prefix = 'cann_ops_transformer_qlifusion.ops.attention'
    ordered = _op(prefix + '.quant_lightning_indexer_ordered.quant_lightning_indexer_ordered')
    shell = _op(prefix + '.quant_lightning_indexer_plan_native_shell.'
                        'quant_lightning_indexer_plan_native_shell')
    fused = _op(prefix + '.quant_lightning_indexer_plan_native_fused.'
                        'quant_lightning_indexer_plan_native_fused')
    group = _op(prefix + '.quant_lightning_indexer_plan_native_group.'
                        'quant_lightning_indexer_plan_native_group')

    spec = fixture.FixtureSpec(
        args.batch, args.width, 2048, args.capacity, args.seq_len, 32,
        'fp8', 'random', 17, 'sparse',
    )
    cpu, device = (fixture.make_fixture(spec, device=target, pin_host=False)
                   for target in ('cpu', 'npu'))
    attrs = state_mod.plan_attributes(device)
    rows = spec.batch * spec.width

    def run_ordered() -> torch.Tensor:
        output = torch.empty((rows, 1, spec.topk), dtype=torch.int32, device=device.query.device)
        result = ordered(**fixture.qli_kwargs(device), topk_out=output)
        torch.npu.synchronize()
        return result.cpu()

    expected_topk = run_ordered()
    shell_state = state_mod.allocate_state(spec, 'npu')
    shell_before = shell_state.snapshot()
    shell_out = torch.empty_like(expected_topk, device=device.query.device)
    shell_out = shell(**fixture.qli_kwargs(device), **shell_state.fused_inputs(device),
                      **attrs, topk_out=shell_out)
    torch.npu.synchronize()
    if not torch.equal(shell_out.cpu(), expected_topk):
        raise AssertionError('native_shell changed standalone ordered TopK')
    for name, before in shell_before.items():
        if not torch.equal(shell_state.tensors[name].cpu(), before):
            raise AssertionError(f'native_shell modified dummy Plan tensor: {name}')

    def allocate_device_state():
        state = state_mod.allocate_state(spec, 'npu')
        # Route 2 uses streamed unique-token GM for compact rankpipe AC.
        requests = spec.batch
        workspace_words = (requests * 8 * spec.topk // 2 + requests * 8 * spec.capacity +
                           requests * 8 * (3 * 4096 + spec.capacity + spec.topk))
        state.tensors['compact_workspace'] = torch.full(
            (workspace_words,), 0x7fffffff, dtype=torch.int32, device=device.query.device,
        )
        return state

    variants = (
        ('native_fused', fused, allocate_device_state()),
        ('native_group', group, allocate_device_state()),
    )
    for phase in range(args.phases):
        standalone.mutate_fixture(cpu, phase)
        standalone.mutate_fixture(device, phase)
        expected_topk = run_ordered()
        for label, operation, state in variants:
            before = {name: value.cpu().numpy().copy()
                      for name, value in state.tensors.items()}
            actual = operation(**fixture.qli_kwargs(device), **state.fused_inputs(device), **attrs)
            torch.npu.synchronize()
            if not torch.equal(actual.cpu(), expected_topk):
                raise AssertionError(f'{label} changed ordered TopK at phase {phase}')
            actual_state = {name: value.cpu().numpy().copy()
                            for name, value in state.tensors.items()}
            validate(_case(device, expected_topk), before, actual_state, True,
                     max_group_rows=4, require_power_two_groups=False)
            if torch.count_nonzero(state.tensors['ready_counts']).item() != 0:
                raise AssertionError(f'{label} did not reset ready_counts')

    print(
        f'PASS native_shell/native_fused/native_group B={spec.batch} width={spec.width} '
        f'seq={spec.max_token} capacity={spec.capacity} phases={args.phases} '
        'ordered_topk=exact plan=golden dummy_plan=unchanged ready_counts=zero'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
