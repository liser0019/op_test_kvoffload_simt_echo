#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ABCDF vs archive LRU compact: correctness adapter for profile_plan_variants.py.

Only shared Plan semantics: all rows active, identity row/request mapping, valid
unique queries/residents and valid source pages. Address generation/copy excluded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import compare as plan

ROOT = Path(__file__).resolve().parent
I32_MAX, BLOCK_SIZE = plan.I32_MAX, plan.BLOCK_SIZE
SCENARIOS, SIMT_STATE, SIMT_OUTPUT = plan.SCENARIOS, plan.SIMT_STATE, plan.SIMT_OUTPUT
array_hash, reference_simt = plan.array_hash, plan.reference_simt
execution_order, restore_state = plan.execution_order, plan.restore_state
VARIANTS = {'abcdf': plan.VARIANTS['abcdf'], 'memfabric': {
    'op_name': 'memfabric_lru_plan', 'kernel_name': 'OffloadLruCompactOps'}}
IMPLEMENTATIONS = tuple(VARIANTS)
DEFAULT_PROFILE_REFERENCE = 'memfabric'


def legacy_layout(topk, capacity, rows, max_token):
    t_bytes, c_bytes = (topk * 4 + 31) & ~31, (capacity * 4 + 31) & ~31
    ub_bytes = 6 * t_bytes + 4 * c_bytes + 4096 + 64
    if ub_bytes > 120 * 1024:
        raise ValueError('MemFabric Plan exceeds comparison UB budget of 120 KiB')
    if 8 * max_token > I32_MAX:
        raise ValueError('MemFabric workspace exceeds supported int32 element limit')
    return {'blocks': 8, 'ub_bytes': ub_bytes, 'mark_elements': 8 * max_token,
            'pos_elements': 8 * max_token, 'epoch_elements': 8,
            'scratch_policy': 'zero initialize once; advance epochs across calls; restore public state only'}


def validate_case(case):
    r, t, c, length = (case[k] for k in ('rows', 'topk', 'capacity', 'max_token'))
    legacy_layout(t, c, r, length)
    if not 0 < t <= c or case.get('active_rows', r) != r:
        raise ValueError('Common domain requires all rows active and 0<topk<=capacity')
    if case.get('block_size', BLOCK_SIZE) <= 0:
        raise ValueError('Common domain requires a positive block size')
    if len(case['req_ids']) != r or not np.array_equal(case['token_to_req'], np.arange(r)):
        raise ValueError('Archive Plan requires one request per row, with identity mapping')
    for row in range(r):
        q = case['topk_indices'][row]
        if len(np.unique(q)) != t or np.any(q < 0) or np.any(q >= length):
            raise ValueError('Common performance domain requires unique valid queries')
        if np.any(q >= case['visible_seq_lens'][row]):
            raise ValueError('Query outside visible length is not supported by archive Plan')
        blocks = q // case.get('block_size', BLOCK_SIZE)
        if np.any(blocks >= case['block_table'].shape[1]):
            raise ValueError('Query outside block table')
        physical = case['block_table'][row, blocks]
        if np.any(physical < 0) or np.any(physical >= case['host_num_blocks']):
            raise ValueError('Invalid source page is not supported by archive Plan')
        resident = case['slot_to_token'][row]
        occupied = resident[resident >= 0]
        if len(occupied) != len(np.unique(occupied)) or np.any(occupied >= length):
            raise ValueError('Common domain requires unique resident tokens')
        if not np.array_equal(np.sort(case['lru_slots'][row]), np.arange(c)):
            raise ValueError('Common domain requires a valid LRU permutation')
    return case


def make_case(*args, **kwargs):
    return validate_case(plan.make_case(*args, **kwargs))


def validate_build(torch):
    from checks.memfabric_build_info import identity
    op = torch.ops.sparse_kv_plan_op
    if not hasattr(op, 'memfabric_plan_info'):
        raise RuntimeError('Shared library lacks MemFabric identity; rebuild this branch')
    actual = json.loads(op.memfabric_plan_info())
    if actual != identity():
        raise RuntimeError('MemFabric source/adapter identity is stale; rebuild this branch')
    return actual


def prepare_npu(torch, case, unused, device, names=IMPLEMENTATIONS):
    validate_case(case)
    # Reuse the exact ABCDF allocation, input upload and initial public snapshot.
    inputs, initial, states = plan.prepare_npu(torch, case, None, device, ('abcdf',))
    if 'abcdf' not in names:
        states.pop('abcdf')
    if 'memfabric' in names:
        initial['memfabric'] = initial['abcdf']
        state = {k: v.clone() for k, v in initial['memfabric'].items()}
        r, t, c, length = (case[k] for k in ('rows', 'topk', 'capacity', 'max_token'))
        for key in SIMT_OUTPUT:
            state[key] = torch.full((r,) if key == 'miss_count' else (r, t), -777,
                                    dtype=torch.int32, device=device)
        state['token_mark'] = torch.zeros((8, length), dtype=torch.int32, device=device)
        state['token_pos'] = torch.zeros((8, length), dtype=torch.int32, device=device)
        state['epochs'] = torch.zeros(8, dtype=torch.int32, device=device)
        # Generic profiler audits poison this temporary array. Marks/epochs stay
        # paired and advance, so old marks can never look like current hits.
        state['workspace'] = state['token_pos']
        state['layout'] = legacy_layout(t, c, r, length)
        state['_calls'] = 0
        state['_rows'] = r
        states['memfabric'] = state
    initial = {n: initial[n] for n in names}
    torch.npu.synchronize()
    return inputs, initial, states


def invoke(torch, name, case, inputs, state):
    if name == 'abcdf':
        return plan.invoke(torch, name, case, inputs, state)
    torch.ops.sparse_kv_plan_op.memfabric_lru_plan(
        inputs['req_ids'], state['last_req_ids'], inputs['topk_indices'], inputs['stable_prefix_lens'],
        state['slot_to_token'], state['lru_slots'], state['current_slots'], state['miss_count'],
        state['miss_tokens'], state['miss_slots'], state['token_mark'], state['token_pos'], state['epochs'],
        case['rows'], case['topk'], case['capacity'], case['max_token'])
    state['_calls'] += 1


def check_npu_result(name, state, expected, phase):
    actual = plan.check_npu_result(name, state, expected, phase)
    if name == 'memfabric':
        per_core = np.array([len(range(i, state['_rows'], 8)) for i in range(8)], np.int64)
        epochs = state['epochs'].cpu().numpy()
        expected_epochs = per_core * state['_calls']
        if np.any(expected_epochs >= 1 << 30):
            raise RuntimeError('Capture too long: epoch overflow path requires separate testing')
        np.testing.assert_array_equal(epochs, expected_epochs, err_msg=phase + '/epochs')
    return actual


def audit_sequence(torch, case, device, steps=4):
    """Continuous NPU state: update only read-only inputs; never rewrite from golden."""
    names = IMPLEMENTATIONS
    inputs, initial, states = prepare_npu(torch, case, None, device, names)
    current = dict(case)
    for step in range(steps):
        validate_case(current)
        expected = {name: reference_simt(current) for name in names}
        for key in ('req_ids', 'topk_indices', 'stable_prefix_lens'):
            inputs[key].copy_(torch.from_numpy(np.ascontiguousarray(current[key])).to(device))
        for name in names:
            invoke(torch, name, current, inputs, states[name])
        torch.npu.synchronize()
        for name in names:
            check_npu_result(name, states[name], expected, f'continuous-{step}')
        current = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in current.items()}
        current.update({k: expected['abcdf'][k].copy() for k in SIMT_STATE})
        if step == 0:  # rollback suffix while keeping request ID
            current['stable_prefix_lens'][:] = current['max_token'] // 2
        elif step == 1:  # new legal query with historical state retained
            current['topk_indices'] = ((current['topk_indices'].astype(np.int64) + 7) % current['max_token']).astype(np.int32)
        elif step == 2:  # request reset
            current['req_ids'] += 100
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--library', type=Path, default=ROOT / 'build/libsimt_plan_variants_ops.so')
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    # Both small tail-core cases and production dimensions; randomized historical order.
    cases = [make_case(r, t, c, length, s, 2026, 64, True)
             for r,t,c,length in [(1,4,8,64),(9,4,8,64),(32,2048,4096,262144)] for s in SCENARIOS]
    if args.cpu_check:
        for case in cases: reference_simt(case)
        print('PASS: 12 shared-domain inputs and reference cases; no kernel execution')
        return
    import torch
    import torch_npu
    if not torch.npu.is_available(): parser.error('No NPU available')
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    from checks.plan_build_info import catalog_identity
    if json.loads(torch.ops.sparse_kv_plan_op.plan_variants_info()) != catalog_identity():
        parser.error('ABCDF library identity is stale; rebuild')
    validate_build(torch)
    device = torch.device(f'npu:{args.device}')
    for case in cases:
        steps = audit_sequence(torch, case, device)
        print(f"PASS rows={case['rows']} topk={case['topk']} {case['scenario']}: {steps} continuous steps")
    print('Correctness passed. Use profile_plan_variants.py --comparison-module compare_memfabric_plan.py for kernel timing.')
if __name__ == '__main__': main()
