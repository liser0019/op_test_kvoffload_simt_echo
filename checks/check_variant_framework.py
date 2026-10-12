#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""检查多版本框架接线：真实 CPU Tensor/schema，模拟设备调用；不生成 NPU 性能结论。"""
import contextlib
import ast
import importlib.util
import io
import json
import re
from pathlib import Path
import sys
import tempfile
import time
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT / 'compare.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

# 使用实际注册层的参数顺序和 mutation 标注。仅后端替换成 CPU 参考回调。
def source_schema(path, marker):
    text = path.read_text()
    start = text.index(marker)
    chunk = text[start:text.index(');', start)]
    return ''.join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', chunk))


plan_schema = source_schema(ROOT / 'csrc/simt/sparse_kv_plan_plugin.cpp', 'm.def(SPARSE_KV_VARIANT_OP_NAME')
echo_schema = source_schema(ROOT / 'csrc/echo/echo_lru_plugin.cpp', 'm.def("echo_lru(')
lib = torch.library.Library('sparse_kv_plan_op', 'FRAGMENT')
context = {}
invocations = []
fault = False
counts = {}
real_prepare = b.prepare_npu


def tracked_prepare(*args, **kwargs):
    inputs, initial, states = real_prepare(*args, **kwargs)
    case = args[1]
    context.update(case=case, inputs=inputs, initial=initial)
    context['expected'] = {'simt': b.reference_simt(case)}
    if 'echo' in states:
        context['expected']['echo'] = b.reference_echo(case, args[2])
    counts.clear()
    # immutable 初始快照与任何工作状态不得共享存储；不同实现也不得共享工作状态。
    for key in b.SIMT_STATE:
        ptrs = [s[key].data_ptr() for n, s in states.items() if n != 'echo']
        assert len(ptrs) == len(set(ptrs))
        for n, state in states.items():
            if n != 'echo':
                assert state[key].data_ptr() != initial[n][key].data_ptr()
    return inputs, initial, states


b.prepare_npu = tracked_prepare


def callback(name):
    def run(*args):
        case = context['case']
        inputs = context['inputs']
        counts[name] = counts.get(name, 0) + 1
        if name == 'echo':
            assert args[0].data_ptr() == inputs['topk_indices'].data_ptr()
            assert args[-1] is False and args[-2] > 0
            state_tensors = args[3:9]
            output_tensors = args[9:13]
            state_keys, output_keys = b.ECHO_STATE, b.ECHO_OUTPUT
            expected = context['expected']['echo']
        else:
            for i, key in enumerate(('req_ids', 'topk_indices', 'stable_prefix_lens', 'visible_seq_lens',
                                     'token_to_req', 'block_table', 'active_rows')):
                assert args[i].data_ptr() == inputs[key].data_ptr(), (name, key)
            assert args[15:] == (case['topk'], case['capacity'], case['max_token'], b.BLOCK_SIZE, case['host_num_blocks'])
            state_tensors = args[7:10]
            output_tensors = args[10:14]
            state_keys, output_keys = b.SIMT_STATE, b.SIMT_OUTPUT
            arrays = b.VARIANTS[name]['capacity_arrays'] if name in b.VARIANTS else 2
            h = max(32, 1 << (2 * case['topk'] - 1).bit_length())
            elements = 3 * h + arrays * case['capacity'] + case['topk']
            required = 1 if elements <= 112 * 1024 // 4 else case['rows'] * elements
            assert args[14].numel() == required, (name, 'workspace allocation')
            expected = context['expected']['simt']
        for key, tensor in zip(state_keys, state_tensors):
            torch.testing.assert_close(tensor, context['initial'][name][key], rtol=0, atol=0)
        for key, tensor in zip((*state_keys, *output_keys), (*state_tensors, *output_tensors)):
            tensor.copy_(torch.from_numpy(expected[key]))
        invocations.append((case['scenario'], name))
        if fault and name == '10_bcde' and counts[name] == 4:
            output_tensors[0].view(-1)[0] = -123
    return run


for name in b.IMPLEMENTATIONS:
    op = b.VARIANTS[name]['op_name'] if name in b.VARIANTS else 'sparse_kv_plan'
    if name == 'echo':
        lib.define(echo_schema)
        lib.impl('echo_lru', callback(name), 'CPU')
    else:
        lib.define(op + plan_schema)
        lib.impl(op, callback(name), 'CPU')
lib.define('echo_lru_core_count() -> int')
lib.impl('echo_lru_core_count', lambda: 2, 'CompositeExplicitAutograd')


class Event:
    def __init__(self, **kwargs):
        self.timestamp = 0
    def record(self):
        self.timestamp = time.perf_counter_ns()
    def synchronize(self):
        pass
    def elapsed_time(self, other):
        return (other.timestamp - self.timestamp) / 1e6


real_device = torch.device
torch.device = lambda *args, **kwargs: real_device('cpu')
torch.npu = types.SimpleNamespace(is_available=lambda: True, set_device=lambda n: None,
                                  get_device_name=lambda n: 'CPU protocol stub; NOT NPU',
                                  synchronize=lambda: None, Event=Event)
torch.ops.load_library = lambda path: None
sys.modules['torch_npu'] = types.SimpleNamespace(__version__='CPU-test-stub')


def run(argv):
    sys.argv = ['compare.py', *argv]
    with contextlib.redirect_stdout(io.StringIO()):
        b.main()


with tempfile.TemporaryDirectory(prefix='variant_protocol_') as temporary:
    tmp = Path(temporary)
    library = tmp / 'stub.so'
    library.touch()
    common = ['--library', str(library), '--json', str(tmp / 'result.json'), '--audit-repeat', '2', '--warmup', '1']
    run(common + ['--rows', '2', '--topk', '4', '--capacity', '8', '--max-token', '64',
                  '--repeat', '13', '--verify-each-sample'])
    result = json.loads((tmp / 'result.json').read_text())
    assert len(invocations) == 4 * 13 * (2 + 1 + 13)
    for scenario in result['scenarios'].values():
        assert scenario['correctness'] == 'passed'
        orders = scenario['execution_orders']
        for name in b.IMPLEMENTATIONS:
            assert scenario[name]['event']['count'] == 13
            assert sorted(order.index(name) for order in orders) == list(range(13))
    full_calls = len(invocations)
    invocations.clear()
    run(common + ['--rows', '1', '--topk', '2048', '--capacity', '8192', '--max-token', '16384',
                  '--scenario', 'warm_75pct', '--repeat', '1', '--implementations', '00_baseline', '03_c_compact_partition'])
    result = json.loads((tmp / 'result.json').read_text())['scenarios']['warm_75pct']
    assert result['00_baseline']['layout']['scratch_path'] == 'GM'
    assert result['03_c_compact_partition']['layout']['scratch_path'] == 'UB'
    (tmp / 'result.json').unlink()
    fault = True
    try:
        run(common + ['--rows', '2', '--topk', '4', '--capacity', '8', '--max-token', '64',
                      '--scenario', 'warm_75pct', '--repeat', '1', '--implementations', 'simt', '10_bcde'])
    except AssertionError:
        pass
    else:
        raise AssertionError('final sample corruption was not detected')
    assert not (tmp / 'result.json').exists(), 'failure must not publish performance JSON'
print(f'PASS: 13 CPU schemas, {full_calls} calls across four scenarios; GM/UB split and failure gate checked. Not NPU timing.')
