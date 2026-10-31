#!/usr/bin/env python3
"""同一MTP0输入，明确调用序号；支持用旧源码身份验证旧库，不改写旧checkout。"""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--implementation', choices=['abcdf', 'joint_bitmap'], required=True)
    parser.add_argument('--repeat', type=int, choices=[1, 5], default=5)
    parser.add_argument('--report', type=Path, required=True)
    preliminary, _ = parser.parse_known_args()
    source = preliminary.source_root.resolve()
    sys.path.insert(0, str(source))
    import joint_plan_common as common
    from joint_plan_cases import make_case, describe
    from harness_io import preflight_args, write_json, record_failure
    common.add_runtime_arguments(parser)
    parser.set_defaults(device=0, library=source / 'build_joint_plan/libmtp_joint_plan_ops.so')
    args = parser.parse_args()
    if args.device != 0:
        parser.error('本轮固定逻辑NPU0')
    case = make_case(32, 2048, 4096, 262144, 0, 'full_cache_75pct', 'independent', 2026, 64)
    report = {'status': 'running', 'implementation': args.implementation,
              'source_root': str(source), 'workload': describe(case), 'calls': []}
    bundle = None
    try:
        report['storage'] = preflight_args(args, args.report.parent)
        torch, _, report['environment'] = common.load_runtime(args)
        bundle = common.prepare(torch, case, args.implementation)
        for index in range(1, args.repeat + 1):
            common.restore(torch, bundle)
            common.poison(bundle)
            torch.npu.synchronize()
            before = common.snapshot(bundle)
            item = {'call_number': index, 'status': 'about_to_invoke'}
            report['calls'].append(item)
            write_json(args.report, report)
            print(f'{args.implementation}/full_cache_75pct/independent/MTP0 call={index}/{args.repeat}', flush=True)
            bundle['phase'] = f'MTP0-probe-{index}/{args.repeat}'
            common.invoke(bundle)
            torch.npu.synchronize()
            item['correctness'] = common.check(case, bundle, before)
            item['status'] = 'passed'
            write_json(args.report, report)
        common.close(bundle)
        report['status'] = 'passed'
        write_json(args.report, report)
    except BaseException as exc:
        if bundle is not None:
            common.dump_failure(case, bundle, args.report.parent / 'failure_state')
        record_failure(args.report, report, exc)
        raise

if __name__ == '__main__':
    main()
