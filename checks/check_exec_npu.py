#!/usr/bin/env python3
"""Server-only A/B/C/D gate; full values/inputs/guards/fallback/continuous-state validation."""
from pathlib import Path
import argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import joint_plan_common as common
from joint_variants import EXEC_NAMES,EXEC_REFS
from joint_plan_cases import make_case,sort_topk
from checks.check_union_next_npu import gate as common_gate
from checks.check_cold_finish_npu import transition_gate
from checks.check_phase0_bounds import boundary_cases
from harness_io import preflight_args,write_json,record_failure


def targeted_gate(torch,names,failure_dir):
    records=[];cases=list(boundary_cases(compact=True))
    for topk in (63,64,65,127,128,129):
        cap=((4*topk+7)//8)*8
        c=make_case(1,topk,cap,262144,3,'cold_reset','shared',block_size=8)
        c['topk_indices'][:]=np.arange(topk,dtype=np.int32)//3 # repeated heads across packet/vector seams
        cases.append((f'SIMD_seams/topk{topk}',c))
    for label,raw in cases:
        c=sort_topk(raw);inputs=common.upload_inputs(torch,c)
        for name in names:
            b=common.prepare(torch,c,name,inputs)
            b['workload']['exec_case']=label
            print('execution targeted:',name,label,flush=True)
            try:records.append({'case':label,'implementation':name,'audit':common.audit(torch,c,b,1,failure_dir)})
            finally:common.close(b)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p);p.set_defaults(device=0)
    p.add_argument('--implementations',nargs='+',choices=tuple(dict.fromkeys((*EXEC_REFS,*EXEC_NAMES))),default=[*EXEC_REFS[:-1],*EXEC_NAMES])
    p.add_argument('--regression-repeat',type=int,default=5);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.regression_repeat<=0 or len(a.implementations)!=len(set(a.implementations)):p.error('positive repeats and distinct implementations')
    storage=preflight_args(a,a.report.parent);out={'status':'running','storage':storage}
    try:
        torch,_,env=common.load_runtime(a);out['environment']=env
        failure_dir=a.report.parent/'failure_state'
        gate=common_gate(torch,a.implementations,a.regression_repeat,failure_dir)
        gate['execution_targeted']=targeted_gate(torch,a.implementations,failure_dir)
        gate['cold_transitions']=transition_gate(torch,a.implementations,failure_dir)
        gate['validation_scope']='execution_abcd_v1; strict common gate + exact boundary/seam semantics + 8-step state continuations'
        out.update(status='passed',gate=gate);write_json(a.report,out)
    except BaseException as exc:record_failure(a.report,out,exc);raise
if __name__=='__main__':main()
