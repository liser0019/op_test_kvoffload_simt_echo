#!/usr/bin/env python3
"""Server-only six-way gate: strict existing checks plus cold/warm/mode continuations."""
from pathlib import Path
import argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import joint_plan_common as common
from joint_variants import COLD_COMPARE_NAMES
from joint_plan_cases import make_case,sort_topk
from checks.check_union_next_npu import gate as common_gate
from harness_io import preflight_args,write_json,record_failure


def transition_gate(torch,names,failure_dir):
    records=[]
    for mtp in (0,1,3):
        base=sort_topk(make_case(2,65,128,262144,mtp,'cold_reset','shared',block_size=8))
        for name in names:
            b=common.prepare(torch,base,name)
            occupied=set(map(int,base['topk_indices'].ravel()))
            missing=np.array([t for t in range(65536,131072) if t not in occupied][:base['topk']],np.int32)
            try:
                for step,label in enumerate(('cold','warm','warm_all_miss','new_request','new_request_warm','split_or_duplicate','join','join_warm')):
                    current={k:v.copy() if hasattr(v,'copy') else v for k,v in base.items()}
                    if step>=3:current['req_ids']=base['req_ids']+17
                    if step==2:current['topk_indices'][:]=missing
                    if step==5:
                        if mtp:current['token_to_req'][:]=[i%2 for i in range(current['rows'])]
                        else:current['topk_indices'][:,1]=current['topk_indices'][:,0]
                    current=sort_topk(current)
                    b['phase']=f'cold-transition/mtp{mtp}/{step+1}/{label}'
                    b['workload']={k:current.get(k) for k in b['workload']}
                    # Only inputs/output poison change here; persistent state is never reset.
                    common.update_inputs(torch,b,current);common.poison(b);torch.npu.synchronize()
                    before=common.snapshot(b)
                    try:
                        common.invoke(b);torch.npu.synchronize();result=common.check(current,b,before)
                    except BaseException:
                        common.dump_failure(current,b,failure_dir);raise
                    records.append({'implementation':name,'mtp':mtp,'step':step+1,'label':label,'audit':result})
            finally:common.close(b)
            print('cold continuation passed:',name,'MTP',mtp,flush=True)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p)
    p.add_argument('--implementations',nargs='+',choices=COLD_COMPARE_NAMES,default=list(COLD_COMPARE_NAMES))
    p.add_argument('--regression-repeat',type=int,default=5);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.regression_repeat<=0 or len(set(a.implementations))!=len(a.implementations):p.error('positive repeats; distinct implementations')
    storage=preflight_args(a,a.report.parent);out={'status':'running','storage':storage}
    try:
        torch,_,env=common.load_runtime(a);out['environment']=env
        result=common_gate(torch,a.implementations,a.regression_repeat,a.report.parent/'failure_state')
        result['cold_transitions']=transition_gate(torch,a.implementations,a.report.parent/'failure_state')
        result['validation_scope']='cold_finish_six_way_v1; existing strict gate + 8-step cold/warm/all-miss/mode transitions'
        out.update(status='passed',gate=result);write_json(a.report,out)
    except BaseException as exc:record_failure(a.report,out,exc);raise
if __name__=='__main__':main()
