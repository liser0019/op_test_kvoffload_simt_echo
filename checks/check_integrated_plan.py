#!/usr/bin/env python3
"""Integrated512 vs frozen parents, actual controller with CPU API model + UBSan only."""
from pathlib import Path
import argparse,ctypes,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_joint_variant_kernels import build,execute
from checks.check_cold_finish import extra_cases,load,check_copies
from checks.check_hardware_sweep import cases
from checks.joint_build_info import expected_integrated_stage
from joint_plan_cases import make_case,sort_topk,initial_arrays


def integration_cases():
    yield from cases();yield from extra_cases()
    # Nonuniform contiguous group sizes: singleton + 2 + 3 + singleton + >4.
    c=make_case(4,8,32,262144,3,'cold_reset','shared',block_size=8)
    c['token_to_req'][:]=[0,1,1,2,2,2,3,0,0,0,0,0,1,2,3,0]
    c['topk_indices'][:]=np.arange(8)+150000
    yield c
    c=make_case(3,8,16,262144,1,'cold_reset','shared',block_size=8)
    c['token_to_req'][:]=[-1,-1,0,0,3,3];yield c
    c=make_case(4,8,32,262144,0,'cold_reset','shared',block_size=8)
    c['token_to_req'][:]=0;yield c # actual four-row group despite fixture MTP0 label
    # One AIV handles many successive mixed-size groups, reusing ctx64 scratch.
    c=make_case(17,8,32,262144,3,'cold_reset','shared',block_size=8)
    c['token_to_req'][:]=np.arange(c['rows'])%c['requests'];yield c
    for mtp in (0,1,3):
        for scene in ('cold_reset','full_cache_75pct'):
            yield make_case(1,2048,4096,262144,mtp,scene,'overlap40')


def compare(candidate,parents,c,before,gm):
    parent=parents[0] if c['mtp']==0 else parents[1]
    kw=dict(turbo=True,merge=True,sorted_union=True)
    reference=execute(parent,c,True,gm,before,union_next=c['mtp']==0,**kw)
    actual=execute(candidate,c,True,gm,before,single_sorted=True,**kw)
    for key in before:np.testing.assert_array_equal(actual[key],reference[key],err_msg='exact-parent/'+key)
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path);a=p.parse_args()
    check_copies();base=ROOT/'csrc/joint_variants'
    assert expected_integrated_stage((base/'cold_ac_stage.h').read_text())==(base/'integrated_stage.h').read_text()
    checks=0
    with tempfile.TemporaryDirectory(prefix='integrated512_cpu_') as tmp:
        d=Path(tmp);parents=[load(d/'union',123),load(d/'ac',124)];candidate=load(d/'integrated',125)
        candidate.cold_calls.restype=ctypes.c_uint32
        for raw in integration_cases():
            c=sort_topk(raw)
            for gm in (False,True):
                before=initial_arrays(c,c['mtp']>0,4*c['topk'] if c['mtp'] else None)
                out=compare(candidate,parents,c,before,gm);checks+=1
                out=compare(candidate,parents,c,out,gm);checks+=1
                c=dict(c,req_ids=c['req_ids']+31)
                compare(candidate,parents,c,out,gm);checks+=1
        # Fixed shapes, actual returned state: joint -> split private -> rejoin.
        for mtp in (0,1,3):
            c=sort_topk(make_case(2,65,128,262144,mtp,'cold_reset','shared',block_size=8))
            c['topk_indices'][:]=np.arange(65)+150000
            out=initial_arrays(c,mtp>0,4*c['topk'] if mtp else None)
            for step in range(8):
                x={k:v.copy() if hasattr(v,'copy') else v for k,v in c.items()}
                if step>=3:x['req_ids']+=17
                if step==2:x['topk_indices'][:]=np.arange(65)+180000
                if step==5:
                    if mtp:x['token_to_req'][:]=np.arange(c['rows'])%2
                    else:x['topk_indices'][:,1]=x['topk_indices'][:,0]
                cold_before=candidate.cold_calls()
                out=compare(candidate,parents,sort_topk(x),out,False);checks+=1
                if step in (1,2,4,7):assert candidate.cold_calls()==cold_before,'warm/all-miss must not trigger reset shortcut'
                if step==0:assert candidate.cold_calls()-cold_before==2
                if step==3:assert candidate.cold_calls()-cold_before==2
        print('integrated512:',checks,'exact-parent comparisons passed; CPU API model + UBSan, NOT CANN/NPU',flush=True)
    if a.report:a.report.write_text(json.dumps({'scope':'CPU API model + UBSan only; no CANN/NPU','exact_parent_comparisons':checks},indent=2)+'\n')
if __name__=='__main__':main()
