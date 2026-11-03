#!/usr/bin/env python3
"""Server-only new ABCD/thread/single-sorted gate; strict old and new protocol checks."""
from pathlib import Path
import argparse, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import joint_plan_common as common
from joint_variants import UNION_NAMES,COLD_NAMES
from joint_plan_cases import make_case,sort_topk
from checks.check_joint_plan import continuous_gate,context_gate,input_write_regression
from harness_io import preflight_args,write_json,record_failure


def gate(torch,names,repeats,failure_dir):
    regression=input_write_regression(torch,repeats,failure_dir,names)
    records=[]
    cases=[]
    for mtp in (0,1,3):
        for scene in ('cold_reset','full_cache_75pct'):
            cases.append((f'small/{scene}/mtp{mtp}',make_case(2,65,128,262144,mtp,scene,'overlap40',block_size=8)))
    cases.append(('overflow_disjoint',make_case(1,65,128,262144,3,overlap='disjoint',block_size=8)))
    for blocked in (False,True):
        c=make_case(1,7,16,262144,3,block_size=8)
        c['slot_to_token'][:]=[200000 if blocked or i<2 else 200000+i for i in range(16)]
        c['topk_indices'][:]=[200000,200000,200001,200002,200002,200020,200021]
        cases.append((f'duplicate_resident/{blocked}',c))
    c=make_case(2,65,128,262144,3,block_size=8);c['active_rows']=5
    c['topk_indices'][0,:6]=[-1,262144,c['context']+2,5,5,6];c['stable_prefix_lens'][0]=-1
    cases.append(('causal_duplicates_partial_active',c))
    for scene in ('cold_reset','full_cache_75pct'):
        cases.append((f'production_ub/{scene}',make_case(1,2048,4096,262144,3,scene,'overlap40')))
    cases.append(('natural_gm',make_case(1,33,32768,262144,1,block_size=64)))
    for duplicate in (False,True):
        c=make_case(2,65,128,262144,0,block_size=8)
        c['topk_indices'][0,:4]=[-1,262144,c['context']+3,5]
        if duplicate:c['topk_indices'][1,1]=c['topk_indices'][1,0]
        cases.append((f'single_sorted_invalid_duplicate/{duplicate}',c))
    if any(n in (*COLD_NAMES,"joint_integrated_512") for n in names):
        c=make_case(257,8,16,262144,1,'full_cache_75pct','shared',block_size=8)
        c['topk_indices'][:]=np.arange(8)+100000;c['last_req_ids'][::4]=-1
        cases.append(('many_groups_mixed_cold_warm',c))
    if "joint_integrated_512" in names:
        c=make_case(4,8,32,262144,3,'cold_reset','shared',block_size=8)
        c['token_to_req'][:]=[0,1,1,2,2,2,3,0,0,0,0,0,1,2,3,0]
        c['topk_indices'][:]=np.arange(8)+150000
        cases.append(('integrated_nonuniform_groups_1_2_3_overflow5',c))
    for label,raw in cases:
        c=sort_topk(raw);inputs=common.upload_inputs(torch,c);record={'case':label,'implementations':[]}
        for name in names:
            b=common.prepare(torch,c,name,inputs)
            try:
                record['implementations'].append({'name':name,'audit':common.audit(torch,c,b,1,failure_dir)})
            finally:common.close(b)
        records.append(record);print('ABCD targeted gate passed:',label,flush=True)
    if "joint_integrated_512" in names:
        # The new adapter/golden must follow actual groups, not a fixture's MTP label.
        c=sort_topk(make_case(4,8,32,262144,0,'cold_reset','shared',block_size=8))
        c['token_to_req'][:]=[0,1,1,1]
        b=common.prepare(torch,c,'joint_integrated_512')
        try:
            records.append({'case':'integrated_actual_groups_despite_mtp0_label','implementations':[
                {'name':'joint_integrated_512','audit':common.audit(torch,c,b,1,failure_dir)}]})
        finally:common.close(b)
    return {'status':'passed','validation_scope':'union_next_targeted_v1; strict values/guards/tail contract; continuous joint/private transitions',
            'implementations':list(names),'input_write_regression':regression,'cases':records,
            'continuous':continuous_gate(torch,failure_dir,names),'context':context_gate(torch,names)}


def main():
    p=argparse.ArgumentParser(description=__doc__);common.add_runtime_arguments(p)
    p.add_argument('--implementations',nargs='+',choices=('abcdf_pages','joint_hw_ac_512','joint_hw_ac_256',*UNION_NAMES,*COLD_NAMES),default=['abcdf_pages','joint_hw_ac_512','joint_hw_ac_256',*UNION_NAMES])
    p.add_argument('--regression-repeat',type=int,default=5);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.regression_repeat<=0 or len(a.implementations)!=len(set(a.implementations)):p.error('positive repeat; no duplicate implementation')
    storage=preflight_args(a,a.report.parent);out={'status':'running','storage':storage}
    try:
        torch,_,env=common.load_runtime(a);out['environment']=env
        out['gate']=gate(torch,a.implementations,a.regression_repeat,a.report.parent/'failure_state')
        out['status']='passed';write_json(a.report,out)
    except BaseException as exc:record_failure(a.report,out,exc);raise
if __name__=='__main__':main()
