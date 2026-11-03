#!/usr/bin/env python3
"""Phase0 endpoints candidate: exact Stage0 bounds + full controller/UBSan CPU tests, NOT NPU."""
from pathlib import Path
import argparse,ctypes,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_cold_finish import load
from checks.check_integrated_plan import integration_cases
from checks.check_joint_variant_kernels import execute
from checks.joint_build_info import expected_phase0_bounds_stage
from joint_plan_cases import make_case,sort_topk,initial_arrays


def boundary_cases(compact=False):
    patterns=[('all_valid',[0,1,2,3,4,5,6],7),
      ('negative_prefix',[-4,-3,-1,0,2,3,5],6),('at_upper',[0,1,2,3,4,7,8],7),
      ('both_ends',[-3,-1,0,1,6,7,9],7),('all_negative',[-7,-6,-5,-4,-3,-2,-1],200),
      ('all_future',[7,8,9,10,11,12,13],7),('upper_zero',[0,1,2,3,4,5,6],0),
      ('upper_negative',[0,1,2,3,4,5,6],-10),
      ('upper_clipped_to_max',[0,1,4093,4094,4095,4096,5000],5000),
      ('duplicates',[0,0,1,2,2,3,4],5)]
    for index,(label,values,visible) in enumerate(patterns):
        mtps=((0,1,3)[index%3],) if compact else (0,1,3)
        for mtp in mtps:
            c=make_case(1,7,16,4096,mtp,'full_cache_75pct','shared',block_size=8)
            c['topk_indices'][:]=values;c['visible_seq_lens'][:]=visible
            yield f'{label}/mtp{mtp}',c
    for token in (-1,0,7):
        c=make_case(1,1,8,4096,0,'cold_reset','shared',block_size=8)
        c['topk_indices'][:]=token;c['visible_seq_lens'][:]=7
        yield f'one_element/{token}',c
    c=make_case(1,7,16,4096,3,'cold_reset','shared',block_size=8)
    c['topk_indices'][:]=np.arange(7);c['visible_seq_lens'][:]=[1,3,6,7]
    yield 'row_specific_causal_upper',c
    c=make_case(2,7,16,4096,3,'full_cache_75pct','shared',block_size=8)
    c['topk_indices'][:]=np.arange(7);c['topk_indices'][0]=[-3,-1,0,1,6,7,9]
    c['visible_seq_lens'][:]=[7,8,9,10,-1,0,3,7]
    yield 'mixed_fast_and_fallback_rows',c


def probe_bounds(lib):
    lib.run_phase0_bounds.argtypes=[ctypes.POINTER(ctypes.c_int32),ctypes.c_uint32,ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
    lib.run_phase0_bounds.restype=None
    rng=np.random.default_rng(2026);checks=0
    vectors=[np.array([],np.int32),np.array([-1],np.int32),np.array([0],np.int32),
      np.array([5],np.int32),np.arange(2048,dtype=np.int32),np.array([-5,-1,0,0,1,7,7,262144],np.int32)]
    vectors += [np.sort(rng.integers(-10,80,size=int(rng.integers(1,129)),dtype=np.int32)) for _ in range(32)]
    for values in vectors:
        # Exact equality at visible boundary and clipping to [0,maxToken].
        uppers=[-9,0,1,7,262144,262150]
        if values.size:uppers += [int(values[-1]),int(values[-1])+1]
        for visible in uppers:
            out=np.zeros(2,np.int32)
            lib.run_phase0_bounds(values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),len(values),visible,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            expected=[np.searchsorted(values,0),np.searchsorted(values,max(0,min(262144,visible)))]
            np.testing.assert_array_equal(out,expected,err_msg=f'actual Phase0 interval/visible={visible}')
            checks+=1
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path);a=p.parse_args()
    base=ROOT/'csrc/joint_variants'
    assert expected_phase0_bounds_stage((base/'integrated_stage.h').read_text())==(base/'phase0_bounds_stage.h').read_text()
    comparisons=0;kw=dict(turbo=True,merge=True,sorted_union=True,single_sorted=True)
    with tempfile.TemporaryDirectory(prefix='phase0_bounds_cpu_') as tmp:
        d=Path(tmp);parent=load(d/'parent',125);candidate=load(d/'bounds',126)
        probes=probe_bounds(candidate)
        for raw in [*integration_cases(),*(c for _,c in boundary_cases())]:
            c=sort_topk(raw)
            for gm in (False,True):
                before=initial_arrays(c,True,4*c['topk'])
                for step in range(3):
                    current=dict(c,req_ids=c['req_ids']+31) if step==2 else c
                    expected=execute(parent,current,True,gm,before,**kw)
                    actual=execute(candidate,current,True,gm,before,**kw)
                    for key in before:np.testing.assert_array_equal(actual[key],expected[key],err_msg='exact-integrated-parent/'+key)
                    before=actual;comparisons+=1
        result={'scope':'CPU API model + UBSan only; not CANN/NPU','phase0_interval_probes':probes,'full_controller_parent_comparisons':comparisons}
        print(json.dumps(result,ensure_ascii=False),flush=True)
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
