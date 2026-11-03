#!/usr/bin/env python3
"""Cold candidates vs exact parent states under CPU API models/UBSan. NOT CANN/NPU."""
from pathlib import Path
import argparse,ctypes,json,tempfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_joint_variant_kernels import build,execute
from checks.check_hardware_sweep import cases
from joint_plan_cases import make_case,sort_topk,initial_arrays


def check_copies():
    for source,target,single in [('union_next_stage.h','cold_union_stage.h',True),('sorted_sweep_stage.h','cold_ac_stage.h',False)]:
        s=(ROOT/'csrc/joint_variants'/source).read_text()
        line=next(x for x in s.splitlines(True) if 'const int64_t reqid=a.reqIds[req];const bool reset=' in x)
        hook='''    if (reset) {
      // Uniform reset: no thread may publish mode/ID until all have captured it.
      asc_syncthreads();
      FinishColdSorted<SORTED_THREADS,SINGLE>(a,first,end,map,unique,ctx,t);
      return;
    }
'''.replace('SINGLE','true' if single else 'false')
        assert (ROOT/'csrc/joint_variants'/target).read_text()==s.replace(line,line+hook),target
    helper=(ROOT/'csrc/joint_variants/cold_finish.h').read_text()
    assert 'SortedScan(' not in helper and 'asc_atomic_' not in helper
    assert 'if(thread==1U)a.rowModes[row]' in helper


def extra_cases():
    # Empty union and full-capacity union, including exact rotation boundaries.
    for mtp in (0,1,3):
        c=make_case(1,8,16,262144,mtp,'cold_reset','overlap40',block_size=8)
        c['topk_indices'][:]=-1;yield c
        c=make_case(1,8 if mtp else 16,16,262144,mtp,'cold_reset','overlap40',block_size=8)
        for row in range(c['rows']):c['topk_indices'][row]=np.arange(c['topk'])+(100+(row%2)*8 if mtp else 100)
        yield c
    c=make_case(7,8,16,262144,1,'full_cache_75pct','shared',block_size=8)
    c['topk_indices'][:]=np.arange(8)+100000;c['last_req_ids'][::4]=-1;yield c
    c=make_case(2,8,16,262144,3,'cold_reset','overlap40',block_size=8);c['active_rows']=0;yield c
    c=make_case(1,8,16,262144,0,'cold_reset',block_size=8);c['topk_indices'][0,:2]=[101,101];yield c


def load(path,mask):
    path.mkdir();lib=ctypes.CDLL(str(build(path,mask)))
    lib.run_variant.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)]
    if mask in (123,124):lib.cold_calls.restype=ctypes.c_uint32
    return lib


def compare(parent,candidate,c,before,gm,single):
    kw=dict(turbo=True,merge=True,sorted_union=True,union_next=single)
    reference=execute(parent,c,c['mtp']>0,gm,before,**kw)
    actual=execute(candidate,c,c['mtp']>0,gm,before,**kw)
    for key in before:np.testing.assert_array_equal(actual[key],reference[key],err_msg='exact-parent/'+key)
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path);a=p.parse_args();check_copies();records=[]
    with tempfile.TemporaryDirectory(prefix='cold_finish_cpu_') as tmp:
        for name,mask,parent_mask,single in [('joint_union_cold_512',123,90,True),('joint_hw_ac_cold_512',124,72,False)]:
            parent=load(Path(tmp)/(name+'_parent'),parent_mask);candidate=load(Path(tmp)/name,mask)
            checks=0;observations=[]
            for raw in [*cases(),*extra_cases()]:
                c=sort_topk(raw)
                for gm in (False,True):
                    before=initial_arrays(c,c['mtp']>0,4*c['topk'] if c['mtp'] else None)
                    out=compare(parent,candidate,c,before,gm,single);checks+=1
                    calls=candidate.cold_calls()
                    out=compare(parent,candidate,c,out,gm,single);checks+=1
                    # A previous private fallback may require a legitimate mode reset
                    # when this next call can join again; exact parent equality is strict.
                    changed=dict(c,req_ids=c['req_ids']+17)
                    compare(parent,candidate,changed,out,gm,single);checks+=1
            # Explicit observed trigger tests with scrambled LRU: all miss is not reset.
            for mtp in (0,1,3):
                for gm in (False,True):
                    c=sort_topk(make_case(2,65,128,262144,mtp,'cold_reset','overlap40',block_size=8))
                    c['topk_indices'][:]=np.arange(65,dtype=np.int32)+150000 # guaranteed fitting union
                    before=initial_arrays(c,mtp>0,4*c['topk'] if mtp else None);n=candidate.cold_calls()
                    out=compare(parent,candidate,c,before,gm,single);checks+=1
                    expected=2 if single or mtp else 0
                    assert candidate.cold_calls()-n==expected,(name,mtp,'cold entry count')
                    observations.append({'mtp':mtp,'gm':gm,'cold_entries':expected})
                    miss_case=dict(c,topk_indices=np.tile(np.arange(65,dtype=np.int32)+180000,(c['rows'],1)))
                    n=candidate.cold_calls();out=compare(parent,candidate,miss_case,out,gm,single);checks+=1
                    assert candidate.cold_calls()==n,'warm all-miss is not a reset'
                    changed=dict(miss_case,req_ids=c['req_ids']+23)
                    n=candidate.cold_calls();out=compare(parent,candidate,changed,out,gm,single);checks+=1
                    assert candidate.cold_calls()-n==expected
                    n=candidate.cold_calls();compare(parent,candidate,changed,out,gm,single);checks+=1
                    assert candidate.cold_calls()==n
            # Continue actual states across joint -> singleton/private -> joint.
            for mtp in (1,3):
                c=sort_topk(make_case(2,8,16,262144,mtp,'cold_reset','shared',block_size=8))
                c['topk_indices'][:]=np.arange(8)+150000
                before=initial_arrays(c,True,4*c['topk'])
                out=compare(parent,candidate,c,before,False,single);checks+=1
                n=candidate.cold_calls();out=compare(parent,candidate,c,out,False,single);checks+=1
                assert candidate.cold_calls()==n
                split=dict(c,token_to_req=np.arange(c['rows'],dtype=np.int32)%2)
                n=candidate.cold_calls();out=compare(parent,candidate,split,out,False,single);checks+=1
                assert candidate.cold_calls()-n==(c['rows'] if single else 0)
                n=candidate.cold_calls();out=compare(parent,candidate,c,out,False,single);checks+=1
                assert candidate.cold_calls()-n==2,'rejoin must reset prior private leaders'
                n=candidate.cold_calls();compare(parent,candidate,c,out,False,single);checks+=1
                assert candidate.cold_calls()==n
            # Real production scratch geometry (one request; CPU timing is irrelevant).
            for mtp in (0,1,3):
                c=sort_topk(make_case(1,2048,4096,262144,mtp,'cold_reset','overlap40'))
                before=initial_arrays(c,mtp>0,4*c['topk'] if mtp else None)
                out=compare(parent,candidate,c,before,False,single);compare(parent,candidate,c,out,False,single);checks+=2
            records.append({'name':name,'exact_parent_comparisons':checks,'observed_cold_entries':observations})
            print(name,checks,'exact parent comparisons passed; CPU only',flush=True)
    if a.report:a.report.write_text(json.dumps({'scope':'CPU API model + UBSan, not CANN or NPU','records':records},indent=2)+'\n')
if __name__=='__main__':main()
