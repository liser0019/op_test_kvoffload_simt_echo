#!/usr/bin/env python3
"""CPU/UBSan comparison to frozen BD8; primitive tails/ranks and multi-warp counts. NOT NPU."""
from pathlib import Path
import argparse,ctypes,json,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_cold_finish import load
from checks.check_joint_variant_kernels import execute,build
from checks.mtp8_cases import cases,transitions
from checks.check_hot_kernels import aligned
from joint_plan_cases import make_case,sort_topk
from joint_variants import mtp7_config


def primitives(lib,mask=195):
    config=mtp7_config(mask);reuse=config['reuse']
    checks=0;rng=np.random.default_rng(7791);chunk=612 if reuse else 512
    packet_words=(40 if reuse else 48)*chunk
    output=(24 if reuse else 32)*chunk
    middle_b=0 if reuse else 24*chunk
    partial=8*chunk if reuse else 0
    ptr=ctypes.c_void_p
    lib.run_eight_progress.argtypes=[ptr,ptr,ctypes.c_uint32,ctypes.c_uint32]
    for topk in (7,65,2048):
        for n in (0,1,31,32,33,63,64,65,511,512,513,2048,4096,*((4895,4896) if reuse else ())):
            packet,pg,po=aligned(packet_words);ctx,cg,co=aligned(128)
            q=rng.integers(0,8*topk,n,dtype=np.int32);packet[output+1:output+2*n:2]=q
            initial=rng.integers(0,123,8,dtype=np.int32);ctx[48:56]=initial;ctx[30]=output
            expected=ctx.copy();expected[48:56]+=np.bincount(q//topk,minlength=8).astype(np.int32)
            old=packet.copy()
            lib.run_eight_progress(packet.ctypes.data,ctx.ctypes.data,n,topk)
            np.testing.assert_array_equal(ctx,expected);np.testing.assert_array_equal(packet[128:],old[128:])
            assert np.all(pg[:po]==-777) and np.all(pg[po+len(packet):]==-777)
            assert np.all(cg[:co]==-777) and np.all(cg[co+128:]==-777);checks+=1
    if not hasattr(lib,'run_eight_rank'):return {'progress_cases':checks}
    lib.run_eight_rank.argtypes=[ptr,ptr,ctypes.c_uint32,ctypes.c_uint32];ranks=0
    for na,nb in ((1,1),(1,65),(7,1),(8,1),(9,7),(15,17),(17,15),(63,64),(64,65),(127,129),(513,511),(4*chunk,4*chunk)):
        for complete in range(4):
            for kind in ('equal','interleaved','separated','random'):
                a=np.zeros(na,np.int32) if kind=='equal' else np.arange(na,dtype=np.int32)*2 if kind=='interleaved' else np.arange(na,dtype=np.int32) if kind=='separated' else np.sort(rng.integers(0,10000,na,dtype=np.int32))
                b=np.zeros(nb,np.int32) if kind=='equal' else np.arange(nb,dtype=np.int32)*2+1 if kind=='interleaved' else np.arange(nb,dtype=np.int32)+10000 if kind=='separated' else np.sort(rng.integers(0,10000,nb,dtype=np.int32))
                packet,pg,po=aligned(packet_words);ctx,cg,co=aligned(128)
                q0=rng.integers(0,4*2048,na,dtype=np.int32);q1=rng.integers(4*2048,8*2048,nb,dtype=np.int32)
                packet.view(np.float32)[16*chunk:16*chunk+2*na:2]=-a.astype(np.float32);packet[16*chunk+1:16*chunk+2*na:2]=q0
                packet.view(np.float32)[middle_b:middle_b+2*nb:2]=-b.astype(np.float32);packet[middle_b+1:middle_b+2*nb:2]=q1
                ctx[0]=1;ctx[104]=na;ctx[105]=nb;ctx[106]=complete;ctx[48:56]=0
                before=packet.copy();expected_ctx=ctx.copy()
                cut=min([int(v[-1]) for k,v in enumerate((a,b)) if not complete&(1<<k)],default=16777216)
                n0=int(np.searchsorted(a,cut,side='right'));n1=int(np.searchsorted(b,cut,side='right'))
                values=np.r_[a[:n0],b[:n1]];ids=np.r_[q0[:n0],q1[:n1]];order=np.argsort(values,kind='stable');n=n0+n1
                expected_ctx[4]=n;expected_ctx[107]=n0;expected_ctx[108]=n1;expected_ctx[48:56]=np.bincount(ids//2048,minlength=8)
                lib.run_eight_rank(packet.ctypes.data,ctx.ctypes.data,chunk,2048)
                np.testing.assert_array_equal(ctx,expected_ctx)
                np.testing.assert_array_equal(-packet.view(np.float32)[output:output+2*n:2],values[order])
                np.testing.assert_array_equal(packet[output+1:output+2*n:2],ids[order])
                unchanged=np.ones(packet_words,dtype=bool)
                unchanged[partial:partial+128]=False;unchanged[output:output+2*n]=False
                np.testing.assert_array_equal(packet[unchanged],before[unchanged],err_msg='live mids, unused outputs and non-partial scratch must remain intact')
                assert np.all(pg[:po]==-777) and np.all(pg[po+len(packet):]==-777)
                assert np.all(cg[:co]==-777) and np.all(cg[co+128:]==-777);ranks+=1
    return {'progress_cases':checks,'rank_prefix_cases':ranks}


def worker(mask):
    with tempfile.TemporaryDirectory(prefix='mtp7_parallel_cpu_') as tmp:
        path=Path(tmp);ref=load(path/'ref',193);lib=load(path/'candidate',mask)
        count=0;kw=dict(turbo=True,merge=True,sorted_union=True,single_sorted=True,mtp8=True)
        for label,c in cases(include_extreme=True):
            before=None
            for step in range(3):
                current=dict(c,req_ids=c['req_ids']+41) if step==2 else c
                expected=execute(ref,current,True,False,before,**kw);actual=execute(lib,current,True,False,before,**kw)
                for k in expected:np.testing.assert_array_equal(actual[k],expected[k],err_msg=f'{mask}/{label}/{k}')
                before=actual;count+=1
        before=None
        for label,c in transitions():
            expected=execute(ref,c,True,False,before,**kw);actual=execute(lib,c,True,False,before,**kw)
            for k in expected:np.testing.assert_array_equal(actual[k],expected[k],err_msg=f'{mask}/{label}/{k}')
            before=actual;count+=1
        for mtp in (0,1,3,7):
            for scenario in ('cold_reset','full_cache_75pct'):
                c=sort_topk(make_case(2,2048,4096,262144,mtp,scenario,'overlap40'));before=None
                for step in range(2):
                    expected=execute(ref,c,True,False,before,**kw);actual=execute(lib,c,True,False,before,**kw)
                    for k in expected:np.testing.assert_array_equal(actual[k],expected[k],err_msg=f'{mask}/production/{k}')
                    before=actual;count+=1
        rng=np.random.default_rng(9023)
        for i in range(32):
            t=int(rng.choice([7,65,129,513,2048]));cap=min(4096,((t*int(rng.choice([1,2,4]))+7)//8)*8)
            c=sort_topk(make_case(1,t,cap,262144,7,'cold_reset' if i%2 else 'full_cache_75pct',str(rng.choice(['shared','overlap40','disjoint'])),seed=i+91,block_size=8))
            if i%3==0:c['topk_indices'][:]=np.sort(rng.integers(-1,max(t*3,8),(8,t),dtype=np.int32),axis=1)
            expected=execute(ref,c,True,False,**kw);actual=execute(lib,c,True,False,**kw)
            for k in expected:np.testing.assert_array_equal(actual[k],expected[k],err_msg=f'{mask}/random/{k}')
            count+=1
        merge_calls={}
        c=sort_topk(make_case(1,2048,4096,262144,7,'full_cache_75pct','overlap40'))
        for label,obj in (('frozen193',ref),('candidate',lib)):
            obj.eight_cpu_merge_calls.restype=ctypes.c_uint64;obj.eight_cpu_reset_merge_calls()
            execute(obj,c,True,False,**kw);count+=1
            merge_calls[label]=obj.eight_cpu_merge_calls()
        multi=path/'multi' ;multi.mkdir();wide=ctypes.CDLL(str(build(multi,mask,model_threads=256)))
        return {'mask':mask,'exact_controller_comparisons':count-2,'controller_executions_for_flow_count':2,'multiwarp_primitives':primitives(wide,mask),'CPU_MrgSort_calls_one_request':merge_calls,'scope':'CPU + UBSan; not CANN/NPU'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=(194,195));a=p.parse_args()
    if a.worker:print(json.dumps(worker(a.worker)),flush=True);return
    for m in (194,195):subprocess.run([sys.executable,'-B',__file__,'--worker',str(m)],check=True)
if __name__=='__main__':main()
