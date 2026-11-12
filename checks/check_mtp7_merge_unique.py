#!/usr/bin/env python3
"""CPU/UBSan exact state comparisons of two merge experiments against frozen196."""
from pathlib import Path
import argparse,json,subprocess,sys,tempfile,ctypes
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_mtp7_fast import worker
from checks.check_cold_finish import load
from checks.check_hot_kernels import aligned
from checks.check_joint_variant_kernels import execute,build
from joint_plan_cases import make_case,sort_topk


def primitives(mask):
    checks=0;rng=np.random.default_rng(3859);ptr=ctypes.c_void_p
    with tempfile.TemporaryDirectory(prefix='merge_unique_primitives_') as tmp:
        root=Path(tmp);reference=load(root/'reference',196);candidate=load(root/'candidate',mask)
        if mask==204:
            reference.run_refine_unique.argtypes=[ptr]*3+[ctypes.c_uint32]*4+[ctypes.c_int32,ctypes.c_uint32]
            candidate.run_fused_unique.argtypes=[ptr]*3+[ctypes.c_uint32]*4+[ctypes.c_int32]+[ctypes.c_uint32]*2
            for topk in (7,65,513,2048):
                q=8*topk;capacity=4096
                for n in sorted(set(min(q,v) for v in (0,1,31,32,33,63,64,65,127,129,511,512,513,4896))):
                    for duplicate in (False,True):
                        original,og,oo=aligned(q+capacity+24576);data,dg,do=aligned(len(original));data[:]=original
                        ctx,cg,co=aligned(128);initial=rng.integers(0,21,8,dtype=np.int32)
                        ctx[48:56]=initial;ctx[56:64]=initial+topk
                        expected,eg,eo=aligned(128);expected[:]=ctx
                        records,rg,ro=aligned(max(128,2*n+128));qid=rng.permutation(q)[:n].astype(np.int32)
                        records.view(np.float32)[:2*n:2]=-(np.arange(n)//(3 if duplicate else 1)+100).astype(np.float32)
                        records[1:2*n:2]=qid
                        reference.run_refine_unique(records.ctypes.data,original.ctypes.data,expected.ctypes.data,n,q,capacity,3,100,1)
                        candidate.run_fused_unique(records.ctypes.data,data.ctypes.data,ctx.ctypes.data,n,q,capacity,3,100,1,topk)
                        expected[48:56]+=np.bincount(qid//topk,minlength=8).astype(np.int32)
                        expected[6]=int(np.any(expected[48:56]<expected[56:64]))
                        np.testing.assert_array_equal(data,original)
                        np.testing.assert_array_equal(ctx,expected)
                        for guard,offset,size in ((og,oo,len(original)),(dg,do,len(data)),(cg,co,128),(eg,eo,128),(rg,ro,len(records))):
                            assert np.all(guard[:offset]==-777) and np.all(guard[offset+size:]==-777)
                        checks+=1
        else:
            candidate.run_mid_pack.argtypes=[ptr]*3+[ctypes.c_uint32]*4
            chunk=612;topk=2048;q=8*topk;words=40*chunk+96
            for na,nb in ((0,0),(1,1),(31,33),(63,65),(127,129),(511,513),(4*chunk,4*chunk)):
                for duplicate in (False,True):
                    packet,pg,po=aligned(words);map_,mg,mo=aligned(q+4096+24576);ctx,cg,co=aligned(128)
                    old_packet=packet.copy();old_map=map_.copy();old_ctx=ctx.copy()
                    compact=[]
                    for half,n in enumerate((na,nb)):
                        token=np.arange(n,dtype=np.int32)//(3 if duplicate else 1)+91
                        qid=(rng.permutation(4*topk)[:n]+half*4*topk).astype(np.int32)
                        off=0 if half else 16*chunk
                        packet.view(np.float32)[off:off+2*n:2]=-token.astype(np.float32)
                        packet[off+1:off+2*n:2]=qid
                        unique,inverse=np.unique(token,return_inverse=True)
                        mids=inverse+half*4*chunk;old_map[qid]=-(mids+2)
                        out=(24+8*half)*chunk
                        old_packet.view(np.float32)[out:out+2*len(unique):2]=-unique.astype(np.float32)
                        old_packet[out+1:out+2*len(unique):2]=np.arange(len(unique))+half*4*chunk
                        old_ctx[104+half]=len(unique)
                        compact.append((off,n))
                    # Preserve rawB; rawA becomes the cleared second-level map after all its reads.
                    old_packet[:8*chunk]=packet[:8*chunk]
                    old_packet[16*chunk:24*chunk]=-1
                    candidate.run_mid_pack(packet.ctypes.data,map_.ctypes.data,ctx.ctypes.data,q,chunk,na,nb)
                    np.testing.assert_array_equal(packet,old_packet)
                    np.testing.assert_array_equal(map_,old_map)
                    np.testing.assert_array_equal(ctx,old_ctx)
                    for guard,offset,size in ((pg,po,words),(mg,mo,len(map_)),(cg,co,128)):
                        assert np.all(guard[:offset]==-777) and np.all(guard[offset+size:]==-777)
                    checks+=1
        # Cross-warp execution of the actual controller and new MID resolver.
        wide=root/'wide';wide.mkdir();lib=ctypes.CDLL(str(build(wide,mask,model_threads=256)))
        for scenario in ('cold_reset','full_cache_75pct'):
            c=sort_topk(make_case(1,65,256,262144,7,scenario,'shared',block_size=8))
            kw=dict(turbo=True,merge=True,sorted_union=True,single_sorted=True,mtp8=True)
            before=None
            for step in range(2):
                expected=execute(reference,c,True,False,before,**kw);actual=execute(lib,c,True,False,before,**kw)
                for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
                before=actual;checks+=1
    return checks


def merge_work(mask):
    """Observe CPU-model merge inputs only; never instrument device kernels."""
    counts={}
    with tempfile.TemporaryDirectory(prefix='mtp7_merge_work_') as tmp:
        c=sort_topk(make_case(1,2048,4096,262144,7,'cold_reset','shared'))
        # Dense within-half duplicates exercise the actual intended reduction.
        for row in range(8):c['topk_indices'][row]=np.arange(2048)//3+100000+(row%2)*100
        for name,m in (('reference',196),('candidate',mask)):
            lib=load(Path(tmp)/name,m)
            lib.eight_cpu_two_way_input_records.restype=ctypes.c_uint64
            lib.eight_cpu_reset_merge_calls()
            execute(lib,c,True,False,turbo=True,merge=True,sorted_union=True,single_sorted=True,mtp8=True)
            counts[name]=lib.eight_cpu_two_way_input_records()
        if mask==203:assert counts['candidate']<counts['reference'],counts
        else:assert counts['candidate']==counts['reference'],counts
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=(203,204));a=p.parse_args()
    if a.worker:
        result=worker(a.worker,reference_mask=196,include_prefix=True)
        result['dedicated_primitives_and_wide_controller']=primitives(a.worker)
        result['CPU_two_way_input_records_dense_duplicates']=merge_work(a.worker)
        print(json.dumps(result),flush=True);return
    for m in (203,204):subprocess.run([sys.executable,'-B',__file__,'--worker',str(m)],check=True)
if __name__=='__main__':main()
