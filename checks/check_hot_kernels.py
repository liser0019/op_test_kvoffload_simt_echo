#!/usr/bin/env python3
"""Real refinement stage/controller under CPU APIs + UBSan; no CANN/NPU/timing validation."""
from pathlib import Path
import argparse,ctypes,json,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_joint_variant_kernels import build,execute
from checks.check_cold_finish import load
from checks.check_integrated_plan import integration_cases
from checks.check_phase0_bounds import boundary_cases
from joint_plan_cases import make_case,sort_topk,initial_arrays
from joint_variants import HOT_SWEEP,hot_config


def aligned(size,fill=-777):
    storage=np.full(size+48,fill,np.int32)
    offset=16+((-storage.ctypes.data//4-16)%8)
    view=storage[offset:offset+size]
    assert size == 0 or view.ctypes.data%32==0
    return view,storage,offset


def primitive_checks(lib):
    lib.run_refine_unique.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint32]*4+[ctypes.c_int32,ctypes.c_uint32]
    lib.run_refine_unique.restype=None
    lib.run_refine_bool.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p]
    lib.run_refine_bool.restype=None;lib.refine_threads.restype=ctypes.c_uint32
    rng=np.random.default_rng(2026);unique_tests=0;scan_tests=0
    for n in (0,1,2,31,32,33,63,64,65,127,128,129,511,512,513,2048,8192):
        for kind in ('same','distinct','mixed'):
            tokens=np.full(n,7,np.int32) if kind=='same' else np.arange(n,dtype=np.int32)+7 if kind=='distinct' else np.sort(rng.integers(7,10000,n,dtype=np.int32))
            q=max(n,1)+17;qids=rng.permutation(q)[:n].astype(np.int32)
            records,guard,offset=aligned(2*n)
            records.view(np.float32)[::2]=-tokens.astype(np.float32);records[1::2]=qids
            original=records.copy()
            for cap in (1,64,4096):
                for initial,last in ((0,-1),(1,7)):
                    data,g,off=aligned(q+cap);ctx,cg,coff=aligned(64);expected=data.copy();ec=ctx.copy()
                    count=initial;prev=last
                    for token,qid in zip(tokens,qids):
                        if int(token)!=prev:
                            if count<cap:expected[q+count]=token
                            count+=1
                        expected[qid]=count-1;prev=int(token)
                    more=n%2
                    ec[0]=int(count<=cap);ec[1]=count;ec[2]=prev;ec[6]=more;ec[29]=1
                    lib.run_refine_unique(records.ctypes.data,data.ctypes.data,ctx.ctypes.data,n,q,cap,initial,last,more)
                    np.testing.assert_array_equal(data,expected);np.testing.assert_array_equal(ctx,ec)
                    np.testing.assert_array_equal(records,original)
                    for buf,start,length in ((g,off,len(data)),(cg,coff,len(ctx)),(guard,offset,len(records))):
                        assert np.all(buf[:start]==-777) and np.all(buf[start+length:]==-777)
                    unique_tests+=1
    # Exact representability boundary and repeated token across a 64-lane seam.
    for tokens in (np.array([0,0,1,16777214,16777215],np.int32),np.full(129,16777215,np.int32)):
        n=len(tokens);q=n;cap=n;records,_,_=aligned(2*n);records.view(np.float32)[::2]=-tokens.astype(np.float32);records[1::2]=np.arange(n)
        data,_,_=aligned(q+cap);ctx,_,_=aligned(64)
        lib.run_refine_unique(records.ctypes.data,data.ctypes.data,ctx.ctypes.data,n,q,cap,0,-1,0)
        unique,inverse=np.unique(tokens,return_inverse=True)
        np.testing.assert_array_equal(data[:q],inverse);np.testing.assert_array_equal(data[q:q+len(unique)],unique);unique_tests+=1
    threads=lib.refine_threads()
    for n in (1,31,32,33,threads-1,threads,threads+1,511,512,513,4096):
        for rate in (0.,.01,.5,1.):
            flags=(rng.random(n)<rate).astype(np.int32);out=np.full(n,-777,np.int32);total=np.zeros(1,np.int32)
            lib.run_refine_bool(flags.ctypes.data,n,out.ctypes.data,total.ctypes.data)
            np.testing.assert_array_equal(out,np.cumsum(flags)-flags);assert total[0]==flags.sum();scan_tests+=1
    return {'SIMD_unique_map_cases':unique_tests,'ballot_scan_cases':scan_tests,'model_sorted_threads':threads}


def run_worker(name):
    kw=dict(turbo=True,merge=True,sorted_union=True,single_sorted=True);comparisons=0
    with tempfile.TemporaryDirectory(prefix='hot_cpu_') as tmp:
        d=Path(tmp);parent=load(d/'parent',138);candidate=load(d/'candidate',HOT_SWEEP[name])
        config=hot_config('joint_'+name);candidate_kw=kw | {'hot_packet':config['packet']}
        extra=[]
        for mtp in (0,1,3):
            c=make_case(1,65,1024,262144,mtp,'full_cache_75pct','shared',block_size=8)
            c['topk_indices'][:]=np.arange(65)+1000
            c['slot_to_token'][:]=np.arange(1024)+33768
            c['slot_to_token'][:,:65]=np.arange(65)+1000
            c['slot_to_token'][:,100:104]=c['context']+1
            extra.append(c)
        for raw in [*integration_cases(),*(c for _,c in boundary_cases(compact=True)),*extra]:
            c=sort_topk(raw)
            for gm in (False,True):
                before=initial_arrays(c,True,4*c['topk'])
                for step in range(3):
                    current=dict(c,req_ids=c['req_ids']+31) if step==2 else c
                    expected=execute(parent,current,True,gm,before,**kw)
                    actual=execute(candidate,current,True,gm,before,**candidate_kw)
                    for key in before:np.testing.assert_array_equal(actual[key],expected[key],err_msg=name+'/'+key)
                    before=actual;comparisons+=1
        # Continue actual returned state across joint -> singleton -> joint and reset.
        for mtp in (0,1,3):
            c=sort_topk(make_case(2,65,128,262144,mtp,'cold_reset','shared',block_size=8));c['topk_indices'][:]=np.arange(65)+150000
            before=initial_arrays(c,True,4*c['topk'])
            for step in range(8):
                current={k:v.copy() if hasattr(v,'copy') else v for k,v in c.items()}
                if step>=3:current['req_ids']+=17
                if step==2:current['topk_indices'][:]=np.arange(65)+180000
                if step==5:
                    if mtp:current['token_to_req'][:]=np.arange(c['rows'])%2
                    else:current['topk_indices'][:,1]=current['topk_indices'][:,0]
                current=sort_topk(current)
                expected=execute(parent,current,True,False,before,**kw);actual=execute(candidate,current,True,False,before,**candidate_kw)
                for key in before:np.testing.assert_array_equal(actual[key],expected[key],err_msg=name+'/transition/'+key)
                before=actual;comparisons+=1
    return {'variant':name,'full_controller_comparisons':comparisons,'scope':'CPU API model + UBSan, NOT CANN/NPU'}


def compact_checks(lib):
    lib.run_hot_compact.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_uint32]*6
    rng=np.random.default_rng(2026);checks=0
    for cap in (1,7,16,63,64,65,127,128,129,4096):
        for u in (0,1,cap):
            for mode in (0,1,2):
                for kind in ('evict','hit','mixed','invalid'):
                    q=max(129,u);ow=(q+cap+7)//8*8;fl=ow+cap;pa=fl+cap
                    data,g,off=aligned(ow+max(3*cap,64));ctx,cg,coff=aligned(64)
                    slots=rng.permutation(cap).astype(np.int32)
                    hit=np.ones(cap,bool) if kind=='hit' else np.zeros(cap,bool) if kind=='evict' else rng.random(cap)<.5
                    inv=(~hit)&(rng.random(cap)<.5)
                    flags=slots | (hit.astype(np.int32)<<16) | (inv.astype(np.int32)<<17)
                    if kind=='invalid':flags[0]=-1
                    data[fl:fl+cap]=flags
                    missing=rng.random(u)<.5
                    owners=rng.integers(0,65536,u,dtype=np.int32);owners[missing]=-1
                    data[ow:ow+u]=owners;expected=data.copy();ec=ctx.copy()
                    ev=[int(f)&0x2ffff for f in flags if f>=0 and not (int(f)&65536)]
                    hi=[int(f)&0x2ffff for f in flags if f>=0 and (int(f)&65536)]
                    if mode==1: # prepared by old SIMT partition in miss-only mode
                        data[pa:pa+len(ev)]=ev;data[pa+cap-len(hi):pa+cap]=hi[::-1];expected=data.copy()
                    else:
                        expected[pa:pa+len(ev)]=ev;expected[pa+cap-len(hi):pa+cap]=hi[::-1]
                    misses=0
                    if mode!=0:
                        for i,owner in enumerate(owners):
                            expected[fl+i]=misses if owner<0 else -1
                            if owner<0:
                                if misses<len(ev):expected[ow+i]=ev[misses]
                                misses+=1
                    ec[0]=int(len(ev)+len(hi)==cap and (mode==0 or misses<=len(ev)))
                    ec[43]=len(ev);ec[44]=len(hi);ec[45]=misses
                    lib.run_hot_compact(data.ctypes.data,ctx.ctypes.data,q,cap,u,len(ev),len(hi),mode)
                    np.testing.assert_array_equal(data,expected,err_msg=f'{cap}/{u}/{mode}/{kind}')
                    np.testing.assert_array_equal(ctx,ec)
                    for buf,start,length in ((g,off,len(data)),(cg,coff,len(ctx))):
                        assert np.all(buf[:start]==-777) and np.all(buf[start+length:]==-777)
                    checks+=1
    return {'SIMD_hot_compaction_cases':checks}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',nargs='+',choices=HOT_SWEEP,default=list(HOT_SWEEP))
    p.add_argument('--worker',choices=[*HOT_SWEEP,'primitives']);p.add_argument('--report',type=Path);a=p.parse_args()
    if a.worker:
        if a.worker=='primitives':
            with tempfile.TemporaryDirectory(prefix='hot_primitives_') as tmp:
                lib=ctypes.CDLL(str(build(Path(tmp),190,model_threads=512)));result=primitive_checks(lib) | compact_checks(lib)
        else:result=run_worker(a.worker)
        if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);return
    results=[]
    # Separate processes avoid accumulating persistent CPU model worker pools.
    with tempfile.TemporaryDirectory(prefix='hot_checks_') as tmp:
        for i,name in enumerate([*a.variants,'primitives']):
            report=Path(tmp)/f'{i}.json'
            subprocess.run([sys.executable,'-B',__file__,'--worker',name,'--report',str(report)],check=True)
            results.append(json.loads(report.read_text()))
    if a.report:a.report.write_text(json.dumps({'scope':'CPU only, not CANN/NPU','checks':results},indent=2)+'\n')
if __name__=='__main__':main()
