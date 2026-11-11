#!/usr/bin/env python3
"""CPU/UBSan验证A/B/AB与冻结196等价，额外检查B16偶数lane散写及跨包UID。"""
from pathlib import Path
import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from checks.check_mtp7_fast import worker
from checks.check_cold_finish import load
from checks.check_hot_kernels import aligned


def packed_primitives(mask):
    if mask == 200:
        return 0
    count=0
    rng=np.random.default_rng(6642)
    with tempfile.TemporaryDirectory(prefix='mtp7_map16_') as tmp:
        root=Path(tmp)
        reference=load(root/'reference',196)
        candidate=load(root/'candidate',mask)
        pointer=ctypes.c_void_p
        reference.run_refine_unique.argtypes=[pointer]*3+[ctypes.c_uint32]*4+[ctypes.c_int32,ctypes.c_uint32]
        candidate.run_prefix_packed_unique.argtypes=[pointer]*3+[ctypes.c_uint32]*4+[ctypes.c_int32]+[ctypes.c_uint32]*3
        for q in (56,520,4104,16384):
            capacity=4096
            words=q//2
            total=words+capacity+32768
            for n in sorted(set(min(q,n) for n in (0,1,31,32,33,63,64,65,127,129,511,512,513,6528))):
                for duplicate in (False,True):
                    ordinary,og,oo=aligned(q+capacity+32768)
                    compact,cg,co=aligned(total)
                    ordinary[:q]=-1
                    compact[:words]=-1
                    c0,g0,o0=aligned(128)
                    c1,g1,o1=aligned(128)
                    c1[:]=c0
                    record,rg,ro=aligned(max(128,2*n+128))
                    token=np.arange(n,dtype=np.int32)//(3 if duplicate else 1)+91
                    payload=rng.permutation(q)[:n].astype(np.int32)
                    record.view(np.float32)[:2*n:2]=-token.astype(np.float32)
                    record[1:2*n:2]=payload
                    initial_count=0
                    initial_last=-1
                    # Independent calls carry both UID count and last token across seams.
                    for first,end in ((0,min(63,n)),(min(63,n),n)):
                        chunk=end-first
                        packet,pg,po=aligned(max(128,2*chunk+128))
                        packet[:2*chunk]=record[first*2:end*2]
                        args=[packet.ctypes.data,ordinary.ctypes.data,c0.ctypes.data,chunk,q,capacity,initial_count,initial_last,int(end<n)]
                        reference.run_refine_unique(*args)
                        args[1]=compact.ctypes.data
                        args[2]=c1.ctypes.data
                        candidate.run_prefix_packed_unique(*args,words,total)
                        np.testing.assert_array_equal(c1,c0)
                        decoded=compact.view(np.uint16)[:q].astype(np.int32)
                        decoded[decoded==65535]=-1
                        np.testing.assert_array_equal(decoded,ordinary[:q])
                        np.testing.assert_array_equal(compact[words:],ordinary[q:])
                        assert np.all(pg[:po]==-777)
                        assert np.all(pg[po+len(packet):]==-777)
                        initial_count=int(c0[1])
                        initial_last=int(c0[2])
                        count+=1
                    for storage,offset,length in ((og,oo,len(ordinary)),(cg,co,total),(g0,o0,128),(g1,o1,128),(rg,ro,len(record))):
                        assert np.all(storage[:offset]==-777)
                        assert np.all(storage[offset+length:]==-777)
    return count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,choices=(200,201,202))
    args=parser.parse_args()
    if args.worker:
        result=worker(args.worker,reference_mask=196,include_prefix=True)
        result['packed_map_primitive_calls']=packed_primitives(args.worker)
        print(json.dumps(result),flush=True)
        return
    for mask in (200,201,202):
        subprocess.run([sys.executable,'-B',__file__,'--worker',str(mask)],check=True)
if __name__=='__main__':main()
