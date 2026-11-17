"""Execute actual fusion Plan source under the existing CPU semantic shim."""
import argparse
import ctypes
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from joint_plan_cases import make_case,sort_topk,initial_arrays,validate,INPUTS
from joint_plan_common import TENSOR_ORDER

ROOT=Path(__file__).resolve().parents[1]

def build(tmp,mode):
    shim=(ROOT/'checks/simt_variants/sorted_cann_cpu.h').read_text()
    shim=shim.replace('  void* GetPhyAddr(){return p;}', '  void* GetPhyAddr(){return p;}\n  template<typename U>LocalTensor<U> ReinterpretCast()const{return {reinterpret_cast<U*>(p),static_cast<uint32_t>(n*sizeof(T)/sizeof(U))};}')
    (tmp/'sorted_cann_cpu.h').write_text(shim)
    for name in ('plan.h','storage.h','resources.h'):
        (tmp/name).write_bytes((ROOT/'csrc/qli_fusion'/name).read_bytes())
    config=(ROOT/'csrc/joint_plan/sparse_kv_plan_config.h').read_text().replace('PLAN_THREADS = 1024U','PLAN_THREADS = 128U')
    # Reduce CPU worker count only; preserve the real 288-byte Scan reservation,
    # so the model does not accidentally grant extra space at the UB tail.
    config=config.replace('PLAN_SCAN_BYTES = (PLAN_SCAN_ELEMENTS * sizeof(int32_t) + 31U) & ~31U','PLAN_SCAN_BYTES = 288U')
    (tmp/'sparse_kv_plan_config.h').write_text(config)
    for name in ('kernel_operator.h','simt_api/asc_simt.h','simt_api/device_atomic_functions.h','simt_api/device_sync_functions.h','simt_api/device_warp_functions.h','simt_api/device_functions.h'):
        target=tmp/name;target.parent.mkdir(exist_ok=True);target.write_text('')
    target=tmp/'plan.so'
    subprocess.run(['g++','-std=c++20','-O1','-pthread','-shared','-fPIC','-fvisibility=hidden',
        '-fsanitize=undefined','-fno-sanitize-recover=all',f'-DQLI_PLAN_STORAGE={mode}',
        '-I'+str(tmp),'-I'+str(ROOT/'checks/simt_variants'),str(ROOT/'checks/qli_fusion/driver.cpp'),'-o',str(target)],check=True)
    return ctypes.CDLL(str(target))

def run(lib,c,before,local=True):
    arrays={k:v.copy() for k,v in before.items()};arrays.update({k:c[k].copy() for k in INPUTS})
    arrays['active_rows']=np.array([c['rows']],np.int32)
    requests=c['requests'];cap=c['capacity'];topk=c['topk']
    arrays['workspace']=np.full(requests*8*topk//2+requests*8*cap+requests*8*(3*4096+cap+topk),-777,np.int32)
    backing={}
    for k,v in arrays.items():
        b=np.full(v.size+64,1234567,v.dtype);b[32:-32]=v.reshape(-1);backing[k]=b;arrays[k]=b[32:-32].reshape(v.shape)
    ptrs=(ctypes.c_void_p*17)(*(arrays[k].ctypes.data for k in TENSOR_ORDER))
    args=np.array([c['rows'],topk,cap,c['max_token'],requests,c['block_size'],local],np.int64)
    lib.fusion_run(ptrs,args.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    for k,b in backing.items():assert np.all(b[:32]==1234567) and np.all(b[-32:]==1234567),k
    for k in INPUTS:np.testing.assert_array_equal(arrays[k],c[k],err_msg=k)
    validate(c,before,arrays,True,cap,float_keys=True,single_sorted=True,max_group_rows=8)
    return {k:arrays[k].copy() for k in before}

def main():
    p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',type=int,default=[1,2,3]);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    calls=0
    for mode in a.modes:
        with tempfile.TemporaryDirectory() as td:
            lib=build(Path(td),mode)
            for mtp in ([0,7] if a.smoke else [0,1,3,7]):
                cap=max(4096,(mtp+1)*2048)
                for scenario in ('cold_reset','full_cache_75pct'):
                    c=sort_topk(make_case(1,2048,cap,262144,mtp,scenario,'overlap40',history_pool_tokens=cap))
                    before=initial_arrays(c,True,cap)
                    state=run(lib,c,before);calls+=1
                    if not a.smoke:
                        # Same device-produced state continued, then request reset.
                        state=run(lib,c,state,False);calls+=1
                        c['req_ids']+=100;run(lib,c,state);calls+=1
                print('mode',mode,'MTP',mtp,'passed',flush=True)
            if not a.smoke:
                c=sort_topk(make_case(2,2048,16384,262144,7,'full_cache_75pct','overlap40',history_pool_tokens=16384))
                run(lib,c,initial_arrays(c,True,16384));calls+=1
                for mtp in (0,7):
                    cap=max(4096,(mtp+1)*2048)
                    for valid in (0,1,63,64,65):
                        c=sort_topk(make_case(1,2048,cap,262144,mtp,'full_cache_75pct','shared',history_pool_tokens=cap))
                        c['topk_indices'][:]=-1
                        if valid:c['topk_indices'][:,:valid]=100000+np.arange(valid,dtype=np.int32)
                        before=initial_arrays(c,True,cap)
                        if valid:before['slot_to_token'][0,:2]=100000 # duplicated resident: retain LRU-last owner
                        run(lib,c,before);calls+=1
                print('mode',mode,'trailing -1 / 63,64,65 / duplicate resident passed',flush=True)
    print('actual fusion Plan CPU semantic calls:',calls,'(not CANN/NPU validation)')

if __name__=='__main__':main()
