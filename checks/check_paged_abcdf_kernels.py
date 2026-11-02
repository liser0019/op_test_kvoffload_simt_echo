#!/usr/bin/env python3
"""Actual paged VF on CPU64 + UBSan; valid page tables only. Not NPU compilation/timing."""
import ctypes, tempfile, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from checks.check_joint_variant_kernels import execute
from joint_plan_cases import make_case

def build(d, mask):
    src=ROOT/('csrc/abcdf_pages' if mask==0 else 'csrc/abcdf_paged_opt')
    s=(src/'plan_kernel_impl.h').read_text()
    (d/'kernel_under_test.h').write_text(s[:s.index('extern "C" __global__')])
    for n in ('variant_config.h','sparse_kv_plan_launch.h','sparse_kv_plan_config.h'):
        (d/n).write_text((src/n).read_text().replace('PLAN_THREADS = 1024U','PLAN_THREADS = 64U'))
    for n in ('kernel_operator.h','simt_api/asc_simt.h','simt_api/device_atomic_functions.h','simt_api/device_sync_functions.h','simt_api/device_warp_functions.h'):
        p=d/n;p.parent.mkdir(exist_ok=True);p.write_text('// CPU shim\n')
    (d/'driver.cpp').write_text('''#include <climits>
#include <numeric>
#include "cpu_runtime.h"
#include "kernel_under_test.h"
extern "C" void run_variant(void** p,const int64_t* d,int32_t* scan) {
 const int64_t blocks=(d[6]+d[8]-1)/d[8];
 std::vector<int32_t> table(d[7]*blocks); std::iota(table.begin(),table.end(),0);
 CpuWorkers workers(sparse_kv_plan::PLAN_THREADS);
 for(int64_t start=0;start<d[0];++start) workers.Run([&]{
 SparseKvPlanRuntimeVf(scan,(int64_t*)p[0],(int64_t*)p[6],(int32_t*)p[1],
 (int32_t*)p[2],(int32_t*)p[3],(int32_t*)p[4],table.data(),(int32_t*)p[7],(int32_t*)p[8],
 (int32_t*)p[9],(int32_t*)p[10],(int32_t*)p[11],(int32_t*)p[12],(int32_t*)p[13],
 d[1],d[2],start,d[0],d[3],d[4],d[5],d[6],d[7],blocks,d[7]*blocks,(int32_t)d[8]); });
}
''')
    lib=d/'test.so'
    subprocess.run(['g++','-std=c++20','-O1','-pthread','-shared','-fPIC','-fsanitize=undefined','-fno-sanitize-recover=all',f'-DPAGED_EXPERIMENT_MASK={mask}','-I'+str(d),'-I'+str(ROOT/'checks/simt_variants'),str(d/'driver.cpp'),'-o',str(lib)],check=True)
    return ctypes.CDLL(str(lib))

def main():
    with tempfile.TemporaryDirectory(prefix='paged_abcdf_cpu_') as tmp:
        for mask in range(4):
            d=Path(tmp)/str(mask);d.mkdir();lib=build(d,mask)
            lib.run_variant.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)]
            lib.run_variant.restype=None
            count=0
            for gm in (False,True):
                for mtp in (0,1,3):
                    for scenario in ('cold_reset','full_cache_75pct','full_cache_90pct','full_cache_stable_hit'):
                        case=make_case(2,65,128,2048,mtp,scenario=scenario,overlap='independent',block_size=8)
                        before=execute(lib,case,False,gm,abcdf=True)
                        # Continue actual state with repeats, causal invalidity, reset and partial active rows.
                        case['topk_indices'][0,:6]=[5,5,6,-1,2048,case['context']+5]
                        case['stable_prefix_lens'][0]=0
                        before=execute(lib,case,False,gm,before,abcdf=True)
                        case['req_ids']+=100;case['active_rows']=max(0,case['rows']-1)
                        execute(lib,case,False,gm,before,abcdf=True)
                        count+=3
            for scenario in ('cold_reset','full_cache_75pct'):
                case=make_case(1,2048,4096,262144,3,scenario=scenario,overlap='independent')
                execute(lib,case,False,False,abcdf=True);count+=1
            print('paged mask',mask,':',count,'CPU VF checks passed',flush=True)
if __name__=='__main__':main()
