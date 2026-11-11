#!/usr/bin/env python3
"""执行同事独立Plan计算体，CPU模型检查语义与保护区，不能替代NPU验证。"""
from pathlib import Path
import ctypes
import json
import subprocess
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import peer_plan
from joint_plan_common import TENSOR_ORDER
from joint_plan_cases import INPUTS, initial_arrays, validate, make_case, sort_topk
from checks.mtp8_cases import cases, transitions

class Tiling(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ('hashCapacity', 'workspaceRowElements')]
    _fields_ += [(n, ctypes.c_int64) for n in ('maxRows', 'topk', 'capacity', 'maxToken', 'maxRequests', 'maxNumBlocks', 'hostNumBlocks')]
    _fields_ += [('blockSize', ctypes.c_int32), ('localMemoryBytes', ctypes.c_uint32)]
    _fields_ += [('jointHashCapacity', ctypes.c_uint64), ('scratchElements', ctypes.c_uint64), ('maxJointRows', ctypes.c_uint32)]

def build(directory):
    source = ROOT / 'csrc/peer_qli_plan'
    (directory/'standalone_body.h').write_bytes((source/'standalone_body.h').read_bytes())
    (directory/'sparse_kv_plan_config.h').write_text('// constants supplied by CPU prefix\n')
    for name in ('kernel_operator.h', 'simt_api/asc_simt.h', 'simt_api/device_atomic_functions.h', 'simt_api/device_sync_functions.h', 'simt_api/device_warp_functions.h', 'simt_api/device_functions.h'):
        p=directory/name
        p.parent.mkdir(exist_ok=True)
        p.write_text('// CPU shim supplied in prefix\n')
    program = '''#include "sorted_cann_cpu.h"
#define GM_ADDR void*
namespace sparse_kv_plan_config {
constexpr uint32_t PLAN_THREADS=64, PLAN_JOINT_THREADS=32, PLAN_WARP_SIZE=32;
constexpr uint32_t PLAN_WARP_COUNT=2, PLAN_SCAN_STORAGE_ELEMENTS=16, PLAN_ROW_UB_LIMIT_ELEMENTS=28672;
}
#include "standalone_body.h"
extern "C" void run_peer(void** p, const SparseKvPlanTilingData* d) {
  SortedCpuBlocks=3;
  for(SortedCpuBlock=0;SortedCpuBlock<SortedCpuBlocks;++SortedCpuBlock)
    RunAbdPlan(p[0],p[1],p[2],p[3],p[4],p[17],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],*d);
}
'''
    (directory/'driver.cpp').write_text(program)
    library=directory/'peer.so'
    subprocess.run(['g++','-std=c++20','-O1','-pthread','-shared','-fPIC','-fsanitize=undefined','-fno-sanitize-recover=all', '-I'+str(directory), '-I'+str(ROOT/'checks/simt_variants'), str(directory/'driver.cpp'), '-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library))
    lib.run_peer.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(Tiling)]
    return lib

def execute(lib,case,before=None):
    width=case['q_len']
    info=peer_plan.layout(case['topk'],case['capacity'],case['rows'],width)
    base=initial_arrays(case,True,width*case['topk']) if before is None else before
    values={k:v.copy() for k,v in base.items()}
    values.update({k:case[k].copy() for k in INPUTS})
    values['active_rows']=np.array([case.get('active_rows',case['rows'])],np.int32)
    values['workspace']=np.full(info['workspace_elements'],0x1234567,np.int32)
    blocks=(case['max_token']+case['block_size']-1)//case['block_size']
    values['block_table']=np.random.default_rng(case['seed']+90001).permutation(case['requests']*blocks).astype(np.int32).reshape(case['requests'],blocks)
    guards={}
    for key,value in list(values.items()):
        backing=np.full(value.size+32,0x5a5a5a5a,value.dtype)
        view=backing[16:-16].reshape(value.shape)
        view[:]=value
        values[key]=view
        guards[key]=backing
    saved={k:v.copy() for k,v in values.items() if k in (*INPUTS,'active_rows','block_table')}
    tiling=Tiling(info['hash_capacity'],3*info['hash_capacity']+case['capacity']+case['topk'],case['rows'],case['topk'],case['capacity'],case['max_token'],case['requests'],blocks,case['requests']*blocks,case['block_size'],info['dynamic_ub_bytes'],info['joint_hash_capacity'],info['scratch_elements'],width)
    pointers=(ctypes.c_void_p*18)(*(values[k].ctypes.data for k in (*TENSOR_ORDER,'block_table')))
    lib.run_peer(pointers,ctypes.byref(tiling))
    for key,g in guards.items():
        np.testing.assert_array_equal(g[:16],0x5a5a5a5a,err_msg=key)
        np.testing.assert_array_equal(g[-16:],0x5a5a5a5a,err_msg=key)
    for key,value in saved.items():
        np.testing.assert_array_equal(values[key],value,err_msg='read-only/'+key)
    actual={k:values[k].copy() for k in base}
    validate(case,base,actual,True,width*case['topk'],merge=True,float_keys=True,single_sorted=True,max_group_rows=width,require_power_two_groups=False)
    return actual

def main():
    with tempfile.TemporaryDirectory(prefix='peer_plan_cpu_') as tmp:
        lib=build(Path(tmp));count=0
        for label,case in cases(include_extreme=True):
            execute(lib,case);count+=1
        before=None
        for label,case in transitions():
            before=execute(lib,case,before);count+=1
        for mtp in (0,1,3,7):
            for scenario in ('cold_reset','full_cache_75pct'):
                case=sort_topk(make_case(1,2048,4096,262144,mtp,scenario,'overlap40'))
                before=execute(lib,case);execute(lib,case,before);count+=2
        print(json.dumps({'status':'passed','CPU_semantic_calls':count,'scope':'real standalone body + UBSan/guards/input invariance; no CANN/NPU'},ensure_ascii=False))
if __name__=='__main__':main()
