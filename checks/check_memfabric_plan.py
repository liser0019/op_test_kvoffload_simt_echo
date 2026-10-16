#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile the byte-identical archive class with CPU UB/DMA shims and compare all public results."""
import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mem_comparison',ROOT/'compare_memfabric_plan.py')
b=importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0,str(ROOT))
spec.loader.exec_module(b)
from checks.memfabric_build_info import identity
identity()
# Reject cases that would compare different semantics rather than different implementations.
case=b.make_case(2,4,8,64,'warm_75pct',2026)
for mode in ('mapping','active','duplicate_query','visible','page','resident','block_size'):
    bad=copy.deepcopy(case)
    if mode=='mapping':bad['token_to_req'][:]=[1,0]
    elif mode=='active':bad['active_rows']=1
    elif mode=='duplicate_query':bad['topk_indices'][0,1]=bad['topk_indices'][0,0]
    elif mode=='visible':bad['visible_seq_lens'][0]=0
    elif mode=='page':bad['block_table'][0,:]=-1
    elif mode=='resident':bad['slot_to_token'][0,1]=bad['slot_to_token'][0,0]
    else:bad['block_size']=0
    try:b.validate_case(bad)
    except ValueError:pass
    else:raise AssertionError('Unsupported common domain accepted: '+mode)
with tempfile.TemporaryDirectory(prefix='memfabric_cpu_') as directory:
    tmp=Path(directory);binary=tmp/'check'
    subprocess.run(['g++','-std=c++17','-O1','-g','-no-pie','-fsanitize=address,undefined','-fno-omit-frame-pointer',
                    '-I'+str(ROOT/'checks/memfabric'),'-I'+str(ROOT/'csrc/memfabric_plan/upstream'),
                    str(ROOT/'checks/memfabric/driver.cpp'),'-o',str(binary)],check=True)
    checks=0
    for rows,t,c,l in [(1,4,8,64),(7,4,8,64),(9,4,8,64),(32,2048,4096,262144)]:
        for scenario in b.SCENARIOS:
            case=b.make_case(rows,t,c,l,scenario,2026,64,True)
            expected=[];payload=[np.array([rows,t,c,l,4],dtype=np.int64).tobytes()]
            for step in range(4):
                b.validate_case(case)
                for key in ['req_ids','topk_indices','stable_prefix_lens']:payload.append(case[key].tobytes())
                if step==0:
                    for key in b.SIMT_STATE:payload.append(case[key].tobytes())
                answer=b.reference_simt(case);expected.append(answer)
                case={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in case.items()}
                case.update({k:answer[k].copy() for k in b.SIMT_STATE})
                if step==0:case['stable_prefix_lens'][:]=l//2
                elif step==1:case['topk_indices']=((case['topk_indices'].astype(np.int64)+7)%l).astype(np.int32)
                elif step==2:case['req_ids']+=100
            source=tmp/'input.bin';result=tmp/'output.bin'
            source.write_bytes(b''.join(payload))
            subprocess.run([str(binary),str(source),str(result)],check=True,stdout=subprocess.PIPE,timeout=30)
            raw=result.read_bytes();offset=0
            for step,answer in enumerate(expected):
                for key in (*b.SIMT_STATE,*b.SIMT_OUTPUT):
                    a=answer[key];size=a.nbytes
                    got=np.frombuffer(raw[offset:offset+size],dtype=a.dtype).reshape(a.shape)
                    np.testing.assert_array_equal(got,a,err_msg=f'{rows}/{scenario}/{step}/{key}')
                    offset+=size
                epochs=np.frombuffer(raw[offset:offset+32],dtype=np.int32);offset+=32
                np.testing.assert_array_equal(epochs,np.array([len(range(i,rows,8))*(step+1) for i in range(8)]))
                checks+=1
            assert offset==len(raw)
    print(f'PASS: {checks} actual archive-class CPU state/oracle checks, including production, 8-core tails, rollback, query change, reset and epochs. Not NPU validation.')
