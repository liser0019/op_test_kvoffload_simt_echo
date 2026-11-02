#!/usr/bin/env python3
"""CPU torch: validate both adapters, readonly page table and exact historical headers. No NPU execution."""
import sys, subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joint_plan_common as c
from joint_plan_cases import make_case
from compare import reference_simt
from profile_joint_mtp import read_kernels, duration_ratios
import tempfile
for name in ('plan_kernel_impl.h','plan_plugin_impl.h','sparse_kv_plan_config.h','sparse_kv_plan_launch.h'):
    assert (ROOT/'csrc/abcdf_pages'/name).read_bytes() == subprocess.check_output(
        ['git','show','75cf8b2:csrc/plan_variants/common/'+name],cwd=ROOT)
old, new = Mock(), Mock()
proxy = SimpleNamespace(**{key:getattr(torch,key) for key in ('int64','int32','full','from_numpy')},
                        npu=SimpleNamespace(synchronize=lambda:None),
                        ops=SimpleNamespace(sparse_kv_plan_op=SimpleNamespace(plan_abcdf=new,plan_abcdf_pages=old)))
for mtp in (0,1,3):
    case=make_case(2,7,16,512,mtp,overlap='independent',block_size=8)
    inputs={k:torch.from_numpy(case[k].copy()) for k in c.INPUTS}
    inputs['active_rows']=torch.tensor([case['rows']],dtype=torch.int32)
    bundles=[c.prepare(proxy,case,name,inputs) for name in ('abcdf','abcdf_pages')]
    assert len(bundles[0]['ordered'])==14 and len(bundles[1]['ordered'])==15
    assert len(bundles[0]['attrs'])==3 and len(bundles[1]['attrs'])==5
    b=bundles[1]
    assert b['ordered'][5] is b['inputs']['block_table']
    assert b['attrs'][-2:]==[8,128]
    for key in c.ABCDF_STATE:
        np.testing.assert_array_equal(bundles[0]['host_initial'][key],b['host_initial'][key])
    gold=reference_simt(case)
    for bundle in bundles:
        for key in (*c.ABCDF_STATE,*c.ABCDF_OUTPUT):
            bundle['state'][key].copy_(torch.from_numpy(gold[key]))
        c.check(case,bundle)
        c.invoke(bundle)
    b['inputs']['block_table'][0,0]=-1
    try:c.check(case,b)
    except AssertionError:pass
    else:raise AssertionError('page input write not detected')
for name in c.PAGED_NAMES[1:]:
    spy=Mock()
    setattr(proxy.ops.sparse_kv_plan_op,"plan_"+name,spy)
    case=make_case(2,7,16,512,3,overlap='independent',block_size=8)
    inputs={k:torch.from_numpy(case[k].copy()) for k in c.INPUTS}
    inputs['active_rows']=torch.tensor([case['rows']],dtype=torch.int32)
    bundle=c.prepare(proxy,case,name,inputs)
    assert len(bundle['ordered'])==15 and len(bundle['attrs'])==5
    gold=reference_simt(case)
    for key in (*c.ABCDF_STATE,*c.ABCDF_OUTPUT):bundle['state'][key].copy_(torch.from_numpy(gold[key]))
    c.check(case,bundle);c.invoke(bundle);assert spy.call_count==1
assert old.call_count==new.call_count==3
names=['abcdf','abcdf_pages']
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'kernel_details.csv'
    path.write_text('Name,Duration(us),Start Time(us)\n'+''.join(
        f'{c.KERNELS[n]},10,{i*100}\n' for i,n in enumerate(names)))
    assert len(read_kernels([path],names,1,names))==2
assert duration_ratios({n:{'p50_us':v} for n,v in zip(names,[10,12])})['abcdf_pages_over_abcdf_p50']==1.2
print('PASS: preserved historical headers; both CPU adapters at MTP0/1/3; page input immutability; distinct invocation/CSV identity and ratio')
