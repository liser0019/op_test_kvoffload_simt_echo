#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end profiler/adapter protocol on CPU; fake timings are discarded, never NPU results."""
import ast
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import compare_memfabric_plan as b
import profile_plan_variants as profile
from checks.plan_build_info import catalog_identity
from checks.memfabric_build_info import identity

# Parse actual C++ schema string literals, rather than maintaining a second ABI.
def schema(path,marker):
    text=path.read_text();start=text.index(marker);part=text[start:text.index(');',start)]
    return ''.join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"',part))
plan_schema=schema(ROOT/'csrc/plan_variants/common/plan_plugin_impl.h','m.def(')
plan_schema=plan_schema[plan_schema.index('('):]
mem_schema=schema(ROOT/'csrc/memfabric_plan/plugin.cpp','m.def(')
lib=torch.library.Library('sparse_kv_plan_op','FRAGMENT')
lib.define('plan_abcdf'+plan_schema)
lib.define(mem_schema)
lib.define('plan_variants_info() -> str')
lib.define('memfabric_plan_info() -> str')
lib.impl('plan_variants_info',lambda:json.dumps(catalog_identity()),'CompositeExplicitAutograd')
lib.impl('memfabric_plan_info',lambda:json.dumps(identity()),'CompositeExplicitAutograd')
context={};calls=[];active=None;inject=False
real_prepare=b.prepare_npu

def prepare(*args,**kwargs):
    result=real_prepare(*args,**kwargs)
    case=args[1];inputs,initial,states=result
    context.update(case=case,inputs=inputs,initial=initial,states=states)
    for key in b.SIMT_STATE:
        assert states['abcdf'][key].data_ptr()!=states['memfabric'][key].data_ptr()
        assert states['abcdf'][key].data_ptr()!=initial['abcdf'][key].data_ptr()
    return result
b.prepare_npu=prepare

def callback(name):
    def run(*args):
        case=context['case'];state=context['states'][name]
        if name=='abcdf':
            public=args[7:14]
            assert args[1].data_ptr()==context['inputs']['topk_indices'].data_ptr()
        else:
            assert args[0].data_ptr()==context['inputs']['req_ids'].data_ptr()
            assert args[2].data_ptr()==context['inputs']['topk_indices'].data_ptr()
            assert args[10].shape==(8,case['max_token']) and args[11].shape==args[10].shape
            assert args[12].shape==(8,)
            assert args[13:]==(case['rows'],case['topk'],case['capacity'],case['max_token'])
            public=(args[1],args[4],args[5],*args[6:10])
            per_core=torch.tensor([len(range(i,case['rows'],8)) for i in range(8)],dtype=torch.int32)
            args[12].add_(per_core)
        for k,t in zip(b.SIMT_STATE,public[:3]):
            torch.testing.assert_close(t,context['initial'][name][k],rtol=0,atol=0)
        answer=b.reference_simt(case)
        for k,t in zip((*b.SIMT_STATE,*b.SIMT_OUTPUT),public):t.copy_(torch.from_numpy(answer[k]))
        calls.append(name)
        if active is not None and active.step_index>=1:
            entry=b.VARIANTS[name]['kernel_name']
            active.rows.append((entry,'AI_VECTOR_CORE',2 if name=='abcdf' else 10))
            if inject and name=='memfabric':state['current_slots'].view(-1)[0]=-123
    return run
lib.impl('plan_abcdf',callback('abcdf'),'CPU')
lib.impl('memfabric_lru_plan',callback('memfabric'),'CPU')

class Profiler:
    def __init__(self,**kw):self.destination=Path(kw['on_trace_ready']);self.rows=[];self.step_index=0
    def __enter__(self):
        global active
        active=self;return self
    def step(self):self.step_index+=1
    def __exit__(self,*exc):
        global active
        active=None
        d=self.destination/'cpu_stub_ascend_pt/ASCEND_PROFILER_OUTPUT';d.mkdir(parents=True)
        with (d/'kernel_details.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['Name','Task Type','Duration(us)'])
            w.writerow(['TensorMove','MEMCPY_ASYNC',9999]);w.writerows(self.rows)
        return False
p=types.SimpleNamespace(_ExperimentalConfig=lambda **kw:None,
    ProfilerLevel=types.SimpleNamespace(Level0=0),AiCMetrics=types.SimpleNamespace(AiCoreNone=0),
    ExportType=types.SimpleNamespace(Text=0),ProfilerActivity=types.SimpleNamespace(CPU=0,NPU=1),
    profile=Profiler,schedule=lambda **kw:kw,tensorboard_trace_handler=lambda path:path)
sys.modules['torch_npu']=types.SimpleNamespace(profiler=p,__version__='CPU-protocol-stub')
real_device=torch.device
torch.device=lambda *a,**kw:real_device('cpu')
torch.npu=types.SimpleNamespace(is_available=lambda:True,set_device=lambda n:None,
    get_device_name=lambda n:'CPU protocol stub, NOT NPU',synchronize=lambda:None)
torch.ops.load_library=lambda p:None
# profile dynamically imports adapter under a different module name.
original_spec=profile.importlib.util.spec_from_file_location
def load_spec(name,location,*args,**kwargs):
    if name=='kv_variant_comparison':
        class Loader:
            def create_module(self,spec):return b
            def exec_module(self,module):pass
        return importlib.util.spec_from_loader(name,Loader())
    return original_spec(name,location,*args,**kwargs)
profile.importlib.util.spec_from_file_location=load_spec
with tempfile.TemporaryDirectory(prefix='memfabric_profile_protocol_') as temporary:
    root=Path(temporary);so=root/'stub.so';so.touch()
    argv=['profile_plan_variants.py','--project',str(ROOT),'--comparison-module','compare_memfabric_plan.py',
          '--library',str(so),'--rows','2','--topk','4','--capacity','8','--max-token','64',
          '--scenario','all','--shuffle-lru','--warmup','0','--repeat','4','--output',str(root/'ok')]
    sys.argv=argv
    with contextlib.redirect_stdout(io.StringIO()):profile.main()
    metadata=json.loads(next((root/'ok').glob('capture_*/capture_metadata.json')).read_text())
    assert metadata['status']=='complete' and metadata['reference']=='memfabric'
    assert len(calls)==4*2*(3+1+4)
    for scene in metadata['scenarios'].values():
        stats={s['implementation']:s for s in scene['kernel_stats']}
        assert set(stats)=={'abcdf','memfabric'}
        assert stats['abcdf']['count']==stats['memfabric']['count']==4
        assert stats['abcdf']['speedup_vs_reference_p50']==5
    inject=True
    sys.argv=[v if v!=str(root/'ok') else str(root/'bad') for v in argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()):profile.main()
    except AssertionError:pass
    else:raise AssertionError('fault was not detected')
    meta=json.loads(next((root/'bad').glob('capture_*/capture_metadata.json')).read_text())
    assert meta['status']!='complete' and not list((root/'bad').rglob('kernel_summary.csv'))
print('PASS: 64 CPU callback calls, shared inputs, restored public state, advancing epochs, profiler warm-up exclusion, CSV copy filtering/counts/reference and fault gate. Fake timings discarded; NOT NPU performance.')
