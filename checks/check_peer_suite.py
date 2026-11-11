"""28请求三路接入、输入身份、构建子集与来源检查。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joint_plan_common as common
import peer_plan
from joint_plan_cases import make_case, sort_topk
from checks.generate_joint_variants import generate

class Checks(unittest.TestCase):
    def test_primary_entry_defaults_and_matrix(self):
        out=subprocess.check_output([sys.executable,'-B',str(ROOT/'run_mtp8.py'),'--suite','peer','--describe-only','--output','/tmp/unused_peer'],text=True)
        report=json.loads(out)
        self.assertEqual(report['implementations'],['abcdf_pages','joint_mtp7_reuse_parallel_512','peer_qli_plan'])
        self.assertEqual(report['target_active_samples'],120)
        self.assertEqual([w['query_rows'] for w in report['workloads'][:4]],[28,56,112,224])
        self.assertEqual(report['repeat'],5)
        self.assertEqual(report['build'],'--peer-only')

    def test_exact_standalone_body_and_source_pinning(self):
        root=ROOT/'csrc/peer_qli_plan'
        source=(root/'upstream/sparse_kv_plan_apt.cpp').read_text()
        self.assertEqual((root/'standalone_body.h').read_text(),source[:source.index('extern "C" __global__')].rstrip() + '\n')
        info=json.loads((root/'upstream/source.json').read_text())
        import hashlib
        for rel,expected in info['files'].items():
            self.assertEqual(hashlib.sha256((root/'upstream'/Path(rel).name).read_bytes()).hexdigest(),expected)
        self.assertEqual(info['commit'],'9ec2110969a9e8502bf6d579c027ac4ea9572016')

    def test_upstream_layout_and_gm_route_are_visible(self):
        self.assertEqual(peer_plan.layout(2048,4096,224,8)['scratch_path'],'GM')
        self.assertEqual(peer_plan.layout(2048,4096,28,1)['scratch_path'],'UB')
        self.assertEqual(peer_plan.layout(2048,4096,224,8)['dynamic_ub_bytes'],49696)
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp,peer_only=True)
            self.assertEqual([p.name for p in Path(tmp).glob('variant_*.cpp')],['variant_mtp7_reuse_parallel_512.cpp'])
        flags=(ROOT/'joint_plan/CMakeLists.txt').read_text()
        self.assertNotIn('-cce-aicore-dcci-before-kernel-end=false',flags)
        self.assertIn('"${PEER_PLAN_DIR}/kernel.cpp"',flags)
        self.assertIn('-O3',flags)

    def test_prepare_normal_entry_and_original_snapshot_validation(self):
        import torch
        c=sort_topk(make_case(2,7,16,256,3,block_size=8))
        inputs={k:torch.from_numpy(c[k].copy()) for k in common.INPUTS}
        inputs['active_rows']=torch.tensor([c['rows']],dtype=torch.int32)
        calls=[]
        class Context:
            def __init__(self,p,attrs,joint):
                self.p=p;self.attrs=attrs;calls.append((p,attrs,joint))
            def info(self):return json.dumps({'dynamic_ub_bytes':peer_plan.layout(7,16,c['rows'],4)['dynamic_ub_bytes'],'joint_enabled':1,'blocks':56})
            def run(self,p,attrs):calls.append(('run',p,attrs))
            def close(self):pass
        with patch.object(torch,'npu',SimpleNamespace(synchronize=lambda:None),create=True), patch.object(torch.classes,'mtp_joint_plan',SimpleNamespace(PreparedPeerQli=Context)):
            bundle=common.prepare(torch,c,peer_plan.NAME,inputs)
            common.invoke(bundle)
            self.assertEqual(len(bundle['ordered']),18)
            self.assertEqual(bundle['attrs'],[7,16,256,8,1,4])
            self.assertIs(bundle['ordered'][-1],bundle['inputs']['block_table'])
            self.assertEqual(bundle['layout']['blocks'],56)
            np.testing.assert_array_equal(bundle['host_initial']['slot_to_token'],c['slot_to_token'])
            self.assertEqual(calls[-1][0],'run')
            common.close(bundle)

if __name__=='__main__':unittest.main()
