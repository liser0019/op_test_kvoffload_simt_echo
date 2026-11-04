#!/usr/bin/env python3
"""CPU-only regression: layout/catalogue, strict fixed-reference cache, readback and CLI size."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,subprocess,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_plan_cases import make_case,sort_topk,layout,INPUTS
from joint_variants import VARIANTS,description,algorithm_mask
from checks.generate_joint_variants import generate
from fixed_plan_check import FixedExpectation
import joint_plan_common as common
from compare import reference_simt


class Checks(unittest.TestCase):
    def test_minimum_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);generate(d,mtp8_only=True)
            self.assertEqual({f.name for f in d.glob('variant_*.cpp')},{'variant_hot_bd_512.cpp','variant_hot_bd8_512.cpp'})
            old=(d/'variant_hot_bd_512.cpp').read_text();new=(d/'variant_hot_bd8_512.cpp').read_text()
            self.assertNotIn('JOINT_MTP8',old);self.assertIn('#define JOINT_MTP8 1',new)
            for t in ('#define JOINT_HOT_BITS 10','#define JOINT_REFINE_BITS 11','#define JOINT_OPT_MASK 72'):self.assertIn(t,new)
        self.assertEqual(VARIANTS['joint_hot_bd8_512'],193);self.assertEqual(algorithm_mask(193),72)
        self.assertEqual(description('joint_hot_bd8_512')['threads'],512)

    def test_layout(self):
        d=layout(2048,4096,256,True,262144,sorted_union=True,mtp8=True)
        self.assertEqual(d['dynamic_ub_bytes'],181024);self.assertEqual(d['workspace_elements'],1);self.assertEqual(d['scratch_path'],'UB')
        self.assertLess(d['dynamic_ub_bytes']+8192+32768,256*1024)
        with self.assertRaisesRegex(ValueError,'exceeds192KiB'):layout(2048,32768,256,True,262144,sorted_union=True,mtp8=True)
        self.assertEqual(layout(2048,4096,128,True,262144,sorted_union=True)['scratch_elements'],24640)

    def test_cases_unchanged_first_row(self):
        for mtp in (1,3,7):
            a=make_case(2,65,256,262144,0,overlap='overlap40',block_size=8)
            b=make_case(2,65,256,262144,mtp,overlap='overlap40',block_size=8)
            for r in range(2):
                for k in ('topk_indices','slot_to_token','lru_slots'):
                    np.testing.assert_array_equal(a[k][r],b[k][r*(mtp+1)])
            self.assertEqual(b['rows'],2*(mtp+1))
            self.assertTrue(np.all(b['visible_seq_lens']<b['max_token']))
        for m in (2,4,5,6,8):
            with self.assertRaises(ValueError):make_case(mtp=m)

    def bundle(self):
        import torch
        c=sort_topk(make_case(1,7,16,1024,0,block_size=8))
        gold=reference_simt(c)
        host={k:c[k].copy() for k in common.ABCDF_STATE}
        for k in common.ABCDF_OUTPUT:host[k]=np.full_like(gold[k],-777777)
        host['workspace']=np.zeros(1,np.int32)
        states={};buffers={}
        for k,a in host.items():
            buf=torch.full((a.size+32,),common.GUARD_VALUE,dtype=torch.int64 if a.dtype==np.int64 else torch.int32)
            v=buf[16:-16].reshape(a.shape);v.copy_(torch.from_numpy(gold[k] if k in gold else a))
            states[k]=v;buffers[k]=buf
        inputs={k:torch.from_numpy(c[k].copy()) for k in INPUTS};inputs['active_rows']=torch.tensor([c['rows']],dtype=torch.int32)
        b={'name':'abcdf','torch':torch,'host_initial':host,'inputs':inputs,'state':states,'buffers':buffers,'fast_fixed_check':True}
        return c,b

    def test_fixed_cache_no_recompute(self):
        c,b=self.bundle()
        with patch('joint_plan_cases.reference_simt',wraps=reference_simt) as model:
            result=common.check(c,b)
            for _ in range(3):self.assertEqual(common.check(c,b),result)
            self.assertEqual(model.call_count,1)
        self.assertEqual(b['check_timing']['reference_builds'],1);self.assertEqual(b['check_timing']['cache_hits'],3)

    def test_cached_checks_detect_corruption(self):
        c,b=self.bundle();common.check(c,b)
        for k in (*common.ABCDF_STATE,'current_slots','miss_count','miss_tokens','miss_slots'):
            view=b['state'][k].reshape(-1);old=view[0].item();view[0]=old+1
            with self.assertRaises(AssertionError,msg=k):common.check(c,b)
            view[0]=old
        for k in INPUTS:
            view=b['inputs'][k].reshape(-1);old=view[0].item();view[0]=old+1
            with self.assertRaises(AssertionError,msg=k):common.check(c,b)
            view[0]=old
        b['buffers']['workspace'][0]=0
        with self.assertRaises(AssertionError):common.check(c,b)

    def test_cache_state_identity_and_continuous_bypass(self):
        c,b=self.bundle();common.check(c,b)
        c['visible_seq_lens'][0]-=1;b['inputs']['visible_seq_lens'][0]-=1
        with self.assertRaisesRegex(RuntimeError,'identity changed'):common.check(c,b)
        with patch.object(common,'_check_uncached',return_value={'status':'continuous'}) as model:
            before={};self.assertEqual(common.check(c,b,before),{'status':'continuous'});model.assert_called_once_with(c,b,before)

    def test_joint_descriptor_tails_remain_strict(self):
        c=sort_topk(make_case(1,7,16,1024,1,block_size=8))
        before={'slot_to_token':c['slot_to_token']}
        actual={'miss_count':np.array([1,0],np.int32),'miss_tokens':np.full((2,7),-1,np.int32),'miss_slots':np.full((2,7),-1,np.int32)}
        result={'groups':[{'first':0,'end':2,'path':'joint'}]}
        e=FixedExpectation(c,before,actual,result)
        bad=copy.deepcopy(actual);bad['miss_tokens'][1,6]=123
        with self.assertRaises(AssertionError):e.check(c,before,bad)

    def test_pre_runtime_failure_preserved(self):
        import run_mtp8
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'results'
            args=['run_mtp8.py','--skip-build','--output',str(out),'--mtp','0','--requests','1','--topk','7','--capacity','64','--scenarios','cold_reset','--min-output-free-bytes','0','--min-temp-free-bytes','0','--min-free-inodes','0']
            exc=RuntimeError('synthetic runtime unavailable')
            with patch.dict('os.environ'),patch.object(tempfile,'tempdir',tempfile.gettempdir()),patch.object(sys,'argv',args),patch.object(common,'load_runtime',side_effect=exc):
                with self.assertRaises(RuntimeError) as caught:run_mtp8.main()
            self.assertIs(caught.exception,exc)
            self.assertEqual(json.loads((out/'RUN_STATUS.json').read_text())['status'],'failed')
            self.assertFalse((out/'COMPLETE.json').exists());self.assertFalse((out/'report.md').exists())

    def test_cli_only_three(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'run_mtp8.py'),'--describe-only','--output','/tmp/unused_mtp8_suite'],capture_output=True,text=True,check=True)
        d=json.loads(r.stdout);self.assertEqual(len(d['implementations']),3);self.assertEqual(d['mtp'],[0,1,3,7])
        self.assertEqual(d['target_active_samples'],120);self.assertEqual(d['repeat'],5)

if __name__=='__main__':unittest.main()
