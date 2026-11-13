#!/usr/bin/env python3
"""8K预算、生成器、分派和报告回归；CPU检查，不是CANN/NPU验证。"""
from pathlib import Path
import hashlib,json,re,subprocess,sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.generate_joint_variants import generate
from checks.capacity8k_cases import capacity_cases
from joint_plan_cases import layout,make_case,sort_topk,describe
from joint_variants import CAPACITY8K_NAMES,description,fused_tune_bits
import run_mtp8

class Checks(unittest.TestCase):
    def test_layout_strict_shape_exception(self):
        with self.assertRaisesRegex(ValueError,'192KiB'):layout(2048,8192,224,True,262144,sorted_union=True,mtp8=True)
        d=layout(2048,8192,224,True,262144,sorted_union=True,mtp8=True,mtp8_capacity8k=True)
        self.assertEqual((d['scratch_elements'],d['dynamic_ub_bytes'],d['workspace_elements'],d['scratch_path']),(49280,197408,1,'UB'))
        for t,c in ((2048,8256),(2048,12288),(4096,8192),(2048,16384)):
            with self.assertRaises(ValueError):layout(t,c,224,True,262144,sorted_union=True,mtp8=True,mtp8_capacity8k=True)
        self.assertEqual(layout(2048,4096,224,True,262144,sorted_union=True,mtp8=True),layout(2048,4096,224,True,262144,sorted_union=True,mtp8=True,mtp8_capacity8k=True))

    def test_wrappers_have_exact_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);generate(p,capacity8k_only=True)
            self.assertEqual({f.stem for f in p.glob('variant_*.cpp')},{'variant_'+n.removeprefix('joint_') for n in CAPACITY8K_NAMES})
            for name,bits in zip(CAPACITY8K_NAMES,(0,3)):
                source=(p/('variant_'+name.removeprefix('joint_')+'.cpp')).read_text()
                for flag,value in (('JOINT_OPT_MASK',72),('JOINT_MTP7_CAPACITY8K',1),('JOINT_MTP7_FUSED_COUNT',1),('JOINT_MTP7_FUSED_TUNE_BITS',bits),('JOINT_MTP7_REUSE',1),('JOINT_MTP7_TILE',0)):
                    self.assertIn(f'#define {flag} {value}\n',source)
                self.assertEqual(description(name)['threads'],512)
            d=(p/'variant_dispatch.h').read_text();self.assertIn('mask <= 213',d)

    def test_old_selected_compute_unchanged(self):
        # The old source must preprocess exactly as pinned FUSED204; don't rewrite its frozen hash.
        args=['cpp','-P','-DJOINT_OPT_MASK=72','-DJOINT_REFINE_BITS=11','-DJOINT_MTP7_OPT=1','-DJOINT_MTP7_REUSE=1','-DJOINT_MTP7_TILE=0','-DJOINT_MTP7_PREFIX=0','-DJOINT_MTP7_MAP16=0','-DJOINT_MTP7_FUSED_COUNT=1','-I'+str(ROOT/'csrc/joint_variants'),'-']
        out=subprocess.run(args,input=(ROOT/'csrc/joint_variants/mtp8_plan_impl.h').read_text(),text=True,capture_output=True,check=True).stdout
        h=hashlib.sha256(re.sub(r'\s+','',out).encode()).hexdigest()
        self.assertEqual(h,json.loads((ROOT/'joint_plan_manifest.json').read_text())['frozen_fused204_preprocessor_sha256'])

    def test_adapter_selected_code_changes_budget_only(self):
        source=(ROOT/'csrc/joint_variants/mtp8_plan_impl.h').read_text()
        source=source.replace('196U*1024U','192U*1024U')
        manifest=json.loads((ROOT/'joint_plan_manifest.json').read_text())
        for bits,key in ((0,'frozen_fused204_preprocessor_sha256'),(3,'capacity8k_parent_ab_preprocessor_sha256')):
            args=['cpp','-P','-DJOINT_OPT_MASK=72','-DJOINT_REFINE_BITS=11','-DJOINT_MTP7_OPT=1','-DJOINT_MTP7_REUSE=1','-DJOINT_MTP7_TILE=0','-DJOINT_MTP7_PREFIX=0','-DJOINT_MTP7_MAP16=0','-DJOINT_MTP7_FUSED_COUNT=1','-DJOINT_MTP7_CAPACITY8K=1',f'-DJOINT_MTP7_FUSED_TUNE_BITS={bits}','-I'+str(ROOT/'csrc/joint_variants'),'-']
            out=subprocess.run(args,input=source,text=True,capture_output=True,check=True).stdout
            self.assertEqual(hashlib.sha256(re.sub(r'\s+','',out).encode()).hexdigest(),manifest[key])

    def test_true8k_workload_and_40pct_causality(self):
        for mtp in (0,1,3,7):
            c=sort_topk(make_case(2,2048,8192,262144,mtp,'full_cache_75pct','overlap40',history_pool_tokens=8192))
            d=describe(c)
            if mtp>=3:self.assertTrue(all(4096 < u <=8192 for u in d['union_sizes']))
            for first in range(0,c['rows'],mtp+1):
                root=set(c['topk_indices'][first])
                for row in range(first,first+mtp+1):
                    self.assertTrue(np.all(c['topk_indices'][row]<c['visible_seq_lens'][row]))
                    self.assertEqual(len(set(c['topk_indices'][row])),2048)
                    self.assertGreaterEqual(len(root & set(c['topk_indices'][row])),820 if row!=first else 2048)

    def test_capacity_boundary_cases(self):
        values={label:describe(c)['union_sizes'][0] for label,c in capacity_cases()}
        self.assertEqual(values['capacity8k/union_exact8192'],8192)
        self.assertEqual(values['capacity8k/union8193_fallback'],8193)
        self.assertEqual(values['capacity8k/union16384_fallback'],16384)
        self.assertEqual(values['capacity8k/no_visible_queries'],0)

    def test_cli_and_ratio_direction(self):
        out=subprocess.check_output([sys.executable,'-B',str(ROOT/'run_mtp8.py'),'--suite','capacity8k','--describe-only','--output','/tmp/unused_capacity8k'],text=True)
        d=json.loads(out);self.assertEqual(d['implementations'],list(CAPACITY8K_NAMES));self.assertEqual(d['target_active_samples'],80)
        self.assertTrue(all(w['capacity']==8192 for w in d['workloads']))
        names=CAPACITY8K_NAMES
        report={'suite':'capacity8k','implementations':names,'runs':[{'workload':{'scenario':'cold_reset','mtp':7},'stats':{n:{'p50_us':100-20*i,'p95_us':110-20*i,'count':5,'correctness':{'joint_groups':28,'overflow_groups':0,'total_transfer_descriptors':9}} for i,n in enumerate(names)}}]}
        text=run_mtp8.report_text(report);self.assertIn('0.800 | 20.0%',text);self.assertIn('原提交原样拒绝8K',text);self.assertNotIn('181024',text)

if __name__=='__main__':unittest.main()
