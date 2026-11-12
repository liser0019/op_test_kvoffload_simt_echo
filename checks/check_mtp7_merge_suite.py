#!/usr/bin/env python3
"""Three-way wiring, unchanged frozen body and intermediate-buffer lifetimes; CPU only."""
from pathlib import Path
import json,re,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.generate_joint_variants import generate
from checks.check_mtp7_prefix_suite import frozen_digest
from joint_plan_cases import layout
from joint_variants import description
import run_mtp8

class Checks(unittest.TestCase):
    def test_three_only_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp,merge_unique_only=True)
            files={p.stem for p in Path(tmp).glob('variant_*.cpp')}
            self.assertEqual(files,{'variant_'+n.removeprefix('joint_') for n in run_mtp8.MERGE_UNIQUE_NAMES})
            for name,flag in (('mtp7_mid_unique_512','MID_UNIQUE'),('mtp7_fused_count_512','FUSED_COUNT')):
                source=(Path(tmp)/('variant_'+name+'.cpp')).read_text()
                self.assertIn('#define JOINT_MTP7_'+flag+' 1',source)
                self.assertNotIn('#define JOINT_MTP7_MAP16 1',source)
                self.assertNotIn('#define JOINT_MTP7_PREFIX 1',source)
                self.assertEqual(description('joint_'+name)['threads'],512)
        cmake=(ROOT/'joint_plan/CMakeLists.txt').read_text()
        self.assertIn('"${JOINT_VARIANT_DIR}/mtp7_merge_unique.h"',cmake)
        self.assertIn('JOINT_MERGE_UNIQUE_ONLY OR JOINT_PREFIX_ONLY',cmake)

    def test_frozen196_unchanged(self):
        manifest=json.loads((ROOT/'joint_plan_manifest.json').read_text())
        for name,expected in manifest['mtp7_prefix_frozen196_preprocessor_sha256'].items():
            self.assertEqual(frozen_digest((ROOT/name).read_text()),expected)

    def test_same_ub_and_regions(self):
        d=layout(2048,4096,224,True,262144,sorted_union=True,mtp8=True)
        self.assertEqual(d['dynamic_ub_bytes'],181024)
        chunk=612;work=d['scratch_elements']-128-8*2048-4096
        self.assertLessEqual(40*chunk,work)
        # Lifetime1: original inputs and raw two half results follow the reference.
        # Lifetime2: rawA/rawB and both compact results must be disjoint.
        second={'rawB':(0,8*chunk),'rawA':(16*chunk,24*chunk),'compactA':(24*chunk,32*chunk),'compactB':(32*chunk,40*chunk)}
        for a,x in second.items():
            for b,y in second.items():
                if a<b:self.assertTrue(x[1]<=y[0] or y[1]<=x[0],(a,b))
        # Lifetime3 final output and MID table are disjoint. Padding stays allocated.
        self.assertLessEqual(16*chunk+8*chunk,work)
        self.assertLessEqual(2*((8*chunk+63)//64*64),16*chunk+64)
        self.assertLessEqual(4*chunk,4095)
        self.assertLess(8*chunk+64,work)
        self.assertTrue(all(x%8==0 for v in second.values() for x in v))

    def test_short_matrix(self):
        raw=subprocess.check_output([sys.executable,'-B',str(ROOT/'run_mtp8.py'),'--suite','merge_unique',
            '--describe-only','--output','/tmp/unused_merge_unique'],text=True)
        x=json.loads(raw)
        self.assertEqual(x['implementations'],list(run_mtp8.MERGE_UNIQUE_NAMES))
        self.assertEqual(x['target_active_samples'],120)
        self.assertEqual(x['build'],'--merge-unique-only')
        self.assertEqual((x['warmup'],x['repeat'],x['regression_repeat']),(2,5,5))
        self.assertEqual([w['query_rows'] for w in x['workloads'][:4]],[28,56,112,224])

    def test_ratio_and_scope(self):
        names=run_mtp8.MERGE_UNIQUE_NAMES
        x={'implementations':names,'repeat':5,'runs':[{'workload':{'scenario':'cold_reset','mtp':7},'stats':{
            n:{'p50_us':100-10*i,'p95_us':110-10*i,'count':5,'block_nums':['56'],
               'correctness':{'joint_groups':28,'overflow_groups':0,'total_transfer_descriptors':456}}
            for i,n in enumerate(names)}}]}
        text=run_mtp8.report_text(x)
        self.assertIn('0.800 | 20.0%',text)
        self.assertIn('三路对照',text)
        self.assertIn('32位映射',text)
        self.assertNotIn('A/B/AB',text)
        self.assertNotIn('实现/ABCDF',text)

if __name__=='__main__':unittest.main()
