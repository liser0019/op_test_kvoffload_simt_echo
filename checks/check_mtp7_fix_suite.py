#!/usr/bin/env python3
"""Catalogue, frozen preprocessor and short-run/report regression. CPU only."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.generate_joint_variants import generate
from joint_variants import supports_eight_rows,description
import run_mtp8

class Checks(unittest.TestCase):
    def test_build_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);generate(p,mtp7_fix_only=True)
            names={'hot_bd8_512':0,'mtp7_parallel_512':1,'mtp7_rank_512':2}
            self.assertEqual({f.stem for f in p.glob('variant_*.cpp')},{'variant_'+n for n in names})
            for n,opt in names.items():
                s=(p/('variant_'+n+'.cpp')).read_text()
                self.assertIn(f'#define JOINT_MTP7_OPT {opt}',s);self.assertIn('#define JOINT_HOT_BITS 10',s)
                self.assertTrue(supports_eight_rows('joint_'+n));self.assertEqual(description('joint_'+n)['UB_limit_bytes'],192*1024)

    def test_short_plan_and_formal(self):
        for extra,repeat,active in (([],5,160),(['--formal'],50,1600)):
            r=subprocess.run([sys.executable,'-B',str(ROOT/'run_mtp8.py'),'--suite','mtp7_fix','--describe-only','--output','/tmp/unused_mtp7',*extra],capture_output=True,text=True,check=True)
            d=json.loads(r.stdout);self.assertEqual(d['build'],'--mtp7-fix-only');self.assertEqual(d['repeat'],repeat);self.assertEqual(d['target_active_samples'],active)
            self.assertEqual(d['implementations'],list(run_mtp8.FIX_NAMES))

    def test_report_uses_frozen_bd8(self):
        names=run_mtp8.FIX_NAMES
        report={'implementations':names,'repeat':5,'mtp_comparison':[],'runs':[{'workload':{'scenario':'cold_reset','mtp':7},'stats':{n:{'p50_us':10+i,'p95_us':12+i,'count':5,'correctness':{'joint_groups':32,'overflow_groups':0,'total_transfer_descriptors':123}} for i,n in enumerate(names)}}]}
        text=run_mtp8.report_text(report);self.assertIn('实现/上轮BD8',text);self.assertIn('joint_mtp7_rank_512',text)

    def test_frozen_compute_preprocessor(self):
        source=(ROOT/'csrc/joint_variants/mtp8_plan_impl.h').read_text()
        source=re.sub(r'static_assert\(JOINT_MTP7_OPT[^\n]+\n','',source)
        r=subprocess.run(['cpp','-P','-DJOINT_OPT_MASK=72','-DJOINT_REFINE_BITS=11','-DJOINT_MTP7_OPT=0','-'],input=source,text=True,capture_output=True,check=True)
        digest=hashlib.sha256(re.sub(r'\s+','',r.stdout).encode()).hexdigest()
        self.assertEqual(digest,json.loads((ROOT/'joint_plan_manifest.json').read_text())['mtp7_fix_frozen_preprocessor_sha256'])

    def test_no_extra_ub_or_kernel_abi(self):
        s=(ROOT/'csrc/joint_variants/mtp8_plan_impl.h').read_text()
        self.assertIn('const uint32_t t=threadIdx.x;',s)
        self.assertIn('auto* partials=records-static_cast<uint32_t>(ctx[30]);',s)
        self.assertIn('for(uint32_t i=t;i<n;i+=SORTED_THREADS)',s)
        self.assertIn('for(uint32_t r=0;r<8U;++r)bins[r]+=row==r?1U:0U;',s)
        self.assertNotIn('++bins[row]',s)
        a=(ROOT/'csrc/joint_plan/launch.h').read_text();self.assertNotIn('perf',a)
if __name__=='__main__':unittest.main()
