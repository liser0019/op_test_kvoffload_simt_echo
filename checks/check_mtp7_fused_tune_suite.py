#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.generate_joint_variants import generate
from joint_plan_cases import layout
from joint_variants import description
import run_mtp8

class Checks(unittest.TestCase):
    def test_eight_only(self):
        with tempfile.TemporaryDirectory() as t:
            generate(t,fused_tune_only=True)
            files=list(Path(t).glob('variant_*.cpp'));self.assertEqual(len(files),8)
            self.assertEqual({p.stem.removeprefix('variant_') for p in files},{n.removeprefix('joint_') for n in run_mtp8.FUSED_TUNE_NAMES})
            for p in files:
                text=p.read_text();self.assertIn('#define JOINT_MTP7_FUSED_COUNT 1',text)
                self.assertNotIn('#define JOINT_MTP7_MAP16 1',text);self.assertNotIn('#define JOINT_MTP7_PREFIX 1',text)
                self.assertEqual(description('joint_'+p.stem.removeprefix('variant_'))['threads'],512)
    def test_frozen204(self):
        args=['cpp','-P','-DJOINT_OPT_MASK=72','-DJOINT_REFINE_BITS=11','-DJOINT_MTP7_OPT=1','-DJOINT_MTP7_REUSE=1','-DJOINT_MTP7_TILE=0','-DJOINT_MTP7_PREFIX=0','-DJOINT_MTP7_MAP16=0','-DJOINT_MTP7_FUSED_COUNT=1','-I'+str(ROOT/'csrc/joint_variants'),'-']
        x=subprocess.run(args,input=(ROOT/'csrc/joint_variants/mtp8_plan_impl.h').read_text(),text=True,capture_output=True,check=True)
        h=hashlib.sha256(re.sub(r'\s+','',x.stdout).encode()).hexdigest()
        self.assertEqual(h,json.loads((ROOT/'joint_plan_manifest.json').read_text())['frozen_fused204_preprocessor_sha256'])
    def test_dcci_ub_and_hooks(self):
        cmake=(ROOT/'joint_plan/CMakeLists.txt').read_text();self.assertNotIn('-cce-aicore-dcci-before-kernel-end=false',cmake)
        self.assertIn('set(JOINT_FLAGS -O3',cmake)
        self.assertEqual(layout(2048,4096,224,True,262144,sorted_union=True,mtp8=True)['dynamic_ub_bytes'],181024)
        s=(ROOT/'csrc/joint_variants/mtp8_plan_impl.h').read_text()
        self.assertIn('fullA=used[0]==lens[0],fullB=used[1]==lens[1]',s)
        self.assertIn('if(ctx[28]==8)',s)
        self.assertIn('ctx[112+begin+k]=static_cast<int32_t>(used[k])',s)
    def test_short_matrix(self):
        out=subprocess.check_output([sys.executable,'-B',str(ROOT/'run_mtp8.py'),'--suite','fused_tune','--describe-only','--output','/tmp/unused_fused_tune'],text=True)
        x=json.loads(out);self.assertEqual(x['target_active_samples'],320)
        self.assertEqual(x['implementations'],list(run_mtp8.FUSED_TUNE_NAMES));self.assertEqual(x['build'],'--fused-tune-only')
        self.assertEqual((x['warmup'],x['repeat'],x['regression_repeat']),(2,5,5))
    def test_report_direction(self):
        names=run_mtp8.FUSED_TUNE_NAMES
        data={'suite':'fused_tune','implementations':names,'runs':[{'workload':{'scenario':'cold_reset','mtp':7},'stats':{n:{'p50_us':100-5*i,'p95_us':110-5*i,'count':5,'block_nums':['56'],'correctness':{'joint_groups':28,'overflow_groups':0,'total_transfer_descriptors':9}} for i,n in enumerate(names)}}]}
        s=run_mtp8.report_text(data);self.assertIn('0.650 | 35.0%',s);self.assertIn('实现/FUSED',s)
        self.assertNotIn('实现/ABCDF',s)
if __name__=='__main__':unittest.main()
