#!/usr/bin/env python3
"""CPU-only bounded-sweep wiring, UB lifetimes and unchanged references."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.generate_joint_variants import generate
from joint_variants import mtp7_config,MTP7_PACK_MASKS,description
from joint_plan_cases import layout
import run_mtp8

class Checks(unittest.TestCase):
    def test_catalog_and_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);generate(d,mtp7_pack_only=True)
            names=['hot_bd8_512','mtp7_parallel_512','mtp7_rank_512',*MTP7_PACK_MASKS]
            self.assertEqual({f.stem for f in d.glob('variant_*.cpp')},{'variant_'+n for n in names})
            for n,mask in MTP7_PACK_MASKS.items():
                s=(d/('variant_'+n+'.cpp')).read_text();c=mtp7_config(mask)
                for token in (f"#define JOINT_MTP7_OPT {c['opt']}",f"#define JOINT_MTP7_REUSE {int(c['reuse'])}",f"#define JOINT_MTP7_TILE {int(c['tile'])}"):
                    self.assertIn(token,s)
                self.assertEqual(description('joint_'+n)['threads'],512)

    def test_same_ub_and_live_regions(self):
        d=layout(2048,4096,256,True,262144,sorted_union=True,mtp8=True)
        self.assertEqual(d['dynamic_ub_bytes'],181024)
        work=d['scratch_elements']-128-8*2048-4096
        chunk=min(612,(work//40)//4*4)
        self.assertEqual(chunk,612);self.assertLessEqual(40*chunk,work)
        # Reg Gather may form indices for inactive lanes. Include the full final64-record tile.
        padded_records=(8*chunk+63)//64*64
        self.assertLessEqual(24*chunk+2*padded_records,work)
        self.assertLessEqual(4*chunk,4095)
        # Current/old input A and B, first mid, final output; mid B reuses input A.
        regions={'midA':(16*chunk,24*chunk),'midB':(0,8*chunk),'out':(24*chunk,40*chunk),'partials':(8*chunk,8*chunk+128)}
        keys=list(regions)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                x,y=regions[a],regions[b];self.assertTrue(x[1]<=y[0] or y[1]<=x[0],(a,b))
        self.assertEqual((chunk*2*4)%32,0)

    def test_frozen_preprocessed_bodies(self):
        manifest=json.loads((ROOT/'joint_plan_manifest.json').read_text())
        s=(ROOT/'csrc/joint_variants/mtp8_plan_impl.h').read_text()
        s=re.sub(r'static_assert\(JOINT_MTP7_OPT[^\n]+\n','',s)
        for opt in (0,1,2):
            r=subprocess.run(['cpp','-P','-DJOINT_OPT_MASK=72','-DJOINT_REFINE_BITS=11',f'-DJOINT_MTP7_OPT={opt}','-DJOINT_MTP7_REUSE=0','-DJOINT_MTP7_TILE=0','-'],input=s,text=True,capture_output=True,check=True)
            digest=hashlib.sha256(re.sub(r'\s+','',r.stdout).encode()).hexdigest()
            self.assertEqual(digest,manifest['mtp7_pack_frozen_preprocessor_sha256'][str(opt)])
        for name,digest in manifest['mtp7_pack_unchanged_sources'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)

    def test_cli_is_bounded_once_through(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'run_mtp8.py'),'--suite','mtp7_pack','--describe-only','--output','/tmp/unused_mtp7_pack'],capture_output=True,text=True,check=True)
        x=json.loads(r.stdout)
        self.assertEqual(len(x['implementations']),8);self.assertEqual(x['target_active_samples'],320)
        self.assertEqual(x['repeat'],5);self.assertEqual(x['build'],'--mtp7-pack-only')
        self.assertEqual(x['mtp'],[0,1,3,7])

    def test_report_compares_parents(self):
        report={'implementations':run_mtp8.PACK_NAMES,'repeat':5,'mtp_comparison':[],'runs':[{'workload':{'scenario':'cold_reset','mtp':7},'stats':{n:{'p50_us':10+i,'p95_us':12+i,'count':5,'correctness':{'joint_groups':32,'overflow_groups':0,'total_transfer_descriptors':123}} for i,n in enumerate(run_mtp8.PACK_NAMES)}}]}
        text=run_mtp8.report_text(report)
        for token in ('实现/A','实现/B','joint_mtp7_reuse_tile_512'):self.assertIn(token,text)
if __name__=='__main__':unittest.main()
