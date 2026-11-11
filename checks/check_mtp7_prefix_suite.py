#!/usr/bin/env python3
"""CPU-only A/B/AB catalogue, frozen196 identity, geometry and bounded runner checks."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from checks.generate_joint_variants import generate
from joint_variants import VARIANTS, MTP7_PREFIX_MASKS, description
from joint_plan_cases import layout
import run_mtp8


def frozen_digest(source):
    result=subprocess.run(['cpp','-P','-DJOINT_OPT_MASK=72','-DJOINT_REFINE_BITS=11',
                           '-DJOINT_MTP7_OPT=1','-DJOINT_MTP7_REUSE=1','-DJOINT_MTP7_TILE=0',
                           '-DJOINT_MTP7_PREFIX=0','-DJOINT_MTP7_MAP16=0','-'],
                          input=source,text=True,capture_output=True,check=True)
    return hashlib.sha256(re.sub(r'\s+','',result.stdout).encode()).hexdigest()


class Checks(unittest.TestCase):
    def test_four_wrappers_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp,prefix_only=True)
            self.assertEqual({p.stem for p in Path(tmp).glob('variant_*.cpp')},
                             {'variant_'+n.removeprefix('joint_') for n in run_mtp8.PREFIX_NAMES})
            for n,m in MTP7_PREFIX_MASKS.items():
                self.assertEqual(VARIANTS['joint_'+n],m)
                source=(Path(tmp)/('variant_'+n+'.cpp')).read_text()
                for token in ('#define JOINT_MTP7_OPT 1','#define JOINT_MTP7_REUSE 1','#define JOINT_MTP7_TILE 0',
                              '#define JOINT_MTP7_PREFIX '+str(int(m in (200,202))),
                              '#define JOINT_MTP7_MAP16 '+str(int(m in (201,202)))):
                    self.assertIn(token,source)
                self.assertEqual(description('joint_'+n)['threads'],512)

    def test_frozen196_body_and_cold_signature(self):
        manifest=json.loads((ROOT/'joint_plan_manifest.json').read_text())
        for name,expected in manifest['mtp7_prefix_frozen196_preprocessor_sha256'].items():
            self.assertEqual(frozen_digest((ROOT/name).read_text()),expected,name)

    def test_same_ub_larger_packet_and_simd_tail(self):
        d=layout(2048,4096,224,True,262144,sorted_union=True,mtp8=True)
        self.assertEqual(d['dynamic_ub_bytes'],181024)
        q=8*2048
        for packed,expected in ((False,612),(True,816)):
            words=(q+1)//2 if packed else q
            work=d['scratch_elements']-128-((words+4096+7)&~7)
            chunk=min(expected,(work//40)//(8 if packed else 4)*(8 if packed else 4))
            self.assertEqual(chunk,expected)
            self.assertLessEqual(40*chunk,work)
            self.assertLessEqual(24*chunk+2*((8*chunk+63)//64*64),work)
            self.assertLessEqual(4*chunk,4095)
            self.assertEqual(2*chunk*4%32,0)
        self.assertEqual((q-q//2)*4,32768)
        self.assertLessEqual(65408+127,65535)

    def test_cli_matrix_once_through(self):
        raw=subprocess.check_output([sys.executable,'-B',str(ROOT/'run_mtp8.py'),
                                     '--suite','prefix','--describe-only','--output','/tmp/unused_prefix'],text=True)
        report=json.loads(raw)
        self.assertEqual(report['implementations'],list(run_mtp8.PREFIX_NAMES))
        self.assertEqual(report['target_active_samples'],160)
        self.assertEqual(report['repeat'],5)
        self.assertEqual(report['warmup'],2)
        self.assertEqual(report['regression_repeat'],5)
        self.assertEqual(report['build'],'--prefix-only')
        self.assertEqual([w['query_rows'] for w in report['workloads'][:4]],[28,56,112,224])

    def test_report_ratio_direction(self):
        names=run_mtp8.PREFIX_NAMES
        report={'implementations':names,'repeat':5,'runs':[{'workload':{'scenario':'cold_reset','mtp':7},
            'stats':{n:{'p50_us':100-10*i,'p95_us':110-10*i,'count':5,'block_nums':['56'],
                        'correctness':{'joint_groups':28,'overflow_groups':0,'total_transfer_descriptors':123}}
                     for i,n in enumerate(names)}}]}
        text=run_mtp8.report_text(report)
        self.assertIn('0.700 | 30.0%',text)
        self.assertIn('28/0',text)
        self.assertIn('Block Num',text)
        self.assertNotIn('实现/ABCDF',text)

if __name__=='__main__':unittest.main()
