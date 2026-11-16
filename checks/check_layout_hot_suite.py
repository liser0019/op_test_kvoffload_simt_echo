import unittest,tempfile,subprocess,sys,json
from pathlib import Path
from joint_variants import LAYOUT_HOT_MASKS,LAYOUT_HOT_NAMES,LAYOUT_HOT_REFS,layout16_mode,layout_hot_bits
from joint_plan_cases import layout
from checks.check_capacity16 import cases
from checks.generate_joint_variants import generate
from checks.layout_hot_suite import select,report_text
ROOT=Path(__file__).resolve().parents[1]
class Checks(unittest.TestCase):
 def test_layout_sizes(self):
  expected={4096:123680,8192:148256,16384:197408}
  for n,m in LAYOUT_HOT_MASKS.items():
   mode=layout16_mode('joint_'+n);bits=layout_hot_bits(m)
   for c in expected:
    d=layout(2048,c,256,True,262144,sorted_union=True,mtp8=True,layout16=mode,layout_hot_bits=bits)
    size=expected[c] if mode==1 or c<16384 else 205600 if bits&2 else 203552
    self.assertEqual(d['dynamic_ub_bytes'],size,n)
    self.assertLessEqual(size,204*1024+288)
    self.assertEqual(d['workspace_elements'],262144 if mode==1 else 1)
 def test_host_formula(self):
  # Compile real host sizing helper independently of torch/CANN.
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'test.cpp').write_text('#include "'+str(ROOT/'csrc/joint_variants/layout_hot_tiling.h')+'"\n#include <cstdio>\nint main(){for(int m=1;m<=2;++m)for(int b:{1,3,5,7,13,15})for(int c:{4096,8192,16384})printf("%d %d %d %llu\\n",m,b,c,(unsigned long long)LayoutHotScratch(c,3*4096+c+2048,m,b));}')
   subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
   for line in subprocess.check_output([str(p/'test')],text=True).splitlines():
    mode,bits,c,size=map(int,line.split());d=layout(2048,c,256,True,262144,sorted_union=True,mtp8=True,layout16=mode,layout_hot_bits=bits)
    self.assertEqual(size,d['scratch_elements'])
 def test_wrappers(self):
  with tempfile.TemporaryDirectory() as t:
   generate(t,layout_hot_only=True);self.assertEqual(len(list(Path(t).glob('*.cpp'))),14)
   for n,m in LAYOUT_HOT_MASKS.items():
    s=(Path(t)/f'variant_{n}.cpp').read_text();self.assertIn(f'#define JOINT_LAYOUT_HOT_BITS {layout_hot_bits(m)}',s);self.assertIn('#define JOINT_MTP7_FUSED_TUNE_BITS 3',s)
 def test_regression_case(self):
  dup=dict(cases())['duplicates'];self.assertEqual((dup['mtp'],dup['capacity']),(7,16384))
 def mock(self):
  names=(*LAYOUT_HOT_REFS,*LAYOUT_HOT_NAMES);runs=[]
  for mtp,c in [(0,4096),(1,4096),(3,8192),(7,16384)]:
   st={n:{'p50_us':100.,'p95_us':101.,'count':5,'correctness':{'total_transfer_descriptors':12}} for n in names}
   for n in LAYOUT_HOT_NAMES:
    st[n]['p50_us']=80 if n.endswith('_abcd_512') else 90
   runs.append({'workload':{'scenario':'full_cache_75pct','mtp':mtp,'capacity':c},'stats':st})
  return {'status':'complete','runs':runs}
 def test_fixed_selection_and_capacity_report(self):
  report=self.mock();names,ranking=select(report)
  self.assertEqual(len(names),5);self.assertEqual(names[-2:],['joint_layout_stage_abcd_512','joint_layout_ub_abcd_512'])
  self.assertIn('full_cache_75pct/3/8192',report_text(report))
  report['status']='failed'
  with self.assertRaises(ValueError):select(report)
if __name__=='__main__':unittest.main()
