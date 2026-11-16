import unittest,tempfile
from pathlib import Path
from joint_plan_cases import layout
from checks.generate_joint_variants import generate
class Layout(unittest.TestCase):
 def test_bytes_and_rejection(self):
  for mode,nbytes,gm in [(1,197408,262144),(2,203552,1)]:
   x=layout(2048,16384,256,True,262144,sorted_union=True,mtp8=True,layout16=mode)
   self.assertEqual(x['dynamic_ub_bytes'],nbytes);self.assertEqual(x['workspace_elements'],gm)
   self.assertEqual(x['scratch_path'],'UB')
   with self.assertRaises(ValueError):layout(2048,32768,256,True,262144,mtp8=True,layout16=mode)
  with self.assertRaises(ValueError):layout(2048,16384,256,True,262144,sorted_union=True,mtp8=True)
 def test_isolated_build(self):
  with tempfile.TemporaryDirectory() as t:
   generate(t,capacity16k_only=True)
   self.assertEqual({p.name for p in Path(t).glob('*.cpp')},{'variant_mtp7_stage16_512.cpp','variant_mtp7_ub16_512.cpp'})
if __name__=='__main__':unittest.main()
