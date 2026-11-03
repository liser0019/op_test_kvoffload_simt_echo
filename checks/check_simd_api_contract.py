#!/usr/bin/env python3
"""CPU compile-time checks for the ShiftLefts scalar contract, NOT CANN/NPU."""
from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]

class Checks(unittest.TestCase):
    def test_scalar_types(self):
        with tempfile.TemporaryDirectory(prefix='simd_api_contract_') as tmp:
            path=Path(tmp)/'probe.cpp'
            for scalar,accepted in (('int16_t',True),('uint32_t',False)):
                path.write_text('#include "sorted_cann_cpu.h"\nvoid probe(){AscendC::Reg::RegTensor<uint32_t> a,b;'
                    'auto mask=AscendC::Reg::CreateMask<uint32_t>();'
                    'AscendC::Reg::ShiftLefts(a,b,'+scalar+'{1},mask);}\n')
                result=subprocess.run(['g++','-std=c++20','-fsyntax-only','-I'+str(ROOT/'checks/simt_variants'),str(path)],capture_output=True,text=True)
                self.assertEqual(result.returncode==0,accepted,result.stderr)
                if not accepted:self.assertIn('CANN ShiftLefts scalar must be int16_t',result.stderr)
    def test_device_source_uses_exact_scalar_types(self):
        source=(ROOT/'csrc/joint_variants/union_next_simd.h').read_text()
        self.assertIn('Reg::ShiftLefts(even,even,int16_t{1},full)',source)
        self.assertIn('Reg::ShiftLefts(packed,prefix,int16_t{16},full)',source)
        self.assertEqual(source.count('Reg::ShiftLefts('),2)
if __name__=='__main__':unittest.main()
