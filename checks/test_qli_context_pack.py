"""CPU checks for the C-only streaming-context packing."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class ContextPack(unittest.TestCase):
    def test_c_8k_plan_reclaims_exactly_32_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'probe.cpp'
            executable=Path(td)/'probe'
            source.write_text('''#include "resources.h"
static_assert(QLI_PLAN_STORAGE==3);
static_assert(qli_fusion_resource::kContextWords==120U);
static_assert(qli_fusion_resource::LocalBytes(8192U)==123648U);
static_assert(qli_fusion_resource::LocalBytes(16384U)==140032U);
int main(){return 0;}
''')
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',
                            '-DQLI_PLAN_STORAGE=3','-I'+str(ROOT/'csrc/qli_fusion'),
                            str(source),'-o',str(executable)],check=True)
            subprocess.run([str(executable)],check=True)

    def test_other_storage_modes_keep_full_context(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'probe.cpp'
            executable=Path(td)/'probe'
            source.write_text('''#include "resources.h"
static_assert(QLI_PLAN_STORAGE==1);
static_assert(qli_fusion_resource::kContextWords==128U);
int main(){return 0;}
''')
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',
                            '-DQLI_PLAN_STORAGE=1','-I'+str(ROOT/'csrc/qli_fusion'),
                            str(source),'-o',str(executable)],check=True)
            subprocess.run([str(executable)],check=True)

    def test_generated_plan_is_current(self):
        subprocess.run(['python3','-B',str(ROOT/'scripts/export_fusion_plan.py'),'--check'],
                       check=True,cwd=ROOT)

if __name__=='__main__':
    unittest.main()
