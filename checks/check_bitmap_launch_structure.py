#!/usr/bin/env python3
"""预处理对照：旧mask不变，mask16仅联合入口少一个VF<false>调用点。非CANN编译。"""
from pathlib import Path
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
BASE = '84769ae5d30b9f1ee1cc04ee34e34b27c86c5ee2'

def preprocess(text, mask):
    # Only outer launch code; standard preprocessor requires no Ascend headers.
    text = text[text.index('// Direct launch adaptation only:'):]
    text = '\n'.join(line for line in text.splitlines() if not line.startswith('#include'))
    return subprocess.run(['g++', '-E', '-P', '-x', 'c++', '-DJOINT_OPT_MASK=' + str(mask), '-'],
                          input=text, text=True, capture_output=True, check=True).stdout

def main():
    old = subprocess.run(['git', 'show', BASE + ':csrc/joint_variants/kernel.cpp'], cwd=ROOT,
                         text=True, capture_output=True, check=True).stdout
    new = (ROOT / 'csrc/joint_variants/kernel.cpp').read_text()
    marker = '// Direct launch adaptation only:'
    assert old[:old.index(marker)] == new[:new.index(marker)], 'VF/algorithm changed'
    for mask in range(1, 16):
        assert preprocess(old, mask) == preprocess(new, mask), f'old mask {mask} changed'
    before, after = preprocess(old, 16), preprocess(new, 16)
    assert before.count('asc_vf_call<SparseKvPlanRuntimeVf<false>>') == 2
    assert after.count('asc_vf_call<SparseKvPlanRuntimeVf<false>>') == 1
    assert after.count('asc_vf_call<SparseKvPlanRuntimeVf<true>>') == 1
    begin = after.index('if constexpr (!AllowJoint)')
    split = after.index('} else {', begin)
    assert before[before.index('if constexpr (!AllowJoint)'):before.index('} else {', before.index('if constexpr (!AllowJoint)'))] == after[begin:split]
    print('PASS: masks1..15 unchanged; mask16 AllowJoint=false unchanged; VF body unchanged; joint single VF<true>')

if __name__ == '__main__':
    main()
