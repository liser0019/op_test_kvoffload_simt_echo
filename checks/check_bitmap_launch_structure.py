#!/usr/bin/env python3
"""预处理对照：当前无页表ABI：mask16/17保留单VF联合启动。非CANN编译。"""
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
    source = (ROOT / 'csrc/joint_variants/kernel.cpp').read_text()
    for mask in (16, 17, 18, 19, 20):
        launch = preprocess(source, mask)
        begin = launch.index('if constexpr (!AllowJoint)')
        split = launch.index('} else {', begin)
        assert launch[begin:split].count('asc_vf_call<SparseKvPlanRuntimeVf<false>>') == 1
        assert launch[split:].count('asc_vf_call<SparseKvPlanRuntimeVf<false>>') == 0
        assert launch[split:].count('asc_vf_call<SparseKvPlanRuntimeVf<true>>') == 1
    print('PASS: bitmap/turbo/A/B/AB private VF<false>; joint single VF<true>, including MTP0')

if __name__ == '__main__':
    main()
