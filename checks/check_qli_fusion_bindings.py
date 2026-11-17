"""Validate generated Torch schemas and host C++ syntax, without CANN/NPU.

The ACLNN call is stubbed for syntax only: this cannot validate CANN linkage or
kernel compilation. No operator is executed.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import types
import torch
from scripts.prepare_qli_fusion import NAMES

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True,type=Path);p.add_argument('--syntax',action='store_true');a=p.parse_args()
    lib=torch.library.Library('qli_fusion_cpu_schema_check','DEF')
    class Builder:
        def __init__(self,name,**kwargs):self.name=name
        def ensure_initialized(self):
            for schema in self.schema():lib.define(schema)
            self.register_meta()
    fake=types.ModuleType('cann_ops_transformer.op_builder');fake.OpBuilder=Builder;fake.get_as_library=lambda:lib
    sys.modules['cann_ops_transformer.op_builder']=fake
    for name in NAMES.values():
        base=a.upstream/'attention'/name/'torch_extension'
        spec=importlib.util.spec_from_file_location('cpu_check_'+name,base/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        print(name,'Torch schema + Meta/PrivateUse1 registration passed')
        if a.syntax:
            from torch.utils.cpp_extension import include_paths
            with tempfile.TemporaryDirectory() as td:
                td=Path(td)
                (td/'aclnn_common.h').write_text('#pragma once\nnamespace at_npu {namespace native {}}\n#define ACLNN_CMD(...) do {} while(0)\n')
                subprocess.run(['g++','-std=c++17','-fsyntax-only','-DTORCH_EXTENSION_NAME=qli_fusion_cpu_syntax',
                    '-I'+str(td),'-I'+sysconfig.get_paths()['include'],*['-I'+s for s in include_paths()],
                    str(base/'csrc'/(name+'.cpp'))],check=True)
            print(name,'host C++ syntax passed (ACLNN stub; not a CANN build)')

if __name__=='__main__':main()
