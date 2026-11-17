"""Compile the actual generated host resource functions with the CPU compiler."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from scripts.prepare_qli_fusion import NAMES,source_id

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True,type=Path);a=p.parse_args()
    manifest=json.loads((a.upstream/'qli_fusion_manifest.json').read_text())
    if manifest['source_id']!=source_id():raise RuntimeError('stale generated upstream overlay')
    records=[]
    for mode,name in NAMES.items():
        host=a.upstream/'attention'/name/'op_host'
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);src=td/'probe.cpp';exe=td/'probe'
            source='''#include <iostream>
#include "fused_resource.h"
#include "qli_plan_resources.h"
int main(){for(unsigned c:{4096U,8192U,16384U}){
 auto plan=qli_fusion_resource::LocalBytes(c);
 auto r=qli_fused_host::ComputeResourceBytes(2,32,2048,plan,qli_fused_host::QliDataPath::FP8_BF16);
 if(r.total>216U*1024U)return 2;
 std::cout<<c<<" "<<plan<<" "<<r.total<<"\\n";
}}
'''
            src.write_text(source.replace('qli_fusion_resource',name+'_resource').replace('qli_fused_host',name+'_host'))
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(host),str(src),'-o',str(exe)],check=True)
            for line in subprocess.check_output([str(exe)],text=True).splitlines():
                c,plan,total=map(int,line.split());records.append({'mode':mode,'capacity':c,'plan_bytes':plan,'fusion_bytes':total})
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
