#!/usr/bin/env python3
"""CPU compile/link of full and merge-only dispatch, including holes; no CANN."""
from pathlib import Path
import subprocess, tempfile,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from checks.generate_joint_variants import generate
from joint_variants import MASKS

def main():
    for subset in ("all","merge","sorted","sweep","union","ac_threads","cold_compare","integrated","refine","exec"):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);generate(d,merge_only=subset=="merge",sorted_only=subset=="sorted",sweep_only=subset=="sweep",union_only=subset=="union",ac_threads_only=subset=="ac_threads",cold_compare_only=subset=="cold_compare",integrated_only=subset=="integrated",refine_only=subset=="refine",exec_only=subset=="exec")
            declaration=(ROOT/'csrc/joint_plan/launch.h').read_text().split('extern "C" void launch_joint_plan',1)[1]
            declaration='extern "C" void launch_joint_plan'+declaration.strip()
            selected={n:m for n,m in MASKS.items() if subset=='all' or (subset=='merge' and n.startswith('merge')) or (subset=='sorted' and n in ('merge','merge_bd','sorted_stream','sorted_mrgsort')) or (subset=='sweep' and (n=='sorted_mrgsort' or n.startswith('hw_'))) or (subset=='union' and (n in ('hw_ac_512','hw_ac_256','hw_ac_cold_512') or n.startswith('union_'))) or (subset=='ac_threads' and n in ('hw_ac_512','hw_ac_256')) or (subset=='cold_compare' and n in ('hw_ac_512','hw_ac_256','union_base_512','union_cold_512','hw_ac_cold_512')) or (subset=='integrated' and n in ('union_cold_512','hw_ac_cold_512','integrated_512','integrated_bounds_512')) or (subset=='exec' and (n in ('integrated_bounds_512','refine_abd_512','refine_bd_512') or n.startswith('exec_'))) or (subset=='refine' and (n=='integrated_bounds_512' or n.startswith('refine_')))}
            assert len(list(d.glob('variant_*.cpp')))==len(selected)
            text='#include <cstdint>\n#include <cassert>\n#include "'+str(ROOT/'csrc/joint_plan/launch.h')+'"\n#include "variant_dispatch.h"\n'
            for name in selected:text+=declaration.replace('launch_joint_plan','launch_joint_plan_'+name).rstrip(';')+' {}\n'
            text+='int main() { assert(JointVariantLaunch(0)==nullptr); assert(JointVariantLaunch(159)==nullptr); assert(JointVariantAlgorithmMask(122)==59); assert(JointVariantAlgorithmMask(72)==72); assert(JointVariantAlgorithmMask(123)==90); assert(JointVariantAlgorithmMask(124)==72); assert(JointVariantAlgorithmMask(125)==72); assert(JointVariantAlgorithmMask(126)==72); assert(JointVariantAlgorithmMask(127)==72); assert(JointVariantAlgorithmMask(142)==72); assert(JointVariantAlgorithmMask(143)==72); assert(JointVariantAlgorithmMask(158)==72);\n'
            for name,mask in MASKS.items():text+=f'assert((JointVariantLaunch({mask}) != nullptr) == {str(name in selected).lower()});\n'
            text+='}\n';(d/'probe.cpp').write_text(text)
            subprocess.run(['g++','-std=c++17',str(d/'probe.cpp'),'-o',str(d/'probe')],check=True)
            subprocess.run([str(d/'probe')],check=True)
    print('all/merge-only/sorted-only/sweep-only dispatch compile+link checks passed (CPU)')
if __name__=='__main__':main()
