"""Real candidate controller/SIMT/SIMD/MTE-semantic CPU models, not CANN/NPU."""
import argparse
from joint_variants import RANKPIPE_MASKS
from checks.check_layout_hot_kernels import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masks',nargs='+',type=int)
    p.add_argument('--model-threads',type=int,choices=(64,128),default=64)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    selected=[(n,m) for n,m in RANKPIPE_MASKS.items() if a.masks is None or m in a.masks]
    if not selected:p.error('no candidates')
    run(selected,a.model_threads,a.smoke)
