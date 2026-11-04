#!/usr/bin/env python3
"""CPU-only buffer reuse/tiled merge checks; no CANN/NPU validation."""
from pathlib import Path
import argparse,json,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from checks.check_mtp7_fast import worker

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=(196,197,198,199));a=p.parse_args()
    if a.worker:print(json.dumps(worker(a.worker)),flush=True);return
    for mask in (196,197,198,199):subprocess.run([sys.executable,'-B',__file__,'--worker',str(mask)],check=True)
if __name__=='__main__':main()
