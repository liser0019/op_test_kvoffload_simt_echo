"""Rank-pipeline catalog, exact rank model, UB lifetime and CANN9.2.1 API regressions."""
import bisect
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

from joint_variants import (RANKPIPE_MASKS,RANKPIPE_NAMES,RANKPIPE_REFS,rankpipe_bits,
                            stage_exec_bits,layout_hot_bits,layout16_mode)
from joint_plan_cases import layout
from checks.generate_joint_variants import generate
from checks.rankpipe_suite import select,report_text,factor_text
ROOT=Path(__file__).resolve().parents[1]


class Checks(unittest.TestCase):
    def mock(self):
        names=(*RANKPIPE_REFS,*RANKPIPE_NAMES[1:]);runs=[]
        for scenario in ('cold_reset','full_cache_75pct'):
            for m,c in ((0,4096),(1,4096),(3,8192),(7,16384)):
                st={n:{'p50_us':100.,'p95_us':101.,'count':5,'correctness':{'total_transfer_descriptors':8}} for n in names}
                st['joint_rankpipe_a_512']['p50_us']=80
                st['joint_rankpipe_ac_512']['p50_us']=85
                if m==7:st['joint_rankpipe_abc_512']['p50_us']=60
                runs.append({'workload':{'scenario':scenario,'mtp':m,'capacity':c},'stats':st})
        return {'status':'complete','runs':runs}

    def test_wrappers_and_catalog(self):
        self.assertEqual({rankpipe_bits(m) for m in RANKPIPE_MASKS.values()},set(range(8)))
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp,rankpipe_only=True)
            names={p.stem[8:] for p in Path(tmp).glob('variant_*.cpp')}
            self.assertEqual(names,{'stage_exec_abcd_512',*RANKPIPE_MASKS})
            for name,mask in RANKPIPE_MASKS.items():
                self.assertEqual((stage_exec_bits(mask),layout_hot_bits(mask),layout16_mode('joint_'+name)),(15,15,1))
                text=(Path(tmp)/('variant_'+name+'.cpp')).read_text()
                self.assertIn('#define JOINT_STAGE_EXEC_BITS 15\n',text)
                self.assertIn(f'#define JOINT_RANKPIPE_BITS {mask-244}\n',text)
            self.assertNotIn('JOINT_RANKPIPE',(Path(tmp)/'variant_stage_exec_abcd_512.cpp').read_text())
            cmake=Path(ROOT/'joint_plan/CMakeLists.txt').read_text()
            pattern=re.search(r'JOINT_RANKPIPE_ONLY AND NOT _name MATCHES "([^"]+)"',cmake).group(1)
            self.assertTrue(all(re.fullmatch(pattern,n) for n in names))

    def test_fixed_selection_and_conditionals(self):
        r=self.mock();chosen,ranking=select(r)
        self.assertEqual(chosen,[*RANKPIPE_REFS,'joint_rankpipe_a_512','joint_rankpipe_ac_512'])
        self.assertEqual(len(ranking),7)
        self.assertIn('full_cache_75pct/7/16384',report_text(r))
        effects=factor_text(r)
        for letter in 'ABC':self.assertEqual(sum(line.startswith(f'| {letter} |') for line in effects.splitlines()),4)
        r['runs'].pop()
        with self.assertRaises(ValueError):select(r)

    def test_exact_word_rank(self):
        rng=random.Random(2026)
        for n in (0,1,31,32,33,63,64,65,2048,4096,8192,16384):
            tokens=sorted(rng.sample(range(262144),n))
            # Deliberately include dense full words and sign-bit boundaries.
            if n>=65:tokens=sorted(set(range(65))|set(tokens))[:n]
            masks={};base={}
            for uid,t in enumerate(tokens):
                w=t>>5
                if w not in masks:base[w]=uid;masks[w]=0
                masks[w]|=1<<(t&31)
            for token in [*tokens,*[rng.randrange(262144) for _ in range(1000)]]:
                w=token>>5;bit=1<<(token&31);mask=masks.get(w,0)
                uid=base[w]+(mask&(bit-1)).bit_count() if mask&bit else None
                lo=bisect.bisect_left(tokens,token)
                self.assertEqual(uid,lo if lo<len(tokens) and tokens[lo]==token else None)

    def test_three_capacity_byte_lifetimes(self):
        records=[]
        for c,expected in ((4096,123680),(8192,148256),(16384,197408)):
            d=layout(2048,c,256,True,262144,sorted_union=True,mtp8=True,layout16=1,layout_hot_bits=15)
            self.assertEqual(d['dynamic_ub_bytes'],expected)
            # All offsets below are bytes from the recycled merge work start.
            phases={'index':[(0,32768),(32768,65536),(65536,65536+2*c)],
                    'owner_partition':[(0,4*c),(4*c,6*c)]}
            for phase,spans in phases.items():
                for lo,hi in spans:self.assertTrue(0<=lo<hi<=98304)
                for i,(lo,hi) in enumerate(spans):
                    for a,b in spans[i+1:]:self.assertTrue(hi<=a or b<=lo)
            # C reuses unique/owner only after all original readers finish.
            self.assertLessEqual(4*c+2*c,98304)
            records.append((c,d['dynamic_ub_bytes']))
        self.assertEqual(len(records),3)

    def test_unpack_contract_rejects_old_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for typ,accepted in (('uint16_t',False),('uint32_t',True)):
                source='#include "sorted_cann_cpu.h"\nvoid f(){AscendC::Reg::RegTensor<uint32_t> v;alignas(32) '+typ+' x[128]{};AscendC::Reg::LoadAlign<'+typ+',AscendC::Reg::LoadDist::DIST_UNPACK_B16>(v,x);}'
                (p/'probe.cpp').write_text(source)
                r=subprocess.run(['g++','-std=c++20','-fsyntax-only','-pthread','-I'+str(ROOT/'checks/simt_variants'),str(p/'probe.cpp')],capture_output=True,text=True)
                self.assertEqual(r.returncode==0,accepted,r.stderr)
                if not accepted:self.assertIn('CANN9.2.1 unpack',r.stderr)

    def test_describe_only(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'run_rankpipe_sweep.py'),'--describe-only','--output','/tmp/unused_rankpipe'],capture_output=True,text=True,check=True)
        d=json.loads(r.stdout);self.assertEqual((len(d['implementations']),d['screen_target_samples']),(10,400))

if __name__=='__main__':unittest.main()
