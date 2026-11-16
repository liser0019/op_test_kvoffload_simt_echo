"""Host与选择逻辑回归，以及独立的排名/清理等价性模型；不是CANN测试。"""
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from checks.generate_joint_variants import generate
from checks.stage_exec_suite import select, report_text, factor_text
from joint_variants import (STAGE_EXEC_MASKS, STAGE_EXEC_NAMES, STAGE_EXEC_REFS,
                            stage_exec_bits, layout_hot_bits, layout16_mode, description)
from joint_plan_cases import layout

ROOT = Path(__file__).resolve().parents[1]


class Checks(unittest.TestCase):
    def mock(self):
        names = (*STAGE_EXEC_REFS, *STAGE_EXEC_NAMES[1:])
        runs = []
        for scenario in ('cold_reset', 'full_cache_75pct'):
            for m, c in ((0, 4096), (1, 4096), (3, 8192), (7, 16384)):
                stats = {n: {'p50_us': 100., 'p95_us': 101., 'count': 5,
                             'correctness': {'total_transfer_descriptors': 100}} for n in names}
                stats['joint_stage_exec_b_512']['p50_us'] = 80.
                stats['joint_stage_exec_ac_512']['p50_us'] = 85.
                # 单工况的临时赢家不可压过全热工况固定赢家。
                if m == 7:
                    stats['joint_stage_exec_abcd_512']['p50_us'] = 60.
                runs.append({'workload': {'scenario': scenario, 'mtp': m, 'capacity': c}, 'stats': stats})
        return {'status': 'complete', 'runs': runs}

    def test_catalog_layout_wrappers(self):
        self.assertEqual(len(STAGE_EXEC_MASKS), 16)
        self.assertEqual({stage_exec_bits(m) for m in STAGE_EXEC_MASKS.values()}, set(range(16)))
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, stage_exec_only=True)
            self.assertEqual(len(list(Path(tmp).glob('*.cpp'))), 17)
            for n, mask in STAGE_EXEC_MASKS.items():
                self.assertEqual(layout16_mode('joint_' + n), 1)
                self.assertEqual(layout_hot_bits(mask), 15)
                text = (Path(tmp) / f'variant_{n}.cpp').read_text()
                self.assertIn(f'#define JOINT_STAGE_EXEC_BITS {mask-228}\n', text)
                self.assertIn('#define JOINT_MTP7_FUSED_TUNE_BITS 3\n', text)
                self.assertEqual(description('joint_' + n)['parent'], STAGE_EXEC_REFS[1])
                for c, size in ((4096, 123680), (8192, 148256), (16384, 197408)):
                    info = layout(2048, c, 256, True, 262144, sorted_union=True,
                                  mtp8=True, layout16=1, layout_hot_bits=15)
                    self.assertEqual(info['dynamic_ub_bytes'], size)
                    self.assertEqual(info['scratch_path'], 'UB')
                    self.assertEqual(info['workspace_elements'], 262144)
            # 冻结入口只包含原layout实现，不定义新因素。
            frozen = (Path(tmp) / 'variant_layout_stage_abcd_512.cpp').read_text()
            self.assertNotIn('JOINT_STAGE_EXEC', frozen)

    def test_fixed_winners_and_factor_table(self):
        r = self.mock()
        chosen, ranked = select(r)
        self.assertEqual(chosen, [*STAGE_EXEC_REFS, 'joint_stage_exec_b_512', 'joint_stage_exec_ac_512'])
        self.assertEqual(len(ranked), 15)
        self.assertIn('full_cache_75pct/7/16384', report_text(r))
        effects = factor_text(r)
        self.assertEqual(sum(line.startswith('| A |') for line in effects.splitlines()), 8)
        self.assertEqual(sum(line.startswith('| D |') for line in effects.splitlines()), 8)
        # 不允许丢失或重复工况后静默选择。
        for bad in ('status', 'missing', 'duplicate'):
            r = self.mock()
            if bad == 'status': r['status'] = 'failed'
            elif bad == 'missing': r['runs'].pop()
            else: r['runs'][-1] = r['runs'][-2]
            with self.assertRaises(ValueError): select(r)

    def test_single_prefix_equivalence(self):
        # 对所有小序列验证全局evict排名；跨64-lane边界另做随机长序列。
        rng = random.Random(2026)
        flags = [list(x) for n in range(9) for x in itertools.product((0, 1), repeat=n)]
        flags += [[rng.randrange(2) for _ in range(257)] for _ in range(50)]
        for hits in flags:
            hc = ec = 0
            for i, hit in enumerate(hits):
                self.assertEqual(i - hc, ec)
                hc += hit
                ec += 1 - hit
            self.assertEqual(ec, len(hits) - hc)

    def test_fused_invalidation_disjoint_writes(self):
        # 独立模型验证全清理→分配与“仅清未分配槽”完全相同；
        # 还显式证明新方案中任意线程交错下clear/write目标不相交。
        rng = random.Random(2026)
        for _ in range(2000):
            c = rng.randrange(1, 130)
            evicts = rng.randrange(c + 1)
            misses = rng.randrange(evicts + 1)
            part = rng.sample(range(c), c)
            invalid = [rng.randrange(2) for _ in range(c)]
            old = list(range(c)); new = old.copy()
            for i in range(c):
                if invalid[i]: old[part[i]] = -1
            for i in range(misses): old[part[i]] = 100000 + i
            for i in range(misses): new[part[i]] = 100000 + i
            for i in range(misses, c):
                if invalid[i]: new[part[i]] = -1
            self.assertEqual(old, new)
            self.assertFalse(set(part[:misses]) & {part[i] for i in range(misses, c) if invalid[i]})
            unused = evicts - misses
            indices = [misses+i if i < unused else i-unused if i < evicts else c-1-(i-evicts) for i in range(c)]
            self.assertEqual(sorted(indices), list(range(c)))

    def test_directory_bounds(self):
        import bisect
        rng = random.Random(2026)
        for max_token in (1, 512, 513, 65535, 262144, 262147, 16777216):
            values = sorted({rng.randrange(max_token) for _ in range(min(2048, max_token))})
            for bins in (512, 1024):
                shift = 0
                while bins << shift < max_token: shift += 1
                directory = [bisect.bisect_left(values, i << shift) for i in range(bins+1)]
                for token in (*values, 0, max_token-1):
                    b = token >> shift
                    self.assertLess(b, bins)
                    pos = bisect.bisect_left(values, token, directory[b], directory[b+1])
                    self.assertEqual(pos, bisect.bisect_left(values, token))

    def test_api_unsigned_unpack_and_masked_store(self):
        # 用实际CPU原语桩执行u16零扩展(含bit15)、masked连续写回。
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            p.joinpath('probe.cpp').write_text('''#include "sorted_cann_cpu.h"
int main(){using namespace AscendC;alignas(32) uint16_t f[64];alignas(32) uint32_t out[64];
for(int i=0;i<64;++i){f[i]=uint16_t(0x8000U+i);out[i]=12345;}
Reg::RegTensor<uint32_t> v;Reg::LoadAlign<uint16_t,Reg::LoadDist::DIST_UNPACK_B16>(v,f);
auto m=Reg::CreateMask<uint32_t>();for(int i=0;i<64;++i){assert(v.v[i]==f[i]);m.v[i]=i%2;}
Reg::StoreAlign<uint32_t,Reg::StoreDist::DIST_NORM_B32>(out,v,m);
for(int i=0;i<64;++i)assert(out[i]==(i%2?f[i]:12345));}''')
            subprocess.run(['g++', '-std=c++20', '-pthread', '-fsanitize=undefined',
                            '-I'+str(ROOT/'checks/simt_variants'), str(p/'probe.cpp'), '-o', str(p/'probe')], check=True)
            subprocess.run([str(p/'probe')], check=True)

    def test_describe_no_npu(self):
        r = subprocess.run([sys.executable, '-B', str(ROOT/'run_stage_exec_sweep.py'),
                            '--describe-only', '--output', '/tmp/unused-stage-exec'], capture_output=True, text=True, check=True)
        d = json.loads(r.stdout)
        self.assertEqual(len(d['implementations']), 18)
        self.assertEqual(d['screen_target_samples'], 720)

if __name__ == '__main__':
    unittest.main()
