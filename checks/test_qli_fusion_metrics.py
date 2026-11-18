"""Synthetic CSV contract tests; these numbers are not performance results."""
import csv
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from fusion.metrics import TOKENS,read_csvs

class Metrics(unittest.TestCase):
    def parse(self,rows,orders=('split','a'),repeat=1,header=None):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'kernel_details.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(header or ['Name','Start Time(us)','Duration(us)','Task Type','Stream ID','Device ID']);w.writerows(rows)
            return read_csvs([p],list(orders),repeat)[0]
    def rows(self):
        return [[TOKENS['split']+'_mix_aic',0,10,'AI_CORE',1,0],
                [TOKENS['split']+'_mix_aiv',1,11,'AI_VECTOR_CORE',1,0],
                ['TensorMove',12,3,'AI_CORE',1,0],
                [TOKENS['plan'],15,4,'AI_VECTOR_CORE',1,0],
                [TOKENS['a']+'_mix_aic',20,12,'AI_CORE',1,0],
                [TOKENS['a']+'_mix_aiv',21,14,'AI_VECTOR_CORE',1,0]]
    def test_overlap_and_gap(self):
        p=self.parse(self.rows());self.assertEqual(p[0]['span_us'],19);self.assertEqual(p[0]['busy_us'],16)
        self.assertEqual(p[0]['between_kernel_gap_us'],3);self.assertEqual(p[1]['busy_us'],15)
    def test_native_indexer_does_not_capture_split_or_fused_kernel_names(self):
        rows = self.rows() + [
            ['quant_lightning_indexer_mix_aic', 40, 8, 'AI_CORE', 1, 0],
            ['QuantLightningIndexer_mix_aiv', 41, 9, 'AI_VECTOR_CORE', 1, 0],
        ]
        split, fused, native = self.parse(rows, ('split', 'a', 'baseline'))
        self.assertEqual(split['plan_us'], 4)
        self.assertEqual(fused['implementation'], 'a')
        self.assertEqual(native['implementation'], 'baseline')
        self.assertEqual(native['span_us'], 10)
        self.assertEqual(native['busy_us'], 10)
        self.assertNotIn('plan_us', native)
    def test_four_row_indexer_stays_separate_from_split_plan(self):
        rows = self.rows()[:4] + [
            [TOKENS['baseline'], 20, 6, 'AI_CORE', 1, 0],
            [TOKENS['ordered4']+'_mix_aic', 30, 7, 'AI_CORE', 1, 0],
            ['QuantLightningIndexerPlanSplit4_mix_aiv', 31, 8, 'AI_VECTOR_CORE', 1, 0],
        ]
        split, native, ordered4 = self.parse(rows, ('split', 'baseline', 'ordered4'))
        self.assertEqual(split['indexer_span_us'], 12)
        self.assertEqual(split['plan_us'], 4)
        self.assertEqual(native['span_us'], 6)
        self.assertEqual(ordered4['span_us'], 9)
        self.assertNotIn('plan_us', ordered4)
    def test_native_and_original_fused_without_split(self):
        native = [TOKENS['baseline'] + '_mix_aic', 0, 8, 'AI_CORE', 1, 0]
        fused = [
            [TOKENS['peer'] + '_mix_aic', 10, 9, 'AI_CORE', 1, 0],
            ['QuantLightningIndexerOrderedFused_mix_aiv', 11, 12, 'AI_VECTOR_CORE', 1, 0],
        ]
        rows = [native, *fused, ['TensorMove', 24, 1, 'AI_CORE', 1, 0]]
        rows += [[n, s + 20, d, t, st, dev] for n, s, d, t, st, dev in fused]
        rows.append([native[0], 44, *native[2:]])
        order = ('baseline', 'peer', 'peer', 'baseline')
        parsed = self.parse(rows, order, repeat=2)
        self.assertEqual(tuple(p['implementation'] for p in parsed), order)
        self.assertEqual([p['span_us'] for p in parsed], [8, 13, 13, 8])
        self.assertEqual([p['busy_us'] for p in parsed], [8, 13, 13, 8])
        self.assertTrue(all('plan_us' not in p for p in parsed))
    def test_matched_topk_roles_preserve_device_and_launch_geometry(self):
        header = ['Name', 'Start Time(us)', 'Duration(us)', 'Device_id', 'Block Num', 'Mix Block Num']
        rows = [
            ['QuantLightningIndexerPlanTopkControl4', 0, 8, 4, 32, 64],
            ['QuantLightningIndexerPlanSplit4', 10, 9, 4, 32, 64],
        ]
        a, b = self.parse(rows, ('topk_control4', 'ordered4'), header=header)
        self.assertEqual((a['span_us'], b['span_us']), (8, 9))
        for sample in (a, b):
            entry = sample['components'][0]
            self.assertEqual((entry['device'], entry['block_num'], entry['mix_block_num']), ('4', '32', '64'))
        rows[1][3] = 3
        with self.assertRaisesRegex(RuntimeError, 'mixed target device'):
            self.parse(rows, ('topk_control4', 'ordered4'), header=header)
    def test_timestamp_quantization_near_contiguous_split(self):
        base=1.0e15
        rows=[[TOKENS['split'],base,533.078,'AI_CORE',1,0],
              [TOKENS['plan'],base+533.078-0.25,14.0,'AI_VECTOR_CORE',1,0],
              [TOKENS['a'],base+600.0,10.0,'AI_CORE',1,0],
              [TOKENS['b'],base+620.0,10.0,'AI_CORE',1,0],
              [TOKENS['c'],base+640.0,10.0,'AI_CORE',1,0]]
        parsed=self.parse(rows,('split','a','b','c'))
        self.assertEqual(parsed[0]['between_kernel_gap_us'],0.0)
        self.assertGreater(parsed[0]['gap_clamped_us'],0.0)
        self.assertLessEqual(parsed[0]['gap_clamped_us'],parsed[0]['gap_tolerance_us'])
    def test_large_absolute_timestamp_preserves_csv_duration(self):
        rows=[[TOKENS['a'],'1000000000000000.001','533.078','AI_CORE',1,0],
              [TOKENS['b'],'1000000000000600.001','10.007','AI_CORE',1,0]]
        parsed=self.parse(rows,('a','b'))
        self.assertEqual(parsed[0]['busy_us'],533.078)
        self.assertEqual(parsed[0]['span_us'],533.078)
        self.assertEqual(parsed[1]['busy_us'],10.007)
        self.assertEqual(parsed[0]['timestamp_origin_us'],'1000000000000000.001')
        self.assertEqual(parsed[0]['components'][0]['absolute_start_us'],rows[0][1])
        json.dumps(parsed) # no Decimal objects leak into capture_metadata.json
    def test_large_precise_split_gap(self):
        rows=[[TOKENS['split'],'1000000000000000','533.078','AI_CORE',1,0],
              [TOKENS['plan'],'1000000000000533.125','14.005','AI_VECTOR_CORE',1,0]]
        p=self.parse(rows,('split',))[0]
        self.assertEqual(p['span_us'],547.130)
        self.assertEqual(p['busy_us'],547.083)
        self.assertEqual(p['between_kernel_gap_us'],0.047)
        self.assertEqual(p['plan_us'],14.005)
    def test_translation_invariance_with_mixed_components(self):
        rows=self.rows();original=self.parse(rows)
        shifted=[[n,str(Decimal('1000000000000000.001')+Decimal(str(s))),d,t,st,dev]
                 for n,s,d,t,st,dev in rows]
        moved=self.parse(shifted)
        for before,after in zip(original,moved):
            for key in ('span_us','busy_us','indexer_span_us','plan_us','between_kernel_gap_us'):
                if key in before:self.assertEqual(before[key],after[key],key)
    def test_large_timestamp_real_overlap_still_rejected(self):
        rows=[[TOKENS['split'],'1000000000000000','533.078','AI_CORE',1,0],
              [TOKENS['plan'],'1000000000000532','14.005','AI_VECTOR_CORE',1,0]]
        with self.assertRaisesRegex(RuntimeError,'before ordered Indexer'):
            self.parse(rows,('split',))
    def test_nonfinite_device_time_rejected(self):
        for bad in ('NaN','Infinity','-Infinity','garbled'):
            with self.subTest(bad=bad),self.assertRaisesRegex(RuntimeError,'invalid device time'):
                self.parse([[TOKENS['a'],'1000000000000000',bad,'AI_CORE',1,0]],('a',))
    def test_missing_component(self):
        with self.assertRaises(RuntimeError):self.parse(self.rows()[:-1],repeat=2,orders=('split','a','split','a'))
    def test_split_candidates_share_plan_kernel_but_keep_separate_samples(self):
        orders = ('final4', 'split4', 'split4', 'final4')
        rows = []
        for index, role in enumerate(orders):
            token = TOKENS['ordered4' if role == 'split4' else role]
            start = index * 30
            rows.extend([
                [token+'_mix_aic', start, 7, 'AI_CORE', 1, 0],
                [token+'_mix_aiv', start+1, 9, 'AI_VECTOR_CORE', 1, 0],
                [TOKENS['plan'], start+12, 3+index, 'AI_VECTOR_CORE', 1, 0],
            ])
        result = self.parse(rows, orders, repeat=2)
        self.assertEqual([sample['implementation'] for sample in result], list(orders))
        self.assertEqual([sample['plan_us'] for sample in result], [3, 4, 5, 6])
        self.assertTrue(all(sample['indexer_span_us'] == 10 for sample in result))
        self.assertTrue(all(sample['between_kernel_gap_us'] == 2 for sample in result))
        with self.assertRaisesRegex(RuntimeError, 'split Indexer changed before Plan'):
            self.parse(rows[:2]+rows[3:], orders, repeat=2)
    def test_peer_aligned4_names_do_not_capture_original_peer(self):
        rows = [
            [TOKENS['peer'], 0, 10, 'AI_CORE', 1, 0],
            [TOKENS['peer4'], 20, 11, 'AI_CORE', 1, 0],
        ]
        parsed = self.parse(rows, ('peer', 'peer4'))
        self.assertEqual([sample['implementation'] for sample in parsed], ['peer', 'peer4'])
        self.assertEqual([sample['span_us'] for sample in parsed], [10, 11])

    def test_factorial_identities_do_not_capture_c0_or_each_other(self):
        rows = [
            [TOKENS['c0'] + '_mix_aic', 0, 8, 'AI_CORE', 1, 0],
            [TOKENS['A'] + '_mix_aic', 10, 9, 'AI_CORE', 1, 0],
            [TOKENS['AB'] + '_mix_aic', 20, 10, 'AI_CORE', 1, 0],
            [TOKENS['ABC'] + '_mix_aic', 30, 11, 'AI_CORE', 1, 0],
            ['QuantLightningIndexerPlanCFB_mix_aiv', 50, 12, 'AI_VECTOR_CORE', 1, 0],
        ]
        parsed = self.parse(rows, ('c0', 'A', 'AB', 'ABC', 'B'))
        self.assertEqual([sample['implementation'] for sample in parsed], ['c0', 'A', 'AB', 'ABC', 'B'])
        self.assertEqual([sample['span_us'] for sample in parsed], [8, 9, 10, 11, 12])

    def test_wrong_order(self):
        with self.assertRaises(RuntimeError):self.parse(self.rows(),orders=('a','split'))
    def test_wrong_units(self):
        with self.assertRaises(RuntimeError):self.parse(self.rows(),header=['Name','Start Time(ns)','Duration(ns)','Task Type','Stream ID','Device ID'])
    def test_no_plan(self):
        with self.assertRaises(RuntimeError):self.parse([r for r in self.rows() if r[0]!=TOKENS['plan']])
    def test_two_repeats(self):
        rows=self.rows();rows+=[[n,s+40,d,t,st,dev] for n,s,d,t,st,dev in self.rows()]
        self.assertEqual(len(self.parse(rows,('split','a','split','a'),2)),4)

if __name__=='__main__':unittest.main()
