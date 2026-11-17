"""Synthetic CSV contract tests; these numbers are not performance results."""
import csv
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
    def test_timestamp_quantization_near_contiguous_split(self):
        base=1.0e15
        rows=[[TOKENS['split'],base,533.078,'AI_CORE',1,0],
              [TOKENS['plan'],base+533.078-0.25,14.0,'AI_VECTOR_CORE',1,0],
              [TOKENS['a'],base+600.0,10.0,'AI_CORE',1,0],
              [TOKENS['b'],base+620.0,10.0,'AI_CORE',1,0],
              [TOKENS['c'],base+640.0,10.0,'AI_CORE',1,0]]
        parsed=self.parse(rows,('split','a','b','c'))
        self.assertEqual(parsed[0]['between_kernel_gap_us'],0.0)
    def test_missing_component(self):
        with self.assertRaises(RuntimeError):self.parse(self.rows()[:-1],repeat=2,orders=('split','a','split','a'))
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
