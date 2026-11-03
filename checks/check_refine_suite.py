#!/usr/bin/env python3
"""CPU-only checks of refinement catalogue, CSV pairing, selection and failure handling."""
from pathlib import Path
import copy,json,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import joint_plan_common as common
import run_refine_sweep as runner
from joint_variants import REFINE_SWEEP,REFINE_NAMES,REFINE_REFS,INTEGRATED_COMPARE_NAMES,description,algorithm_mask
from checks.generate_joint_variants import generate
from profile_joint_mtp import read_kernels,execution_order,duration_ratios
from run_union_next_sweep import validate_cold_capture

class Checks(unittest.TestCase):
    def report(self,names=None):
        names=names or list(runner.DEFAULT_NAMES)
        return {'implementations':names,'runs':[{'comparison':'same_requests','workload':{'scenario':s,'overlap':'overlap40','mtp':m,'requests':32,'query_rows':32*(m+1),'q_len':m+1,'topk':2048,'capacity':4096,'max_token':262144},'stats':{n:{'p50_us':10+i,'p95_us':20+i,'count':50,'correctness':{'status':'passed'}} for i,n in enumerate(names)}} for s in ('cold_reset','full_cache_75pct') for m in (0,1,3)]}
    def test_catalog_and_frozen_integrated_suite(self):
        self.assertEqual(list(REFINE_SWEEP.values()),list(range(127,143)))
        self.assertEqual(len(INTEGRATED_COMPARE_NAMES),5)
        for bits,name in enumerate(REFINE_NAMES):
            d=description(name);self.assertEqual(d['refine_bits'],bits);self.assertEqual(d['threads'],512)
            self.assertEqual(d['private_fallback_threads'],1024);self.assertEqual(algorithm_mask(127+bits),72)
    def test_minimum_build_has_distinct_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);generate(p,refine_only=True)
            files=list(p.glob('variant_*.cpp'));self.assertEqual(len(files),17)
            for bits,name in enumerate(REFINE_SWEEP):
                source=(p/('variant_'+name+'.cpp')).read_text()
                for token in ('#define JOINT_REFINE 1',f'#define JOINT_REFINE_BITS {bits}','#define JOINT_OPT_MASK 72'):self.assertIn(token,source)
                self.assertNotIn('#define JOINT_INTEGRATED',source)
    def test_cli_defaults_and_all_formal_option(self):
        def describe(*args):
            r=subprocess.run([sys.executable,'-B',str(ROOT/'run_refine_sweep.py'),'--describe-only',*args],capture_output=True,text=True,check=True);return json.loads(r.stdout)
        d=describe();self.assertEqual(d['count'],18);self.assertEqual(d['screen_repeat'],5);self.assertEqual(d['confirm_repeat'],50)
        self.assertEqual(d['confirm_warmup'],10);self.assertEqual(d['mtp'],[0,1,3]);self.assertEqual(d['regression_repeat'],5)
        self.assertIsNone(describe('--full-confirm')['screen_repeat'])
    def test_ranking_retains_control_and_union_of_workload_winners(self):
        names=list(runner.DEFAULT_NAMES);r=self.report(names);a=REFINE_NAMES[1];b=REFINE_NAMES[2]
        r['runs'][0]['stats'][a]['p50_us']=1;r['runs'][1]['stats'][b]['p50_us']=1
        selected,ranking=runner.select_candidates(r,names,1)
        self.assertTrue(set(REFINE_REFS)<=set(selected));self.assertIn(a,selected);self.assertIn(b,selected)
        self.assertEqual(len(ranking),6);self.assertNotIn(REFINE_REFS[-1],ranking[0]['ranking'])
        with self.assertRaises(ValueError):runner.select_candidates(r,names,0)
    def test_csv_target_names_count_units_order(self):
        names=runner.DEFAULT_NAMES;order=[n for i in range(5) for n in execution_order(names,i)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'kernel_details.csv';p.write_text('Name,Duration(us),Start Time(us)\nTensorMove,3,0\n'+'\n'.join(f'{common.KERNELS[n]},1,{i*10+10}' for i,n in enumerate(order))+'\n')
            self.assertEqual(len(read_kernels([p],names,5,order)),90)
            with self.assertRaises(RuntimeError):read_kernels([p],names,6,order)
            with self.assertRaises(RuntimeError):read_kernels([p],names,5,list(reversed(order)))
            p.write_text(p.read_text().replace('Duration(us)','Duration(ms)'))
            with self.assertRaises(RuntimeError):read_kernels([p],names,5,order)
    def test_ratio_direction_and_slow_results_retained(self):
        r=self.report();name=REFINE_NAMES[-1]
        r['runs'][0]['stats'][name]['p50_us']=100
        text=runner.summary_text(Path('/workspace/confirm/capture_metadata.json'),r,Path('/workspace/screen/capture_metadata.json'))
        self.assertIn('100.000',text);self.assertIn('data_index.json',text)
        for n in runner.DEFAULT_NAMES:self.assertIn(n,text)
        ratios=duration_ratios({REFINE_REFS[1]:{'p50_us':20},REFINE_REFS[2]:{'p50_us':40},name:{'p50_us':10}})
        self.assertEqual(ratios[f'{name}_over_{REFINE_REFS[1]}_p50'],.5)
        self.assertEqual(ratios[f'{name}_over_{REFINE_REFS[2]}_p50'],.25)
    def test_report_requires_complete_matrix(self):
        r=self.report();validate_cold_capture(r,runner.DEFAULT_NAMES,['cold_reset','full_cache_75pct'])
        r['runs'].pop()
        with self.assertRaises(RuntimeError):validate_cold_capture(r,runner.DEFAULT_NAMES,['cold_reset','full_cache_75pct'])
    def test_gate_failure_keeps_log_and_stops_before_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'results';argv=['run_refine_sweep.py','--skip-build','--output',str(output),'--temp-dir',str(Path(tmp)/'tmp'),'--min-output-free-bytes','0','--min-temp-free-bytes','0','--min-free-inodes','0'];seen=[]
            def fail(command,**kwargs):
                seen.append(command);kwargs['stdout'].write('RuntimeError: synthetic refine gate failed\n');return subprocess.CompletedProcess(command,7)
            with patch.dict(os.environ),patch.object(tempfile,'tempdir',None),patch.object(sys,'argv',argv),patch.object(runner.subprocess,'run',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'gate failed rc=7'):runner.main()
            self.assertEqual(len(seen),1);self.assertIn('checks/check_refine_npu.py',seen[0])
            self.assertFalse((output/'COMPLETE.json').exists());self.assertFalse((output/'report.md').exists())
            self.assertEqual(json.loads((output/'RUN_STATUS.json').read_text())['status'],'failed')
            self.assertIn('synthetic refine gate failed',(output/'gate.log').read_text())
if __name__=='__main__':unittest.main()
