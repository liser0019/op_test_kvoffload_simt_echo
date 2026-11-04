#!/usr/bin/env python3
"""CPU-only checks of refinement catalogue, CSV pairing, selection and failure handling."""
from pathlib import Path
import copy,json,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import joint_plan_common as common
import run_hot_sweep as runner
from joint_variants import HOT_SWEEP,HOT_NAMES,HOT_REFS,INTEGRATED_COMPARE_NAMES,description,algorithm_mask
from checks.generate_joint_variants import generate
from profile_joint_mtp import read_kernels,execution_order,duration_ratios
from run_union_next_sweep import validate_cold_capture

class Checks(unittest.TestCase):
    def report(self,names=None):
        names=names or list(runner.DEFAULT_NAMES)
        return {'implementations':names,'runs':[{'comparison':'same_requests','workload':{'scenario':s,'overlap':'overlap40','mtp':m,'requests':32,'query_rows':32*(m+1),'q_len':m+1,'topk':2048,'capacity':4096,'max_token':262144},'stats':{n:{'p50_us':10+i,'p95_us':20+i,'count':50,'correctness':{'status':'passed'}} for i,n in enumerate(names)}} for s in ('cold_reset','full_cache_75pct') for m in (0,1,3)]}
    def test_catalog_and_frozen_integrated_suite(self):
        self.assertEqual(list(HOT_SWEEP.values()),list(range(159,193)))
        self.assertEqual(len(INTEGRATED_COMPARE_NAMES),5)
        for bits,name in enumerate(HOT_NAMES):
            d=description(name);self.assertEqual(d['hot_config']['bits'],bits if bits<32 else 0);self.assertEqual(d['threads'],512)
            self.assertEqual(d['private_fallback_threads'],1024);self.assertEqual(algorithm_mask(159+bits),72)
    def test_minimum_build_has_distinct_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);generate(p,hot_only=True)
            files=list(p.glob('variant_*.cpp'));self.assertEqual(len(files),37)
            for bits,name in enumerate(HOT_SWEEP):
                source=(p/('variant_'+name+'.cpp')).read_text()
                for token in ('#define JOINT_REFINE 1','#define JOINT_REFINE_BITS 11',f'#define JOINT_HOT_BITS {bits if bits<32 else 0}','#define JOINT_HOT 1','#define JOINT_OPT_MASK 72'):self.assertIn(token,source)
                self.assertNotIn('#define JOINT_INTEGRATED',source)
    def test_packet_growth_and_feature_contracts(self):
        from joint_plan_cases import layout
        from joint_variants import hot_config
        base=layout(2048,4096,128,True,262144,False,True,True)
        large=layout(2048,4096,128,True,262144,False,True,True,hot_packet=True)
        self.assertEqual(base['scratch_elements'],24640);self.assertEqual(large['scratch_elements'],28672)
        self.assertEqual(large['dynamic_ub_bytes'],114976)
        self.assertEqual(layout(2048,4096,32,False,262144),layout(2048,4096,32,False,262144,hot_packet=True))
        for shape in ((65,128),(2048,32768)):
            kw=dict(topk=shape[0],capacity=shape[1],rows=128,joint=True,max_token=262144,turbo=True,sorted_union=True)
            self.assertEqual(layout(**kw),layout(**kw,hot_packet=True))
        self.assertTrue(hot_config('joint_hot_part_512')['partition']);self.assertFalse(hot_config('joint_hot_part_512')['miss'])
        self.assertFalse(hot_config('joint_hot_miss_512')['partition']);self.assertTrue(hot_config('joint_hot_miss_512')['miss'])
    def test_cli_defaults_and_all_formal_option(self):
        def describe(*args):
            r=subprocess.run([sys.executable,'-B',str(ROOT/'run_hot_sweep.py'),'--describe-only',*args],capture_output=True,text=True,check=True);return json.loads(r.stdout)
        d=describe();self.assertEqual(d['count'],38);self.assertEqual(d['screen_repeat'],5);self.assertEqual(d['confirm_repeat'],50)
        self.assertEqual(d['confirm_warmup'],10);self.assertEqual(d['mtp'],[0,1,3]);self.assertEqual(d['regression_repeat'],5)
        self.assertIsNone(describe('--full-confirm')['screen_repeat'])
    def test_ranking_retains_control_and_union_of_workload_winners(self):
        names=list(runner.DEFAULT_NAMES);r=self.report(names);a=HOT_NAMES[1];b=HOT_NAMES[2]
        r['runs'][0]['stats'][a]['p50_us']=1;r['runs'][1]['stats'][b]['p50_us']=1
        selected,ranking=runner.select_candidates(r,names,1)
        self.assertTrue(set(HOT_REFS)<=set(selected));self.assertIn(a,selected);self.assertIn(b,selected)
        self.assertEqual(len(ranking),6);self.assertNotIn(HOT_REFS[-1],ranking[0]['ranking'])
        with self.assertRaises(ValueError):runner.select_candidates(r,names,0)
    def test_csv_target_names_count_units_order(self):
        names=runner.DEFAULT_NAMES;order=[n for i in range(5) for n in execution_order(names,i)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'kernel_details.csv';p.write_text('Name,Duration(us),Start Time(us)\nTensorMove,3,0\n'+'\n'.join(f'{common.KERNELS[n]},1,{i*10+10}' for i,n in enumerate(order))+'\n')
            self.assertEqual(len(read_kernels([p],names,5,order)),190)
            with self.assertRaises(RuntimeError):read_kernels([p],names,6,order)
            with self.assertRaises(RuntimeError):read_kernels([p],names,5,list(reversed(order)))
            p.write_text(p.read_text().replace('Duration(us)','Duration(ms)'))
            with self.assertRaises(RuntimeError):read_kernels([p],names,5,order)
    def test_ratio_direction_and_slow_results_retained(self):
        r=self.report();name=HOT_NAMES[-1]
        r['runs'][0]['stats'][name]['p50_us']=100
        text=runner.summary_text(Path('/workspace/confirm/capture_metadata.json'),r,Path('/workspace/screen/capture_metadata.json'))
        self.assertIn('100.000',text);self.assertIn('data_index.json',text)
        for n in runner.DEFAULT_NAMES:self.assertIn(n,text)
        ratios=duration_ratios({HOT_REFS[2]:{'p50_us':20},HOT_REFS[4]:{'p50_us':40},name:{'p50_us':10}})
        self.assertEqual(ratios[f'{name}_over_{HOT_REFS[2]}_p50'],.5)
        self.assertEqual(ratios[f'{name}_over_{HOT_REFS[4]}_p50'],.25)
    def test_report_requires_complete_matrix(self):
        r=self.report();validate_cold_capture(r,runner.DEFAULT_NAMES,['cold_reset','full_cache_75pct'])
        r['runs'].pop()
        with self.assertRaises(RuntimeError):validate_cold_capture(r,runner.DEFAULT_NAMES,['cold_reset','full_cache_75pct'])
    def test_gate_failure_keeps_log_and_stops_before_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'results';argv=['run_hot_sweep.py','--skip-build','--output',str(output),'--temp-dir',str(Path(tmp)/'tmp'),'--min-output-free-bytes','0','--min-temp-free-bytes','0','--min-free-inodes','0'];seen=[]
            def fail(command,**kwargs):
                seen.append(command);kwargs['stdout'].write('RuntimeError: synthetic execution gate failed\n');return subprocess.CompletedProcess(command,7)
            with patch.dict(os.environ),patch.object(tempfile,'tempdir',None),patch.object(sys,'argv',argv),patch.object(runner.subprocess,'run',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'gate failed rc=7'):runner.main()
            self.assertEqual(len(seen),1);self.assertIn('checks/check_hot_npu.py',seen[0])
            self.assertFalse((output/'COMPLETE.json').exists());self.assertFalse((output/'report.md').exists())
            self.assertEqual(json.loads((output/'RUN_STATUS.json').read_text())['status'],'failed')
            self.assertIn('synthetic execution gate failed',(output/'gate.log').read_text())
if __name__=='__main__':unittest.main()
