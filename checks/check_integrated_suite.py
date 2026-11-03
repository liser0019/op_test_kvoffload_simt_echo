#!/usr/bin/env python3
"""CPU checks of integration dispatch/layout/profiling orchestration, no torch/CANN/NPU."""
from pathlib import Path
import copy,json,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import joint_plan_common as common
import run_integrated_plan as runner
from checks.generate_joint_variants import generate
from checks.joint_build_info import expected_integrated_stage
from joint_plan_cases import layout
from joint_variants import INTEGRATED_COMPARE_NAMES as NAMES,description,supports_single_sorted
from profile_joint_mtp import read_kernels,execution_order,duration_ratios
from run_union_next_sweep import validate_cold_capture

class Checks(unittest.TestCase):
    def test_exact_stage_transform_and_private_duplicate_fallback(self):
        base=ROOT/'csrc/joint_variants'
        source=(base/'integrated_stage.h').read_text()
        self.assertEqual(source,expected_integrated_stage((base/'cold_ac_stage.h').read_text()))
        self.assertIn('rows==1U && i && a.topkIndices[qb+i]==a.topkIndices[qb+i-1U]',source)
        self.assertIn('if(t==1U)a.rowModes[row]=rows==1U?ROW_PRIVATE:',source)
    def test_minimal_build_and_host_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);generate(p,integrated_only=True)
            self.assertEqual({f.stem for f in p.glob('variant_*.cpp')},{'variant_'+n.split('_',1)[1] for n in NAMES[1:]})
            source=(p/'variant_integrated_512.cpp').read_text()
            for token in ('#define JOINT_OPT_MASK 72','#define JOINT_COLD_FINISH 1','#define JOINT_INTEGRATED 1'):self.assertIn(token,source)
        self.assertTrue(supports_single_sorted(NAMES[-1]));self.assertFalse(supports_single_sorted(NAMES[2]))
        for rows in (32,64,128):
            candidate=layout(2048,4096,rows,True,262144,False,True,True,False)
            self.assertEqual(candidate['scratch_elements'],24640) # ctx64 + 4T map + C + 3C work
            self.assertEqual(candidate['dynamic_ub_bytes'],98848)
        self.assertEqual(description(NAMES[-1])['control_words'],64)
    def test_cli_does_not_expand_matrix(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'run_integrated_plan.py'),'--describe-only'],capture_output=True,text=True,check=True)
        info=json.loads(r.stdout)
        self.assertEqual(info['implementations'],list(NAMES));self.assertEqual(info['repeat'],50)
        self.assertEqual(info['warmup'],10);self.assertEqual(info['regression_repeat'],5)
        self.assertEqual(info['build_mode'],'--integrated-only');self.assertEqual(info['mtp'],[0,1,3])
        self.assertIn('no screening',info['selection'])
    def test_csv_exact_order_counts_units_and_parent_ratios(self):
        order=[n for i in range(5) for n in execution_order(NAMES,i)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'kernel_details.csv'
            p.write_text('Name,Duration(us),Start Time(us)\nTensorMove,1,0\n'+'\n'.join(f'{common.KERNELS[n]},1,{i*10+10}' for i,n in enumerate(order))+'\n')
            self.assertEqual(len(read_kernels([p],NAMES,5,order)),20)
            with self.assertRaises(RuntimeError):read_kernels([p],NAMES,6,order)
            with self.assertRaises(RuntimeError):read_kernels([p],NAMES,5,list(reversed(order)))
            p.write_text(p.read_text().replace('Duration(us)','Duration(ms)'))
            with self.assertRaises(RuntimeError):read_kernels([p],NAMES,5,order)
        ratios=duration_ratios({n:{'p50_us':10} for n in NAMES})
        for parent in NAMES[:3]:self.assertEqual(ratios[f'{NAMES[-1]}_over_{parent}_p50'],1)
    def test_report_keeps_slow_results_and_requires_full_matrix(self):
        r={'implementations':list(NAMES),'runs':[]}
        for scene in ('cold_reset','full_cache_75pct'):
            for m in (0,1,3):
                r['runs'].append({'comparison':'same_requests','workload':{'scenario':scene,'overlap':'overlap40','mtp':m,'requests':32,'query_rows':32*(m+1),'q_len':m+1,'topk':2048,'capacity':4096,'max_token':262144},
                    'stats':{n:{'p50_us':100 if n==NAMES[-1] else 10,'p95_us':101,'count':50,'correctness':{'status':'passed'}} for n in NAMES}})
        validate_cold_capture(r,NAMES,['cold_reset','full_cache_75pct'])
        text=runner.summary_text(Path('/workspace/example/capture_metadata.json'),r)
        self.assertIn('10.000',text)
        for name in NAMES:self.assertIn(name,text)
        bad=copy.deepcopy(r);bad['runs'].pop()
        with self.assertRaises(RuntimeError):validate_cold_capture(bad,NAMES,['cold_reset','full_cache_75pct'])
    def test_gate_failure_stops_before_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'results'
            argv=['run_integrated_plan.py','--skip-build','--output',str(output),'--temp-dir',str(Path(tmp)/'tmp'),'--min-output-free-bytes','0','--min-temp-free-bytes','0','--min-free-inodes','0']
            seen=[]
            def fail(command,**kwargs):
                seen.append(command);kwargs['stdout'].write('RuntimeError: synthetic gate failed\n');return subprocess.CompletedProcess(command,7)
            with patch.dict(os.environ),patch.object(tempfile,'tempdir',None),patch.object(sys,'argv',argv),patch.object(runner.subprocess,'run',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'gate failed rc=7'):runner.main()
            self.assertEqual(len(seen),1);self.assertIn('checks/check_cold_finish_npu.py',seen[0])
            self.assertFalse((output/'COMPLETE.json').exists());self.assertFalse((output/'report.md').exists())
            self.assertEqual(json.loads((output/'RUN_STATUS.json').read_text())['status'],'failed')
            self.assertTrue((output/'gate.log').is_file())
if __name__=='__main__':unittest.main()
