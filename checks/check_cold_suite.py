#!/usr/bin/env python3
"""CPU orchestration and cold-source regressions; NOT CANN/NPU."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint_variants import COLD_COMPARE_NAMES,VARIANTS,description
from checks.check_cold_finish import check_copies
from checks.generate_joint_variants import generate
from run_union_next_sweep import build_mode_for,cold_summary_text,validate_cold_capture
from checks.check_union_next_runner import report
from profile_joint_mtp import read_kernels,execution_order,duration_ratios
import joint_plan_common as common

class Checks(unittest.TestCase):
    def test_only_reset_hook_changes_stages(self):check_copies()
    def test_six_way_minimal_build_wrappers(self):
        self.assertEqual(len(COLD_COMPARE_NAMES),6)
        self.assertEqual(build_mode_for(COLD_COMPARE_NAMES),'--cold-compare-only')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);generate(p,cold_compare_only=True)
            self.assertEqual(len(list(p.glob('variant_*.cpp'))),5)
            for stem in ('union_cold_512','hw_ac_cold_512'):
                self.assertIn('#define JOINT_COLD_FINISH 1',(p/('variant_'+stem+'.cpp')).read_text())
            for stem in ('union_base_512','hw_ac_512','hw_ac_256'):
                self.assertNotIn('JOINT_COLD_FINISH',(p/('variant_'+stem+'.cpp')).read_text())
    def test_cli_defaults_no_screen(self):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'run_union_next_sweep.py'),'--cold-compare','--describe-only'],capture_output=True,text=True,check=True)
        x=json.loads(result.stdout);self.assertEqual(x['implementations'],list(COLD_COMPARE_NAMES))
        self.assertEqual(x['build_mode'],'--cold-compare-only');self.assertIn('no screening',x['selection'])
        self.assertEqual(x['confirm_repeat'],50);self.assertEqual(x['scenarios'],['cold_reset','full_cache_75pct'])
        rejected=subprocess.run([sys.executable,'-B',str(ROOT/'run_union_next_sweep.py'),'--cold-compare','--describe-only','--implementations','abcdf_pages','joint_hw_ac_512','joint_hw_ac_256'],capture_output=True,text=True)
        self.assertNotEqual(rejected.returncode,0)
    def test_exact_csv_names_and_parent_ratios(self):
        names=COLD_COMPARE_NAMES;order=[n for step in range(5) for n in execution_order(names,step)]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'kernel_details.csv';rows=['Name,Duration(us),Start Time(us)','TensorMove,1,0']
            rows += [f'{common.KERNELS[n]},1,{10+i*10}' for i,n in enumerate(order)]
            p.write_text('\n'.join(rows)+'\n');self.assertEqual(len(read_kernels([p],names,5,order)),30)
            with self.assertRaises(RuntimeError):read_kernels([p],names,5,list(reversed(order)))
        ratios=duration_ratios({n:{'p50_us':10} for n in names})
        self.assertEqual(ratios['joint_union_cold_512_over_joint_union_base_512_p50'],1)
        self.assertEqual(ratios['joint_hw_ac_cold_512_over_joint_hw_ac_512_p50'],1)
    def test_incomplete_matrix_cannot_succeed(self):
        r={'implementations':list(COLD_COMPARE_NAMES),'runs':[]}
        for scene in ('cold_reset','full_cache_75pct'):
            for m in (0,1,3):
                r['runs'].append({'comparison':'same_requests','workload':{'scenario':scene,'overlap':'overlap40','mtp':m,'requests':32,'query_rows':32*(m+1),'q_len':m+1,'topk':2048,'capacity':4096,'max_token':262144},'stats':{n:{} for n in COLD_COMPARE_NAMES}})
        validate_cold_capture(r,COLD_COMPARE_NAMES,['cold_reset','full_cache_75pct'])
        for kind in ('missing','duplicate','dimension'):
            bad=copy.deepcopy(r)
            if kind=='missing':bad['runs'].pop()
            elif kind=='duplicate':bad['runs'].append(bad['runs'][0])
            else:bad['runs'][0]['workload']['query_rows']=1
            with self.assertRaises(RuntimeError):validate_cold_capture(bad,COLD_COMPARE_NAMES,['cold_reset','full_cache_75pct'])
    def test_report_keeps_slow_candidates(self):
        r=report();r['runs']=r['runs'][:1]
        r['runs'][0]['stats']={n:{'p50_us':100 if n.endswith('cold_512') else 10,'p95_us':101,'count':50,'correctness':{'status':'passed'}} for n in COLD_COMPARE_NAMES}
        text=cold_summary_text(Path('/workspace/example/capture_metadata.json'),r,COLD_COMPARE_NAMES)
        for n in COLD_COMPARE_NAMES:self.assertIn(n,text)
        self.assertIn('10.000',text)
if __name__=='__main__':unittest.main()
