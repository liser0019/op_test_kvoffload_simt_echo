#!/usr/bin/env python3
"""CPU-only orchestration regressions: independent selection, input identity and incomplete runs."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_hardware_sweep import select_candidates,validate_confirmation,read_complete,REFS

def report():
    r={'environment':{k:'same' for k in ('device','device_name','hostname','library_sha256','identity')},'status':'complete','runs':[]}
    for mtp in (0,1,3):
        stats={n:{'p50_us':v,'p95_us':v+1,'count':5,'samples_us':[v]*5,'correctness':{'status':'passed'}}
            for n,v in [(REFS[0],100),(REFS[1],95),('joint_hw_a',1 if mtp==0 else 80 if mtp==1 else 90),('joint_hw_b',85 if mtp==1 else 70)]}
        r['runs'].append({'comparison':'same_requests','workload':{'scenario':'full_cache_75pct','overlap':'overlap40','mtp':mtp,'logical_input_sha256':str(mtp)},'stats':stats})
    return r

class Checks(unittest.TestCase):
    def test_selection_ignores_mtp0_and_keeps_references(self):
        names,ranking=select_candidates(report(),1)
        self.assertEqual(names[:2],list(REFS));self.assertEqual(set(names[2:]),{'joint_hw_a','joint_hw_b'})
        self.assertEqual(len(ranking),2)
    def test_confirmation_identity(self):
        a=report();b=copy.deepcopy(a);validate_confirmation(a,b)
        b['runs'][0]['workload']['logical_input_sha256']='wrong'
        with self.assertRaises(RuntimeError):validate_confirmation(a,b)
        b=copy.deepcopy(a);b['environment']['library_sha256']='wrong'
        with self.assertRaises(RuntimeError):validate_confirmation(a,b)
        b=copy.deepcopy(a);b['runs'].append(b['runs'][0])
        with self.assertRaises(RuntimeError):validate_confirmation(a,b)
    def test_partial_or_wrong_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'joint_matrix_test';p.mkdir();f=p/'capture_metadata.json';r=report();f.write_text(json.dumps(r))
            with self.assertRaises(RuntimeError):read_complete(root)
            (p/'COMPLETE.json').write_text('{}');read_complete(root,expected_count=5)
            with self.assertRaises(RuntimeError):read_complete(root,expected_count=50)
            r['status']='running';f.write_text(json.dumps(r))
            with self.assertRaises(RuntimeError):read_complete(root)
    def test_old_and_new_script_entry_files_exist(self):
        for n in ('checks/check_merge_abcd_npu.py','profile_joint_mtp.py','run_hardware_sweep.py'):self.assertTrue((ROOT/n).is_file())
        self.assertNotIn('check_sorted_hardware_npu.py',(ROOT/'scripts/test_sorted_hardware.sh').read_text())
if __name__=='__main__':unittest.main()
