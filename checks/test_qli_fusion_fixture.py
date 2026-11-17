"""CPU checks of fixture layout and cache initialization, not Indexer accuracy."""
import unittest
import numpy as np
import torch
from fusion.glm52_inputs import Shape,make_inputs
from profile_qli_fusion import make_plan_case,operator_kwargs,check_ordered_topk

class Fixture(unittest.TestCase):
    def test_ordered_topk_valid_prefix_contract(self):
        check_ordered_topk(np.array([[2,5,9,11],[0,1,2,-1],[-1,-1,-1,-1]]),[20,3,0])
        check_ordered_topk(np.array([[2,5,9,11]]),[20])
    def test_ordered_topk_rejects_broken_prefix_or_visibility(self):
        for row,visible in [([2,9,5,11],20),([2,-1,5,-1],3),
                            ([0,1,2,-2],3),([0,1,3,-1],3),([0,1,1,-1],3)]:
            with self.subTest(row=row),self.assertRaises(RuntimeError):
                check_ordered_topk(np.array([row]),[visible])
    def test_actual_checked_metadata_is_passed(self):
        x=make_inputs(Shape(batch=1,width=2,seq_len=128,topk=8,capacity=32),seed=2026)
        actual={k:v.clone() for k,v in x.plan_metadata.items() if k!='host_block_table'}
        actual['active_rows']=torch.tensor([2],dtype=torch.int32)
        b={'inputs':actual,'state':{'workspace':torch.zeros(1,dtype=torch.int32)}}
        ready=torch.zeros(2,dtype=torch.int32);kw=operator_kwargs(x,b,ready)
        for k in ('req_ids','stable_prefix_lens','visible_seq_lens','token_to_req','active_rows'):
            self.assertIs(kw[k],actual[k])
        self.assertIs(kw['ready_counts'],ready)
    def test_packed_alias_and_causality(self):
        x=make_inputs(Shape(batch=2,width=8,seq_len=128,topk=8,capacity=32),seed=2026)
        key=x.qli['key'];scale=x.qli['key_dequant_scale']
        self.assertEqual(key.untyped_storage().data_ptr(),scale.untyped_storage().data_ptr())
        self.assertEqual(key.stride()[0],x.shape.page_bytes)
        self.assertEqual(scale.stride()[0],x.shape.page_bytes//4)
        np.testing.assert_array_equal(x.plan_metadata['visible_seq_lens'].numpy(),np.tile(np.arange(121,129),2))
        np.testing.assert_array_equal(x.qli['actual_seq_lengths_query'].numpy(),[8,16])
    def test_reproducible_input(self):
        s=Shape(batch=1,width=2,seq_len=128,topk=8,capacity=32)
        a=make_inputs(s,seed=2026);b=make_inputs(s,seed=2026)
        for k in a.qli:
            if torch.is_tensor(a.qli[k]):self.assertTrue(torch.equal(a.qli[k].view(torch.uint8),b.qli[k].view(torch.uint8)),k)
    def test_first_row_75pct_full_capacity(self):
        s=Shape(batch=1,width=8,seq_len=32768,capacity=16384)
        x=make_inputs(s,seed=2026)
        indices=np.tile(np.arange(2048,dtype=np.int32),(8,1))
        c=make_plan_case(x,indices,7,'full_cache_75pct',2026)
        resident=c['slot_to_token'][0]
        self.assertEqual(len(np.unique(resident)),16384)
        self.assertEqual(np.isin(indices[0],resident).sum(),1536)
        for row in c['slot_to_token']:np.testing.assert_array_equal(row,resident)

if __name__=='__main__':unittest.main()
