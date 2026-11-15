"""CPU-only loss/alignment/negative-control tests. No LED, GPU or optimizer step."""
import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from tokenizers import Tokenizer
from stage_i_unlikelihood_v1 import *


class ULTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
        cls.text='[REVIEW_CONSENSUS]\nCustomers praise the fresh pizza. Some report slow service.\n\n[VISUAL_EVIDENCE]\nA plate of pizza is on a table.'

    def view(self,kind):
        return build_negative(self.text,self.tok,'synthetic_only',kind)

    def test_sentence_view_exists(self):
        self.assertTrue(self.view('sentence')['eligible'])

    def test_word_view_exists(self):
        self.assertTrue(self.view('word')['eligible'])

    def test_deterministic(self):
        self.assertEqual(self.view('word'),self.view('word'))

    def test_positive_unchanged(self):
        before=self.tok.encode(self.text).ids
        self.view('sentence');self.view('word')
        self.assertEqual(before,self.tok.encode(self.text).ids)

    def test_negative_roundtrip_and_mask(self):
        for kind in ('sentence','word'):
            v=self.view(kind)
            encoded=self.tok.encode(v['negative_text'])
            self.assertEqual(encoded.ids,v['negative_ids'])
            self.assertEqual(len(v['negative_ids']),len(v['penalty_mask']))
            self.assertGreater(sum(v['penalty_mask']),0)
            start,end=v['inserted_char_span']
            for i,m in enumerate(v['penalty_mask']):
                if m:
                    a,b=encoded.offsets[i]
                    self.assertGreaterEqual(a,start);self.assertLessEqual(b,end)
                    self.assertNotIn(encoded.ids[i],[0,1,2])

    def test_gold_branch_conflicts_excluded(self):
        for kind in ('sentence','word'):
            v=self.view(kind);original=self.tok.encode(self.text).ids
            for i,m in enumerate(v['penalty_mask']):
                if m and v['negative_ids'][:i]==original[:i] and i<len(original):
                    self.assertNotEqual(v['negative_ids'][i],original[i])

    def test_overflow_skipped_not_truncated(self):
        v=build_negative(self.text,self.tok,'synthetic','sentence',limit=2)
        self.assertFalse(v['eligible']);self.assertGreater(v['rejected_candidates']['overflow'],0)

    def test_bad_markers_rejected(self):
        with self.assertRaises(ValueError): build_negative('No markers.',self.tok,'synthetic','word')

    def test_empty_visual_no_visual_negative(self):
        text='[REVIEW_CONSENSUS]\nCustomers praise the pizza.\n\n[VISUAL_EVIDENCE]\n'
        v=build_negative(text,self.tok,'synthetic','sentence')
        self.assertTrue(v['eligible']);self.assertNotIn('[VISUAL_EVIDENCE]',v['negative_text'])

    def test_existing_word_repetition_not_selected(self):
        text='[REVIEW_CONSENSUS]\nBang Bang.\n\n[VISUAL_EVIDENCE]\n'
        # First Bang would reproduce the existing gold continuation and is excluded.
        first=text.index('Bang')
        self.assertNotIn((first,first+4),candidates(text,'word'))

    def test_unknown_negative_kind(self):
        with self.assertRaises(ValueError): candidates(self.text,'semantic')

    def test_uniform_loss_matches_closed_form(self):
        logits=torch.zeros(1,2,5,requires_grad=True)
        loss=unlikelihood_loss(logits,torch.tensor([[3,4]]),torch.tensor([[True,False]]))
        self.assertAlmostEqual(loss.item(),-math.log(.8),places=6)

    def test_only_masked_positions_receive_gradient(self):
        logits=torch.zeros(1,3,6,requires_grad=True)
        loss=unlikelihood_loss(logits,torch.tensor([[0,3,2]]),torch.tensor([[False,True,False]]))
        loss.backward()
        self.assertEqual(logits.grad[0,0].abs().sum().item(),0)
        self.assertEqual(logits.grad[0,2].abs().sum().item(),0)
        self.assertGreater(logits.grad[0,1,3].item(),0) # descent lowers negative token

    def test_empty_mask_differentiable_zero(self):
        logits=torch.zeros(1,2,5,requires_grad=True)
        loss=unlikelihood_loss(logits,torch.tensor([[0,2]]),torch.zeros(1,2,dtype=torch.bool))
        loss.backward();self.assertEqual(loss.item(),0);self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_fp16_logits_upcast(self):
        loss=unlikelihood_loss(torch.zeros(1,1,5,dtype=torch.float16),torch.tensor([[3]]),torch.tensor([[True]]))
        self.assertEqual(loss.dtype,torch.float32)

    def test_large_finite_probability_gradient(self):
        logits=torch.tensor([[[0.,0.,0.,8.,0.]]],requires_grad=True)
        loss=unlikelihood_loss(logits,torch.tensor([[3]]),torch.tensor([[True]]))
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_exact_probability_one_stops(self):
        with self.assertRaises(ValueError):
            unlikelihood_loss(torch.tensor([[[-1000.,-1000.,-1000.,1000.]]]),torch.tensor([[3]]),torch.tensor([[True]]))

    def test_nonfinite_logits_stop(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                unlikelihood_loss(torch.full((1,1,4),value),torch.tensor([[3]]),torch.tensor([[True]]))

    def test_special_tokens_never_penalized(self):
        for value in (0,1,2):
            with self.assertRaises(ValueError):
                unlikelihood_loss(torch.zeros(1,1,5),torch.tensor([[value]]),torch.tensor([[True]]))

    def test_shape_mismatch_stops(self):
        with self.assertRaises(ValueError):
            unlikelihood_loss(torch.zeros(1,2,5),torch.tensor([[3]]),torch.tensor([[True]]))

    def test_wrong_mask_dtype_stops(self):
        with self.assertRaises(ValueError):
            unlikelihood_loss(torch.zeros(1,1,5),torch.tensor([[3]]),torch.tensor([[1]]))

    def test_invalid_ids_stop(self):
        with self.assertRaises(ValueError):
            unlikelihood_loss(torch.zeros(1,1,5),torch.tensor([[9]]),torch.tensor([[True]]))

    @staticmethod
    def fake_inputs():
        return dict(input_ids=torch.tensor([[0,3,2]]),attention_mask=torch.ones(1,3,dtype=torch.long),
            global_attention_mask=torch.tensor([[1,0,0]]),labels=torch.tensor([[0,4,2]]),
            negative_ids=torch.tensor([[0,3,4,2]]),negative_penalty_mask=torch.tensor([[False,False,True,False]]))

    class Fake:
        def __init__(self): self.calls=[];self.logits=torch.zeros(1,4,6,requires_grad=True)
        def __call__(self,**kw):
            self.calls.append(kw)
            return SimpleNamespace(loss=torch.tensor(2.,requires_grad=True),logits=self.logits)

    def test_control_and_treatment_both_forward_negative(self):
        for w in (0.,.1):
            m=self.Fake();loss,_=paired_objective(m,self.fake_inputs(),w,True)
            self.assertEqual(len(m.calls),2)
            self.assertAlmostEqual(loss.item(),2+w*(-math.log(5/6)),places=6)

    def test_explicit_decoder_shift_alignment(self):
        m=self.Fake();inputs=self.fake_inputs();paired_objective(m,inputs,.1,True)
        self.assertEqual(m.calls[1]['decoder_input_ids'].tolist(),[[2,0,3,4]])
        self.assertEqual(m.calls[1]['decoder_attention_mask'].tolist(),[[1,1,1,1]])
        self.assertNotIn('labels',m.calls[1])
        for k in ('input_ids','attention_mask','global_attention_mask'):
            self.assertTrue(torch.equal(m.calls[0][k],m.calls[1][k]))

    def test_eval_is_pure_CE(self):
        m=self.Fake();loss,_=paired_objective(m,self.fake_inputs(),.1,False)
        self.assertEqual(len(m.calls),1);self.assertEqual(loss.item(),2.)

    def test_empty_negative_skips_in_both_arms(self):
        for weight in (0.,.1):
            m=self.Fake();inputs=self.fake_inputs();inputs['negative_penalty_mask'].zero_()
            paired_objective(m,inputs,weight,True);self.assertEqual(len(m.calls),1)

    def test_no_negative_keys_leak_to_positive(self):
        m=self.Fake();paired_objective(m,self.fake_inputs(),.1,True)
        self.assertFalse(any(k.startswith('negative_') for k in m.calls[0]))

    def test_logging_callback_JSON_serialization(self):
        logs=[];m=self.Fake();paired_objective(m,self.fake_inputs(),.1,True,logs.append)
        self.assertEqual(json.loads(json.dumps(logs,allow_nan=False))[0]['penalized_tokens'],1)

    def test_weight_search_not_supported(self):
        with self.assertRaises(ValueError): paired_objective(self.Fake(),self.fake_inputs(),.2,True)

    def test_authorization_template_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';p.write_text(json.dumps({'GPU_execution_approved':False}))
            with self.assertRaises(ValueError):
                authorization_gate(p,{'experiment':'unapproved'}, {},'train','control_ce')

    def test_unknown_arm_or_mode_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';p.write_text('{}')
            with self.assertRaises(ValueError):authorization_gate(p,{}, {},'train','other')
            with self.assertRaises(ValueError):authorization_gate(p,{}, {},'sweep','control_ce')

    def approval(self):
        return {'experiment':'synthetic','GPU_execution_approved':True,'approval_source':'researcher_message',
            'researcher_explicit_authorization':'synthetic unit-test fixture only, not an actual approval',
            'config_sha256':'fixture','preflight_manifest_sha256':'fixture','runner_sha256':'fixture',
            'approved_arms':['control_ce','treatment_ul'],'approved_modes':['feasibility','train','evaluate'],
            'max_training_updates_total':256,'max_gate_updates_total':6,'max_generation_outputs_total':48,
            'held_out_access':False,'Teacher_calls':False,'execution_authorized_at_utc':'2026-10-19T00:00:00+00:00',
            'new_execution_deadline_utc':'2026-10-20T00:00:00+00:00','prior_stop_loss_override_explicit':True}

    def check_synthetic_approval(self,a):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.json';p.write_text(json.dumps(a))
            with patch('stage_i_unlikelihood_v1.sha',return_value='fixture'):
                return authorization_gate(p,{'experiment':'synthetic'}, {},'train','control_ce',
                        datetime.fromisoformat('2026-10-19T01:00:00+00:00'))

    def test_hash_mismatch_rejected(self):
        a=self.approval();a['config_sha256']='wrong'
        with self.assertRaises(ValueError):self.check_synthetic_approval(a)

    def test_expired_authorization_rejected(self):
        a=self.approval();a['new_execution_deadline_utc']='2026-10-19T00:30:00+00:00'
        with self.assertRaises(ValueError):self.check_synthetic_approval(a)

    def test_prior_final_attempt_not_implicitly_overridden(self):
        a=self.approval();a['prior_stop_loss_override_explicit']=False
        with self.assertRaises(ValueError):self.check_synthetic_approval(a)

    def test_scope_expansion_rejected(self):
        a=self.approval();a['max_training_updates_total']=750
        with self.assertRaises(ValueError):self.check_synthetic_approval(a)

    def test_more_than_48hours_rejected(self):
        a=self.approval();a['new_execution_deadline_utc']='2026-10-25T00:00:00+00:00'
        with self.assertRaises(ValueError):self.check_synthetic_approval(a)

    def test_runner_import_has_no_model_load(self):
        spec=importlib.util.spec_from_file_location('prepared_ul_runner',ROOT/'scripts/run_stage_i_unlikelihood_pilot_v1.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.preflight))

    def test_loss_primitives_do_not_initialize_CUDA(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__': unittest.main(verbosity=2)
