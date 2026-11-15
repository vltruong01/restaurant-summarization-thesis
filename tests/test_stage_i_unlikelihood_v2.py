"""Focused v2 CPU regressions. Synthetic autograd only; no LED/GPU execution."""
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_unlikelihood_v2 as v2


class SentenceRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
        cls.text='[REVIEW_CONSENSUS]\nCustomers praise the fresh pizza. Some report slow service.\n\n[VISUAL_EVIDENCE]\nA plate of pizza is on a table.'

    def test_word_negatives_are_rejected(self):
        with self.assertRaises(ValueError):v2.build_negative(self.text,self.tok,'synthetic','word')

    def test_sentence_negatives_retain_alignment(self):
        n=v2.build_negative(self.text,self.tok,'synthetic','sentence')
        self.assertTrue(n['eligible']);self.assertEqual(n['kind'],'sentence')
        self.assertEqual(n['negative_ids'],self.tok.encode(n['negative_text']).ids)
        original=self.tok.encode(self.text).ids
        self.assertEqual(original[:n['branch_token_position']],n['negative_ids'][:n['branch_token_position']])

    def test_ineligible_business_is_not_fabricated(self):
        text='[REVIEW_CONSENSUS]\nGood.\n\n[VISUAL_EVIDENCE]\n'
        n=v2.build_negative(text,self.tok,'synthetic','sentence')
        self.assertFalse(n['eligible']);self.assertEqual(n['reason'],'NO_SAFE_BOUNDED_NEGATIVE')

    def test_overflow_does_not_fallback_to_word(self):
        n=v2.build_negative(self.text,self.tok,'synthetic','sentence',limit=2)
        self.assertFalse(n['eligible']);self.assertEqual(n['kind'],'sentence')

    def test_config_changes_are_minimal(self):
        a=v2.read(ROOT/'configs/stage_i_unlikelihood_pilot_v1.json')
        b=v2.read(v2.CONFIG)
        changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
        self.assertEqual(changed,{'experiment','run_root','negative_design','gate_gradient_diagnostics','practical_quality_benchmark'})
        self.assertFalse(b['GPU_execution_approved']);self.assertIsNone(b['new_execution_deadline_utc'])

    def test_fixed_ids_and_orders_are_byte_identical(self):
        old=ROOT/'outputs/stage_i/unlikelihood_preparation_v1'
        for name in ('TRAINING_MANIFEST.json','EVALUATION_MANIFEST.json','ADAPTER_INVENTORY.json'):
            self.assertEqual((old/name).read_bytes(),(v2.PREP/name).read_bytes())

    def test_all512_sentence_negatives_eligible(self):
        rows=[json.loads(s) for s in (v2.PREP/'NEGATIVE_VIEWS.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),512)
        self.assertTrue(all(r['eligible'] and r['kind']=='sentence' and any(r['penalty_mask']) for r in rows))
        self.assertTrue(all(len(r['negative_ids'])<=768 for r in rows))

    def test_original256_sentence_views_identical(self):
        old=[json.loads(s) for s in (ROOT/'outputs/stage_i/unlikelihood_preparation_v1/NEGATIVE_VIEWS.jsonl').read_text(encoding='utf-8').splitlines()]
        new=[json.loads(s) for s in (v2.PREP/'NEGATIVE_VIEWS.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(sum(a['kind']=='sentence' for a in old),256)
        self.assertTrue(all(a==b for a,b in zip(old,new) if a['kind']=='sentence'))

    def test_benchmark_is_saved_non_causal_same24(self):
        from compare_stage_i_unlikelihood_v2 import validate_benchmark
        cfg=v2.read(v2.CONFIG);saved=validate_benchmark(cfg)
        self.assertEqual(len(saved),24)
        self.assertTrue(all(r['epoch']==3 and r['arm']=='Expanded_full_epoch3' for r in saved.values()))

    def test_v1_hashes_and_deadlines_remain_bound(self):
        oldroot=ROOT/'outputs/stage_i/unlikelihood_preparation_v1'
        receipt=v2.read(oldroot/'DELIVERY_RECEIPT.json')
        self.assertEqual(v2.sha(ROOT/'configs/stage_i_unlikelihood_pilot_v1.json'),receipt['config_sha256'])
        self.assertEqual(v2.sha(oldroot/'PREFLIGHT_MANIFEST.json'),receipt['preflight_manifest_sha256'])
        self.assertEqual(v2.read(v2.CONFIG)['old_deadlines_preserved'],v2.read(ROOT/'configs/stage_i_unlikelihood_pilot_v1.json')['old_deadlines_preserved'])

    def test_runner_import_does_not_load_model(self):
        spec=importlib.util.spec_from_file_location('ul_runner_v2',v2.RUNNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(callable(module.preflight));self.assertFalse(torch.cuda.is_initialized())


class GradientSignalTests(unittest.TestCase):
    def setup_losses(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        return p,(p*p).sum(),(p*.003).sum()+2

    def probe(self,p,ce,ul,weight=.1,scale=1.):
        return v2.gradient_contributions(ce,ul,[('synthetic.lora_A.default.weight',p)],weight,scale)

    def test_actual_gradient_norm_and_loss_ratio(self):
        p,ce,ul=self.setup_losses();r=self.probe(p,ce,ul)
        self.assertAlmostEqual(r['CE_gradient_norm'],math.sqrt(20),places=6)
        self.assertAlmostEqual(r['UL_gradient_norm_raw'],math.sqrt(2)*.003,places=8)
        self.assertAlmostEqual(r['weighted_UL_loss_over_CE'],float(ul.detach())*.1/5,places=7)

    def test_weak_signal_reported_without_weight_change(self):
        p,ce,ul=self.setup_losses();r=self.probe(p,ce,ul)
        self.assertIn('WEIGHTED_UL_NORM_BELOW_1_PERCENT_OF_CE_DESCRIPTIVE_ONLY',r['signal_flags'])
        self.assertEqual(r['effective_weight'],.1);self.assertFalse(r['weight_adjusted'])

    def test_nonzero_UL_loss_with_zero_gradient_detected(self):
        p,ce,_=self.setup_losses();ul=(p*0).sum()+3
        r=self.probe(p,ce,ul)
        self.assertEqual(r['UL_loss_raw'],3.);self.assertEqual(r['UL_gradient_norm_raw'],0.)
        self.assertIn('ZERO_UL_GRADIENT_DESPITE_POSSIBLY_NONZERO_LOSS',r['signal_flags'])

    def test_constant_UL_missing_gradient_is_not_effective_signal(self):
        p,ce,_=self.setup_losses();r=self.probe(p,ce,torch.tensor(3.))
        self.assertEqual(r['UL_gradient_norm_raw'],0.)
        self.assertTrue(next(iter(r['per_tensor'].values()))['UL_missing'])

    def test_control_counterfactual_is_not_applied(self):
        p,ce,ul=self.setup_losses();r=self.probe(p,ce,ul,weight=0.)
        self.assertGreater(r['UL_gradient_norm_weighted_0_1'],0.)
        self.assertEqual(r['UL_gradient_norm_effective'],0.)
        self.assertEqual(r['effective_combined_gradient_norm'],r['CE_gradient_norm'])

    def test_AMP_scaling_is_unscaled_in_report(self):
        p,ce,ul=self.setup_losses();a=self.probe(p,ce,ul);b=self.probe(p,ce,ul,scale=1024.)
        self.assertAlmostEqual(a['UL_gradient_norm_raw'],b['UL_gradient_norm_raw'],places=10)
        self.assertAlmostEqual(a['CE_gradient_norm'],b['CE_gradient_norm'],places=10)

    def test_accumulated_grad_buffers_unchanged(self):
        p,ce,ul=self.setup_losses();p.grad=torch.tensor([8.,9.]);old=p.grad.clone()
        self.probe(p,ce,ul);self.assertTrue(torch.equal(p.grad,old))

    def test_no_buffer_created_if_originally_none(self):
        p,ce,ul=self.setup_losses();self.probe(p,ce,ul);self.assertIsNone(p.grad)

    def test_actual_combined_backward_still_works(self):
        p,ce,ul=self.setup_losses();self.probe(p,ce,ul)
        (ce+.1*ul).backward()
        self.assertTrue(torch.allclose(p.grad,torch.tensor([2.0003,4.0003]),atol=1e-6))

    def test_antiparallel_contribution_not_interpreted_as_improvement(self):
        p,ce,_=self.setup_losses();r=self.probe(p,ce,100-ce)
        self.assertAlmostEqual(r['CE_UL_gradient_cosine'],-1.,places=7)
        self.assertAlmostEqual(r['effective_combined_gradient_norm'],.9*r['CE_gradient_norm'],places=6)

    def test_frozen_backbone_not_probe_target(self):
        p,ce,ul=self.setup_losses()
        with self.assertRaises(ValueError):v2.gradient_contributions(ce,ul,[('backbone.weight',p)],.1)

    def test_nonfinite_gradient_stops(self):
        p=torch.nn.Parameter(torch.tensor([0.]))
        with self.assertRaises(ValueError):self.probe(p,p.sum()+1,p.sqrt().sum()+2)

    def test_invalid_scale_stops(self):
        p,ce,ul=self.setup_losses()
        for s in (0.,float('inf')):
            with self.assertRaises(ValueError):self.probe(p,ce,ul,scale=s)

    def test_all48_lora_tensors_reported(self):
        ps=[torch.nn.Parameter(torch.tensor([1.])) for _ in range(48)]
        ce=sum(p.square().sum() for p in ps);ul=sum(p.sum()*.002 for p in ps)
        r=v2.gradient_contributions(ce,ul,[(f'module{i}.lora_A.default.weight',p) for i,p in enumerate(ps)],.1)
        self.assertEqual(r['parameter_tensors'],48);self.assertEqual(len(r['per_tensor']),48)

    def test_diagnostic_output_JSON_serializable(self):
        p,ce,ul=self.setup_losses();r=self.probe(p,ce,ul)
        self.assertTrue(json.loads(json.dumps(r,allow_nan=False))['RNG_unchanged'])


if __name__=='__main__':unittest.main(verbosity=2)
