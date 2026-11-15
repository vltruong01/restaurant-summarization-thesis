"""Offline boundary tests only: meta shapes, CPU tensors/data and denied execution gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from accelerate.data_loader import DataLoaderShard

from stage_i_paired_pilot import (ROOT,ARMS,read,sha,expected_modules,meta_probe,initialized_A,module_seed,
    epoch_orders,PairedEpochSampler,gradient_record,verify_mask_batch,checked_preparation,execution_gate,
    verify_reference_freeze,paired_bootstrap)
from stage_hi_led_data import LEDCompleteCollator
from stage_i_decoding_diagnostics import structure,termination

CONFIG=ROOT/'configs/stage_i_paired_lora_pilot_v1.json'


class PilotPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.prep,cls.manifest=checked_preparation(CONFIG)
        cloud=ROOT/cls.config['frozen_training_root']
        mm=read(cloud/'assets/models/led-base-16384.config_manifest.json');cls.mc=read(cloud/mm['config_file'])

    def test_actual_meta_module_attachment_and_freezing(self):
        for arm,count in zip(ARMS,[294912,589824]):
            record=meta_probe(self.config,self.mc,arm)
            self.assertEqual(record['trainable_parameters'],count)
            self.assertEqual(set(record['modules']),expected_modules(arm))
            self.assertTrue(record['meta_only']);self.assertFalse(record['forward_backward'])
        self.assertEqual(record['encoder_local'],12);self.assertEqual(record['encoder_global'],12);self.assertEqual(record['decoder'],24)

    def test_independent_decoder_initialization_ignores_scope_order(self):
        names=sorted(expected_modules(ARMS[0]));all_names=sorted(expected_modules(ARMS[1]),reverse=True)
        a={n:initialized_A((8,768),42,n) for n in names}
        b={n:initialized_A((8,768),42,'base_model.model.'+n) for n in all_names}
        self.assertTrue(all(torch.equal(a[n],b[n]) for n in names))
        self.assertNotEqual(module_seed(42,names[0]),module_seed(42,names[1]))
        frozen=read(self.prep/'initialization_hashes.json')['per_module_sha256']
        for n in names:
            digest=hashlib.sha256(a[n].numpy().tobytes()+torch.zeros((768,8)).numpy().tobytes()).hexdigest()
            self.assertEqual(digest,frozen[n])

    def test_orders_are_equal_across_arms_and_distinct_epochs(self):
        plan=read(self.prep/'epoch_orders.json')['indices'];self.assertEqual(plan,epoch_orders(512,42,2))
        self.assertNotEqual(plan[0],plan[1]);self.assertEqual(sorted(plan[0]),list(range(512)))
        for arm in ARMS:
            sampler=PairedEpochSampler(plan);sampler.set_epoch(1);self.assertEqual(list(sampler),plan[1])

    def test_accelerate_propagates_sampler_epoch_on_CPU(self):
        plan=epoch_orders(12,42,2);loader=DataLoader(list(range(12)),batch_size=1,sampler=PairedEpochSampler(plan))
        shard=DataLoaderShard(loader.dataset,device=torch.device('cpu'),batch_sampler=loader.batch_sampler,rng_types=[])
        for epoch in [0,1]:
            shard.set_epoch(epoch);self.assertEqual([x.item() for x in shard],plan[epoch])

    def test_no_extra_epochs_or_partial_sampler(self):
        with self.assertRaises(ValueError):PairedEpochSampler([[0,0]])
        with self.assertRaises(ValueError):PairedEpochSampler([[0,1],[1,0]]).set_epoch(2)

    def test_original_training_and_evaluation_IDs_preserved(self):
        old=read(ROOT/'outputs/stage_i/training_failure_audit_v1/proposed_pilot_manifest.json');new=read(self.prep/'pilot_manifest.json')
        for key in ['training_business_ids','validation_loss_business_ids','generation_evaluation_random20','generation_evaluation_targeted4']:
            self.assertEqual(new[key],old[key])
        self.assertFalse(set(new['training_business_ids'])&set(new['validation_loss_business_ids']))

    def test_masks_shift_and_no_truncation(self):
        features=[{'input_ids':[0,15,16,2],'labels':[0,50,2]},{'input_ids':[0,2],'labels':[0,20,21,2]}]
        batch=LEDCompleteCollator()(features);verify_mask_batch(batch,features)
        altered={k:v.clone() for k,v in batch.items()};altered['labels'][0,-1]=1
        with self.assertRaises(ValueError):verify_mask_batch(altered,features)
        with self.assertRaises(ValueError):LEDCompleteCollator()([{'input_ids':[0]*5001+[2],'labels':[0,2]}])

    def test_initial_zero_A_gradients_then_active_required(self):
        # Synthetic parameters/assigned gradients only; no forward/backward or optimizer update.
        a=torch.nn.Parameter(torch.ones((2,2)));b=torch.nn.Parameter(torch.ones((2,2)));frozen=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        class Fake:
            def named_parameters(self):return iter([('x.lora_A.default.weight',a),('x.lora_B.default.weight',b),('base.weight',frozen)])
        a.grad=torch.zeros_like(a);b.grad=torch.ones_like(b)
        self.assertFalse(gradient_record(Fake())['x.lora_A.default.weight']['nonzero'])
        with self.assertRaises(ValueError):gradient_record(Fake(),True)
        a.grad=torch.ones_like(a);self.assertTrue(gradient_record(Fake(),True))
        b.grad.fill_(float('nan'))
        with self.assertRaises(ValueError):gradient_record(Fake(),True)
        b.grad=torch.ones_like(b);frozen.grad=torch.ones_like(frozen)
        with self.assertRaises(ValueError):gradient_record(Fake(),True)

    def test_scheduler_horizon_and_warmup_without_training(self):
        from transformers import get_linear_schedule_with_warmup
        p=torch.nn.Parameter(torch.zeros(1));optimizer=torch.optim.AdamW([p],lr=2e-4)
        schedule=get_linear_schedule_with_warmup(optimizer,7,128);fn=schedule.lr_lambdas[0]
        self.assertEqual(fn(0),0);self.assertEqual(fn(7),1);self.assertEqual(fn(128),0)
        self.assertEqual(self.config['training']['steps'],512//8*2)
        self.assertAlmostEqual(fn(64),(128-64)/(128-7))

    def test_execution_denied_before_GPU_or_model_loading(self):
        with patch('torch.cuda.is_available',side_effect=AssertionError('CUDA reached')):
            for mode in ['feasibility','train','generate']:
                with self.assertRaises(ValueError):execution_gate(CONFIG,mode,ARMS[0],None,False)
                with self.assertRaises(ValueError):execution_gate(CONFIG,mode,ARMS[0],None,True)
                with self.assertRaises(ValueError):execution_gate(CONFIG,mode,ARMS[0],self.prep/'approval_example_NOT_APPROVED.json',True)

    def test_explicit_scope_and_hash_binding_gate(self):
        with tempfile.TemporaryDirectory() as td:
            a=read(self.prep/'approval_example_NOT_APPROVED.json');a.update(approved=True,scopes=['feasibility'])
            p=Path(td)/'SYNTHETIC_GATE_TEST_ONLY.json';p.write_text(json.dumps(a),encoding='utf-8')
            self.assertEqual(execution_gate(CONFIG,'feasibility',ARMS[0],p,True)[0]['condition'],'filtered')
            with self.assertRaises(ValueError):execution_gate(CONFIG,'train',ARMS[0],p,True)
            a['config_sha256']='wrong';p.write_text(json.dumps(a),encoding='utf-8')
            with self.assertRaises(ValueError):execution_gate(CONFIG,'feasibility',ARMS[0],p,True)

    def test_training_needs_both_reviewed_feasibility_reports(self):
        with tempfile.TemporaryDirectory() as td:
            a=read(self.prep/'approval_example_NOT_APPROVED.json');a.update(approved=True,scopes=['train'])
            p=Path(td)/'SYNTHETIC_GATE_TEST_ONLY.json';p.write_text(json.dumps(a),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Both feasibility'):execution_gate(CONFIG,'train',ARMS[0],p,True)

    def test_runtime_version_drift_denied(self):
        with patch('importlib.metadata.version',return_value='0.0.0'):
            with self.assertRaisesRegex(ValueError,'Runtime version drift'):checked_preparation(CONFIG)

    def test_stale_feasibility_code_binding_denied(self):
        with tempfile.TemporaryDirectory(dir=self.prep) as td:
            p=Path(td);a=read(self.prep/'approval_example_NOT_APPROVED.json');a.update(approved=True,scopes=['train'])
            reports={}
            for arm in ARMS:
                r=p/(arm+'.json');r.write_text(json.dumps({'status':'TECHNICAL_FEASIBILITY_PASS','arm':arm,'config_sha256':sha(CONFIG),'binding':{'preparation_manifest_sha256':'old'}}),encoding='utf-8')
                reports[arm]={'path':r.relative_to(ROOT).as_posix(),'sha256':sha(r)}
            a['reviewed_feasibility']=reports;approval=p/'SYNTHETIC_TEST_ONLY.json';approval.write_text(json.dumps(a),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Stale feasibility binding'):execution_gate(CONFIG,'train',ARMS[0],approval,True)

    def test_reference_freeze_gates_generation_not_training_preparation(self):
        with tempfile.TemporaryDirectory(dir=self.prep) as td:
            p=Path(td);a=read(self.prep/'approval_example_NOT_APPROVED.json');a.update(approved=True,scopes=['train','generate'])
            reports={}
            for arm in ARMS:
                r=p/(arm+'.json');r.write_text(json.dumps({'status':'TECHNICAL_FEASIBILITY_PASS','arm':arm,'config_sha256':sha(CONFIG),
                    'binding':{'preparation_manifest_sha256':sha(self.prep/'PREPARATION_MANIFEST.json'),
                    'runner_sha256':sha(ROOT/'scripts/run_stage_i_paired_lora_pilot_v1.py'),'helper_sha256':sha(ROOT/'scripts/stage_i_paired_pilot.py')}}),encoding='utf-8')
                reports[arm]={'path':r.relative_to(ROOT).as_posix(),'sha256':sha(r)}
            a['reviewed_feasibility']=reports;f=p/'SYNTHETIC_TEST_ONLY.json';f.write_text(json.dumps(a),encoding='utf-8')
            self.assertEqual(execution_gate(CONFIG,'train',ARMS[0],f,True)[0]['condition'],'filtered')
            with self.assertRaisesRegex(ValueError,'Freeze source units'):execution_gate(CONFIG,'generate',ARMS[0],f,True)

    def test_unverified_teacher_candidates_cannot_be_reference_truth(self):
        value=read(self.prep/'reference_freeze_template.json')
        with self.assertRaises(ValueError):verify_reference_freeze(value,self.config,self.prep)
        rows=[json.loads(x) for x in (self.prep/'reference_unit_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(rows);self.assertTrue(all(r['human_verified'] is False and r['support_status'] is None for r in rows))

    def test_reference_domain_quote_and_provenance_validation(self):
        # Entirely synthetic unit-test evidence, not user annotation or scientific labels.
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);case={'business_id':'SYNTHETIC_TEST','source':'[REVIEWS]\n[REV_1] stars=5\nGood soup.\n\n[VISUAL_CONTEXT]\n',
                             'selected_review_ids':['TEST_REV'],'selected_photo_ids':[]}
            (p/'evaluation_inputs.jsonl').write_text(json.dumps(case)+'\n',encoding='utf-8');(p/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md').write_text('SYNTHETIC_TEST',encoding='utf-8')
            value={'case_inputs_sha256':sha(p/'evaluation_inputs.jsonl'),'protocol_sha256':sha(p/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),
                'researcher_confirmed_complete':True,'business_reviews':{'SYNTHETIC_TEST':{'human_verified':True,'positive_negative_and_specificity_checked':True}},
                'units':[{'unit_id':'TEST','business_id':'SYNTHETIC_TEST','domain':'review','text':'Good soup.','importance':'core','polarity':'positive',
                 'support_status':'supported','human_verified':True,'confirmed_by':'SYNTHETIC_TEST_NOT_HUMAN','aspect':'food','restaurant_specific':True,
                 'evidence':[{'marker':'REV_1','review_id':'TEST_REV','quote':'Good soup.'}]}]}
            verify_reference_freeze(value,self.config,p)
            bad=deepcopy(value);bad['units'][0]['evidence'][0]['quote']='Invented.'
            with self.assertRaises(ValueError):verify_reference_freeze(bad,self.config,p)
            bad=deepcopy(value);bad['units'][0]['domain']='visual'
            with self.assertRaises(ValueError):verify_reference_freeze(bad,self.config,p)

    def test_zero_caption_and_malformed_headers_never_safe_empty(self):
        good='[REVIEW_CONSENSUS]\nReview.\n\n[VISUAL_EVIDENCE]\n'
        self.assertTrue(structure(good,'filtered',0)['empty_visual_compliant'])
        self.assertFalse(structure(good+'There is a bagel.','filtered',0)['empty_visual_compliant'])
        self.assertFalse(structure(good.replace('VISUAL_EVIDENCE','VISUAL__EVIDENCE'),'filtered',0)['empty_visual_compliant'])
        self.assertFalse(termination([2,0,45])['eos_after_decoder_start'])
        self.assertTrue(termination([2]+[0]*766+[2])['eos_at_cap_may_be_forced'])

    def test_paired_uncertainty_reproducible_and_missing_not_zero(self):
        a=[1,2,3,None];b=[2,3,4,10];x=paired_bootstrap(a,b)
        self.assertEqual(x,paired_bootstrap(a,b));self.assertEqual(x['paired_n'],3);self.assertEqual(x['missing_pairs'],1)
        self.assertEqual(x['CI95'],[1,1]);self.assertIsNone(paired_bootstrap([None],[1])['CI95'])
        with self.assertRaises(ValueError):paired_bootstrap([1],[2,3])

    def test_no_quality_win_cutoffs_in_active_config(self):
        self.assertIn('no rigid numerical',self.config['quality_decision'])
        protocol=(self.prep/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md').read_text(encoding='utf-8')
        self.assertIn('superseded',protocol);self.assertIn('no weighted composite',protocol.lower())


if __name__=='__main__':unittest.main(verbosity=2)
