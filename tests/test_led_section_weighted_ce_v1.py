"""CPU fixtures only: no pretrained backbone, generation, or research optimizer run."""
import copy
import io
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import led_section_weighted_ce_v1 as w
from teacher_redesign_token_count_v1 import load_tokenizer
from led_large_decoder_label_view_v1 import CorrectedLabelCollator


class WeightedCE(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(813)
        logits=torch.randn(2,5,11,requires_grad=True)
        labels=torch.tensor([[4,5,6,2,-100],[4,7,2,-100,-100]])
        roles=torch.tensor([[w.HEADER,w.REVIEW,w.VISUAL,w.EOS,w.IGNORE],
                            [w.HEADER,w.REVIEW,w.EOS,w.IGNORE,w.IGNORE]],dtype=torch.int8)
        return logits,labels,roles

    def test_all_ones_matches_native_CE_value_and_every_logit_gradient(self):
        x,y,r=self.fixture();base=F.cross_entropy(x.flatten(0,1),y.flatten(),ignore_index=-100)
        result,_=w.weighted_cross_entropy(x,y,r,1.,1.,1.)
        torch.testing.assert_close(result,base,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(torch.autograd.grad(result,x,retain_graph=True)[0],
                                   torch.autograd.grad(base,x)[0],rtol=1e-6,atol=1e-7)

    def test_exact_weighted_formula_and_contribution_sum(self):
        x,y,r=self.fixture();loss,e=w.weighted_cross_entropy(x,y,r)
        ce=F.cross_entropy(x.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
        weight=torch.where(r==w.REVIEW,1.5,1.)*(y!=-100)
        torch.testing.assert_close(loss,(ce*weight).sum()/weight.sum())
        self.assertAlmostEqual(e['valid_weight_sum'],8.)
        self.assertAlmostEqual(sum(v['normalized_loss_contribution'] for v in e['sections'].values()),loss.item(),places=6)

    def test_padding_extension_changes_neither_value_nor_valid_gradients(self):
        x,y,r=self.fixture();loss,_=w.weighted_cross_entropy(x,y,r)
        xx=torch.cat([x,torch.full((2,17,11),7.)],dim=1)
        yy=F.pad(y,(0,17),value=-100);rr=F.pad(r,(0,17),value=w.IGNORE)
        padded,_=w.weighted_cross_entropy(xx,yy,rr)
        torch.testing.assert_close(loss,padded)
        g=torch.autograd.grad(padded,xx)[0];self.assertTrue((g[:,5:]==0).all())

    def test_EOS_and_headers_keep_neutral_weight_and_EOS_gradient(self):
        x,y,r=self.fixture();loss,e=w.weighted_cross_entropy(x,y,r)
        self.assertEqual(e['sections']['EOS']['weight_sum'],2.)
        self.assertEqual(e['sections']['headers_separators']['weight_sum'],2.)
        g=torch.autograd.grad(loss,x)[0]
        self.assertGreater(g[0,3].norm().item(),0.)
        self.assertTrue((g[y==-100]==0).all())

    def test_no_visual_body_is_valid_and_review_weight_is_still_normalized(self):
        x=torch.randn(1,4,10,requires_grad=True);y=torch.tensor([[5,6,2,-100]])
        r=torch.tensor([[w.HEADER,w.REVIEW,w.EOS,w.IGNORE]])
        loss,e=w.weighted_cross_entropy(x,y,r)
        self.assertEqual(e['sections']['visual']['tokens'],0)
        self.assertEqual(e['valid_weight_sum'],3.5);self.assertTrue(torch.isfinite(loss))

    def test_mask_mismatch_all_ignored_and_nonfinite_fail_closed(self):
        x,y,r=self.fixture()
        for yy,rr in [(torch.full_like(y,-100),torch.full_like(r,-1)),(y,torch.zeros_like(r))]:
            with self.assertRaises(ValueError):w.weighted_cross_entropy(x,yy,rr)
        x=x.detach().clone();x[0,0,0]=float('nan')
        with self.assertRaises(ValueError):w.weighted_cross_entropy(x,y,r)

    def test_accumulation_is_mean_of_eight_per_example_normalized_losses(self):
        torch.manual_seed(14);p=torch.randn(8,5,11,requires_grad=True)
        labels=torch.tensor([[4,5,6,2,-100]]*8);roles=torch.tensor([[0,1,2,3,-1]]*8)
        total=sum(w.weighted_cross_entropy(p[i:i+1],labels[i:i+1],roles[i:i+1])[0] for i in range(8))/8
        expected=torch.autograd.grad(total,p,retain_graph=True)[0]
        for i in range(8):
            loss,_=w.weighted_cross_entropy(p[i:i+1],labels[i:i+1],roles[i:i+1]);(loss/8).backward()
        torch.testing.assert_close(p.grad,expected)

    def test_CPU_autocast_has_FP32_finite_loss_and_gradients(self):
        x,y,r=self.fixture()
        with torch.autocast('cpu',dtype=torch.bfloat16):loss,_=w.weighted_cross_entropy(x,y,r)
        self.assertEqual(loss.dtype,torch.float32);loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
        # CPU bfloat16 is a numerical fixture, NOT CUDA-FP16 feasibility evidence.

    def test_scaled_backward_unscale_matches_unscaled_without_optimizer_update(self):
        x,y,r=self.fixture();loss,_=w.weighted_cross_entropy(x,y,r)
        normal=torch.autograd.grad(loss,x,retain_graph=True)[0]
        scaled=torch.autograd.grad(loss*65536.,x)[0]/65536.
        torch.testing.assert_close(normal,scaled);self.assertTrue(torch.isfinite(scaled).all())

    def test_synthetic_checkpoint_roundtrip_preserves_next_gradient_and_optimizer_state(self):
        # One tiny CPU fixture update, no LED/LoRA research checkpoint or GPU involved.
        torch.manual_seed(3);param=torch.nn.Parameter(torch.randn(1,4,11));opt=torch.optim.AdamW([param],lr=.0002)
        y=torch.tensor([[4,5,6,2]]);r=torch.tensor([[0,1,2,3]])
        loss,_=w.weighted_cross_entropy(param,y,r);loss.backward();opt.step();opt.zero_grad()
        buffer=io.BytesIO();torch.save(dict(param=param.detach().clone(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),buffer)
        nextloss,_=w.weighted_cross_entropy(param,y,r);nextgrad=torch.autograd.grad(nextloss,param)[0]
        buffer.seek(0);s=torch.load(buffer,map_location='cpu',weights_only=False)
        restored=torch.nn.Parameter(s['param']);other=torch.optim.AdamW([restored],lr=.0002);other.load_state_dict(s['optimizer'])
        again,_=w.weighted_cross_entropy(restored,y,r)
        torch.testing.assert_close(again,nextloss);torch.testing.assert_close(torch.autograd.grad(again,restored)[0],nextgrad)
        self.assertEqual(int(other.state[restored]['step']),1)


class SectionIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=load_tokenizer(ROOT/'outputs/stage_i/teacher_target_v2_preparation_v1/stage_a_sources_32/tokenizer')
        cls.prep=ROOT/'outputs/stage_i/led_large_section_wce_v1'
        cls.index=w.SectionMaskIndex(cls.prep/'SECTION_MASK_VIEWS.npz',w.hashlib.sha256((cls.prep/'SECTION_MASK_VIEWS.npz').read_bytes()).hexdigest())

    def test_space_only_offsets_and_unicode_keep_frozen_ids(self):
        text=w.REVIEW_HEADER+'Café  shrimp: praised, but one diner disagrees.'+w.VISUAL_SEPARATOR+'A plate.'
        ids,roles,_=w.segment_target(text,self.tok)
        self.assertEqual(ids,self.tok.encode(text).ids[1:]);self.assertEqual(roles[-1],w.EOS)
        self.assertEqual(len(ids),len(roles))

    def test_empty_visual_duplicate_missing_or_injected_markers(self):
        ids,roles,_=w.segment_target(w.REVIEW_HEADER+'Useful review.'+w.VISUAL_SEPARATOR,self.tok)
        self.assertNotIn(w.VISUAL,roles)
        for text in ['Useful review.',w.REVIEW_HEADER+'Text.',w.REVIEW_HEADER+'Text.'+w.VISUAL_SEPARATOR+w.VISUAL_SEPARATOR]:
            with self.assertRaises(ValueError):w.segment_target(text,self.tok)

    def test_all_2200_immutable_labels_lookup_exactly_with_stress_padding(self):
        with np.load(self.prep/'SECTION_MASK_VIEWS.npz',allow_pickle=False) as p:
            self.assertEqual(len(p['business_ids']),2200)
            for i in range(2200):
                a,b=p['offsets'][i:i+2];ids=p['labels'][a:b].tolist();expected=p['roles'][a:b].tolist()
                y=torch.tensor([ids+[-100]*(768-len(ids))])
                actual=self.index.for_labels(y)[0].tolist()
                self.assertEqual(actual,expected+[-1]*(768-len(ids)))

    def test_unknown_changed_or_interior_ignored_labels_rejected(self):
        for row in [[9,10,2],[5,-100,2],[0,5,2]]:
            with self.assertRaises(ValueError):self.index.for_labels(torch.tensor([row]))

    def test_pinned_LED_shift_single_alignment_without_BOS(self):
        from transformers.models.led.modeling_led import shift_tokens_right
        labels=torch.tensor([[10975,4629,2,-100,-100]])
        before=labels.clone();shifted=shift_tokens_right(labels,1,2)
        self.assertEqual(shifted.tolist(),[[2,10975,4629,2,1]])
        self.assertTrue(torch.equal(labels,before))

    def test_wrapper_retains_forward_labels_modes_cache_and_loss_modeloutput(self):
        from transformers.modeling_outputs import Seq2SeqLMOutput
        with np.load(self.prep/'SECTION_MASK_VIEWS.npz',allow_pickle=False) as p:
            ids=p['labels'][p['offsets'][0]:p['offsets'][1]].tolist()
        labels=torch.tensor([ids]);logits=torch.randn(1,len(ids),50265,requires_grad=True)
        captured={}
        class Fake:
            training=True
            config=SimpleNamespace(use_cache=False)
            def __call__(self,**kw):
                captured.update(kw)
                return Seq2SeqLMOutput(loss=F.cross_entropy(logits.flatten(0,1),labels.flatten()),logits=logits)
        model=Fake();original=labels.clone();rng=torch.get_rng_state().clone()
        with patch.object(w,'_INDEX',self.index):out=w.section_forward(model,{'labels':labels},use_cache=False)
        self.assertIs(captured['labels'],labels);self.assertFalse(captured['use_cache'])
        self.assertTrue(torch.equal(labels,original));self.assertIs(out.loss,out['loss'])
        self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertTrue(model.training)
        out.loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_runner_objective_is_used_in_train_and_gate_and_replay_uses_compute_loss(self):
        import inspect
        import run_stage_i_led_large_section_wce_v1 as runner
        from led_large_dynamic_amp_v2 import install_trainer
        train=inspect.getsource(runner.train);gate=inspect.getsource(runner.feasible)
        self.assertIn('output = section_forward(model,inputs',train)
        self.assertIn('loss=section_forward(model,batch',gate)
        self.assertIn('Seq2SeqTrainer.training_step(trainer,trainer.model,inputs)',inspect.getsource(install_trainer))
        self.assertIn('require(resume is None',train) # No hidden continuation/retry.
        parent=(ROOT/'scripts/run_stage_i_led_large_decoder_policy_v2.py').read_text()
        self.assertEqual(inspect.getsource(runner.collator),parent[parent.index('def collator('):parent.index('\ndef checkpoint_verified')].rstrip()+'\n')

    def test_approval_missing_or_false_refuses_before_GPU(self):
        import stage_i_led_large_section_wce_v1 as helper
        with self.assertRaises(ValueError):helper.authorize(None,'feasibility',False)


if __name__=='__main__':unittest.main()
