"""CPU numerical regression. Full pretrained CUDA/FP16 remains untested.

One fixture uses exact untruncated frozen TRAIN IDs with the pinned LED
implementation, all12+12 layers and456 LoRA tensors, but REDUCED hidden width.
That fixture is a software contract test, not pretrained-model feasibility.
"""
from pathlib import Path
import copy
import json
import random
import sys
import types
import unittest
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import led_large_contrastive_sequential_v1 as seq

ROOT=Path(__file__).resolve().parents[1]

class TinyConditional(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(32,8);self.head=torch.nn.Linear(8,32)
        self.config=types.SimpleNamespace(use_cache=True)
    def forward(self,input_ids,attention_mask,global_attention_mask,labels,use_cache=False):
        random.random();np.random.rand();torch.rand(())
        shifted=torch.cat([torch.full_like(labels[:,:1],2),labels[:,:-1]],dim=1).clamp_min(0)
        x=F.dropout(self.embedding(shifted),p=.2,training=self.training)
        logits=self.head(x)
        return types.SimpleNamespace(loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100))

def tiny_batch():
    positive=dict(input_ids=torch.tensor([[0,7,8,2]]),attention_mask=torch.ones(1,4,dtype=torch.long),
        global_attention_mask=torch.tensor([[1,0,0,0]]),labels=torch.tensor([[4,5,9,2,-100]]))
    negative={k:v.clone() for k,v in positive.items()};negative['labels'][0,1]=6
    return positive,negative

def joint(model,p,n):
    sp=-model(**p,use_cache=False).loss.float();sn=-model(**n,use_cache=False).loss.float()
    return -sp+seq.WEIGHT*F.softplus((seq.MARGIN-sp+sn)/seq.TEMPERATURE)

class SequentialCPU(unittest.TestCase):
    def setUp(self):
        random.seed(3);np.random.seed(3);torch.manual_seed(3)
    def test_zero_weight_delegates_existing_ce_exactly(self):
        p,n=tiny_batch();m=TinyConditional();state=seq.capture_rng(torch.device('cpu'))
        a=m(**p,use_cache=False).loss;ga=torch.autograd.grad(a,tuple(m.parameters()))
        seq.restore_rng(state,torch.device('cpu'));b=seq.loss(m,p,n,weight=0.)
        gb=torch.autograd.grad(b,tuple(m.parameters()))
        self.assertTrue(torch.equal(a,b));self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))
    def test_full_joint_matches_streamed_all_rng_components(self):
        p,n=tiny_batch();m=TinyConditional();params=tuple(m.parameters());state=seq.capture_rng(torch.device('cpu'))
        value=joint(m,p,n);expected=torch.autograd.grad(value,params);end=seq.capture_rng(torch.device('cpu'))
        seq.restore_rng(state,torch.device('cpu'));events=[];got_loss=seq.loss(m,p,n,recorder=events.append)
        got=torch.autograd.grad(got_loss,params)
        self.assertTrue(torch.equal(value,got_loss))
        self.assertTrue(all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(expected,got)))
        self.assertTrue(seq.same_rng(seq.capture_rng(torch.device('cpu')),end))
        self.assertEqual(events[-1]['event'],'backward_rng_restored')
    def test_scaled_eight_example_window_equivalence_cpu_not_fp16(self):
        p,n=tiny_batch();m=TinyConditional();params=tuple(m.parameters());state=seq.capture_rng(torch.device('cpu'))
        sums=[torch.zeros_like(x) for x in params];scale=65536.
        for _ in range(8):
            grads=torch.autograd.grad(joint(m,p,n)*scale/8,params)
            sums=[a+b for a,b in zip(sums,grads)]
        end=seq.capture_rng(torch.device('cpu'));seq.restore_rng(state,torch.device('cpu'))
        for _ in range(8):(seq.loss(m,p,n)*scale/8).backward()
        self.assertTrue(all(torch.allclose(a/scale,b.grad/scale,atol=1e-6,rtol=1e-5) for a,b in zip(sums,params)))
        self.assertTrue(seq.same_rng(seq.capture_rng(torch.device('cpu')),end))
    def test_backward_restores_entry_state_even_if_rng_used_between(self):
        p,n=tiny_batch();m=TinyConditional();v=seq.loss(m,p,n)
        torch.rand(7);random.random();np.random.rand(2);entry=seq.capture_rng(torch.device('cpu'))
        v.backward();self.assertTrue(seq.same_rng(seq.capture_rng(torch.device('cpu')),entry))
    def test_source_mutation_stops(self):
        p,n=tiny_batch();n['input_ids'][0,1]=9
        with self.assertRaises(ValueError):seq.loss(TinyConditional(),p,n)
    def test_bos_regression_stops(self):
        p,n=tiny_batch();n['labels'][0,0]=0
        with self.assertRaises(ValueError):seq.loss(TinyConditional(),p,n)
    def test_no_optimizer_or_parameter_mutation(self):
        p,n=tiny_batch();m=TinyConditional();before={k:v.clone() for k,v in m.state_dict().items()}
        seq.loss(m,p,n).backward()
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
    def test_score_mismatch_restores_rng_and_records_failure(self):
        p,n=tiny_batch();m=TinyConditional();events=[];v=seq.loss(m,p,n,recorder=events.append)
        original=seq._check_score
        def fail(*args):raise ValueError('Injected mismatch')
        seq._check_score=fail;entry=seq.capture_rng(torch.device('cpu'))
        try:
            with self.assertRaises(ValueError):v.backward()
        finally:seq._check_score=original
        self.assertTrue(seq.same_rng(seq.capture_rng(torch.device('cpu')),entry))
        self.assertIn('replay_failure',[x['event'] for x in events])

class PinnedLEDCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import LEDConfig,LEDForConditionalGeneration
        from peft import LoraConfig,TaskType,get_peft_model
        from led_large_decoder_label_view_v1 import CorrectedLabelCollator
        cls.previous_threads=torch.get_num_threads();torch.set_num_threads(2)
        torch.manual_seed(17)
        # Exact real TRAIN data, full layer/module coverage; width reduced for safe CPU testing.
        data=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
        pos=json.loads((data/'positives/EzJI5x25U5GgbYBWhu5ZQQ.json').read_text(encoding='utf-8'))
        neg=json.loads((data/'pairs/P10.json').read_text(encoding='utf-8'))
        assert neg['business_id']==pos['business_id']
        collate=CorrectedLabelCollator()
        cls.p=collate([dict(input_ids=pos['input_ids'],labels=pos['corrected_labels'])])
        cls.n=collate([dict(input_ids=pos['input_ids'],labels=neg['negative_labels'])])
        config=LEDConfig(vocab_size=50265,d_model=16,encoder_layers=12,decoder_layers=12,
            encoder_attention_heads=1,decoder_attention_heads=1,encoder_ffn_dim=32,decoder_ffn_dim=32,
            attention_window=[1024]*12,max_encoder_position_embeddings=16384,max_decoder_position_embeddings=1024,
            dropout=.1,attention_dropout=0.,activation_dropout=0.,encoder_layerdrop=0.,decoder_layerdrop=0.,
            pad_token_id=1,bos_token_id=0,eos_token_id=2,decoder_start_token_id=2,use_cache=False)
        base=LEDForConditionalGeneration(config)
        cls.m=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',task_type=TaskType.SEQ_2_SEQ_LM,target_modules='all-linear'))
        for name,param in cls.m.named_parameters():
            if 'lora_B' in name:torch.nn.init.normal_(param,std=.01)
        cls.m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        cls.m.train();cls.params=tuple(p for p in cls.m.parameters() if p.requires_grad)
        assert len(cls.params)==456
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.previous_threads)
    def test_real_untruncated_train_batch_lambda_zero_exact_ce(self):
        self.assertGreater(self.p['input_ids'].shape[1],4000)
        state=seq.capture_rng(torch.device('cpu'))
        a=self.m(**self.p,use_cache=False).loss;ga=torch.autograd.grad(a,self.params)
        seq.restore_rng(state,torch.device('cpu'));b=seq.loss(self.m,self.p,self.n,weight=0.)
        gb=torch.autograd.grad(b,self.params)
        self.assertTrue(torch.equal(a,b));self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))
    def test_pinned_led_dropout_checkpointed_joint_vs_sequential(self):
        state=seq.capture_rng(torch.device('cpu'));a=joint(self.m,self.p,self.n)
        expected=torch.autograd.grad(a,self.params);end=seq.capture_rng(torch.device('cpu'))
        seq.restore_rng(state,torch.device('cpu'));events=[];b=seq.loss(self.m,self.p,self.n,recorder=events.append)
        got=torch.autograd.grad(b,self.params)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(all(torch.isfinite(x).all() for x in got))
        errors=[float((x-y).abs().max()) for x,y in zip(expected,got)]
        self.assertLess(max(errors),1e-5)
        self.assertTrue(all(torch.allclose(x,y,atol=1e-5,rtol=1e-4) for x,y in zip(expected,got)))
        self.assertTrue(seq.same_rng(seq.capture_rng(torch.device('cpu')),end))
        print('PINNED_LED_REDUCED_WIDTH_REAL_BATCH: max_gradient_abs_error=',max(errors),'tensors=',len(got),flush=True)

if __name__=='__main__':unittest.main()
