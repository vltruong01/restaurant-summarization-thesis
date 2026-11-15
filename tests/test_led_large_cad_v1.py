"""Reduced random LED/LoRA only: no original weights, labels or held-out data."""
from pathlib import Path
import sys
import unittest
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
from transformers import LEDConfig, LEDForConditionalGeneration, GenerationConfig
from led_large_cad_v1 import CADSession, contextless_source, mix_logits, SCAFFOLD

torch.set_num_threads(1)


def tiny():
    torch.manual_seed(42)
    c = LEDConfig(vocab_size=31, d_model=16, encoder_layers=12, decoder_layers=2,
                  encoder_attention_heads=2, decoder_attention_heads=2,
                  encoder_ffn_dim=32, decoder_ffn_dim=32, attention_window=[4]*12,
                  max_encoder_position_embeddings=64, max_decoder_position_embeddings=64,
                  pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
                  dropout=.1, attention_dropout=.1)
    return LEDForConditionalGeneration(c).eval()


def batch(ids):
    x=torch.tensor([ids]);m=torch.ones_like(x);g=torch.zeros_like(x);g[:,0]=1
    return dict(input_ids=x, attention_mask=m, global_attention_mask=g)


def gen():
    return GenerationConfig(max_length=18, num_beams=4, num_return_sequences=1,
        do_sample=False, early_stopping=True, use_cache=True, forced_eos_token_id=2,
        decoder_start_token_id=2, eos_token_id=2, pad_token_id=1, bos_token_id=0)


class CADTests(unittest.TestCase):
    def test_scaffold_removes_all_facts(self):
        x=contextless_source('[BUSINESS_INFORMATION]\nname=SECRET\n[REVIEWS]\n[REV_1] nice\n[VISUAL_CONTEXT]\n[IMG_CAP_1] red')
        self.assertEqual(x,SCAFFOLD)
        self.assertNotIn('SECRET',x)

    def test_ambiguous_source_rejected(self):
        with self.assertRaises(ValueError):contextless_source('[REVIEWS]')

    def test_formula_and_alpha0_identity(self):
        a=torch.tensor([[-2.,3.,.5]]);b=torch.tensor([[1.,-1.,2.]])
        self.assertIs(mix_logits(a,b,0),a)
        self.assertTrue(torch.equal(mix_logits(a,b,.3),1.3*a-.3*b))

    def test_nonfinite_and_unlocked_alpha_rejected(self):
        for alpha in (.2,float('nan')):
            with self.assertRaises(ValueError):mix_logits(torch.ones(1),torch.ones(1),alpha)
        with self.assertRaises(ValueError):mix_logits(torch.ones(1),torch.tensor([float('inf')]),.3)

    def test_beam4_alpha0_exact_and_restore(self):
        model=tiny();x=batch([0,6,7,8,9,2]);ctx=batch([0,10,2]);state=torch.get_rng_state()
        attrs=set(model.__dict__);weights={n:p.detach().clone() for n,p in model.named_parameters()}
        with torch.no_grad():
            vanilla=model.generate(**x,generation_config=gen())
            torch.set_rng_state(state)
            with CADSession(model,ctx,0) as s:
                alpha0=model.generate(**x,generation_config=gen())
                self.assertGreater(s.steps,1);self.assertEqual(s.reorders,s.steps)
                self.assertTrue(all(r['alpha0_raw_logits_bit_exact'] for r in s.rows))
        self.assertTrue(torch.equal(vanilla,alpha0));self.assertEqual(set(model.__dict__),attrs)
        self.assertEqual(alpha0[0,0],2);self.assertEqual(alpha0[0,-1],2)
        self.assertTrue(all(torch.equal(weights[n],p) and p.grad is None for n,p in model.named_parameters()))

    def test_nonzero_deterministic_and_two_encoders_rng(self):
        model=tiny();x=batch([0,6,7,8,9,2]);ctx=batch([0,10,2]);state=torch.get_rng_state()
        values=[]
        for _ in range(2):
            torch.set_rng_state(state)
            with torch.no_grad(),CADSession(model,ctx,.3) as s:
                values.append(model.generate(**x,generation_config=gen()))
                self.assertTrue(all(r['branches_finite'] and r['conditional_contextless_prefix_exact'] for r in s.rows))
            actual=torch.get_rng_state();torch.set_rng_state(state)
            for _ in range(24):torch.rand([])
            self.assertTrue(torch.equal(actual,torch.get_rng_state()))
        self.assertTrue(torch.equal(*values))

    def test_cache_matches_contextless_full_prefix(self):
        model=tiny();x=batch([0,6,7,8,9,2]);ctx=batch([0,10,2]);errors=[]
        with torch.no_grad():
            with CADSession(model,ctx,.3) as s:
                original=s._hook
                # Compare the actual cached contextless logits against a separate
                # full-prefix reference at every step, including beam reordering.
                def audit(module,args,kwargs,out):
                    if s.inside:return None
                    s.inside=True
                    try:
                        full=module(encoder_outputs=s.encoder,attention_mask=s.mask,
                            decoder_input_ids=s.prefix,use_cache=False,return_dict=True).logits[:,-1:,:]
                    finally:s.inside=False
                    result=original(module,args,kwargs,out)
                    # Hook's mixed logits use the cached branch. Recover reference mix.
                    s.inside=True
                    try:
                        conditional=module(**kwargs).logits[:,-1:,:]
                    finally:s.inside=False
                    expected=mix_logits(conditional,full,.3)
                    errors.append(float((result.logits-expected).abs().max()))
                    return result
                s.handles[0].remove();s.handles=[model.register_forward_hook(audit,with_kwargs=True)]
                model.generate(**x,generation_config=gen())
        self.assertTrue(errors and max(errors)<1e-6,errors)

    def test_peft_generation_hook_integration(self):
        from peft import get_peft_model,LoraConfig,TaskType
        from unittest.mock import patch
        # Same exclusion of unused bitsandbytes dispatch as the accepted parent
        # loader. This is ordinary FP32 LoRA, never a quantized model test.
        with patch('peft.tuners.lora.model.is_bnb_available',return_value=False), \
             patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
            model=get_peft_model(tiny(),LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=2,lora_alpha=4,
                target_modules=['q_proj','v_proj'],lora_dropout=.1)).eval()
        x=batch([0,6,7,8,9,2]);ctx=batch([0,10,2])
        with torch.no_grad():
            ref=model.generate(**x,generation_config=gen())
            with CADSession(model,ctx,0) as s:
                actual=model.generate(**x,generation_config=gen())
                self.assertGreater(s.steps,1)
        self.assertTrue(torch.equal(ref,actual))

    def test_exception_cleanup(self):
        m=tiny();attrs=set(m.__dict__)
        with self.assertRaisesRegex(RuntimeError,'planned'):
            with torch.no_grad(),CADSession(m,batch([0,10,2]),.3):raise RuntimeError('planned')
        self.assertEqual(attrs,set(m.__dict__));self.assertEqual(len(m._forward_hooks),0)

    def test_no_grad_and_eval_required(self):
        m=tiny()
        with self.assertRaises(ValueError):
            with CADSession(m,batch([0,10,2]),.3):pass
        m.train()
        with self.assertRaises(ValueError):
            with torch.no_grad(),CADSession(m,batch([0,10,2]),.3):pass

    @unittest.skipUnless(os.environ.get('CAD_RUN_CUDA_TESTS') == '1', 'Explicit CUDA technical test only')
    def test_cuda_fp16_alpha0_and_nonzero(self):
        self.assertTrue(torch.cuda.is_available())
        model=tiny().cuda();x={k:v.cuda() for k,v in batch([0,6,7,8,9,2]).items()}
        ctx={k:v.cuda() for k,v in batch([0,10,2]).items()};state=torch.get_rng_state()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            vanilla=model.generate(**x,generation_config=gen())
            torch.set_rng_state(state)
            with CADSession(model,ctx,0):alpha0=model.generate(**x,generation_config=gen())
            self.assertTrue(torch.equal(vanilla,alpha0))
            results=[]
            for _ in range(2):
                torch.set_rng_state(state)
                with CADSession(model,ctx,.3) as s:
                    results.append(model.generate(**x,generation_config=gen()))
                    self.assertTrue(all(r['branches_finite'] for r in s.rows))
            self.assertTrue(torch.equal(*results))


if __name__=='__main__':unittest.main()
