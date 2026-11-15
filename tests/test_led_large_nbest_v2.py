"""CPU-only synthetic regression tests; never loads pretrained weights or generates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import led_large_nbest_rng_v2 as guard
import stage_i_led_large_nbest_v1 as old
import run_stage_i_led_large_nbest_v2 as runner
PARENT = ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
sys.path.insert(0, str(PARENT/'scripts'))
from led_large_dynamic_amp_v2 import capture_rng, restore_rng
from run_stage_i_d050_feasibility_v1 import fingerprint


class Memory:
    def __init__(self, folder, fail=False):
        self.samples=[];self.failure=None;self.folder=folder;self.fail=fail
    def sample(self): self.samples.append(dict(used=10,total=100,reserved=10))
    def summary(self): return dict(headroom_fraction=.9)
    def check(self):
        # Receipt ordering is asserted at the exact point a later guard may fail.
        assert (self.folder/'raw/call.json').is_file()
        assert (self.folder/'calls/call.json').is_file()
        if self.fail: raise ValueError('Synthetic headroom failure')


class RNGGuard(unittest.TestCase):
    def setUp(self): self.original=capture_rng()
    def tearDown(self): restore_rng(self.original)

    def states(self):
        b=capture_rng();b['cuda']=[torch.zeros(8,dtype=torch.uint8)]  # synthetic device state, CPU tensor
        a=copy.deepcopy(b);a['cpu']=guard.expected_cpu_state(b['cpu'],torch)
        return b,a,copy.deepcopy(b)

    def report(self,b,a,r): return guard.rng_report(b,a,r,torch,fingerprint)

    def invoke(self, folder, *, kind='healthy', memory_fail=False, restore=restore_rng):
        attempts=[];returned=[]
        def generate():
            attempts.append(1)
            if kind=='exception': raise RuntimeError('Synthetic model error; no real forward')
            for _ in range(12): torch.rand([])
            if kind=='python': random.random()
            if kind=='numpy': np.random.random()
            n=2 if kind=='count' else 1
            return torch.tensor([[2,20,2]]*n)
        def decode(ids):
            if kind=='decode': raise ValueError('Synthetic decoder error')
            return 'Synthetic useful review text.'
        screen=lambda *a:dict(status='FAIL' if kind=='collapse' else 'PASS_TECHNICAL_ONLY')
        def run():
            return guard.recorded_generation(folder=folder,key='call',generate=generate,decode=decode,
                screen=screen,torch=torch,memory=Memory(folder,memory_fail),capture=capture_rng,
                restore=restore,fingerprint=fingerprint,synchronize=lambda:None,
                metadata=dict(source_sha256='fixture',checkpoint_sha256='fixture'),
                expected_sequences=1,on_returned=returned.append,eval_check=lambda:kind!='mode')
        return run,attempts,returned

    def test_pinned_encoder_source_and_version(self):
        import transformers
        from transformers.models.led import modeling_led
        self.assertEqual(transformers.__version__,'4.51.3')
        self.assertEqual(hashlib.sha256(Path(modeling_led.__file__).read_bytes()).hexdigest(),guard.LED_SOURCE_SHA256)
        src=inspect.getsource(modeling_led.LEDEncoder.forward)
        self.assertLess(src.index('dropout_probability = torch.rand([])'),src.index('if self.training and (dropout_probability'))

    def test_actual_tiny_twelve_layer_LED_encoder_eval_CPU_transition(self):
        from transformers import LEDConfig
        from transformers.models.led.modeling_led import LEDEncoder
        cfg=LEDConfig(vocab_size=32,d_model=8,encoder_layers=12,decoder_layers=1,
            encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=16,
            decoder_ffn_dim=16,attention_window=4,max_encoder_position_embeddings=32,
            encoder_layerdrop=.2,dropout=.1)
        encoder=LEDEncoder(cfg).cpu().eval()
        ids=torch.tensor([[0,10,11,12,13,14,15,2]])
        mask=torch.ones_like(ids);global_mask=torch.zeros_like(ids);global_mask[:,0]=1
        before=capture_rng()
        with torch.no_grad(): result=encoder(input_ids=ids,attention_mask=mask,global_attention_mask=global_mask)
        after=capture_rng()
        self.assertTrue(torch.isfinite(result.last_hidden_state).all().item())
        self.assertFalse(torch.equal(before['cpu'],after['cpu']))
        self.assertTrue(torch.equal(guard.expected_cpu_state(before['cpu'],torch),after['cpu']))
        restore_rng(before)
        self.assertEqual(self.report(before,after,capture_rng())['status'],'PASS')

    def test_prediction_uses_independent_generator(self):
        b=capture_rng();guard.expected_cpu_state(b['cpu'],torch)
        self.assertEqual(guard.component_fingerprints(b,fingerprint),guard.component_fingerprints(capture_rng(),fingerprint))

    def test_expected_CPU_twelve_draws(self):
        b,a,r=self.states();self.assertEqual(self.report(b,a,r)['status'],'PASS')

    def test_zero_CPU_draws_stop(self):
        b,a,r=self.states();a['cpu']=b['cpu'];self.assertFalse(self.report(b,a,r)['expected_transition_matches']['cpu'])

    def test_thirteenth_CPU_draw_stops(self):
        b,a,r=self.states();g=torch.Generator();g.set_state(a['cpu']);torch.rand([],generator=g);a['cpu']=g.get_state()
        self.assertEqual(self.report(b,a,r)['status'],'FAIL')

    def test_unexpected_Python_change(self):
        b,a,r=self.states();random.random();a['python']=random.getstate()
        self.assertFalse(self.report(b,a,r)['expected_transition_matches']['python'])

    def test_unexpected_NumPy_change(self):
        b,a,r=self.states();np.random.random();a['numpy']=np.random.get_state()
        self.assertFalse(self.report(b,a,r)['expected_transition_matches']['numpy'])

    def test_unexpected_CUDA_change_synthetic(self):
        b,a,r=self.states();a['cuda'][0][0]=1
        self.assertFalse(self.report(b,a,r)['expected_transition_matches']['cuda'])

    def test_CUDA_device_count_change_synthetic(self):
        b,a,r=self.states();a['cuda'].append(torch.zeros(8,dtype=torch.uint8))
        self.assertEqual(self.report(b,a,r)['status'],'FAIL')

    def test_inexact_restoration_stops(self):
        b,a,r=self.states();r['cpu']=a['cpu'];self.assertEqual(self.report(b,a,r)['status'],'FAIL')

    def test_real_full_CPU_snapshot_restore(self):
        b=capture_rng();random.random();np.random.random();torch.rand([]);restore_rng(b)
        self.assertEqual(guard.component_fingerprints(b,fingerprint),guard.component_fingerprints(capture_rng(),fingerprint))

    def test_healthy_receipts_and_restoration(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);b=capture_rng();run,attempts,returned=self.invoke(p);rows,elapsed=run()
            self.assertEqual(attempts,[1]);self.assertEqual(returned,[1]);self.assertEqual(rows[0]['token_ids'],[2,20,2])
            self.assertGreaterEqual(elapsed,0);self.assertTrue((p/'calls/call_PASS.json').is_file())
            self.assertEqual(old.read(p/'calls/call.json')['rng']['status'],'PASS')
            self.assertEqual(guard.component_fingerprints(b,fingerprint),guard.component_fingerprints(capture_rng(),fingerprint))

    def check_failed_receipts(self,kind,**kwargs):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);run,attempts,returned=self.invoke(p,kind=kind,**kwargs)
            with self.assertRaises(ValueError):run()
            self.assertEqual(attempts,[1]);self.assertTrue((p/'calls/call_FAILURE.json').is_file())
            receipt=old.read(p/'calls/call.json')
            self.assertGreaterEqual(receipt['seconds'],0);self.assertIn('memory',receipt)
            self.assertTrue((p/'raw/call.json').is_file());self.assertNotIn('retry',list(p.glob('*')))
            return receipt

    def test_Python_guard_failure_flushes_raw_and_restores(self):
        b=capture_rng();r=self.check_failed_receipts('python');self.assertEqual(r['raw_token_ids'],[[2,20,2]])
        self.assertEqual(guard.component_fingerprints(b,fingerprint),guard.component_fingerprints(capture_rng(),fingerprint))

    def test_NumPy_guard_failure_flushes(self): self.check_failed_receipts('numpy')
    def test_CUDA_guard_failure_and_restore_with_synthetic_device_state(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);device=[torch.zeros(8,dtype=torch.uint8)]
            def capture(): return dict(capture_rng(),cuda=copy.deepcopy(device))
            def restore(s):
                restore_rng(dict(s,cuda=[]));device[:]=copy.deepcopy(s['cuda'])
            def generate():
                for _ in range(12):torch.rand([])
                device[0][0]=1
                return torch.tensor([[2,20,2]])
            with self.assertRaisesRegex(ValueError,'Unexpected RNG transition'):
                guard.recorded_generation(folder=p,key='call',generate=generate,decode=lambda s:'Synthetic text',
                    screen=lambda *a:dict(status='PASS_TECHNICAL_ONLY'),torch=torch,memory=Memory(p),
                    capture=capture,restore=restore,fingerprint=fingerprint,synchronize=lambda:None,
                    metadata={},expected_sequences=1,on_returned=lambda n:None,eval_check=lambda:True)
            rng=old.read(p/'calls/call.json')['rng']
            self.assertFalse(rng['expected_transition_matches']['cuda'])
            self.assertTrue(all(rng['exact_restoration_matches'].values()))
    def test_partial_generation_exception_closeout(self):
        r=self.check_failed_receipts('exception');self.assertIsNone(r['raw_returned_sequences']);self.assertIn('generation_traceback',r)
    def test_wrong_return_count_flushes(self): self.assertEqual(self.check_failed_receipts('count')['raw_returned_sequences'],2)
    def test_collapse_flushes(self): self.check_failed_receipts('collapse')
    def test_decode_failure_preserves_ids(self): self.assertEqual(self.check_failed_receipts('decode')['raw_token_ids'],[[2,20,2]])
    def test_memory_guard_has_already_flushed_receipts(self): self.check_failed_receipts('healthy',memory_fail=True)
    def test_model_mode_drift_stops(self): self.check_failed_receipts('mode')
    def test_failed_restore_is_not_suppressed(self): self.check_failed_receipts('healthy',restore=lambda s:None)

    def test_N1_N4_identity_and_padding(self):
        a=dict(text='same',token_ids=[2,20,2]);b=dict(text='same',token_ids=[2,20,2,1])
        self.assertTrue(old.compare_top1(a,b)['single_vs_nbest_content_exact'])
        with self.assertRaises(ValueError):old.compare_top1(a,dict(b,text='different'))

    def test_identity_failure_preserves_both_prior_call_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[]
            for name in ('N1','N4'):
                p=root/name;p.mkdir();run,_,_=self.invoke(p);rows.append(run()[0][0])
            rows[1]['text']='Synthetic different text'
            with self.assertRaises(ValueError):old.compare_top1(*rows)
            self.assertTrue(all((root/n/'calls/call_PASS.json').is_file() for n in ('N1','N4')))

    def test_V1_or_pending_authorization_cannot_launch_V2(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';p.write_text(json.dumps(dict(phases=['training_pilot'],experiment=old.EXPERIMENT,
                GPU_inference_approved=True,researcher_explicit_authorization=True)))
            with self.assertRaises(ValueError):runner.authorization(p,'training_pilot',{},True)
            with self.assertRaises(ValueError):runner.authorization(p,'validation_pair',{},True)

    def test_validation_CLI_is_disabled(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage_i_led_large_nbest_v2.py'),'validation_pair'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('invalid choice',r.stderr)

    def test_scientific_settings_unchanged(self):
        a=old.read(ROOT/'configs/stage_i_led_large_nbest_v1.json');b=old.read(ROOT/'configs/stage_i_led_large_nbest_v2.json')
        for key in ('parent_bindings','runtime_pins','backbone_id','backbone_revision','backbone_sha256','adapter_sha256',
                    'original_checkpoint','generation_override','unchanged_num_beams'):
            self.assertEqual(a[key],b[key],key)
        self.assertEqual(a['budgets']['training_pilot'],b['budgets']['training_pilot'])


if __name__=='__main__':unittest.main(verbosity=2)
