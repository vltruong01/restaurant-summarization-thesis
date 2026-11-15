"""Small CPU recovery-contract fixtures, no LED model or research optimizer update."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import led_large_dynamic_amp_v2 as amp
import run_stage_i_led_large_section_wce_recovery_v1 as recovery

class RecoveryTests(unittest.TestCase):
    def test_extended_scale_can_recover_below_256_without_optimizer_update(self):
        model=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(456)])
        for p in model:p.grad=torch.ones_like(p)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        class Scaler:
            scale=256.
            def get_scale(self):return self.scale
        scaler=Scaler();events=[]
        r=amp.Recovery(Path('unused'),1250,lambda:None,lambda *a:None,lambda p,x:events.append(x))
        r.successful_updates=906;r.overflow_skips=8;r.rng=amp.capture_rng()
        def evidence(model):return dict(missing=[],nonfinite=['fixture'] if scaler.scale>=256 else [])
        def skip():scaler.scale/=2
        def repeat():
            for p in model:p.grad=torch.ones_like(p)
        policy=dict(amp.POLICY,max_overflow_skips=16,minimum_scale=1.)
        with patch.object(amp,'POLICY',policy),patch('led_large_gradient_evidence_v3.inspect_gradients',evidence),patch('led_large_gradient_evidence_v3.snapshot_raw_gradients'):
            r.finish(model,optimizer,scaler,lambda:None,skip,repeat)
        self.assertEqual(scaler.scale,128.);self.assertEqual(r.overflow_skips,9)
        self.assertTrue(all(torch.equal(p.detach(),torch.ones(1)) for p in model))
        self.assertEqual(r.successful_updates,906);self.assertTrue(r.pending)

    def test_scale_floor_still_stops_without_fallback(self):
        model=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(456)])
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        scaler=type('Scaler',(),{'get_scale':lambda self:1.})()
        r=amp.Recovery(Path('unused'),1250,lambda:None,lambda *a:None,lambda *a:None)
        r.overflow_skips=16
        with patch.object(amp,'POLICY',dict(amp.POLICY,max_overflow_skips=16,minimum_scale=1.)),patch('led_large_gradient_evidence_v3.inspect_gradients',return_value=dict(missing=[],nonfinite=['fixture'])),patch('led_large_gradient_evidence_v3.snapshot_raw_gradients'):
            with self.assertRaisesRegex(ValueError,'limit reached'):r.finish(model,optimizer,scaler,lambda:None,lambda:self.fail('must not skip again'),lambda:None)

    def test_missing_gradients_are_not_treated_as_overflow(self):
        model=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(456)])
        r=amp.Recovery(Path('unused'),1250,lambda:None,lambda *a:None,lambda *a:None)
        with patch('led_large_gradient_evidence_v3.inspect_gradients',return_value=dict(missing=['fixture'],nonfinite=[])):
            with self.assertRaisesRegex(ValueError,'Missing adapter'):r.finish(model,None,None,lambda:None,lambda:None,lambda:None)

    def test_active_log_prefix_discards_only_uncheckpointed_suffix(self):
        raw=b'{"step":749}\n{"step":750}\n{"step":751}\n'
        self.assertEqual(recovery.prefix_bytes(raw,750),b'{"step":749}\n{"step":750}\n')
        self.assertTrue(raw.endswith(b'{"step":751}\n'))

    def test_train_extension_keeps_integrity_and_progress_guards(self):
        train,_=recovery.installed_train_function()
        self.assertIn('Only the authorized checkpoint750 recovery is allowed',train.__code__.co_consts)
        import inspect
        source=inspect.getsource(recovery.parent.train)
        self.assertIn('Uncheckpointed updates exist; rollback/repeated optimization NOT authorized',source)
        self.assertIn('resume_checkpoint_integrity_sha256',source)
        self.assertIn('_load_rng_state',source)

    def test_checkpoint_save_signature_is_pinned(self):
        from transformers import Trainer
        import inspect
        self.assertEqual(list(inspect.signature(Trainer._save_checkpoint).parameters),['self','model','trial'])

if __name__=='__main__':unittest.main()
