"""Focused CPU-only contracts for the gate-to-training handoff. No model execution."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import stage_i_led_large_full_train_v5 as m


class FullTrainingTests(unittest.TestCase):
    def test_scientific_config_unchanged_except_explicit_amp(self):
        old = m.read(ROOT/'configs/stage_i_led_large_colab_v2.json')
        c = m.read(m.CONFIG)
        self.assertEqual(c, dict(old, amp_scaler=m.AMP, accepted_gate_binding=m.GATE_BINDING,
                                 accepted_gate_report_sha256=m.GATE_SHA, full_training_version=5))
        self.assertFalse(c['training_authorized'])

    def test_real_scaler_public_state_configuration_without_gpu(self):
        import torch
        from torch.cuda.amp import grad_scaler
        with patch.object(grad_scaler, 'amp_definitely_not_available', return_value=False), \
             patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden')):
            scaler = grad_scaler.GradScaler(enabled=True)
            accelerator = SimpleNamespace(scaler=scaler)
            result = m.configure_amp_scaler(accelerator, dict(amp_scaler=m.AMP))
            self.assertIs(accelerator.scaler, scaler)
            self.assertEqual(scaler.state_dict()['scale'], 1024.)
            self.assertTrue(result['same_scaler_object'])
            self.assertIsNone(scaler._scale)
            self.assertEqual(scaler.state_dict()['_growth_tracker'], 0)
            with self.assertRaisesRegex(ValueError, 'defaults differ'):
                m.configure_amp_scaler(accelerator, dict(amp_scaler=m.AMP))

    def test_no_scaler_no_fallback(self):
        with self.assertRaisesRegex(ValueError, 'GradScaler required'):
            m.configure_amp_scaler(SimpleNamespace(scaler=None), dict(amp_scaler=m.AMP))

    def test_gradient_callback_actually_serializes(self):
        gradients = {f'layer{i}.lora_{s}.weight':dict(finite=True,nonzero=True,norm=.001)
                     for i in range(228) for s in ('A','B')}
        import stage_i_paired_pilot
        with tempfile.TemporaryDirectory() as d, patch.object(stage_i_paired_pilot,'gradient_record',return_value=gradients) as fn:
            callback = m.callback_class()(Path(d), {}, None)
            callback.on_pre_optimizer_step(None, SimpleNamespace(global_step=2), None, model=object())
            result = json.loads((Path(d)/'gradient_checks.jsonl').read_text())
            self.assertEqual(result['step'], 3)
            self.assertEqual(result['gradients'], gradients)
            self.assertTrue(fn.call_args.kwargs['expect_later_A'])

    def test_checkpoint_callback_binds_new_runner_and_refuses_missing(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'binding', return_value={'version':5}):
            root = Path(d);cp=root/'checkpoint-250';cp.mkdir()
            callback=m.callback_class()(root,{},None)
            state=SimpleNamespace(global_step=250)
            with self.assertRaisesRegex(ValueError,'Incomplete epoch checkpoint'):
                callback.on_save(None,state,None)
            for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                         'scaler.pt','rng_state.pth','trainer_state.json'):
                (cp/name).write_bytes(b'offline-contract-fixture')
            callback.on_save(None,state,None)
            proof=m.read(cp/'checkpoint_integrity.json')
            self.assertEqual(proof['binding'], {'version':5})
            self.assertEqual(len(proof['files']),7)
            self.assertTrue(all(m.sha(cp/n)==h for n,h in proof['files'].items()))

    def test_no_approval_stops_before_output_or_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'new'
            p=subprocess.run([sys.executable,str(m.RUNNER),'train','--output',str(out),'--gate-output',str(Path(d)/'gate')],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('GPU disabled',p.stderr)
            self.assertFalse(out.exists())

    def test_evaluation_and_training_math_preserved(self):
        old=ast.parse((ROOT/'scripts/run_stage_i_led_large_colab_v2.py').read_text())
        new=ast.parse(m.RUNNER.read_text())
        def fn(tree,name):
            return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('evaluate','optimizer_scheduler'):
            if name=='evaluate':
                self.assertEqual(ast.dump(fn(old,name),include_attributes=False),ast.dump(fn(new,name),include_attributes=False))
            else:
                self.assertIs(m.optimizer_scheduler,m.frozen.optimizer_scheduler)
        newtrainer=next(n for n in ast.walk(fn(new,'train')) if isinstance(n,ast.ClassDef) and n.name=='Trainer')
        oldtrainer=next(n for n in ast.walk(fn(old,'train')) if isinstance(n,ast.ClassDef) and n.name=='Trainer')
        newtrainer.body=[n for n in newtrainer.body if n.name!='create_accelerator_and_postprocess']
        self.assertEqual(ast.dump(newtrainer,include_attributes=False),ast.dump(oldtrainer,include_attributes=False))
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='feasible' for n in new.body))

    def test_obsolete_gate_approval_rejected_without_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json'
            p.write_text(json.dumps(dict(GPU_execution_approved=True,training_authorized=False,
                                        researcher_explicit_authorization='historical gate only')))
            with self.assertRaisesRegex(ValueError,'Full-training approval missing'):
                m.authorize(p,'train',True)


if __name__ == '__main__':
    unittest.main()
