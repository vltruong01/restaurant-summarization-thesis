"""Portable CPU-only checks of the fresh3090 handoff. Never initialize CUDA."""
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
sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_led_large_vast_fresh_v1 as m


class FreshVastTests(unittest.TestCase):
    def test_science_matches_colab_v5(self):
        old=m.read(ROOT/'configs/stage_i_led_large_full_train_v5.json')
        new=m.read(m.CONFIG);new.pop('vast_fresh')
        self.assertEqual(new,old)
        self.assertEqual(new['training'],dict(businesses=2000,epochs=5,steps=1250,steps_per_epoch=250,warmup_steps=63))
        self.assertEqual(new['expected_trainable_parameters'],4915200)
        self.assertEqual(new['expected_modules'],228)

    def test_dataset_order_and_frozen_functions_reused(self):
        self.assertIs(m.load_model,m.frozen.load_model)
        self.assertIs(m.optimizer_scheduler,m.frozen.optimizer_scheduler)
        self.assertIs(m.generation,m.frozen.generation)
        self.assertIs(m.training_arguments,m.frozen.training_arguments)

    def test_training_and_decoding_preserve_original_code(self):
        old=ast.parse((ROOT/'scripts/run_stage_i_led_large_full_train_v5.py').read_text())
        new=ast.parse(m.RUNNER.read_text())
        def fn(t,n): return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
        for name in ('train','evaluate'):
            a=fn(old,name);b=fn(new,name)
            # Ignore only explicit hardware guard and provenance field/path naming.
            txt='\n'.join(line for line in ast.unparse(b).splitlines() if line.strip()!='check_required_gpu(torch)')
            txt=txt.replace('ACCEPTED_VAST_GATE.json','ACCEPTED_V4_GATE.json').replace('gate_output=str(out)',"gate_output=a['_gate_output']")
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(ast.parse(txt).body[0],include_attributes=False))

    def test_gate_retains_original_numerical_checks(self):
        old=ast.parse((ROOT/'scripts/run_stage_i_led_large_gradient_gate_v4.py').read_text())
        new=ast.parse(m.RUNNER.read_text())
        def fn(t):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='feasible')
        clean='\n'.join(line for line in ast.unparse(fn(new)).splitlines() if line.strip()!='check_required_gpu(torch)')
        self.assertEqual(ast.dump(fn(old),include_attributes=False),ast.dump(ast.parse(clean).body[0],include_attributes=False))

    def test_hardware_mismatch_refused(self):
        import run_stage_i_led_large_vast_fresh_v1 as r
        cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,
            get_device_name=lambda:'Tesla T4',get_device_capability=lambda:(7,5),
            get_device_properties=lambda _:SimpleNamespace(total_memory=24*1024**3))
        with self.assertRaisesRegex(ValueError,'hardware differs'):r.check_required_gpu(SimpleNamespace(cuda=cuda))
        cuda.get_device_name=lambda:'NVIDIA GeForce RTX 3090';cuda.get_device_capability=lambda:(8,6)
        r.check_required_gpu(SimpleNamespace(cuda=cuda))
        cuda.get_device_properties=lambda _:SimpleNamespace(total_memory=6*1024**3)
        with self.assertRaisesRegex(ValueError,'memory missing'):r.check_required_gpu(SimpleNamespace(cuda=cuda))

    def test_no_approval_refused(self):
        with self.assertRaisesRegex(ValueError,'GPU disabled'):m.authorize(None,'train',False)

    def test_old_approval_and_tampering_refused(self):
        a=m.read(ROOT/'VAST_RESEARCHER_AUTHORIZATION.json')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json'
            for key,value,pattern in [('experiment','LED_LARGE_F_FULL_TRAIN_V5','provenance'),
                ('runner_sha256','0'*64,'hash mismatch'),('execution_deadline_utc','2099-01-01T00:00:00+00:00','deadline'),
                ('held_out_access',True,'Additional scope'),('max_training_updates',1300,'budget')]:
                p.write_text(json.dumps(dict(a,**{key:value})))
                with self.assertRaisesRegex(ValueError,pattern):m.authorize(p,'train',True)

    def test_deadline_not_reset(self):
        from datetime import datetime,timezone
        a=m.read(ROOT/'VAST_RESEARCHER_AUTHORIZATION.json')
        self.assertEqual(a['execution_deadline_utc'],'2026-10-21T17:11:48+00:00')
        with self.assertRaisesRegex(ValueError,'expired'):
            m.check_deadline(a,datetime(2026,10,22,tzinfo=timezone.utc))

    def test_real_amp_state_cpu_only(self):
        import torch
        from torch.cuda.amp import grad_scaler
        with patch.object(grad_scaler,'amp_definitely_not_available',return_value=False),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')):
            scaler=grad_scaler.GradScaler(enabled=True);acc=SimpleNamespace(scaler=scaler)
            result=m.configure_amp_scaler(acc,dict(amp_scaler=m.AMP))
            self.assertEqual(scaler.state_dict()['scale'],1024.)
            self.assertIs(acc.scaler,scaler);self.assertTrue(result['same_scaler_object'])

    def test_gradient_callback_actual_json_serialization(self):
        import stage_i_paired_pilot
        grads={f'module{i}.lora_{k}.weight':dict(finite=True,nonzero=True,norm=.01) for i in range(228) for k in ('A','B')}
        with tempfile.TemporaryDirectory() as d,patch.object(stage_i_paired_pilot,'gradient_record',return_value=grads):
            cb=m.callback_class()(Path(d),{},None)
            cb.on_pre_optimizer_step(None,SimpleNamespace(global_step=2),None,model=object())
            row=json.loads((Path(d)/'gradient_checks.jsonl').read_text())
            self.assertEqual(row['step'],3);self.assertEqual(len(row['gradients']),456)

    def test_checkpoint_callback_new_binding(self):
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)/'checkpoint-250';cp.mkdir();cb=m.callback_class()(Path(d),{},None)
            state=SimpleNamespace(global_step=250)
            with self.assertRaisesRegex(ValueError,'Incomplete'):cb.on_save(None,state,None)
            for n in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'):
                (cp/n).write_bytes(b'cpu-contract-fixture-not-real-checkpoint')
            cb.on_save(None,state,None)
            self.assertEqual(m.read(cp/'checkpoint_integrity.json')['binding'],m.binding())

    def test_no_resume_argument_or_automatic_retry(self):
        tree=ast.parse(m.RUNNER.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        text=ast.unparse(main)
        self.assertNotIn('--resume',text);self.assertIn('train(c, a, out)',text)
        shell=(ROOT/'RUN_FRESH_F.sh').read_text()
        self.assertIn('set -euo pipefail',shell);self.assertIn('test ! -e "$OUT"',shell)
        self.assertNotIn('while ',shell)

    def test_gate_never_reuses_T4_report(self):
        import inspect
        src=inspect.getsource(m.require_gate)
        self.assertNotIn('GATE_SHA',src);self.assertIn("binding()",src)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'feasibility';p.mkdir()
            (p/'FAILURE_STOP_NO_RETRY.json').write_text('{}')
            (p/'REPORT.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Failed gate'):m.require_gate(Path(d),{})


if __name__=='__main__':unittest.main()
