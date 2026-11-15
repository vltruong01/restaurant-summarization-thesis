"""Offline validation of a proposed numerical amendment. No GPU or model execution."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class AMPCandidateTests(unittest.TestCase):
    def test_only_feasibility_math_change_is_initial_scale(self):
        trees=[]
        for version in (3,4):
            p=ROOT/f'scripts/run_stage_i_led_large_gradient_gate_v{version}.py'
            tree=ast.parse(p.read_text(encoding='utf-8'))
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='feasible')
            calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='GradScaler']
            self.assertEqual(len(calls),1)
            if version==4:
                self.assertEqual(next(k.value.value for k in calls[0].keywords if k.arg=='init_scale'),1024.0)
                calls[0].keywords=[k for k in calls[0].keywords if k.arg!='init_scale']
            trees.append(ast.dump(fn,include_attributes=False))
        self.assertEqual(*trees)

    def test_actual_scaler_initialization_without_cuda(self):
        import torch
        from torch.cuda.amp import grad_scaler
        with patch.object(grad_scaler,'amp_definitely_not_available',return_value=False), \
             patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')):
            scaler=grad_scaler.GradScaler(init_scale=1024.0,enabled=True)
            self.assertEqual(scaler.state_dict(),dict(scale=1024.0,growth_factor=2.0,backoff_factor=0.5,growth_interval=2000,_growth_tracker=0))
            self.assertIsNone(scaler._scale)

    def test_original_all_scientific_config_fields_unchanged(self):
        a=json.loads((ROOT/'configs/stage_i_led_large_gradient_gate_v3.json').read_text())
        b=json.loads((ROOT/'configs/stage_i_led_large_gradient_gate_v4.json').read_text())
        self.assertEqual({k:v for k,v in a.items() if k!='diagnostic_scope'},
                         {k:v for k,v in b.items() if k not in ('diagnostic_scope','amp_scaler')})
        self.assertTrue(b['diagnostic_scope']['scientific_settings_changed'])

    def test_new_approval_required_and_no_deadline_reset(self):
        a=json.loads((ROOT/'outputs/stage_i/led_large_gradient_gate_v4_preparation/GPU_APPROVAL_TEMPLATE_NOT_APPROVED.json').read_text())
        self.assertFalse(a['GPU_execution_approved']);self.assertFalse(a['training_authorized'])
        self.assertEqual(a['experiment'],'LED_LARGE_GRADIENT_GATE_V4')
        self.assertEqual(a['execution_deadline_utc'],'2026-10-21T17:11:48+00:00')

    def test_no_approval_stops_before_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            p=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage_i_led_large_gradient_gate_v4.py'),'feasibility','--output',str(Path(d)/'run')],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('GPU disabled',p.stderr)
            self.assertFalse((Path(d)/'run').exists())

    def test_original_strict_gradient_and_skip_guards_retained(self):
        s=(ROOT/'scripts/run_stage_i_led_large_gradient_gate_v4.py').read_text()
        for text in ('grads = gradient_record(model,expect_later_A=step>=3)',"'AMP skipped update'",'for step in range(1,4):','for j in range(8):',"'Measured GPU headroom below 15%; STOP, no fallback'"):
            self.assertIn(text,s)
        self.assertNotIn('def train(',s)


if __name__=='__main__':unittest.main()
