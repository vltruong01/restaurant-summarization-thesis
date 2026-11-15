"""CPU-only diagnostic regression tests. No LED forward, backward or GPU operation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from led_large_gradient_evidence_v3 import finite_value, inspect_gradients, snapshot_raw_gradients


class Parameters:
    def __init__(self):
        self.a=torch.nn.Parameter(torch.ones(2,3))
        self.b=torch.nn.Parameter(torch.ones(3,2))
        self.a.grad=torch.zeros_like(self.a)
        self.b.grad=torch.ones_like(self.b)
    def named_parameters(self):
        yield 'test.lora_A.default.weight',self.a
        yield 'test.lora_B.default.weight',self.b


class EvidenceTests(unittest.TestCase):
    def test_zero_A_is_finite_not_missing(self):
        r=inspect_gradients(Parameters(),detailed=True)
        self.assertEqual(r['missing'],[]);self.assertEqual(r['nonfinite'],[])
        self.assertEqual(r['tensors']['test.lora_A.default.weight']['finite_nonzero'],0)

    def test_missing_is_distinguished(self):
        m=Parameters();m.a.grad=None;r=inspect_gradients(m)
        self.assertEqual(r['missing'],['test.lora_A.default.weight']);self.assertEqual(r['nonfinite'],[])

    def test_nan_and_signed_infinities_counted(self):
        m=Parameters();m.a.grad.flatten()[:3]=torch.tensor([float('nan'),float('inf'),-float('inf')])
        r=inspect_gradients(m,detailed=True);v=r['tensors']['test.lora_A.default.weight']
        self.assertEqual((v['nan'],v['positive_inf'],v['negative_inf'],v['finite']),(1,1,1,3))
        json.dumps(r,allow_nan=False)

    def test_all_bad_tensors_reported_before_assertion(self):
        m=Parameters();m.a.grad.fill_(float('nan'));m.b.grad=None
        r=inspect_gradients(m)
        self.assertEqual(len(r['missing']),1);self.assertEqual(len(r['nonfinite']),1)

    def test_no_grad_parameter_or_rng_mutation(self):
        m=Parameters();before=[p.detach().clone() for _,p in m.named_parameters()]
        grad=[p.grad.clone() for _,p in m.named_parameters()];rng=torch.get_rng_state().clone()
        inspect_gradients(m,detailed=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for i,(_,p) in enumerate(m.named_parameters()):
            self.assertTrue(torch.equal(before[i],p));self.assertTrue(torch.equal(grad[i],p.grad))

    def test_no_finite_values_json_safe(self):
        m=Parameters();m.a.grad.fill_(float('nan'))
        r=inspect_gradients(m,detailed=True)
        self.assertIsNone(r['tensors']['test.lora_A.default.weight']['finite_max_abs'])
        json.dumps(r,allow_nan=False)

    def test_raw_snapshot_preserves_values_and_exclusive_create(self):
        with tempfile.TemporaryDirectory() as d:
            m=Parameters();m.a.grad[0,0]=float('inf');p=Path(d)/'raw.pt'
            snapshot_raw_gradients(m,p);loaded=torch.load(p,map_location='cpu',weights_only=True)
            self.assertTrue(torch.equal(loaded['test.lora_A.default.weight'],m.a.grad))
            with self.assertRaises(FileExistsError):snapshot_raw_gradients(m,p)

    def test_nonfinite_scalar_serialization(self):
        self.assertEqual(finite_value(float('nan')),'nan');self.assertEqual(finite_value(float('inf')),'inf')

    def test_original_gradient_gate_retained(self):
        from stage_i_paired_pilot import gradient_record
        m=Parameters();self.assertEqual(len(gradient_record(m)),2)
        m.a.grad=None
        with self.assertRaisesRegex(ValueError,'Missing/nonfinite'):gradient_record(m)
        m.a.grad=torch.full_like(m.a,float('inf'))
        with self.assertRaisesRegex(ValueError,'Missing/nonfinite'):gradient_record(m)

    def test_scientific_config_identical(self):
        old=json.loads((ROOT/'configs/stage_i_led_large_colab_v2.json').read_text())
        new=json.loads((ROOT/'configs/stage_i_led_large_gradient_gate_v3.json').read_text())
        self.assertEqual({k:v for k,v in new.items() if k!='diagnostic_scope'},old)

    def test_gate_algorithm_statements_preserved(self):
        text=(ROOT/'scripts/run_stage_i_led_large_gradient_gate_v3.py').read_text()
        compile(text,'<v3>','exec')
        for s in ('for step in range(1,4):','for j in range(8):',"torch.autocast('cuda',dtype=torch.float16)",
                  'torch.cuda.amp.GradScaler(enabled=True)','scaler.scale(loss/8).backward()',
                  'grads = gradient_record(model,expect_later_A=step>=3)',
                  "scaler.step(opt);scaler.update();sched.step()",'range(2)',
                  'headroom_fraction','restore_feas_checkpoint'):
            self.assertIn(s,text)
        self.assertLess(text.index("write(p/f'UNSCALED_GRADIENTS_step{step}.json'"),text.index('grads = gradient_record'))
        self.assertNotIn('def train(',text);self.assertNotIn('def evaluate(',text)

    def test_no_approval_stops_before_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            p=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage_i_led_large_gradient_gate_v3.py'),
                              'feasibility','--output',str(Path(d)/'run')],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('GPU disabled',p.stderr)
            self.assertFalse((Path(d)/'run').exists())

    def test_unapproved_template_and_preserved_deadline(self):
        a=json.loads((ROOT/'outputs/stage_i/led_large_gradient_gate_v3/GPU_APPROVAL_TEMPLATE_NOT_APPROVED.json').read_text())
        self.assertFalse(a['GPU_execution_approved']);self.assertFalse(a['training_authorized'])
        self.assertEqual(a['scopes'],['feasibility'])
        self.assertEqual(a['execution_deadline_utc'],'2026-10-21T17:11:48+00:00')


if __name__=='__main__':unittest.main()
