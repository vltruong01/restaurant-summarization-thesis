"""CPU-only operational adapter tests; no CUDA/model calls."""
import sys, unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_led_large_boundary_simple_gate_v1 import GateSafety

class GateSafetyTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(uuid='GPU-test',name='NVIDIA GeForce RTX 3090',memory_total_MiB=24576,driver='pinned')
        self.simple=Mock()
        self.simple.snapshot.return_value=dict(self.identity,compute_pids=[])
        self.simple.check.return_value=self.identity
        self.g=GateSafety(self.simple,self.identity)
    def test_preload_must_remain_idle(self):
        self.assertEqual(self.g.identity_check(),self.identity)
        self.simple.snapshot.return_value=dict(self.identity,compute_pids=[999999])
        with self.assertRaises(ValueError):self.g.identity_check()
    def test_loaded_and_between_arm_checks_use_stable_context_guard(self):
        self.assertEqual(self.g.after_load(),self.identity)
        self.assertTrue(self.g.model_seen)
        self.g.identity_check();self.assertEqual(self.simple.check.call_count,2)
    def test_loaded_context_conflict_propagates(self):
        self.simple.check.side_effect=ValueError('Foreign or ambiguous process')
        with self.assertRaises(ValueError):self.g.after_load()
    def test_device_drift_rejects(self):
        self.simple.snapshot.return_value=dict(self.identity,uuid='wrong',compute_pids=[])
        with self.assertRaises(ValueError):self.g.identity_check()
    def test_unchanged_science_and_explicit_identity_injection(self):
        import ast, inspect, run_led_large_boundary_simple_gate_v1 as m
        tree=ast.parse(inspect.getsource(m))
        calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='run_pair']
        self.assertEqual(len(calls),1)
        self.assertIn('identity',[x.arg for x in calls[0].keywords])
        self.assertFalse(any(isinstance(x,ast.Attribute) and x.attr in ('generate','backward','step') for x in ast.walk(tree)))
        self.assertNotIn('os.getpid',inspect.getsource(m))

if __name__=='__main__':unittest.main(verbosity=2)
