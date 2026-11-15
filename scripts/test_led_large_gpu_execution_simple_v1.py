"""CPU safety/contract tests. No NVIDIA queries, CUDA or model imports."""
import ast
import os
from pathlib import Path
import tempfile
import unittest
import led_large_gpu_execution_simple_v1 as s
from run_led_large_ce3_lp12_simple_v2 import generation_b


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(instance_id=57562275,GPU_uuid='GPU-fixture')
        self.item=dict(id=57562275,actual_status='running',intended_status='running',
                       num_gpus=1,gpu_name='RTX 3090',gpu_ram=24576,gpu_frac=1/6,machine_id=24033)
        self.gpu='GPU-fixture, NVIDIA GeForce RTX 3090, 24576, 2, 570.86.15\n'

    def test_provider_one_full_GPU_with_host_resource_fraction(self):
        self.assertEqual(s.validate_allocation(self.item,self.config)['gpu_frac'],1/6)

    def test_wrong_provider_instance_or_state_rejected(self):
        for delta in ({'id':1},{'actual_status':'stopped'},{'intended_status':'stopped'}):
            with self.assertRaises(ValueError):s.validate_allocation(dict(self.item,**delta),self.config)

    def test_provider_wrong_GPU_count_type_RAM_or_fraction_rejected(self):
        for delta in ({'num_gpus':2},{'gpu_name':'RTX 4090'},{'gpu_ram':4096},{'gpu_frac':0},{'gpu_frac':None}):
            with self.assertRaises(ValueError):s.validate_allocation(dict(self.item,**delta),self.config)

    def test_idle_GPU_inventory(self):
        x=s.parse_inventory(self.gpu,'',self.config)
        self.assertEqual(x['compute_pids'],[])
        self.assertIsNone(s.validate_processes([],before_model=True))

    def test_existing_workload_rejected(self):
        with self.assertRaisesRegex(ValueError,'Existing GPU'):s.validate_processes([99],before_model=True)

    def test_stable_driver_PID_is_not_compared_with_container_PID(self):
        observed=987654321
        self.assertNotEqual(observed,os.getpid())
        expected=s.validate_processes([observed])
        self.assertEqual(s.validate_processes([observed],expected),expected)

    def test_multiple_missing_changed_processes_rejected(self):
        for observed,expected in [([10,11],None),([],None),([11],frozenset([10]))]:
            with self.assertRaises(ValueError):s.validate_processes(observed,expected)

    def test_malformed_process_inventory_rejected(self):
        for p in ('GPU-other, 123','GPU-fixture, unknown','GPU-fixture, 0','GPU-fixture, -4','GPU-fixture, 1, extra','GPU-fixture, 1\nGPU-fixture, 1'):
            with self.assertRaises(ValueError):s.parse_inventory(self.gpu,p,self.config)

    def test_wrong_visibility_UUID_or_VRAM_rejected(self):
        for g in ('',self.gpu+self.gpu,self.gpu.replace('GPU-fixture','GPU-other'),self.gpu.replace('24576','4096')):
            with self.assertRaises(ValueError):s.parse_inventory(g,'',self.config)

    def test_only_scientific_change_is_length_penalty(self):
        original=dict(num_beams=4,length_penalty=1.0,max_length=768,do_sample=False,num_return_sequences=1,use_cache=True,forced_eos_token_id=2)
        result=generation_b(original)
        self.assertEqual([k for k in original if original[k]!=result[k]],['length_penalty'])
        self.assertEqual(result['length_penalty'],1.2)
        self.assertEqual(original['length_penalty'],1.0)

    @unittest.skipUnless(os.name=='posix','flock is verified on execution Linux, not Windows')
    def test_common_lock_rejects_concurrent_launch_and_releases(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'run.lock';fd=s.acquire_run_lock(path)
            try:
                with self.assertRaisesRegex(ValueError,'Another cooperating'):s.acquire_run_lock(path)
            finally:os.close(fd)
            second=s.acquire_run_lock(path);os.close(second)

    def test_new_runner_has_no_hidden_raw_PID_gate_or_extra_model_call(self):
        source=Path(__file__).with_name('run_led_large_ce3_lp12_simple_v2.py').read_text()
        ast.parse(source)
        self.assertNotIn('guard.gpu_identity',source)
        self.assertNotIn('run_led_large_boundary_global_feasibility_v1',source)
        self.assertEqual(source.count('model.generate('),1)
        self.assertNotIn('.backward(',source)
        self.assertIn('SystemExit(run())',source)


if __name__=='__main__':unittest.main(verbosity=2)
