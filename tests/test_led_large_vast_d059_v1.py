"""Focused offline tests for D059 method binding, GPU health gate and error preservation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_led_large_vast_d059_v1 as m
import run_stage_i_led_large_vast_d059_v1 as runner
from check_led_large_vast_host_v1 import inspect_host

class MethodAndScope(unittest.TestCase):
    def test_D059_method_mapping_and_large_scope(self):
        c=m.read(m.CONFIG)
        self.assertTrue(m.validate_d059_reference(c))
        self.assertEqual(c['expected_modules'],228)
        self.assertEqual(c['expected_trainable_parameters'],4915200)
        self.assertEqual(c['training'],dict(businesses=2000,epochs=5,steps=1250,steps_per_epoch=250,warmup_steps=63))
        self.assertEqual(c['evaluation_epochs'],[3,5])
        self.assertEqual(c['generation']['num_beams'],4)
        self.assertEqual(c['amp_scaler']['init_scale'],65536.)

    def test_changed_LR_is_rejected(self):
        c=m.read(m.CONFIG);c['optimization']['learning_rate']=1e-3
        with self.assertRaisesRegex(ValueError,'learning_rate|optimization'):
            m.validate_d059_reference(c)

    def test_new_time_policy_cannot_authorize_old_run_or_unbounded_steps(self):
        a=dict(experiment=m.EXPERIMENT,execution_window=m.EXECUTION_WINDOW,
               execution_deadline_utc=None,max_training_updates=1250,evaluation_outputs_max=48)
        self.assertIsNone(m.check_deadline(a))
        with self.assertRaisesRegex(ValueError,'scope'):
            m.check_deadline(dict(a,experiment='LED_LARGE_VAST_DYNAMIC_65536_V2'))
        with self.assertRaisesRegex(ValueError,'ceilings'):
            m.check_deadline(dict(a,max_training_updates=1500))

    def test_authorization_hash_and_ROU_guards_remain(self):
        b=dict(config_sha256='config_fixture',runner_sha256='runner_fixture',preflight_manifest_sha256='preflight_fixture')
        a=dict(experiment=m.EXPERIMENT,execution_window=m.EXECUTION_WINDOW,
            execution_deadline_utc=None,max_training_updates=1250,evaluation_outputs_max=48,
            GPU_execution_approved=True,training_authorized=True,approval_source='researcher_message',
            researcher_explicit_authorization='CPU fixture only',scopes=['feasibility','train','evaluate'],
            dynamic_overflow_policy=m.POLICY,feasibility_updates_max=3,feasibility_microbatches_max=88,
            nominal_feasibility_microbatches=24,technical_generation_calls_max=2,amp_scaler=m.AMP,
            held_out_access=False,Teacher_calls=False,RO_U_training_authorized=False,
            conditional_train_after_feasibility_pass=True,**b)
        with tempfile.TemporaryDirectory() as d,patch.object(m,'binding',return_value=b):
            p=Path(d)/'fixture.json';p.write_text(json.dumps(a))
            self.assertIn('_approval_sha256',m.authorize(p,'train',True))
            p.write_text(json.dumps(dict(a,runner_sha256='changed')))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):m.authorize(p,'train',True)
            p.write_text(json.dumps(dict(a,RO_U_training_authorized=True)))
            with self.assertRaisesRegex(ValueError,'scope forbidden'):m.authorize(p,'train',True)

class ErrorPreservation(unittest.TestCase):
    def meter(self):
        obj=runner.MemoryMeter(SimpleNamespace())
        obj.thread=SimpleNamespace(join=lambda **kwargs:None)
        def failed_sample():raise RuntimeError('CUDA mock query failed')
        obj.sample=failed_sample
        return obj

    def test_cleanup_failure_keeps_primary_exception(self):
        mm=self.meter()
        self.assertFalse(mm.__exit__(AssertionError,AssertionError('attention fixture'),None))
        self.assertIn('CUDA mock query failed',mm.exit_failure)

    def test_cleanup_failure_without_primary_still_fails(self):
        with self.assertRaisesRegex(RuntimeError,'CUDA mock query failed'):
            self.meter().__exit__(None,None,None)

    def test_background_monitor_failure_cannot_pass(self):
        mm=self.meter();mm.failure='monitor failed'
        with self.assertRaisesRegex(ValueError,'monitor failed'):mm.__exit__(None,None,None)

class HostReadiness(unittest.TestCase):
    def run_queries(self,device_error=False,active=False,name='NVIDIA GeForce RTX 3090'):
        responses=[('GPU 0: fixture',255 if device_error else 0),
            (name+', GPU-FIXTURE, 0000:42:00.0, 580.173.02, 24576\n',0),
            ('GPU-FIXTURE, 4789, python\n' if active else '',0)]
        self.calls=[]
        def run(args,**kwargs):
            self.calls.append(args);text,code=responses[len(self.calls)-1]
            return subprocess.CompletedProcess(args,code,stdout=text,stderr='')
        return inspect_host(run)

    def test_healthy_idle_single3090(self):
        self.assertTrue(self.run_queries()['status'].startswith('PASS_'))

    def test_unavailable_GPU_blocks_before_other_queries(self):
        self.assertEqual(self.run_queries(device_error=True)['status'],'FAIL')
        self.assertEqual(len(self.calls),1)

    def test_remaining_compute_process_blocks(self):
        result=self.run_queries(active=True)
        self.assertEqual(result['status'],'FAIL');self.assertIn('compute processes',result['error'])

    def test_wrong_GPU_blocks(self):
        self.assertEqual(self.run_queries(name='Different GPU')['status'],'FAIL')

if __name__=='__main__':unittest.main()
