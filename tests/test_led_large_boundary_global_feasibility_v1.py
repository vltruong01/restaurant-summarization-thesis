"""CPU-only scope, routing and fail-closed regression tests. No LED model loads."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_led_large_boundary_global_feasibility_v1 as g


def fake_binding(arm): return {'attention_arm':arm, 'config_sha256':'locked_gate_only'}


class Authorization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'approval.json'
        self.bind = patch.object(g, 'binding', fake_binding); self.bind.start(); self.addCleanup(self.bind.stop)
        now = datetime.now(timezone.utc)
        self.a = dict(experiment=g.EXPERIMENT, GPU_execution_approved=True, approval_source='researcher_message',
            researcher_explicit_authorization='TEST ONLY, NEVER USED FOR GPU',
            bindings={x:fake_binding(x) for x in g.ARMS}, execution_authorized_at_utc=(now-timedelta(minutes=1)).isoformat(),
            execution_deadline_utc=(now+timedelta(hours=1)).isoformat(),
            researcher_absolute_stop_utc=(now+timedelta(hours=1)).isoformat(),
            defense_reserve_at_least_14_days_confirmed=True, **g.expected_contract())
    def save(self): self.path.write_text(json.dumps(self.a), encoding='utf-8')
    def accept(self): self.save(); return g.authorize(self.path, True)
    def test_correct_gate_only_contract(self):
        a = self.accept(); self.assertEqual(a['_approval_sha256'], g.sha(self.path))
        self.assertEqual(a['feasibility_updates_suite_max'], 6)
    def test_template_and_missing_execute_denied(self):
        self.save()
        with self.assertRaises(ValueError): g.authorize(self.path, False)
        self.a['GPU_execution_approved'] = False
        with self.assertRaises(ValueError): self.accept()
    def test_training_evaluation_or_automatic_continuation_denied(self):
        for key, value in [('training_authorized',True), ('evaluation_authorized',True),
                ('conditional_train_after_both_complete_gates_pass',True), ('scopes',['feasibility','train']),
                ('max_training_updates',1250), ('evaluation_outputs_max',48), ('early_sanity_outputs_max',4)]:
            with self.subTest(key=key):
                before = copy.deepcopy(self.a); self.a[key] = value
                with self.assertRaises(ValueError): self.accept()
                self.a = before
    def test_extra_unknown_permission_denied(self):
        self.a['extra_training_permission'] = True
        with self.assertRaises(ValueError): self.accept()
    def test_historical_receipt_cannot_authorize_new_gate(self):
        self.a['experiment'] = 'LED_LARGE_BOUNDARY_GLOBAL_V1'
        with self.assertRaises(ValueError): self.accept()
    def test_gate_receipt_cannot_authorize_old_training_wrapper(self):
        self.save()
        with self.assertRaises(ValueError): g.old.authorize(self.path, True, 'train')
    def test_code_binding_and_memory_policy_locked(self):
        for key, value in [('bindings',{}), ('minimum_memory_headroom',.14), ('feasibility_updates_max',4),
                           ('maximum_runtime_seconds',7201), ('automatic_retry',True),
                           ('amp_scaler',{}), ('dynamic_overflow_policy',{})]:
            with self.subTest(key=key):
                before = copy.deepcopy(self.a); self.a[key] = value
                with self.assertRaises(ValueError): self.accept()
                self.a = before
    def test_boolean_integer_substitution_rejected(self):
        self.a['training_authorized'] = 0
        with self.assertRaises(ValueError): self.accept()
    def test_deadline_48h_expiry_reserve_and_timezone(self):
        now = datetime.now(timezone.utc)
        for change in [dict(execution_deadline_utc=(now-timedelta(seconds=1)).isoformat()),
                dict(execution_deadline_utc=(now+timedelta(hours=49)).isoformat(),
                    researcher_absolute_stop_utc=(now+timedelta(hours=50)).isoformat()),
                dict(defense_reserve_at_least_14_days_confirmed=None),
                dict(execution_authorized_at_utc=now.replace(tzinfo=None).isoformat())]:
            with self.subTest(change=change):
                a = dict(self.a, **change)
                with self.assertRaises(ValueError): g.check_time(a)
    def test_runtime_cap_checked_in_native_hook(self):
        with self.assertRaises(ValueError): g.check_time(dict(self.a, _runtime_stop_monotonic=-1))
    def test_public_cli_has_no_train_or_evaluate(self):
        for mode in ('train','evaluate','resume'):
            with patch.object(sys,'argv',['runner',mode,'--parent-root','.']), patch.object(g,'execute') as call:
                with self.assertRaises(SystemExit): g.main()
                call.assert_not_called()


class Routing(unittest.TestCase):
    def test_new_binding_and_clock_injection_restores_old_functions(self):
        original_b, original_t = g.old.binding, g.old.check_time
        @contextmanager
        def native_route(*args):
            self.assertIs(g.old.binding, g.binding)
            self.assertIs(g.old.check_time, g.check_time)
            yield
        with patch.object(g.old, 'route_parent', native_route):
            with g.route(None,None,None,'boundary',None): pass
        self.assertIs(g.old.binding, original_b); self.assertIs(g.old.check_time, original_t)
    def test_matching_initialization_requirement_in_inherited_pair_guard(self):
        with tempfile.TemporaryDirectory() as t, patch.object(g.old,'binding',fake_binding):
            suite = Path(t); a = {'_approval_sha256':'approval'}
            reports = {x:dict(hardware_profile={'same':True}) for x in g.ARMS}
            h = SimpleNamespace(binding=None, require_gate=lambda out,a:reports[out.name])
            for x in g.ARMS:
                g.write(suite/x/'RESOURCE_GATE.json',dict(status='PASS',binding=fake_binding(x),
                    approval_sha256='approval',headroom_fraction=.15))
                g.write(suite/x/'feasibility/INITIAL_MODEL_BINDINGS.json',dict(initialization={'LoRA':'identical'}))
            g.old.require_both_gates(suite,a,h)
            p = suite/'boundary/feasibility/INITIAL_MODEL_BINDINGS.json'
            p.write_text(json.dumps(dict(initialization={'LoRA':'different'})),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'initialization'): g.old.require_both_gates(suite,a,h)
    def test_foreign_gpu_workload_and_multiple_gpu_rejected(self):
        good = SimpleNamespace(stdout='GPU-test, NVIDIA GeForce RTX 3090, 0000:01:00.0, 24576, 590.0\n')
        with patch.object(g.subprocess,'run',side_effect=[good,SimpleNamespace(stdout='99999999\n')]):
            with self.assertRaisesRegex(ValueError,'Foreign'): g.gpu_identity()
        with patch.object(g.subprocess,'run',return_value=SimpleNamespace(stdout=good.stdout*2)):
            with self.assertRaisesRegex(ValueError,'single-GPU'): g.gpu_identity()


class PairFlow(unittest.TestCase):
    def run_fake(self, failure=None):
        with tempfile.TemporaryDirectory() as t:
            suite = Path(t); calls = []; records = {}
            a = {'_approval_sha256':'approval'}
            h = SimpleNamespace(require_gate=lambda out,a:records[out.name])
            b = SimpleNamespace(write=g.write)
            def feasible(c,a,out):
                calls.append(out.name+':gate')
                if failure == 'first_gate' and out.name == 'bos_only': raise ValueError('gate failed')
                for step in (1,2,3): b.write(out/'feasibility'/f'update_{step}.json',dict(step=step))
                records[out.name] = dict(seconds=120,headroom_fraction=.30,optimizer_updates=3,generation_calls=2)
            b.feasible = feasible
            # Deliberately omit train/evaluate; a call would be a regression.
            def cache(c,a,out,b):
                calls.append(out.name+':cache')
                g.write(out/'RESOURCE_GATE.json',dict(status='PASS',binding=fake_binding(out.name),
                    approval_sha256='approval',headroom_fraction=.149 if failure == 'low_memory' else .30,
                    optimizer_updates=0,new_generate_calls=0,cache_tokens=768,full_cache_batch=4))
            @contextmanager
            def route(*args): yield
            identities = ['same']*5
            if failure == 'identity': identities[3] = 'changed'
            identity = Mock(side_effect=identities)
            with patch.object(g,'check_time'), patch.object(g,'binding',fake_binding), patch.object(g,'route',route), \
                 patch.object(g.old,'cache_stress',cache), patch.object(g.old,'require_both_gates') as both:
                if failure:
                    with self.assertRaises(ValueError): g.run_pair(suite,a,dict.fromkeys(g.ARMS,{}),None,h,b,identity)
                    self.assertFalse((suite/'FEASIBILITY_COMPLETE_STOP_FOR_RESEARCHER_REVIEW.json').exists())
                    both.assert_not_called()
                    return calls
                result = g.run_pair(suite,a,dict.fromkeys(g.ARMS,{}),None,h,b,identity)
                self.assertFalse(result['full_training_authorized'])
                self.assertFalse(result['evaluation_authorized'])
                self.assertEqual(result['discarded_optimizer_updates'],6)
                self.assertEqual(result['TRAIN_technical_generation_calls'],4)
                self.assertEqual(len(list(suite.glob('*/feasibility/update_*.json'))),6)
                self.assertEqual(both.call_count,1)
                return calls
    def test_exact_pair_and_unconditional_stop_after_pass(self):
        self.assertEqual(self.run_fake(),['bos_only:gate','bos_only:cache','boundary:gate','boundary:cache'])
    def test_first_gate_failure_does_not_attempt_cache_or_second_arm(self):
        self.assertEqual(self.run_fake('first_gate'),['bos_only:gate'])
    def test_insufficient_cache_memory_stops_before_second_arm(self):
        self.assertEqual(self.run_fake('low_memory'),['bos_only:gate','bos_only:cache'])
    def test_gpu_change_stops_before_second_arm(self):
        self.assertEqual(self.run_fake('identity'),['bos_only:gate','bos_only:cache'])
    def test_incomplete_timing_is_not_reported_as_measured_throughput(self):
        r = g.estimate_schedule(dict.fromkeys(g.ARMS,dict(seconds=100)),dict.fromkeys(g.ARMS,[]))
        self.assertIsNone(r['boundary']['approximate_1250_update_compute_hours'])
    def test_linux_watchdog_stops_worker_group_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            proc = Mock(pid=12345,returncode=-15)
            proc.wait.side_effect = [subprocess.TimeoutExpired('worker',1),-15]
            fake_os = SimpleNamespace(name='posix',environ=dict(os.environ),killpg=Mock())
            now = datetime.now(timezone.utc)
            a = dict(_approval_sha256='test',maximum_runtime_seconds=7200,
                execution_deadline_utc=(now+timedelta(hours=1)).isoformat())
            approval = root/'a.json'; approval.write_text('{}',encoding='utf-8')
            with patch.object(g,'os',fake_os), patch.object(g,'verify_overlay',return_value={'run_root':'runs/GATE'}), \
                 patch.object(g.subprocess,'Popen',return_value=proc):
                with self.assertRaisesRegex(ValueError,'wall cap'): g.execute(root,approval,a)
            fake_os.killpg.assert_called_once_with(12345,g.signal.SIGTERM)
            self.assertEqual(g.read(root/'runs/GATE/STOP_FAILURE.json')['status'],'WATCHDOG_STOP_NO_RETRY')
    def test_successful_worker_exit_without_complete_receipt_is_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); proc = Mock(pid=12345); proc.wait.return_value=0
            fake_os=SimpleNamespace(name='posix',environ=dict(os.environ))
            a=dict(_approval_sha256='test',maximum_runtime_seconds=7200,
                execution_deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
            approval=root/'a.json';approval.write_text('{}',encoding='utf-8')
            with patch.object(g,'os',fake_os), patch.object(g,'verify_overlay',return_value={'run_root':'runs/GATE'}), \
                 patch.object(g.subprocess,'Popen',return_value=proc):
                with self.assertRaisesRegex(ValueError,'Incomplete'): g.execute(root,approval,a)
            self.assertEqual(g.read(root/'runs/GATE/STOP_FAILURE.json')['status'],'INCOMPLETE_STOP_NO_RETRY')
    def test_existing_namespace_never_reused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'runs/GATE').mkdir(parents=True)
            fake_os=SimpleNamespace(name='posix',environ=dict(os.environ))
            with patch.object(g,'os',fake_os), patch.object(g,'verify_overlay',return_value={'run_root':'runs/GATE'}), \
                 patch.object(g.subprocess,'Popen') as popen:
                with self.assertRaises(FileExistsError):g.execute(root,Path('unused'),{})
                popen.assert_not_called()


if __name__ == '__main__': unittest.main(verbosity=2)
