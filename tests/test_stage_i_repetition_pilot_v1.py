"""Offline authorization and negative controls; approvals exist in memory only.

The subprocess installs an import blocker for Torch/Transformers/PEFT and
executes only the absent-approval path. No model or GPU operations are used.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import stage_i_repetition_pilot_v1 as pilot


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)


class AuthorizationTests(unittest.TestCase):
    digest = 'a' * 64

    def approval(self):
        return {
            'experiment': 'repetition_guard_pilot_v1',
            'GPU_execution_approved': True,
            'approval_source': 'researcher_message',
            'researcher_explicit_authorization': 'SYNTHETIC IN-MEMORY UNIT TEST ONLY; NOT REAL APPROVAL',
            'execution_authorized_at_utc': '2026-10-19T11:00:00+00:00',
            'config_sha256': self.digest,
            'preflight_manifest_sha256': self.digest,
            'runner_sha256': self.digest,
            'generation_calls_max': 24,
            'condition': 'filtered',
            'training_authorized': False,
            'Teacher_calls_authorized': False,
            'held_out_access': False,
            'automatic_retry': False,
            'fallback_authorized': False,
            'RO_U_Shuffled_authorized': False,
            'viability_deadline_utc': '2026-10-22T07:11:22.846767+00:00',
        }

    def invoke(self, value):
        with patch.object(pilot, 'read', return_value=copy.deepcopy(value)), \
             patch.object(pilot, 'sha', return_value=self.digest), \
             patch.object(pilot, 'datetime', FixedDatetime), \
             patch.object(Path, 'is_file', return_value=True):
            return pilot.authorization('SYNTHETIC_IN_MEMORY_ONLY.json')

    def test_missing_approval_rejected_without_read_or_model_import(self):
        with patch.object(pilot, 'read') as reader:
            with self.assertRaisesRegex(ValueError, 'approval required'):
                pilot.authorization(None)
            reader.assert_not_called()

    def test_missing_approval_file_rejected(self):
        with patch.object(Path, 'is_file', return_value=False), patch.object(pilot, 'read') as reader:
            with self.assertRaisesRegex(ValueError, 'approval required'):
                pilot.authorization('DOES_NOT_EXIST.json')
            reader.assert_not_called()

    def test_in_memory_exact_hash_bound_approval_passes_without_writing(self):
        value = self.approval()
        snapshot = copy.deepcopy(value)
        with patch.object(pilot, 'write') as writer:
            actual = self.invoke(value)
            writer.assert_not_called()
        self.assertEqual(actual, snapshot)
        self.assertEqual(value, snapshot)

    def test_old_d060_authorization_is_not_transferable(self):
        value = self.approval()
        value.update(experiment='D-060', decision='D-060', generation_calls_max=48)
        with self.assertRaisesRegex(ValueError, 'not the new guard execution approval'):
            self.invoke(value)

    def test_false_or_non_boolean_approval_rejected(self):
        for approval in (False, None, 1, 'true'):
            value = self.approval()
            value['GPU_execution_approved'] = approval
            with self.subTest(approval=approval), self.assertRaises(ValueError):
                self.invoke(value)

    def test_wrong_or_absent_explicit_approval_source_rejected(self):
        for change in ({'approval_source': 'model_assessment'},
                       {'approval_source': None}, {'researcher_explicit_authorization': ''}):
            value = self.approval()
            value.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Explicit approval source'):
                self.invoke(value)

    def test_each_hash_binding_is_mandatory_and_exact(self):
        for field in ('config_sha256', 'preflight_manifest_sha256', 'runner_sha256'):
            for altered in (None, 'b' * 64):
                value = self.approval()
                value[field] = altered
                with self.subTest(field=field, altered=altered), self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    self.invoke(value)

    def test_generation_scope_cannot_expand_or_change_condition(self):
        for change in ({'generation_calls_max': 25}, {'generation_calls_max': 23},
                       {'condition': 'unfiltered'}, {'condition': 'test'}):
            value = self.approval()
            value.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'scope mismatch'):
                self.invoke(value)

    def test_all_forbidden_operations_need_explicit_false_denial(self):
        fields = ('training_authorized', 'Teacher_calls_authorized', 'held_out_access',
                  'automatic_retry', 'fallback_authorized', 'RO_U_Shuffled_authorized')
        for field in fields:
            for wrong in (True, None, 0, 'false'):
                value = self.approval()
                value[field] = wrong
                with self.subTest(field=field, wrong=wrong), self.assertRaisesRegex(ValueError, 'Forbidden scope'):
                    self.invoke(value)
            value = self.approval()
            del value[field]
            with self.subTest(field=field, wrong='MISSING'), self.assertRaisesRegex(ValueError, 'Forbidden scope'):
                self.invoke(value)

    def test_future_non_utc_and_malformed_approval_timestamp_rejected(self):
        for timestamp in ('2026-10-19T13:00:00+00:00', '2026-10-19T11:00:00+07:00', 'not-a-date'):
            value = self.approval()
            value['execution_authorized_at_utc'] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.invoke(value)

    def test_deadline_cannot_be_reset_extended_or_omitted(self):
        for deadline in ('2026-10-25T15:10:48.941817+00:00', '2026-10-22T07:11:23+00:00', None):
            value = self.approval()
            value['viability_deadline_utc'] = deadline
            with self.subTest(deadline=deadline), self.assertRaisesRegex(ValueError, 'Deadline extended'):
                self.invoke(value)

    def test_expired_clock_fails_without_tolerance_change(self):
        class ExpiredDatetime(FixedDatetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 22, 7, 11, 22, 846767, tzinfo=timezone.utc)
        with patch.object(pilot, 'datetime', ExpiredDatetime):
            with self.assertRaisesRegex(ValueError, 'deadline expired'):
                pilot.deadline()

    def test_execute_without_approval_cannot_reach_heavy_imports(self):
        probe = r'''
import importlib.abc, runpy, sys
class DenyModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft', 'safetensors'}:
            raise AssertionError('FORBIDDEN_MODEL_IMPORT:' + fullname)
sys.meta_path.insert(0, DenyModelImports())
sys.path.insert(0, 'scripts')
sys.argv = ['run_stage_i_repetition_guard_pilot_v1.py', '--execute']
try:
    runpy.run_path('scripts/run_stage_i_repetition_guard_pilot_v1.py', run_name='__main__')
except ValueError as exc:
    assert 'approval required' in str(exc), str(exc)
    assert not any(k in sys.modules for k in ('torch', 'transformers', 'peft', 'safetensors'))
    print('ABSENT_APPROVAL_BLOCKED_BEFORE_HEAVY_IMPORTS')
else:
    raise AssertionError('Absent approval unexpectedly executed')
'''
        result = subprocess.run([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ABSENT_APPROVAL_BLOCKED_BEFORE_HEAVY_IMPORTS', result.stdout)
        self.assertNotIn('FORBIDDEN_MODEL_IMPORT', result.stdout + result.stderr)


class PreflightNegativeControls(unittest.TestCase):
    """All inputs here are synthetic in-memory validation fixtures."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.digest = 'a' * 64
        self.config = {
            'GPU_execution_approved': False, 'training_authorized': False,
            'scope': {'condition': 'filtered', 'businesses': 24, 'random': 20,
                      'targeted': 4, 'generation_calls_max': 24, 'training_updates': 0},
            'historical_integrity_manifest': 'synthetic_historical_manifest.json',
            'runtime_pins': {'synthetic_package': '1.0'},
            'expected_generation_config': 'synthetic_generation.json',
            'generation': {'num_beams': 4},
            'cases': 'synthetic_cases.jsonl', 'tokenizer_directory': 'synthetic_tokenizer',
            'run_root': 'FRESH_SYNTHETIC_NAMESPACE_NOT_A_TRAINING_RUN',
        }
        self.generation = {'num_beams': 4, 'do_sample': False, 'max_length': 768,
                           'max_new_tokens': None, 'renormalize_logits': False,
                           'watermarking_config': None, 'no_repeat_ngram_size': 0}
        self.manifest = {'files': {'synthetic_protected_bytes.bin': self.digest}}
        self.historical = {'files': {'synthetic_historical_bytes.bin': self.digest}}
        self.cases, self.encodes = [], {}
        for n in range(24):
            source = f'SYNTHETIC VALIDATION EVIDENCE {n}'
            target = f'SYNTHETIC COMPARISON TARGET {n}'
            if n == 0:
                source += '\n\n[VISUAL_CONTEXT]\n'
                target += '\n\n[VISUAL_EVIDENCE]\n'
            inputs, labels = [0, 100 + n, 2], [0, 200 + n, 2]
            self.encodes[source], self.encodes[target] = inputs, labels
            self.cases.append({'business_id': f'SYNTHETIC_{n}', 'condition': 'filtered',
                               'split': 'validation', 'subset': 'random' if n < 20 else 'targeted',
                               'source': source, 'target': target, 'input_ids': inputs,
                               'label_ids': labels, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                               'selected_photo_ids': [] if n == 0 else ['SYNTHETIC_PHOTO'],
                               'baseline_prediction': {'token_ids': [55], 'text': 'SYNTHETIC SAVED BASELINE'}})
        self.original = copy.deepcopy(self.cases)
        self.ids = [case['business_id'] for case in self.cases]

    def invoke(self, wrong_digest=False, wrong_runtime=False):
        def reader(path):
            path = Path(path)
            if path == pilot.CONFIG:
                return copy.deepcopy(self.config)
            if path == pilot.PREP / 'PREFLIGHT_MANIFEST.json':
                return copy.deepcopy(self.manifest)
            if path.name == 'synthetic_historical_manifest.json':
                return copy.deepcopy(self.historical)
            if path.name == 'synthetic_generation.json':
                return copy.deepcopy(self.generation)
            if path.name == 'decoding_configurations.json':
                return {'experiments': {'baseline': {'generation_kwargs': {'num_beams': 4}}}}
            if path.name == 'subset_manifest.json':
                return {'execution_business_ids': list(self.ids)}
            raise AssertionError(('Unexpected fixture read', path))
        def row_reader(path):
            return copy.deepcopy(self.cases if Path(path).name == 'synthetic_cases.jsonl' else self.original)
        encodes = self.encodes
        class TokenizerDouble:
            @classmethod
            def from_file(cls, path):
                return cls()
            def encode(self, text):
                return SimpleNamespace(ids=encodes[text])
            def decode(self, ids, skip_special_tokens=True):
                return 'SYNTHETIC SAVED BASELINE'
        with patch.object(pilot, 'ROOT', self.root), patch.object(pilot, 'read', side_effect=reader), \
             patch.object(pilot, 'rows', side_effect=row_reader), \
             patch.object(pilot, 'sha', return_value='b' * 64 if wrong_digest else self.digest), \
             patch.object(pilot.importlib.metadata, 'version', return_value='2.0' if wrong_runtime else '1.0'), \
             patch.dict(sys.modules, {'tokenizers': SimpleNamespace(Tokenizer=TokenizerDouble)}):
            return pilot.preflight()

    def test_full_synthetic_preflight_passes_without_real_model_or_data(self):
        config, manifest, cases, report = self.invoke()
        self.assertEqual(report['cases'], 24)
        self.assertEqual(report['D051_strict_replay'], 'FAIL_UNCHANGED')
        self.assertEqual(report['status'], 'OFFLINE_PASS_NO_MODEL_OR_GPU_EXECUTION')

    def test_frozen_file_and_runtime_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Integrity mismatch'):
            self.invoke(wrong_digest=True)
        with self.assertRaisesRegex(ValueError, 'Runtime drift'):
            self.invoke(wrong_runtime=True)

    def test_scope_generation_and_training_flag_changes_rejected(self):
        for key in ('GPU_execution_approved', 'training_authorized'):
            original = self.config[key]
            self.config[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'not a researcher authorization'):
                self.invoke()
            self.config[key] = original
        self.config['scope']['training_updates'] = 1
        with self.assertRaisesRegex(ValueError, 'Scope changed'):
            self.invoke()

    def test_frozen_generation_semantics_cannot_change(self):
        for key, wrong in (('num_beams', 1), ('max_length', 769), ('max_new_tokens', 768),
                           ('no_repeat_ngram_size', 4), ('renormalize_logits', True)):
            original = self.generation[key]
            self.generation[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.invoke()
            self.generation[key] = original

    def test_source_tokens_and_ids_cannot_change(self):
        self.cases[0]['source_sha256'] = 'b' * 64
        self.original = copy.deepcopy(self.cases)
        with self.assertRaisesRegex(ValueError, 'Source hash differs'):
            self.invoke()

    def test_existing_output_namespace_blocks_retry_overwrite(self):
        (self.root / self.config['run_root']).mkdir()
        with self.assertRaisesRegex(ValueError, 'namespace already exists'):
            self.invoke()


if __name__ == '__main__':
    unittest.main(verbosity=2)
