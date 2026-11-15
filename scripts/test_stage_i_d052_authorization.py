"""CPU/file-only denial tests for D-052; never load a model or call CUDA."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stage_i_d052_authorization as gate


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root/'configs/original.json'
        self.prep = self.root/'prep'
        self.wrapper = self.root/'wrapper.py'
        self.request = self.root/'request.txt'
        self.approval_path = self.root/'approval.json'
        self.config = {'condition': 'filtered', 'run_root': 'runs',
                       'training': {'businesses': 512, 'epochs': 2, 'steps': 128,
                                    'steps_per_epoch': 64, 'warmup_steps': 7},
                       'optimization': {'physical_batch_size': 1, 'gradient_accumulation_steps': 8}}
        for path in [self.config_path, self.wrapper, self.request,
                     self.root/gate.DECISION, self.prep/'PREPARATION_MANIFEST.json',
                     self.prep/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md',
                     self.root/'scripts/run_stage_i_paired_lora_pilot_v1.py',
                     self.root/'scripts/stage_i_paired_pilot.py']:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('immutable fixture '+path.name, encoding='utf-8')
        self.put(self.prep/'pilot_manifest.json', {
            'training_business_ids': [f't{i}' for i in range(512)],
            'validation_loss_business_ids': [f'v{i}' for i in range(200)]})
        for name, value in [('ROOT', self.root), ('CONFIG', self.config_path), ('REQUEST', self.request)]:
            p = patch.object(gate, name, value); p.start(); self.addCleanup(p.stop)
        check_patch = patch.object(gate, 'checked_preparation', return_value=(self.config, self.prep, {}))
        self.check = check_patch.start(); self.addCleanup(check_patch.stop)
        self.put(self.root/gate.EVIDENCE['d050_a'], {
            'status': 'TECHNICAL_FEASIBILITY_PASS', 'arm': gate.ARMS[0], 'replay_allclose': True,
            'replay_tolerance': {'rtol': 1e-5, 'atol': 1e-6},
            'binding': gate._original_binding(self.config_path, self.prep, gate.ARMS[0])})
        self.put(self.root/gate.EVIDENCE['d050_b'], {
            'status': 'AFFECTED_EXECUTION_STOPPED',
            'error': {'message': 'Checkpoint/RNG replay not numerically reproducible'},
            'config_sha256': gate.sha(self.config_path)})
        self.put(self.root/gate.EVIDENCE['d051_readiness'], {
            'assessment': 'CONDITIONAL', 'strict_replay_gate': 'FAIL_UNCHANGED',
            'criterion_amendment_made': False, 'long_run_harmlessness_proven': False,
            'strict_original_tolerance': {'rtol': 1e-5, 'atol': 1e-6}})
        self.put(self.root/gate.EVIDENCE['d051_strict'], {
            'outcome': 'STRICT_REPLAY_TOLERANCE_FAILURE', 'rtol': 1e-5, 'atol': 1e-6,
            'tolerances_changed': False})
        self.put(self.root/gate.EVIDENCE['d051_comparison'], {
            'overall': {'numel': 589824, 'failing_elements': 18, 'allclose': False,
                        'all_finite': True, 'max_abs': 3.723427653312683e-6},
            'rtol': 1e-5, 'atol': 1e-6})
        self.put(self.root/gate.EVIDENCE['d051_binding'],
                 gate._original_binding(self.config_path, self.prep, gate.ARMS[1]))
        self.approval = {
            'decision': 'D-052', 'approved': True, 'scopes': ['train'], 'arms': list(gate.ARMS),
            'scientific_scope': deepcopy(gate.SCOPE),
            'conditional_reproducibility_acceptance': deepcopy(gate.ACCEPTANCE),
            'config_sha256': gate.sha(self.config_path),
            'preparation_manifest_sha256': gate.sha(self.prep/'PREPARATION_MANIFEST.json'),
            'protocol_sha256': gate.sha(self.prep/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),
            'wrapper_sha256': gate.sha(self.wrapper),
            'authorization_module_sha256': gate.sha(Path(gate.__file__)),
            'user_request': {'path': str(self.request), 'sha256': gate.sha(self.request)},
            'decision_record': {'path': str(self.root/gate.DECISION), 'sha256': gate.sha(self.root/gate.DECISION)},
            'evidence': {r: gate.sha(self.root/r) for r in gate.EVIDENCE.values()},
        }
        self.save()

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def save(self): self.put(self.approval_path, self.approval)

    def call(self, arm=None, execute=True):
        return gate.conditional_training_gate(self.approval_path, arm or gate.ARMS[0], execute, self.wrapper)

    def complete_a(self):
        aout = self.root/'runs'/gate.ARMS[0]/'train'
        weights = aout/'best_adapter/adapter_model.safetensors'
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b'not model weights: gate fixture only')
        self.put(aout/'completion.json', {
            'status': 'PILOT_TRAINING_COMPLETE_NO_GENERATION', 'steps': 128, 'epochs': 2,
            'arm': gate.ARMS[0], 'generation_calls': 0,
            'binding': gate._original_binding(self.config_path, self.prep, gate.ARMS[0]),
            'best_adapter_sha256': gate.sha(weights)})
        self.put(aout/'d052_authorization_binding.json', {
            'decision': 'D-052', 'approval_sha256': gate.sha(self.approval_path),
            'wrapper_sha256': gate.sha(self.wrapper)})
        self.put(self.root/'outputs/stage_i/d052_paired_pilot_v1/A_decoder_only/observation_completion.json', {
            'status': 'TRAINING_AND_AUDIT_PASS', 'error': None, 'actual_optimizer_steps': 128})

    def test_only_explicit_fresh_training_accepts_known_fail(self):
        config, prep, approval = self.call()
        self.assertIs(config, self.config)
        self.assertEqual(prep, self.prep)
        self.assertEqual(approval['conditional_reproducibility_acceptance']['strict_replay_status'], 'FAIL_UNCHANGED')
        self.check.assert_called_once_with(self.config_path)

    def test_missing_execute_denied(self):
        with self.assertRaises(ValueError): self.call(execute=False)

    def test_missing_approval_denied(self):
        self.approval_path.unlink()
        with self.assertRaises(ValueError): self.call()

    def test_approval_false_denied(self):
        self.approval['approved'] = False; self.save()
        with self.assertRaises(ValueError): self.call()

    def test_extra_scope_or_wrong_arm_denied(self):
        for scopes in [['generate'], ['train', 'generate'], ['feasibility'], []]:
            self.approval['scopes'] = scopes; self.save()
            with self.subTest(scopes=scopes), self.assertRaises(ValueError): self.call()
        with self.assertRaises(ValueError): self.call(arm='unknown')

    def test_original_preparation_gate_cannot_be_bypassed(self):
        self.check.side_effect = ValueError('Prepared config changed')
        with self.assertRaisesRegex(ValueError, 'Prepared config changed'): self.call()

    def test_config_or_wrapper_binding_drift_denied(self):
        for field in ['config_sha256', 'wrapper_sha256', 'authorization_module_sha256',
                      'preparation_manifest_sha256', 'protocol_sha256']:
            old = self.approval[field]; self.approval[field] = '0'*64; self.save()
            with self.subTest(field=field), self.assertRaises(ValueError): self.call()
            self.approval[field] = old

    def test_request_decision_and_evidence_hash_drift_denied(self):
        for path in [self.request, self.root/gate.DECISION,
                     self.root/gate.EVIDENCE['d051_readiness']]:
            old = path.read_bytes(); path.write_bytes(old+b' changed')
            with self.subTest(path=path.name), self.assertRaises(ValueError): self.call()
            path.write_bytes(old)

    def test_hashed_retroactive_pass_still_denied(self):
        path = self.root/gate.EVIDENCE['d051_readiness']
        value = gate.read(path); value['strict_replay_gate'] = 'PASS'; self.put(path, value)
        self.approval['evidence'][gate.EVIDENCE['d051_readiness']] = gate.sha(path); self.save()
        with self.assertRaisesRegex(ValueError, 'assessment changed'): self.call()

    def test_acceptance_tolerance_or_scope_drift_denied(self):
        self.approval['conditional_reproducibility_acceptance']['original_tolerance']['atol'] = 1e-5; self.save()
        with self.assertRaises(ValueError): self.call()
        self.approval['conditional_reproducibility_acceptance'] = deepcopy(gate.ACCEPTANCE)
        self.approval['scientific_scope']['epochs'] = 3; self.save()
        with self.assertRaises(ValueError): self.call()

    def test_existing_train_or_generation_path_denied(self):
        folder = self.root/'runs'/gate.ARMS[0]/'train'; folder.mkdir(parents=True)
        with self.assertRaises(ValueError): self.call()
        folder.rmdir(); generated = folder.parent/'generation'; generated.mkdir()
        with self.assertRaises(ValueError): self.call()

    def test_b_requires_completed_a_and_audit(self):
        with self.assertRaisesRegex(ValueError, 'A must complete'): self.call(gate.ARMS[1])
        self.complete_a()
        self.assertEqual(self.call(gate.ARMS[1])[2]['decision'], 'D-052')
        observed = self.root/'outputs/stage_i/d052_paired_pilot_v1/A_decoder_only/observation_completion.json'
        self.put(observed, {'status': 'FAILED_AFTER_CORE', 'error': 'audit failed', 'actual_optimizer_steps': 128})
        with self.assertRaisesRegex(ValueError, 'observation audit'): self.call(gate.ARMS[1])

    def test_b_denies_changed_a_checkpoint(self):
        self.complete_a()
        (self.root/'runs'/gate.ARMS[0]/'train/best_adapter/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'): self.call(gate.ARMS[1])

    def test_finite_numeric_scope_is_required_even_after_rehash(self):
        path = self.root/gate.EVIDENCE['d051_comparison']; value = gate.read(path)
        value['overall']['failing_elements'] = 1800; self.put(path, value)
        self.approval['evidence'][gate.EVIDENCE['d051_comparison']] = gate.sha(path); self.save()
        with self.assertRaisesRegex(ValueError, 'numerical evidence'): self.call()


if __name__ == '__main__': unittest.main(verbosity=2)
