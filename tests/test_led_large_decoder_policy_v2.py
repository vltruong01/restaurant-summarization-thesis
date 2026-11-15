"""Offline regression tests: tensors, state dictionaries and fake inference only."""
import copy
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
# Re-run all eleven accepted label/shift/source/config tests without changing them.
from test_led_large_decoder_policy_v1 import LabelPolicy
import led_large_decoder_early_sanity_v2 as early
import stage_i_led_large_decoder_policy_v2 as helper
import run_stage_i_led_large_decoder_policy_v2 as runner
from led_large_dynamic_amp_v2 import capture_rng
from run_stage_i_d050_feasibility_v1 import fingerprint


class TechnicalCriteria(unittest.TestCase):
    def test_single_BOS_followed_by_useful_text_is_warning_and_PASS(self):
        r = early.sanity_result([2, 0] + list(range(10, 35)) + [2], 'Useful business information.')
        self.assertEqual(r['status'], 'PASS_TECHNICAL_ONLY')
        self.assertEqual(r['longest_BOS_run'], 1)
        self.assertTrue(r['warnings']); self.assertFalse(r['errors'])

    def test_four_BOS_is_warning_five_is_hard_failure(self):
        for run, expected in ((4, 'PASS_TECHNICAL_ONLY'), (5, 'FAIL_STOP'), (766, 'FAIL_STOP')):
            with self.subTest(run=run):
                r = early.sanity_result([2] + [0]*run + list(range(10, 35)) + [2], 'content')
                self.assertEqual(r['status'], expected)
                self.assertEqual(r['longest_BOS_run'], run)

    def test_nonconsecutive_BOS_is_recorded_without_loop_failure(self):
        r = early.sanity_result([2] + [0, 10]*8 + [2], 'Useful content')
        self.assertEqual(r['status'], 'PASS_TECHNICAL_ONLY')
        self.assertEqual(r['BOS_after_start'], 8); self.assertEqual(r['longest_BOS_run'], 1)

    def test_blank_or_whitespace_output_hard_fails(self):
        for ids, text in (([2, 2], ''), ([2]+list(range(10, 35))+[2], ' \n\t')):
            self.assertEqual(early.sanity_result(ids, text)['status'], 'FAIL_STOP')

    def test_near_empty_special_dominated_without_BOS_loop_hard_fails(self):
        r = early.sanity_result([2] + [0, 3]*10 + [10, 2], 'One')
        self.assertEqual(r['longest_BOS_run'], 1)
        self.assertEqual(r['status'], 'FAIL_STOP')
        self.assertIn('Near-empty special-token-dominated output', r['errors'])

    def test_short_readable_content_alone_is_not_quality_rejection(self):
        r = early.sanity_result([2, 10, 11, 2], 'Open daily.')
        self.assertEqual(r['status'], 'PASS_TECHNICAL_ONLY'); self.assertFalse(r['semantic_quality_GO'])

    def test_cap_hit_with_content_is_warning_even_without_EOS(self):
        for eos in (True, False):
            ids = [2] + [10, 11]*383 + [2 if eos else 12]
            r = early.sanity_result(ids, 'Useful but capped text')
            self.assertEqual(len(ids), 768); self.assertTrue(r['cap_hit'])
            self.assertEqual(r['status'], 'PASS_TECHNICAL_ONLY')
            self.assertIn('Generation length cap reached', r['warnings'])

    def test_cap_does_not_override_collapse_failure(self):
        r = early.sanity_result([2]+[0]*766+[2], '')
        self.assertEqual(r['status'], 'FAIL_STOP'); self.assertTrue(r['cap_hit'])
        self.assertGreaterEqual(len(r['errors']), 2)

    def test_wrong_decoder_start_and_invalid_token_are_protocol_failures(self):
        for ids in ([], [0, 10, 2], [2, -1, 2], [2, 50265, 2], [2, True, 2]):
            self.assertEqual(early.sanity_result(ids, 'text')['status'], 'FAIL_STOP')


class StateObject:
    def __init__(self, value): self.value = copy.deepcopy(value)
    def state_dict(self): return self.value


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Parameter(torch.ones(2))
        self.lora.grad = torch.zeros(2)
        self.child = torch.nn.Dropout(0)
        self.child.eval()  # Deliberately mixed flags: restore each module individually.
        self.active_adapters = ['default']
        self.disable_adapters = False
        self.merged_adapters = []
        self.config = SimpleNamespace(use_cache=False)
        self.outputs = []
        self.calls = 0
        self.raise_inference = False

    def generate(self, **kwargs):
        # Fake fixture only: never calls any model forward or GPU kernel.
        random.random(); np.random.random(); torch.rand(1)
        self.config.use_cache = True
        if self.raise_inference:
            raise RuntimeError('SYNTHETIC INFERENCE FAILURE')
        ids = self.outputs[self.calls]; self.calls += 1
        return torch.tensor([ids])


def make_trainer(model):
    scaler = StateObject(dict(scale=65536., _growth_tracker=250))
    return SimpleNamespace(model=model, optimizer=StateObject(dict(state={0: dict(step=torch.tensor(250.), exp_avg=torch.ones(2))},
        param_groups=[dict(lr=.00016, params=[0])])), lr_scheduler=StateObject(dict(last_epoch=250, _step_count=251)),
        accelerator=SimpleNamespace(scaler=scaler, optimizer_step_was_skipped=False),
        amp_recovery=SimpleNamespace(state=lambda s: dict(successful_updates=250, scale=s.state_dict()['scale'])),
        state=SimpleNamespace(global_step=250, epoch=1., max_steps=1250, log_history=[]),
        control=SimpleNamespace(should_training_stop=False, should_save=False))


class Continuity(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch('torch.cuda.is_available', return_value=False))
        self.stack.enter_context(patch('torch.cuda._lazy_init', side_effect=AssertionError('GPU FORBIDDEN')))
        self.stack.enter_context(patch('peft.get_peft_model_state_dict', side_effect=lambda m: {'lora': m.lora}))
        self.addCleanup(self.stack.close)
        self.model = FakeModel(); self.trainer = make_trainer(self.model)

    def test_success_restores_RNG_mixed_modes_cache_and_all_training_state(self):
        before = early.continuity_snapshot(self.trainer)
        rng = fingerprint(capture_rng()); evidence = {}
        with early.preserve_training_state(self.model, self.trainer, evidence):
            self.model.eval(); self.model.config.use_cache = True
            random.random(); np.random.random(); torch.rand(3)
        self.assertEqual(evidence['status'], 'PASS')
        self.assertTrue(all(evidence['checks'].values()))
        self.assertEqual(before, early.continuity_snapshot(self.trainer))
        self.assertEqual(fingerprint(capture_rng()), rng)
        self.assertTrue(self.model.training); self.assertFalse(self.model.child.training)
        self.assertFalse(self.model.config.use_cache)

    def test_inference_exception_restores_states_without_hiding_original_error(self):
        evidence = {}; rng = fingerprint(capture_rng())
        with self.assertRaisesRegex(RuntimeError, 'ORIGINAL'):
            with early.preserve_training_state(self.model, self.trainer, evidence):
                self.model.eval(); self.model.config.use_cache = True; torch.rand(2)
                raise RuntimeError('ORIGINAL')
        self.assertEqual(evidence['status'], 'PASS')
        self.assertEqual(fingerprint(capture_rng()), rng); self.assertTrue(self.model.training)

    def test_CUDA_RNG_round_trip_using_synthetic_states_without_GPU(self):
        # Exercise the real capture/restore CUDA branch with CPU byte tensors only.
        cuda = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
        def restore(states): cuda[:] = [x.clone() for x in states]
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.get_rng_state_all',
                side_effect=lambda: [x.clone() for x in cuda]), patch('torch.cuda.set_rng_state_all', side_effect=restore):
            evidence = {}
            with early.preserve_training_state(self.model, self.trainer, evidence):
                cuda[0].add_(1)
            self.assertEqual(cuda[0].tolist(), [1, 2, 3])
            self.assertTrue(evidence['checks']['RNG_restored'])

    def test_detects_each_optimizer_scheduler_scaler_counter_or_gradient_mutation(self):
        changes = [lambda t: t.optimizer.value['state'][0]['step'].add_(1),
                   lambda t: t.lr_scheduler.value.update(last_epoch=251),
                   lambda t: t.accelerator.scaler.value.update(scale=32768.),
                   lambda t: setattr(t.state, 'global_step', 251),
                   lambda t: setattr(t.control, 'should_training_stop', True),
                   lambda t: t.model.lora.grad.add_(1),
                   lambda t: setattr(t.model, 'disable_adapters', True)]
        for change in changes:
            self.model = FakeModel(); self.trainer = make_trainer(self.model)
            evidence = {}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'continuity'):
                with early.preserve_training_state(self.model, self.trainer, evidence): change(self.trainer)
            self.assertEqual(evidence['status'], 'FAIL_STOP')

    def test_adapter_value_mutation_detected_without_silent_restoration(self):
        evidence = {}
        with self.assertRaisesRegex(ValueError, 'adapter_unchanged'):
            with early.preserve_training_state(self.model, self.trainer, evidence):
                with torch.no_grad(): self.model.lora.add_(1)
        self.assertTrue(torch.equal(self.model.lora, torch.full((2,), 2.)))

    def test_primary_exception_preserved_when_continuity_also_fails(self):
        evidence = {}
        with self.assertRaisesRegex(RuntimeError, 'ORIGINAL') as caught:
            with early.preserve_training_state(self.model, self.trainer, evidence):
                self.trainer.state.global_step += 1
                raise RuntimeError('ORIGINAL')
        self.assertEqual(evidence['status'], 'FAIL_STOP')
        self.assertIn('continuity', caught.exception.__notes__[0])

    def run_fixture(self, ids, text, fail=False):
        cfg = helper.read(helper.CONFIG)
        self.model.outputs = [ids] * 4; self.model.raise_inference = fail
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d); folder = root / 'train'; checkpoint = folder / 'checkpoint-250'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'checkpoint_integrity.json').write_text('{}')
            (checkpoint / 'adapter_model.safetensors').write_bytes(b'SYNTHETIC ONLY')
            cases = root / cfg['early_sanity']['cases_file']; cases.parent.mkdir(parents=True)
            cases.write_text('\n'.join(json.dumps(dict(business_id=bid, subset=subset, input_ids=[0,9,2], label_ids=[0,10,11,2],
                source_sha256='fixture')) for bid, subset in zip(cfg['early_sanity']['business_ids'], cfg['early_sanity']['subsets'])))
            gen = root / cfg['effective_generation_config_file']; gen.parent.mkdir(parents=True)
            gen.write_text(json.dumps(helper.read(ROOT / cfg['effective_generation_config_file'])))
            stack.enter_context(patch('transformers.LEDTokenizerFast.from_pretrained', return_value=SimpleNamespace(decode=lambda *a, **k: text)))
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda t: t))
            stack.enter_context(patch('torch.autocast', side_effect=lambda *a, **k: nullcontext()))
            def write(path, value): path.write_text(json.dumps(value), encoding='utf-8')
            error = None; report = None
            try:
                report = early.run_early_sanity(self.model, folder,
                    dict(early_sanity_generation_approved=True, early_sanity_outputs_max=4), cfg,
                    SimpleNamespace(check=lambda: None), root, dict(fixture=True), write, lambda a: None, self.trainer)
            except (ValueError, RuntimeError) as exc: error = exc
            raw = [json.loads((folder / 'early_sanity' / (bid + '.json')).read_text()) for bid in cfg['early_sanity']['business_ids']
                   if (folder / 'early_sanity' / (bid + '.json')).is_file()]
            saved_report = folder / 'early_sanity/REPORT.json'
            saved = json.loads(saved_report.read_text()) if saved_report.exists() else None
            continuity = json.loads((folder / 'early_sanity/STATE_CONTINUITY.json').read_text())
            return report, error, raw, saved, continuity

    def test_four_raw_predictions_and_diagnostics_preserved_on_BOS_collapse(self):
        ids = [2]+[0]*5+list(range(10,35))+[2]
        _, error, raw, saved, continuity = self.run_fixture(ids, 'text')
        self.assertIsInstance(error, ValueError); self.assertEqual(len(raw), 4)
        self.assertTrue(all(r['token_ids'] == ids for r in raw))
        self.assertEqual(saved['generation_calls'], 4); self.assertEqual(saved['status'], 'FAIL_STOP')
        self.assertFalse(saved['quality_GO']); self.assertEqual(continuity['status'], 'PASS')

    def test_warning_outputs_continue_and_preserve_four_records(self):
        report, error, raw, saved, continuity = self.run_fixture([2,0]+list(range(10,35))+[2], 'useful text')
        self.assertIsNone(error); self.assertEqual(len(raw), 4)
        self.assertEqual(report['status'], 'PASS_TECHNICAL_ONLY'); self.assertFalse(report['quality_GO'])
        self.assertTrue(all(r['technical_screen']['warnings'] for r in raw))
        self.assertTrue(all(continuity['checks'].values()))

    def test_inference_runtime_failure_saves_continuity_but_no_fake_predictions(self):
        _, error, raw, saved, continuity = self.run_fixture([2,10,2], 'text', fail=True)
        self.assertIsInstance(error, RuntimeError); self.assertEqual(raw, []); self.assertIsNone(saved)
        self.assertEqual(continuity['status'], 'PASS'); self.assertTrue(self.model.training)


class MinimalScope(unittest.TestCase):
    def test_config_changes_only_early_sanity_and_execution_namespace(self):
        old = helper.read(ROOT / 'configs/stage_i_led_large_decoder_policy_v1.json')
        new = helper.read(helper.CONFIG)
        keys = {k for k in old if old[k] != new[k]}
        self.assertEqual(keys, {'experiment', 'run_root', 'vast_fresh', 'early_sanity'})
        prior = dict(old['early_sanity']); del prior['requires_nonempty_no_BOS_natural_EOS_min16_content']
        self.assertEqual(new['early_sanity'], dict(prior, technical_criteria=early.CRITERIA))
        self.assertEqual(new['decoder_label_policy'], old['decoder_label_policy'])
        self.assertEqual(new['label_views_manifest'], old['label_views_manifest'])

    def test_accepted_label_module_and_new_files_match_manifest(self):
        manifest = helper.read(helper.PREP / 'PREFLIGHT_MANIFEST.json')
        self.assertEqual(helper.sha(ROOT/'scripts/led_large_decoder_label_view_v1.py'), manifest['accepted_label_module_sha256'])
        for rel, digest in manifest['files'].items(): self.assertEqual(helper.sha(ROOT/rel), digest, rel)

    def test_new_template_is_unapproved_and_scope_stays_54_calls(self):
        a = helper.read(helper.PREP / 'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')
        self.assertFalse(a['GPU_execution_approved']); self.assertIsNone(a['execution_authorized_at_utc'])
        self.assertEqual((a['technical_generation_calls_max'], a['early_sanity_outputs_max'], a['evaluation_outputs_max'], a['all_generation_calls_max']), (2,4,48,54))
        with self.assertRaisesRegex(ValueError, 'not approved'): helper.authorize(helper.PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json', 'train', True)

    def test_old_scope_rejected_even_if_flags_synthetically_approved(self):
        a = helper.read(ROOT/'outputs/stage_i/led_large_decoder_policy_v1/AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')
        a.update(GPU_execution_approved=True, training_authorized=True, scientific_amendment_approved=True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'SYNTHETIC.json';p.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError, 'scope'): helper.authorize(p, 'train', True)

    def test_no_resume_or_GPU_without_new_approval(self):
        with self.assertRaisesRegex(ValueError, 'Fresh corrected'): runner.train({}, {}, Path('unused'), resume=Path('old'))
        with patch.object(runner, 'preflight', return_value={}), patch.object(sys, 'argv', ['runner', 'train']), patch.object(runner, 'check_required_gpu', side_effect=AssertionError('GPU FORBIDDEN')):
            with self.assertRaisesRegex(ValueError, 'OFFLINE ONLY'): runner.main()

    def test_callback_passes_live_trainer_to_epoch1_guard(self):
        class Base:
            def __init__(self): self.trainer=SimpleNamespace(model=object());self.folder=Path('fixture');self.approval={};self.memory=object()
            def on_save(self,*a,**kw): pass
        with patch.object(helper, '_core_callback_class', return_value=Base), patch.object(early, 'run_early_sanity', side_effect=ValueError('sanity failed')) as screen:
            c=helper.callback_class()()
            with self.assertRaisesRegex(ValueError, 'sanity failed'): c.on_save(None,SimpleNamespace(global_step=250),None)
            self.assertIs(screen.call_args.kwargs['trainer'], c.trainer)


if __name__ == '__main__':
    unittest.main()
