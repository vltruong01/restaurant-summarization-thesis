"""Offline tests: real frozen RO/U labels, tensor masks/shift, fail-closed contracts.
No LED model is instantiated; no forward/generate/backward/optimizer or CUDA call.
"""
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'scripts'))
PARENT = Path(os.environ.get('LED_MATCHED_PARENT_ROOT',
                            R / 'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE')).resolve()
sys.path.append(str(PARENT / 'scripts'))
import led_large_matched_family_v1 as m
import led_large_shuffled_fixture_v1 as s
import led_large_final_evaluation_spec_v1 as e
from tokenizers import Tokenizer
from led_large_decoder_label_view_v1 import CorrectedLabelCollator
from stage_hi_led_data import LEDCompleteCollator
from transformers.models.led.modeling_led import shift_tokens_right

P = R / m.PREP


class ConditionRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = m.read(PARENT / 'configs/stage_i_led_large_decoder_policy_v2.json')
        cls.data = {c: m.condition_data(m.read(m.paths(c)), PARENT) for c in m.CONDITIONS}

    def test_counts_and_frozen_identity_order_across_conditions(self):
        ro, u = (self.data[c] for c in m.CONDITIONS)
        self.assertEqual([len(x) for x in ro], [2000, 200])
        for a, b in zip(ro, u):
            self.assertEqual(a.business_ids, b.business_ids)
        self.assertFalse(set(ro[0].business_ids) & set(ro[1].business_ids))

    def test_original_condition_inputs_differ_and_are_not_F_substitutes(self):
        ro, u = (self.data[c][0] for c in m.CONDITIONS)
        self.assertNotEqual(ro[0]['input_ids'], u[0]['input_ids'])
        self.assertEqual(ro[0]['input_ids'], ro.original[0]['input_ids'])
        self.assertEqual(u[0]['input_ids'], u.original[0]['input_ids'])

    def test_all4400_labels_change_only_initial_BOS(self):
        for pair in self.data.values():
            for ds in pair:
                for i in range(len(ds)):
                    self.assertEqual(ds[i]['labels'], ds.original[i]['labels'][1:])
                    self.assertNotIn(0, ds[i]['labels'])
                    self.assertEqual(ds[i]['labels'][-1], 2)

    def test_actual_RO_U_collation_preserves_source_masks_shapes_and_shift(self):
        for pair in self.data.values():
            ds = pair[0]
            features = [ds[0], ds[1]]
            batch = CorrectedLabelCollator()(features)
            old = LEDCompleteCollator()([ds.original[0], ds.original[1]])
            for key in ('input_ids', 'attention_mask', 'global_attention_mask'):
                self.assertTrue((batch[key] == old[key]).all().item())
            self.assertEqual(batch['labels'].shape, old['labels'].shape)
            shifted = shift_tokens_right(batch['labels'], 1, 2)
            for i, f in enumerate(features):
                self.assertEqual(shifted[i, 0].item(), 2)
                self.assertEqual(shifted[i, 1:len(f['labels'])].tolist(), f['labels'][:-1])
                self.assertEqual(batch['labels'][i, len(f['labels'])-1].item(), 2)
                self.assertTrue((batch['labels'][i, len(f['labels']):] == -100).all().item())

    def test_exact_config_allowlist_and_unchanged_scientific_parameters(self):
        science = ('lora', 'optimization', 'generation', 'budgets', 'runtime_pins', 'training', 'seed',
                   'amp_scaler', 'dynamic_overflow_policy', 'global_attention', 'initialization', 'loss')
        for c in m.CONDITIONS:
            new = m.read(m.paths(c))
            self.assertEqual(new, m.expected_config(self.old, c))
            for key in science:
                self.assertEqual(new[key], self.old[key])

    def test_unknown_condition_is_not_accepted_as_folder_rename(self):
        for c in ('filtered', 'shuffled', 'test', 'RO'):
            with self.assertRaises(ValueError):
                m.expected_config(self.old, c)

    def test_all24_cases_have_their_own_condition_sources(self):
        for condition in m.CONDITIONS:
            val = self.data[condition][1]
            cases = [json.loads(x) for x in (P / f'{condition}_validation_cases.jsonl').read_text().splitlines()]
            self.assertEqual(len(cases), 24)
            self.assertEqual(sum(x['subset'] == 'random' for x in cases), 20)
            for row in cases:
                i = val.business_ids.index(row['business_id'])
                self.assertEqual(row['condition'], condition)
                self.assertEqual(row['input_ids'], val[i]['input_ids'])
                self.assertEqual(row['label_ids'][1:], val[i]['labels'])

    def test_gate_stress_cases_are_real_TRAIN_maxima(self):
        for condition in m.CONDITIONS:
            ds = self.data[condition][0]
            gate = m.read(P / f'{condition}_FEASIBILITY_CASES.json')
            self.assertTrue(set(gate['business_ids']) <= set(ds.business_ids))
            self.assertEqual(max(x['source'] for x in gate['cases']), max(len(ds[i]['input_ids']) for i in range(len(ds))))
            self.assertEqual(max(x['original_target'] for x in gate['cases']), max(len(ds.original[i]['labels']) for i in range(len(ds))))

    def test_no_model_execution_entry_in_offline_materializer(self):
        source = (R / 'scripts/prepare_led_large_matched_family_v1.py').read_text()
        self.assertNotIn('.generate(', source)
        self.assertNotIn('.backward(', source)
        self.assertNotIn('LEDForConditionalGeneration(', source)


class Authorization(unittest.TestCase):
    def receipt(self):
        now = datetime.now(timezone.utc)
        return dict(execution_authorized_at_utc=now.isoformat(), execution_deadline_utc=(now+timedelta(hours=24)).isoformat(),
                    researcher_absolute_stop_utc=(now+timedelta(hours=30)).isoformat(),
                    defense_reserve_at_least_14_days_confirmed=True, max_training_updates=1250,
                    feasibility_updates_max=3, all_generation_calls_max=6)

    def test_fresh_valid_clock(self):
        m.check_deadline(self.receipt())

    def test_expired_clock(self):
        a = self.receipt()
        with self.assertRaises(ValueError):
            m.check_deadline(a, datetime.now(timezone.utc)+timedelta(days=3))

    def test_unconfirmed_reserve_or_naive_clock_or_excess_budget_rejected(self):
        for key, val in [('defense_reserve_at_least_14_days_confirmed', False), ('all_generation_calls_max', 54),
                         ('execution_authorized_at_utc', '2026-10-24T00:00:00')]:
            a = self.receipt(); a[key] = val
            with self.assertRaises(ValueError):
                m.check_deadline(a)

    def test_no_GPU_or_evaluate_scope_in_pending_contracts(self):
        for condition, name in [('reviews_only', 'RO'), ('unfiltered', 'U')]:
            path = R / 'approvals' / f'{name}_PENDING.json'
            with self.assertRaises(ValueError):
                m.authorize(path, condition, 'train')
            self.assertEqual(m.read(path)['scopes'], ['feasibility', 'train'])


class Shuffle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = m.read(P / 'SHUFFLED_SOURCE_FIXTURES.json')
        cls.tok = Tokenizer.from_file(str(PARENT / 'assets/tokenizers/led-base-16384/tokenizer.json'))

    def run_fixture(self, rows=None, edges=None, **kw):
        return s.build_mapping(rows or self.fixture['rows'], edges or self.fixture['explicit_test_edges'], self.tok, **kw)

    def test_real_train_validation_fixture_determinism(self):
        self.assertEqual(self.run_fixture(), self.run_fixture())
        self.assertEqual(self.run_fixture(), m.read(P / 'SHUFFLED_FIXTURE_RESULTS.json'))

    def test_reviews_fixed_counts_and_empty_caption_handling(self):
        original = {r['business_id']: r for r in self.fixture['rows']}
        result = self.run_fixture()
        for row in result['mappings']:
            self.assertEqual(s.partition(row['source'])[0], s.partition(original[row['recipient_id']]['source'])[0])
            if not row['caption_count']:
                self.assertIsNone(row['donor_id'])
            else:
                self.assertNotEqual(row['recipient_id'], row['donor_id'])
                self.assertEqual(original[row['donor_id']]['split'], row['split'])

    def test_no_silent_production_cuisine_fallback(self):
        with self.assertRaises(ValueError):
            self.run_fixture(fixture_only=False)

    def test_self_match_rejected(self):
        edges = copy.deepcopy(self.fixture['explicit_test_edges']); bid = next(iter(edges)); edges[bid] = [bid]
        with self.assertRaises(ValueError):
            self.run_fixture(edges=edges)

    def test_cross_split_rejected(self):
        rows = copy.deepcopy(self.fixture['rows']); edge = next(iter(self.fixture['explicit_test_edges'].items()))
        next(r for r in rows if r['business_id'] == edge[1][0])['split'] = 'validation' if next(r for r in rows if r['business_id'] == edge[0])['split'] == 'train' else 'train'
        with self.assertRaises(ValueError):
            self.run_fixture(rows=rows)

    def test_caption_count_and_token_overflow_rejected(self):
        for change in ('count', 'overflow'):
            rows = copy.deepcopy(self.fixture['rows']); row = next(r for r in rows if r['selected_photo_ids'])
            if change == 'count': row['selected_photo_ids'] = []
            else: row['source'] += ' very' * 6000
            with self.assertRaises(ValueError):
                self.run_fixture(rows=rows)

    def test_held_out_and_unapproved_no_match_rejected(self):
        rows = copy.deepcopy(self.fixture['rows']); rows[0]['split'] = 'test'
        with self.assertRaises(ValueError): self.run_fixture(rows=rows)
        edges = copy.deepcopy(self.fixture['explicit_test_edges']); edges[next(iter(edges))] = []
        with self.assertRaises(ValueError): self.run_fixture(edges=edges)


class Evaluation(unittest.TestCase):
    def test_target_free_schema_and_BOS_only(self):
        r = dict(business_id='SYNTHETIC', split='train', condition='RO', input_ids=[0, 40, 2, 1],
                 attention_mask=[1, 1, 1, 0], global_attention_mask=[1, 0, 0, 0], source_sha256='fixture',
                 selected_review_ids=['fixture'], selected_photo_ids=[])
        self.assertTrue(e.target_free_record(r))
        for k, value in [('labels', [40, 2]), ('target', 'text'), ('global_attention_mask', [1, 1, 0, 0])]:
            bad = dict(r); bad[k] = value
            with self.assertRaises(ValueError): e.target_free_record(bad)

    def test_sampling_is_metadata_only_deterministic_and_exactly30(self):
        frame = [dict(business_id=f'F{i}', split='train', review_tokens=i*10, usable_photo_count=i%6) for i in range(100)]
        self.assertEqual(e.sample_fixture(frame), e.sample_fixture(frame))
        self.assertEqual(len(e.sample_fixture(frame)['ids']), 30)
        frame[0]['quality'] = 5
        with self.assertRaises(ValueError): e.sample_fixture(frame)

    def test_heldout_sampler_is_not_enabled(self):
        with self.assertRaises(ValueError):
            e.sample_fixture([dict(business_id='X', split='test', review_tokens=500, usable_photo_count=0)], 1)

    def test_360_blinded_slots_RO_visual_NA_and_no_fabricated_scores(self):
        f = e.annotation_fixture([f'F{i}' for i in range(30)])
        self.assertEqual(len(f['tasks']), 360)
        self.assertTrue(e.validate_ratings(f['tasks'], f['PRIVATE_CONDITION_KEY'], require_complete=False))
        with self.assertRaises(ValueError): e.validate_ratings(f['tasks'], f['PRIVATE_CONDITION_KEY'])
        self.assertTrue(all(not t['researcher_confirmed'] for t in f['tasks']))

    def test_invalid_reference_substitution_blocked(self):
        for origin in ('D053_validation', 'Teacher', 'model_assisted_only'):
            with self.assertRaises(ValueError):
                e.reference_contract(dict(split='test', independent_human_approved=True, business_count=100,
                                          unresolved=0, reference_sha256='x', approval_receipt_sha256='y', origin=origin))

    def test_zero_caption_visual_abstention_preserves_original_combined_rubric(self):
        businesses = [f'F{i}' for i in range(30)]
        f = e.annotation_fixture(businesses, caption_counts={(b, 'F'): 0 for b in businesses})
        rows = [t for t in f['tasks'] if f['PRIVATE_CONDITION_KEY'][t['alias']]['condition'] == 'F']
        self.assertTrue(all(r['visual_contribution_applicability'] == 'N/A_NO_CAPTIONS' for r in rows))
        self.assertTrue(all(r['scores']['visual_grounding_and_contribution'] is None for r in rows))
        self.assertTrue(all(len(r['scores']) == 6 for r in rows))
        self.assertTrue(e.validate_ratings(f['tasks'], f['PRIVATE_CONDITION_KEY'], require_complete=False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
