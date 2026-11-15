"""D-029 guards: unchanged math, rare negatives, provenance and report-only routing."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from analyze_stage_f_consistency_stability import (
    ROOT, BASE, ASSISTED, analyze_rows, consistency_rows, validate_declared_prefix,
)
from stage_f_math import threshold_search
from stage_f_verification_workflow import read


def fixture(values, labels):
    return [{'case_id': str(i), 'clip_score': v, 'decision_score': round(v, 6), 'positive': y}
            for i, (v, y) in enumerate(zip(values, labels))]


class ConsistencyStabilityTests(unittest.TestCase):
    def test_rare_negatives_use_same_canonical_grid_and_ties(self):
        values = [.15, .30, .35, .35, .30]; labels = [False, True, True, True, False]
        grid = [i/100 for i in range(-100, 101)]
        result, table, _ = analyze_rows(fixture(values, labels), grid, 20, 42)
        canonical, canonical_table = threshold_search(values, labels, grid, minimum_class=1)
        self.assertEqual(table, canonical_table)
        self.assertEqual(result['full_fit']['threshold'], canonical['threshold'])
        # The generic historical requirement for other criteria remains unchanged.
        with self.assertRaises(ValueError): threshold_search(values, labels, grid)

    def test_bootstrap_reproducible_and_preserves_each_class_count(self):
        rows = fixture([.1, .2, .3, .4, .5], [False, False, True, True, True])
        a = analyze_rows(rows, [.1, .2, .3, .4, .5], 30, 42)
        self.assertEqual(a, analyze_rows(rows, [.1, .2, .3, .4, .5], 30, 42))
        for draw in a[2]:
            self.assertEqual(draw['resample_fit']['positive_labels'], 3)
            self.assertEqual(draw['resample_fit']['negative_labels'], 2)
        for fold in a[0]['leave_one_negative_out']:
            self.assertEqual(fold['fold_fit']['positive_labels'], 3)
            self.assertEqual(fold['fold_fit']['negative_labels'], 1)

    def test_one_negative_lono_does_not_invent_threshold(self):
        result, _, _ = analyze_rows(fixture([.1, .3], [False, True]), [.1, .2, .3], 5, 42)
        self.assertEqual(result['leave_one_negative_out'][0]['status'], 'UNIDENTIFIABLE_ONE_CLASS_FOLD')
        self.assertIsNone(result['leave_one_negative_out'][0]['selected_threshold'])
        with self.assertRaises(ValueError):
            analyze_rows(fixture([.3], [True]), [.2], 5, 42)

    def test_uncertain_and_unverified_ai_labels_never_enter_fit(self):
        cases = [{'case_id': str(i), 'kind': 'image_caption', 'business_id': 'b',
                  'photo_ids': [str(i)], 'normalized_captions': ['caption'], 'raw_captions': ['caption']}
                 for i in range(4)]
        actual = {str(i): {'human_verified': i != 3,
                          'human_final_label': {'consistency': label},
                          'ai_suggested_label': {'consistency': 'inconsistent'}}
                  for i, label in enumerate(['consistent', 'inconsistent', 'uncertain', ''])}
        scores = {str(i): {'clip_score': .2} for i in range(4)}
        before = copy.deepcopy(actual)
        rows, uncertain, pending = consistency_rows({'cases': cases}, actual, scores, {'b': 'calibration'}, set(actual))
        self.assertEqual([r['case_id'] for r in rows], ['0', '1'])
        self.assertEqual(uncertain, ['2']); self.assertEqual(pending, ['3'])
        self.assertEqual(actual, before)
        with self.assertRaises(AssertionError):
            consistency_rows({'cases': cases}, actual, scores, {'b': 'K_study'}, set(actual))

    def test_current_confirmed_older_prefix_usable_pending_batch_not_required(self):
        dataset = read(BASE/'human_check/annotation_dataset.json')
        order = read(ASSISTED/'verification_order.json')
        selection = read(ASSISTED/'active_selection.json')
        ai = read(ASSISTED/'ai_suggestions.json'); labels = read(ASSISTED/'latest_human_export.json')
        _, prefix = validate_declared_prefix(dataset, order, selection, ai, labels)
        self.assertEqual(prefix, {'image_cases': 130, 'pair_cases': 140})
        invalid = copy.deepcopy(labels); invalid['active_case_ids'][0], invalid['active_case_ids'][1] = invalid['active_case_ids'][1], invalid['active_case_ids'][0]
        with self.assertRaises(AssertionError):
            validate_declared_prefix(dataset, order, selection, ai, invalid)

    def test_old_cli_routes_before_any_import_expansion_or_freeze(self):
        import calibrate_stage_f_thresholds as legacy
        with patch('sys.argv', ['calibrate', '--labels', 'NOT_READ.json']), \
             patch('stage_f_d033_protocol.active_d033',return_value=False), \
             patch('stage_f_d032_protocol.active_d032', return_value=False), \
             patch('stage_f_d031_protocol.active_d031', return_value=False), \
             patch('stage_f_d030_protocol.active_d030', return_value=False), \
             patch('analyze_stage_f_consistency_stability.active_refinement', return_value=True), \
             patch('analyze_stage_f_consistency_stability.run') as report_only, \
             patch.object(legacy, 'write') as mutation, \
             patch.object(legacy, 'balance_and_expansion') as expansion:
            legacy.main()
            report_only.assert_called_once_with(Path('NOT_READ.json'))
            mutation.assert_not_called(); expansion.assert_not_called()


if __name__ == '__main__': unittest.main()
