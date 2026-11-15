"""Synthetic CPU tests only; no model, CUDA, dataset, or teacher API execution."""
import math
import json
import unittest

import numpy as np
import torch

from stage_i_d051_numeric import compare_tensors, compare_tensor_maps, compare_nested_states


class D051NumericTests(unittest.TestCase):
    def test_near_zero_safe_relative_errors_and_failure_counts(self):
        a = torch.tensor([0.0, 1e-13, 0.0, 2.0])
        b = torch.tensor([0.0, 2e-13, 2e-6, 2.0])
        row = compare_tensors(a, b)
        self.assertFalse(row['allclose'])
        self.assertEqual(row['failing_elements'], 1)
        self.assertEqual(row['failing_pct'], 25.0)
        self.assertEqual(row['near_zero_both_count_at_atol'], 2)
        self.assertAlmostEqual(row['max_relative_symmetric'], 1.0)
        self.assertTrue(math.isfinite(row['relative_l2']))

    def test_torch_isclose_orientation_is_preserved(self):
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([1.11], dtype=torch.float64)
        self.assertTrue(compare_tensors(a, b, rtol=.1, atol=0)['allclose'])
        self.assertFalse(compare_tensors(b, a, rtol=.1, atol=0)['allclose'])

    def test_metrics_use_fp64_but_acceptance_preserves_fp16(self):
        a = torch.tensor([.01, .02], dtype=torch.float16)
        b = a.clone(); b[0] += .0001
        row = compare_tensors(a, b)
        self.assertEqual(row['allclose'], bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6)))
        expected = ((b.double() - a.double()).square().sum() / 2).sqrt().item()
        self.assertEqual(row['rms'], expected)

    def test_aggregates_weight_elements_not_tensor_means(self):
        stem = 'base_model.model.led.encoder.layers.0.self_attn.longformer_self_attn.query'
        names = [stem + '.lora_A.default.weight', stem + '.lora_B.weight']
        a = {names[0]: torch.tensor([1.0]), names[1]: torch.tensor([2., 2., 2.])}
        b = {names[0]: torch.tensor([3.0]), names[1]: torch.tensor([2., 2., 2.])}
        result = compare_tensor_maps(a, b)
        self.assertEqual(len(result['per_module']), 1)
        group = next(iter(result['per_module'].values()))
        self.assertEqual(group['branch'], 'encoder_local')
        self.assertEqual(result['overall']['rms'], 1.0)
        self.assertAlmostEqual(result['overall']['relative_l2'], 2 / math.sqrt(13))
        self.assertEqual(result['overall']['failing_pct'], 25.0)

    def test_branch_assignment_and_unmatched_keys(self):
        keys = ['led.encoder.layers.0.self_attn.longformer_self_attn.query_global.lora_A.weight',
                'led.decoder.layers.0.encoder_attn.v_proj.lora_B.default.weight']
        a = {key: torch.zeros(2) for key in keys}
        result = compare_tensor_maps(a, {keys[0]: torch.zeros(2), 'extra': torch.zeros(2)})
        self.assertEqual(result['missing_from_replay'], [keys[1]])
        self.assertEqual(result['extra_in_replay'], ['extra'])
        self.assertFalse(result['overall']['allclose'])
        self.assertEqual(result['per_tensor'][keys[0]]['branch'], 'encoder_global')
        self.assertTrue(compare_tensor_maps(a, a)['overall']['exact'])

    def test_empty_nonfinite_and_structure_never_silently_pass(self):
        self.assertEqual(compare_tensors(torch.empty(0), torch.empty(0))['rms'], 0.0)
        row = compare_tensors(torch.tensor([float('nan'), 1.]), torch.tensor([float('nan'), 1.]))
        self.assertFalse(row['all_finite']); self.assertFalse(row['allclose'])
        self.assertIsNone(row['rms'])
        with self.assertRaises(ValueError):
            compare_tensors(torch.ones(1), torch.ones(1, dtype=torch.float64))
        with self.assertRaises(ValueError):
            compare_tensors(torch.ones(1), torch.ones(2))

    def test_nested_states_exact_and_tolerant_are_distinct(self):
        a = {'state': {0: {'exp_avg': torch.tensor([1.0]), 'step': torch.tensor(2.)}},
             'lr': .0002, 'rng': ('tag', np.array([1, 2], dtype=np.uint32))}
        b = {'state': {0: {'exp_avg': torch.tensor([1.000001]), 'step': torch.tensor(2.)}},
             'lr': .0002, 'rng': ('tag', np.array([1, 2], dtype=np.uint32))}
        row = compare_nested_states(a, b)
        self.assertFalse(row['exact']); self.assertTrue(row['allclose'])
        self.assertEqual(row['different_paths'], ["root['state'][0]['exp_avg']"])
        self.assertTrue(compare_nested_states(a, a)['exact'])
        self.assertFalse(compare_nested_states({'step': 1000000}, {'step': 1000001})['allclose'])

    def test_nested_shape_type_and_missing_paths(self):
        row = compare_nested_states({'x': torch.ones(1), 'y': [1]}, {'x': torch.ones(2), 'y': (1,), 'z': None})
        self.assertFalse(row['exact']); self.assertFalse(row['allclose'])
        self.assertEqual(len(row['structural_mismatches']), 3)
        self.assertIn("root['z']", [r['path'] for r in row['structural_mismatches']])

    def test_discrete_rng_and_nonfinite_scalar_stay_json_serializable(self):
        row = compare_nested_states({'rng': np.array([1000000000], dtype=np.uint32), 'step': np.int64(2)},
                                    {'rng': np.array([1000000001], dtype=np.uint32), 'step': np.int64(2)})
        self.assertFalse(row['allclose'])
        json.dumps(row, allow_nan=False)
        row = compare_nested_states({'loss': float('nan')}, {'loss': 1.0})
        self.assertFalse(row['allclose'])
        json.dumps(row, allow_nan=False)

    def test_diagnostics_do_not_mutate_values_or_cpu_rng(self):
        torch.manual_seed(6)
        a = torch.randn(5); b = a.clone()
        saved = torch.get_rng_state().clone(); value = a.clone()
        compare_tensors(a, b); compare_tensor_maps({'x': a}, {'x': b}); compare_nested_states({'x': a}, {'x': b})
        self.assertTrue(torch.equal(torch.get_rng_state(), saved))
        self.assertTrue(torch.equal(a, value))


if __name__ == '__main__':
    unittest.main()
