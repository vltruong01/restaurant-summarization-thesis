"""CPU-only tests of the narrow terminal repair using pinned HF beam utilities.

No real model, forward, generation, training, network, or held-out data is used.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HF_HUB_OFFLINE'] = '1'

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import stage_i_repetition_guard_v1 as v1
import stage_i_repetition_guard_v2 as v2


class SyntheticTokenizer:
    eos_token_id = 2
    pad_token_id = 1
    pieces = {3: 'The ', 4: 'service ', 5: 'service', 6: 'staff', 7: '.',
              8: 'Good ', 9: 'food', 16: 'Several reviews praise the ',
              17: 'scallops', 18: ' and '}

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ''.join(self.pieces.get(i, 'x') for i in ids if i not in (0, 1, 2))


class TerminalRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers.generation.utils import GenerationMixin
        cls.torch = torch
        cls.mixin = GenerationMixin
        cls.tokenizer = SyntheticTokenizer()

    def prefix(self, pieces, width=767):
        ids = [2] + list(pieces)
        return [2] + [1] * (width - len(ids)) + list(pieces)

    def prefixes(self, good_rows=()):
        bad = [16, 17, 18, 17]  # Pending repeated list item, not yet closed.
        good = [8, 9]
        return self.torch.tensor([self.prefix(good if i in good_rows else bad)
                                  for i in range(4)], dtype=self.torch.long)

    def forced_scores(self):
        result = self.torch.full((4, 32), -float('inf'), dtype=self.torch.float32)
        result[:, 2] = 0.
        return result

    def harness(self):
        torch, mixin = self.torch, self.mixin
        class Harness:
            _gather_beams = staticmethod(mixin._gather_beams)
            _get_top_k_continuations = mixin._get_top_k_continuations
            def _update_finished_beams(self, *args, **kwargs):
                self.original_arguments = (args, kwargs)
                self.original_result = mixin._update_finished_beams(self, *args, **kwargs)
                return self.original_result
        return Harness()

    def finalize_with_pinned_utilities(self, guard, prefixes, processed, prior_finished=False):
        """Actual pinned top8/finished-beam arithmetic; no model or generate."""
        torch = self.torch
        host = self.harness()
        running = torch.full((1, 4, 768), 1, dtype=torch.long)
        running[0, :, :767] = prefixes
        running_indices = torch.zeros((1, 4, 767), dtype=torch.int32)
        accumulated = processed.reshape(1, -1) + torch.zeros((1, 128))
        top_scores, top_sequences, top_indices = host._get_top_k_continuations(
            accumulated_log_probs=accumulated, running_sequences=running,
            running_beam_indices=running_indices, cur_len=767, decoder_prompt_len=1,
            do_sample=False, beams_to_keep=8, num_beams=4, vocab_size=32, batch_size=1,
        )
        # The unchanged safety hook ignores impossible -Inf backup slots.
        v2.SelectedCandidateSafetyCheck(guard)(top_sequences.reshape(8, 768))
        sequences = torch.full((1, 4, 768), 1, dtype=torch.long)
        sequences[:, :, 0] = 2
        scores = torch.full((1, 4), -1e9)
        indices = torch.full((1, 4, 767), -1, dtype=torch.int32)
        finished = torch.zeros((1, 4), dtype=torch.bool)
        if prior_finished:
            path = [2, 8, 9, 7, 2]
            sequences[0, 0, :len(path)] = torch.tensor(path)
            scores[0, 0] = -.125
            finished[0, 0] = True
            indices[0, 0, :len(path) - 1] = 0
        arguments = dict(
            sequences=sequences, topk_running_sequences=top_sequences,
            beam_scores=scores, topk_log_probs=top_scores,
            beam_indices=indices, topk_running_beam_indices=top_indices,
            is_sent_finished=finished,
            next_token_hits_stopping_criteria=torch.ones((1, 8), dtype=torch.bool),
            top_num_beam_mask=torch.tensor([True] * 4 + [False] * 4),
            num_beams=4, cur_len=767, decoder_prompt_len=1,
            length_penalty=1.0, early_stopping=True,
        )
        with v2.FinishedBeamObserver(host, guard) as observer:
            result = host._update_finished_beams(**arguments)
            self.assertIs(result, host.original_result)
            self.assertIs(host.original_arguments[1]['sequences'], arguments['sequences'])
            self.assertEqual(observer.last_snapshot['sequences'].data_ptr(), result[0].data_ptr())
        # Reproduce the actual pinned HF output crop, including the decoder
        # prompt. This must not manufacture a completed path by finding EOS.
        generated_length = int(((result[2][0, 0] + 1).bool()).sum())
        returned = result[0][0, 0, :1 + generated_length].tolist()
        return host, observer, result, returned

    def test_surface_policy_and_selection_safety_are_exact_v1_objects(self):
        self.assertIs(v2.analyze, v1.analyze)
        self.assertIs(v2.SelectedCandidateSafetyCheck, v1.SelectedCandidateSafetyCheck)
        self.assertIs(v2.TerminalRepetitionGuardLogitsProcessor, v2.RepetitionGuardLogitsProcessor)

    def test_terminal_rows_pruned_property_reports_only_rejected_terminal_rows(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        self.assertEqual(guard.terminal_rows_pruned, 0)
        guard(self.prefixes(good_rows=(3,)), self.forced_scores())
        self.assertEqual(guard.terminal_rows_pruned, 3)
        self.assertEqual(guard.terminal_rows_pruned,
                         sum(item['event'] == 'TERMINAL_EOS_ROW_REJECTED'
                             for item in guard.terminal_row_decisions))
        with self.assertRaises(AttributeError):
            guard.terminal_rows_pruned = 0

    def test_three_dead_terminal_rows_one_legal_eos_finishes(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        prefixes, original = self.prefixes(good_rows=(3,)), self.forced_scores()
        prefix_copy, score_copy = prefixes.clone(), original.clone()
        processed = guard(prefixes, original)
        self.assertTrue(self.torch.isneginf(processed[:3]).all().item())
        self.assertEqual(float(processed[3, 2]), 0.)
        self.assertEqual(guard.tested_candidates, 4)
        self.assertEqual(guard.blocked_candidates, 3)
        self.assertTrue(self.torch.equal(prefixes, prefix_copy))
        self.assertTrue(self.torch.equal(original, score_copy))
        host, observer, result, returned = self.finalize_with_pinned_utilities(guard, prefixes, processed)
        proof = observer.verify_returned(returned)
        self.assertTrue(proof['actual_finished_beam_verified'])
        self.assertEqual(proof['completion_origin'], 'terminal_max_length_EOS')
        self.assertEqual(proof['generated_EOS_position'], 767)
        self.assertEqual(proof['terminal_rows_rejected'], 3)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_one_dead_three_legal_rows_evaluates_all_rows(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        result = guard(self.prefixes(good_rows=(1, 2, 3)), self.forced_scores())
        self.assertTrue(self.torch.isneginf(result[0]).all().item())
        self.assertTrue(self.torch.isfinite(result[1:, 2]).all().item())
        self.assertEqual(len(guard.terminal_row_decisions), 4)

    def test_all_terminal_rows_dead_can_return_prior_genuinely_finished(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        prefixes = self.prefixes()
        processed = guard(prefixes, self.forced_scores())
        self.assertTrue(self.torch.isneginf(processed).all().item())
        _, observer, result, returned = self.finalize_with_pinned_utilities(
            guard, prefixes, processed, prior_finished=True)
        proof = observer.verify_returned(returned)
        self.assertEqual(proof['completion_origin'], 'earlier_finished_EOS')
        self.assertEqual(proof['actual_normalized_sequence_score'], -.125)
        self.assertEqual(returned, [2, 8, 9, 7, 2])
        self.assertEqual(proof['terminal_rows_rejected'], 4)

    def test_all_terminal_rows_dead_placeholder_is_typed_case_failure(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        prefixes = self.prefixes()
        processed = guard(prefixes, self.forced_scores())
        _, observer, result, returned = self.finalize_with_pinned_utilities(guard, prefixes, processed)
        self.assertFalse(bool(result[3][0, 0]))
        self.assertEqual(float(result[1][0, 0]), -1e9)
        with self.assertRaisesRegex(v2.NoCompliantCompletion, 'placeholders'):
            observer.verify_returned(returned)

    def test_terminal_trace_records_every_row_without_nonstandard_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.jsonl'
            guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '', trace_path=path)
            guard(self.prefixes(good_rows=(3,)), self.forced_scores())
            records = [json.loads(line, parse_constant=lambda value: self.fail(value))
                       for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(r['prefix_length'] == 767 and len(r['prefix_ids']) == 767 for r in records))
            self.assertFalse(any(r['EOS_unmasked_or_bypassed'] for r in records))

    def test_terminal_contract_rejects_non_eos_finite_nonzero_nan_inf_or_shape(self):
        mutations = (
            lambda scores: scores.__setitem__((0, 3), -5.),
            lambda scores: scores.__setitem__((0, 2), -1.),
            lambda scores: scores.__setitem__((0, 2), float('nan')),
            lambda scores: scores.__setitem__((0, 2), float('inf')),
            lambda scores: scores.__setitem__((0, 2), -float('inf')),
        )
        for mutate in mutations:
            scores = self.forced_scores()
            mutate(scores)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')(self.prefixes(), scores)
        with self.assertRaises(ValueError):
            v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')(self.prefixes()[:3], self.forced_scores()[:3])

    def test_prior_invalid_prefix_still_fails_instead_of_being_pruned(self):
        prefixes = self.prefixes()
        prefixes[0] = self.torch.tensor(self.prefix([8, 9, 7, 8, 9, 7]))
        with self.assertRaisesRegex(ValueError, 'Invalid prior terminal prefix'):
            v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')(prefixes, self.forced_scores())

    def test_terminal_call_cannot_silently_retry(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        guard(self.prefixes(good_rows=(3,)), self.forced_scores())
        with self.assertRaisesRegex(ValueError, 'twice'):
            guard(self.prefixes(good_rows=(3,)), self.forced_scores())

    def test_nonterminal_branch_outputs_counters_and_trace_bytes_match_v1(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            a = v1.RepetitionGuardLogitsProcessor(self.tokenizer, '', trace_path=Path(tmp) / 'v1.jsonl')
            b = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '', trace_path=Path(tmp) / 'v2.jsonl')
            prefixes = torch.tensor([self.prefix([3, 4, 5], width=12)] * 4)
            scores = torch.full((4, 32), -float('inf'))
            scores[:, 7] = 100.
            for token in range(20, 32):
                scores[:, token] = 90. - token
            x, y = a(prefixes, scores), b(prefixes, scores)
            self.assertTrue(torch.equal(x, y))
            for name in ('calls', 'tested_candidates', 'blocked_candidates', 'selected_continuations_checked'):
                self.assertEqual(getattr(a, name), getattr(b, name))
            self.assertEqual((Path(tmp) / 'v1.jsonl').read_bytes(), (Path(tmp) / 'v2.jsonl').read_bytes())
            self.assertFalse(b.terminal_seen)
            for key in a.last_scores_by_prefix:
                self.assertTrue(torch.equal(a.last_scores_by_prefix[key], b.last_scores_by_prefix[key]))

    def test_nonterminal_failure_behavior_remains_v1(self):
        prefix = self.torch.tensor([self.prefix([8, 9], width=12)] * 4)
        scores = self.torch.full((4, 32), -float('inf'))
        for cls in (v1.RepetitionGuardLogitsProcessor, v2.RepetitionGuardLogitsProcessor):
            with self.subTest(cls=cls), self.assertRaisesRegex(ValueError, 'All scores masked'):
                cls(self.tokenizer, '')(prefix, scores)

    def test_observer_restores_method_after_normal_and_exception_paths(self):
        host = self.harness()
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        original_function = host._update_finished_beams.__func__
        with v2.FinishedBeamObserver(host, guard):
            self.assertIn('_update_finished_beams', host.__dict__)
        self.assertNotIn('_update_finished_beams', host.__dict__)
        self.assertIs(host._update_finished_beams.__func__, original_function)
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            with v2.FinishedBeamObserver(host, guard):
                raise RuntimeError('synthetic interruption')
        self.assertNotIn('_update_finished_beams', host.__dict__)

    def test_observer_preserves_existing_instance_override(self):
        host = self.harness()
        override = host._update_finished_beams
        host._update_finished_beams = override
        with v2.FinishedBeamObserver(host, v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')):
            pass
        self.assertIs(host._update_finished_beams, override)

    def test_observer_unbound_method_and_tampering_are_fatal(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            with v2.FinishedBeamObserver(object(), v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')):
                pass
        host = self.harness()
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            with v2.FinishedBeamObserver(host, v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')):
                host._update_finished_beams = lambda: None
        self.assertNotIn('_update_finished_beams', host.__dict__)

    def test_observer_requires_actual_snapshot_and_path_binding(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        with self.assertRaisesRegex(ValueError, 'No actual'):
            v2.FinishedBeamObserver(self.harness(), guard).verify_returned([2, 8, 9, 2])
        prefixes = self.prefixes(good_rows=(3,))
        _, observer, _, returned = self.finalize_with_pinned_utilities(guard, prefixes, guard(prefixes, self.forced_scores()))
        altered = returned.copy()
        altered[-2] = 6
        with self.assertRaisesRegex(ValueError, 'not bound'):
            observer.verify_returned(altered)
        with self.assertRaises(ValueError):
            observer.verify_returned(returned + [6])
        with self.assertRaisesRegex(ValueError, 'lacks a generated EOS'):
            observer.verify_returned(returned[:-1])

    def test_observer_strict_final_guard_is_not_bypassed(self):
        guard = v2.RepetitionGuardLogitsProcessor(self.tokenizer, '')
        prefixes = self.prefixes()
        _, observer, _, _ = self.finalize_with_pinned_utilities(guard, prefixes, guard(prefixes, self.forced_scores()))
        snapshot = observer.last_snapshot
        path = [2, 8, 9, 7, 8, 9, 7, 2]
        snapshot['sequences'][0, 0, :] = 1
        snapshot['sequences'][0, 0, :len(path)] = self.torch.tensor(path)
        snapshot['finished'][0, 0] = True
        snapshot['scores'][0, 0] = -.5
        with self.assertRaisesRegex(ValueError, 'Guard violation in final sequence'):
            observer.verify_returned(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
