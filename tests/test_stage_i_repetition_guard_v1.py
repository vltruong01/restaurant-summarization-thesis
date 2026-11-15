"""Independent adversarial guard tests: CPU scores and cached tokenizer only.

No model, forward, generation, training, network fetch, or held-out input is used.
Synthetic examples test the declared surface policy, not factual correctness.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HF_HUB_OFFLINE'] = '1'

import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import stage_i_repetition_guard_v1 as guard_module
from stage_i_repetition_guard_v1 import (
    RepetitionGuardLogitsProcessor, SelectedCandidateSafetyCheck, analyze,
)


def rules(report):
    return {item['rule'] for item in report['violations']}


def flags(report):
    return {item['rule'] for item in report['flags']}


class SurfacePolicyTests(unittest.TestCase):
    def test_adjacent_repeat_requires_closed_second_word(self):
        self.assertNotIn('adjacent_word', rules(analyze('The service service')))
        self.assertIn('adjacent_word', rules(analyze('The service service.')))
        self.assertIn('adjacent_word', rules(analyze('The service service', finalize=True)))

    def test_word_extension_remains_legal(self):
        self.assertNotIn('adjacent_word', rules(analyze('The service services.')))

    def test_case_whitespace_and_unicode_word_detection(self):
        self.assertIn('adjacent_word', rules(analyze('Good GOOD food.')))
        self.assertIn('adjacent_word', rules(analyze('café\tcafé.')))
        self.assertIn('adjacent_word', rules(analyze('It\u2019s it\u2019s good.')))

    def test_sentence_and_line_boundaries_are_not_literal_stutters(self):
        self.assertNotIn('adjacent_word', rules(analyze('service.Service was good.')))
        self.assertNotIn('adjacent_word', rules(analyze('service\nservice was good.')))
        self.assertNotIn('adjacent_word', rules(analyze('service, service was good.')))

    def test_exact_source_reduplication_is_logged_not_verified(self):
        source = 'The menu lists Bang Bang shrimp.'
        result = analyze('Bang Bang shrimp was popular.', source)
        self.assertNotIn('adjacent_word', rules(result))
        self.assertEqual(len(result['source_reduplication_exceptions']), 1)
        item = result['source_reduplication_exceptions'][0]
        self.assertEqual(item['source_quote'], 'Bang Bang')
        self.assertEqual(item['basis'], 'verbatim_source_pair_not_verified_name')
        self.assertEqual(source[item['source_start']:item['source_end']], item['source_quote'])

    def test_absent_source_pair_does_not_get_silent_exception(self):
        self.assertIn('adjacent_word', rules(analyze('Bang Bang shrimp.', 'The menu lists shrimp.')))
        self.assertIn('adjacent_word', rules(analyze('Bang Bang shrimp.', 'Bang, Bang shrimp.')))

    def test_fixed_grammatical_double_is_flagged_not_automatically_corrected(self):
        for text in ('Customers had had enough.', 'They said that that dish was good.'):
            with self.subTest(text=text):
                for source in ('', text):
                    result = analyze(text, source)
                    self.assertNotIn('adjacent_word', rules(result))
                    self.assertIn('grammatical_reduplication_not_blocked', flags(result))
        for text in ('Customers had had had enough.', 'They said that that that dish was good.'):
            with self.subTest(text=text):
                self.assertIn('adjacent_word', rules(analyze(text)))

    def test_source_exception_does_not_disable_sentence_guard(self):
        result = analyze('Bang Bang shrimp. Bang Bang shrimp.', 'Bang Bang shrimp.')
        self.assertIn('duplicate_sentence', rules(result))
        self.assertNotIn('adjacent_word', rules(result))

    def test_source_double_does_not_authorize_unbounded_reduplication(self):
        result = analyze('Bang Bang Bang shrimp.', 'The menu lists Bang Bang shrimp.')
        self.assertIn('adjacent_word', rules(result))

    def test_source_verbatim_longer_run_exception_is_bounded(self):
        source = 'A reviewer wrote no no no in the review.'
        self.assertNotIn('adjacent_word', rules(analyze('no no no.', source)))
        self.assertIn('adjacent_word', rules(analyze('no no no no.', source)))

    def test_completed_duplicate_sentence_normalizes_words(self):
        result = analyze('The food was good. THE  food\nwas good!')
        self.assertIn('duplicate_sentence', rules(result))
        self.assertEqual(sum(v['rule'] == 'duplicate_sentence' for v in result['violations']), 1)

    def test_unfinished_duplicate_sentence_is_only_closed_by_eos(self):
        text = 'The food was good. The food was good'
        self.assertNotIn('duplicate_sentence', rules(analyze(text)))
        self.assertIn('duplicate_sentence', rules(analyze(text, finalize=True)))

    def test_sentence_prefix_with_contrast_is_flagged_not_rejected(self):
        result = analyze('The fish was good. The fish was good, but the tacos were poor.')
        self.assertNotIn('duplicate_sentence', rules(result))
        self.assertIn('phrase_extension_after_repeat', flags(result))

    def test_sentence_prefix_filler_remains_explicit_review_flag(self):
        result = analyze('The food was good. The food was good and also very good overall.')
        self.assertNotIn('duplicate_sentence', rules(result))
        self.assertIn('phrase_extension_after_repeat', flags(result))

    def test_different_sentences_preserve_opinion_contrast(self):
        result = analyze('Some reviewers liked the food. Other reviewers disliked the food.')
        self.assertFalse(result['violations'])

    def test_decimals_abbreviations_and_initials_do_not_make_false_sentences(self):
        for text in ('Dr. A was helpful. Dr. B was helpful.',
                     'It costs 3.50. It costs 3.75.', 'Ms. A was kind. Ms. B was kind.'):
            with self.subTest(text=text):
                self.assertNotIn('duplicate_sentence', rules(analyze(text)))

    def test_completed_duplicate_list_items(self):
        for text in ('Service was slow, slow.', 'They served red curry, red curry.',
                     'Service was slow or slow.', 'Service was slow and slow.'):
            with self.subTest(text=text):
                self.assertIn('duplicate_list_item', rules(analyze(text)))

    def test_shared_noun_and_unfinished_list_item_are_not_rejected(self):
        for text in ('They served fish, fish tacos.', 'They served fish and fish tacos.',
                     'Service was slow, slow'):
            with self.subTest(text=text):
                self.assertNotIn('duplicate_list_item', rules(analyze(text)))
        self.assertIn('duplicate_list_item', rules(analyze('Service was slow, slow', finalize=True)))

    def test_negated_list_prefix_preserves_positive_negative_contrast(self):
        for text in ('The service was not slow, slow service was rare.',
                     'The food was not good, good service helped.',
                     'Some said the food was not good, but others called it good.'):
            with self.subTest(text=text):
                self.assertNotIn('duplicate_list_item', rules(analyze(text)))

    def test_section_histories_are_separate(self):
        text = '[REVIEW_CONSENSUS]\nThe food was good.\n[VISUAL_EVIDENCE]\nThe food was good.'
        self.assertNotIn('duplicate_sentence', rules(analyze(text)))

    def test_repeated_marker_does_not_reset_history(self):
        text = '[REVIEW_CONSENSUS]\nThe food was good.\n[REVIEW_CONSENSUS]\nThe food was good.'
        result = analyze(text)
        self.assertIn('duplicate_sentence', rules(result))
        self.assertIn('repeated_marker', flags(result))

    def test_marker_shared_prefix_and_empty_visual_representation_are_untouched(self):
        for text in ('[REVIEW_CONSENSUS]\nGood food.\n[VISUAL_EVIDENCE]\nNo reliable visual evidence.',
                     '[REVIEW_CONSENSUS]\nGood food.\n[VISUAL_', '[REVIEW_'):
            with self.subTest(text=text):
                self.assertFalse(analyze(text)['violations'])

    def test_unknown_marker_only_flagged(self):
        result = analyze('[OTHER_MARKER]\nGood food.')
        self.assertIn('unexpected_marker', flags(result))
        self.assertFalse(result['violations'])

    def test_bracketed_normal_prose_does_not_bypass_guard(self):
        result = analyze('[The food was good. The food was good.]')
        self.assertIn('duplicate_sentence', rules(result))
        self.assertTrue(result['flags'])

    def test_empty_inputs_and_non_string_contract(self):
        self.assertFalse(analyze('')['violations'])
        for text, source in ((None, ''), ('text', None), ([], '')):
            with self.subTest(text=text, source=source), self.assertRaises(ValueError):
                analyze(text, source)

    def test_raw_input_and_source_never_rewritten(self):
        text = '[REVIEW_CONSENSUS]\nThe service service was good. Good food. Good food.'
        source = 'Customer said service was good.'
        snapshot = copy.deepcopy((text, source))
        result = analyze(text, source, finalize=True)
        self.assertEqual((text, source), snapshot)
        self.assertEqual(set(result), {'violations', 'flags', 'source_reduplication_exceptions'})
        json.dumps(result, allow_nan=False)


class CachedLedTokenizerCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LEDTokenizer
        cls.torch = torch
        cls.tokenizer = LEDTokenizer.from_pretrained(
            'allenai/led-base-16384', revision='38335783885b338d93791936c54bb4be46bebed9',
            local_files_only=True,
        )
        cls.vocab_size = len(cls.tokenizer)
        cls.good_tokens = []
        for word in ('s', 'ing', 'ly', 'ful', 'ness', 'fulfilling', 'great', 'kind',
                     'nice', 'tasty', 'helpful', 'prompt', 'cozy', 'fresh', 'new',
                     'lovely', 'pleasant', 'flavorful', 'staff', 'food'):
            # Non-whitespace continuations extend the pending last word. They
            # are synthetic legal options, not claims of grammatical quality.
            ids = cls.tokenizer.encode(word, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in cls.good_tokens:
                cls.good_tokens.append(ids[0])
        if len(cls.good_tokens) < 12:
            raise AssertionError('Need 12 actual one-token legal continuations for CPU fixtures')
        cls.dot = cls.single('.')
        cls.eos = cls.tokenizer.eos_token_id

    @classmethod
    def single(cls, text):
        ids = cls.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise AssertionError(('Fixture expected one BPE token', text, ids))
        return ids[0]

    def prefixes(self, texts):
        rows = [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]
        width = max(map(len, rows))
        return self.torch.tensor([[self.tokenizer.pad_token_id] * (width - len(r)) + r for r in rows],
                                 dtype=self.torch.long, device='cpu')

    def scores(self, preferred=None):
        scores = self.torch.full((4, self.vocab_size), -float('inf'), dtype=self.torch.float32)
        for i, token in enumerate(self.good_tokens[:12]):
            scores[:, token] = 90. - i
        if preferred is not None:
            scores[:, preferred] = 100.
        return scores

    def test_cached_actual_bpe_space_and_no_space_word_variants(self):
        self.assertNotEqual(self.tokenizer.encode('service', add_special_tokens=False),
                            self.tokenizer.encode(' service', add_special_tokens=False))
        ids = self.tokenizer.encode('café café.', add_special_tokens=False)
        decoded = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        self.assertIn('adjacent_word', rules(analyze(decoded)))

    def test_masks_duplicate_closing_token_without_modifying_inputs(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        ids = self.prefixes(['The service service'] * 4)
        scores = self.scores(self.dot)
        original_ids, original_scores = ids.clone(), scores.clone()
        result = processor(ids, scores)
        self.assertTrue(self.torch.isneginf(result[:, self.dot]).all().item())
        self.assertTrue(self.torch.equal(ids, original_ids))
        self.assertTrue(self.torch.equal(scores, original_scores))
        for token in self.good_tokens[:8]:
            self.assertTrue(self.torch.equal(result[:, token], scores[:, token]))

    def test_eos_finalizes_pending_duplicate(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        result = processor(self.prefixes(['The service service'] * 4), self.scores(self.eos))
        self.assertTrue(self.torch.isneginf(result[:, self.eos]).all().item())

    def test_subword_extension_not_blanket_word_ban(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        token = self.single('s')
        result = processor(self.prefixes(['The service service'] * 4), self.scores(token))
        self.assertTrue(self.torch.isfinite(result[:, token]).all().item())

    def test_actual_beam_reordering_has_no_row_history_leak(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        ids = self.prefixes(['The service service', 'The service staff',
                             'The food food', 'The food staff'])
        scores = self.scores(self.dot)
        first = processor(ids, scores)
        order = self.torch.tensor([3, 0, 1, 2])
        second = processor(ids[order], scores[order])
        self.assertTrue(self.torch.equal(second, first[order]))
        self.assertTrue(self.torch.isneginf(first[[0, 2], self.dot]).all().item())
        self.assertTrue(self.torch.isfinite(first[[1, 3], self.dot]).all().item())

    def test_eos_after_safe_completion_and_source_exception_remain_allowed(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, 'Bang Bang shrimp.')
        result = processor(self.prefixes(['Bang Bang shrimp.'] * 4), self.scores(self.eos))
        self.assertTrue(self.torch.isfinite(result[:, self.eos]).all().item())

    def test_all_inf_and_all_forbidden_fail_without_fallback(self):
        ids = self.prefixes(['The service service'] * 4)
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        scores = self.torch.full((4, self.vocab_size), -float('inf'))
        with self.assertRaisesRegex(ValueError, 'All scores masked'):
            processor(ids, scores)
        scores[:, self.dot] = 1.
        with self.assertRaisesRegex(ValueError, 'No finite legal continuation'):
            processor(ids, scores)

    def test_nan_and_positive_inf_fail_before_frontier(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value):
                scores = self.scores()
                scores[0, self.dot] = value
                with self.assertRaisesRegex(ValueError, 'NaN/\+Inf'):
                    RepetitionGuardLogitsProcessor(self.tokenizer, '')(
                        self.prefixes(['Good food'] * 4), scores)

    def test_wrong_beam_and_eos_contracts_rejected(self):
        with self.assertRaises(ValueError):
            RepetitionGuardLogitsProcessor(self.tokenizer, '', num_beams=1)
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        with self.assertRaises(ValueError):
            processor(self.prefixes(['Good food'] * 3), self.scores()[:3])

    def test_invalid_prior_prefix_and_final_fail_without_rewrite(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        with self.assertRaisesRegex(ValueError, 'invalid beam prefix'):
            processor(self.prefixes(['The service service.'] * 4), self.scores())
        ids = self.tokenizer.encode('Good food. Good food.', add_special_tokens=False)
        snapshot = ids.copy()
        with self.assertRaisesRegex(ValueError, 'final sequence'):
            processor.validate_final(ids)
        self.assertEqual(ids, snapshot)

    def test_strict_json_trace_records_masks_and_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.jsonl'
            processor = RepetitionGuardLogitsProcessor(self.tokenizer, 'Only source.', trace_path=path)
            processor(self.prefixes(['The service service'] * 4), self.scores(self.dot))
            records = [json.loads(line, parse_constant=lambda x: self.fail('Nonstandard JSON: ' + x))
                       for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(records)
            self.assertTrue(all(r['event'] == 'MASK' for r in records))
            self.assertTrue(all(r['source_sha256'] == processor.source_sha256 for r in records))
            self.assertTrue(all(r['rejected'][0]['token_id'] == self.dot for r in records))

    def test_actual_selected_top8_safety_and_impossible_slots(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        ids = self.prefixes(['Good food'] * 4)
        processor(ids, self.scores())
        selected = self.torch.stack([
            self.torch.cat((ids[i % 4], self.torch.tensor([self.good_tokens[i]]))) for i in range(8)
        ])
        result = SelectedCandidateSafetyCheck(processor)(selected)
        self.assertEqual(result.shape, (8,))
        self.assertFalse(result.any().item())
        self.assertEqual(processor.selected_continuations_checked, 8)
        # -Inf backup slots are impossible HF transitions, not legal selections.
        selected[:, -1] = self.dot
        SelectedCandidateSafetyCheck(processor)(selected)
        self.assertEqual(processor.selected_continuations_checked, 8)

    def test_fp32_tie_risk_injected_illegal_untested_selected_candidate_fails(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        ids = self.prefixes(['The service service'] * 4)
        scores = self.scores()
        # Lower-scoring tail lies below the eight tested legal candidates.
        scores[:, self.dot] = -100.
        result = processor(ids, scores)
        self.assertTrue(self.torch.isfinite(result[:, self.dot]).all().item())
        selected = self.torch.stack([
            self.torch.cat((ids[i % 4], self.torch.tensor([self.dot if i == 0 else self.good_tokens[i]])))
            for i in range(8)
        ])
        with self.assertRaisesRegex(ValueError, 'Selected continuation violates'):
            SelectedCandidateSafetyCheck(processor)(selected)

    def test_duplicate_prefix_unequal_ranks_cannot_hide_available_illegal_tail(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        ids = self.prefixes(['The service service'] * 4)
        scores = self.scores()
        scores[:, self.dot] = -100.  # Untested, still finite in early rows.
        scores[-1, self.dot] = 100.  # Tested and masked in final same-prefix row.
        result = processor(ids, scores)
        self.assertTrue(self.torch.isfinite(result[0, self.dot]).item())
        self.assertTrue(self.torch.isneginf(result[-1, self.dot]).item())
        selected = self.torch.stack([
            self.torch.cat((ids[i % 4], self.torch.tensor([self.dot if i == 0 else self.good_tokens[i]])))
            for i in range(8)
        ])
        with self.assertRaisesRegex(ValueError, 'Selected continuation violates'):
            SelectedCandidateSafetyCheck(processor)(selected)

    def test_selected_hook_rejects_unknown_prefix_or_changed_top8_shape(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        ids = self.prefixes(['Good food'] * 4)
        processor(ids, self.scores())
        selected = self.torch.stack([
            self.torch.cat((ids[i % 4], self.torch.tensor([self.good_tokens[i]]))) for i in range(8)
        ])
        with self.assertRaisesRegex(ValueError, 'top8'):
            SelectedCandidateSafetyCheck(processor)(selected[:4])
        selected[0, 0] = self.tokenizer.unk_token_id
        with self.assertRaisesRegex(ValueError, 'not bound'):
            SelectedCandidateSafetyCheck(processor)(selected)

    def test_typical_four_row_full_vocab_cpu_benchmark_and_no_cuda(self):
        processor = RepetitionGuardLogitsProcessor(self.tokenizer, '')
        ids = self.prefixes(['The staff were friendly'] * 4)
        scores = self.torch.full((4, self.vocab_size), -100.)
        for i, token in enumerate(self.good_tokens[:12]):
            scores[:, token] = -float(i)
        begin = time.perf_counter()
        result = processor(ids, scores)
        elapsed = time.perf_counter() - begin
        self.assertEqual(tuple(result.shape), (4, self.vocab_size))
        self.assertEqual(result.device.type, 'cpu')
        self.assertEqual(processor.tested_candidates, 32)
        self.assertFalse(self.torch.cuda.is_initialized())
        print(json.dumps({'offline_cpu_guard_benchmark': {
            'beam_rows': 4, 'vocabulary': self.vocab_size, 'tested_candidates': 32,
            'seconds': elapsed, 'torch_cpu_threads': self.torch.get_num_threads(),
            'model_forward_calls': 0, 'gpu_execution': False,
        }}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
