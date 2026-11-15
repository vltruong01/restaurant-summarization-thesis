"""Guard minimal-quality boundaries and distinguish prompt/BOS from generated EOS."""
import unittest
from pathlib import Path
import yaml
from caption_basic_quality import evaluate, normalize, termination_metadata

POLICY = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/basic_caption_quality.yaml').read_text())


class BasicQualityTests(unittest.TestCase):
    def test_missing_period_passes_without_changing_raw(self):
        raw = 'a plate of dumplings with sauce and green onions on it'
        a = evaluate({'raw_caption': raw}, POLICY)
        self.assertFalse(a['basic_quality_fail'])
        self.assertEqual(a['normalized_caption'], raw+'.')

    def test_quotes_whitespace_and_empty(self):
        self.assertEqual(normalize('  "a bowl\n of rice."  ', POLICY), '"a bowl of rice."')
        self.assertEqual(normalize('"a bowl of rice"', POLICY), '"a bowl of rice."')
        self.assertEqual(normalize('  ', POLICY), '')
        self.assertTrue(evaluate({'raw_caption': '!!!'}, POLICY)['basic_quality_fail'])

    def test_generic_rule_is_exact_and_specific_caption_stays(self):
        self.assertTrue(evaluate({'raw_caption': 'A plate of food on a table.'}, POLICY)['basic_quality_fail'])
        self.assertFalse(evaluate({'raw_caption': 'a plate of dumplings on a table'}, POLICY)['basic_quality_fail'])
        self.assertFalse(evaluate({'raw_caption': 'a plate of food on a table with a fork and knife'}, POLICY)['basic_quality_fail'])

    def test_recursive_three_word_phrase_is_diagnostic(self):
        a = evaluate({'raw_caption': 'a picture of a picture of a building'}, POLICY)
        self.assertFalse(a['basic_quality_fail'])
        self.assertTrue(a['short_repetition_diagnostic'])
        self.assertTrue(evaluate({'raw_caption': 'a close up of a close up of a burger'}, POLICY)['basic_quality_fail'])

    def test_opt_bos_eos_same_id_excluded_before_eos_check(self):
        a = termination_metadata([2, 30, 40, 50], 1, [2], 1, 3)
        self.assertFalse(a['eos_seen'])
        self.assertTrue(a['token_cap_reached'])

    def test_prompt_eos_and_post_eos_padding_do_not_count_as_generated(self):
        a = termination_metadata([2, 4, 5, 60, 2, 1, 1], 3, [2], 1, 4)
        self.assertTrue(a['eos_seen'])
        self.assertEqual(a['effective_generated_tokens'], 2)
        self.assertFalse(a['token_cap_reached'])

    def test_eos_at_cap_not_truncation(self):
        a = termination_metadata([2, 60, 61, 2], 1, [2], 1, 3)
        result = evaluate({'raw_caption': 'a plate of dumplings on a table', **a}, POLICY)
        self.assertTrue(a['token_cap_reached'])
        self.assertFalse(result['basic_quality_fail'])
        a['eos_seen'] = False
        self.assertIn('evidenced_truncation', evaluate({'raw_caption': 'a plate of dumplings on a table', **a}, POLICY)['basic_quality_flags'])

    def test_coco_language_model_newline_eos_is_not_tokenizer_eos(self):
        a = termination_metadata([2, 60, 61, 50118], 1, [50118], 1, 3)
        self.assertTrue(a['eos_seen'])
        self.assertEqual(a['termination_reason'], 'eos_at_cap')
        result = evaluate({'raw_caption': 'a plate of dumplings on a table\n', **a}, POLICY)
        self.assertFalse(result['basic_quality_fail'])


if __name__ == '__main__':
    unittest.main()
