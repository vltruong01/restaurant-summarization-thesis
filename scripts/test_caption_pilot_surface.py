"""Raw-caption surface checks must flag issues without changing model output."""
import unittest,yaml
from blip2_caption_pilot import ROOT,CONFIG,surface


class CaptionSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))

    def test_missing_punctuation_is_flagged_not_added(self):
        caption='A red bowl with noodles on a table'
        result=surface(caption,self.cfg)
        self.assertIn('no_terminal_punctuation',result['surface_flags'])
        self.assertEqual(caption,'A red bowl with noodles on a table')
        self.assertFalse(result['surface_eligible'])

    def test_explicit_generic_phrase_even_with_punctuation(self):
        result=surface('A plate of food on a table.',self.cfg)
        self.assertGreaterEqual(result['word_count'],5)
        self.assertIn('exact_generic_phrase',result['surface_flags'])

    def test_surface_pass_is_not_semantic_verification(self):
        result=surface('A wooden table with two white plates.',self.cfg)
        self.assertTrue(result['surface_eligible'])
        self.assertEqual(result['grammar_information_quality'],'manual_review_pending')


if __name__=='__main__':unittest.main()
