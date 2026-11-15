"""Synthetic regression checks for raw diagnostics, routing and empty-caption behavior."""
import unittest,copy
from s1000_final_eval_core_v1 import *
from stage_k_reference_free_analysis_v1 import component_diagnostics
class ReferenceFreeTests(unittest.TestCase):
    def sample(self,text='[VISUAL_EVIDENCE]\nA red plate. A red plate.',ids=None):
        return dict(text=text,task='visual',token_ids=ids or [2,10,11,2],independent_generation=True,generation_status='SUCCESS'),dict(source='[TASK_VISUAL_EVIDENCE]\n[VISUAL_CONTEXT]\n[IMG_CAP_1]\nA red plate.')
    def test_repeated_full_sentences(self):
        r,c=self.sample();d=component_diagnostics(r,c,1)
        self.assertEqual(d['repeated_full_sentence_occurrences'],1);self.assertGreater(d['repeated_3gram_fraction'],0)
    def test_decoder_start_is_not_termination(self):
        r,c=self.sample(ids=[2,10,11]);d=component_diagnostics(r,c,1)
        self.assertFalse(d['EOS_seen_after_decoder_start']);self.assertFalse(d['last_token_EOS'])
    def test_forced_EOS_at_cap_not_proof_complete(self):
        r,c=self.sample(ids=[2]+[10]*766+[2]);d=component_diagnostics(r,c,1)
        self.assertTrue(d['at_max_length']);self.assertTrue(d['forced_EOS_at_cap_possible'])
    def test_empty_caption_generated_claim_flag(self):
        r,c=self.sample();c['source']='[TASK_VISUAL_EVIDENCE]\n[VISUAL_CONTEXT]\n'
        d=component_diagnostics(r,c,0);self.assertTrue(d['empty_caption_nonempty_output']);self.assertEqual(d['source_lexical_precision_proxy'],0.)
    def test_empty_output_and_wrong_section(self):
        r,c=self.sample(text='[VISUAL_EVIDENCE]\n');self.assertTrue(component_diagnostics(r,c,1)['empty'])
        r['text']='[REVIEW_CONSENSUS]\nGood food.';self.assertTrue(component_diagnostics(r,c,1)['malformed'])
    def test_caption_markers_not_lexical_evidence(self):
        r,c=self.sample(text='[VISUAL_EVIDENCE]\nIMG CAP TASK VISUAL CONTEXT')
        self.assertEqual(component_diagnostics(r,c,1)['source_lexical_precision_proxy'],0.)
    def test_review_no_visual_penalty(self):
        r,c=self.sample(text='[REVIEW_CONSENSUS]\nGood food.');r['task']='review';c['source']='[TASK_REVIEW_CONSENSUS]\n[BUSINESS_INFORMATION]\n[REVIEWS]\n[REV_1] stars=4\nGood food.'
        d=component_diagnostics(r,c,0);self.assertIsNone(d['caption_count']);self.assertFalse(d['empty_caption_nonempty_output'])
if __name__=='__main__':unittest.main(verbosity=2)
