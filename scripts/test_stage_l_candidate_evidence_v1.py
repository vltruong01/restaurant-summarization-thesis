"""Synthetic source-alignment checks only; never opens participant drafts."""
import copy
import unittest
from import_stage_l_candidate_evidence_v1 import select_candidates


class EvidenceSelection(unittest.TestCase):
    def setUp(self):
        self.source = {'reviews': [{'review_id': 'R', 'text': 'Soup is good. Service is slow.'}], 'photos': [{'photo_id': 'P'}]}
        self.reference = {'review_consensus': 'Soup is praised.', 'visual_evidence': 'A blue bowl.'}
        self.candidate = {'review_claims': [{'output_span': 'Soup is praised.', 'evidence': [{'review_id': 'R', 'quote': 'Soup is good.'}]}],
                          'visual_claims': [{'output_span': 'A blue bowl.', 'evidence': [{'photo_id': 'P', 'direct_image_viewed': True}]}]}

    def test_candidates_cannot_certify_human_photo_inspection(self):
        original = copy.deepcopy(self.candidate)
        result, summary, counts = select_candidates(self.candidate, self.reference, self.source)
        self.assertFalse(result['visual_claims'][0]['evidence'][0]['direct_image_viewed'])
        self.assertEqual(self.candidate, original)
        self.assertEqual(counts['review_links'], 1)
        self.assertEqual(summary['unlinked_review_spans'], [])

    def test_foreign_id_and_nonliteral_quotes_never_imported(self):
        self.candidate['review_claims'][0]['evidence'] = [{'review_id': 'OTHER', 'quote': 'Soup is good.'}, {'review_id': 'R', 'quote': 'Soup was excellent.'}]
        self.candidate['visual_claims'][0]['evidence'][0]['photo_id'] = 'OTHER'
        result, summary, counts = select_candidates(self.candidate, self.reference, self.source)
        self.assertEqual(result, {'review_claims': [], 'visual_claims': []})
        self.assertEqual(counts['bad_links'], 3)
        self.assertEqual(len(summary['unlinked_review_spans']), 1)

    def test_current_prose_is_never_replaced_to_fit_stale_candidate(self):
        self.reference['review_consensus'] = 'Different human revision.'
        before = copy.deepcopy(self.reference)
        result, summary, counts = select_candidates(self.candidate, self.reference, self.source)
        self.assertEqual(result['review_claims'], [])
        self.assertEqual(counts['span_mismatches'], 1)
        self.assertEqual(self.reference, before)
        self.assertTrue(summary['unlinked_review_spans'][0].startswith('[Câu chữ khác'))


if __name__ == '__main__': unittest.main()
