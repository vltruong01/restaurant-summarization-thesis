import copy
import unittest
from stage_g_canonical_review_ids import resolve_review, citation_diagnostics


class CanonicalMetadata(unittest.TestCase):
    def fixture(self):
        job = {'business_id': 'A', 'evidence': {'reviews': [
            {'review_marker': 'REV_1', 'review_id': 'source-1', 'text': 'Good bread.'},
            {'review_marker': 'REV_2', 'review_id': 'source-2', 'text': 'Happy hour is good.'}]}}
        output = {'business_id': 'A', 'claims': [{'text': 'Bread is praised.', 'polarity': 'positive',
            'support': [{'review_marker': 'REV_1', 'review_id': 'mis-copied-redundant-ID',
                         'evidence_span_ids': ['REV_1_SEG_1']}]}]}
        return job, output

    def test_source_marker_and_span_ids_determine_canonical_original_id(self):
        job, output = self.fixture();before = copy.deepcopy(output)
        resolved, provenance = resolve_review(job, output)
        self.assertEqual(output, before)
        self.assertEqual(resolved['claims'][0]['support'][0]['review_id'], 'source-1')
        self.assertEqual(resolved['claims'][0]['support'][0]['quote'], 'Good bread.')
        self.assertEqual(resolved['claims'][0]['text'], output['claims'][0]['text'])
        self.assertTrue(provenance[0]['redundant_reported_ID_mismatch'])
        self.assertEqual(citation_diagnostics(job, output)[0]['teacher_reported_review_id'], 'mis-copied-redundant-ID')

    def test_wrong_marker_span_pair_still_rejected(self):
        job, output = self.fixture()
        output['claims'][0]['support'][0]['evidence_span_ids'] = ['REV_2_SEG_1']
        with self.assertRaises(ValueError): resolve_review(job, output)

    def test_unknown_span_still_rejected(self):
        job, output = self.fixture()
        output['claims'][0]['support'][0]['evidence_span_ids'] = ['REV_1_SEG_99']
        with self.assertRaises(ValueError): resolve_review(job, output)

    def test_other_business_still_rejected(self):
        job, output = self.fixture();output['business_id'] = 'B'
        with self.assertRaises(ValueError): resolve_review(job, output)

    def test_matching_metadata_has_no_diagnostics(self):
        job, output = self.fixture();output['claims'][0]['support'][0]['review_id'] = 'source-1'
        self.assertEqual(citation_diagnostics(job, output), [])


if __name__ == '__main__':
    unittest.main()
