import copy
import unittest
from stage_g_evidence_id_protocol import review_input, resolve_review
from prepare_stage_g_review_evidence_ids import evidence_spans
from run_stage_g_teacher_pilot import validate_output
from prepare_stage_g_teacher_pilot import read, ROOT


class SourceSpanEvidence(unittest.TestCase):
    def fixtures(self):
        job = {'business_id': 'A', 'task': 'review_consensus', 'evidence': {'reviews': [
            {'review_marker': 'REV_1', 'review_id': 'r1', 'stars': 4, 'date': '2020',
             'text': 'Good food.  Friendly staff!\nSecond paragraph.'},
            {'review_marker': 'REV_2', 'review_id': 'r2', 'stars': 5, 'date': '2021',
             'text': 'Staff were friendly.\n\nPrice was fair.'}]}}
        output = {'business_id': 'A', 'claims': [{'text': 'Reviewers praise friendly staff.',
            'polarity': 'positive', 'support': [
                {'review_marker': 'REV_1', 'review_id': 'r1', 'evidence_span_ids': ['REV_1_SEG_2']},
                {'review_marker': 'REV_2', 'review_id': 'r2', 'evidence_span_ids': ['REV_2_SEG_1']}]}]}
        return job, output

    def test_complete_input_reconstructed_without_duplicate_text(self):
        job, _ = self.fixtures()
        outgoing = review_input(job)
        for original, sent in zip(job['evidence']['reviews'], outgoing['reviews']):
            self.assertEqual(''.join(s['text'] for s in sent['evidence_spans']), original['text'])
            self.assertNotIn('text', sent)

    def test_exact_source_offsets_quotes_and_unchanged_claim(self):
        job, output = self.fixtures()
        before = copy.deepcopy(output)
        resolved, provenance = resolve_review(job, output)
        self.assertEqual(resolved['claims'][0]['support'][0]['quote'], 'Friendly staff!\n')
        self.assertEqual(output, before)
        self.assertEqual(resolved['claims'][0]['text'], output['claims'][0]['text'])
        self.assertEqual(provenance[0]['start_character'], 12)
        schema = read(ROOT / 'schemas/stage_g_teacher_pilot_v1.json')['review_consensus']
        self.assertTrue(validate_output(job, resolved, schema)['structural_and_evidence_binding_valid'])

    def test_invented_span_rejected(self):
        job, output = self.fixtures()
        output['claims'][0]['support'][0]['evidence_span_ids'] = ['REV_1_SEG_99']
        with self.assertRaises(ValueError): resolve_review(job, output)

    def test_span_from_wrong_review_rejected(self):
        job, output = self.fixtures()
        output['claims'][0]['support'][0]['evidence_span_ids'] = ['REV_2_SEG_1']
        with self.assertRaises(ValueError): resolve_review(job, output)

    def test_wrong_original_review_ID_rejected(self):
        job, output = self.fixtures()
        output['claims'][0]['support'][0]['review_id'] = 'different'
        with self.assertRaises(ValueError): resolve_review(job, output)

    def test_missing_or_repeated_span_rejected(self):
        job, output = self.fixtures()
        for ids in [[], ['REV_1_SEG_2', 'REV_1_SEG_2']]:
            output['claims'][0]['support'][0]['evidence_span_ids'] = ids
            with self.assertRaises(ValueError): resolve_review(job, output)

    def test_one_review_still_does_not_establish_recurrence(self):
        job, output = self.fixtures()
        output['claims'][0]['support'].pop()
        resolved, _ = resolve_review(job, output)
        schema = read(ROOT / 'schemas/stage_g_teacher_pilot_v1.json')['review_consensus']
        with self.assertRaises(ValueError): validate_output(job, resolved, schema)

    def test_unicode_whitespace_and_offsets_are_exact(self):
        text = 'Caf\u00e9 is good.\r\n\r\nPrice: 3.50.  Nice!\n'
        spans = evidence_spans({'text': text, 'review_marker': 'REV_8'})
        self.assertEqual(''.join(s['exact_source_text'] for s in spans), text)
        for span in spans:
            self.assertEqual(text[span['start_character']:span['end_character']], span['exact_source_text'])


if __name__ == '__main__':
    unittest.main()
