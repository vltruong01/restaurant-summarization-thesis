import copy
import unittest
from pathlib import Path
import json
from run_stage_g_teacher_pilot import validate_output

SCHEMAS = json.loads((Path(__file__).resolve().parents[1] / 'schemas/stage_g_teacher_pilot_v1.json').read_text())


class TeacherEvidenceValidation(unittest.TestCase):
    def review(self):
        job = {'business_id': 'A', 'task': 'review_consensus', 'evidence': {'reviews': [
            {'review_marker': 'REV_1', 'review_id': 'r1', 'text': 'The service was friendly.'},
            {'review_marker': 'REV_2', 'review_id': 'r2', 'text': 'Friendly staff helped us.'}]}}
        output = {'business_id': 'A', 'claims': [{'text': 'Reviewers describe friendly service.',
            'polarity': 'positive', 'support': [
                {'review_marker': 'REV_1', 'review_id': 'r1', 'quote': 'service was friendly'},
                {'review_marker': 'REV_2', 'review_id': 'r2', 'quote': 'Friendly staff'}]}]}
        return job, output

    def visual(self):
        job = {'business_id': 'A', 'task': 'visual_evidence', 'condition': 'filtered',
               'evidence': {'captions': [{'caption_id': 'IMG_CAP_1', 'photo_id': 'p1',
                                          'caption': 'a plate with rice.'}]}}
        output = {'business_id': 'A', 'condition': 'filtered', 'assessments': [
            {'caption_id': 'IMG_CAP_1', 'photo_id': 'p1', 'note': '', 'claims': [
                {'text': 'A plate contains rice.', 'caption_quote': 'plate with rice',
                 'verdict': 'verified', 'explanation': 'Visible in the mapped photo.'}]}]}
        return job, output

    def test_valid_review_is_not_human_ground_truth(self):
        job, output = self.review()
        result = validate_output(job, output, SCHEMAS[job['task']])
        self.assertFalse(result['review_consensus_grounding_human_verified'])

    def test_wrong_original_review_ID_rejected(self):
        job, output = self.review()
        output['claims'][0]['support'][0]['review_id'] = 'another-review'
        with self.assertRaises(ValueError): validate_output(job, output, SCHEMAS[job['task']])

    def test_fabricated_quote_rejected(self):
        job, output = self.review()
        output['claims'][0]['support'][0]['quote'] = 'The food was excellent.'
        with self.assertRaises(ValueError): validate_output(job, output, SCHEMAS[job['task']])

    def test_single_review_recurring_claim_rejected(self):
        job, output = self.review()
        output['claims'][0]['support'].pop()
        with self.assertRaises(ValueError): validate_output(job, output, SCHEMAS[job['task']])

    def test_wrong_photo_and_missing_assessment_rejected(self):
        job, output = self.visual()
        wrong = copy.deepcopy(output); wrong['assessments'][0]['photo_id'] = 'p2'
        with self.assertRaises(ValueError): validate_output(job, wrong, SCHEMAS[job['task']])
        output['assessments'] = []
        with self.assertRaises(ValueError): validate_output(job, output, SCHEMAS[job['task']])

    def test_photo_only_quote_rejected(self):
        job, output = self.visual()
        output['assessments'][0]['claims'][0]['caption_quote'] = 'a red table'
        with self.assertRaises(ValueError): validate_output(job, output, SCHEMAS[job['task']])

    def test_evidence_ID_does_not_enter_target_sentence(self):
        job, output = self.visual()
        output['assessments'][0]['claims'][0]['text'] += ' [IMG_CAP_1]'
        with self.assertRaises(ValueError): validate_output(job, output, SCHEMAS[job['task']])

    def test_zero_caption_condition_is_valid_empty(self):
        job, output = self.visual()
        job['evidence']['captions'] = []; output['assessments'] = []
        self.assertTrue(validate_output(job, output, SCHEMAS[job['task']])['structural_and_evidence_binding_valid'])


if __name__ == '__main__':
    unittest.main()
