"""Symbolic boundary tests only: no real-business V2 draft or generation."""
import copy
import json
from pathlib import Path
import unittest

try:
    from test_export_boundaries import fixture
    from validate_teacher_target_v2_acceptance_v1 import check_record
    from count_led_tokens import load_tokenizer
except ModuleNotFoundError:
    from test_teacher_target_redesign_pilot_v1 import fixture
    from validate_teacher_target_v2_acceptance_v1 import check_record
    from teacher_redesign_token_count_v1 import load_tokenizer


class V2BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(__file__).resolve().parents[1]
        if not (folder / 'TARGET_V2_OUTPUT_SCHEMA.json').is_file():
            folder = folder / 'outputs/stage_i/teacher_target_v2_preparation_v1/stage_a_sources_32'
        cls.schema = json.loads((folder / 'TARGET_V2_OUTPUT_SCHEMA.json').read_text(encoding='utf-8'))
        cls.tok = load_tokenizer(folder / 'tokenizer')

    def symbolic(self):
        row, source = fixture()
        source['split'] = 'train'
        source['caption_counts'] = {'unfiltered': 1, 'filtered': 0}
        source['exact_frozen_source_inputs'] = {c: {'utf8_sha256': '0' * 64} for c in row['targets']}
        row.update(schema_version=2, policy_id=self.schema['properties']['policy_id']['const'],
            policy_sha256=self.schema['properties']['policy_sha256']['const'],
            source_inputs_sha256={c: '0' * 64 for c in row['targets']},
            status='PROVISIONAL_GPT_WORK_V2_REQUIRES_RESEARCHER_REVIEW',
            researcher_confirmed=False, uncertainties=[])
        row['review_claims'][0]['evidence_characterization'] = {
            'basis': 'mixed_or_uncertain', 'salience_reason': 'symbolic boundary fixture', 'uncertainty_note': ''}
        return row, source

    def test_outside_guidance_warns_without_rejecting(self):
        row, source = self.symbolic()
        result = check_record(row, source, self.schema, self.tok)
        self.assertEqual(result['mechanical']['status'], 'PASS_MECHANICAL_ONLY')
        self.assertTrue(result['guidance_warnings'])
        self.assertFalse(result['researcher_quality_confirmation'])

    def test_source_binding_mismatch_blocks(self):
        row, source = self.symbolic()
        row['source_inputs_sha256']['filtered'] = '1' * 64
        with self.assertRaisesRegex(ValueError, 'source bindings'):
            check_record(row, source, self.schema, self.tok)

    def test_wrong_policy_hash_blocks(self):
        import jsonschema
        row, source = self.symbolic()
        row['policy_sha256'] = '1' * 64
        with self.assertRaises(jsonschema.ValidationError):
            check_record(row, source, self.schema, self.tok)

    def test_no_fabricated_researcher_confirmation(self):
        import jsonschema
        row, source = self.symbolic()
        row['researcher_confirmed'] = True
        with self.assertRaises(jsonschema.ValidationError):
            check_record(row, source, self.schema, self.tok)

    def test_no_acceptance_validation_business(self):
        row, source = self.symbolic()
        source['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'TRAIN32-only'):
            check_record(row, source, self.schema, self.tok)

    def test_hard_768_limit_never_truncates(self):
        row, source = self.symbolic()
        body = ' '.join(['alpha'] * 900)
        row['canonical_review_consensus'] = body
        row['review_claims'][0]['text'] = body
        for target in row['targets'].values():
            target['text'] = target['text'].replace('alpha', body)
        before = copy.deepcopy(row)
        with self.assertRaisesRegex(ValueError, '768label ceiling'):
            check_record(row, source, self.schema, self.tok)
        self.assertEqual(row, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
