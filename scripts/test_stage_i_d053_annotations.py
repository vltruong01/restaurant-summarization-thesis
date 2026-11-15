"""Synthetic, offline scientific-integrity tests; no model or private data."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stage_i_d053_annotations import make_initial_state, reference_freeze, validate_annotations, BINDING_KEYS

STAMP = '2026-10-16T12:00:00+07:00'


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {'business_id': 'B1', 'subset': 'random', 'split': 'validation', 'condition': 'filtered',
             'reviews': [{'marker': 'REV_1', 'review_id': 'R1', 'text': 'I liked the smoky soup, but the service was slow.'}],
             'captions': [{'caption_id': 'IMG_CAP_1', 'photo_id': 'P1', 'text': 'a bowl on a wooden table.'}]},
            {'business_id': 'B2', 'subset': 'targeted', 'split': 'validation', 'condition': 'filtered',
             'reviews': [{'marker': 'REV_1', 'review_id': 'R2', 'text': 'The noodles were bland.'}], 'captions': []},
        ]
        self.candidates = [
            {'unit_id': 'C1', 'business_id': 'B1', 'domain': 'review', 'text': 'A reviewer liked soup.'},
            {'unit_id': 'C2', 'business_id': 'B1', 'domain': 'visual', 'text': 'A bowl on a table.'},
            {'unit_id': 'C3', 'business_id': 'B2', 'domain': 'review', 'text': 'A reviewer disliked the noodles.'},
        ]
        self.bindings = {k: str(i + 1) * 64 for i, k in enumerate(BINDING_KEYS)}
        self.state = make_initial_state(self.cases, self.candidates, self.bindings, 'synthetic')

    def validate(self, state=None, final=False):
        return validate_annotations(state or self.state, self.cases, self.candidates, self.bindings, final)

    @staticmethod
    def snapshot(state, kind, entity_id, record):
        state['history'].append({'kind': kind, 'entity_id': entity_id, 'confirmed_by': record['confirmed_by'],
                                 'confirmed_at': record['confirmed_at'], 'snapshot': copy.deepcopy(record)})

    def complete(self):
        state = copy.deepcopy(self.state)
        state['reviewer_id'] = 'Researcher'
        state['researcher_confirmed_complete'] = True
        for i, c in enumerate(self.candidates):
            unit = {'unit_id': 'U' + str(i + 1), 'business_id': c['business_id'], 'domain': c['domain'],
                    'text': c['text'], 'support_status': 'supported', 'human_verified': True,
                    'confirmed_by': 'Researcher', 'confirmed_at': STAMP,
                    'importance': 'core', 'informativeness': 'useful_specific',
                    'polarity': None if c['domain'] == 'visual' else ('negative' if i == 2 else 'positive'),
                    'aspect': 'food', 'restaurant_specific': True, 'atomicity_confirmed': True,
                    'evidence_confirmed': True, 'visual_photo_confirmed': c['domain'] == 'visual',
                    'provenance': {'origin': 'candidate_split', 'parent_candidate_id': c['unit_id']}}
            if c['domain'] == 'visual':
                unit['evidence'] = [{'marker': 'IMG_CAP_1', 'photo_id': 'P1', 'quote': 'a bowl on a wooden table.'}]
            else:
                unit['evidence'] = [{'marker': 'REV_1', 'review_id': 'R2' if i == 2 else 'R1',
                                     'quote': 'The noodles were bland.' if i == 2 else 'I liked the smoky soup'}]
            state['units'].append(unit)
            record = state['candidate_annotations'][c['unit_id']]
            record.update(support_label='supported', human_verified=True, confirmed_by='Researcher',
                          confirmed_at=STAMP, importance='core', informativeness='useful_specific', final_unit_ids=[unit['unit_id']])
            self.snapshot(state, 'candidate_confirmed', c['unit_id'], record)
            self.snapshot(state, 'unit_confirmed', unit['unit_id'], unit)
        for bid, record in state['business_reviews'].items():
            record.update(human_verified=True, positive_negative_and_specificity_checked=True,
                          confirmed_by='Researcher', confirmed_at=STAMP)
            self.snapshot(state, 'business_confirmed', bid, record)
        return state

    def test_initial_state_is_valid_but_not_human_ground_truth(self):
        before = copy.deepcopy(self.state)
        result = self.validate()
        self.assertTrue(result['valid'])
        self.assertFalse(result['ready_for_freeze'])
        self.assertEqual(result['counts']['verified_candidates'], 0)
        self.assertEqual(before, self.state)
        with self.assertRaises(ValueError):
            reference_freeze(self.state, self.cases, self.candidates, self.bindings)

    def test_fully_confirmed_sheet_exports_d049_shape_without_editing(self):
        state = self.complete()
        before = copy.deepcopy(state)
        self.assertTrue(self.validate(state, True)['valid'])
        frozen = reference_freeze(state, self.cases, self.candidates, self.bindings)
        self.assertEqual(frozen['case_inputs_sha256'], self.bindings['case_inputs_sha256'])
        self.assertEqual(frozen['protocol_sha256'], self.bindings['protocol_sha256'])
        self.assertEqual(frozen['units'], state['units'])
        self.assertEqual(frozen['candidate_annotations'], state['candidate_annotations'])
        self.assertEqual(state, before)

    def test_unknown_missing_or_changed_candidate_ids_are_rejected(self):
        self.state['candidate_annotations']['C4'] = self.state['candidate_annotations'].pop('C1')
        self.assertFalse(self.validate()['valid'])

    def test_bindings_are_exact(self):
        self.state['bindings']['candidate_sha256'] = 'a' * 64
        self.assertFalse(self.validate()['valid'])

    def test_boolean_spoof_does_not_count_as_human_confirmation(self):
        self.state['candidate_annotations']['C1']['human_verified'] = 1
        self.assertFalse(self.validate()['valid'])

    def test_changed_confirmed_record_requires_new_confirmation_snapshot(self):
        state = self.complete()
        state['candidate_annotations']['C1']['support_label'] = 'uncertain'
        self.assertFalse(self.validate(state)['valid'])
        state = self.complete()
        state['units'][0]['text'] = 'Altered claim'
        self.assertFalse(self.validate(state)['valid'])

    def test_blank_and_nonexact_quotes_rejected_even_in_draft(self):
        for quote in (' ', 'smoky beef soup'):
            state = self.complete()
            state['units'][0]['evidence'][0]['quote'] = quote
            self.assertFalse(self.validate(state)['valid'])

    def test_foreign_original_ids_and_wrong_domains_are_rejected(self):
        for field, value in [('review_id', 'R2'), ('marker', 'IMG_CAP_1')]:
            state = self.complete()
            state['units'][0]['evidence'][0][field] = value
            self.assertFalse(self.validate(state)['valid'])

    def test_visual_requires_caption_support_and_photo_confirmation(self):
        state = self.complete()
        state['units'][1]['visual_photo_confirmed'] = False
        self.snapshot(state, 'unit_confirmed', 'U2', state['units'][1])
        self.assertFalse(self.validate(state, True)['valid'])
        state['units'][1]['visual_photo_confirmed'] = True
        state['units'][1]['evidence'][0]['quote'] = 'red soup'
        self.snapshot(state, 'unit_confirmed', 'U2', state['units'][1])
        self.assertFalse(self.validate(state)['valid'])

    def test_zero_caption_business_cannot_have_visual_reference(self):
        state = self.complete()
        state['units'][1]['business_id'] = 'B2'
        self.assertFalse(self.validate(state)['valid'])

    def test_visual_reference_cannot_acquire_review_opinion_polarity(self):
        for polarity in ('positive', 'negative', 'mixed', 'neutral'):
            with self.subTest(polarity=polarity):
                state = self.complete()
                state['units'][1]['polarity'] = polarity
                self.snapshot(state, 'unit_confirmed', 'U2', state['units'][1])
                result = self.validate(state)
                self.assertFalse(result['valid'])
                self.assertTrue(any('review-opinion polarity' in message for message in result['errors']))

    def test_uncertain_preserved_as_uncertain_and_business_retained(self):
        state = self.complete()
        record = state['candidate_annotations']['C2']
        record.update(support_label='uncertain', reviewer_notes='Object is obscured.', final_unit_ids=[])
        state['units'] = [u for u in state['units'] if u['unit_id'] != 'U2']
        self.snapshot(state, 'candidate_confirmed', 'C2', record)
        self.assertTrue(self.validate(state, True)['valid'])
        frozen = reference_freeze(state, self.cases, self.candidates, self.bindings)
        self.assertEqual(frozen['exclusion_log'][0]['support_label'], 'uncertain')
        self.assertIn('B1', frozen['business_reviews'])
        self.assertEqual(len(frozen['business_reviews']), 2)

    def test_candidate_priorities_optional_but_final_atomic_unit_priorities_required(self):
        state = self.complete()
        for cid, record in state['candidate_annotations'].items():
            record.update(importance=None, informativeness=None)
            self.snapshot(state, 'candidate_confirmed', cid, record)
        self.assertTrue(self.validate(state, True)['valid'])
        record = state['candidate_annotations']['C2']
        record.update(support_label='uncertain', reviewer_notes='Identity cannot be determined.', final_unit_ids=[])
        state['units'] = [u for u in state['units'] if u['unit_id'] != 'U2']
        self.snapshot(state, 'candidate_confirmed', 'C2', record)
        self.assertTrue(self.validate(state, True)['valid'])
        record['support_label'] = 'unsupported'
        self.snapshot(state, 'candidate_confirmed', 'C2', record)
        self.assertTrue(self.validate(state, True)['valid'])
        state['units'][0]['importance'] = None
        self.snapshot(state, 'unit_confirmed', 'U1', state['units'][0])
        self.assertFalse(self.validate(state, True)['valid'])

    def test_unconfirmed_exclusion_transition_preserves_units_but_cannot_freeze(self):
        for label in ('uncertain', 'unsupported', 'contradicted', None):
            with self.subTest(label=label):
                state = self.complete()
                previous_units = copy.deepcopy(state['units'])
                record = state['candidate_annotations']['C2']
                record.update(support_label=label, human_verified=False, confirmed_by=None, confirmed_at=None,
                              reviewer_notes='Decision is being re-inspected.')
                state['researcher_confirmed_complete'] = False
                draft = self.validate(state)
                self.assertTrue(draft['valid'], draft['errors'])
                self.assertFalse(draft['ready_for_freeze'])
                self.assertFalse(self.validate(state, True)['valid'])
                self.assertEqual(state['units'], previous_units)
                if label is not None:
                    record.update(human_verified=True, confirmed_by='Researcher', confirmed_at=STAMP)
                    self.snapshot(state, 'candidate_confirmed', 'C2', record)
                    self.assertFalse(self.validate(state)['valid'])

    def test_needs_revision_draft_never_automatically_becomes_supported(self):
        state = self.complete()
        record = state['candidate_annotations']['C2']
        record.update(support_label='needs_revision', corrected_claim_text='A bowl is visible.',
                      revision_disposition='pending_unresolved', final_unit_ids=[])
        state['units'] = [u for u in state['units'] if u['unit_id'] != 'U2']
        self.snapshot(state, 'candidate_confirmed', 'C2', record)
        self.assertTrue(self.validate(state)['valid'])
        self.assertFalse(self.validate(state, True)['valid'])
        record.update(revision_disposition='excluded_from_reference', reviewer_notes='Not reliable enough to include.')
        self.snapshot(state, 'candidate_confirmed', 'C2', record)
        self.assertTrue(self.validate(state, True)['valid'])

    def test_resolved_revision_requires_separate_supported_atomic_unit(self):
        state = self.complete()
        record = state['candidate_annotations']['C2']
        record.update(support_label='needs_revision', corrected_claim_text='A bowl is visible.',
                      revision_disposition='resolved_with_confirmed_units')
        self.snapshot(state, 'candidate_confirmed', 'C2', record)
        self.assertTrue(self.validate(state, True)['valid'])
        state['units'][1]['atomicity_confirmed'] = False
        self.snapshot(state, 'unit_confirmed', 'U2', state['units'][1])
        self.assertFalse(self.validate(state, True)['valid'])

    def test_researcher_can_add_important_omitted_source_information(self):
        state = self.complete()
        unit = copy.deepcopy(state['units'][0])
        unit.update(unit_id='ADDED', text='A reviewer reported slow service.', polarity='negative', aspect='service',
                    provenance={'origin': 'researcher_added', 'parent_candidate_id': None},
                    evidence=[{'marker': 'REV_1', 'review_id': 'R1', 'quote': 'the service was slow.'}])
        state['units'].append(unit)
        self.snapshot(state, 'unit_confirmed', 'ADDED', unit)
        self.assertTrue(self.validate(state, True)['valid'])

    def test_source_completeness_is_required_per_business(self):
        state = self.complete()
        state['business_reviews']['B2']['positive_negative_and_specificity_checked'] = False
        self.snapshot(state, 'business_confirmed', 'B2', state['business_reviews']['B2'])
        self.assertFalse(self.validate(state, True)['valid'])

    def test_simple_correct_information_not_rejected_for_low_density(self):
        state = self.complete()
        state['units'][1]['informativeness'] = 'low_information'
        self.snapshot(state, 'unit_confirmed', 'U2', state['units'][1])
        self.assertTrue(self.validate(state, True)['valid'])

    def test_malformed_import_is_invalid_not_uncaught_exception(self):
        self.state['candidate_annotations']['C1']['final_unit_ids'] = [{}]
        self.assertFalse(self.validate()['valid'])

    def test_json_schema_accepts_initial_and_final_state_when_available(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest('jsonschema optional; stdlib validator remains authoritative')
        schema = json.loads((Path(__file__).resolve().parents[1] / 'schemas/stage_i_d053_annotation_v1.schema.json').read_text(encoding='utf-8'))
        jsonschema.validate(self.state, schema)
        jsonschema.validate(self.complete(), schema)

    def test_cli_package_checks_bound_files_and_never_overwrites_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            files = {'source_cases.json': json.dumps(self.cases),
                     'candidates_UNVERIFIED.jsonl': '\n'.join(json.dumps(c) for c in self.candidates),
                     'D049_FROZEN_EVALUATION_PROTOCOL.md': 'Frozen synthetic protocol',
                     'FROZEN_EVALUATION_CONFIG.json': '{}'}
            for name, value in files.items():
                (package / name).write_text(value, encoding='utf-8')
            bindings = dict(self.bindings)
            for name, key in [('source_cases.json', 'source_cases_sha256'), ('candidates_UNVERIFIED.jsonl', 'candidate_sha256'),
                              ('D049_FROZEN_EVALUATION_PROTOCOL.md', 'protocol_sha256'),
                              ('FROZEN_EVALUATION_CONFIG.json', 'evaluation_config_sha256')]:
                bindings[key] = hashlib.sha256((package / name).read_bytes()).hexdigest()
            bundle_id = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
            (package / 'bindings.json').write_text(json.dumps({'bundle_id': bundle_id, 'bindings': bindings}), encoding='utf-8')
            (package / 'annotations.json').write_text(json.dumps(make_initial_state(self.cases, self.candidates, bindings, bundle_id)), encoding='utf-8')
            command = [sys.executable, str(Path(__file__).with_name('stage_i_d053_annotations.py')),
                       '--package', str(package), '--annotations', str(package / 'annotations.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)['ready_for_freeze'])
            result = subprocess.run(command + ['--final'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = package / 'report.json'
            report.write_text('Existing evidence', encoding='utf-8')
            result = subprocess.run(command + ['--output', str(report)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(report.read_text(encoding='utf-8'), 'Existing evidence')
            (package / 'source_cases.json').write_text(json.dumps(self.cases) + ' ', encoding='utf-8')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('hash mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
