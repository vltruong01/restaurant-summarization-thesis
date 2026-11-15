"""CPU-only boundary regressions; tiny symbolic fixtures are not business summaries."""
import copy
from pathlib import Path
import unittest

try:
    from count_led_tokens import split_target, target_counts, validate_proposal, load_tokenizer
except ModuleNotFoundError:
    from teacher_redesign_token_count_v1 import split_target, target_counts, validate_proposal, load_tokenizer


def fixture():
    # Symbols exercise formatting/evidence boundaries without drafting a target for any business.
    source = {'business_id': 'X', 'reviews': [{'evidence_id': 'X::REV_1', 'text': 'alpha'}],
              'captions': {'unfiltered': [{'evidence_id': 'X::U::IMG_CAP_1', 'photo_id': 'P1',
                                          'model_visible_caption': 'beta'}], 'filtered': []}}
    ro = '[REVIEW_CONSENSUS]\nalpha'
    proposal = {'business_id': 'X', 'split': 'train', 'canonical_review_consensus': 'alpha',
        'review_claims': [{'claim_id': 'R1', 'text': 'alpha', 'evidence': [{'evidence_id': 'X::REV_1', 'quote': 'alpha'}]}],
        'targets': {'reviews_only': {'text': ro, 'visual_claims': []},
            'unfiltered': {'text': ro + '\n\n[VISUAL_EVIDENCE]\nbeta', 'visual_claims': [{
                'claim_id': 'U1', 'text': 'beta', 'evidence': [{'evidence_id': 'X::U::IMG_CAP_1', 'quote': 'beta', 'photo_id': 'P1'}]}]},
            'filtered': {'text': ro + '\n\n[VISUAL_EVIDENCE]\n', 'visual_claims': []}}}
    return proposal, source


class ExportBoundaryTests(unittest.TestCase):
    def test_exact_canonical_and_empty_F_pass(self):
        p, s = fixture()
        self.assertEqual(validate_proposal(p, s)['status'], 'PASS_MECHANICAL_ONLY')

    def test_zero_caption_assertion_rejected(self):
        p, s = fixture()
        p['targets']['filtered']['text'] += 'beta'
        with self.assertRaisesRegex(ValueError, 'zero condition-visible'):
            validate_proposal(p, s)

    def test_canonical_copy_change_rejected(self):
        p, s = fixture()
        p['targets']['filtered']['text'] = p['targets']['filtered']['text'].replace('alpha', 'gamma')
        with self.assertRaisesRegex(ValueError, 'consensus differs'):
            validate_proposal(p, s)

    def test_wrong_condition_caption_rejected(self):
        p, s = fixture()
        p['targets']['unfiltered']['visual_claims'][0]['evidence'][0]['evidence_id'] = 'X::F::IMG_CAP_1'
        with self.assertRaisesRegex(ValueError, 'Wrong-condition'):
            validate_proposal(p, s)

    def test_wrong_photo_mapping_rejected(self):
        p, s = fixture()
        p['targets']['unfiltered']['visual_claims'][0]['evidence'][0]['photo_id'] = 'P2'
        with self.assertRaisesRegex(ValueError, 'photo mapping'):
            validate_proposal(p, s)

    def test_nonliteral_review_quote_rejected(self):
        p, s = fixture()
        p['review_claims'][0]['evidence'][0]['quote'] = 'gamma'
        with self.assertRaisesRegex(ValueError, 'nonliteral quote'):
            validate_proposal(p, s)

    def test_cross_photo_claim_rejected(self):
        p, s = fixture()
        s['captions']['unfiltered'].append({'evidence_id': 'X::U::IMG_CAP_2',
            'photo_id': 'P2', 'model_visible_caption': 'gamma'})
        p['targets']['unfiltered']['visual_claims'][0]['evidence'].append({
            'evidence_id': 'X::U::IMG_CAP_2', 'photo_id': 'P2', 'quote': 'gamma'})
        with self.assertRaisesRegex(ValueError, 'within one photo'):
            validate_proposal(p, s)

    def test_RO_visual_header_rejected(self):
        with self.assertRaisesRegex(ValueError, 'RO must not'):
            split_target('[REVIEW_CONSENSUS]\nalpha\n\n[VISUAL_EVIDENCE]\n', 'reviews_only')

    def test_token_denominator_and_no_truncation(self):
        portable = Path(__file__).resolve().parents[1] / 'tokenizer'
        if portable.is_dir():
            tok = load_tokenizer(portable)
        else:
            from source_budget_audit_template import pinned_tokenizer
            tok, _ = pinned_tokenizer()
        p, _ = fixture()
        for condition, record in p['targets'].items():
            counts = target_counts(tok, record['text'], condition)
            self.assertEqual(counts['original_tokenizer_BOS_content_EOS_tokens'],
                             counts['corrected_training_content_EOS_label_tokens'] + 1)
            self.assertFalse(counts['truncation_performed'])
        self.assertEqual(target_counts(tok, p['targets']['filtered']['text'], 'filtered')['visual_body_standalone_no_special'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
