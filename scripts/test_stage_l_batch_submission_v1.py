import copy
import tempfile
import unittest
from pathlib import Path

from test_stage_l_reference_workflow_v1 import fixture, draft_for
from stage_l_reference_app_v1 import ATTESTATIONS
from submit_stage_l_confirmed_drafts_v1 import execute, prepare

AUTHOR = {'id': 'author01', 'role': 'author'}


class BatchSubmissionTests(unittest.TestCase):
    def test_reported_confirmation_submits_without_checker_or_prose_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = fixture(Path(tmp)/'workspace'); b = next(iter(s.frame))
            d = draft_for(s, b); s.apply(b, AUTHOR, 'save', {'expected_version': 0, 'draft': d})
            confirmation = {'author_id': AUTHOR['id'], 'scope': 'all_100_current_drafts', 'author_attestations_reported': {k: True for k in ATTESTATIONS}, 'statement_verbatim': 'SYNTHETIC TEST ONLY'}
            self.assertEqual(AUTHOR['id'], 'author01')
            before = copy.deepcopy(s.get(b)[0])
            result = execute(s, confirmation, Path(tmp)/'receipt')
            r, v = s.get(b)
            self.assertEqual(r['status'], 'SUBMITTED'); self.assertEqual(v, 2)
            self.assertEqual(r['review_consensus'], before['review_consensus'])
            self.assertEqual(r['checker'], before['checker'])
            self.assertFalse(result['checker_actions_performed'])
            self.assertIsNone(prepare(s, b, AUTHOR['id'], confirmation))

    def test_missing_evidence_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = fixture(Path(tmp)/'workspace'); b = next(iter(s.frame))
            d = draft_for(s, b); d['review_claims'] = []
            s.apply(b, AUTHOR, 'save', {'expected_version': 0, 'draft': d})
            with self.assertRaisesRegex(ValueError, 'Missing component evidence'):
                prepare(s, b, AUTHOR['id'], {})
            self.assertEqual(s.get(b)[0]['status'], 'DRAFT')


if __name__ == '__main__':
    unittest.main()
