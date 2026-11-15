import copy
import json
from pathlib import Path
import tempfile
import unittest

from stage_l_reference_app_v1 import digest, reference_content
from test_stage_l_reference_workflow_v1 import fixture, draft_for, AUTHOR, CHECKER, AUTHOR_ATTEST, CHECKER_ATTEST
from import_stage_l_offline_checker_v1 import validate, execute


class OfflineCheckerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.store = fixture(self.root/'app', n=2)
        self.store.add_user(CHECKER['id'], 'checker', 'Synthetic-Test-Passphrase')
        rows = []
        for order, b in enumerate(self.store.frame, 1):
            self.store.apply(b, AUTHOR, 'submit', {'expected_version': 0, 'draft': draft_for(self.store, b), 'attestations': AUTHOR_ATTEST})
            r, v = self.store.get(b)
            rows.append({'order': order, 'business_id': b, 'business_name': self.store.source(b)['business_metadata']['name'], 'reference_version': v, 'reference_content_sha256': digest(reference_content(r)), 'status_at_export': 'SUBMITTED', 'decision': 'APPROVE' if order == 2 else 'REQUEST_REVISION', 'comment': '' if order == 2 else 'SYNTHETIC: revise wording'})
        self.doc = {'format': 'stage_l_offline_checker_review_v1', 'checker_id': CHECKER['id'], 'checker_declarations': CHECKER_ATTEST, 'reviews': rows}

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_exact_decisions_and_preserves_content(self):
        before = {b: reference_content(self.store.get(b)[0]) for b in self.store.frame}
        source = self.root/'decisions.json'; source.write_text(json.dumps(self.doc), encoding='utf8')
        receipt = execute(self.store, source, self.root/'receipt')
        self.assertEqual(receipt['audit']['counts']['APPROVED'], 1)
        self.assertEqual(receipt['audit']['counts']['NEEDS_REVISION'], 1)
        self.assertFalse(receipt['final_reference_freeze_performed'])
        for b, content in before.items(): self.assertEqual(reference_content(self.store.get(b)[0]), content)
        with self.assertRaisesRegex(ValueError, 'Stale reference version'):
            validate(self.store, self.doc)

    def test_rejects_wrong_hash_missing_declarations_and_duplicates_without_writes(self):
        for case in ['hash', 'declarations', 'duplicate']:
            doc = copy.deepcopy(self.doc)
            if case == 'hash': doc['reviews'][0]['reference_content_sha256'] = '0'*64
            elif case == 'declarations': doc['checker_declarations']['sources_checked'] = False
            else: doc['reviews'][1] = copy.deepcopy(doc['reviews'][0])
            with self.assertRaises(ValueError): validate(self.store, doc)
            self.assertEqual(self.store.audit()['counts']['SUBMITTED'], 2)


if __name__ == '__main__': unittest.main()
