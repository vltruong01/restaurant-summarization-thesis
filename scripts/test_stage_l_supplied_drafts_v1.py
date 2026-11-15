"""Synthetic-only intake regressions. Never reads genuine participant files/DB."""
import copy
from pathlib import Path
import tempfile
import unittest

from import_stage_l_supplied_drafts_v1 import assemble, import_records, parse_text
from test_stage_l_reference_workflow_v1 import fixture


def text_record(bid, review='Café customers liked the soup.\nAnother paragraph.', visual='A blue bowl is visible.'):
    return f'BUSINESS_ID: {bid}\n\n[REVIEW_CONSENSUS]\n\n{review}\n\n[VISUAL_EVIDENCE]\n\n{visual}\n'


class Parsing(unittest.TestCase):
    def test_numbered_corrections_do_not_leak_next_title_into_prose(self):
        raw = '\ufeff007 — Café One\r\n' + text_record('ONE') + '\n009 — Café Two\n' + text_record('TWO')
        records = parse_text(raw)
        self.assertEqual(records['ONE']['review_consensus'], 'Café customers liked the soup.\nAnother paragraph.')
        self.assertEqual(records['ONE']['visual_evidence'], 'A blue bowl is visible.')
        self.assertEqual(records['TWO']['visual_evidence'], 'A blue bowl is visible.')

    def test_duplicate_and_incomplete_records_are_refused(self):
        for raw in [text_record('ONE') + text_record('ONE'),
                    'BUSINESS_ID: ONE\n[REVIEW_CONSENSUS]\nMissing visual.',
                    text_record('ONE', visual=''),
                    text_record('ONE') + '\nUnexpected trailing material\n002 — Next\nFooter']:
            with self.assertRaises(ValueError):
                parse_text(raw)

    def test_exact_seven_replacements_with_93_untouched(self):
        ids = [f'CASE_{i:03d}' for i in range(1, 101)]
        patches = [ids[i - 1] for i in (7, 9, 14, 18, 56, 71, 72)]
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td)/f'{i}.txt' for i in range(3)]
            paths[0].write_text(''.join(text_record(b) for b in ids[:50]), encoding='utf-8')
            paths[1].write_text(''.join(text_record(b) for b in ids[50:]), encoding='utf-8')
            paths[2].write_text(''.join(text_record(b, 'Supplied revised wording.') for b in patches), encoding='utf-8')
            merged, origin = assemble(paths, ids, patches)
            self.assertEqual(list(merged), ids)
            self.assertEqual(sum(r['review_consensus'] == 'Supplied revised wording.' for r in merged.values()), 7)
            self.assertEqual(sum(origin[b] == 2 for b in ids), 7)
            paths[2].write_text(''.join(text_record(b) for b in patches[:-1]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'replacement'):
                assemble(paths, ids, patches)


class DraftImport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = fixture(Path(self.temp.name)/'demo', 2)
        self.store.add_user('demo_author', 'author', 'synthetic-only-password')
        self.records = [{'business_id': b, 'source_bundle_sha256': row['sha256'],
                         'review_consensus': f'Supplied draft for {b}.', 'visual_evidence': 'A synthetic image.',
                         'provenance': {'submission_file': 'SYNTHETIC.txt', 'submission_sha256': 'a'*64}}
                        for b, row in self.store.frame.items()]

    def tearDown(self):
        self.temp.cleanup()

    def run_import(self, records=None):
        return import_records(self.store, records or self.records, 'demo_author', 'b'*64)

    def test_verbatim_drafts_and_journal_without_human_certifications(self):
        self.assertEqual(self.run_import()['imported'], 2)
        for record in self.records:
            r, version = self.store.get(record['business_id'])
            self.assertEqual(version, 1)
            self.assertEqual(r['review_consensus'], record['review_consensus'])
            self.assertEqual(r['visual_evidence'], record['visual_evidence'])
            self.assertEqual(r['status'], 'DRAFT')
            self.assertEqual(r['review_claims'], [])
            self.assertEqual(r['visual_claims'], [])
            self.assertFalse(r['author']['human_written_attestation'])
            self.assertFalse(r['checker']['independent_check_attestation'])
            self.assertIn('Administrative import', r['notes'])
        self.assertEqual(self.store.audit()['journal_errors'], [])
        with self.store.connect() as c:
            import json
            self.assertTrue(all(json.loads(row[0])['attestations'] == {} for row in c.execute('SELECT event_json FROM events')))

    def test_repeat_is_noop_and_preserves_history(self):
        self.run_import()
        head = self.store.audit()['event_chain_head_sha256']
        result = self.run_import()
        self.assertEqual(result['imported'], 0)
        self.assertEqual(result['existing_identical_skipped'], 2)
        self.assertEqual(self.store.audit()['event_chain_head_sha256'], head)

    def test_late_conflict_refuses_whole_preflight_without_partial_writes(self):
        self.run_import([self.records[1]])
        changed = copy.deepcopy(self.records)
        changed[1]['review_consensus'] = 'Replacement is not approved.'
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            self.run_import(changed)
        self.assertEqual(self.store.get(self.records[0]['business_id'])[1], 0)
        self.assertEqual(self.store.get(self.records[1]['business_id'])[1], 1)

    def test_wrong_binding_or_checker_role_refused_before_writes(self):
        changed = copy.deepcopy(self.records)
        changed[1]['source_bundle_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Source binding'):
            self.run_import(changed)
        self.store.add_user('demo_checker', 'checker', 'synthetic-checker-password')
        with self.assertRaisesRegex(ValueError, 'author account'):
            import_records(self.store, self.records, 'demo_checker', 'b'*64)
        self.assertEqual(self.store.audit()['nonempty_drafts'], 0)


if __name__ == '__main__':
    unittest.main()
