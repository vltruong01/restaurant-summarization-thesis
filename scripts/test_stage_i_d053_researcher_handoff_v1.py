"""Offline approval-import/guard checks against the immutable D053 handoff.

No actual unit is confirmed by these tests; corrupted records exist only in RAM.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'researcher_finalization'
if not OUT.is_dir():OUT=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1'
SOURCE=OUT/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
if not SOURCE.is_dir():SOURCE=ROOT


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class ApprovalImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=read(OUT/'incoming/D053_AI_DRAFT_24_RESTAURANTS_EDITED.json')
        cls.state=read(OUT/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json')
        cls.approval=read(OUT/'RESEARCHER_APPROVAL_RECORD.json')
        cls.cases=read(SOURCE/'source_cases.json')
        cls.candidates=[json.loads(s) for s in (SOURCE/'candidates_UNVERIFIED.jsonl').read_text().splitlines()]
        cls.binding=read(SOURCE/'bindings.json')['bindings']
        spec=importlib.util.spec_from_file_location('d053_frozen_handoff_test',SOURCE/'code/stage_i_d053_annotations.py')
        cls.validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.validator)

    def test_01_submitted_original_remains_unconfirmed_exact_bytes(self):
        self.assertEqual(sha(OUT/'incoming/D053_AI_DRAFT_24_RESTAURANTS_EDITED.json'),'4f2a6ec958e80a1f888d100673aeedd00ed18d68af2dd243ae6247f3fdc009a8')
        self.assertTrue(all(not a['human_verified'] for a in self.original['candidate_annotations'].values()))
        self.assertFalse(self.original['history']);self.assertFalse(self.original['units'])

    def test_02_approval_record_binds_actual_message_and_original(self):
        a=self.approval
        self.assertEqual(hashlib.sha256(a['approval_message_verbatim'].encode()).hexdigest(),a['approval_message_sha256'])
        self.assertEqual(a['submitted_json_sha256'],sha(OUT/'incoming/D053_AI_DRAFT_24_RESTAURANTS_EDITED.json'))
        self.assertIsNone(a['actual_prior_review_timestamp']);self.assertIsNone(a['conversation_message_id'])
        self.assertIn('NOT an invented time',a['timestamp_meaning'])

    def test_03_no_labels_or_candidate_ids_changed(self):
        self.assertEqual(set(self.original['candidate_annotations']),set(self.state['candidate_annotations']))
        for cid,a in self.original['candidate_annotations'].items():
            b=self.state['candidate_annotations'][cid]
            for k in ('support_label','corrected_claim_text','importance','informativeness','reviewer_notes'):
                self.assertEqual(a[k],b[k])
        self.assertEqual(Counter(a['support_label'] for a in self.state['candidate_annotations'].values()),{'supported':203,'needs_revision':21,'uncertain':2})

    def test_04_import_events_are_snapshot_bound_not_fabricated_UI(self):
        self.assertEqual(len(self.state['history']),226)
        for e in self.state['history']:
            self.assertEqual(e['snapshot'],self.state['candidate_annotations'][e['entity_id']])
            self.assertIn('conversation',e['recording_method']);self.assertTrue(e['review_timestamp_unknown'])
            self.assertEqual(e['confirmed_at'],self.approval['recorded_at_utc'])

    def test_05_uncertain_never_produces_reference_units(self):
        rows=read(OUT/'UNCERTAIN_EXCLUDED_FROM_SCORING.json');self.assertEqual(len(rows),2)
        ids={r['candidate_id'] for r in rows}
        self.assertTrue(all(r['excluded_from_scoring'] for r in rows))
        self.assertTrue(all(not self.state['candidate_annotations'][cid]['final_unit_ids'] for cid in ids))
        self.assertFalse(any(u['parent_candidate_id'] in ids for u in self.state['units']))

    def test_06_all21_corrections_exact_and_alternatives_not_applied(self):
        rows=read(OUT/'CORRECTION_CAPTION_PHOTO_AUDIT.json');self.assertEqual(len(rows),21)
        self.assertEqual(sum(r['caption_scope_assessment']=='REQUIRES_RESEARCHER_DECISION' for r in rows),15)
        for r in rows:
            a=self.original['candidate_annotations'][r['candidate_id']]
            self.assertEqual(r['approved_corrected_claim_text'],a['corrected_claim_text'])
            self.assertFalse(r['alternative_applied']);self.assertFalse(r['included_in_final_scoring'])

    def test_07_unit_proposals_are_exact_approved_wording_spans(self):
        byid={c['unit_id']:c for c in self.candidates}
        self.assertEqual(len(self.state['units']),321)
        for u in self.state['units']:
            a=self.original['candidate_annotations'][u['parent_candidate_id']]
            text=a['corrected_claim_text'] if a['support_label']=='needs_revision' else byid[u['parent_candidate_id']]['text']
            self.assertEqual(u['exact_parent_wording'],text)
            self.assertEqual(u['text'],text[u['parent_wording_start']:u['parent_wording_end']])
            self.assertTrue(u['parent_context_required'])

    def test_08_unit_human_atomicity_and_metadata_not_forged(self):
        for u in self.state['units']:
            self.assertFalse(u['human_verified']);self.assertFalse(u['atomicity_confirmed']);self.assertFalse(u['evidence_confirmed'])
            self.assertIsNone(u['importance']);self.assertIsNone(u['aspect']);self.assertIsNone(u['restaurant_specific'])
            self.assertEqual(u['construction_status'],'DRAFT_NOT_SCORING_ELIGIBLE')

    def test_09_zero_caption_business_has_no_visual_unit(self):
        ids={c['business_id'] for c in self.cases if not c['captions']};self.assertEqual(len(ids),1)
        self.assertFalse(any(u['domain']=='visual' and u['business_id'] in ids for u in self.state['units']))

    def test_10_draft_valid_but_original_final_guard_blocks(self):
        result=self.validator.validate_annotations(self.state,self.cases,self.candidates,self.binding,False)
        self.assertTrue(result['valid']);self.assertFalse(result['ready_for_freeze'])
        with self.assertRaises(ValueError):self.validator.reference_freeze(self.state,self.cases,self.candidates,self.binding)

    def test_11_foreign_quote_rejected_without_changing_disk(self):
        bad=copy.deepcopy(self.state);bad['units'][0]['evidence'][0]['quote']='SYNTHETIC_NOT_IN_ORIGINAL_EVIDENCE'
        r=self.validator.validate_annotations(bad,self.cases,self.candidates,self.binding,False)
        self.assertFalse(r['valid']);self.assertTrue(any('exact substring' in e for e in r['errors']))

    def test_12_omissions_are_proposals_not_new_approved_reference_facts(self):
        rows=read(OUT/'POTENTIAL_IMPORTANT_OMISSIONS_FOR_DECISION.json');self.assertEqual(len(rows),5)
        cases={c['business_id']:c for c in self.cases}
        for row in rows:
            r=next(r for r in cases[row['business_id']]['reviews'] if r['marker']==row['marker'])
            self.assertEqual(row['review_id'],r['review_id']);self.assertIn(row['quote'],r['text'])
            self.assertFalse(row['human_approved']);self.assertFalse(row['automatically_added_to_units'])
        self.assertFalse(any(u['provenance']['origin']=='researcher_added' for u in self.state['units']))

    def test_13_all90_original_photos_match_frozen_hash(self):
        photos=read(OUT/'PHOTO_INSPECTION_LOG.json')['images'];self.assertEqual(len(photos),90)
        self.assertEqual(len({p['photo_id'] for p in photos}),90)
        for p in photos:self.assertEqual(sha(SOURCE/p['image_file']),p['sha256'])

    def test_14_business_and_global_completeness_not_silently_forced(self):
        self.assertEqual(len(self.state['business_reviews']),24)
        self.assertFalse(self.state['researcher_confirmed_complete'])
        self.assertTrue(all(not b['human_verified'] for b in self.state['business_reviews'].values()))
        self.assertIn('all24',self.approval['researcher_stated_review_scope'])

    def test_15_original_scientific_validator_and_config_hash(self):
        self.assertEqual(sha(SOURCE/'code/stage_i_d053_annotations.py'),'b5762faf33e66591094ca384a56ad3a8b7c967d96d313f10cc6020c133f2b752')
        self.assertEqual(sha(SOURCE/'FROZEN_EVALUATION_CONFIG.json'),'a139ee88824b6b5489b871cdbd5b9c13317139444f7a95033d5d9c5019a48461')

    def test_16_schema_accepts_draft_but_does_not_grant_finality(self):
        import jsonschema
        schema=read(SOURCE/'schemas/stage_i_d053_annotation_v1.schema.json')
        jsonschema.Draft202012Validator(schema).validate(self.state)
        self.assertFalse(self.validator.validate_annotations(self.state,self.cases,self.candidates,self.binding,True)['valid'])


if __name__=='__main__':unittest.main(verbosity=2)
