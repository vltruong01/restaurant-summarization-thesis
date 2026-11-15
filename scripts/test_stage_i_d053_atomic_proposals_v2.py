"""New-proposal integrity checks. No scientific acceptance or human labeling."""
import hashlib,json,re,unittest
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reference_proposals'
if not OUT.exists():OUT=ROOT/'outputs/stage_i/d053_reference_proposals_v2'
OLD=ROOT/'researcher_finalization'
if not OLD.exists():OLD=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1'
SOURCE=ROOT if (ROOT/'source_cases.json').exists() else OLD/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class ProposalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.units=read(OUT/'PROPOSED_ATOMIC_REFERENCE_UNITS.json')['units'];cls.cases={c['business_id']:c for c in read(SOURCE/'source_cases.json')};cls.approved=read(OLD/'RESEARCHER_APPROVED_DECISIONS.json');cls.conflicts=read(OUT/'CAPTION_SCOPE_RECOMMENDATIONS.json');cls.omissions=read(OUT/'OMISSION_RECOMMENDATIONS.json')
 def test_01_prior_226_decisions_preserved(self):
  expected=next(r['sha256'] for r in read(OUT/'INPUT_PRESERVATION.json')['files'] if r['name']=='approved_candidate_ledger');self.assertEqual(sha(OLD/'RESEARCHER_APPROVED_DECISIONS.json'),expected);self.assertEqual(len(self.approved['decisions']),226)
 def test_02_no_new_human_confirmation(self):
  for u in self.units:self.assertIs(u['human_verified'],False);self.assertIs(u['scoring_eligible'],False);self.assertIsNone(u['confirmed_by']);self.assertIsNone(u['researcher_decision'])
 def test_03_new_units_are_not_321_fragments(self):
  self.assertNotEqual(len(self.units),321);self.assertTrue(all(u['text'].endswith('.') and len(u['text'].split())>=4 for u in self.units))
 def test_04_lineage_is_complete_not_one_to_one(self):
  rows=read(OUT/'ORIGINAL_321_SPAN_LINEAGE.json');self.assertEqual(len(rows),321);self.assertEqual(len({r['old_span_id'] for r in rows}),321);self.assertTrue(all('Parent-level' in r['mapping_scope'] and r['new_parent_proposal_ids'] for r in rows));self.assertTrue(any(len(r['new_parent_proposal_ids'])>1 for r in rows))
 def test_05_unique_ids_and_propositions(self):
  self.assertEqual(len({u['proposal_id'] for u in self.units}),len(self.units));keys=[(u['business_id'],u['domain'],u['text'].lower(),u['proposed_metadata']['polarity'],tuple(sorted(e.get('photo_id','') for e in u['evidence']))) for u in self.units];self.assertEqual(len(keys),len(set(keys)))
 def test_06_only_24_validation_businesses(self):
  self.assertEqual(len(self.cases),24);self.assertEqual({u['business_id'] for u in self.units},set(self.cases));self.assertTrue(all(c['split']=='validation' for c in self.cases.values()))
 def test_07_no_uncertain_or_zero_caption_visual_units(self):
  uncertain={d['candidate_id'] for d in self.approved['decisions'] if d['decision']=='uncertain'};self.assertEqual(len(uncertain),2);self.assertFalse(any(u.get('parent_candidate_id') in uncertain for u in self.units));self.assertFalse(any(u['business_id']=='j3kGicU3mPG_ciIKG1Vuyg' and u['domain']=='visual' for u in self.units))
 def test_08_every_evidence_binding_exact(self):
  for u in self.units:
   c=self.cases[u['business_id']];dom=u['domain'];pool={r['marker' if dom=='review' else 'caption_id']:r for r in c['reviews' if dom=='review' else 'captions']}
   for e in u['evidence']:
    r=pool[e['marker']];field='review_id' if dom=='review' else 'photo_id';self.assertEqual(e[field],r[field]);self.assertIn(e['quote'],r['text'])
 def test_09_visual_images_present_and_original(self):
  for c in self.cases.values():
   for cap in c['captions']:self.assertTrue((SOURCE/cap['image_file']).is_file());self.assertFalse(Path(cap['image_file']).is_absolute());self.assertNotIn('..',Path(cap['image_file']).parts)
 def test_10_metadata_complete_but_only_proposed(self):
  for u in self.units:
   m=u['proposed_metadata'];self.assertIn(m['importance'],['core','supporting']);self.assertIn(m['informativeness'],['useful_specific','useful_general','low_information']);self.assertTrue(m['aspect']);self.assertIsInstance(m['restaurant_specific'],bool)
   if u['domain']=='visual':self.assertIsNone(m['polarity'])
   else:self.assertIn(m['polarity'],['positive','negative','mixed','neutral'])
 def test_11_caption_decisions_pending_11_keep_4_exclude(self):
  self.assertEqual(Counter(r['recommended_action'] for r in self.conflicts),{'include_caption_safe_revision':11,'exclude':4});self.assertTrue(all(not r['human_verified'] and r['researcher_decision'] is None for r in self.conflicts))
 def test_12_no_flagged_photo_only_details_in_alternatives(self):
  by={r['candidate_id']:r['proposed_caption_safe_wording'].lower() for r in self.conflicts}
  checks={'3v0tSwHXR9LhpsoGXLgsTA_IMG_CAP_2_CANDIDATE':['shrimp'],'Ktg3ahIxk0JIkJwXAqu2ew_IMG_CAP_1_CANDIDATE':['green'],'haPQJ2oy3529RMzpWrA39A_IMG_CAP_2_CANDIDATE':['barbecue','sides'],'aZN5R7Xfea0GkDBwBZOQrQ_IMG_CAP_2_CANDIDATE':['glass of beer'],'gcQSAQIx-KQOdf5E8HF6Lw_IMG_CAP_2_CANDIDATE':['tea','amber']}
  for cid,forbidden in checks.items():
   for word in forbidden:self.assertNotIn(word,by[cid])
 def test_13_omissions_have_roles_and_exact_sources(self):
  self.assertEqual(Counter(o['recommended_reference_role'] for o in self.omissions),{'core':3,'supporting':1,'excluded':1});self.assertTrue(all(o['researcher_decision'] is None for o in self.omissions));self.assertEqual(sum(u['origin']=='source_omission_proposal' for u in self.units),7)
 def test_14_both_review_sentiments_retained(self):
  for bid in self.cases:
   polarities={u['proposed_metadata']['polarity'] for u in self.units if u['business_id']==bid and u['domain']=='review'};self.assertTrue({'positive','negative'}<=polarities,bid)
 def test_15_ambiguous_literal_pricing_not_silently_fixed(self):
  rows=[u for u in self.units if u.get('evidence_caution')];self.assertEqual(len(rows),1);self.assertEqual(rows[0]['recommendation'],'exclude');self.assertIn('did tell',rows[0]['recommendation_reason'])
 def test_16_frozen_protocol_and_config_hashes(self):
  self.assertEqual(sha(SOURCE/'code/stage_i_d053_annotations.py'),'b5762faf33e66591094ca384a56ad3a8b7c967d96d313f10cc6020c133f2b752');self.assertEqual(sha(SOURCE/'FROZEN_EVALUATION_CONFIG.json'),'a139ee88824b6b5489b871cdbd5b9c13317139444f7a95033d5d9c5019a48461')
 def test_17_old_sha_receipt_reconciles_published_not_release_candidate(self):
  r=read(OUT/'ZIP_RECEIPT_RECONCILIATION.json');self.assertEqual(r['actual_sha256'],'af82bc27e934924c67ec813114caaa24748532e4a257c23a04411c73d7219a3d');self.assertEqual(r['actual_sha256'],r['published_sidecar_sha256']);self.assertNotEqual(r['actual_sha256'],r['release_candidate_sha256']);self.assertFalse(r['previous_delivered_archive_rewritten'])
 def test_18_version_binds_actual_proposal_files(self):
  v=read(OUT/'PROPOSAL_VERSION.json');self.assertEqual(v['files'],{name:sha(OUT/name) for name in v['files']});self.assertEqual(v['proposal_id'],hashlib.sha256(json.dumps(v['files'],sort_keys=True).encode()).hexdigest())
 def test_19_blank_decision_template(self):
  s=read(OUT/'RESEARCHER_DECISIONS_TEMPLATE.json');self.assertFalse(s['history']);self.assertFalse(s['researcher_attestation']);self.assertFalse(s['final_reference_freeze']);self.assertEqual(s['new_human_confirmed_unit_count'],0)
  for g,n in [('corrections',15),('omissions',5),('business_packages',24)]:self.assertEqual(len(s[g]),n);self.assertTrue(all(d['choice'] is None and d['confirmed_at'] is None for d in s[g].values()))
 def test_20_parent_coverage_and_consolidation(self):
  parents={u.get('parent_candidate_id') for u in self.units if u.get('parent_candidate_id')};self.assertEqual(len(parents),224);notes=read(OUT/'CONSOLIDATION_AND_ATOMICITY_NOTES.json');self.assertFalse(notes['one_to_one_span_conversion']);self.assertGreaterEqual(len(notes['retired_during_context_curation']),5)
 def test_21_original_annotation_hash_unchanged(self):
  self.assertEqual(sha(OLD/'incoming/D053_AI_DRAFT_24_RESTAURANTS_EDITED.json'),'4f2a6ec958e80a1f888d100673aeedd00ed18d68af2dd243ae6247f3fdc009a8')
 def test_22_no_fragment_or_dangling_review_prefix(self):
  self.assertFalse(any(re.match(r'^(That |Those |And |While |But |Food opinions are mixed)',u['text']) for u in self.units))
if __name__=='__main__':unittest.main(verbosity=2)
