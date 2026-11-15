"""Accepted D053 data and negative controls; mutations stay solely in memory."""
import copy,hashlib,json,unittest
from collections import Counter
from pathlib import Path
import jsonschema
from check_stage_i_d053_final_references_v2 import load_context,arguments,check,read,sha
from stage_i_d053_annotations import validate_annotations
from stage_i_d053_curated_references_v2 import validate_curated_annotations,freeze_curated_annotations
ROOT=Path(__file__).resolve().parents[1]
class FinalReferenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=load_context(ROOT)
 def altered(self):return copy.deepcopy(self.c)
 def reject(self,c):self.assertFalse(validate_curated_annotations(*arguments(c))['valid'])
 def test_01_full_offline_reference_check_passes(self):self.assertTrue(check(ROOT)['valid'])
 def test_02_correct_three_role_populations(self):
  c=self.c;self.assertEqual(Counter(u['importance'] for u in c['state']['units']),{'core':261,'supporting':315});self.assertEqual(len(c['state']['approved_exclusion_log']),88)
 def test_03_no_old_unapproved_flags_overwritten(self):
  self.assertTrue(all(not u['human_verified'] and not u['scoring_eligible'] for u in self.c['proposals']));self.assertTrue(all(not r['human_verified'] and not r['researcher_approved'] for r in self.c['curation']['unit_recommendations'].values()))
 def test_04_all_226_prior_labels_and_original_text_preserved(self):
  for cid,a in self.c['state']['candidate_annotations'].items():
   old=self.c['prior_annotations'][cid]
   for k in ['support_label','corrected_claim_text','importance','informativeness']:self.assertEqual(a[k],old[k])
 def test_05_two_uncertain_parents_remain_excluded_even_with_corrected_text(self):
  c=self.c;uncertain={k for k,v in c['state']['candidate_annotations'].items() if v['support_label']=='uncertain'};self.assertEqual(len(uncertain),2)
  self.assertTrue(all(not c['state']['candidate_annotations'][cid]['final_unit_ids'] for cid in uncertain));self.assertFalse(any(u['parent_candidate_id'] in uncertain for u in c['state']['units']))
  self.assertEqual(c['state']['candidate_annotations']['gpSv4OvISIeyBAqcNN5bqw_IMG_CAP_3_CANDIDATE']['corrected_claim_text'],'the bar at a restaurant.')
 def test_06_all_90_photos_byte_identical(self):
  for c in self.c['cases']:
   for p in c['captions']:self.assertEqual(sha(self.c['source']/p['image_file']),sha(self.c['final']/'incoming/extracted'/p['image_file']))
 def test_07_frozen_v1_code_and_config_unchanged(self):
  self.assertEqual(sha(self.c['source']/'code/stage_i_d053_annotations.py'),'b5762faf33e66591094ca384a56ad3a8b7c967d96d313f10cc6020c133f2b752');self.assertEqual(sha(self.c['source']/'FROZEN_EVALUATION_CONFIG.json'),'a139ee88824b6b5489b871cdbd5b9c13317139444f7a95033d5d9c5019a48461')
 def test_08_legacy_gate_still_blocks_exactly_24_supported_parents(self):
  c=self.c;r=validate_annotations(c['state'],c['cases'],c['candidates'],c['bindings'],final=True);self.assertFalse(r['valid']);self.assertEqual(len(r['errors']),24);self.assertTrue(all('supported candidate needs separately confirmed' in e for e in r['errors']))
 def test_09_excluded_units_never_in_scoring_units(self):
  s=self.c['state'];self.assertFalse({u['unit_id'] for u in s['units']}&{u['proposal_id'] for u in s['approved_exclusion_log']})
 def test_10_missing_core_unit_rejected(self):
  c=self.altered();c['state']['units'].pop(0);self.reject(c)
 def test_11_wording_not_approved_rejected(self):
  c=self.altered();c['state']['units'][0]['text']='Invented restaurant claim.';self.reject(c)
 def test_12_silently_reworded_quote_rejected(self):
  c=self.altered();c['state']['units'][0]['evidence'][0]['quote']='Paraphrased source';self.reject(c)
 def test_13_priority_mutation_rejected(self):
  c=self.altered();c['state']['units'][0]['importance']='supporting';self.reject(c)
 def test_14_unconfirmed_photo_rejected(self):
  c=self.altered();next(u for u in c['state']['units'] if u['domain']=='visual')['visual_photo_confirmed']=False;self.reject(c)
 def test_15_global_confirmation_pending_is_not_waived(self):
  c=self.altered();c['state']['researcher_confirmed_complete']=False;self.reject(c)
 def test_16_confirmation_snapshot_tamper_rejected(self):
  c=self.altered();next(e for e in c['state']['history'] if e['kind']=='unit_confirmed')['snapshot']['text']='Tampered';self.reject(c)
 def test_17_bad_frozen_binding_rejected(self):
  c=self.altered();c['state']['bindings']['protocol_sha256']='0'*64;self.reject(c)
 def test_18_new_unapproved_curated_role_rejected(self):
  c=self.altered();next(iter(c['curation']['unit_recommendations'].values()))['recommended_action']='exclude';self.reject(c)
 def test_19_prior_supported_label_reclassified_rejected(self):
  c=self.altered();next(a for a in c['state']['candidate_annotations'].values() if a['support_label']=='supported')['support_label']='needs_revision';self.reject(c)
 def test_20_uncertain_claim_cannot_be_promoted(self):
  c=self.altered();a=c['state']['candidate_annotations']['gpSv4OvISIeyBAqcNN5bqw_IMG_CAP_3_CANDIDATE'];a['support_label']='supported';self.reject(c)
 def test_21_excluded_unit_cannot_enter_denominator(self):
  c=self.altered();u=copy.deepcopy(c['state']['units'][0]);u['unit_id']=c['state']['approved_exclusion_log'][0]['proposal_id'];c['state']['units'].append(u);self.reject(c)
 def test_22_missing_exclusion_audit_record_rejected(self):
  c=self.altered();c['state']['approved_exclusion_log'].pop();self.reject(c)
 def test_23_exclusion_without_approved_reason_rejected(self):
  c=self.altered();c['state']['approved_exclusion_log'][0]['reason']='';self.reject(c)
 def test_24_approval_message_tampering_rejected(self):
  c=self.altered();c['approval']['approval_message_verbatim']+=' fake';self.reject(c)
 def test_25_recommendation_source_not_approval_rejected(self):
  c=self.altered();c['approval']['record_type']='AI_RECOMMENDATION';self.reject(c)
 def test_26_fabricated_ui_event_source_rejected(self):
  c=self.altered();next(e for e in c['state']['history'] if e['kind']=='unit_confirmed')['recording_method']='UI_click';self.reject(c)
 def test_27_other_pending_conditions_not_filtered(self):
  c=self.altered();c['state']['units'][0]['atomicity_confirmed']=False;self.reject(c)
 def test_28_missing_non_scoring_parent_proof_rejected(self):
  c=self.altered();c['state']['curated_reference_policy']['all_children_excluded_parent_ids'].pop();self.reject(c)
 def test_29_prior_uncertain_suggested_text_not_silently_approved_or_cleared(self):
  c=self.altered();c['state']['candidate_annotations']['gpSv4OvISIeyBAqcNN5bqw_IMG_CAP_3_CANDIDATE']['corrected_claim_text']='';self.reject(c)
 def test_30_missing_extension_version_rejected(self):
  c=self.altered();c['state']['curated_reference_policy']['extension_version']='bypass';self.reject(c)
 def test_31_each_business_keeps_positive_and_negative_review_evidence(self):
  for c in self.c['cases']:
   p={u['polarity'] for u in self.c['state']['units'] if u['business_id']==c['business_id'] and u['domain']=='review'};self.assertTrue({'positive','negative'}<=p)
 def test_32_zero_caption_case_has_zero_visual_units(self):
  self.assertFalse(any(u['domain']=='visual' and u['business_id']=='j3kGicU3mPG_ciIKG1Vuyg' for u in self.c['state']['units']))
 def test_33_business_distribution_matches_approved_counts(self):
  rows=read(self.c['final']/'REFERENCE_DISTRIBUTIONS_BY_BUSINESS.json');self.assertEqual(len(rows),24);self.assertEqual(sum(r['core'] for r in rows),261);self.assertEqual(sum(r['supporting'] for r in rows),315);self.assertEqual(sum(r['excluded'] for r in rows),88)
 def test_34_scope_and_importance_not_conflated(self):
  rows=read(self.c['final']/'CAPTION_AND_OMISSION_RESOLUTION_AUDIT.json')['captions'];self.assertEqual(Counter(r['recommended_choice'] for r in rows),{'include_caption_safe_revision':10,'exclude':5});self.assertEqual(sum(r['recommended_choice']=='include_caption_safe_revision' and not r['scoring_unit_ids'] for r in rows),3)
 def test_35_five_omission_decisions_six_scoring_units_one_excluded(self):
  rows=read(self.c['final']/'CAPTION_AND_OMISSION_RESOLUTION_AUDIT.json')['omissions'];self.assertEqual(Counter(r['recommended_choice'] for r in rows),{'core':3,'supporting':1,'excluded':1});self.assertEqual(sum(len(r['scoring_unit_ids']) for r in rows),6);self.assertEqual(sum(len(r['excluded_unit_ids']) for r in rows),1)
 def test_36_core_and_supporting_not_mandatory_word_for_word(self):
  p=read(self.c['final']/'SCORING_POPULATIONS_AND_INTERPRETATION.json');self.assertTrue(p['no_automatic_664_denominator']);self.assertTrue(p['no_universal_mandatory_576_denominator']);self.assertFalse(p['teacher_targets_are_independent_ground_truth']);self.assertFalse(p['evaluation_config_changed'])
 def test_37_extension_schema_rejects_missing_exclusion_id(self):
  p=copy.deepcopy(self.c['state']['curated_reference_policy']);p['approved_unit_exclusion_ids'].pop();schema=read(self.c['schema']/'stage_i_d053_curated_exclusion_extension_v2.schema.json')
  with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(p,schema)
 def test_38_malformed_metadata_fails_closed(self):
  c=self.altered();c['state']['units'][0]['importance']=[];self.reject(c)
 def test_39_exclusion_is_not_a_false_assertion_of_factual_support(self):
  self.assertTrue(all(not u['truth_verification_implied_by_exclusion'] and u['human_approved_exclusion'] for u in self.c['state']['approved_exclusion_log']))
 def test_40_exact_frozen_output_and_no_new_ui_claims(self):
  c=self.c;self.assertEqual(freeze_curated_annotations(*arguments(c)),read(c['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json'));a=c['approval'];self.assertIsNone(a['conversation_message_id']);self.assertIsNone(a['actual_prior_review_timestamp']);self.assertFalse(a['ui_interactions_fabricated']);self.assertFalse(a['separate_signature_required'])
if __name__=='__main__':unittest.main(verbosity=2)
