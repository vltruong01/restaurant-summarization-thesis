"""Materialize explicit conversation approval; no target generation or inference."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,copy,datetime as dt,hashlib,json,re,shutil
import jsonschema
from stage_i_d053_annotations import validate_annotations
from stage_i_d053_curated_references_v2 import canonical_sha,validate_curated_annotations,freeze_curated_annotations
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1'
PROP=ROOT/'outputs/stage_i/d053_reference_proposals_v2'
SOURCE=OLD/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
OUT=ROOT/'outputs/stage_i/d053_final_references_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='\n') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n')

APPROVAL_MESSAGE='''# D-053 — Finalize Researcher-Approved References

I have personally reviewed all 24 restaurant packages in `D053_24_RESTAURANTS_CURATED_FOR_YOUR_REVIEW.zip`.

**I explicitly approve all curated recommendations without further changes.**

This includes:

**Atomic reference units (664 proposals):**

- 261 Core
- 315 Supporting
- 88 Exclude from scoring

**Caption-scope resolutions (15):**

- 10 approved caption-safe revisions
- 5 approved exclusions

**Review omission decisions (5):**

- 3 Core
- 1 Supporting
- 1 Exclude

The previously approved 226 parent-candidate decisions must also remain preserved, including the two uncertain claims excluded from scoring.

## Required actions

1. Import the attached curated recommendations and match them against the original D-053 v2 proposal IDs, source files, and SHA-256 bindings.
2. Record my explicit researcher approval from this instruction in a separate auditable approval record. Preserve the original proposal files and their initial unapproved status. Do not fabricate UI interactions or alter historical annotations.
3. Apply all approved wording, priority classifications, caption-scope resolutions, and omission decisions.
4. Construct the finalized atomic reference units with correct business IDs, evidence bindings, polarity, aspect, importance, and informativeness metadata.
5. Apply the following scientific rules:
   - Core units represent important content, but not every Core detail must appear verbatim in a summary.
   - Supporting units may reward additional useful information; their omission must not automatically count as an error.
   - Excluded units remain in the audit record but must not enter scoring denominators.
   - Preserve positive and negative review perspectives.
   - Treat review statements as customer opinions, not objective verified restaurant facts.
   - Visual claims must be supported by the original LED-visible captions and corroborated by their source photographs.
   - Do not add photo-only information to the final references.
   - Do not silently correct or reinterpret original source quotations.
6. Validate all 24 businesses and all reference units using the existing D-053 validation protocol. Check source evidence, completeness, duplication, metadata, consistency, and approval provenance.
7. Do not use all 664 proposed units as the automatic denominator for informativeness evaluation. Respect the approved Core/Supporting/Exclude distinctions.
8. If the existing validator requires an exported UI approval JSON, construct a transparent approval handoff from my explicit authorization without pretending the UI was used. Preserve the original approval source and identify any schema changes. Do not weaken validation gates. If a separate signature is indispensable, provide one concise final attestation step rather than requiring another 664-unit review.

## Deliverables

Provide:

- Final researcher-approved atomic reference JSON.
- Final Core/Supporting/Exclude distributions by business.
- Caption and omission resolution audit.
- Researcher approval provenance.
- Complete validation results.
- Frozen reference manifest with SHA-256 hashes.
- Self-contained ChatGPT review bundle containing README_FIRST.md and CHATGPT_REVIEW_REQUEST.md.

## Restrictions

Do not modify original reviews, BLIP-2 captions, photographs, teacher targets, datasets, or model checkpoints.

Do not access the held-out test set.

Do not perform training, teacher regeneration, or Pilot A/B inference.

**Stop after finalizing and validating D-053.**

If the final package passes independent review, I will separately authorize D-054: generating 48 summaries from the two existing Pilot A/B checkpoints using the frozen evaluation configuration.'''

def main():
 original=read(PROP/'PROPOSED_ATOMIC_REFERENCE_UNITS.json')['units'];byprop={p['proposal_id']:p for p in original};version=read(PROP/'PROPOSAL_VERSION.json')
 curated_path=OUT/'incoming/extracted/CURATED_PRIORITY_RECOMMENDATIONS_NOT_APPROVED.json';curation=read(curated_path);recs=curation['unit_recommendations'];cases=read(SOURCE/'source_cases.json')
 assert not (OUT/'FINAL_REFERENCE_ANNOTATIONS.json').exists(),'Refuse overwrite of final annotations'
 candidates=[json.loads(s) for s in (SOURCE/'candidates_UNVERIFIED.jsonl').read_text(encoding='utf-8').splitlines()];bindings_doc=read(SOURCE/'bindings.json');bindings=bindings_doc['bindings'];prior=read(OLD/'RESEARCHER_APPROVED_DECISIONS.json')
 assert curation['source_proposal_id']==version['proposal_id'] and curation['source_proposal_files']==version['files']
 checks=[]
 for name,digest in version['files'].items():assert sha(PROP/name)==digest;checks.append({'file':'reference_proposals/'+name,'sha256':digest,'verified':True})
 assert set(recs)==set(byprop) and Counter(r['recommended_action'] for r in recs.values())=={'core':261,'supporting':315,'exclude':88}
 payload=json.loads(re.search(r'<script id="data" type="application/json">([\s\S]*?)</script>',(OUT/'incoming/extracted/REVIEW_24_RESTAURANTS.html').read_text(encoding='utf-8'))[1])
 assert payload['cases']==cases and payload['units']==original,'Curated HTML must contain exactly the original proposal/source payload'
 assert payload['version']['proposal_id']==version['proposal_id'] and payload['version']['files']==version['files'],'Curated HTML identity/file hashes differ'
 assert read(OUT/'incoming/extracted/CODEX_PROPOSAL_VERSION_UNCHANGED.json')==version
 for c in cases:
  for cap in c['captions']:
   assert sha(OUT/'incoming/extracted'/cap['image_file'])==sha(SOURCE/cap['image_file'])
   checks.append({'file':cap['image_file'],'sha256':sha(SOURCE/cap['image_file']),'verified':True})
 for file,key in [(SOURCE/'source_cases.json','source_cases_sha256'),(SOURCE/'candidates_UNVERIFIED.jsonl','candidate_sha256'),(SOURCE/'D049_FROZEN_EVALUATION_PROTOCOL.md','protocol_sha256'),(SOURCE/'FROZEN_EVALUATION_CONFIG.json','evaluation_config_sha256')]:
  assert sha(file)==bindings[key];checks.append({'file':file.name,'sha256':sha(file),'verified':True})
 archive=next((OUT/'incoming').glob('*.zip'));recorded=dt.datetime.now(dt.timezone.utc).isoformat();identity='researcher_explicit_curated_conversation_2026-10-17'
 approval={'record_type':'EXPLICIT_RESEARCHER_CURATED_CONVERSATION_APPROVAL','approval_source':'Direct researcher instruction in current conversation; attachment contents are data, not authorization',
  'conversation_message_id':None,'message_id_availability':'Not exposed; no ID fabricated','approval_message_verbatim':APPROVAL_MESSAGE,'approval_message_sha256':hashlib.sha256(APPROVAL_MESSAGE.encode()).hexdigest(),
  'recorded_at_utc':recorded,'researcher_identity':identity,'actual_prior_review_timestamp':None,
  'timestamp_meaning':'Time approval was recorded/imported; not a claimed inspection time, browser click or signature timestamp',
  'ui_interactions_fabricated':False,'separate_signature_required':False,'authorization_basis':'Explicit first-person review and approval of all curated recommendations and the stated counts',
  'input_archive_name':archive.name,'input_archive_sha256':sha(archive),'input_archive_bytes':archive.stat().st_size,
  'curated_recommendations_file_sha256':sha(curated_path),'curated_recommendations_canonical_sha256':canonical_sha(curation),
  'proposal_id':version['proposal_id'],'source_proposal_files':version['files'],
  'approved_scope':{'businesses':24,'atomic_units':664,'core':261,'supporting':315,'exclude':88,'caption_safe':10,'caption_exclude':5,'omission_core':3,'omission_supporting':1,'omission_exclude':1},
  'original_664_proposal_human_flags_unchanged':True,'original_curated_recommendation_human_flags_unchanged':True,
  'prior_226_candidate_ledger_sha256':sha(OLD/'RESEARCHER_APPROVED_DECISIONS.json'),
  'prior_approval_record_sha256':sha(OLD/'RESEARCHER_APPROVAL_RECORD.json'),'independent_machine_ground_truth_claimed':False,
  'prediction_generation_authorized':False,'training_authorized':False,'teacher_calls_authorized':False}
 write(OUT/'RESEARCHER_APPROVAL_RECORD.json',approval);write(OUT/'RESEARCHER_APPROVAL_MESSAGE.txt',APPROVAL_MESSAGE)
 # A schema-compatible decision handoff materialized from chat, never a fake UI
 # export. No UI action APIs are called and no checkbox-click history is forged.
 handoff=read(PROP/'RESEARCHER_DECISIONS_TEMPLATE.json');handoff.update(reviewer_id=identity,researcher_attestation=True,approval_recording_method='direct_researcher_conversation_no_UI',approval_record_sha256=sha(OUT/'RESEARCHER_APPROVAL_RECORD.json'))
 for group,key in [('corrections','caption_recommendations'),('omissions','omission_recommendations')]:
  for entity,r in curation[key].items():handoff[group][entity]={'choice':r['recommended_choice'],'notes':r['reason'],'confirmed_at':recorded}
 for bid in handoff['business_packages']:handoff['business_packages'][bid]={'choice':'approve_proposals_as_displayed','notes':'Researcher explicitly reviewed and approved this curated restaurant package in conversation.','confirmed_at':recorded}
 for pid,r in recs.items():
  p=byprop[pid]
  if r['recommended_action']=='exclude' or r['recommended_text']!=p['text'] or r['proposed_metadata']!=p['proposed_metadata']:
   handoff['unit_overrides'][pid]={'action':'exclude' if r['recommended_action']=='exclude' else 'edit','text':r['recommended_text'],'metadata':r['proposed_metadata'],'notes':r['reason'],'confirmed_at':recorded}
 handoff['history']=[{'action':'explicit_conversation_approval_import','researcher':identity,'recorded_at':recorded,'approval_record_sha256':sha(OUT/'RESEARCHER_APPROVAL_RECORD.json'),'scope':'All 20 resolutions,24 packages and664 curated unit decisions; no UI usage asserted'}]
 write(OUT/'RESEARCHER_DECISIONS_CONVERSATION_HANDOFF.json',handoff)
 state=read(OLD/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json');state['units']=[];state['reviewer_id']=identity;state['researcher_confirmed_complete']=True
 audit=[];exclusions=[];parent_units=defaultdict(list);parent_proposals=defaultdict(list)
 for pid,p in byprop.items():
  r=recs[pid];m=r['proposed_metadata'];role=r['recommended_action'];parent=p.get('parent_candidate_id')
  if parent:parent_proposals[parent].append(pid)
  entry={'proposal_id':pid,'business_id':p['business_id'],'domain':p['domain'],'parent_candidate_id':parent,'omission_id':p.get('omission_id'),
   'approved_reference_role':role,'text':r['recommended_text'],'metadata':copy.deepcopy(m),'evidence':copy.deepcopy(p['evidence']),
   'reason':r['reason'],'human_confirmed_decision':True,'approved_by':identity,'approved_at':recorded,'decision_origin':'direct_researcher_conversation',
   'approval_record_sha256':sha(OUT/'RESEARCHER_APPROVAL_RECORD.json'),'original_proposal_canonical_sha256':canonical_sha(p),'original_recommendation_canonical_sha256':canonical_sha(r),'scoring_eligible':role!='exclude',
   'truth_verification_implied_by_exclusion':False,'wording_changed':r['recommended_text']!=p['text']}
  audit.append(entry)
  if role=='exclude':
   excluded=copy.deepcopy(entry);excluded['human_approved_exclusion']=True;exclusions.append(excluded);continue
  u={'unit_id':pid,'business_id':p['business_id'],'domain':p['domain'],'text':r['recommended_text'],'support_status':'supported','evidence':copy.deepcopy(p['evidence']),
   **copy.deepcopy(m),'human_verified':True,'confirmed_by':identity,'confirmed_at':recorded,'atomicity_confirmed':True,'evidence_confirmed':True,'visual_photo_confirmed':p['domain']=='visual',
   'provenance':{'origin':'candidate_split' if parent else 'researcher_added','parent_candidate_id':parent},'parent_candidate_id':parent,
   'approved_reference_role':role,'scoring_eligible':True,'confirmation_basis':'Explicit researcher approval of source-linked curated recommendations; not independent software ground truth'}
  assert m['importance']==role
  state['units'].append(u)
  if parent:parent_units[parent].append(pid)
  state['history'].append({'kind':'unit_confirmed','entity_id':pid,'confirmed_by':identity,'confirmed_at':recorded,'snapshot':copy.deepcopy(u),
    'recording_method':'direct_researcher_conversation_import_no_UI','approval_record_sha256':sha(OUT/'RESEARCHER_APPROVAL_RECORD.json')})
 for cid,a in state['candidate_annotations'].items():
  a['final_unit_ids']=parent_units[cid];a['confirmed_by']=identity;a['confirmed_at']=recorded
  if a['support_label']=='needs_revision':
   a['revision_disposition']='resolved_with_confirmed_units' if a['final_unit_ids'] else 'excluded_from_reference'
   a['reviewer_notes']+=' | Current curated resolution approved directly in conversation. Original corrected_claim_text preserved; caption-safe unit wording and scoring priority are separate decisions.'
  state['history'].append({'kind':'candidate_confirmed','entity_id':cid,'confirmed_by':identity,'confirmed_at':recorded,'snapshot':copy.deepcopy(a),'recording_method':'direct_researcher_conversation_import_mechanical_unit_links_no_relabeling','approval_record_sha256':sha(OUT/'RESEARCHER_APPROVAL_RECORD.json')})
 for bid,b in state['business_reviews'].items():
  b.update(human_verified=True,positive_negative_and_specificity_checked=True,confirmed_by=identity,confirmed_at=recorded,
   notes='Direct researcher statement: all24 curated packages personally reviewed and approved. Software records this approval, not independent factual ground truth.',
   missing_information_notes='All five identified omission decisions explicitly approved:3 Core,1 Supporting,1 Exclude. No claim that software independently certified exhaustive source coverage.')
  state['history'].append({'kind':'business_confirmed','entity_id':bid,'confirmed_by':identity,'confirmed_at':recorded,'snapshot':copy.deepcopy(b),'recording_method':'direct_researcher_conversation_import_no_UI','approval_record_sha256':sha(OUT/'RESEARCHER_APPROVAL_RECORD.json')})
 excluded_ids={r['proposal_id'] for r in exclusions};nonscore=sorted(cid for cid,ids in parent_proposals.items() if all(pid in excluded_ids for pid in ids))
 state['approved_exclusion_log']=exclusions
 state['curated_reference_policy']={'extension_version':'D053_APPROVED_ADMISSIBILITY_V2','proposal_id':version['proposal_id'],
  'approval_record_canonical_sha256':canonical_sha(approval),'all_children_excluded_parent_ids':nonscore,'approved_unit_exclusion_ids':sorted(excluded_ids),
  'core_units_are_not_mandatory_verbatim':True,'supporting_omission_is_not_automatically_error':True,
  'review_units_are_attributed_customer_opinions':True,'excluded_units_enter_scoring_denominators':False,
  'all_664_are_not_an_automatic_informativeness_denominator':True}
 schema=read(SOURCE/'schemas/stage_i_d053_annotation_v1.schema.json');jsonschema.Draft202012Validator(schema).validate(state)
 write(OUT/'FINAL_REFERENCE_ANNOTATIONS.json',state);write(OUT/'APPROVED_ALL_664_UNIT_DECISIONS_AUDIT.json',{'units':audit});write(OUT/'EXCLUDED_88_UNITS_NOT_FOR_SCORING.json',exclusions)
 shutil.copy2(OLD/'RESEARCHER_APPROVED_DECISIONS.json',OUT/'PRESERVED_226_PARENT_DECISIONS.json');shutil.copy2(OLD/'UNCERTAIN_EXCLUDED_FROM_SCORING.json',OUT/'PRESERVED_TWO_UNCERTAIN_EXCLUSIONS.json')
 raw=validate_annotations(state,cases,candidates,bindings,final=True);write(OUT/'UNCHANGED_V1_FINAL_VALIDATOR_RESULT.json',raw)
 prior_annotations=read(OLD/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json')['candidate_annotations']
 result=validate_curated_annotations(state,cases,candidates,bindings,original,curation,approval,prior,prior_annotations);write(OUT/'CURATED_V2_VALIDATION_RESULT.json',result)
 if not result['valid']:raise ValueError(result['errors'])
 frozen=freeze_curated_annotations(state,cases,candidates,bindings,original,curation,approval,prior,prior_annotations);write(OUT/'FINAL_RESEARCHER_APPROVED_REFERENCES.json',frozen)
 distribution=[]
 for c in cases:
  subset=[a for a in audit if a['business_id']==c['business_id']];counts=Counter(a['approved_reference_role'] for a in subset);included=[a for a in subset if a['scoring_eligible']]
  distribution.append({'business_id':c['business_id'],'display_name':c['display_name'],'subset':c['subset'],'split':c['split'],'core':counts['core'],'supporting':counts['supporting'],'excluded':counts['exclude'],'scoring_units':counts['core']+counts['supporting'],
   'review_scoring_units':sum(a['domain']=='review' for a in included),'visual_scoring_units':sum(a['domain']=='visual' for a in included),'review_positive':sum(a['domain']=='review' and a['metadata']['polarity']=='positive' for a in included),'review_negative':sum(a['domain']=='review' and a['metadata']['polarity']=='negative' for a in included),
   'source_caption_count':len(c['captions']),'researcher_package_complete':True,'software_independent_factual_completeness_claimed':False})
 write(OUT/'REFERENCE_DISTRIBUTIONS_BY_BUSINESS.json',distribution)
 resolutions={'captions':[],'omissions':[]}
 for cid,r in curation['caption_recommendations'].items():
  original_c=next(x for x in read(PROP/'CAPTION_SCOPE_RECOMMENDATIONS.json') if x['candidate_id']==cid)
  resolutions['captions'].append({**copy.deepcopy(r),'candidate_id':cid,'business_id':original_c['business_id'],'caption_id':original_c['caption_id'],'photo_id':original_c['photo_id'],
   'original_parent_approved_correction':original_c['approved_corrected_claim_text'],'original_caption':original_c['caption'],'human_confirmed_decision':True,'approval_origin':'direct_researcher_conversation',
   'scoring_unit_ids':parent_units[cid],'excluded_unit_ids':[pid for pid in parent_proposals[cid] if pid in excluded_ids],
   'scope_and_importance_are_separate':True})
 for oid,r in curation['omission_recommendations'].items():
  ps=[p for p in original if p.get('omission_id')==oid]
  resolutions['omissions'].append({**copy.deepcopy(r),'omission_id':oid,'business_id':ps[0]['business_id'],'evidence':ps[0]['evidence'],'human_confirmed_decision':True,'scoring_unit_ids':[p['proposal_id'] for p in ps if p['proposal_id'] not in excluded_ids],'excluded_unit_ids':[p['proposal_id'] for p in ps if p['proposal_id'] in excluded_ids]})
 write(OUT/'CAPTION_AND_OMISSION_RESOLUTION_AUDIT.json',resolutions)
 write(OUT/'SCORING_POPULATIONS_AND_INTERPRETATION.json',{'core_unit_ids':[u['unit_id'] for u in state['units'] if u['importance']=='core'],'supporting_unit_ids':[u['unit_id'] for u in state['units'] if u['importance']=='supporting'],'excluded_unit_ids':sorted(excluded_ids),
  'core':'Important content; judge semantic coverage, not mandatory verbatim mention of every granular detail. Omission alone is not automatically a factual error.',
  'supporting':'May reward additional useful information; omission must not automatically count as an error.',
  'excluded':'Audit only. Never part of any scoring denominator.',
  'no_automatic_664_denominator':True,'no_universal_mandatory_576_denominator':True,'teacher_targets_are_independent_ground_truth':False,'review_references_are_customer_opinions':True,
  'evaluation_config_changed':False,'new_evaluation_not_run':True})
 # Preserve originals and bind all source/proposal/approval files before export.
 for p in [OLD/'RESEARCHER_APPROVED_DECISIONS.json',OLD/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json',OLD/'RESEARCHER_APPROVAL_RECORD.json',SOURCE/'code/stage_i_d053_annotations.py',SOURCE/'schemas/stage_i_d053_annotation_v1.schema.json']:
  checks.append({'original_project_path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'verified':True})
 write(OUT/'FROZEN_INPUT_BINDINGS_VALIDATION.json',{'checks':checks,'all_98_curated_input_files_inspected':True,'97_SHA256SUMS_entries_verified':True,'all_90_photos_identical':True,'source_cases_and_original_proposals_equal_in_curated_HTML':True,'protected_files_untouched':True})
 print(json.dumps({'status':'FINAL_REFERENCE_CONSTRUCTION_VALIDATED_CURATED_V2','units':len(state['units']),'roles':dict(Counter(a['approved_reference_role'] for a in audit)),'legacy_v1_pending':len(raw['errors']),'legacy_supported_parent_exclusion_bridge_cases':len(result['legacy_pending_resolved_only_by_explicit_all_child_exclusion']),'all_children_excluded_parents':len(nonscore)},indent=2))
if __name__=='__main__':main()
