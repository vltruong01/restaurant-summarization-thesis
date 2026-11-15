"""Strict D053 approved-admissibility bridge; frozen v1 validator is unchanged.

Support of a parent claim is independent of whether its approved children are
useful scoring references. Only the exact all-children-explicitly-excluded case
can resolve v1's mandatory-child pending condition. All v1 errors, unit readiness,
quote, binding and confirmation checks remain mandatory. No semantic AI judge.
"""
from collections import Counter,defaultdict
import copy,hashlib,json
from stage_i_d053_annotations import validate_annotations,BINDING_KEYS

def canonical_sha(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _validate_curated_annotations(state,cases,candidates,bindings,proposals,curation,approval,prior_decisions,prior_annotations):
 errors=[]
 def check(ok,msg):
  if not ok:errors.append(msg)
 raw=validate_annotations(state,cases,candidates,bindings,final=False)
 errors.extend(raw['errors'])
 byproposal={p['proposal_id']:p for p in proposals}; recs=curation.get('unit_recommendations',{}); units=state.get('units',[]);byunit={u['unit_id']:u for u in units}
 check(len(proposals)==len(byproposal)==664,'Original 664 proposal IDs required')
 check(set(recs)==set(byproposal),'Curated proposal ID set differs')
 check(Counter(r['recommended_action'] for r in recs.values())=={'core':261,'supporting':315,'exclude':88},'Approved role distribution differs')
 check(approval.get('record_type')=='EXPLICIT_RESEARCHER_CURATED_CONVERSATION_APPROVAL','Missing direct conversation approval')
 check(approval.get('ui_interactions_fabricated') is False,'Approval source must not impersonate UI')
 check(state.get('reviewer_id')==approval.get('researcher_identity'),'Researcher identity differs from explicit approval')
 check(approval.get('approval_message_sha256')==hashlib.sha256(approval.get('approval_message_verbatim','').encode()).hexdigest(),'Approval message hash mismatch')
 check(approval.get('curated_recommendations_canonical_sha256')==canonical_sha(curation),'Curated decisions not bound to approval')
 check(approval.get('approved_scope')=={'businesses':24,'atomic_units':664,'core':261,'supporting':315,'exclude':88,'caption_safe':10,'caption_exclude':5,'omission_core':3,'omission_supporting':1,'omission_exclude':1},'Explicit approval scope differs')
 check(curation.get('source_proposal_id')==approval.get('proposal_id'),'Proposal approval binding mismatch')
 check(curation.get('source_proposal_files')==approval.get('source_proposal_files'),'Proposal file hash bindings differ')
 exclusions=state.get('approved_exclusion_log',[]);excluded={u['proposal_id']:u for u in exclusions}
 check(len(excluded)==len(exclusions)==88,'88 distinct excluded decisions required')
 accepted={k for k,v in recs.items() if v['recommended_action']!='exclude'};excluded_expected=set(recs)-accepted
 check(set(byunit)==accepted and len(units)==len(byunit)==576,'Scoring units must be exactly 261 Core +315 Supporting, never all 664')
 check(set(excluded)==excluded_expected and not set(excluded)&set(byunit),'Excluded units must stay outside scoring')
 parent_proposals=defaultdict(list)
 bycase={c['business_id']:c for c in cases}
 for pid,p in byproposal.items():
  if p.get('parent_candidate_id'):parent_proposals[p['parent_candidate_id']].append(pid)
  r=recs.get(pid,{})
  case=bycase[p['business_id']];dom=p['domain'];source={x['marker' if dom=='review' else 'caption_id']:x for x in case['reviews' if dom=='review' else 'captions']}
  for e in p['evidence']:
   x=source.get(e['marker'],{});field='review_id' if dom=='review' else 'photo_id'
   check(e.get(field)==x.get(field) and bool(e.get('quote')) and e['quote'] in x.get('text',''),'Exact permitted source binding failed, including excluded audit evidence: '+pid)
  check(r.get('proposal_id')==pid and r.get('business_id')==p['business_id'],'Curated ID/business mismatch: '+pid)
  check(r.get('proposed_metadata',{}).get('polarity')==p['proposed_metadata']['polarity'],'Opinion polarity changed without approved original binding: '+pid)
  if pid in byunit:
   u=byunit[pid];m=r['proposed_metadata']
   check(u['text']==r['recommended_text'],'Final wording differs from approved wording: '+pid)
   check(u['evidence']==p['evidence'],'Source evidence was modified: '+pid)
   check(u['business_id']==p['business_id'] and u['domain']==p['domain'],'Unit business/domain mismatch: '+pid)
   check(u.get('parent_candidate_id')==p.get('parent_candidate_id'),'Unit parent changed: '+pid)
   check(u.get('scoring_eligible') is True and u.get('approved_reference_role')==r['recommended_action'],'Scoring role mismatch: '+pid)
   for key in ['importance','informativeness','aspect','restaurant_specific','polarity']:check(u.get(key)==m[key],'Approved metadata mismatch '+pid+' / '+key)
   check(u['importance']==r['recommended_action'],'Core/Supporting must match approved role: '+pid)
   check(u.get('confirmed_by')==approval['researcher_identity'] and u.get('confirmed_at')==approval['recorded_at_utc'],'Unit confirmation provenance differs: '+pid)
  if pid in excluded:
   e=excluded[pid]
   check(e.get('human_approved_exclusion') is True and e.get('scoring_eligible') is False,'Exclusion not explicitly approved: '+pid)
   check(e.get('decision_origin')=='direct_researcher_conversation' and e.get('approved_at')==approval['recorded_at_utc'] and e.get('approved_by')==approval['researcher_identity'],'Exclusion provenance missing: '+pid)
   check(e.get('reason')==r['reason'] and bool(e.get('reason')),'Exclusion reason missing or changed: '+pid)
   check(e.get('text')==r['recommended_text'] and e.get('evidence')==p['evidence'] and e.get('metadata')==r['proposed_metadata'],'Excluded audit content changed: '+pid)
   check(e.get('business_id')==p['business_id'] and e.get('parent_candidate_id')==p.get('parent_candidate_id'),'Excluded scope differs: '+pid)
 annotations=state.get('candidate_annotations',{});previous={r['candidate_id']:r for r in prior_decisions['decisions']}
 check(len(previous)==226 and set(previous)==set(annotations),'Original 226 candidate decisions required')
 for cid,r in previous.items():
  a=annotations.get(cid,{})
  check(a.get('support_label')==r['decision'],'Prior candidate label altered: '+cid)
  for field in ['corrected_claim_text','importance','informativeness']:
   check(a.get(field)==prior_annotations.get(cid,{}).get(field),'Prior original annotation field altered: '+cid+' / '+field)
  if r['approved_corrected_claim_text'] is not None:
   check(a.get('corrected_claim_text')==r['approved_corrected_claim_text'],'Approved correction altered: '+cid)
 # Resolve only a representational pending condition, never any evidence or
 # provenance error. Every child must be an approved exclusion in the canonical
 # curation; all exclusions are independently accounted for above.
 non_scoring_parents={cid for cid,ids in parent_proposals.items() if ids and all(pid in excluded_expected for pid in ids)}
 allowed_pending={cid+': supported candidate needs separately confirmed atomic unit(s)' for cid in non_scoring_parents if annotations.get(cid,{}).get('support_label')=='supported'}
 check(set(raw['pending'])==allowed_pending,'Unresolved v1 readiness condition not authorized by all-child exclusion: '+str(set(raw['pending'])-allowed_pending))
 policy=state.get('curated_reference_policy',{})
 check(policy.get('extension_version')=='D053_APPROVED_ADMISSIBILITY_V2','Missing versioned admissibility extension')
 check(policy.get('all_children_excluded_parent_ids')==sorted(non_scoring_parents),'Approved non-scoring-parent coverage mismatch')
 check(policy.get('approved_unit_exclusion_ids')==sorted(excluded_expected),'Approved exclusion IDs mismatch')
 check(policy.get('approval_record_canonical_sha256')==canonical_sha(approval),'Approval record fingerprint mismatch')
 check(policy.get('proposal_id')==curation.get('source_proposal_id'),'Policy proposal binding differs')
 check(policy.get('core_units_are_not_mandatory_verbatim') is True and policy.get('supporting_omission_is_not_automatically_error') is True,'Importance interpretation rules missing')
 # Caption scope and importance are two different axes: an approved safe wording
 # may still be excluded for low information. A scope-excluded parent may not
 # contribute even one scoring unit.
 cap_recs=curation.get('caption_recommendations',{})
 check(Counter(r['recommended_choice'] for r in cap_recs.values())=={'include_caption_safe_revision':10,'exclude':5},'Caption resolutions differ')
 for cid,r in cap_recs.items():
  ids=parent_proposals.get(cid,[]);check(bool(ids),'Unknown caption-resolution parent '+cid)
  if ids:check(r['original_caption']==byproposal[ids[0]]['evidence'][0]['quote'],'Curated caption text differs from original evidence: '+cid)
  if r['recommended_choice']=='exclude':check(all(pid in excluded_expected for pid in ids),'Scope-excluded caption entered scoring: '+cid)
  if r['recommended_choice']=='include_caption_safe_revision':check(' '.join(recs[pid]['recommended_text'] for pid in ids)==r['safe_wording'],'Caption-safe approved wording differs: '+cid)
 for oid,r in curation.get('omission_recommendations',{}).items():
  ids=[p['proposal_id'] for p in proposals if p.get('omission_id')==oid]
  check(bool(ids),'Unknown omission '+oid)
  expected='exclude' if r['recommended_choice']=='excluded' else r['recommended_choice']
  check(all(recs[pid]['recommended_action']==expected for pid in ids),'Omission role inconsistency '+oid)
 check(Counter(r['recommended_choice'] for r in curation.get('omission_recommendations',{}).values())=={'core':3,'supporting':1,'excluded':1},'Omission scope differs')
 # A claim of direct confirmation must carry the actual import source. Old
 # history stays intact; the latest new snapshots bind the current unit links.
 approval_file_sha=hashlib.sha256((json.dumps(approval,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
 latest={(e.get('kind'),e.get('entity_id')):e for e in state.get('history',[]) if e.get('kind') in ['unit_confirmed','candidate_confirmed','business_confirmed']}
 for kind,ids in [('unit_confirmed',byunit),('candidate_confirmed',annotations),('business_confirmed',bycase)]:
  for eid in ids:
   event=latest.get((kind,eid),{})
   check(event.get('recording_method','').startswith('direct_researcher_conversation') and event.get('approval_record_sha256')==approval_file_sha,'Confirmation import source missing or falsified: '+eid)
 for bid,b in state.get('business_reviews',{}).items():
  check(b.get('confirmed_by')==approval['researcher_identity'] and b.get('confirmed_at')==approval['recorded_at_utc'],'Business confirmation does not match approval: '+bid)
 for declared in curation.get('business_review_summary',[]):
  actual=Counter(r['recommended_action'] for r in recs.values() if r['business_id']==declared['business_id'])
  check(dict(actual)==declared['counts'],'Curated business summary counts differ: '+declared['business_id'])
 # Exact duplicates within one business/domain/photo evidence scope must not
 # inflate denominators. Different photo or polarity is not auto-collapsed.
 seen=set()
 for u in units:
  key=(u['business_id'],u['domain'],' '.join(u['text'].lower().split()),u['polarity'],tuple(sorted(e.get('photo_id','') for e in u['evidence'])))
  check(key not in seen,'Exact duplicate scoring proposition: '+u['unit_id']);seen.add(key)
 return {'valid':not errors,'ready_for_freeze':not errors,'errors':errors,
  'legacy_v1_structural_and_evidence_valid':raw['valid'],'legacy_v1_ready_for_freeze':raw['ready_for_freeze'],
  'legacy_pending_resolved_only_by_explicit_all_child_exclusion':sorted(allowed_pending),
  'legacy_original_pending_count':len(raw['pending']),'all_children_non_scoring_parents':len(non_scoring_parents),
  'counts':{'businesses':24,'candidates':226,'scoring_units':len(units),'core':sum(u['importance']=='core' for u in units),'supporting':sum(u['importance']=='supporting' for u in units),'excluded':len(exclusions)},
  'software_does_not_establish_independent_human_truth':True,'schema_or_existing_validator_weakened':False,
  'compatibility_notice':'Frozen v1 validator remains unchanged and reports its mandatory-child pending conditions. Versioned v2 adds exact, approval-bound non-scoring admissibility accounting; all v1 errors and all unit/evidence/confirmation checks remain fatal.'}

def validate_curated_annotations(*args,**kwargs):
 try:return _validate_curated_annotations(*args,**kwargs)
 except (KeyError,TypeError,ValueError,AttributeError) as exc:
  return {'valid':False,'ready_for_freeze':False,'errors':['Malformed curated reference data: '+str(exc)]}

def freeze_curated_annotations(state,cases,candidates,bindings,proposals,curation,approval,prior_decisions,prior_annotations):
 result=validate_curated_annotations(state,cases,candidates,bindings,proposals,curation,approval,prior_decisions,prior_annotations)
 if not result['valid']:raise ValueError('Curated reference freeze blocked: '+'; '.join(result['errors']))
 frozen=copy.deepcopy(state);frozen.update({key:bindings[key] for key in BINDING_KEYS})
 frozen['status']='RESEARCHER_CONFIRMED_SOURCE_REFERENCES_D053_CURATED_V2'
 frozen['reference_format_version']=2
 frozen['compatibility_notice']='Use approved-admissibility v2 validation. An unmodified v1 finalization/evaluation loader intentionally refuses all-child-excluded Supported parents; never bypass this gate.'
 return frozen
