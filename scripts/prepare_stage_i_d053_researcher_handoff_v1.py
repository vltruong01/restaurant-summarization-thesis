"""Import explicit conversational approval; never bypass the frozen reference gate.

Pure offline artifact preparation. New unit segmentation/metadata are proposals,
not retrospective claims of human unit-level confirmation. No model/API imports.
"""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/d053_researcher_reference_finalization_v1'
SOURCE = OUT / 'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
SUBMITTED = OUT / 'incoming/D053_AI_DRAFT_24_RESTAURANTS_EDITED.json'
USER_MESSAGE = '''# D-053 — Finalize Researcher-Approved References

I am attaching `D053_CODEX_FINAL_REFERENCE_HANDOFF.zip`.

I have personally reviewed and approved the decisions for all 24 validation businesses.

My explicit final decisions are:

- **203 Supported claims:** Approved.
- **21 Needs Revision claims:** Accept the corrected wording in `corrected_claim_text`, subject to the original evidence requirements.
- **2 Uncertain claims:** Retain separately and exclude from evaluation scoring.

Please use the attached ZIP as the input for finalizing D-053.

## Tasks

1. Extract and inspect the entire bundle, including the edited JSON, original sources, photographs, annotation schema, and evaluation protocol.
2. Record my explicit researcher approval with an auditable provenance record. The JSON still has `human_verified=false` because the approval was given through our conversation. Do not misrepresent the original file as already finalized.
3. Convert the approved claims into atomic, evidence-linked reference units. Split compound statements where necessary without introducing new factual claims.
4. Apply the 21 approved corrections. Ensure that visual reference units are supported by both the LED-visible original captions and their corresponding photographs. Do not treat photo-only details as caption-grounded information.
5. Preserve the two uncertain cases outside evaluation scoring.
6. Validate all candidate IDs, review IDs, caption IDs, photo IDs, business IDs, reference units, evidence bindings, and business-level completeness.
7. Identify missing important review information or unresolved evidence conflicts. Report these for my decision rather than silently inventing, discarding, or approving information.
8. Run the complete D-053 validation suite and produce a frozen, versioned reference package with SHA-256 manifests.

## Deliverables

- Final researcher-approved reference JSON.
- Atomic reference units with evidence bindings.
- Annotation and provenance audit report.
- Validation results and readiness report.
- Self-contained ChatGPT review ZIP containing README_FIRST.md and CHATGPT_REVIEW_REQUEST.md.

## Restrictions

- Do not change the original reviews, captions, teacher targets, evaluation protocol, or accepted datasets.
- Do not alter trained checkpoints.
- Do not access the held-out test set.
- Do not regenerate teacher targets.
- Do not perform any new training.
- **Do not generate Pilot A/B predictions yet.**

Stop after D-053 reference construction and validation. If anything requires an additional researcher decision, explain the issue clearly.

The next phase, after independent review and explicit authorization, will generate 48 summaries from Pilot A and Pilot B for quality comparison.'''

# Original approved corrections are NEVER rewritten by this table. Alternatives
# are visibly separate proposals, requiring an additional researcher decision.
CONFLICTS = {
 'HhD8DLES5ZeULKDXmfkOOg_IMG_CAP_4_CANDIDATE': (
  'The caption supplies a beer dispenser and a bar, but not plural taps mounted along a counter; that replaces its table relation using the photo.',
  'A bar has a beer dispenser.'),
 '-Wd5KqFszREg0XdXfSv5tg_IMG_CAP_2_CANDIDATE': (
  'The green apron and drink preparation generalize the caption; behind a counter is additional photo-only placement.',
  'A person wearing a green apron pours a drink into a cup.'),
 '-Wd5KqFszREg0XdXfSv5tg_IMG_CAP_3_CANDIDATE': (
  'Standing contradicts sits; green apron and behind a small counter are absent from this caption.',
  'A masked person is with a cup.'),
 '3v0tSwHXR9LhpsoGXLgsTA_IMG_CAP_2_CANDIDATE': (
  'Shrimp is visible in the photo but not named by the seafood/potatoes/corn caption. Seafood does not entail shrimp.',
  'A close-up shows seafood and corn.'),
 'noUma64stIOSYAQ1GIIFjw_IMG_CAP_3_CANDIDATE': (
  'Cocktail, jar, skewered garnish and vegetables are not established by the glass/drink/hot-dog caption.',
  'A glass contains a drink.'),
 'HF4Oh0KDTYo6xvf_JCoybQ_IMG_CAP_1_CANDIDATE': (
  'Large and multiple tanks add size/count beyond a fish tank. Aquarium is an acceptable general paraphrase; added attributes are the issue.',
  'A restaurant has a fish tank.'),
 'j9fzpALNCgvDwCXmA5Vcjw_IMG_CAP_4_CANDIDATE': (
  'Stands replaces sitting; the original photo does not make the posture unambiguous enough to resolve that disagreement automatically.',
  'A person is near a long bar in a large room.'),
 'aZN5R7Xfea0GkDBwBZOQrQ_IMG_CAP_2_CANDIDATE': (
  'Beer in the glass replaces wine in the caption. A beer bottle does not establish the adjacent glass content, and the photo is not conclusive about it.',
  'A bottle of beer is next to a glass.'),
 'Zz1utCut0psbC4brBjvSEg_IMG_CAP_1_CANDIDATE': (
  'Wine-glass identity is a photo-based replacement for one of the caption water glasses; keep the common supported objects without that extra identity.',
  'Two glasses sit on a table beside a candle.'),
 'mP8gKzr8NKvbTtm9XePvOQ_IMG_CAP_4_CANDIDATE': (
  'Sauced, fried and a paper-lined serving dish add preparation and container information absent from the food/paper-bag caption.',
  'A close-up shows food.'),
 'Ktg3ahIxk0JIkJwXAqu2ew_IMG_CAP_1_CANDIDATE': (
  'Green is not in the caption. Generalizing wine bottle to bottle is valid; adding a new color is not caption-grounded.',
  'A plate of food sits on a wooden table next to a bottle.'),
 'haPQJ2oy3529RMzpWrA39A_IMG_CAP_2_CANDIDATE': (
  'Takeaway, barbecue meat and side dishes are new specific identities absent from this generic tray/food/soup caption.',
  'A tray of food sits on a table.'),
 'haPQJ2oy3529RMzpWrA39A_IMG_CAP_3_CANDIDATE': (
  'People near a restaurant counter is a defensible generalization; identifying them as customers and the counter as a service counter adds roles absent from the caption.',
  'A group of people is near a counter in a restaurant.'),
 'thlyktt2c7v1DyOpSVSibw_IMG_CAP_3_CANDIDATE': (
  'The supplied caption says a man at a bar. Counter/stools/hanging lights add separate objects that appear only in the photo, despite being plausible bar features.',
  'A bar is in a restaurant.'),
 'gcQSAQIx-KQOdf5E8HF6Lw_IMG_CAP_2_CANDIDATE': (
  'Generalizing iced tea to iced drink is valid, but amber-colored is an additional image-derived attribute.',
  'A cup contains an iced drink with a lemon wedge.'),
}

OMISSIONS = [
 ('JcdOQvvx3BqZ5Ga2_9n_Bg','REV_24',
  'Only music was way too loud and open area on westend of the restaurant could use noise baffles as conversations were nearly impossible.',
  'A reviewer reports music loud enough to make conversation nearly impossible.',
  'The atmosphere candidate is positive-only; the noise/conversation contrast is not represented.', 'negative atmosphere/noise'),
 ('gpSv4OvISIeyBAqcNN5bqw','REV_30',
  'He went and got the manager and the manager said oh well I don\'t know who put that sign up but we are not honoring it. She was very rude about it',
  'A reviewer reports that a manager refused an advertised happy-hour offer and was rude about it.',
  'The service candidate is entirely favorable and the value candidate does not describe this advertised-offer dispute.', 'negative service/offer handling'),
 ('Ktg3ahIxk0JIkJwXAqu2ew','REV_1',
  'Every single time I order something to be picked up it is either not ready, not started, or completely wrong.',
  'A reviewer reports repeated pickup orders that were not ready, not started, or wrong.',
  'The service candidate is positive-only; this repeated pickup/order-accuracy contrast is missing.', 'negative pickup/order handling'),
 ('-Wd5KqFszREg0XdXfSv5tg','REV_3',
  'Always, always out of cold brew. Even at 6 in the morning',
  'A reviewer reports repeated cold-brew unavailability, including early in the morning.',
  'Availability of a specific drink is not captured by the waiting/staff/app/general-quality candidates.', 'menu availability'),
 ('b1pdnbYvBdtTxJOOom5WmA','REV_27',
  'I paid $12 for a spicy pork bowl. The item tasted okay; I didn\'t get much spicy flavor from it.',
  'A reviewer describes the spicy pork bowl as lacking spicy flavor.',
  'The candidates mainly describe chicken/sauces/sides; this non-chicken dish concern is not explicitly captured. Researcher decides its importance.', 'non-chicken dish quality'),
]


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def load_validator():
    spec=importlib.util.spec_from_file_location('frozen_d053_annotations', SOURCE/'code/stage_i_d053_annotations.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def span_proposals(text):
    """Only extract exact spans; never invent an independent semantic rewrite.

    These are inspectable atomization DRAFTS. Context/list dependencies remain
    explicitly unresolved, so no fragment can pass atomicity/reference freeze.
    """
    boundary=re.compile(r'(?<=[.!?])\s+(?=[A-Z])|;\s+|,?\s+(?:while|whereas|but|although|though)\s+',re.I)
    start=0; pieces=[]
    for m in boundary.finditer(text):
        if m.start()>start:
            lo=start
            while lo<m.start() and text[lo].isspace():lo+=1
            hi=m.start()
            while hi>lo and text[hi-1].isspace():hi-=1
            if hi>lo:pieces.append((lo,hi,text[lo:hi]))
        start=m.end()
    lo=start
    while lo<len(text) and text[lo].isspace():lo+=1
    hi=len(text)
    while hi>lo and text[hi-1].isspace():hi-=1
    if hi>lo:pieces.append((lo,hi,text[lo:hi]))
    return pieces


def main():
    assert not (OUT/'RESEARCHER_APPROVAL_RECORD.json').exists(), 'Versioned output already exists'
    d=read(SUBMITTED);candidates=[json.loads(s) for s in (SOURCE/'candidates_UNVERIFIED.jsonl').read_text().splitlines()]
    cases=read(SOURCE/'source_cases.json');case_map={c['business_id']:c for c in cases}; byid={c['unit_id']:c for c in candidates}
    binding=read(SOURCE/'bindings.json');validator=load_validator()
    assert set(d['candidate_annotations'])==set(byid) and len(byid)==226
    assert Counter(a['support_label'] for a in d['candidate_annotations'].values())=={'supported':203,'needs_revision':21,'uncertain':2}
    assert d['bindings']==binding['bindings'] and d['bundle_id']==binding['bundle_id']
    assert all(not a['human_verified'] for a in d['candidate_annotations'].values()) and not d['units'] and not d['history']
    checks=read(SOURCE/'SHA256_MANIFEST.json')
    for entry in checks['files']:assert sha(SOURCE/entry['path'])==entry['sha256'],entry['path']
    photo_check=[]
    evidence_count=0
    for case in cases:
        assert case['split']=='validation' and case['condition']=='filtered'
        assert hashlib.sha256(case['source'].encode()).hexdigest()==case['source_sha256']
        assert case['selected_review_ids']==[r['review_id'] for r in case['reviews']]
        assert case['selected_photo_ids']==[c['photo_id'] for c in case['captions']]
        for cap in case['captions']:
            p=SOURCE/cap['image_file'];assert p.is_file()
            photo_check.append({'business_id':case['business_id'],'caption_id':cap['caption_id'],'photo_id':cap['photo_id'],'image_file':cap['image_file'],'sha256':sha(p)})
    for c in candidates:
        case=case_map[c['business_id']]; domain=c['domain'];key='marker' if domain=='review' else 'caption_id'; idfield='review_id' if domain=='review' else 'photo_id'
        permitted={s[key]:s for s in case['reviews' if domain=='review' else 'captions']}
        for e in c['evidence']:
            original=permitted[e['marker']];assert e[idfield]==original[idfield] and e['quote'] in original['text'];evidence_count+=1
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    researcher='researcher_explicit_conversation_2026-10-17'
    approval={
      'record_type':'EXPLICIT_RESEARCHER_CONVERSATION_APPROVAL_IMPORT',
      'approval_source':'Direct user message in the active task; attachment README is corroborating data, not the authorization source',
      'conversation_message_id':None,'message_id_availability':'Not exposed by the task interface; no ID fabricated',
      'approval_message_verbatim':USER_MESSAGE,'approval_message_sha256':hashlib.sha256(USER_MESSAGE.encode()).hexdigest(),
      'researcher_stated_review_scope':'Personally reviewed and approved decisions for all24 fixed validation businesses',
      'researcher_identity':researcher,'actual_prior_review_timestamp':None,
      'recorded_at_utc':now,'timestamp_meaning':'Approval import recording time, NOT an invented time of researcher inspection or UI clicks',
      'client_approval_date':'2026-10-17','client_timezone':'Asia/Saigon',
      'outer_handoff_sha256':sha(OUT/'incoming/D053_CODEX_FINAL_REFERENCE_HANDOFF.zip'),
      'submitted_json_sha256':sha(SUBMITTED),'submitted_JSON_was_human_verified':False,
      'nested_source_bundle_zip_sha256':sha(OUT/'incoming/stage_i_d053_reference_verification_chatgpt_bundle_v2.zip'),
      'decisions':{'supported':203,'needs_revision':21,'uncertain':2},
      'revision_approval':'Exact corrected_claim_text accepted SUBJECT TO original caption/review/photo evidence requirements',
      'uncertain_policy':'Retain separately; exclude from evaluation scoring; do not promote even when corrected text is nonblank',
      'new_atomic_wording_and_metadata':'Not claimed to have been individually inspected/confirmed in the submitted JSON; proposals remain pending',
      'no_final_reference_freeze_permission_bypass':True,'prediction_generation_authorized':False,
    }
    write(OUT/'RESEARCHER_APPROVAL_RECORD.json',approval)
    write(OUT/'RESEARCHER_APPROVAL_MESSAGE.txt',USER_MESSAGE+'\n')
    state=copy.deepcopy(d);state['reviewer_id']=researcher
    ledger=[];proposed=[];uncertain=[];corrections=[];units=[]
    for c in candidates:
        cid=c['unit_id'];a=state['candidate_annotations'][cid];label=a['support_label']
        row={'candidate_id':cid,'business_id':c['business_id'],'domain':c['domain'],'decision':label,
             'original_candidate_text':c['text'],'approved_corrected_claim_text':a['corrected_claim_text'] if label=='needs_revision' else None,
             'original_evidence':copy.deepcopy(c['evidence']),'approved_by':researcher,'approval_record':'RESEARCHER_APPROVAL_RECORD.json',
             'original_file_human_verified':False,'approval_recorded_at_utc':now,
             'decision_is_human_confirmed_via_conversation':True,'evaluation_eligibility_not_implied_by_approval':True}
        ledger.append(row)
        a.update(human_verified=True,confirmed_by=researcher,confirmed_at=now)
        if label=='uncertain':
            uncertain.append({**row,'excluded_from_scoring':True,'final_unit_ids':[],
                              'reason':'Explicit researcher decision: uncertain; corrected text, if any, is not separately approved for scoring'})
        else:
            text=a['corrected_claim_text'] if label=='needs_revision' else c['text']
            assert text.strip()
            conflict=CONFLICTS.get(cid)
            if label=='needs_revision':
                e=c['evidence'][0]
                corrections.append({**row,'caption':c['text'],'caption_id':e['marker'],'photo_id':e['photo_id'],
                   'image_file':'images/'+e['photo_id']+'.jpg',
                   'caption_scope_assessment':'REQUIRES_RESEARCHER_DECISION' if conflict else 'NO_NEW_CAPTION_FACT_IDENTIFIED',
                   'reason':conflict[0] if conflict else 'Correction removes an unsupported specificity or uses an ordinary caption-entailed generalization.',
                   'alternative_for_researcher_review_only':conflict[1] if conflict else None,
                   'alternative_applied':False,'original_approved_correction_preserved_exactly':True,
                   'photo_assessment_role':'Codex inspection aid, not an independent human annotation',
                   'included_in_final_scoring':False})
            for n,(lo,hi,fragment) in enumerate(span_proposals(text),1):
                uid='D053_'+cid+'_P'+str(n).zfill(2)
                unit={'unit_id':uid,'business_id':c['business_id'],'domain':c['domain'],'text':fragment,
                  'support_status':None if conflict else 'supported','evidence':copy.deepcopy(c['evidence']),
                  'importance':a['importance'],'informativeness':a['informativeness'],
                  'aspect':None,'restaurant_specific':None,'polarity':None,
                  'human_verified':False,'confirmed_by':None,'confirmed_at':None,
                  'atomicity_confirmed':False,'evidence_confirmed':False,'visual_photo_confirmed':False,
                  'parent_candidate_id':cid,'provenance':{'origin':'candidate_split','parent_candidate_id':cid},
                  'construction_status':'DRAFT_NOT_SCORING_ELIGIBLE',
                  'exact_parent_wording':text,'parent_wording_start':lo,'parent_wording_end':hi,
                  'parent_context_required':True,
                  'construction_note':'Exact clause span only. Dependent phrasing/lists and unit-specific evidence attribution require final semantic atomization; no generated standalone facts.',
                  'approval_scope':'Parent candidate decision/wording approved via conversation; derived unit metadata/atomicity are not invented human confirmations',
                  'caption_scope_conflict':bool(conflict),'researcher_decision_required':True}
                assert unit['text']==text[lo:hi]
                units.append(unit);a['final_unit_ids'].append(uid)
                proposed.append(copy.deepcopy(unit))
        state['history'].append({'kind':'candidate_confirmed','entity_id':cid,'confirmed_by':researcher,'confirmed_at':now,
          'snapshot':copy.deepcopy(a),'recording_method':'materialized_explicit_conversation_approval_not_a_UI_click',
          'approval_record':'RESEARCHER_APPROVAL_RECORD.json','review_timestamp_unknown':True})
    state['units']=units
    # A request to check completeness is not a license to conceal new omissions.
    # Keep business/global reference gates pending until conflicts and unit fields resolve.
    state['researcher_confirmed_complete']=False
    write(OUT/'RESEARCHER_APPROVED_DECISIONS.json',{'status':'CANDIDATE_DECISIONS_APPROVED_REFERENCE_ELIGIBILITY_PENDING',
       'counts':dict(Counter(r['decision'] for r in ledger)),'approval_record':'RESEARCHER_APPROVAL_RECORD.json','decisions':ledger})
    write(OUT/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json',state)
    write(OUT/'ATOMIC_SPAN_PROPOSALS_NOT_FINAL.json',{'status':'NOT_FINAL_ATOMIC_REFERENCES',
      'notice':'Clause-span construction proposals preserve approved text; do not score fragments or count them as finalized atomic units.',
      'unit_count':len(proposed),'units':proposed})
    write(OUT/'UNCERTAIN_EXCLUDED_FROM_SCORING.json',uncertain)
    write(OUT/'CORRECTION_CAPTION_PHOTO_AUDIT.json',corrections)
    omission_rows=[]
    for n,(bid,marker,quote,proposal,reason,aspect) in enumerate(OMISSIONS,1):
        case=case_map[bid];r=next(r for r in case['reviews'] if r['marker']==marker);assert quote in r['text']
        omission_rows.append({'issue_id':'OMISSION_'+str(n),'business_id':bid,'marker':marker,'review_id':r['review_id'],
           'quote':quote,'proposed_information_for_decision':proposal,'aspect':aspect,'reason':reason,
           'human_approved':False,'automatically_added_to_units':False,'status':'RESEARCHER_DECISION_REQUIRED',
           'nearby_candidates':[c['unit_id'] for c in candidates if c['business_id']==bid and c['domain']=='review']})
    write(OUT/'POTENTIAL_IMPORTANT_OMISSIONS_FOR_DECISION.json',omission_rows)
    used={(c['business_id'],e['review_id']) for c in candidates if c['domain']=='review' for e in c['evidence']}
    full_inventory=[]
    for case in cases:
        for r in case['reviews']:
            full_inventory.append({'business_id':case['business_id'],'marker':r['marker'],'review_id':r['review_id'],
                'text':r['text'],'candidate_cited_review':(case['business_id'],r['review_id']) in used,
                'uncited_does_not_mean_uncovered':True})
    write(OUT/'ALL_REVIEW_EVIDENCE_INVENTORY.json',full_inventory)
    write(OUT/'PHOTO_INSPECTION_LOG.json',{'scope':'All90 photos inspected in contact sheets; originals remain available and unchanged. One original reopened to check overlapping burger count: four burgers, no count conflict.',
      'not_independent_human_ground_truth':True,'images':photo_check,'inspection_resolution_limitation':'Contact-sheet inspection is not a guarantee of every fine-grained visual detail.',
      'supported_visual_candidates':67,'revised_visual_candidates':21,'uncertain_visual_candidates':2})
    original_validation=validator.validate_annotations(d,cases,candidates,binding['bindings'],final=False)
    draft_validation=validator.validate_annotations(state,cases,candidates,binding['bindings'],final=False)
    final_validation=validator.validate_annotations(state,cases,candidates,binding['bindings'],final=True)
    assert original_validation['valid'] and draft_validation['valid'] and not final_validation['valid']
    write(OUT/'SUBMITTED_DRAFT_VALIDATION.json',original_validation)
    write(OUT/'IMPORTED_DRAFT_VALIDATION.json',draft_validation)
    write(OUT/'FINAL_ACCEPTANCE_BLOCKED.json',final_validation)
    try:
        validator.reference_freeze(state,cases,candidates,binding['bindings'])
    except ValueError as exc:
        write(OUT/'FREEZE_GUARD_EXPECTED_BLOCK.json',{'status':'EXPECTED_BLOCK_PASS','original_guard_not_modified':True,
           'reason_excerpt':str(exc)[:1500],'final_reference_written':False})
    else:raise AssertionError('Incomplete references unexpectedly passed freeze')
    missing=[{'unit_id':u['unit_id'],'parent_candidate_id':u['parent_candidate_id'],'business_id':u['business_id'],
              'importance':u['importance'],'aspect':u['aspect'],'restaurant_specific':u['restaurant_specific'],
              'polarity':u['polarity'],'informativeness_inherited_from_submitted_decision':u['informativeness'],
              'atomicity_confirmed':False,'human_metadata_confirmation':False} for u in units]
    write(OUT/'UNIT_METADATA_AND_ATOMICITY_PENDING.json',missing)
    report={'status':'RESEARCHER_DECISIONS_RECORDED_REFERENCE_FREEZE_BLOCKED',
      'businesses':24,'random_businesses':20,'targeted_businesses':4,'reviews':len(full_inventory),'photos':len(photo_check),
      'candidate_counts':{'supported':203,'needs_revision':21,'uncertain':2},'human_confirmed_candidate_decisions':226,
      'submitted_JSON_original_human_verified_count':0,'candidate_ids_and_labels_preserved':True,
      'exact_candidate_evidence_checks':evidence_count,'nested_original_manifest_files_verified':len(checks['files']),
      'revision_texts_preserved_and_applied_in_proposals':21,'caption_scope_conflicts_for_researcher':len(CONFLICTS),
      'corrections_without_new_caption_fact_identified':len(corrections)-len(CONFLICTS),
      'uncertain_excluded':len(uncertain),'uncertain_reference_units':0,
      'draft_clause_span_proposals':len(units),'final_atomic_reference_units':0,
      'all226_candidate_importance_fields_missing':True,'new_unit_metadata_and_atomicity_not_human_confirmed':True,
      'potential_omissions_for_researcher_decision':len(omission_rows),
      'uncited_review_count':sum(not r['candidate_cited_review'] for r in full_inventory),
      'uncited_is_not_a_semantic_omission_test':True,
      'business_completeness':'Researcher approval of decisions recorded for all24; final reference completeness cannot be asserted while new evidence/omission/unit-metadata issues remain',
      'no_new_clear_conflict_identified_in_67_supported_visual_candidates_on_inspection':True,
      'semantic_coverage_audit':'Concrete nonexhaustive omission findings; full independent semantic exhaustiveness of652 reviews is not claimed',
      'final_reference_freeze_created':False,'frozen_protocol_or_validator_modified':False,
      'teacher_calls':0,'student_predictions':0,'training':False,'held_out_test_access':False}
    write(OUT/'REFERENCE_CONSTRUCTION_READINESS.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
