"""Prepare curated D053 atomic/reference decisions, with no human confirmation.

Recipe rows are explicit semantic proposals, not regex fragments. Keyword lookup
only retrieves candidate evidence; it is neither a factual verifier nor a newly
introduced experimental scoring method. Frozen annotations/data stay untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1'
SOURCE=PREVIOUS/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
OUT=ROOT/'outputs/stage_i/d053_reference_proposals_v2'
EXCLUDE_CONFLICTS={
 'noUma64stIOSYAQ1GIIFjw_IMG_CAP_3_CANDIDATE': 'Only a generic glass/drink remains caption-grounded; the more informative jar/cocktail/garnish details would be photo-only.',
 'mP8gKzr8NKvbTtm9XePvOQ_IMG_CAP_4_CANDIDATE': 'Only generic food remains; fried/sauced/container details cannot be recovered from the supplied caption.',
 'haPQJ2oy3529RMzpWrA39A_IMG_CAP_2_CANDIDATE': 'The safe overlap is a generic tray of food on a table; barbecue identity and side dishes are not caption-grounded.',
 'thlyktt2c7v1DyOpSVSibw_IMG_CAP_3_CANDIDATE': 'The safe overlap is only a generic restaurant bar; stools/lights cannot be imported from the photo.'}

# Explicitly remove duplicated/summary pieces without relabeling their parents.
# Parent IDs still each have at least one own, substantive proposal, preserving
# compatibility with the existing one-primary-parent provenance constraint.
CONSOLIDATIONS=[
 {'business_id':'JcdOQvvx3BqZ5Ga2_9n_Bg','parents':[1], 'removed':'steaks repeated after steak and potatoes',
  'reason':'Steak praise is represented once; potatoes are a separate food opinion.'},
 {'business_id':'gpSv4OvISIeyBAqcNN5bqw','parents':[1,2], 'removed':'A second generic positive-pizza unit from candidate2',
  'reason':'Candidate1 supplies pizza praise; candidate2 supplies the distinct negative pizza attributes.'},
 {'business_id':'Zz1utCut0psbC4brBjvSEg','parents':[3], 'removed':'attentive / appropriately attentive counted twice',
  'reason':'One attentiveness opinion, not two near-synonymous reference facts.'},
 {'business_id':'OpnIXsrqaK-e8Wz5gco7Vw','parents':[3,4,6], 'removed':'Repeated positive fish-taco opinion and generic repeated poor-value phrase',
  'reason':'Candidate3 supplies praise; candidate4 supplies negative flavor; candidate6 retains the specific $18 value criticism.'},
 {'business_id':'Ktg3ahIxk0JIkJwXAqu2ew','parents':[1,5], 'removed':'A duplicate generic value-concern unit under candidate1',
  'reason':'Candidate5 retains positive and negative pricing opinions; candidate1 retains fish-quality concerns.'},
 {'business_id':'ALL','parents':[], 'removed':'Food opinions are mixed / service experiences vary / list introductions as standalone units',
  'reason':'These are context or summaries of the actual positive/negative propositions, not additional denominator items.'},
 {'business_id':'ALL','parents':[], 'removed':'Synonymous praise such as friendly/kind/polite or bland/tasteless as separate repeated opinions',
  'reason':'One opinion dimension is represented once. Distinct cooking, texture, flavor, price and sentiment dimensions remain separate.'},
 {'business_id':'haPQJ2oy3529RMzpWrA39A','parents':[1,2],'removed':'A generic meat-saltiness unit repeated beside the more specific brisket-saltiness complaint',
  'reason':'Keep the specific supported saltiness proposition once; broad parent approval and exact sources remain archived.'},
 {'business_id':'gcQSAQIx-KQOdf5E8HF6Lw','parents':[2],'removed':'Generic incorrect-order summary beside its missing-ingredient and missing-donut details',
  'reason':'The cited examples are missing-item errors; the summary adds no independent detail.'},
]


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):
    s=''.join(c for c in unicodedata.normalize('NFKD',s.lower()) if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+',' ',s).strip()
def write_new(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def recipes():
    result=[];seen_businesses=[]
    for path in sorted((ROOT/'scripts').glob('d053_atomic_recipes_v2*.txt')):
        bid=None
        for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            if not line or line.startswith('#'):continue
            if line.startswith('@'):
                bid=line[1:];seen_businesses.append(bid);continue
            fields=line.split('|');assert len(fields)==7,(path,number,line)
            key,polarity,importance,aspect,specific,text,anchors=fields
            cid=bid+('_IMG_CAP_'+key[1:]+'_CANDIDATE' if key.startswith('V') else '_REVIEW_CANDIDATE_'+key)
            # Remove dangling discourse references; source IDs retain the actual
            # within-review connection rather than asserting a new reviewer.
            text=text.replace('Other selected reviewers','Some selected reviewers').replace('Other selected reviews','Some selected reviews')
            text=text.replace('Another selected reviewer','A selected reviewer').replace('That selected reviewer','A selected reviewer')
            text=text.replace('That wine-tasting reviewer','A selected wine-tasting reviewer')
            text=text.replace('That duplicate-charge reviewer','A selected reviewer reporting a duplicate charge')
            text=text.replace('That mobile-order reviewer','A selected reviewer reporting a mobile-order problem')
            text=text.replace('That reviewer','A selected reviewer')
            text=text.replace('Those price-critical reviews','Price-critical selected reviews')
            result.append({'candidate_id':cid,'business_id':bid,'domain':'visual' if key.startswith('V') else 'review',
              'text':text,'proposed_metadata':{'importance':importance,
              'informativeness':'useful_specific' if specific=='1' else 'low_information' if specific=='L' else 'useful_general',
              'aspect':aspect,'restaurant_specific':specific=='1','polarity':None if key.startswith('V') else polarity},
              'evidence_retrieval_anchors':anchors.split('~'),'recipe_file':path.name,'recipe_line':number})
    assert len(seen_businesses)==len(set(seen_businesses))==24
    return result


def evidence_for(recipe,candidate,case):
    if recipe['domain']=='visual':return copy.deepcopy(candidate['evidence']),'EXACT_ORIGINAL_CAPTION',[]
    anchors=[norm(a) for a in recipe['evidence_retrieval_anchors']]
    parent_review_ids={e['review_id'] for e in candidate['evidence']}
    matches=[{'marker':r['marker'],'review_id':r['review_id'],'quote':r['text']} for r in case['reviews']
             if r['review_id'] in parent_review_ids and any(a in norm(r['text']) for a in anchors)]
    if matches:return select_quotes(recipe,matches),'PARENT_REVIEW_CONTEXT_PROPOSED',[]
    # Transparent additional retrieval within the SAME selected reviews, never
    # external/train/test material. Preserve an exact full quote around the hit.
    matches=[]
    for r in case['reviews']:
        if any(a in norm(r['text']) for a in anchors):
            matches.append({'marker':r['marker'],'review_id':r['review_id'],'quote':r['text']})
    if matches:return select_quotes(recipe,matches),'SAME_SELECTED_REVIEW_POOL_RETRIEVAL_PROPOSED',[]
    return copy.deepcopy(candidate['evidence']),'RETRIEVAL_GAP_REQUIRES_CURATION',[recipe['candidate_id']+' / '+recipe['text']]


def select_quotes(recipe,matches):
    """Propose a compact evidence display, not a semantic support judgment.

    Whole original quotes remain verbatim. Word overlap/sentiment cues prioritize
    which citations to inspect; no classifier/threshold is added to evaluation.
    Full parent evidence and full selected reviews remain independently available.
    """
    words=set(norm(recipe['text']).split())-set('a an the selected some other reviewers reviews reviewer describe describes report reports find finds praise praises one that as is are in on of for and with to it'.split())
    positive=('great','good','delicious','excellent','amazing','friendly','attentive','fresh','tasty','helpful','perfect','love','enjoy','reasonable')
    negative=('bad','poor','slow','rude','bland','cold','disappoint','overpriced','over priced','missing','wrong','not','salty','hard','wait','dry','greasy')
    polarity=recipe['proposed_metadata']['polarity']
    def rank(e):
        t=norm(e['quote']);tokens=set(t.split());score=4*len(words&tokens)
        pos=sum(x in t for x in positive);neg=sum(x in t for x in negative)
        if polarity=='positive':score+=pos-neg
        if polarity=='negative':score+=neg-pos
        return -score,len(e['quote']),e['marker'],e['quote']
    ordered=sorted(matches,key=rank)
    minimum=2 if re.search(r'\b(Two|Multiple) selected review',recipe['text']) else 1
    chosen=[];reviews=set()
    for e in ordered:
        if e['review_id'] in reviews:continue
        chosen.append(copy.deepcopy(e));reviews.add(e['review_id'])
        if len(chosen)>=minimum:break
    return chosen


def prepare(audit_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    cases=read(SOURCE/'source_cases.json');bycase={c['business_id']:c for c in cases}
    candidates=[json.loads(s) for s in (SOURCE/'candidates_UNVERIFIED.jsonl').read_text().splitlines()]
    byid={c['unit_id']:c for c in candidates}
    approved=read(PREVIOUS/'RESEARCHER_APPROVED_DECISIONS.json')
    decisions={r['candidate_id']:r for r in approved['decisions']}
    old=read(PREVIOUS/'ATOMIC_SPAN_PROPOSALS_NOT_FINAL.json')['units']
    spans=defaultdict(list)
    for u in old:spans[u['parent_candidate_id']].append(u['unit_id'])
    proposals=[];perparent=Counter();gaps=[]
    for r in recipes():
        cid=r['candidate_id'];c=byid[cid];assert c['domain']==r['domain'] and c['business_id']==r['business_id']
        assert decisions[cid]['decision']!='uncertain'
        ev,mode,missing=evidence_for(r,c,bycase[r['business_id']]);gaps.extend(missing)
        perparent[cid]+=1;pid='D053_R2_'+cid+'_U'+str(perparent[cid]).zfill(2)
        caution=(r['candidate_id']=='7u40cyMxw4iGosMdkesoEQ_REVIEW_CANDIDATE_4' and r['proposed_metadata']['aspect']=='value.pricing_clarity')
        proposals.append({'proposal_id':pid,'business_id':r['business_id'],'domain':r['domain'],
          'origin':'candidate_reformulation','parent_candidate_id':cid,'source_span_ids':spans[cid],
          'span_mapping_scope':'Parent lineage, not a claim that each old fragment is independently retained',
          'text':r['text'],'proposed_metadata':r['proposed_metadata'],'evidence':ev,
          'evidence_selection_status':mode,'evidence_retrieval_anchors':r['evidence_retrieval_anchors'],
          'recipe_source':{'file':r['recipe_file'],'line':r['recipe_line']},
          'parent_original_text':c['text'],'parent_researcher_approved_correction':decisions[cid]['approved_corrected_claim_text'],
          'recommendation':'exclude' if cid in EXCLUDE_CONFLICTS or caution else 'include',
          'recommendation_reason':('The literal REV_6 text says they "did tell me how much". Do not silently correct a possible typo to "did not"; recommend excluding only this ambiguous pricing-clarity unit while preserving the approved parent decision and its separate bun/chicken value unit.' if caution else EXCLUDE_CONFLICTS.get(cid,'Retains a distinct parent-supported proposition; metadata is proposed for researcher review.')),
          'evidence_caution':caution,
          'requires_caption_resolution':cid in {a['candidate_id'] for a in read(PREVIOUS/'CORRECTION_CAPTION_PHOTO_AUDIT.json') if a['caption_scope_assessment']=='REQUIRES_RESEARCHER_DECISION'},
          'human_verified':False,'confirmed_by':None,'confirmed_at':None,'researcher_decision':None,
          'atomicity_assessment':'Codex proposes one independently assessable opinion, attribute or relation; not human-confirmed',
          'scoring_eligible':False})
    assert set(perparent)=={c['unit_id'] for c in candidates if decisions[c['unit_id']]['decision']!='uncertain'}
    # Curated omissions retain the earlier exact quotes; none becomes an approved
    # addition merely because a proposal is fully formulated.
    omissions=read(PREVIOUS/'POTENTIAL_IMPORTANT_OMISSIONS_FOR_DECISION.json')
    specs={
      'OMISSION_1':('core','Contrasts with the positive ambience candidate: inability to converse is a material dining limitation.',[
        ('A selected reviewer reports music so loud that conversation was nearly impossible.','atmosphere.noise','negative')]),
      'OMISSION_2':('core','Adds an important negative service/value experience absent from the favorable service candidate. It remains an attributed allegation, not an established restaurant policy.',[
        ('A selected reviewer reports refusal to honor an advertised happy-hour offer.','value.advertised_offer','negative'),
        ('The same selected reviewer describes the manager as rude during that dispute.','service.manager_friendliness','negative')]),
      'OMISSION_3':('core','Adds the missing negative pickup/order-reliability contrast to favorable service descriptions; one reviewer reports repeated experiences.',[
        ('A selected reviewer reporting repeated pickup problems describes orders not being ready.','service.pickup_readiness','negative'),
        ('A selected reviewer reporting repeated pickup problems describes incorrect orders.','service.pickup_accuracy','negative')]),
      'OMISSION_4':('supporting','A concrete product-availability detail worth retaining, but less central than the recurring waiting/staffing experience. Do not turn one reviewer into a business-wide prevalence claim.',[
        ('A selected reviewer reports repeated cold-brew unavailability, including early morning.','menu.cold_brew_availability','negative')]),
      'OMISSION_5':('excluded','Peripheral one-review comment on a non-signature dish. Broad portion/value concerns already exist; omitting this detail is an importance recommendation, not a claim that the review is false. Optional supporting inclusion remains available.',[
        ('A selected reviewer finds the spicy pork bowl insufficiently spicy.','food.pork_bowl_spiciness','negative')])}
    omission_review=[]
    for o in omissions:
        level,reason,parts=specs[o['issue_id']];ids=[]
        for i,(text,aspect,polarity) in enumerate(parts,1):
            text=text.replace('The same selected reviewer','A selected reviewer')
            pid='D053_R2_'+o['issue_id']+'_U'+str(i).zfill(2);ids.append(pid)
            proposals.append({'proposal_id':pid,'business_id':o['business_id'],'domain':'review',
              'origin':'source_omission_proposal','parent_candidate_id':None,'omission_id':o['issue_id'],'source_span_ids':[],
              'text':text,'proposed_metadata':{'importance':'core' if level=='core' else 'supporting',
                'informativeness':'useful_specific','aspect':aspect,'restaurant_specific':True,'polarity':polarity},
              'evidence':[{'marker':o['marker'],'review_id':o['review_id'],'quote':o['quote']}],
              'evidence_selection_status':'EXACT_PREVIOUS_OMISSION_QUOTE_PROPOSED',
              'recommendation':'exclude' if level=='excluded' else 'include','recommendation_reason':reason,
              'human_verified':False,'confirmed_by':None,'confirmed_at':None,'researcher_decision':None,'scoring_eligible':False})
        omission_review.append({**o,'recommended_reference_role':level,'recommendation_reason':reason,
          'proposed_unit_ids':ids,'human_verified':False,'researcher_decision':None,
          'approved_candidate_decisions_unchanged':True})
    curation=read(ROOT/'scripts/d053_evidence_curation_v2.json')['overrides']
    retired=[]
    for p in proposals:
        change=curation.get(p['proposal_id'],{})
        if change.get('retire_reason'):
            retired.append({'proposal_id':p['proposal_id'],'parent_candidate_id':p['parent_candidate_id'],
                            'text':p['text'],'reason':change['retire_reason']})
        if change.get('text'):p['text']=change['text']
        if change.get('review_markers'):
            pool={r['marker']:r for r in bycase[p['business_id']]['reviews']}
            p['evidence']=[{'marker':m,'review_id':pool[m]['review_id'],'quote':pool[m]['text']} for m in change['review_markers']]
            p['evidence_selection_status']='MANUALLY_CURATED_SOURCE_CONTEXT_PROPOSED'
        # The complete parent citation context is retained alongside the proposed
        # compact bindings. Neither lexical retrieval nor this curation is human
        # annotation or an independent factual-ground-truth determination.
        if p.get('parent_candidate_id'):
            p['parent_evidence_context']=copy.deepcopy(byid[p['parent_candidate_id']]['evidence'])
    proposals=[p for p in proposals if p['proposal_id'] not in {x['proposal_id'] for x in retired}]
    # Bind attribution to the evidence actually displayed, rather than turning
    # a parent-level list into an unsupported prevalence assertion per item.
    verbs={'describe':'describes','praise':'praises','report':'reports','find':'finds',
           'consider':'considers','appreciate':'appreciates','enjoy':'enjoys','favor':'favors',
           'recommend':'recommends','criticize':'criticizes','object':'objects',
           'rate':'rates','express':'expresses','do':'does','encounter':'encounters',
           'especially':'especially','still':'still','do not rank':'does not rank'}
    for p in proposals:
        if p['domain']!='review' or len({e['review_id'] for e in p['evidence']})!=1:continue
        match=re.match(r'(?:Some |Price-critical )?[Ss]elected (reviewers|reviews) (.+)',p['text'])
        if not match:continue
        noun,rest=match.groups()
        for before,after in sorted(verbs.items(),key=lambda x:-len(x[0])):
            if rest.startswith(before+' '):
                rest=after+rest[len(before):];break
        # Adverbs before verbs need their own agreement.
        rest=rest.replace('especially praise ','especially praises ').replace('still praise ','still praises ')
        p['text']='A selected '+('reviewer' if noun=='reviewers' else 'review')+' '+rest
    # Exact duplicate proposals in the SAME photo/evidence scope would inflate
    # reference counts. Do not deduplicate separate photographs or sentiments.
    signatures=set()
    for p in proposals:
        signature=(p['business_id'],p['domain'],norm(p['text']),p['proposed_metadata']['polarity'],
                   tuple(sorted({e.get('photo_id','') for e in p['evidence']})) if p['domain']=='visual' else '')
        assert signature not in signatures,('Duplicate curated proposition',p['proposal_id']);signatures.add(signature)
        c=bycase[p['business_id']];dom=p['domain'];mapping={x['marker' if dom=='review' else 'caption_id']:x for x in c['reviews' if dom=='review' else 'captions']}
        for e in p['evidence']:
            x=mapping[e['marker']];field='review_id' if dom=='review' else 'photo_id'
            assert e[field]==x[field] and e['quote'] in x['text']
    audit={'proposals':len(proposals),'candidate_proposals':sum(p['origin']=='candidate_reformulation' for p in proposals),
      'omission_proposals':sum(p['origin']=='source_omission_proposal' for p in proposals),
      'review_proposals':sum(p['domain']=='review' for p in proposals),'visual_proposals':sum(p['domain']=='visual' for p in proposals),
      'metadata_importance':dict(Counter(p['proposed_metadata']['importance'] for p in proposals)),
      'retrieval_status':dict(Counter(p['evidence_selection_status'] for p in proposals)),
      'retrieval_gaps':gaps,'retired_redundant_proposals':retired,'human_confirmed_new_proposals':0,'final_reference_freeze_created':False}
    (OUT/'WORKING_EVIDENCE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'WORKING_PROPOSALS.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if audit_only:
        print(json.dumps(audit,ensure_ascii=False,indent=2));return
    assert not gaps,'Resolve evidence retrieval gaps before publication'
    write_new(OUT/'PROPOSED_ATOMIC_REFERENCE_UNITS.json',{'status':'RESEARCHER_APPROVAL_PENDING_NOT_SCORING_READY','units':proposals})
    write_new(OUT/'OMISSION_RECOMMENDATIONS.json',omission_review)
    write_new(OUT/'CONSOLIDATION_AND_ATOMICITY_NOTES.json',{'notes':CONSOLIDATIONS,'retired_during_context_curation':retired,
      'original_clause_spans':321,'new_proposals':len(proposals),'one_to_one_span_conversion':False,
      'atomicity_policy':'One attributed opinion dimension, concrete attribute or explicit relation; synonymous opinions are consolidated. Independent dish identities, toppings, sentiment directions and costs/waits are separated. Relations may have multiple object arguments.',
      'new_metadata_policy':'All values are Codex recommendations, not researcher labels. Core: material food/service/value/safety or defining setting. Supporting: secondary dishes/amenities/visual details. Specificity means a distinguishing source detail, not global uniqueness.',
      'review_attribution_policy':'Neutral selected-review attribution avoids transferring aggregate parent prevalence to every list item. Single-review/count-specific wording is retained where the source claim requires it; full parent wording remains in provenance.',
      'no_semantic_verification_claim_from_keyword_lookup':True})
    perparent_ids=defaultdict(list)
    for p in proposals:
        if p.get('parent_candidate_id'):perparent_ids[p['parent_candidate_id']].append(p['proposal_id'])
    write_new(OUT/'ORIGINAL_321_SPAN_LINEAGE.json',[{'old_span_id':u['unit_id'],'parent_candidate_id':u['parent_candidate_id'],
      'new_parent_proposal_ids':perparent_ids[u['parent_candidate_id']],
      'disposition':'Reformulated or consolidated at parent-meaning level; old fragment is not a separate scoring unit',
      'mapping_scope':'Parent-level lineage only, not exact one-to-one semantic alignment'} for u in old])
    conflicts=[]
    for oldc in read(PREVIOUS/'CORRECTION_CAPTION_PHOTO_AUDIT.json'):
        if oldc['caption_scope_assessment']!='REQUIRES_RESEARCHER_DECISION':continue
        cid=oldc['candidate_id'];children=[p for p in proposals if p.get('parent_candidate_id')==cid]
        conflicts.append({**oldc,'proposed_caption_safe_wording':' '.join(p['text'] for p in children),
          'recommended_action':'exclude' if cid in EXCLUDE_CONFLICTS else 'include_caption_safe_revision',
          'recommendation_reason':EXCLUDE_CONFLICTS.get(cid,'Keep the ordinary facts common to the caption and image; remove unsupported identities, attributes or posture.'),
          'proposed_unit_ids':[p['proposal_id'] for p in children],
          'human_verified':False,'researcher_decision':None,'old_corrected_wording_not_overwritten':True})
    write_new(OUT/'CAPTION_SCOPE_RECOMMENDATIONS.json',conflicts)
    inputs={
      'previous_published_zip':ROOT/'outputs/stage_i/stage_i_d053_researcher_reference_resolution_bundle_v1.zip',
      'approved_candidate_ledger':PREVIOUS/'RESEARCHER_APPROVED_DECISIONS.json',
      'original_submitted_JSON':PREVIOUS/'incoming/D053_AI_DRAFT_24_RESTAURANTS_EDITED.json',
      'previous_approval_record':PREVIOUS/'RESEARCHER_APPROVAL_RECORD.json',
      'previous_imported_annotation_state':PREVIOUS/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json',
      'source_cases':SOURCE/'source_cases.json','original_candidates':SOURCE/'candidates_UNVERIFIED.jsonl',
      'frozen_evaluation_config':SOURCE/'FROZEN_EVALUATION_CONFIG.json','frozen_protocol':SOURCE/'D049_FROZEN_EVALUATION_PROTOCOL.md'}
    write_new(OUT/'INPUT_PRESERVATION.json',{'files':[{'name':k,'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for k,p in inputs.items()],
       'human_approved_candidate_decisions':226,'existing_decisions_changed':0,'frozen_datasets_changed':0})
    # Reconcile the previously delivered archive, not its differently hashed
    # release candidate. Never rewrite either historical archive or receipt.
    published=inputs['previous_published_zip'];receipt=read(PREVIOUS/'REVIEW_BUNDLE_EXPORT_RECEIPT.json')
    sidecar=published.with_suffix('.zip.sha256').read_text(encoding='ascii').split()[0]
    tested=read(PREVIOUS/'PUBLISHED_ZIP_TESTS/COMPLETE_D053_VALIDATION_SUMMARY.json')
    actual=sha(published);assert actual==receipt['sha256']==sidecar==tested['published_final_zip_sha256']
    rc=PREVIOUS/'release_candidate/stage_i_d053_researcher_reference_resolution_bundle_v1.zip'
    write_new(OUT/'ZIP_RECEIPT_RECONCILIATION.json',{
      'status':'PUBLISHED_ARCHIVE_RECEIPT_SIDECAR_AND_TEST_RECORD_MATCH',
      'archive_name':published.name,'actual_sha256':actual,'actual_bytes':published.stat().st_size,
      'canonical_published_receipt_sha256':receipt['sha256'],'published_sidecar_sha256':sidecar,
      'published_extracted_test_sha256':tested['published_final_zip_sha256'],
      'release_candidate_sha256':sha(rc),'release_candidate_scope':'Earlier archive before test logs were embedded; NOT the delivered published archive',
      'release_candidate_receipts_retained_as_history':True,'previous_delivered_archive_rewritten':False,
      'inside_zip_self_hash_policy':'Only file hashes are authoritative inside a ZIP; archive byte hash is in an external receipt/sidecar to avoid self-reference.'})
    # Proposal identity binds the actual new content; a later user export must
    # carry these hashes. It is unrelated to the archive's byte hash.
    file_hashes={p.name:sha(p) for p in [OUT/'PROPOSED_ATOMIC_REFERENCE_UNITS.json',OUT/'CAPTION_SCOPE_RECOMMENDATIONS.json',OUT/'OMISSION_RECOMMENDATIONS.json']}
    pid=hashlib.sha256(json.dumps(file_hashes,sort_keys=True).encode()).hexdigest()
    write_new(OUT/'PROPOSAL_VERSION.json',{'proposal_id':pid,'files':file_hashes,'status':'PREPARED_NOT_HUMAN_CONFIRMED'})
    template={'schema_version':'D053_RESEARCHER_RESOLUTION_DECISIONS_v2','proposal_id':pid,'proposal_files':file_hashes,
      'reviewer_id':'','corrections':{r['candidate_id']:{'choice':None,'notes':'','confirmed_at':None} for r in conflicts},
      'omissions':{r['issue_id']:{'choice':None,'notes':'','confirmed_at':None} for r in omission_review},
      'business_packages':{c['business_id']:{'choice':None,'notes':'','confirmed_at':None} for c in cases},
      'unit_overrides':{},'history':[],'researcher_attestation':False,
      'final_reference_freeze':False,'new_human_confirmed_unit_count':0}
    write_new(OUT/'RESEARCHER_DECISIONS_TEMPLATE.json',template)
    write_new(OUT/'PROPOSAL_PREPARATION_COUNTS.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    prepare(args.audit_only)
