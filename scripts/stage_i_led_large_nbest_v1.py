"""N-best contracts and conditional source-only ranking; no ML/network imports."""
import copy
import hashlib
import itertools
import json
import re
from pathlib import Path

EXPERIMENT = 'LED_LARGE_NBEST_RERANKING_V1'
CASE_KEYS = {'business_id','split','condition','source','source_sha256','input_ids','reviews','captions'}
POLICY = dict(review_coverage=40, opinion_balance=20, visual_coverage=15,
              unsupported_review=-25, contradicted_claim=-40, false_consensus=-30,
              unsupported_visual=-40, uncertain_claim=-10, redundant_claim=-10,
              generic_review_fraction=-10, minimum_supported_review_facts=2,
              minimum_review_coverage=.40,require_useful_visual_if_inventory_nonempty=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def chunks(source):
    require(source.count('[VISUAL_CONTEXT]')==1,'Ambiguous visual boundary')
    review,visual=source.split('[VISUAL_CONTEXT]')
    def parse(text,pattern):
        matches=list(re.finditer(pattern,text));result={}
        for i,m in enumerate(matches):
            key=m.group(1);require(key not in result,'Duplicate evidence marker')
            result[key]=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
        return result
    return parse(review,r'\[(REV_\d+)\]'),parse(visual,r'\[(IMG_CAP_\d+)\]')


def source_case(row, input_ids):
    reviews,captions=chunks(row['source'])
    result={k:row[k] for k in ('business_id','split','condition','source','source_sha256')}
    result.update(input_ids=list(input_ids),reviews=reviews,captions=captions)
    validate_case(result)
    return result


def validate_case(case):
    require(set(case)==CASE_KEYS,'Forbidden/unknown case keys; target/reference/photo payload prohibited')
    require(case['split'] in ('train','validation'),'Held-out content prohibited')
    require(case['condition'] in ('filtered','unfiltered','reviews_only','shuffled'),'Unknown condition')
    require(hashlib.sha256(case['source'].encode('utf-8')).hexdigest()==case['source_sha256'],'Source hash differs')
    require(chunks(case['source'])==(case['reviews'],case['captions']),'Evidence is not exact model-visible source')
    ids=case['input_ids']
    require(0<len(ids)<=5000 and ids[0]==0 and ids[-1]==2,'Incomplete/oversize frozen input')
    require(all(type(x) is int and 0<=x<50265 for x in ids),'Invalid source token ID')


def selection_packet(case,candidates):
    validate_case(case)
    require(1<=len(candidates)<=4,'At most four candidates')
    require(len({x['candidate_id'] for x in candidates})==len(candidates),'Duplicate candidate ID')
    return dict(business_id=case['business_id'],condition=case['condition'],
                source_sha256=case['source_sha256'],reviews=copy.deepcopy(case['reviews']),
                captions=copy.deepcopy(case['captions']),
                candidates=[dict(candidate_id=x['candidate_id'],text=x['text']) for x in candidates])


def generation_pair(original):
    require(original['num_beams']==4 and original['num_return_sequences']==1
            and original['do_sample'] is False,'Expected original deterministic beam4')
    single=copy.deepcopy(original);nbest=copy.deepcopy(original);nbest['num_return_sequences']=4
    require([k for k in single if single[k]!=nbest[k]]==['num_return_sequences'],'Unexpected decoding change')
    return single,nbest


def unpad(ids):
    # Only remove trailing PAD introduced by multi-sequence finalization, never trim EOS/content.
    ids=list(ids)
    while ids and ids[-1]==1:
        ids.pop()
    require(ids and ids[0]==2 and 1 not in ids,'Interior PAD/wrong decoder start')
    return ids


def compare_top1(single, nbest, historical=None):
    require(single['text']==nbest['text'] and unpad(single['token_ids'])==unpad(nbest['token_ids']),
            'N=1 vs N=4 content discrepancy; stop before quality comparison')
    result=dict(single_vs_nbest_content_exact=True,
                single_vs_nbest_raw_ids_exact=single['token_ids']==nbest['token_ids'],
                allowed_difference='TRAILING_PAD_ONLY',historical_checked=historical is not None)
    if historical is not None:
        require(single['source_sha256']==historical['source_sha256'],'Historical source differs')
        require(single['checkpoint_sha256']==historical['checkpoint_sha256'],'Historical checkpoint differs')
        require(single['text']==historical['text'] and unpad(single['token_ids'])==unpad(historical['token_ids']),
                'Historical top1 mismatch; preserve raw outputs/environment, no comparison or retry')
        result['historical_content_ids_and_text_exact']=True
        result['historical_raw_ids_exact']=single['token_ids']==historical['token_ids']
    return result


def lexical_diversity(candidates):
    sets=[set(re.findall(r'\w+',x['text'].lower())) for x in candidates]
    distances=[1-len(a&b)/len(a|b) if a|b else 0 for a,b in itertools.combinations(sets,2)]
    return dict(exact_unique_content_ids=len({tuple(unpad(x['token_ids'])) for x in candidates}),
                exact_unique_texts=len({x['text'] for x in candidates}),
                mean_pairwise_word_set_distance=sum(distances)/len(distances) if distances else 0,
                semantic_diversity='UNASSESSED_REQUIRES_SOURCE_REVIEW')


def bodies(text):
    a='[REVIEW_CONSENSUS]';b='[VISUAL_EVIDENCE]'
    require(text.count(a)==text.count(b)==1 and text.startswith(a) and text.index(a)<text.index(b),'Malformed section headers')
    j=text.index(b)
    return {'review':(len(a),j),'visual':(j+len(b),len(text))}


def evidence(packet, domain, refs):
    table=packet['reviews' if domain=='review' else 'captions']
    for ref in refs:
        require(set(ref)=={'marker','quote'} and ref['marker'] in table,'Wrong modality/evidence marker')
        require(ref['quote'].strip() and ref['quote'] in table[ref['marker']],'Quote absent from model-visible evidence')


def score_candidate(packet, inventory, candidate, annotation):
    """Conditional proposal: requires audited judgments; quote checks do NOT prove entailment."""
    require(set(inventory)=={'business_id','source_sha256','assessor_kind','facts','contrasts'},'Forbidden inventory fields')
    require(inventory['business_id']==packet['business_id'] and inventory['source_sha256']==packet['source_sha256'],'Foreign inventory')
    require(inventory['assessor_kind'] in ('researcher','model_assisted'),'Assessor provenance required')
    facts={f['fact_id']:f for f in inventory['facts']}
    require(len(facts)==len(inventory['facts']) and facts,'Nonempty unique source-only inventory required')
    require(len({(f['domain'],f['text'].strip().casefold()) for f in facts.values()})==len(facts),'Duplicate fact propositions')
    for f in facts.values():
        require(set(f)=={'fact_id','domain','text','salience','aspect','polarity','evidence'},'Forbidden fact fields')
        require(f['domain'] in ('review','visual') and f['salience'] in ('recurring','distinctive','material_isolated'),'Invalid source fact')
        require(f['evidence'],'Source fact lacks evidence');evidence(packet,f['domain'],f['evidence'])
        if f['domain']=='review' and f['salience']=='recurring':
            require(len({r['marker'] for r in f['evidence']})>=2,'Recurring fact needs independent review markers')
    review={k:f for k,f in facts.items() if f['domain']=='review'}
    visual={k:f for k,f in facts.items() if f['domain']=='visual'}
    require(review,'Review salience inventory cannot be empty')
    require(set(annotation)=={'candidate_id','assessor_kind','atomicity_reviewed','claims'},'Forbidden annotation fields')
    require(annotation['candidate_id']==candidate['candidate_id'] and annotation['atomicity_reviewed'] is True,'Atomic claim review required')
    require(annotation['assessor_kind'] in ('researcher','model_assisted'),'Assessor provenance required')
    text=candidate['text'];spans=bodies(text);covered=set();claim_ids=set();occupied=set()
    errors={k:0 for k in ('unsupported_review','contradicted_claim','false_consensus','unsupported_visual','uncertain_claim','redundant_claim')}
    review_claims=generic=0
    for c in annotation['claims']:
        require(set(c)=={'claim_id','start','end','domain','verdict','evidence','covered_fact_ids','false_consensus','generic','redundant_with','rationale'},'Forbidden claim fields')
        require(c['domain'] in spans and c['verdict'] in ('supported','unsupported','contradicted','uncertain'),'Invalid judgment')
        lo,hi=spans[c['domain']];start,end=c['start'],c['end']
        require(type(start) is int and type(end) is int and lo<=start<end<=hi,'Claim span outside body')
        require(not occupied.intersection(range(start,end)),'Overlapping claims must be atomized')
        occupied.update(range(start,end))
        require(c['claim_id'] not in claim_ids,'Duplicate claim ID')
        evidence(packet,c['domain'],c['evidence'])
        require(type(c['generic']) is bool and type(c['false_consensus']) is bool,'Explicit boolean judgments required')
        require(c['rationale'].strip(),'Verification rationale required')
        if c['verdict'] in ('supported','contradicted'):
            require(c['evidence'],'Supported/contradicted judgment needs exact source quote')
        if c['false_consensus']:
            require(c['verdict']!='supported','False consensus cannot be marked supported')
            errors['false_consensus']+=1
        ids=set(c['covered_fact_ids'])
        require(ids<=set(facts) and all(facts[f]['domain']==c['domain'] for f in ids),'Foreign/wrong-modality fact coverage')
        require(not c['generic'] or not ids,'Generic prose cannot claim distinctive fact coverage')
        if c['verdict']=='supported' and not c['false_consensus']:
            covered.update(ids)
        else:
            require(not ids,'Unverified claim cannot claim supported fact coverage')
        if c['verdict']=='contradicted':errors['contradicted_claim']+=1
        if c['verdict']=='uncertain':errors['uncertain_claim']+=1
        if c['domain']=='visual' and (not packet['captions'] or c['verdict']!='supported'):
            errors['unsupported_visual']+=1
        if c['domain']=='review':
            review_claims+=1;generic+=int(c['generic'])
            if c['verdict']=='unsupported':errors['unsupported_review']+=1
        if c['redundant_with'] is not None:
            require(c['redundant_with'] in claim_ids,'Redundancy must reference an earlier claim')
            errors['redundant_claim']+=1
        claim_ids.add(c['claim_id'])
    needed={i for lo,hi in spans.values() for i in range(lo,hi) if text[i].isalnum()}
    require(needed<=occupied,'Unannotated candidate content; all atomic claims must be reviewed')
    rv_covered=covered&set(review);vv_covered=covered&set(visual)
    weight=lambda f:2 if f['salience'] in ('recurring','distinctive') else 1
    R=sum(weight(review[k]) for k in rv_covered)/sum(map(weight,review.values()))
    V=len(vv_covered)/len(visual) if visual else 0
    pairs=inventory['contrasts']
    for pair in pairs:
        require(len(pair)==2 and set(pair)<=set(review),'Invalid same-aspect contrast')
        a,b=(review[k] for k in pair)
        require(a['aspect']==b['aspect'] and {a['polarity'],b['polarity']}=={'positive','negative'},'Contrast must preserve aspect and polarity')
    B=sum(set(pair)<=covered for pair in pairs)/len(pairs) if pairs else 0
    empty_visual=not text[spans['visual'][0]:spans['visual'][1]].strip()
    reasons=[]
    if not text[spans['review'][0]:spans['review'][1]].strip():reasons.append('Empty review')
    if not packet['captions'] and not empty_visual:reasons.append('Zero-caption visual assertion')
    if len(rv_covered)<min(POLICY['minimum_supported_review_facts'],len(review)):reasons.append('Too few supported salient review facts; generic/cautious candidate')
    if R<POLICY['minimum_review_coverage']:reasons.append('Review undercoverage below fixed 40% source-salience floor')
    if visual and not vv_covered:reasons.append('No useful supported visual fact despite available source inventory')
    # Penalties are counts, not error rates; adding generic supported text cannot dilute them.
    features=dict(review_coverage=R,opinion_balance=B,visual_coverage=V,
                  generic_review_fraction=generic/review_claims if review_claims else 1,**errors)
    score=sum(POLICY[k]*v for k,v in features.items())
    return dict(candidate_id=candidate['candidate_id'],score=score,eligible=not reasons,
                exclusion_reasons=reasons,features=features,covered_review_fact_ids=sorted(rv_covered),
                status='PROVISIONAL_SOURCE_JUDGMENTS_NOT_GROUND_TRUTH')


def rerank(case,candidates,inventory,annotations):
    packet=selection_packet(case,candidates)
    require(len(annotations)==len(candidates) and {a['candidate_id'] for a in annotations}=={c['candidate_id'] for c in candidates},'Complete candidate annotations required')
    byid={a['candidate_id']:a for a in annotations}
    scores=[score_candidate(packet,inventory,c,byid[c['candidate_id']]) for c in candidates]
    eligible=[(s,i) for i,s in enumerate(scores) if s['eligible']]
    if not eligible:
        return dict(status='ALL_REJECTED_NO_REWRITE_NO_SILENT_FALLBACK',selected_candidate_id=None,scores=scores)
    s,i=max(eligible,key=lambda x:(x[0]['score'],-x[1]))
    return dict(status='PROVISIONAL_SELECTION',selected_candidate_id=s['candidate_id'],
                selected_text=candidates[i]['text'],scores=scores,rewritten=False)


def diversity_gate(assessment, expected_ids):
    require(assessment['split']=='train' and assessment['assessor_kind']=='researcher'
            and assessment['source_only_review_confirmed'] is True,'Researcher/source-only TRAIN diversity review required')
    rows=assessment['cases'];require([r['business_id'] for r in rows]==expected_ids and len(rows)==12,'Wrong pilot set/order')
    for r in rows:
        require(type(r['materially_distinct_profiles']) is int and r['materially_distinct_profiles']>=1,'Missing semantic profiles')
        require(type(r['oracle_material_gain_without_grounding_regression']) is bool and r['rationale'].strip(),'Oracle explanation required')
    distinct=sum(r['materially_distinct_profiles']>=2 for r in rows)
    gains=sum(r['oracle_material_gain_without_grounding_regression'] for r in rows)
    return dict(status='PASS_TO_CONDITIONAL_RERANKER_PREPARATION' if distinct>=6 and gains>=4 else 'STOP_INSUFFICIENT_CANDIDATE_HEADROOM',
                material_diversity_businesses=distinct,oracle_gain_businesses=gains,n=12,
                population_quality_GO=False)


def verification_audit(audit,train_ids):
    """Compare provisional verifier decisions with independent source-only TRAIN adjudication."""
    require(audit['split']=='train' and audit['assessor_kind']=='researcher'
            and audit['independent_of_provisional_verifier'] is True,'Independent TRAIN verification required')
    rows=audit['claims'];require(1<=len(rows)<=96,'Audit cap: 96 atomic claims')
    tp=fp=fn=tn=unknown=0;seen=set()
    for r in rows:
        require(r['business_id'] in train_ids and r['rationale'].strip(),'Foreign/unexplained verification row')
        key=(r['business_id'],r['claim_key']);require(key not in seen,'Duplicate audit item');seen.add(key)
        require(r['gold'] in ('supported','unsupported','contradicted','uncertain')
                and r['predicted'] in ('supported','unsupported','contradicted','uncertain'),'Unknown verification category')
        if 'uncertain' in (r['gold'],r['predicted']):unknown+=1;continue
        # Positive means a grounding error detected; precision controls rejection of valid content.
        truth=r['gold']!='supported';pred=r['predicted']!='supported'
        tp+=int(truth and pred);fp+=int(not truth and pred)
        fn+=int(truth and not pred);tn+=int(not truth and not pred)
    positives=tp+fn;negatives=tn+fp
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/positives if positives else 0
    false_rejection=fp/negatives if negatives else 1
    sufficient=positives>=20 and negatives>=20 and unknown/len(rows)<=.20
    passed=sufficient and precision>=.90 and recall>=.80 and false_rejection<=.10
    return dict(status='PASS_TRAIN_ONLY' if passed else 'STOP_VERIFIER_INSUFFICIENT_OR_INACCURATE',
                TP=tp,FP=fp,FN=fn,TN=tn,uncertain=unknown,grounding_error_precision=precision,
                grounding_error_recall=recall,supported_claim_false_rejection_rate=false_rejection,
                sufficient_class_counts=sufficient,population_quality_GO=False)
