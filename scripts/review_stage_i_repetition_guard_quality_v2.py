"""Save retrospective, source-linked provisional AI observations; no model calls.

The observations are written by Codex after reading these saved outputs. This is
not an automatic factual-label generator, target editor, or researcher approval.
"""
import json
import re
from datetime import datetime,timezone
from pathlib import Path
from stage_i_repetition_pilot_v1 import ROOT,read,rows,write,sha


def review():
    run=ROOT/'outputs/stage_i/repetition_guard_pilot_v2'
    if not (run/'COMPLETION.json').exists():raise ValueError('Wait for terminal completion')
    cases=rows(ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl')
    sources=read(ROOT/'outputs/stage_i/d053_reference_verification_v1/source_cases.json')
    refs=read(ROOT/'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json')
    oldb={p['business_id']:p for p in rows(ROOT/'outputs/stage_i/decoding_beam_ngram_B_local_v1/filtered/predictions.jsonl')}
    findings=[]
    inspected={'uue0fLINmQeX3GBt0RH2YQ','XZWc_p8tvZo4_P6S4c-aeA','QIP3gjb6TbFLyjUNawVUCQ',
               'owGbndFzUQJlziBDELVzAg','DcTKcRZxcGQWpERp7U6w8w','bBXHLwJ1EctFvwll7Uucng','Kh8HJSPXJ5nBDGQYYluxew'}
    def add(index,claim,classification,explanation,cap_ids=(),review_markers=(),ref_fragments=()):
        case=cases[index-1];source=sources[index-1];bid=case['business_id']
        p=run/'raw_cases'/(bid+'.json')
        if not p.exists():return
        pred=read(p)
        if claim not in pred['text']:raise ValueError('AI finding quote is not exact: '+bid+' '+claim)
        caps=[{**c,'original_photo_AI_inspected_this_review':c['photo_id'] in inspected}
              for c in source['captions'] if c['caption_id'] in cap_ids]
        if set(c['caption_id'] for c in caps)!=set(cap_ids):raise ValueError('Unknown caption ID')
        revs=[r for r in source['reviews'] if r['marker'] in review_markers]
        if set(r['marker'] for r in revs)!=set(review_markers):raise ValueError('Unknown review ID')
        units=[{'unit_id':u['unit_id'],'text':u['text'],'approved_reference_role':u['approved_reference_role'],
                'evidence_markers':[e['marker'] for e in u['evidence']]}
               for u in refs['units'] if u['business_id']==bid and any(f.casefold() in u['text'].casefold() for f in ref_fragments)]
        if ref_fragments and not units:raise ValueError('No approved reference match')
        findings.append({'business_id':bid,'subset':case['subset'],'display_name':source['display_name'],
            'exact_new_claim_or_excerpt':claim,'classification':classification,'explanation':explanation,
            'LED_caption_PHOTO_evidence':caps,'REV_evidence_verbatim':revs,'approved_reference_links':units,
            'source_sha256':case['source_sha256'],'new_prediction_sha256':sha(p),
            'claim_also_verbatim_in_historical_F':claim in case['baseline_prediction']['text'],
            'claim_also_verbatim_in_saved_B':claim in oldb[bid]['text'],
            'human_verified':False,'assessment_status':'PROVISIONAL_AI_RESEARCHER_REVIEW_REQUIRED'})

    add(6,'The crab legs receive repeated praise, including lobster tail and scallops with lobster tail.',
        'SEMANTIC_REPETITION_AND_CATEGORY_INCOHERENCE',
        'One example of many minimally varied seafood/scallop sentences. Exact-sentence bans can be evaded by articles, conjunctions and suffixes. The ending is forced at768 tokens, with no visual section. This is not a usable completed summary.',
        review_markers=('REV_25',),ref_fragments=('oily sauce','overcooked crab','long waits','portions small'))
    add(6,'one review found the scallops disappointing','UNSUPPORTED_REVIEW_CLAIM',
        'Selected scallop evidence in REV25 praises their size and cooking; the full selected review source does not establish a disappointing-scallops opinion. Do not substitute negative crab/shrimp opinions for scallops.',
        review_markers=('REV_25','REV_3'),ref_fragments=('overcooked crab',))
    add(7,'A group of people is sitting at a table.','UNSUPPORTED_VISUAL_CLAIM_RETAINED_FROM_BASELINES',
        'The supplied captions describe beer vessels and a set dining table, with no group of people. The original IMG_CAP2 photo inspected shows the set table with empty seating. The same sentence persists from F and B.',('IMG_CAP_1','IMG_CAP_2'),ref_fragments=('Thai-hot spice','Pad Thai meal','large portions'))
    add(8,'The bar has a lot of tables and chairs in a dimly lit room.','CAPTION_SUPPORTED_BUT_SEMANTICALLY_REDUNDANT',
        'This ordinary scene description is supported by IMG_CAP2. Splitting and repeating the same bar/tables/chairs/room facts inflates the visual section without adding useful information. Source-caption support assessed; this review did not reinspect that photo.',('IMG_CAP_2',))
    add(9,'The fish is on a plate.','UNSUPPORTED_CAPTION_SPECIFICITY_RETAINED_FROM_B',
        'IMG_CAP1 describes fish in a tank; IMG_CAP2 says food on a plate, not specifically fish. Recombining the tank fish and plate scene is not caption-grounded even if fish might independently be plausible.',('IMG_CAP_1','IMG_CAP_2'),ref_fragments=('fish tank','aquarium-centered',))
    add(10,'The photo shows a close-up of a biscuit.','UNSUPPORTED_VISUAL_IDENTITY_RETAINED_FROM_BASELINES',
        'None of the five LED-visible captions identifies a biscuit. The close-up caption identifies a coffee machine, confirmed in the inspected original photo. Biscuit information in reviews does not authorize adding it to the visual section.',('IMG_CAP_3','IMG_CAP_5'),review_markers=('REV_4',),ref_fragments=('Waverly biscuit','coffee flavor','food waits'))
    add(12,'The glasses are on a wooden rack.','UNSUPPORTED_CROSS_PHOTO_RELATION',
        'IMG_CAP1 places glasses on a table; IMG_CAP3 places wine bottles in a wooden rack. Both original photos were inspected. The new output changes the incorrect platter/plate-on-rack relationship in F/B into glasses-on-rack, retaining the binding failure.',('IMG_CAP_1','IMG_CAP_3'))
    add(16,'including the Bang Bang shrimp appetizer, which was praised by several reviewers.',
        'SUPPORTED_BUSINESS_SPECIFIC_DETAIL_PRESERVED',
        'Multiple selected reviews praise this named dish. The source-bound repetition exception appropriately preserves Bang Bang rather than banning a legitimate name. This is a retained success from F/B, not a new quality gain.',
        review_markers=('REV_6','REV_19','REV_20'))
    add(16,'A slice of pineapple is on a plate.','NEW_UNSUPPORTED_VISUAL_RELATION',
        'The caption and original inspected photo put the pineapple slice on the martini glass. Neither F nor B has this exact plate sentence; B already has the separate pineapple-on-pineapple error retained in the new output.',('IMG_CAP_3',))
    add(17,'The group is wearing a neon sign for a bar in a dark room.','UNSUPPORTED_CROSS_PHOTO_ACTION',
        'The captions separately describe a neon sign and people at tables/stools. Both original photos were inspected. They do not support people wearing a sign. Preventing exact repeats can coexist with newly phrased nonsense.',('IMG_CAP_3','IMG_CAP_4'))
    add(21,'A bagel is on a plate.','ZERO_CAPTION_VISUAL_FABRICATION_REMAINS',
        'This business has zero Filtered captions and no allowed PHOTO/IMG_CAP evidence. Its approved visual section must remain empty. F and B already fail this rule; the new output continues to hallucinate plate/bagel/cream-cheese scenes. A review mention of bagels does not make visual evidence.',
        ref_fragments=('iced coffee extremely bitter','incorrect orders','missing order items'))
    # A complete software census is separate from selected qualitative examples.
    evidence={'status':'PROVISIONAL_AI_SELECTED_EVIDENCE_REVIEW_NOT_RESEARCHER_GO',
        'human_verified':False,'researcher_GO_NO_GO':'NOT_ASSIGNED',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'All raw texts available for side-by-side researcher review; these eleven findings are selected evidence-linked examples, not exhaustive claim-by-claim factual labels or numerical Core coverage.',
        'prior_five_unchanged_assessments':'outputs/stage_i/repetition_guard_quality_assessment_v1/AI_REVIEW_random_first8.json (only its five completed assessments apply; original pending statuses preserved)',
        'Core_rules':'Core examples are priorities, not verbatim requirements; Supporting omissions are not automatic errors. No new informativeness denominator or annotation approval is created.',
        'original_photo_AI_inspected_ids':sorted(inspected),'findings':findings,
        'quality_conclusion_provisional':'The execution crash is repaired. Surface-rule compliance does not establish a substantive quality rescue: semantic loops, unsupported visual bindings, missing information and zero-caption fabrication remain. Do not silently deploy this guard or launch another experiment.',
        'frozen_reference_sha256':sha(ROOT/'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json')}
    out=ROOT/'outputs/stage_i/repetition_guard_quality_assessment_v2'
    out.mkdir(exist_ok=False)
    write(out/'AI_QUALITY_REVIEW.json',evidence)
    print({'findings':len(findings),'human_verified':False,'quality_GO':'NOT_ASSIGNED'})


if __name__=='__main__':review()
