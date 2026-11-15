"""Bind Codex-authored assessment notes to immutable output/source spans.

No LLM call, automatic human approval, target generation, or new factual scorer.
The recipes are explicitly model-assisted judgments, not scientific human labels.
"""
import json,csv
from collections import Counter
from stage_i_d054 import OUT,read,write,text_sha,preflight,require

def assemble():
    ctx,*_=preflight();cases={c['business_id']:c for c in ctx['cases']}
    refs=read(ctx['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json')['units']
    predictions=[json.loads(s) for s in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    recipes=read(OUT/'CODEX_SOURCE_ASSESSMENT_RECIPES.json')
    require(len(recipes['outputs'])==48,'Need notes for all48 outputs')
    require({(r['business_id'],r['arm']) for r in recipes['outputs']}=={(p['business_id'],p['arm']) for p in predictions},'Assessment scope mismatch')
    rows=[]
    for recipe in recipes['outputs']:
        p=next(p for p in predictions if p['business_id']==recipe['business_id'] and p['arm']==recipe['arm'])
        c=cases[p['business_id']];units=[u for u in refs if u['business_id']==c['business_id']]
        evidence={r['marker']:{**r,'domain':'review'} for r in c['reviews']}
        evidence.update({r['caption_id']:{**r,'marker':r['caption_id'],'domain':'visual'} for r in c['captions']})
        claims=[]
        for n,claim in enumerate(recipe.get('claims',[])):
            text=claim['text'];start=p['text'].find(text);require(start>=0,'Claim quotation not in output: '+text)
            bound=[]
            for marker in claim.get('markers',[]):
                e=evidence[marker];bound.append({k:e[k] for k in ['marker','domain','text']+(['review_id'] if e['domain']=='review' else ['photo_id','image_file'])})
            claims.append({**claim,'claim_id':p['arm']+'_'+c['business_id']+'_AI_EXAMPLE_'+str(n+1),
                'start':start,'end':start+len(text),'evidence':bound,'human_verified':False,
                'coverage':'Representative assessable claim/span; not exhaustive atomic segmentation of repetitive output'})
        examples=[]
        for suffix in recipe.get('important_omission_examples',[]):
            matching=[u for u in units if u['unit_id']==suffix or u['unit_id'].endswith(c['business_id']+'_'+suffix)]
            require(len(matching)==1,'Reference suffix missing/ambiguous '+suffix);examples.append(matching[0])
        recovered=[]
        for match in recipe.get('reference_matches',[]):
            matching=[u for u in units if u['unit_id']==match['suffix'] or u['unit_id'].endswith(c['business_id']+'_'+match['suffix'])]
            require(len(matching)==1,'Reference match invalid')
            span=match['output_text'];start=p['text'].find(span);require(start>=0,'Reference match lacks exact output span')
            recovered.append({'unit':matching[0],'state':match['state'],'output_text':span,'start':start,'end':start+len(span),'human_verified':False})
        rows.append({**{k:recipe[k] for k in ['business_id','arm','summary','semantic_redundancy','review_grounding','visual_grounding','informativeness','contrast']},
          'subset':c['subset'],'prediction_text_sha256':p['text_sha256'],'human_verified':False,
          'assessment_origin':'Codex source-based model-assisted inspection; not independent human output assessment',
          'claims':claims,'source_reference_matches_examples':recovered,'important_core_omission_examples':examples,
          'full_output_atomic_grounding_score':None,'complete_unit_coverage_score':None,
          'unassessed_units_are_not_automatically_omitted':True,
          'note':'No exhaustive human-equivalent coverage/precision score is claimed. Exact examples and qualitative source review support the interpretation.'})
    write(OUT/'CODEX_SOURCE_BASED_ASSESSMENT.json',{'status':'MODEL_ASSISTED_QUALITATIVE_INSPECTION_NOT_RESEARCHER_CONFIRMED',
      'outputs_reviewed':48,'source_references':{'core':261,'supporting':315,'excluded':88},
      'formal_human_grounding_and_coverage':'N/A pending actual output annotations',
      'no_new_embedding_or_threshold':True,'outputs':rows,'paired_case_assessments':recipes['paired_case_assessments']})
    md=['# Model-assisted source-based assessment — not human output ground truth','',
      'All48 outputs were inspected. The examples below are auditable but are not an exhaustive atomic claim/coverage annotation. Formal human metrics remain N/A. Supporting omissions are not automatically errors; excluded units never enter a denominator.','']
    for c in ctx['cases']:
        md+=['## '+c['business_id']+' — '+c['display_name']+' ('+c['subset']+')','']
        for arm in ['A_decoder_only','B_encoder_decoder']:
            r=next(r for r in rows if r['business_id']==c['business_id'] and r['arm']==arm)
            md+=['### '+arm,'',r['summary'],'',r['informativeness'],'',r['review_grounding'],'',r['visual_grounding'],'',r['contrast'],'']
            for claim in r['claims']:
                md+=['> '+claim['text'],'','Codex assessment: **'+claim['label']+'**. '+claim['explanation'],'']
                for e in claim['evidence']:
                    md+=['Evidence '+e['marker']+' / '+e.get('review_id',e.get('photo_id'))+':','',e['text'],'']
            for u in r['important_core_omission_examples']:
                md+=['Important omission example: '+u['unit_id']+' — '+u['text'],'']
                for e in u['evidence']:md+=['Evidence '+e['marker']+' / '+e.get('review_id',e.get('photo_id'))+': '+e['quote'],'']
        paired=next(r for r in recipes['paired_case_assessments'] if r['business_id']==c['business_id'])
        md+=['Paired Codex interpretation: '+paired['assessment'],'']
    write(OUT/'SOURCE_BASED_QUALITY_ASSESSMENT.md','\n'.join(md)+'\n')
    comparison=[]
    for c in ctx['cases']:
        a=next(r for r in rows if r['business_id']==c['business_id'] and r['arm']=='A_decoder_only')
        b=next(r for r in rows if r['business_id']==c['business_id'] and r['arm']=='B_encoder_decoder')
        pair=next(r for r in recipes['paired_case_assessments'] if r['business_id']==c['business_id'])
        comparison.append({'business_id':c['business_id'],'subset':c['subset'],'A_summary':a['summary'],'B_summary':b['summary'],
          'paired_model_assisted_assessment':pair['assessment'],'human_verified':False,
          'A_reference_matches_examples':';'.join(m['unit']['unit_id'] for m in a['source_reference_matches_examples']),
          'B_reference_matches_examples':';'.join(m['unit']['unit_id'] for m in b['source_reference_matches_examples'])})
    with (OUT/'PAIRED_QUALITY_INSPECTION.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    print(json.dumps({'outputs':len(rows),'claim_examples':sum(len(r['claims']) for r in rows),'human_labels_created':0}))

if __name__=='__main__':assemble()
