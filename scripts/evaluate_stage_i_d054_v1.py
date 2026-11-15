"""Offline D054 metrics and evidence dossiers; never produces model outputs/labels."""
import argparse,csv,json,re
from collections import Counter
from stage_i_d054 import OUT,ROOT,read,write,load_compatible_comparison,text_sha,preflight
from stage_i_d054_text import segments,unique_segments

def evidence_dossier(predictions,cases,units):
    by={c['business_id']:c for c in cases}
    return [{'business_id':p['business_id'],'arm':p['arm'],'subset':p['subset'],
      'prediction_text_sha256':text_sha(p['text']),'raw_text':p['text'],
      'unique_verbatim_segments':unique_segments(p['text']),
      'review_evidence':by[p['business_id']]['reviews'],
      'caption_evidence':by[p['business_id']]['captions'],
      'approved_reference_units':[u for u in units if u['business_id']==p['business_id']],
      'automatic_support_labels':None,'note':'Segments are retrieval aids, not atomic claim annotations or factual judgments.'} for p in predictions]

def evaluate():
    predictions=[json.loads(s) for s in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    evaluation=load_compatible_comparison(predictions)
    write(OUT/'AUTOMATIC_EVALUATION.json',evaluation)
    write(OUT/'PAIRED_COMPARISON.json',evaluation['paired'])
    write(OUT/'PENDING_RESEARCHER_OUTPUT_ANNOTATIONS.json',evaluation['pending_researcher_output_annotations'])
    ctx,*_=preflight();units=read(ctx['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json')['units']
    dossiers=evidence_dossier(predictions,ctx['cases'],units);write(OUT/'SOURCE_GROUNDING_DOSSIER.json',dossiers)
    write(OUT/'UNICODE_AND_SURFACE_AUDIT.json',{'status':'Descriptive rendering diagnostics, not a new quality score or success threshold','outputs':[
        {'business_id':p['business_id'],'arm':p['arm'],'decoded_characters':len(p['text']),
         'nonwhitespace_characters':len(re.sub(r'\s','',p['text'])),
         'nonASCII_codepoints':{'U+%04X'%ord(c):n for c,n in Counter(p['text']).items() if ord(c)>127},
         'nonwhitespace_after_removing_bracketed_markers':len(re.sub(r'\s','',re.sub(r'\[[^\]\n]*\]','',p['text']))),
         'raw_text_sha256':p['text_sha256'],'text_was_not_normalized':True} for p in predictions]})
    summaries=[]
    for c in ctx['cases']:
        row={'business_id':c['business_id'],'subset':c['subset'],'caption_count':len(c['captions'])}
        for arm,short in [('A_decoder_only','A'),('B_encoder_decoder','B')]:
            r=next(r for r in evaluation['per_output'] if r['business_id']==c['business_id'] and r['arm']==arm)
            a=r['automatic'];rep=a['repetition'];term=a['termination'];st=a['structure']
            for key,val in {'tokens':term['generated_tokens_including_decoder_start'],'cap_hit':term['cap_reached'],
                'early_eos':term['eos_before_length_cap'],'headers_compliant':st['section_headers_compliant'],
                'zero_caption_compliant':st['empty_visual_structurally_compliant'],
                'repeated_sentence':rep['exact_repeated_sentence'],'duplicate_sentence_fraction':rep['duplicate_sentence_fraction'],
                'adjacent_repeated_word':rep['consecutive_repeated_word'],'adjacent_pair_fraction':rep['consecutive_repeated_word_pair_fraction'],
                'lexical4_duplicate_fraction':rep['lexical_ngrams']['4']['duplicate_fraction'],
                'lexical4_distinct':rep['lexical_ngrams']['4']['distinct_n']}.items():row[short+'_'+key]=val
        for m in ['tokens','duplicate_sentence_fraction','adjacent_pair_fraction','lexical4_duplicate_fraction','lexical4_distinct']:
            row['B_minus_A_'+m]=None if row['A_'+m] is None or row['B_'+m] is None else row['B_'+m]-row['A_'+m]
        summaries.append(row)
    with (OUT/'PAIRED_PER_BUSINESS.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    write(OUT/'PAIRED_PER_BUSINESS.json',summaries)
    return evaluation,dossiers,summaries

if __name__=='__main__':
    evaluate();print(json.dumps({'outputs':48,'status':'OFFLINE_METRICS_AND_EVIDENCE_SAVED','human_quality_labels_created':0}))
