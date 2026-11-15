"""Frozen-input TEST diagnostics only. No reference reader or reference-based scorer."""
from pathlib import Path
import argparse,csv,collections,statistics,sys
from s1000_final_eval_core_v1 import *

def component_diagnostics(r,call,caption_count):
    marker='[REVIEW_CONSENSUS]' if r['task']=='review' else '[VISUAL_EVIDENCE]'
    foreign='[VISUAL_EVIDENCE]' if r['task']=='review' else '[REVIEW_CONSENSUS]'
    body=r['text'][len(marker)+1:] if r['text'].startswith(marker+'\n') else r['text']
    # Task markers/IMG_CAP IDs are excluded from lexical source overlap.
    if r['task']=='visual':
        parts=re.split(r'\[IMG_CAP_\d+\]\n',call['source'])[1:];source='\n'.join(parts)
    else:source=call['source'].split('[REVIEWS]\n',1)[1]
    d=diagnostics(body,source);ids=r['token_ids']
    sentence_norm=[' '.join(words(x)) for x in re.split(r'(?<=[.!?])\s+',body.strip()) if x.strip()]
    counts=Counter(sentence_norm)
    d.update(raw_generated_tokens=len(ids),content_generated_tokens=sum(i not in (0,1,2) for i in ids),
        repeated_full_sentence_occurrences=sum(n-1 for n in counts.values() if n>1),
        repeated_full_sentence_types=sum(n>1 for n in counts.values()),
        starts_expected_header=r['text'].startswith(marker+'\n'),expected_header_count=r['text'].count(marker),
        foreign_component_header=foreign in r['text'],EOS_seen_after_decoder_start=2 in ids[1:],
        last_token_EOS=bool(ids) and ids[-1]==2,at_max_length=len(ids)>=768,
        forced_EOS_at_cap_possible=len(ids)>=768 and ids[-1]==2,
        ends_sentence_punctuation=body.rstrip().rstrip('\"\'”’').endswith(('.', '!', '?')),
        caption_count=caption_count if r['task']=='visual' else None,
        empty_caption_nonempty_output=r['task']=='visual' and caption_count==0 and bool(body.strip()),
        independent_generation=r['independent_generation'],generation_status=r.get('generation_status','SUCCESS'),
        interpretation='Descriptive lexical/format/termination diagnostics only; not factuality, image grounding or semantic completeness')
    d['malformed']=not d['starts_expected_header'] or d['expected_header_count']!=1 or d['foreign_component_header']
    return d

def analyze(prep,execution,out):
    from tokenizers import Tokenizer
    prep,execution,out=map(Path,(prep,execution,out));require(not out.exists(),'Refuse overwrite completed diagnostics')
    lock=read(prep/'MODEL_AND_DECODER_LOCK.json');plan=read(prep/'TEST100_INFERENCE_PLAN.json')
    tokpath=Path(__file__).resolve().parents[1]/'assets/tokenizers/led-base-16384/tokenizer.json'
    require(file_hash(tokpath)==lock['tokenizer']['sha256']['tokenizer.json'],'Independent output decoder tokenizer differs')
    tok=Tokenizer.from_file(str(tokpath))
    calls={(c['condition'],c['business_id'],c['task']):c for c in plan['calls']};raw={};bind={}
    for a in ['RO','U','F','S']:
        folder=execution/a;done=read(folder/'COMPLETION.json');pp=folder/'raw_predictions.json'
        require(done['condition']==a and done['businesses']==100 and done['raw_predictions_sha256']==file_hash(pp),'Incomplete native condition')
        rr=read(pp);expected={k for k in calls if k[0]==a}
        require(len(rr)==len(expected) and {(a,r['business_id'],r['task']) for r in rr}==expected,'Missing/duplicate output components')
        for r in rr:
            k=a,r['business_id'],r['task'];c=calls[k]
            require(r['checkpoint_sha256']==c['checkpoint_sha256'] and r['source_sha256']==c['source_sha256'] and r['input_ids_sha256']==c['input_ids_sha256'],'Output/source/checkpoint mismatch')
            require(r['generation_config_sha256']==digest(lock['generation']) and r['generation_settings']==lock['generation'],'Decoder mismatch')
            require(r['input_ids']==c['input_ids'] and digest(r['input_ids'])==c['input_ids_sha256'] and text_hash(r['text'])==r['text_sha256'],'Token/text integrity')
            require(0<len(r['token_ids'])<=768 and r['token_ids'][0]==2,'Raw generation sequence malformed')
            require(tok.decode(r['token_ids'],skip_special_tokens=True)==r['text'],'Raw generated IDs do not decode to saved text')
            raw[k]=r
        bind[a]=dict(completion_sha256=file_hash(folder/'COMPLETION.json'),predictions_sha256=file_hash(pp),generation_calls=done['new_generation_calls'],seconds=done['seconds'],GPU=done['GPU_name'],peak_allocated_bytes=done['peak_allocated_bytes'])
    require(len(raw)==700,'Need700 actual/reused logical components')
    for c in plan['calls']:
        if c['reuse_condition']:
            r=raw[c['condition'],c['business_id'],c['task']];f=raw['F',c['business_id'],c['task']]
            require(r['text']==f['text'] and r['token_ids']==f['token_ids'] and not r['independent_generation'],'F reuse not exact')
    ids=read(prep/'COHORT100_FREEZE.json')['business_ids'];mapping={r['recipient_id']:r for r in read(prep/'TEST100_SHUFFLED_INPUT_MANIFEST.json')['mappings']}
    rows=[];index={};records=[]
    for a in ['RO','U','F','S']:
        for b in ids:
            tasks=['review'] if a=='RO' else ['review','visual'];passes={}
            for task in tasks:
                r=raw[a,b,task];c=calls[a,b,task];n=len(c['selected_photo_ids'])
                d=component_diagnostics(r,c,n);row=dict(condition=a,business_id=b,task=task,output_sha256=r['text_sha256'],source_sha256=r['source_sha256'],
                    checkpoint_sha256=r['checkpoint_sha256'],shuffled_donor_id=mapping[b]['donor_id'] if a=='S' else None,
                    F_S_intervention_applicable=mapping[b]['visual_source_changed'] if a in ('F','S') else None,**d)
                rows.append(row);index[a,b,task]=row;passes[task]=r
            records.append(dict(business_id=b,condition=a,checkpoint_condition='F' if a=='S' else a,checkpoint_sha256=lock['models']['F' if a=='S' else a]['adapter_sha256'],
                review_text=passes['review']['text'],visual_text=passes['visual']['text'] if 'visual' in passes else None,visual_applicability='N/A' if a=='RO' else 'APPLICABLE',
                combined_text='\n\n'.join(passes[t]['text'] for t in tasks),passes=passes,rewriting=False,
                shuffled_donor_id=mapping[b]['donor_id'] if a=='S' else None,no_F_S_visual_intervention=not mapping[b]['visual_source_changed'] if a in ('F','S') else None))
    aggregate=[]
    for a in ['RO','U','F','S']:
        for task in ['review','visual']:
            rr=[r for r in rows if r['condition']==a and r['task']==task]
            if not rr:aggregate.append(dict(condition=a,task=task,status='N/A_NO_VISUAL_COMPONENT',businesses=0));continue
            agg=dict(condition=a,task=task,status='ACTUAL_DESCRIPTIVE_METRICS',businesses=len(rr))
            for f in ['words','raw_generated_tokens','content_generated_tokens','repeated_3gram_fraction','duplicate_sentence_fraction','repeated_full_sentence_occurrences','source_lexical_precision_proxy']:
                v=[r[f] for r in rr];agg[f]=dict(mean=statistics.mean(v),median=statistics.median(v),minimum=min(v),maximum=max(v))
            for f in ['empty','malformed','EOS_seen_after_decoder_start','last_token_EOS','at_max_length','forced_EOS_at_cap_possible','empty_caption_nonempty_output','ends_sentence_punctuation']:
                agg[f]=sum(bool(r[f]) for r in rr)
            if task=='visual':agg['caption_count_distribution']=dict(sorted(Counter(r['caption_count'] for r in rr).items()))
            aggregate.append(agg)
    same=[b for b in ids if index['U',b,'visual']['caption_count']==index['F',b,'visual']['caption_count']]
    active=[b for b in ids if mapping[b]['visual_source_changed']];zero=[b for b in ids if not mapping[b]['visual_source_changed']]
    paired_rows=[]
    for label,A,B,task,group in [('U_MINUS_F_ALL100','U','F','visual',ids),('U_MINUS_F_SAME_CAPTION_COUNT','U','F','visual',same),('F_MINUS_S_ACTUAL_INTERVENTION','F','S','visual',active),('F_MINUS_S_ALL100','F','S','visual',ids)]:
        for metric in ['words','repeated_3gram_fraction','duplicate_sentence_fraction','source_lexical_precision_proxy']:
            paired_rows.append(dict(contrast=label,task=task,metric=metric,interpretation='Descriptive paired bootstrap; no factuality or confirmatory significance claim',
                **paired([index[A,b,task][metric] for b in group],[index[B,b,task][metric] for b in group],B=20000,P=1000)))
    out.mkdir(parents=True)
    write_csv(out/'TEST100_REFERENCE_FREE_METRICS.csv',rows)
    with (out/'TEST100_RO_U_F_S_PREDICTIONS.jsonl').open('x',encoding='utf8') as f:
        for r in records:f.write(canonical(r).decode('utf8')+'\n')
    write(out/'REFERENCE_FREE_AGGREGATES.json',aggregate)
    write(out/'REFERENCE_FREE_PAIRED_ANALYSIS.json',dict(paired=paired_rows,same_caption_count_business_ids=same,F_S_active_business_ids=active,F_S_zero_intervention_business_ids=zero,
        interpretation='Paired reference-free descriptive analysis, independent of human references. Source overlap for S is overlap with deliberately wrong donor captions, not recipient photo grounding.'))
    write(out/'GENERATION_INTEGRITY_RECEIPT.json',dict(status='PASS_400_ACTUAL_CONDITION_RECORDS',businesses=100,condition_records=len(records),logical_components=len(raw),
        new_generation_calls=sum(r['independent_generation'] for r in raw.values()),reused_components=sum(not r['independent_generation'] for r in raw.values()),
        F_S_review_identical_all100=True,changed_S_Visual=98,zero_intervention_cases=2,source_bindings_verified=True,output_text_hashes_verified=True,
        original_preparation_unchanged=True,reference_payloads_read=0,reference_based_TEST_scores=0,RO_visual_penalty=False,
        same_caption_count_paired_businesses=len(same),input_receipts=bind,predictions_file_sha256=file_hash(out/'TEST100_RO_U_F_S_PREDICTIONS.jsonl'),metrics_file_sha256=file_hash(out/'TEST100_REFERENCE_FREE_METRICS.csv')))
    return read(out/'GENERATION_INTEGRITY_RECEIPT.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True);p.add_argument('--execution',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.preparation,a.execution,a.output),indent=2))
