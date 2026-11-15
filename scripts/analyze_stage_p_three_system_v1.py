"""Recompute descriptive resource/equivalence diagnostics from frozen native records.
No generation, model loading, scoring of humans, or paid service calls.
"""
import argparse,csv,json,hashlib,re,statistics,collections
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def csvout(p,rows):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def md(rows):
    keys=list(rows[0]);return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(f'{x[k]:.3f}' if isinstance(x[k],float) else str(x[k]) for k in keys)+' |' for x in rows)
def analyze(root,output):
    n=root/'NATIVE';o=output;o.mkdir(parents=True,exist_ok=True)
    led=read(n/'LED_BENCHMARK/BENCHMARK_RECORDS_62.json');measured=[x for x in led if x['phase']!='WARMUP'];old=read(n/'INPUTS/ORIGINAL_LED_F_COMPONENTS_30.json');original={(x['business_id'],x['task']):x for x in old}
    qc=[read(p) for p in sorted((n/'INPUTS/QWEN_CAPTION_CALLS').glob('*.json'))];qi=[read(p) for p in sorted((n/'QWEN_IMAGE/CALLS').glob('*.json'))];assert len(measured)==60 and len(qc)==30 and len(qi)==15
    resources=read(n/'INPUTS/QWEN_CAPTION_RESOURCES.json');ic=read(n/'QWEN_IMAGE/COMPLETION.json');ir=read(n/'QWEN_IMAGE/RESTORATION.json');lc=read(n/'LED_BENCHMARK/COMPLETION.json')
    calls=[]
    for x in measured:
        calls.append(dict(system='LED F',repeat=x['phase'],business_id=x['business_id'],task=x['task'],input_tokens=x['unpadded_input_tokens'],padded_input_tokens=x['padded_input_tokens'],image_tokens=0,output_tokens=x['generated_token_count'],generation_seconds=x['generation_seconds'],tokens_per_second=x['generated_tokens_per_second'],peak_allocated_GiB=x['VRAM']['peak_allocated_bytes']/1024**3,peak_reserved_GiB=x['VRAM']['peak_reserved_bytes']/1024**3))
    for label,rows in [('Qwen-Caption',qc),('Qwen-Image',qi)]:
        for x in rows:
            calls.append(dict(system=label,repeat='single',business_id=x['business_id'],task=x['task'],input_tokens=x.get('input_tokens',x.get('input_provenance',{}).get('input_tokens')),padded_input_tokens=x.get('input_tokens',x.get('input_provenance',{}).get('input_tokens')),image_tokens=x.get('input_provenance',{}).get('image_token_count',0),output_tokens=x['output_tokens'],generation_seconds=x['generation_seconds'],tokens_per_second=x['output_tokens']/x['generation_seconds'],peak_allocated_GiB=x['VRAM']['peak_allocated_bytes']/1024**3,peak_reserved_GiB=x['VRAM']['peak_reserved_bytes']/1024**3))
    csvout(o/'PER_CALL_RESOURCES.csv',calls)
    summaries=[]
    for label,repeat,selection in [('LED F','REPEAT_1',[x for x in calls if x['repeat']=='REPEAT_1']),('LED F','REPEAT_2',[x for x in calls if x['repeat']=='REPEAT_2']),('Qwen-Caption','single',[x for x in calls if x['system']=='Qwen-Caption']),('Qwen-Image Visual only','single',[x for x in calls if x['system']=='Qwen-Image'])]:
        total=sum(x['generation_seconds'] for x in selection);tokens=sum(x['output_tokens'] for x in selection)
        summaries.append(dict(system=label,repeat=repeat,Review_seconds=sum(x['generation_seconds'] for x in selection if x['task']=='review'),Visual_seconds=sum(x['generation_seconds'] for x in selection if x['task']=='visual'),total_seconds=total,seconds_per_restaurant=total/15,output_tokens=tokens,tokens_per_second=tokens/total,peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in selection),peak_reserved_GiB=max(x['peak_reserved_GiB'] for x in selection)))
    qreview=[x for x in calls if x['system']=='Qwen-Caption' and x['task']=='review'];qvisual=[x for x in calls if x['system']=='Qwen-Image'];both=qreview+qvisual
    summaries.append(dict(system='Qwen-Image constructed Review+Visual',repeat='separate sessions',Review_seconds=sum(x['generation_seconds'] for x in qreview),Visual_seconds=sum(x['generation_seconds'] for x in qvisual),total_seconds=sum(x['generation_seconds'] for x in both),seconds_per_restaurant=sum(x['generation_seconds'] for x in both)/15,output_tokens=sum(x['output_tokens'] for x in both),tokens_per_second=sum(x['output_tokens'] for x in both)/sum(x['generation_seconds'] for x in both),peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in both),peak_reserved_GiB=max(x['peak_reserved_GiB'] for x in both)))
    csvout(o/'RESOURCE_COMPARISON.csv',summaries)
    per=[]
    for bid in sorted({x['business_id'] for x in calls}):
        for s,rep in [('LED F','REPEAT_1'),('LED F','REPEAT_2'),('Qwen-Caption','single'),('Qwen-Image','single')]:
            rows=[x for x in calls if x['business_id']==bid and x['system']==s and x['repeat']==rep]
            if s=='Qwen-Image':rows+= [x for x in qreview if x['business_id']==bid]
            per.append(dict(business_id=bid,system=s,repeat=rep,constructed=s=='Qwen-Image',review_seconds=sum(x['generation_seconds'] for x in rows if x['task']=='review'),visual_seconds=sum(x['generation_seconds'] for x in rows if x['task']=='visual'),total_seconds=sum(x['generation_seconds'] for x in rows),output_tokens=sum(x['output_tokens'] for x in rows)))
    csvout(o/'PER_RESTAURANT_RESOURCES.csv',per)
    eq=[];diff=[]
    reps={p:{(x['business_id'],x['task']):x for x in measured if x['phase']==p} for p in ['REPEAT_1','REPEAT_2']}
    for x in measured:
        k=x['business_id'],x['task'];prior=original[k];a=reps['REPEAT_1'][k];b=reps['REPEAT_2'][k];same=x['token_ids']==prior['token_ids'];exact=x['text']==prior['text']
        assert same==x['original_tokens_exact'] and exact==x['original_text_exact'];assert textsha(prior['text'])==prior['text_sha256']
        row=dict(business_id=k[0],task=k[1],repeat=x['phase'],original_tokens_exact=same,original_text_exact=exact,between_repetitions_exact=a['token_ids']==b['token_ids'],original_text_sha256=prior['text_sha256'],rerun_text_sha256=x['text_sha256']);eq.append(row)
        if not exact:diff.append(dict(**row,original_text=prior['text'],rerun_text=x['text'],original_token_ids=prior['token_ids'],rerun_token_ids=x['token_ids']))
    csvout(o/'LED_EQUIVALENCE_60.csv',eq);put(o/'LED_ALL_DIFFERENCES.json',diff)
    eqsummary={p:dict(text_exact=sum(x['original_text_exact'] for x in eq if x['repeat']==p),tokens_exact=sum(x['original_tokens_exact'] for x in eq if x['repeat']==p),review_exact=sum(x['original_text_exact'] for x in eq if x['repeat']==p and x['task']=='review'),visual_exact=sum(x['original_text_exact'] for x in eq if x['repeat']==p and x['task']=='visual'),calls=30) for p in reps}
    totals=[x['total_seconds'] for x in summaries[:2]]
    variability=dict(repetitions=2,total_generation_seconds_mean=statistics.mean(totals),sample_SD_seconds=statistics.stdev(totals),minimum_seconds=min(totals),maximum_seconds=max(totals),between_repeat_token_exact=sum(reps['REPEAT_1'][k]['token_ids']==reps['REPEAT_2'][k]['token_ids'] for k in reps['REPEAT_1']),matched_calls=30)
    put(o/'LED_EQUIVALENCE_SUMMARY.json',dict(original_equivalence=eqsummary,variability=variability,original_outputs_replaced=False,causal_attribution='Undetermined: original RTX3080/Torch2.1.0+cu118 versus current RTX3090/Torch2.9.1+cu128. Same verified weights, sources, TF4.51.3, decoder and precision policy. Framework and hardware both changed; no single cause established.'))
    diagnostics=[]
    for system,rows in [('Original LED F',old),('Qwen-Caption',qc),('Qwen-Image Visual',qi)]:
        for x in rows:
            text=x['text'];plain=re.sub(r'^\[[A-Z_]+\]\s*','',text);words=re.findall(r"\b[\w'-]+\b",plain.lower());grams=list(zip(*(words[i:] for i in range(4))))
            sentences=[re.sub(r'\s+',' ',s.strip().lower()) for s in re.split(r'(?<=[.!?])\s+',plain) if s.strip()]
            task=x['task'];expected='[REVIEW_CONSENSUS]' if task=='review' else '[VISUAL_EVIDENCE]'
            diagnostics.append(dict(system=system,business_id=x['business_id'],task=task,words=len(words),empty=not plain.strip(),header_missing=not text.startswith(expected),thinking_tag=('<think>' in text or '</think>' in text),exact_duplicate_sentences=len(sentences)-len(set(sentences)),repeated_4gram_occurrences=len(grams)-len(set(grams)),repeated_4gram_fraction=(len(grams)-len(set(grams)))/len(grams) if grams else 0,hit_token_cap=x.get('flags',{}).get('hit_token_cap',x.get('flags',{}).get('hit_length_cap',False)),text_sha256=textsha(text)))
    csvout(o/'GENERATION_DIAGNOSTICS.csv',diagnostics)
    ds=[]
    for key in sorted({(x['system'],x['task']) for x in diagnostics}):
        rows=[x for x in diagnostics if (x['system'],x['task'])==key]
        ds.append(dict(system=key[0],task=key[1],calls=len(rows),mean_words=statistics.mean(x['words'] for x in rows),min_words=min(x['words'] for x in rows),max_words=max(x['words'] for x in rows),empty=sum(x['empty'] for x in rows),header_missing=sum(x['header_missing'] for x in rows),thinking=sum(x['thinking_tag'] for x in rows),token_caps=sum(x['hit_token_cap'] for x in rows),duplicate_sentence_occurrences=sum(x['exact_duplicate_sentences'] for x in rows)))
    csvout(o/'DIAGNOSTIC_SUMMARY.csv',ds)
    shared=read(n/'HISTORICAL_SHARED_PREPARATION/BLIP_GENERATION_REPORT.json')['elapsed_seconds']+read(n/'HISTORICAL_SHARED_PREPARATION/CAPTION_SCORING_REPORT.json')['elapsed_seconds']
    put(o/'RESOURCE_SUMMARY.json',dict(comparison=summaries,LED_variability=variability,load_seconds=dict(LED=lc['model_loading_seconds'],Qwen_Caption=resources['load_seconds'],Qwen_Image=ic['model_loading_seconds']),image_CPU_preprocessing_seconds=sum(x['CPU_image_preprocessing_seconds'] for x in qi),image_GPU_transfer_seconds=sum(x['GPU_transfer_seconds'] for x in qi),image_CPU_postprocessing_seconds=sum(x['CPU_postprocessing_seconds'] for x in qi),historical_shared_preparation_subtotal_seconds=shared,diagnostics=ds))
    o.joinpath('RESOURCE_COMPARISON.md').write_text('# Three-system resource measurements\n\n'+md(summaries)+'\n\nLED measured twice after two warm-up calls; Qwen measurements are single passes including their pilot. The Qwen-Image combined row constructs cost from an earlier Review session plus the new Visual session. No second Review generation occurred. Allocated/reserved columns are per-generation allocator peaks; load peaks and sampled whole-device measurements are in native receipts.\n\n'+f'LED total mean {statistics.mean(totals):.3f} s, sample SD {statistics.stdev(totals):.3f} s (n=2, no statistical performance claim). Loading: LED {lc["model_loading_seconds"]:.3f} s; Qwen-Caption {resources["load_seconds"]:.3f} s; Qwen-Image {ic["model_loading_seconds"]:.3f} s.\n\n'+f'Qwen-Image CPU preprocessing {sum(x["CPU_image_preprocessing_seconds"] for x in qi):.3f} s; GPU transfer {sum(x["GPU_transfer_seconds"] for x in qi):.3f} s; CPU postprocessing {sum(x["CPU_postprocessing_seconds"] for x in qi):.3f} s, separately measured outside model.generate.\n\n'+f'Historical shared BLIP generation and CLIP/SBERT scoring subtotal: {shared:.3f} s. This is historical RTX3080/mixed CPU component accounting, not a current3090 full-pipeline benchmark. BLIP generation report already includes model loading: do not add it again. Selection/acquisition and some CPU costs were not measured. All three systems inherit the same selected-photo provenance. Actual incremental caption preparation for this extension is zero. The direct-image variant bypasses caption text at inference, but still inherits caption-based selection. Do not infer monetary cost from VRAM.\n',encoding='utf8')
    o.joinpath('LED_EQUIVALENCE.md').write_text('# Frozen LED equivalence\n\n'+json.dumps(eqsummary,indent=2)+'\n\nBoth repetitions match each other on all30 token sequences. Each matches22/30 original frozen texts/tokens (9/15 Review,13/15 Visual);8 differ. All original and rerun texts/token IDs for every difference are preserved in LED_ALL_DIFFERENCES.json. Original predictions and90 human ratings remain immutable. Therefore current resource measurements are not an exact reproduction of all historically rated strings.\n\nThe original runtime used RTX3080 and Torch2.1.0+cu118; current measurements use RTX3090 and Torch2.9.1+cu128. Transformers4.51.3, weights, source IDs/tokens, masks, FP32+FP16-autocast policy and locked4-beam decoder match. The observed difference cannot be uniquely attributed to GPU or Torch without another experiment, which was not run.\n',encoding='utf8')
    o.joinpath('PRELIMINARY_DIAGNOSTICS.md').write_text('# Preliminary generation diagnostics\n\n'+md(ds)+'\n\nThese are descriptive string diagnostics, not human ratings or confirmed factual-error counts. Repeated4grams can be legitimate recurring terms; exact sentence repetition is case/whitespace normalized. Word counts remove the leading section header; token counts use different tokenizers and include EOS (LED excludes the decoder seed). Length is not a quality score. See native flags for EOS and raw token caps. Semantic observations are separately identified as model-assisted, preliminary and source-limited. All outputs remain verbatim.\n',encoding='utf8')
    return dict(resource_rows=len(summaries),diagnostic_rows=len(diagnostics),equivalence_rows=len(eq),image_calls=len(qi),photo_presentations=sum(len(x['selected_photo_ids']) for x in qi),image_CPU_threads=ir['torch_CPU_threads'])
def plot(root,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=read(root/'REPORTS/RESOURCE_SUMMARY.json');rows=data['comparison'];labels=['LED F\nrepeat1','LED F\nrepeat2','Qwen\nCaption','Qwen Image\nVisual only','Qwen Image\nconstructed full']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    review=[x['Review_seconds'] for x in rows];visual=[x['Visual_seconds'] for x in rows]
    axes[0].bar(labels,review,label='Review');axes[0].bar(labels,visual,bottom=review,label='Visual');axes[0].set_ylabel('Synchronized model.generate seconds /15 restaurants');axes[0].legend()
    axes[1].bar(labels,[x['peak_allocated_GiB'] for x in rows],label='Allocated');axes[1].scatter(labels,[x['peak_reserved_GiB'] for x in rows],color='black',marker='_',s=180,label='Reserved');axes[1].set_ylim(0,24);axes[1].set_ylabel('Peak generation VRAM (GiB)');axes[1].legend()
    fig.suptitle('Same RTX3090; different models, tokenizers, budgets and numerical policies');output.mkdir(parents=True,exist_ok=True);fig.savefig(output/'RESOURCE_COMPARISON.png',dpi=150);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path);p.add_argument('--plots',action='store_true');a=p.parse_args();out=a.out or a.root/'REPORTS';print(json.dumps(analyze(a.root,out))); 
    if a.plots:plot(a.root,out/'FIGURES')
