"""Two separately gated reference tracks using the frozen S1000 metric contract."""
import argparse
import csv
import importlib.metadata
import math
from pathlib import Path
import sys
import time
from s1000_final_eval_core_v1 import *
from score_s1000_automatic_prepared_v1 import strip_header,descriptive_score
from validate_s1000_human_references_v1 import validate_all

METRICS=['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']
CONTRASTS=[('review','RO'),('review','U'),('visual','U'),('visual','S')]
TRACK_NAMES={'human':'HUMAN_REFERENCE_PRIMARY','teacher':'AI_TEACHER_REFERENCE_SUPPLEMENTARY'}

def load_frozen_predictions(preparation,stage_k):
    from tokenizers import Tokenizer
    prep,stage_k=Path(preparation),Path(stage_k);plan=read(prep/'TEST100_INFERENCE_PLAN.json');lock=read(prep/'MODEL_AND_DECODER_LOCK.json')
    ids=read(prep/'COHORT100_FREEZE.json')['business_ids'];require(len(ids)==len(set(ids))==100,'Frozen TEST100 membership')
    calls={(r['condition'],r['business_id'],r['task']):r for r in plan['calls']};require(len(calls)==700,'Frozen component plan')
    tokenizer=Path(__file__).resolve().parents[1]/'assets/tokenizers/led-base-16384/tokenizer.json'
    require(file_hash(tokenizer)==lock['tokenizer']['sha256']['tokenizer.json'],'Frozen LED tokenizer changed')
    tok=Tokenizer.from_file(str(tokenizer));rows=[];bindings={}
    for condition in ['RO','U','F','S']:
        folder=stage_k/'EXECUTION'/condition;done=read(folder/'COMPLETION.json');path=folder/'raw_predictions.json';raw=read(path)
        require(done['condition']==condition and done['businesses']==100 and done['plan_sha256']==file_hash(prep/'TEST100_INFERENCE_PLAN.json'),'Native completion identity')
        require(done['raw_predictions_sha256']==file_hash(path),'Native raw prediction hash changed')
        expected={k for k in calls if k[0]==condition}
        require(len(raw)==len(expected) and {(condition,r['business_id'],r['task']) for r in raw}==expected,'Missing/duplicate native components')
        bindings[condition]={'completion_sha256':file_hash(folder/'COMPLETION.json'),'raw_predictions_sha256':file_hash(path)}
        for r in raw:
            key=condition,r['business_id'],r['task'];c=calls[key]
            require(r['condition']==condition and all(r[k]==c[k] for k in ['checkpoint_sha256','source_sha256','input_ids_sha256']),'Output model/input identity')
            require(r['input_ids']==c['input_ids'] and digest(r['input_ids'])==r['input_ids_sha256'],'Actual input token binding')
            require(r['generation_settings']==lock['generation'] and r['generation_config_sha256']==digest(lock['generation']),'Frozen decoder binding')
            require(text_hash(r['text'])==r['text_sha256'] and tok.decode(r['token_ids'],skip_special_tokens=True)==r['text'],'Raw output token/text binding')
            require(text_hash(c['source'])==c['source_sha256'],'Frozen source text binding')
            rows.append(r)
    by={(r['condition'],r['business_id'],r['task']):r for r in rows};mapping=read(prep/'TEST100_SHUFFLED_INPUT_MANIFEST.json')['mappings'];donors={r['recipient_id']:r for r in mapping}
    require(set(donors)==set(ids),'Shuffled recipient frame')
    for c in plan['calls']:
        if c['reuse_condition']:
            a=by[c['condition'],c['business_id'],c['task']];b=by['F',c['business_id'],c['task']]
            require(a['text']==b['text'] and a['token_ids']==b['token_ids'] and not a['independent_generation'],'F reuse differs')
    permutation=0
    for b in ids:
        f,s=by['F',b,'review'],by['S',b,'review'];require(f['text']==s['text'] and f['token_ids']==s['token_ids'],'F/S Review invariant')
        donor=donors[b]['donor_id'] or b;f,s=by['F',donor,'visual'],by['S',b,'visual']
        require(f['text']==s['text'] and f['token_ids']==s['token_ids'],'S Visual is not its F donor output');permutation+=1
    active=[b for b in ids if donors[b]['visual_source_changed']]
    require(len(rows)==700 and len(active)==98,'Frozen output/intervention frame')
    return rows,calls,ids,active,{'status':'PASS_FROZEN_400_RECORDS_700_COMPONENTS','condition_bindings':bindings,'F_S_review_exact_pairs':100,
        'S_visual_exact_F_donor_pairs':permutation,'active_visual_intervention':len(active),'RO_visual_rows':0,
        'preparation_bindings':{name:file_hash(prep/name) for name in ['COHORT100_FREEZE.json','MODEL_AND_DECODER_LOCK.json','TEST100_INFERENCE_PLAN.json','TEST100_SHUFFLED_INPUT_MANIFEST.json','METRIC_LOCK.json','HUMAN_EVALUATION_COHORT30.json']}}

def human_references(prep,export):
    export=Path(export);package=Path(prep)/'TEST100_REFERENCE_ANNOTATION_PACKAGE'
    freeze=read(export/'REFERENCE_FREEZE.json')
    require(freeze.get('human_written_and_second_checked') is True and freeze['businesses']==100,'Full genuine human-reference freeze required')
    require(freeze['source_manifest_sha256']==file_hash(package/'ANNOTATION_PACKAGE_MANIFEST.json'),'Human source manifest differs')
    require(file_hash(export/'APPROVED_REFERENCES.json')==freeze['reference_file_sha256'],'Human aggregate reference file changed')
    refs,report=validate_all(package,export/'ANNOTATIONS',approved=True);require(not report['errors'],'100 independent human approvals required')
    aggregate=read(export/'APPROVED_REFERENCES.json')
    require(len(aggregate)==100 and len({r['business_id'] for r in aggregate})==100,'Missing/duplicate aggregate human references')
    require({r['business_id']:r for r in aggregate}=={r['business_id']:r for r in refs},'Aggregate and per-business human references differ')
    for r in refs:
        require(r.get('human_written') is not False and r.get('ai_generated') is not True and r.get('origin') in [None,'HUMAN_WRITTEN','HUMAN_REFERENCE'],'Nonhuman reference cannot enter primary track')
        require(freeze['content_hashes'][r['business_id']]==digest(reference_content(r)),'Human content freeze differs')
    return {r['business_id']:{'review_consensus':r['review_consensus'],'visual_evidence':r['visual_evidence'],'reference_sha256':digest(reference_content(r))} for r in refs},report

def metric_runtime(prep,model):
    prep,model=Path(prep),Path(model);lock=read(prep/'METRIC_LOCK.json');wheel=prep/'METRIC_ASSETS/bert_score-0.3.13-py3-none-any.whl'
    require(file_hash(wheel)=='bbbb4c7fcdaa46d7681aff49f37f96faa09ed74e1b150e659bdc6b58a66989b9','Frozen BERTScore wheel differs')
    if str(wheel) not in sys.path:sys.path.insert(0,str(wheel))
    require(all(importlib.metadata.version(k)==v for k,v in lock['packages'].items()),'Frozen metric package versions differ')
    require(file_hash(model/'pytorch_model.bin')==lock['bertscore']['weights_sha256'],'Frozen scorer weights differ')
    for name,h in lock['bertscore']['small_file_sha256'].items():require(file_hash(model/name)==h,'Frozen metric tokenizer/config differs')
    import torch
    from bert_score import BERTScorer
    from stage_k_bert_loader_compat_v1 import legacy_binary_cpu_load
    torch.set_num_threads(1);torch.manual_seed(42)
    with legacy_binary_cpu_load(model/'pytorch_model.bin'):
        scorer=BERTScorer(model_type=str(model),num_layers=17,batch_size=8,nthreads=1,all_layers=False,idf=False,device='cpu',lang='en',rescale_with_baseline=False,use_fast_tokenizer=False)
    scorer._tokenizer.model_max_length=512
    return scorer,{'packages':lock['packages'],'metric_lock_sha256':file_hash(prep/'METRIC_LOCK.json'),'weights_sha256':file_hash(model/'pytorch_model.bin'),
        'revision':lock['bertscore']['revision'],'device':'cpu','precision':'FP32','layer':17,'idf':False,'baseline_rescaling':False,'batch_size':8,'prefix_max_tokens':512}

def score_components(raw,calls,references,scorer,track):
    require(track in TRACK_NAMES,'Reference track')
    rows=[];pairs={}
    for r in raw:
        key=r['condition'],r['business_id'],r['task'];c=calls[key];ref=references[r['business_id']]
        candidate=strip_header(r['text'],r['task']);target=ref['review_consensus' if r['task']=='review' else 'visual_evidence']
        require(isinstance(target,str),'Reference component must be text, not a missing field')
        encode=lambda text:scorer._tokenizer.encode(text.strip(),add_special_tokens=True,add_prefix_space=True,truncation=False)
        cl,rl=len(encode(candidate)),len(encode(target));pair=(candidate,target)
        if pair not in pairs:pairs[pair]=len(pairs)
        expected='[REVIEW_CONSENSUS]\n' if r['task']=='review' else '[VISUAL_EVIDENCE]\n'
        foreign='[VISUAL_EVIDENCE]' if r['task']=='review' else '[REVIEW_CONSENSUS]'
        rows.append({'condition':r['condition'],'business_id':r['business_id'],'reference_business_id':r['business_id'],'task':r['task'],'reference_type':TRACK_NAMES[track],
            'output_sha256':r['text_sha256'],'reference_sha256':ref['reference_sha256'],'source_sha256':r['source_sha256'],'checkpoint_sha256':r['checkpoint_sha256'],
            'empty_candidate':not bool(candidate.strip()),'empty_reference':not bool(target.strip()),'malformed_header':not r['text'].startswith(expected),
            'foreign_component_marker':foreign in r['text'],'at_generation_cap':len(r['token_ids'])>=768,
            'bert_candidate_tokens_untruncated':cl,'bert_reference_tokens_untruncated':rl,'bert_candidate_prefix_truncated':cl>512,'bert_reference_prefix_truncated':rl>512,
            **descriptive_score(candidate,target,c['source']),'_pair':pair})
    # Cache by BOTH candidate and recipient reference, never by donor output alone.
    # This preserves exact F/S Review equality instead of separate padded batches.
    nonempty=[pair for pair in pairs if pair[0].strip() and pair[1].strip()];scores={pair:(0.,0.,0.) for pair in pairs}
    if nonempty:
        import torch
        with torch.no_grad():p,r,f=scorer.score([x[0] for x in nonempty],[x[1] for x in nonempty],batch_size=8,verbose=False)
        for i,pair in enumerate(nonempty):scores[pair]=(float(p[i]),float(r[i]),float(f[i]))
    for row in rows:
        pair=row.pop('_pair');p,r,f=scores[pair];require(all(math.isfinite(x) for x in [p,r,f]),'Nonfinite BERTScore')
        row.update(bertscore_precision=p,bertscore_recall=r,bertscore_f1=f)
    by={(r['condition'],r['business_id'],r['task']):r for r in rows}
    for b in references:
        if ('F',b,'review') in by and ('S',b,'review') in by:
            require(all(by['F',b,'review'][m]==by['S',b,'review'][m] for m in METRICS),'F/S Review scores differ')
    return rows,{'unique_candidate_reference_pairs':len(pairs),'nonempty_pairs_forwarded':len(nonempty),
        'empty_policy':'If either component is empty, ROUGE and BERTScore P/R/F are zero, including both-empty; retain every business.',
        'pair_cache':'Exact candidate text + recipient reference text; no donor-reference lookup','F_S_review_scores_identical':True}

def summarize(rows,ids,active,flagged_review_ids=(),bootstrap=20000,signflips=50000):
    by={(r['condition'],r['business_id'],r['task']):r for r in rows};aggregate=[];comparisons=[]
    for condition,task in [('RO','review'),('U','review'),('F','review'),('S','review'),('U','visual'),('F','visual'),('S','visual')]:
        rr=[by[condition,b,task] for b in ids]
        aggregate.append({'condition':condition,'task':task,'reference_type':rr[0]['reference_type'],'n':len(rr),**{m:sum(r[m] for r in rr)/len(rr) for m in METRICS},
            'candidate_prefix_truncated':sum(r['bert_candidate_prefix_truncated'] for r in rr),'reference_prefix_truncated':sum(r['bert_reference_prefix_truncated'] for r in rr),
            'empty_candidates':sum(r['empty_candidate'] for r in rr),'empty_references':sum(r['empty_reference'] for r in rr)})
    for task,other in CONTRASTS:
        for metric in METRICS:
            a=[by['F',b,task][metric] for b in ids];b=[by[other,b,task][metric] for b in ids];primary=metric=='bertscore_f1'
            result=paired(a,b,B=bootstrap,P=signflips if primary else 1)
            if not primary:
                result.pop('signflip_two_sided_p');result.pop('signflip_replicates')
            comparisons.append({'task':task,'contrast':'F-'+other,'metric':metric,'reference_type':rows[0]['reference_type'],
                'mean_F':sum(a)/len(a),'mean_comparator':sum(b)/len(b),'inference':'Prespecified primary BERTScore F1 test' if primary else 'Secondary descriptive paired CI',**result})
    primary=[r for r in comparisons if r['metric']=='bertscore_f1']
    for r,p in zip(primary,holm([r['signflip_two_sided_p'] for r in primary])):r['holm_p_four_prespecified_tests']=p
    sensitivity=[]
    groups=[('VISUAL_ACTUAL_INTERVENTION_98','visual','S',list(active)),('VISUAL_EXCLUDING_ZERO_F_CAPTIONS_98','visual','U',list(active))]
    clean=[b for b in ids if b not in set(flagged_review_ids)]
    if flagged_review_ids:
        groups += [('REVIEW_EXCLUDING_PREEXISTING_GENERATION_FLAGS','review',other,clean) for other in ['RO','U']]
    for label,task,other,group in groups:
        for metric in METRICS:
            result=paired([by['F',b,task][metric] for b in group],[by[other,b,task][metric] for b in group],B=bootstrap,P=1)
            result.pop('signflip_two_sided_p');result.pop('signflip_replicates')
            sensitivity.append({'label':label,'task':task,'contrast':'F-'+other,'metric':metric,'secondary_descriptive_only':True,
                'mean_F':sum(by['F',b,task][metric] for b in group)/len(group),'mean_comparator':sum(by[other,b,task][metric] for b in group)/len(group),**result})
    return aggregate,comparisons,sensitivity

def write_tables(out,aggregate,comparisons,track):
    title=TRACK_NAMES[track];header=['Condition','Task','n','ROUGE-1 F1','ROUGE-2 F1','ROUGE-L F1','BERT P','BERT R','BERT F1']
    md=[f'# {title}\n\nScores are macro means on a 0–1 scale. No Review/Visual composite is computed. RO Visual is N/A.\n','| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']
    tex=['\\begin{tabular}{llrllllll}','Condition & Task & n & R1 & R2 & RL & BERT P & BERT R & BERT F1 \\\\', '\\hline']
    for r in aggregate:
        cells=[r['condition'],r['task'],str(r['n'])]+[f'{r[m]:.4f}' for m in METRICS]
        md.append('| '+' | '.join(cells)+' |');tex.append(' & '.join(cells)+' \\\\')
    tex.append('\\end{tabular}')
    (out/'AUTOMATIC_METRIC_TABLES.md').write_text('\n'.join(md)+'\n',encoding='utf8');(out/'AUTOMATIC_METRIC_TABLES.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    write_csv(out/'AGGREGATE_METRICS.csv',aggregate)
    # JSON preserves differing primary/secondary inference fields without fake values.
    write(out/'PAIRED_COMPARISONS.json',comparisons)
    write_csv(out/'PAIRED_COMPARISONS.csv',[{k:r.get(k) for k in ['task','contrast','metric','reference_type','n','mean_F','mean_comparator','mean_difference','CI95','seed','bootstrap_replicates','inference','signflip_two_sided_p','holm_p_four_prespecified_tests']} for r in comparisons])
    paired_md=['# Paired business-level comparisons\n\nFull TEST100; F minus comparator. 95% percentile bootstrap, PCG64 seed 20261028, 20,000 replicates. Four prespecified BERT F1 tests use 50,000 sign flips and Holm adjustment; other intervals are descriptive.\n',
        '| Task | Contrast | Metric | n | Mean F | Mean comparator | Difference | 95% CI |','|---|---|---|---|---|---|---|---|']
    for r in comparisons:paired_md.append('| '+' | '.join([r['task'],r['contrast'],r['metric'],str(r['n']),f"{r['mean_F']:.4f}",f"{r['mean_comparator']:.4f}",f"{r['mean_difference']:.4f}",f"[{r['CI95'][0]:.4f}, {r['CI95'][1]:.4f}]"])+' |')
    (out/'PAIRED_COMPARISONS.md').write_text('\n'.join(paired_md)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True);p.add_argument('--stage-k',type=Path,required=True)
    p.add_argument('--track',choices=['human','teacher'],required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--bert-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--researcher-release',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Output must be new');require(a.researcher_release.is_file(),'Stage M researcher release missing')
    # Validate all 100 references BEFORE loading/forwarding a scoring model.
    if a.track=='human':refs,validation=human_references(a.preparation,a.references)
    else:
        from stage_m_teacher_references_v1 import load_teacher_references
        refs,validation=load_teacher_references(a.preparation,a.references)
    raw,calls,ids,active,proof=load_frozen_predictions(a.preparation,a.stage_k)
    require(set(refs)==set(ids),'References must cover exact complete TEST100')
    started=time.perf_counter();scorer,runtime=metric_runtime(a.preparation,a.bert_model)
    rows,cache=score_components(raw,calls,refs,scorer,a.track)
    flagged=sorted({r['business_id'] for r in rows if r['condition']=='F' and r['task']=='review' and (r['foreign_component_marker'] or r['malformed_header'] or r['at_generation_cap'])})
    aggregate,comparisons,sensitivity=summarize(rows,ids,active,flagged)
    a.output.mkdir(parents=True);write_csv(a.output/'PER_BUSINESS_COMPONENT_METRICS.csv',rows)
    write_tables(a.output,aggregate,comparisons,a.track);write(a.output/'SENSITIVITY_ANALYSIS.json',{'flagged_review_ids':flagged,'secondary_only':sensitivity,'full100_is_primary':True})
    write(a.output/'COMPLETION.json',{'status':'COMPLETED_MEASURED_TEST100','reference_type':TRACK_NAMES[a.track],'businesses':100,'condition_records':400,'components':700,
        'prediction_integrity':proof,'reference_validation':validation,'runtime':runtime,'score_cache':cache,'seconds':time.perf_counter()-started,
        'researcher_release_sha256':file_hash(a.researcher_release),'reference_files':{p.relative_to(a.references).as_posix():file_hash(p) for p in a.references.rglob('*.json')},
        'metric_file_sha256':file_hash(a.output/'PER_BUSINESS_COMPONENT_METRICS.csv'),'evaluator_sha256':file_hash(Path(__file__)),'no_training_or_inference':True,
        'limitations':['Reference similarity is not verified factuality/image grounding.','Fixed 512-token prefix BERTScore may omit suffixes; every row reports truncation.',
            'Pipeline ancestry/supervision and known original-image reuse remain confounders.','Human-authorship/independence attestations are not independent identity authentication.']})
    print('Completed '+TRACK_NAMES[a.track]+' only; reference tracks are never merged.')

if __name__=='__main__':main()
