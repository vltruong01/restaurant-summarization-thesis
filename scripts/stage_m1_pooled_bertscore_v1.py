"""Compute remaining exact Stage M pairs together; reuse completed RO scores.

Pooling lets the ORIGINAL BERTScore implementation deduplicate repeated reference
sentences internally. Metric versions/weights/FP32/layer/batch/CPU/prefix unchanged.
The original Stage M score_components consumes actual cached P/R/F, never proxies.
"""
from pathlib import Path
import contextlib
import csv
import math
import time
from stage_m1_existing_teacher_metrics_v1 import OUT,ROOT,PREP,load_targets,normalized_predictions,MODEL_ORDER,strip_header,score_components,metric_runtime
from s1000_final_eval_core_v1 import read,write,write_csv,file_hash,require,text_hash,digest

class ExactPairCache:
    def __init__(self,tokenizer,values):self._tokenizer=tokenizer;self.values=values
    def score(self,candidates,references,**kwargs):
        values=[self.values[c,r] for c,r in zip(candidates,references)]
        return [v[0] for v in values],[v[1] for v in values],[v[2] for v in values]

def main():
    require(not (OUT/'METRIC_COMPUTATION_COMPLETE.json').exists(),'Refuse rescore completed evaluation')
    data,_,audit=load_targets();raw,calls,refs,ids,proof=normalized_predictions(data)
    require(read(OUT/'PREDICTION_ALIGNMENT_AUDIT.json')==proof,'Input binding changed')
    rodir=OUT/'SCORED/RO';roreceipt=read(rodir/'COMPLETION.json');require(roreceipt['status']=='ACTUAL_SCORED_VALIDATION200' and roreceipt['per_business_sha256']==file_hash(rodir/'PER_BUSINESS_COMPONENT_METRICS.csv'),'Prior RO completion changed')
    ro=list(csv.DictReader((rodir/'PER_BUSINESS_COMPONENT_METRICS.csv').open(encoding='utf8')));require(len(ro)==200,'Complete RO rows required')
    models=[m for m in MODEL_ORDER if m!='RO'];pairs=[];seen=set()
    for m in models:
        for r in raw[m]:
            b,t=r['business_id'],r['task'];pair=(strip_header(r['text'],t),refs[m][b]['review_consensus' if t=='review' else 'visual_evidence'])
            if pair[0].strip() and pair[1].strip() and pair not in seen:pairs.append(pair);seen.add(pair)
    # This changes compute grouping only, not scientific scoring settings.
    separated_unique_sentences=sum(len({strip_header(r['text'],r['task']) for r in raw[m]}|{ref[k] for ref in refs[m].values() for k in ['review_consensus','visual_evidence']}) for m in models)
    pooled_unique_sentences=len({s for p in pairs for s in p})
    write(OUT/'COMPUTATION_ONLY_POOLING_AMENDMENT.json',{'reason':'Avoid repeatedly encoding identical references across conditions/checkpoints on the same CPU; independent of which score is larger.',
        'prior_RO_scores_reused':True,'prior_RO_receipt_sha256':file_hash(rodir/'COMPLETION.json'),'remaining_models':models,'distinct_nonempty_candidate_reference_pairs':len(pairs),
        'separate_call_unique_sentence_count_sum':separated_unique_sentences,'pooled_call_unique_sentences':pooled_unique_sentences,
        'metric_lock_sha256_unchanged':file_hash(PREP/'METRIC_LOCK.json'),'source_reference_and_prediction_hashes_unchanged':True,
        'batch_size':8,'CPU_threads':1,'precision':'FP32','implementation':'Original bert_score.BERTScorer.score plus exact candidate/reference pair result cache.',
        'not_new_predictions_targets_or_experiment':True,'known_numerical_limit':'Normal FP32 padding/batch roundoff can differ at sub-micro scale between pooled/separate calls; synthetic direct/cache equivalence is checked, all reports record actual execution grouping.'})
    scorer,runtime=metric_runtime(PREP,ROOT/'outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large');require(runtime==read(OUT/'METRIC_RUNTIME.json'),'Metric contract changed')
    # Genuine short CPU checks, no TEST references. Compare same algorithm/cached lookup.
    technical_candidates=['A red plate is on a table.','Soup is served in a bowl.'];technical_refs=['A red plate is on a table.','A bowl contains soup.']
    p,r,f=scorer.score(technical_candidates,technical_refs,batch_size=8,verbose=False)
    cache=ExactPairCache(scorer._tokenizer,{pair:(float(p[i]),float(r[i]),float(f[i])) for i,pair in enumerate(zip(technical_candidates,technical_refs))})
    cp,cr,cf=cache.score(technical_candidates[::-1],technical_refs[::-1]);require(all(abs(cf[i]-float(f[1-i]))<1e-12 for i in range(2)),'Actual CPU cache alignment failed')
    p2,r2,f2=scorer.score(technical_candidates[::-1],technical_refs[::-1],batch_size=8,verbose=False)
    require(all(abs(float(f2[i])-cf[i])<1e-6 for i in range(2)),'Original direct/pool ordering technical consistency failed')
    write(OUT/'POOLED_CPU_CACHE_TECHNICAL_VERIFICATION.json',{'status':'PASS_ACTUAL_CPU_SYNTHETIC','pairs':2,'exact_cache_reordering_tolerance':1e-12,'direct_original_BERTScore_reordering_max_difference':max(abs(float(f2[i])-cf[i]) for i in range(2)),
        'CPU_FP32_batch8_threads1_unchanged':True,'TEST_references_scored':0,'pool_script_sha256':file_hash(Path(__file__))})
    started=time.perf_counter();print(f'Original BERTScore pooled CPU call: {len(pairs)} pairs, {pooled_unique_sentences} unique sentences instead of {separated_unique_sentences}',flush=True)
    with (OUT/'POOLED_BERTSCORE_PROGRESS.log').open('w',encoding='utf8',buffering=1) as log,contextlib.redirect_stderr(log):
        p,r,f=scorer.score([x[0] for x in pairs],[x[1] for x in pairs],batch_size=8,verbose=True)
    values={pair:(float(p[i]),float(r[i]),float(f[i])) for i,pair in enumerate(pairs)};require(all(all(math.isfinite(x) for x in v) for v in values.values()),'Nonfinite pooled scores')
    write(OUT/'EXACT_POOLED_PAIR_SCORE_CACHE.json',[{'candidate_sha256':text_hash(pair[0]),'reference_sha256':text_hash(pair[1]),'precision':v[0],'recall':v[1],'f1':v[2]} for pair,v in values.items()])
    cached=ExactPairCache(scorer._tokenizer,values);allrows=ro
    for model in models:
        rows,cacheproof=score_components(raw[model],calls[model],refs[model],cached,'teacher');raw_lookup={(r['business_id'],r['task']):r for r in raw[model]}
        for row in rows:
            task,b=row['task'],row['business_id'];condition='unfiltered' if model=='U' else 'filtered';target=data['validation',condition][b];reference=refs[model][b]['review_consensus' if task=='review' else 'visual_evidence']
            row.update(split='validation',reference_type='EXISTING_AI_TEACHER_VALIDATION_TARGET_AGREEMENT',reference_condition='COMMON_RO_U_F_REVIEW' if task=='review' else condition,
                component_reference_sha256=text_hash(reference),full_target_sha256=text_hash(target['target']),target_artifact_sha256=target['target_artifact_sha256'],
                reference_scope='COMMON_IDENTICAL_REVIEW' if task=='review' else 'CONDITION_SPECIFIC_U_VISUAL' if model=='U' else 'COMMON_FILTERED_VISUAL_FOR_F_CE3_V1_500',model_role='CURRENT_SELECTED' if model in ['RO','U','F'] else 'HISTORICAL_SUPPLEMENTARY',
                parent_raw_output_sha256=raw_lookup[b,task]['parent_raw_output_sha256'],joint_extraction_flag=raw_lookup[b,task]['joint_extraction_flag'],output_hash_scope='EXTRACTED_COMPONENT_VIEW_WITH_EXPECTED_HEADER' if model=='CE3' else 'ORIGINAL_RAW_TASK_OUTPUT')
        directory=OUT/'SCORED'/model;directory.mkdir(parents=True);write_csv(directory/'PER_BUSINESS_COMPONENT_METRICS.csv',rows)
        write(directory/'COMPLETION.json',{'status':'ACTUAL_SCORED_VALIDATION200','businesses':200,'components':len(rows),'prediction_binding':next(p for p in proof if p['model']==model),'score_cache':cacheproof,
            'metric_lock_sha256':runtime['metric_lock_sha256'],'per_business_sha256':file_hash(directory/'PER_BUSINESS_COMPONENT_METRICS.csv'),'split':'validation','not_TEST_or_human_reference':True,
            'compute_grouping':'Pooled remaining models through original BERTScore; exact pair cache; no metric setting change'})
        allrows+=rows;print('Completed and saved '+model,flush=True)
    require(len(allrows)==1800,'Complete matrix required');write_csv(OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv',allrows)
    write(OUT/'METRIC_COMPUTATION_COMPLETE.json',{'status':'PASS_ACTUAL_VALIDATION200_ROUGE_AND_BERTSCORE','businesses':200,'model_records':1000,'components':1800,'selected_RO_U_F_components':1000,'historical_CE3_V1_500_components':800,
        'pooled_forward_and_serialization_seconds':time.perf_counter()-started,'prior_RO_result_reused':True,'prior_RO_seconds_from_runtime_receipt_to_completion':(rodir/'COMPLETION.json').stat().st_mtime-(OUT/'METRIC_RUNTIME.json').stat().st_mtime,
        'computation_grouping_amendment_sha256':file_hash(OUT/'COMPUTATION_ONLY_POOLING_AMENDMENT.json'),'metrics_sha256':file_hash(OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv'),
        'TEST_scored':0,'TRAIN_scored':0,'Teacher_calls':0,'LED_generation_calls':0})
    print('All1800 actual VALIDATION component metrics complete',flush=True)

if __name__=='__main__':main()
