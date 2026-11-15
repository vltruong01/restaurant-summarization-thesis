"""VALIDATION-only CPU resource amendment; metric definition/TEST lock unchanged.

Validate FP32 numeric consistency first, then score ALL five models on6 CPU
threads through the original BERTScore/Stage M implementation. No LED/Teacher
generation. Completed CPU1 RO measurements stay intact as a numerical audit.
"""
import contextlib
import csv
import time
from pathlib import Path
import psutil
from stage_m1_existing_teacher_metrics_v1 import ROOT,OUT,PREP,DATA,MODEL_ORDER,jsonl,normalized_predictions,metric_runtime,score_components,strip_header
from stage_m1_pooled_bertscore_v1 import ExactPairCache
from s1000_final_eval_core_v1 import read,write,write_csv,file_hash,text_hash,require

def main():
    require(not (OUT/'METRIC_COMPUTATION_COMPLETE.json').exists(),'Refuse repeat finished metrics')
    plan=read(OUT/'EVALUATION_PLAN_BEFORE_SCORING.json');data={}
    for c in ['reviews_only','unfiltered','filtered']:
        p=DATA/f'validation_{c}.jsonl';binding=next(r for r in plan['target_dataset_bindings'] if r['split']=='validation' and r['condition']==c)
        require(file_hash(p)==binding['sha256'],'Validated dataset changed');data['validation',c]={r['business_id']:r for r in jsonl(p)}
    raw,calls,refs,ids,proof=normalized_predictions(data);require(proof==read(OUT/'PREDICTION_ALIGNMENT_AUDIT.json'),'Prediction bindings changed')
    scorer,runtime=metric_runtime(PREP,ROOT/'outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large')
    import torch
    # Numerical/performance check on one already-approved VALIDATION component.
    # This is hardware scheduling, not metric/model/target selection.
    probe=raw['RO'][0];b=probe['business_id'];candidates=[strip_header(probe['text'],'review')];references=[refs['RO'][b]['review_consensus']]
    times={};values={}
    for threads in [1,6]:
        torch.set_num_threads(threads);start=time.perf_counter();p,r,f=scorer.score(candidates,references,batch_size=8,verbose=False)
        times[str(threads)]=time.perf_counter()-start;values[str(threads)]=[float(p[0]),float(r[0]),float(f[0])]
    difference=max(abs(a-b) for a,b in zip(values['1'],values['6']));require(difference<1e-5,'CPU resource change failed FP32 consistency; do not use amended execution')
    write(OUT/'CPU_THREAD_AMENDMENT_TECHNICAL_CHECK.json',{'status':'PASS_ACTUAL_VALIDATION_TECHNICAL_NUMERIC_CHECK','business_id':b,'candidate_sha256':text_hash(candidates[0]),'reference_sha256':text_hash(references[0]),
        'P_R_F_by_CPU_threads':values,'seconds_by_CPU_threads':times,'max_abs_P_R_F_difference':difference,'prespecified_numerical_tolerance':1e-5,
        'not_a_quality_comparison_or_new_prediction':True,'TEST_businesses_scored':0})
    # Stop only the earlier, incomplete CPU metric worker after the check passes.
    stopped=[]
    for process in psutil.process_iter(['pid','cmdline','name']):
        try:
            argv=process.info['cmdline'] or []
            if process.pid!=psutil.Process().pid and any(Path(s).name=='stage_m1_pooled_bertscore_v1.py' for s in argv):
                process.terminate();process.wait(timeout=30);stopped.append(process.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    write(OUT/'VALIDATION_ONLY_CPU_EXECUTION_AMENDMENT.json',{'reason':'Original one-thread CPU embedding batches took minutes and implied multi-hour evaluation. Use already available local CPU cores, not a new paid resource.',
        'torch_CPU_threads':6,'BERTScorer_IDF_tokenization_nthreads':1,'precision':'FP32','batch_size':8,'model_layers':17,'metric_versions_weights_tokenization_IDF_rescaling_prefix_unchanged':True,
        'TEST_METRIC_LOCK_file_unchanged_sha256':file_hash(PREP/'METRIC_LOCK.json'),'applies_to':'Retrospective supplementary VALIDATION200 only; all five final model rows computed under the SAME six-thread execution.',
        'prior_RO_CPU1_results_preserved_and_compared_not_overwritten':True,'stopped_only_incomplete_own_CPU_metric_PIDs':stopped,'new_Teacher_LED_training_or_human_work':False,
        'scientific_limit':'FP32 operation scheduling can cause tiny roundoff; actual numeric check and all200 RO CPU1-vs6 differences are reported. No metric definition or TEST protocol amendment.'})
    torch.set_num_threads(6);runtime={**runtime,'torch_CPU_threads':6,'BERTScorer_nthreads':1,'execution_amendment':'VALIDATION_ONLY_CPU_EXECUTION_AMENDMENT.json'}
    write(OUT/'ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json',runtime)
    pairs=[];seen=set()
    for m in MODEL_ORDER:
        for row in raw[m]:
            b,t=row['business_id'],row['task'];pair=(strip_header(row['text'],t),refs[m][b]['review_consensus' if t=='review' else 'visual_evidence'])
            if pair[0].strip() and pair[1].strip() and pair not in seen:pairs.append(pair);seen.add(pair)
    write(OUT/'FINAL_CPU6_POOLED_INPUT_FRAME.json',{'models':MODEL_ORDER,'component_records':1800,'unique_nonempty_text_reference_pairs':len(pairs),'unique_embedding_sentences':len({s for p in pairs for s in p}),
        'all200_business_ids_per_model':True,'same_original_Stage_M_preprocessing_and_scoring':True})
    print(f'Final VALIDATION BERTScore: all five models, {len(pairs)} actual pairs, CPU FP32 threads6, batch8',flush=True);start=time.perf_counter()
    with (OUT/'CPU6_POOLED_BERTSCORE_PROGRESS.log').open('w',encoding='utf8',buffering=1) as log,contextlib.redirect_stderr(log):
        p,r,f=scorer.score([x[0] for x in pairs],[x[1] for x in pairs],batch_size=8,verbose=True)
    values={pair:(float(p[i]),float(r[i]),float(f[i])) for i,pair in enumerate(pairs)};cache=ExactPairCache(scorer._tokenizer,values);allrows=[]
    for model in MODEL_ORDER:
        rows,cacheproof=score_components(raw[model],calls[model],refs[model],cache,'teacher');lookup={(x['business_id'],x['task']):x for x in raw[model]}
        for row in rows:
            b,t=row['business_id'],row['task'];c='reviews_only' if model=='RO' else 'unfiltered' if model=='U' else 'filtered';target=data['validation',c][b];reference=refs[model][b]['review_consensus' if t=='review' else 'visual_evidence']
            row.update(split='validation',reference_type='EXISTING_AI_TEACHER_VALIDATION_TARGET_AGREEMENT',reference_condition='COMMON_RO_U_F_REVIEW' if t=='review' else c,
                component_reference_sha256=text_hash(reference),full_target_sha256=text_hash(target['target']),target_artifact_sha256=target['target_artifact_sha256'],
                reference_scope='COMMON_IDENTICAL_REVIEW' if t=='review' else 'CONDITION_SPECIFIC_U_VISUAL' if model=='U' else 'COMMON_FILTERED_VISUAL_FOR_F_CE3_V1_500',model_role='CURRENT_SELECTED' if model in ['RO','U','F'] else 'HISTORICAL_SUPPLEMENTARY',
                parent_raw_output_sha256=lookup[b,t]['parent_raw_output_sha256'],joint_extraction_flag=lookup[b,t]['joint_extraction_flag'],output_hash_scope='EXTRACTED_COMPONENT_VIEW_WITH_EXPECTED_HEADER' if model=='CE3' else 'ORIGINAL_RAW_TASK_OUTPUT',execution_torch_CPU_threads=6)
        folder=OUT/'FINAL_SCORED'/model;folder.mkdir(parents=True);write_csv(folder/'PER_BUSINESS_COMPONENT_METRICS.csv',rows)
        write(folder/'COMPLETION.json',{'status':'ACTUAL_SCORED_VALIDATION200','businesses':200,'components':len(rows),'prediction_binding':next(x for x in proof if x['model']==model),'score_cache':cacheproof,
            'per_business_sha256':file_hash(folder/'PER_BUSINESS_COMPONENT_METRICS.csv'),'runtime_sha256':file_hash(OUT/'ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json'),'split':'validation','not_TEST_or_human_reference':True})
        allrows+=rows;print('Final scores saved '+model,flush=True)
    # Validate hardware grouping against the complete previously computed CPU1 RO.
    old=list(csv.DictReader((OUT/'SCORED/RO/PER_BUSINESS_COMPONENT_METRICS.csv').open(encoding='utf8')));old={x['business_id']:x for x in old}
    deltas=[]
    for row in allrows:
        if row['condition']=='RO':
            earlier=old[row['business_id']];require(row['component_reference_sha256']==earlier['component_reference_sha256'] and row['output_sha256']==earlier['output_sha256'],'RO control alignment changed')
            deltas.append(max(abs(row[m]-float(earlier[m])) for m in ['bertscore_precision','bertscore_recall','bertscore_f1']))
    require(len(deltas)==200 and max(deltas)<1e-5,'CPU1-vs6 whole-RO numerical consistency failed; investigate, do not publish')
    write(OUT/'ALL200_RO_CPU1_VS_CPU6_NUMERIC_AUDIT.json',{'status':'PASS','businesses':200,'max_abs_P_R_F_difference':max(deltas),'mean_max_abs_difference':sum(deltas)/200,'tolerance':1e-5,'all_source_output_reference_hashes_identical':True})
    require(len(allrows)==1800,'Incomplete final matrix');write_csv(OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv',allrows)
    write(OUT/'METRIC_COMPUTATION_COMPLETE.json',{'status':'PASS_ACTUAL_VALIDATION200_ROUGE_AND_BERTSCORE','businesses':200,'model_records':1000,'components':1800,'selected_RO_U_F_components':1000,'historical_CE3_V1_500_components':800,
        'final_CPU6_pool_seconds':time.perf_counter()-start,'all_final_models_same_six_thread_CPU_FP32_runtime':True,'prior_RO_CPU1_control_preserved':True,
        'metrics_sha256':file_hash(OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv'),'TEST_scored':0,'TRAIN_scored':0,'Teacher_calls':0,'LED_generation_calls':0})
    print('COMPLETE actual1800 VALIDATION component metrics, including BERTScore',flush=True)

if __name__=='__main__':main()
