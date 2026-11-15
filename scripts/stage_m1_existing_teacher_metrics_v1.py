"""Existing TRAIN/VALIDATION target audit and actual frozen VALIDATION scoring.

No generation, paid API, training or human-reference modification is implemented.
Reuses Stage M metric runtime, component scoring and the prespecified bootstrap.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
from s1000_final_eval_core_v1 import read,write,write_csv,file_hash,text_hash,digest,require,isolated,paired
from stage_m_evaluation_v1 import metric_runtime,score_components,METRICS,strip_header

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_m1/existing_teacher_validation200_v1'
PREP=ROOT/'outputs/final_evaluation/s1000_preparation_v1'
DATA=ROOT/'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset'
MATCHED=ROOT/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1'
HIST=ROOT/'outputs/stage_i/led_large_filtered_validation200_selection_v1'
MODEL_ORDER=['RO','U','F','CE3','V1_500']
SEPARATOR='\n\n[VISUAL_EVIDENCE]\n'

def jsonl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def target_parts(text,condition):
    require(text.startswith('[REVIEW_CONSENSUS]\n'),'Original target Review header')
    if condition=='reviews_only':
        require('[VISUAL_EVIDENCE]' not in text,'RO target cannot contain Visual');return strip_header(text,'review'),''
    require(text.count(SEPARATOR)==1,'Original target component boundary')
    review,visual=text.split(SEPARATOR);return strip_header(review,'review'),visual

def joint_parts(text):
    # Same conservative fallback as the existing validation diagnostic sections().
    # Retain malformed whole output as Review and an empty Visual; no repair/retry.
    if text.count('[REVIEW_CONSENSUS]')!=1 or text.count('[VISUAL_EVIDENCE]')!=1:return text,'','INVALID_HEADERS_WHOLE_TEXT_REVIEW_EMPTY_VISUAL'
    parts=text.split('[REVIEW_CONSENSUS]',1)[1].split('[VISUAL_EVIDENCE]',1)
    return parts[0].strip(),parts[1].strip(),None

def load_targets():
    splitfile=ROOT/'data/processed/yelp_2300_v1/business_split.csv';splits={r['business_id']:r['split'] for r in csv.DictReader(splitfile.open(encoding='utf8'))}
    completion=read(DATA.parent/'completion_report.json');require(completion['status']=='STAGE_G_COMPLETE_VALIDATED_WITH_DOCUMENTED_LIMITATIONS' and completion['targets']==6600,'Canonical Stage G completion')
    datasets={};ledger=[];artifacts={};inventory=[]
    for split,n in [('train',2000),('validation',200)]:
        for condition in ['reviews_only','unfiltered','filtered']:
            p=DATA/f'{split}_{condition}.jsonl';rows=jsonl(p);h=file_hash(p)
            require(completion['dataset_sha256'][p.relative_to(ROOT).as_posix()]==h,'Teacher dataset changed since canonical completion')
            require(len(rows)==n and len({r['business_id'] for r in rows})==n,'Teacher cohort duplicate/missing')
            require({r['business_id'] for r in rows}=={b for b,s in splits.items() if s==split},'Dataset/split identity')
            for r in rows:
                require(r['split']==split and r['condition']==condition and text_hash(r['source'])==r['source_sha256'],'Teacher source/split/condition changed')
                require(r['target_content_origin']=='teacher_API_only_frozen_serialization' and r['independent_human_verified'] is False,'Teacher origin changed')
                path=(ROOT/r['target_artifact']).resolve();require(path.is_relative_to(ROOT.resolve()),'Foreign Teacher artifact')
                if path not in artifacts:artifacts[path]=(read(path),file_hash(path))
                original,ah=artifacts[path]
                require(ah==r['target_artifact_sha256'] and original['business_id']==r['business_id'] and original['split']==split and original['targets'][condition]==r['target'],'Original target artifact binding')
                review,visual=target_parts(r['target'],condition)
                ledger.append({'business_id':r['business_id'],'split':split,'condition':condition,'source_sha256':r['source_sha256'],'full_target_sha256':text_hash(r['target']),
                    'review_reference_sha256':text_hash(review),'visual_reference_sha256':None if condition=='reviews_only' else text_hash(visual),'empty_visual_reference':not visual if condition!='reviews_only' else None,
                    'target_artifact':r['target_artifact'],'target_artifact_sha256':ah,'artifact_status':original.get('status'),'imported_D041_without_regeneration':r.get('imported_D041_without_regeneration',False),
                    'origin':'EXISTING_AI_TEACHER_NOT_HUMAN','reference_scope':'COMMON_REVIEW_OWN_CONDITION_VISUAL'})
            datasets[split,condition]={r['business_id']:r for r in rows}
            inventory.append({'file':p.relative_to(ROOT).as_posix(),'sha256':h,'split':split,'condition':condition,'records':n})
    equivalence=[]
    for split,n in [('train',2000),('validation',200)]:
        equal_review=equal_visual=0
        for b in datasets[split,'filtered']:
            parts={c:target_parts(datasets[split,c][b]['target'],c) for c in ['reviews_only','unfiltered','filtered']}
            equal_review+=parts['reviews_only'][0]==parts['unfiltered'][0]==parts['filtered'][0]
            equal_visual+=parts['unfiltered'][1]==parts['filtered'][1]
        require(equal_review==n,'Review references cannot be treated as common')
        equivalence.append({'split':split,'businesses':n,'RO_U_F_Review_byte_identical':equal_review,'U_F_Visual_byte_identical':equal_visual,'U_F_Visual_different':n-equal_visual})
    audit={'status':'PASS_6600_CANONICAL_TEACHER_TARGET_RECORDS','records':len(ledger),'unique_businesses':2200,'unique_original_target_artifacts_rehashed':len(artifacts),
        'dataset_inventory':inventory,'equivalence':equivalence,'completion_sha256':file_hash(DATA.parent/'completion_report.json'),'business_split_sha256':file_hash(splitfile),
        'TRAIN_and_VALIDATION_disjoint':True,'TEST_target_records_in_canonical_Stage_G':0,'artifact_status_counts':dict(Counter(a[0].get('status','') for a in artifacts.values())),
        'provisional_D041_note':'Some original artifact status strings predate canonical production acceptance; exact imports are bound by final Stage G completion/dataset hashes. None is human-verified.',
        'TRAIN_scored':False,'TEST_reference_status':'PENDING — no matching references available; workspace search separately recorded'}
    return datasets,ledger,audit

def normalized_predictions(datasets):
    from tokenizers import Tokenizer
    lock=read(PREP/'MODEL_AND_DECODER_LOCK.json');historical=read(HIST/'OFFLINE_BINDINGS.json');ids=sorted(datasets['validation','filtered'])
    require(set(ids)==set(read(HIST/'COHORT_MANIFEST.json')['business_ids']),'Historical validation cohort differs')
    tokenizer=ROOT/'assets/tokenizers/led-base-16384/tokenizer.json';require(file_hash(tokenizer)==lock['tokenizer']['sha256']['tokenizer.json'],'LED tokenizer drift');tok=Tokenizer.from_file(str(tokenizer))
    files={'RO':MATCHED/'RECOVERY_R1/DOWNLOADED/VALIDATION_RESULTS/RO/predictions.json','U':MATCHED/'RECOVERY_R1/DOWNLOADED/VALIDATION_RESULTS/U/predictions.json','F':HIST/'MERGED/V1_1000/predictions.json',
           'CE3':HIST/'MERGED/CE3/predictions.json','V1_500':HIST/'MERGED/V1_500/predictions.json'}
    allraw={};allcalls={};refs={};proof=[]
    for model,p in files.items():
        rows=read(p);require(len(rows)==200 and {r['business_id'] for r in rows}==set(ids),'Incomplete/duplicate VALIDATION predictions')
        condition='reviews_only' if model=='RO' else 'unfiltered' if model=='U' else 'filtered'
        expected=lock['models'][model]['adapter_sha256'] if model in lock['models'] else historical['checkpoint_bindings'][model]['adapter_sha256']
        if model in lock['models']:require(file_hash(p)==lock['models'][model]['predictions_sha256'],'Selected validation output file changed')
        raw=[];calls={};references={};joint_flags=[]
        for r in rows:
            b=r['business_id'];source=datasets['validation',condition][b];reviewref,visualref=target_parts(source['target'],condition)
            references[b]={'review_consensus':reviewref,'visual_evidence':visualref,'reference_sha256':digest([reviewref,visualref])}
            review_source='[TASK_REVIEW_CONSENSUS]\n'+datasets['validation','reviews_only'][b]['source']
            visual_source='' if condition=='reviews_only' else isolated(source['source'])[1]
            if model=='CE3':
                require(set(r['passes'])=={'CE3'},'CE3 joint pass identity');g=r['passes']['CE3'];parts=joint_parts(g['text']);joint_flags.append({'business_id':b,'flag':parts[2]})
                virtual=[('review',parts[0],g,source['source']),('visual',parts[1],g,source['source'])]
            else:
                require(set(r['passes'])==({'review'} if model=='RO' else {'review','visual'}),'Source-separated pass frame')
                virtual=[(t,g['text'],g,review_source if t=='review' else visual_source) for t,g in r['passes'].items()]
            for task,text,g,src in virtual:
                require(g['business_id']==b and g['checkpoint_sha256']==expected,'Prediction business/checkpoint identity')
                require(g['source_sha256']==text_hash(src) and g['input_ids_sha256']==digest(tok.encode(src).ids),'Frozen prediction source/input tokens differ from correct Teacher condition')
                require(text_hash(g['text'])==g['text_sha256'] and tok.decode(g['token_ids'],skip_special_tokens=True)==g['text'],'Saved raw token/text differs')
                rr={**g,'condition':model,'task':task,'text':text,'text_sha256':text_hash(text)}
                rr['parent_raw_output_sha256']=g['text_sha256'];rr['joint_extraction_flag']=parts[2] if model=='CE3' else None
                # CE3 component views are exact extracted spans, not regenerated text.
                if model=='CE3':
                    rr['text']=('[REVIEW_CONSENSUS]\n' if task=='review' else '[VISUAL_EVIDENCE]\n')+text;rr['text_sha256']=text_hash(rr['text'])
                raw.append(rr);calls[model,b,task]={'source':src}
            expected_combined=r['passes']['CE3']['text'] if model=='CE3' else '\n\n'.join(r['passes'][t]['text'] for t in ['review','visual'] if t in r['passes'])
            require(r['combined_text']==expected_combined,'Combined output rewritten')
        allraw[model]=raw;allcalls[model]=calls;refs[model]=references
        proof.append({'model':model,'role':'CURRENT_SELECTED_RO_U_F' if model in ['RO','U','F'] else 'HISTORICAL_SAME_COHORT_SUPPLEMENTARY','prediction_file':p.relative_to(ROOT).as_posix(),'prediction_sha256':file_hash(p),
            'businesses':200,'components':len(raw),'checkpoint_sha256':expected,'Teacher_reference_condition':condition,'split':'validation','all_raw_token_text_and_source_input_hashes_verified':True,
            'joint_extraction':'Existing sections() policy; retain invalid whole text as Review and empty Visual; raw parent output untouched' if model=='CE3' else None,'CE3_joint_header_flags':[x for x in joint_flags if x['flag']]})
    return allraw,allcalls,refs,ids,proof

def prepare():
    require(not OUT.exists(),'New output required');OUT.mkdir(parents=True)
    data,ledger,audit=load_targets();write_csv(OUT/'ALL6600_TARGET_PROVENANCE_LEDGER.csv',ledger);write(OUT/'TARGET_AND_SPLIT_AUDIT.json',audit)
    raw,calls,refs,ids,proof=normalized_predictions(data);write(OUT/'PREDICTION_ALIGNMENT_AUDIT.json',proof)
    write(OUT/'COHORT200.json',{'split':'validation','business_ids':ids,'selection':'All frozen200 validation IDs; no exclusions','TEST_business_ids_disjoint':True})
    plan={'models':MODEL_ORDER,'split':'validation','current_family':['RO','U','F'],'historical_supplementary':['CE3','V1_500'],'condition_records':1000,'component_pairs':1800,
        'reference_contract':'Common byte-identical Review; own-condition Visual. No U-vs-F Visual inferential comparison with differing references.',
        'metric_lock_sha256':file_hash(PREP/'METRIC_LOCK.json'),'models_provenance':proof,'target_dataset_bindings':audit['dataset_inventory'],
        'primary_analysis':'Descriptive paired200-business bootstrap95% CI seed20261028 B20000; no new confirmatory p-values or winner selection.',
        'empty_policy':'Same Stage M: either empty -> six zeros, including both-empty. Retain all200; report empty and truncation counts.',
        'TRAIN_metrics':'Not calculated; no matched canonical full-TRAIN prediction cohort in this evaluation family. TRAIN targets audited, not independent evaluation.',
        'TEST_metrics':'PENDING — no matching references available','new_LED_or_Teacher_generation':False}
    write(OUT/'EVALUATION_PLAN_BEFORE_SCORING.json',plan)
    print('PASS actual6600 targets and1800 VALIDATION prediction-component bindings; evaluation plan frozen',flush=True)

class ProgressScorer:
    def __init__(self,scorer):self.scorer=scorer;self._tokenizer=scorer._tokenizer
    def score(self,*args,**kwargs):
        print(f'BERTScore CPU forward: {len(args[0])} distinct nonempty pairs',flush=True)
        return self.scorer.score(*args,**kwargs)

def score():
    require((OUT/'EVALUATION_PLAN_BEFORE_SCORING.json').is_file(),'Freeze audited inputs first')
    data,_,audit=load_targets();raw,calls,refs,ids,proof=normalized_predictions(data)
    require(read(OUT/'PREDICTION_ALIGNMENT_AUDIT.json')==proof,'Prediction bindings changed after plan freeze')
    scorer,runtime=metric_runtime(PREP,ROOT/'outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large');write(OUT/'METRIC_RUNTIME.json',runtime)
    started=time.perf_counter();allrows=[]
    for model in MODEL_ORDER:
        print('Scoring VALIDATION200 '+model,flush=True)
        rows,cache=score_components(raw[model],calls[model],refs[model],ProgressScorer(scorer),'teacher')
        raw_lookup={(r['business_id'],r['task']):r for r in raw[model]}
        for row in rows:
            task=row['task'];b=row['business_id'];condition='reviews_only' if model=='RO' else 'unfiltered' if model=='U' else 'filtered';target=data['validation',condition][b]
            reference=refs[model][b]['review_consensus' if task=='review' else 'visual_evidence']
            row.update(split='validation',reference_type='EXISTING_AI_TEACHER_VALIDATION_TARGET_AGREEMENT',reference_condition='COMMON_RO_U_F_REVIEW' if task=='review' else condition,
                component_reference_sha256=text_hash(reference),full_target_sha256=text_hash(target['target']),target_artifact_sha256=target['target_artifact_sha256'],
                reference_scope='COMMON_IDENTICAL_REVIEW' if task=='review' else 'CONDITION_SPECIFIC_U_VISUAL' if model=='U' else 'COMMON_FILTERED_VISUAL_FOR_F_CE3_V1_500',
                model_role='CURRENT_SELECTED' if model in ['RO','U','F'] else 'HISTORICAL_SUPPLEMENTARY',
                parent_raw_output_sha256=raw_lookup[b,task]['parent_raw_output_sha256'],joint_extraction_flag=raw_lookup[b,task]['joint_extraction_flag'],
                output_hash_scope='EXTRACTED_COMPONENT_VIEW_WITH_EXPECTED_HEADER' if model=='CE3' else 'ORIGINAL_RAW_TASK_OUTPUT')
        directory=OUT/'SCORED'/model;directory.mkdir(parents=True);write_csv(directory/'PER_BUSINESS_COMPONENT_METRICS.csv',rows)
        write(directory/'COMPLETION.json',{'status':'ACTUAL_SCORED_VALIDATION200','businesses':200,'components':len(rows),'prediction_binding':next(p for p in proof if p['model']==model),'score_cache':cache,
            'metric_lock_sha256':runtime['metric_lock_sha256'],'per_business_sha256':file_hash(directory/'PER_BUSINESS_COMPONENT_METRICS.csv'),'split':'validation','not_TEST_or_human_reference':True})
        allrows+=rows;print('Completed '+model,flush=True)
    require(len(allrows)==1800,'Complete available evaluation matrix required')
    write_csv(OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv',allrows)
    write(OUT/'METRIC_COMPUTATION_COMPLETE.json',{'status':'PASS_ACTUAL_VALIDATION200_ROUGE_AND_BERTSCORE','businesses':200,'model_records':1000,'components':1800,
        'selected_RO_U_F_components':1000,'historical_CE3_V1_500_components':800,'seconds_after_model_load':time.perf_counter()-started,
        'metrics_sha256':file_hash(OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv'),'TEST_scored':0,'TRAIN_scored':0,'Teacher_calls':0,'LED_generation_calls':0})

def report():
    require(read(OUT/'METRIC_COMPUTATION_COMPLETE.json')['status'].startswith('PASS_ACTUAL'),'No measured metrics completion')
    rows=list(csv.DictReader((OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv').open(encoding='utf8')));ids=read(OUT/'COHORT200.json')['business_ids'];by={(r['condition'],r['business_id'],r['task']):r for r in rows};aggregate=[];comparisons=[]
    for model in MODEL_ORDER:
        for task in ['review','visual']:
            rr=[r for r in rows if r['condition']==model and r['task']==task]
            if not rr:continue
            require(len(rr)==200,'Do not silently drop validation cases')
            aggregate.append({'model':model,'model_role':rr[0]['model_role'],'task':task,'split':'validation','n':200,'reference_scope':rr[0]['reference_scope'],
                **{m:sum(float(r[m]) for r in rr)/200 for m in METRICS},
                **{m:sum(r[m]=='True' for r in rr) for m in ['empty_candidate','empty_reference','bert_candidate_prefix_truncated','bert_reference_prefix_truncated','at_generation_cap']}})
    for task,A,B in [('review','F','RO'),('review','F','U'),('review','F','CE3'),('review','F','V1_500'),('visual','F','CE3'),('visual','F','V1_500')]:
        require(all(by[A,b,task]['component_reference_sha256']==by[B,b,task]['component_reference_sha256'] for b in ids),'Paired comparison references differ')
        for m in METRICS:
            a=[float(by[A,b,task][m]) for b in ids];v=[float(by[B,b,task][m]) for b in ids];result=paired(a,v,B=20000,P=1);result.pop('signflip_two_sided_p');result.pop('signflip_replicates')
            comparisons.append({'task':task,'contrast':A+'-'+B,'metric':m,'split':'validation','same_reference_verified_all_pairs':True,'reference_scope':by[A,ids[0],task]['reference_scope'],
                'mean_A':sum(a)/200,'mean_B':sum(v)/200,'analysis':'Descriptive paired percentile95% bootstrap, not new confirmatory tests or model selection',**result})
    write_csv(OUT/'AGGREGATE_VALIDATION200_METRICS.csv',aggregate);write(OUT/'PAIRED_SAME_REFERENCE_COMPARISONS.json',comparisons)
    write_csv(OUT/'PAIRED_SAME_REFERENCE_COMPARISONS.csv',comparisons)
    for task in ['review','visual']:
        md=[f'# Actual VALIDATION200 {task} — existing AI Teacher target agreement\n\nAll200 businesses retained. Scores0–1. No TEST or human-reference scores. RO Visual N/A.\n','| Model | n | Reference | ROUGE-1 | ROUGE-2 | ROUGE-L | BERT P | BERT R | BERT F1 |','|---|---|---|---|---|---|---|---|---|']
        tex=['\\begin{tabular}{lrlllllll}','Model & n & Reference & R1 & R2 & RL & BERT P & BERT R & BERT F1 \\\\','\\hline']
        for r in aggregate:
            if r['task']!=task:continue
            label='Common Review' if task=='review' else 'U-specific' if r['model']=='U' else 'Filtered-specific'
            cells=[r['model'],'200',label]+[f'{r[m]:.4f}' for m in METRICS];md.append('| '+' | '.join(cells)+' |');tex.append(' & '.join(cells).replace('V1_500','V1\\_500')+' \\\\')
        tex.append('\\end{tabular}');(OUT/f'{task.upper()}_VALIDATION200_TABLE.md').write_text('\n'.join(md)+'\n',encoding='utf8');(OUT/f'{task.upper()}_VALIDATION200_TABLE.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    pairedmd=['# Same-reference descriptive paired comparisons\n\nAll200 businesses, F minus comparator;95% percentile bootstrap,20,000 replicates, seed20261028. No U-vs-F Visual test: targets differ for all200. No new p-values.\n','| Task | Contrast | Metric | Mean difference | 95% CI | n |','|---|---|---|---|---|---|']
    for r in comparisons:pairedmd.append(f"| {r['task']} | {r['contrast']} | {r['metric']} | {r['mean_difference']:.4f} | [{r['CI95'][0]:.4f}, {r['CI95'][1]:.4f}] | {r['n']} |")
    (OUT/'PAIRED_SAME_REFERENCE_COMPARISONS.md').write_text('\n'.join(pairedmd)+'\n',encoding='utf8')
    # Verify prior ROUGE where body/header/empty policies genuinely agree.
    old=read(MATCHED/'COMPARISON/ALL200_AUTOMATED_DIAGNOSTICS.json');checks=[]
    for r in old:
        current=by[r['condition'],r['business_id'],r['task']];prior=r['reference_similarity']
        differences=None if prior is None else {name:float(current[name+'_fmeasure'])-prior[name] for name in ['rouge1','rouge2','rougeL']}
        checks.append({'model':r['condition'],'business_id':r['business_id'],'task':r['task'],'output_hash_identical':r['text_sha256']==current['output_sha256'],
            'prior_reference_empty_excluded':prior is None,'absolute_differences':differences,'all_three_reproduced':differences is not None and all(abs(x)<1e-12 for x in differences.values())})
    write(OUT/'PRIOR_ROUGE_REUSE_VERIFICATION.json',{'policy':'Prior valid ROUGE verified; cheap full-text recomputation cross-checks hashes/values. Current Stage M retains empty-reference rows as0 instead of excluding them.',
        'rows':len(checks),'all_three_scores_reproduced':sum(r['all_three_reproduced'] for r in checks),'prior_empty_reference_excluded':sum(r['prior_reference_empty_excluded'] for r in checks),'checks':checks})
    print(json.dumps(aggregate,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','score','report']);a=p.parse_args();globals()[a.command]()

if __name__=='__main__':main()
