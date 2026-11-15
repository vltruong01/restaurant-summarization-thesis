"""Complete the predeclared pilot task inventory once, preserving failures without retries/promotion."""
from pathlib import Path
import json
from collections import Counter

from prepare_stage_g_teacher_pilot import ROOT, read, local_path, file_sha, digest, canonical, immutable
from run_stage_g_teacher_pilot_v4_1_recovery import credentials, job_result, visual_claims

CONFIG=ROOT/'configs/stage_g_teacher_pilot_v4_1_grounding.json'


def run():
    cfg=read(CONFIG);folder=ROOT/cfg['output_directory'];out=folder/'execution_v2_quote_recovery'
    prepared=read(folder/'preparation_report.json')
    for relative,expected in prepared['input_sha256'].items():
        if file_sha(ROOT/relative)!=expected:raise ValueError('Frozen source/protocol changed')
    jobs=read(folder/'pilot_jobs.json');businesses=read(folder/'pilot_businesses.json')
    assert len(businesses)==20 and {b['split'] for b in businesses}=={'train','validation'}
    assert digest(canonical(jobs))==prepared['jobs_sha256']
    immutable(out/'collection_control_manifest.json',{
        'collector_sha256':file_sha(Path(__file__)), 'config_sha256':file_sha(CONFIG),
        'reason':'Finish all60predeclared task slots, keep failures failed; no further retry or target repair',
        'generation_functions':'unchanged run_stage_g_teacher_pilot_v4_1_recovery.job_result',
        'existing_failure_regenerated':False,'failed_task_promoted':False,'automatic_retries':0,
        'model_settings_schema_prompts_inputs_unchanged':True,'production_generation':False})
    base,provider,key=credentials();schemas=read(ROOT/cfg['schema'])
    prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    failures=[];results=[];completed=[]
    for business in businesses:
        outputs={};bid=business['business_id']
        for job in [j for j in jobs if j['business_id']==bid]:
            path=out/'checkpoints'/(job['job_id']+'.json')
            if path.exists() and read(path)['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                failures.append({'job_id':job['job_id'],'business_id':bid,'failure':read(path)})
                print('Preserved failed task; no retry: '+job['job_id'],flush=True)
                continue
            try:
                cp=job_result(job,cfg,prompts[job['task']],schemas[job['task']],base,provider,key,out,CONFIG)
            except RuntimeError:
                # The unchanged producer already persisted failure and original raw response.
                failures.append({'job_id':job['job_id'],'business_id':bid,'failure':read(path)})
                print('Rejected task retained; continuing independent planned tasks only: '+job['job_id'],flush=True)
                continue
            results.append(cp);outputs[job['condition']]=cp.get('resolved_output',cp['output'])
        if set(outputs)!={'shared','unfiltered','filtered'}:
            continue
        consensus=' '.join(c['text'] for c in outputs['shared']['claims'])
        targets={'reviews_only':'[REVIEW_CONSENSUS]\n'+consensus};evidence={}
        for condition in ['unfiltered','filtered']:
            evidence[condition]=visual_claims(outputs[condition]['assessments'])
            targets[condition]=targets['reviews_only']+'\n\n[VISUAL_EVIDENCE]\n'+' '.join(c['text'] for c in evidence[condition])
        immutable(out/'provisional_targets'/(bid+'.json'),{
            'status':'PROVISIONAL_PILOT_TARGETS_NOT_HUMAN_VERIFIED_NOT_PRODUCTION',
            'business_id':bid,'split':business['split'],'targets':targets,
            'evidence_records':outputs,'human_verified':False,
            'review_quote_origin':'code_resolved_from_source_span_IDs',
            'visual_target_claims_with_all_evidence':evidence,
            'empty_review_consensus_flag':not consensus,'review_IDs_text_order_shared':True,
            'zero_filtered_edge_case':business['zero_filtered_edge_case']})
        completed.append(business)
    api=[r for r in results if r['api_called']]
    report={'status':'ALL20_PILOT_BUSINESSES_ATTEMPTED_WITH_REJECTED_TASKS' if failures else 'ALL20_PILOT_STRUCTURAL_PASS',
        'businesses_attempted':20,'businesses_with_valid_three_condition_targets':len(completed),
        'valid_target_texts':len(completed)*3,'successful_API_task_records':len(api),
        'failed_task_count':len(failures),'failed_tasks':failures,
        'task_slots_accounted_for':len(results)+len(failures),
        'deterministic_empty_tasks':sum(not r['api_called'] for r in results),
        'valid_businesses_by_split':dict(Counter(b['split'] for b in completed)),
        'successful_input_tokens':sum(r['usage'].get('input_tokens',0) for r in api),
        'successful_output_tokens':sum(r['usage'].get('output_tokens',0) for r in api),
        'human_verified_businesses':0,'production_method_frozen':False,'held_out_teacher_targets':0,
        'teacher_failures_not_manually_rewritten_or_promoted':True,'LED_training_started':False}
    assert report['task_slots_accounted_for']==60
    immutable(out/'completion_20_attempted_businesses.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    run()
