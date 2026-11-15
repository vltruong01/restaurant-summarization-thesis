"""One authorized six-call trial, two untouched businesses, no automatic retry or scale-up."""
import gzip
import json
import os
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import stage_g_production_d042 as frozen
import run_stage_g_business_id_guard_production_v1 as parent
import stage_g_business_id_guard_v1 as guard
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import stage_g_teacher_stream_diagnostics_v1 as transport
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL, KEY_VARIABLE

ROOT=frozen.ROOT
CONFIG=ROOT/'configs/stage_g_coletoken_guarded_concurrency6_trial_v1.json'
TRIAL=frozen.FOLDER/'coletoken_guarded_concurrency6_trial_v1'
OUT=TRIAL/'execution_v1'
SCRIPT=Path(__file__).resolve()


def stats(values):
    return {'n':len(values),'min':round(min(values),3),'mean':round(statistics.mean(values),3),
            'median':round(statistics.median(values),3),'max':round(max(values),3)} if values else None


def append_note(note):
    p=ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md'; t=p.read_text(encoding='utf-8')
    if note not in t: p.write_text(t+'\n\n'+note+'\n',encoding='utf-8')


def main():
    lock=parent.restored.shared_lock()
    assert not (TRIAL/'manifest.json').exists(), 'Preserve earlier attempt; no automatic rerun'
    base_cfg,jobs,businesses,imports=parent.checked_inputs()
    state=frozen.read(parent.OUT/'execution_progress.json')
    assert not state['current_jobs'] and state['status']=='STOPPED_REQUIRES_REVIEW'
    cfg=frozen.read(CONFIG)
    assert all(cfg[k]==base_cfg[k] for k in frozen.FIXED_KEYS)
    assert cfg['identifier_guard']==base_cfg['identifier_guard']
    assert cfg['execution']['maximum_concurrency']==6 and cfg['execution']['maximum_new_API_calls']==6
    prior_index=frozen.read(parent.RUN/'reuse_index.json')
    grouped=defaultdict(list)
    for j in jobs: grouped[j['business_id']].append(j)
    selected_businesses=[]
    for b in businesses:
        js=grouped[b['business_id']]
        if len(js)!=3 or any(j.get('skip_api_empty_captions') for j in js): continue
        if any(j['job_id'] in prior_index for j in js): continue
        if any((parent.OUT/sub/(j['job_id']+'.json')).exists()
               for j in js for sub in ('checkpoints','request_intents')): continue
        selected_businesses.append(b)
        if len(selected_businesses)==2: break
    assert len(selected_businesses)==2
    selected=[j for b in selected_businesses for j in grouped[b['business_id']]]
    assert len(selected)==len({j['job_id'] for j in selected})==6
    assert all(j['split'] in ('train','validation') for j in selected)
    prior={p.relative_to(ROOT).as_posix():frozen.file_sha(p)
           for p in parent.OUT.rglob('*') if p.is_file()}
    schemas=frozen.read(ROOT/cfg['schema'])
    prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    for j in selected:
        body=frozen.canonical(guard.payload_for_job(j,cfg,prompts[j['task']],schemas[j['task']]))
        p=OUT/'raw_requests'/(j['job_id']+'.json.gz'); p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f: f.write(gzip.compress(body,mtime=0))
    frozen.immutable(TRIAL/'manifest.json',{
        'approved_by':'user','authorization':'Try six requests in parallel',
        'maximum_concurrency':6,'maximum_calls':6,'automatic_retries':0,
        'business_ids':[b['business_id'] for b in selected_businesses],
        'job_ids':[j['job_id'] for j in selected],
        'selection':'first_two_untouched_nonempty_businesses_in_frozen_order',
        'scientific_settings_and_guard_unchanged':True,'existing_outputs_regenerated':False,
        'prior_failed_source_job_retried':False,'prior_failure_remains_production_blocker':True,
        'config_sha256':frozen.file_sha(CONFIG),'parent_config_sha256':frozen.file_sha(parent.CONFIG),
        'code_sha256':{rel:frozen.file_sha(ROOT/rel) for rel in (
            'scripts/run_stage_g_guarded_concurrency6_trial_v1.py','scripts/stage_g_business_id_guard_v1.py',
            'scripts/stage_g_teacher_stream_diagnostics_v1.py')},
        'previous_execution_sha256':prior,'transport':'original_D042_Coletoken',
        'socket_idle_timeout_seconds':120,'wall_deadline_seconds':180,
        'continuous_production_concurrency6_started':False,'prepared_utc':frozen.now()})
    append_note('User-approved bounded six-request concurrency trial prepared: two untouched businesses in frozen order, three existing-condition jobs each; retain D043 ID guard and original D042 model/prompts/schema/content/evidence checks/Coletoken transport. Exactly6 new calls, no duplicate successful jobs, no automatic retries or full-six-worker production. The16tj source-span failure remains untouched and unresolved. Trial namespace: outputs/stage_g/teacher_production_d042_v1/coletoken_guarded_concurrency6_trial_v1.')
    print(json.dumps({'offline_preflight':'PASS','businesses':[b['business_id'] for b in selected_businesses],
                      'planned_API_calls':6,'regeneration':False}),flush=True)
    assert not os.environ.get(KEY_VARIABLE), 'Private teacher credentials only'
    base,provider,key=credentials(); assert (base,provider,cfg['requested_model'])==(BASE_URL,PROVIDER,MODEL)
    legacy.call=transport.call
    barrier=threading.Barrier(6); mutex=threading.RLock(); active=set(); results=[]; peak=[0]
    started=time.monotonic()

    def publish(status='RUNNING_SIX_REQUEST_TRIAL'):
        with mutex:
            progress={'status':status,'provider':provider,'requested_model':MODEL,'pid':os.getpid(),
                'current_jobs':sorted(active),'completed_requests':len(results),'maximum_concurrency':6,
                'observed_active_workers_peak':peak[0],'planned_calls':6,'automatic_retries':0,
                'results':results[:],'updated_utc':frozen.now(),'production_unchanged':True}
            frozen.atomic(TRIAL/'execution_progress.json',progress)
            frozen.atomic(ROOT/'outputs/stage_g/active_teacher_trial_pointer.json',{
                'status':status,'trial_directory':TRIAL.relative_to(ROOT).as_posix(),
                'progress_path':(TRIAL/'execution_progress.json').relative_to(ROOT).as_posix(),
                'maximum_concurrency':6,'production_resume_after_trial':False,'updated_utc':progress['updated_utc']})

    def worker(j):
        transport.set_context(OUT/'stream_diagnostics',j['job_id'])
        barrier.wait(timeout=30)
        with mutex: active.add(j['job_id']); peak[0]=max(peak[0],len(active)); publish()
        try:
            cp=legacy.job_result(j,cfg,prompts[j['task']],guard.schema_for_job(j,schemas[j['task']]),
                                 base,provider,key,OUT,CONFIG)
            guard.verify_checkpoint(j,cfg,OUT,CONFIG)
            return {'job_id':j['job_id'],'business_id':j['business_id'],'task':j['task'],
                    'passed':True,'seconds':cp.get('seconds'),'response_id':cp.get('response_id')}
        except Exception as e:
            return {'job_id':j['job_id'],'business_id':j['business_id'],'task':j['task'],
                    'passed':False,'error_type':type(e).__name__}
        finally:
            with mutex: active.discard(j['job_id']); publish()

    publish()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(worker,j):j for j in selected}
        for future in as_completed(futures):
            result=future.result()
            with mutex: results.append(result); publish()
            print(json.dumps(result),flush=True)
    batch_seconds=round(time.monotonic()-started,3)
    timings=defaultdict(list); usage=Counter(); models=Counter(); tiers=Counter(); efforts=Counter(); events=[]; failures=[]
    for j in selected:
        jid=j['job_id']; d=frozen.read(OUT/'stream_diagnostics'/(jid+'.json'))
        timings[j['task']].append(d['elapsed_seconds'])
        events += [(datetime.fromisoformat(d['started_utc']).timestamp(),1),
                   (datetime.fromisoformat(d['finished_utc']).timestamp(),-1)]
        cp=frozen.read(OUT/'checkpoints'/(jid+'.json'))
        if cp['status']=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW': guard.verify_checkpoint(j,cfg,OUT,CONFIG)
        else: failures.append({'job_id':jid,'checkpoint':cp,'transport':d})
        p=OUT/'raw_responses'/(jid+'.json')
        if p.exists():
            raw=frozen.read(p); models[raw.get('model','unknown')]+=1; tiers[raw.get('service_tier','unknown')]+=1
            efforts[(raw.get('reasoning') or {}).get('effort','unknown')]+=1
            u=raw.get('usage') or {}
            usage['responses_with_usage']+=bool(u); usage['input_tokens']+=u.get('input_tokens',0)
            usage['output_tokens']+=u.get('output_tokens',0)
            usage['reasoning_tokens']+=(u.get('output_tokens_details') or {}).get('reasoning_tokens',0)
    live=overlap=0
    for timestamp,delta in sorted(events,key=lambda e:(e[0],e[1])):
        live+=delta; overlap=max(overlap,live)
    complete=[]
    for b in selected_businesses:
        cp=[frozen.read(OUT/'checkpoints'/(j['job_id']+'.json')) for j in grouped[b['business_id']]]
        if not all(c['status']=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for c in cp): continue
        content={j['condition']:c.get('resolved_output',c['output']) for j,c in zip(grouped[b['business_id']],cp)}
        frozen.immutable(OUT/'targets'/(b['business_id']+'.json'),frozen.build_target(b,content))
        complete.append(b['business_id'])
    for rel,sha in prior.items(): assert frozen.file_sha(ROOT/rel)==sha, rel
    calls=len(list((OUT/'request_intents').glob('*.json'))); assert calls<=6
    passed=sum(r['passed'] for r in results)
    report={'status':'SIX_REQUEST_GUARDED_TRIAL_PASS' if passed==6 else 'SIX_REQUEST_GUARDED_TRIAL_WITH_FAILURES',
        'passed':passed,'failed':6-passed,'paid_calls':calls,'maximum_concurrency':6,
        'observed_client_transport_overlap_peak':overlap,'observed_worker_peak':peak[0],
        'batch_seconds':batch_seconds,'transport_seconds_by_task':{k:stats(v) for k,v in timings.items()},
        'sum_transport_seconds':round(sum(sum(v) for v in timings.values()),3),
        'returned_models':dict(models),'returned_service_tiers':dict(tiers),'returned_reasoning_efforts':dict(efforts),
        'usage_returned':dict(usage),'results':results,'failures':failures,'complete_businesses':complete,
        'condition_aligned_targets':len(complete)*3,'prior_artifacts_unchanged':True,
        'business_id_guard_and_evidence_validation_unchanged':True,'teacher_text_rewritten':False,
        'automatic_retries':0,'continuous_production_concurrency6_started':False,
        'overlap_note':'Client in-flight overlap; not proof of simultaneous provider compute or sixfold speedup',
        'prior_source_span_failure_unresolved':True,'historical_production_complete_businesses':state['businesses_completed'],
        'trial_outputs_reusable_after_compatibility_verification':True,'completed_utc':frozen.now()}
    frozen.immutable(TRIAL/'report.json',report)
    publish(report['status'])
    lines=['# Bounded six-request guarded Coletoken trial','',
        f"Result: {passed}/6 PASS; client transport overlap peak={overlap}; batch wall time={batch_seconds}s.",'',
        '| Job | Result | Seconds |','| --- | --- | --- |']
    lines += [f"| {r['job_id']} | {'PASS' if r['passed'] else 'FAIL'} | {r.get('seconds','see diagnostics')} |" for r in results]
    lines += ['', 'Same frozen teacher model/prompts/settings/evidence/ID guard and original Coletoken transport. No existing successful component was regenerated. All raw API responses, diagnostics and failed attempts are preserved. Two completed businesses, if present, are saved with the frozen serializer for later verified reuse. The prior16tj review source-span failure remains a production blocker. No full six-request production, test targets or LED training started. Client request overlap does not establish provider compute concurrency or a sixfold speedup.']
    (TRIAL/'TRIAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    append_note(f"Six-request trial result: {passed}/6 structural/ID/evidence PASS, overlap peak={overlap}, batch={batch_seconds}s, {len(complete)} complete businesses/{len(complete)*3} targets saved separately for verified reuse. Original153-business production remains stopped on its preserved source-span failure; no automatic six-worker scale-up. Report: outputs/stage_g/teacher_production_d042_v1/coletoken_guarded_concurrency6_trial_v1/report.json. Automated PASS is not independent factual certification.")
    print(json.dumps({k:report[k] for k in ('status','passed','failed','paid_calls','observed_client_transport_overlap_peak','batch_seconds','transport_seconds_by_task','returned_models','complete_businesses')},indent=2),flush=True)


if __name__=='__main__':
    try: main()
    except Exception as e:
        frozen.atomic(TRIAL/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(e).__name__,
                      'updated_utc':frozen.now(),'automatic_retries':0})
        print(json.dumps({'runner_error_type':type(e).__name__}),flush=True)
        raise SystemExit(1)
