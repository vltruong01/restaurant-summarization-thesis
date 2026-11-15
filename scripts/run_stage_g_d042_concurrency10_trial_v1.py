"""One user-approved ten-request scheduling trial; inference/scientific inputs unchanged."""
import gzip
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from stage_g_production_d042 import (
    ROOT, CONFIG, FOLDER, OUT, PILOT_OUT, read, file_sha, canonical, immutable,
    atomic, check_inputs, location, verify_checkpoint, save_completed_business, now,
)
from run_stage_g_production_d042 import runner_lock, progress, update_docs
from run_stage_g_teacher_pilot_v4_1_recovery import job_result, payload
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL

TRIAL = FOLDER / 'execution_control_concurrency10_trial_v1'


def numeric_stats(values):
    return {'n':len(values),'min':min(values) if values else None,
            'mean':statistics.mean(values) if values else None,
            'median':statistics.median(values) if values else None,
            'max':max(values) if values else None}


def recent_four_baseline(jobs):
    """Observed prior batch timestamps; reference only, not a matched causal benchmark."""
    index = {j['job_id']: j for j in jobs}
    lines = (OUT / 'batch_log.jsonl').read_text(encoding='utf-8').splitlines()
    selected=[]
    for line in reversed(lines):
        row=json.loads(line)
        if len(row['results'])!=4 or not all(r['passed'] for r in row['results']): continue
        timings=[]; tasks=[]
        for r in row['results']:
            d=read(OUT/'stream_diagnostics'/(r['job_id']+'.json'))
            timings.append(d);tasks.append(index[r['job_id']]['task'])
        from datetime import datetime
        starts=[datetime.fromisoformat(d['started_utc']).timestamp() for d in timings]
        ends=[datetime.fromisoformat(d['finished_utc']).timestamp() for d in timings]
        elapsed=max(ends)-min(starts)
        selected.append({'job_ids':[r['job_id'] for r in row['results']],
                         'finished_utc':row['finished_utc'],'task_mix':dict(Counter(tasks)),
                         'batch_seconds':round(elapsed,3), 'requests_per_minute':240/elapsed,
                         'request_seconds':[d['elapsed_seconds'] for d in timings]})
        if len(selected)==5:break
    return {'batches':list(reversed(selected)),
            'task_mix':dict(Counter(index[j]['task'] for b in selected for j in b['job_ids'])),
            'request_seconds':numeric_stats([v for b in selected for v in b['request_seconds']]),
            'pooled_requests_per_minute':60*4*len(selected)/sum(b['batch_seconds'] for b in selected) if selected else None,
            'matched_request_benchmark':False}


def trial_progress(snapshot,current,started):
    state=dict(snapshot,status='RUNNING',phase='concurrency10_bounded_trial',current_jobs=current,
               maximum_concurrency=10,baseline_concurrency=4,updated_utc=now(),
               trial_only=True,trial_started_utc=started,
               new_API_attempts=len(list((OUT/'request_intents').glob('*.json'))))
    atomic(OUT/'execution_progress.json',state)
    return state


def main():
    lock=runner_lock()  # Same OS-exclusive lock as the paused production runner.
    assert (FOLDER/'STOP_REQUESTED.json').exists(), 'Drain/pause original worker first'
    assert not (TRIAL/'manifest.json').exists(), 'One bounded trial only; no automatic repeat'
    cfg,jobs,businesses,imports=check_inputs()
    snapshot,accepted,complete=progress(jobs,businesses,imports,[],'concurrency10_preflight')
    assert not snapshot['failed_or_unresolved'], 'Existing failure/unresolved attempt blocks new calls'
    candidates=[j for j in jobs if j['job_id'] not in accepted and j['job_id'] not in imports
                and not j.get('skip_api_empty_captions')]
    trial_jobs=candidates[:10];assert len(trial_jobs)==10
    baseline=recent_four_baseline(jobs)
    frozen_cfg_sha=file_sha(CONFIG)
    previous={p.relative_to(ROOT).as_posix():file_sha(p)
              for d in (OUT/'checkpoints',OUT/'raw_responses',OUT/'targets') for p in d.glob('*.json')}
    immutable(TRIAL/'previous_successful_artifact_sha256.json',previous)
    immutable(TRIAL/'manifest.json',{'approved':True,'approved_by':'user',
        'authorization_basis':'User explicitly requested first testing10 concurrent requests',
        'scope':'one batch of next10 previously unstarted paid train/validation jobs',
        'maximum_concurrency':10,'maximum_new_API_calls':10,'automatic_retries':0,
        'config_sha256':frozen_cfg_sha,'runner_sha256':file_sha(Path(__file__)),
        'job_ids':[j['job_id'] for j in trial_jobs],'jobs_sha256':file_sha(FOLDER/'production_jobs.json'),
        'scientific_settings_or_request_payload_changed':False,'baseline_return_concurrency':4,
        'production_maximum10_approved':False,'provider_pricing_research_required':False,
        'failure_policy':'drain_all_submitted_calls_stop_no_retry_on_any_failure',
        'test_payloads':0,'Codex_configuration_or_authentication_access':False})
    schemas=read(ROOT/cfg['schema']);prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    for j in trial_jobs:
        assert j['split'] in ('train','validation')
        for d in ('checkpoints','request_intents','raw_responses','stream_diagnostics'):
            assert not (OUT/d/(j['job_id']+'.json')).exists()
        raw_request=canonical(payload(j,cfg,prompts[j['task']],schemas[j['task']]))
        path=OUT/'raw_requests'/(j['job_id']+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
        compressed=gzip.compress(raw_request,mtime=0)
        if path.exists():assert path.read_bytes()==compressed
        else:path.write_bytes(compressed)
    # This approval is an operational scheduling overlay; original frozen config stays byte-identical.
    base,provider,key=credentials();assert (base,provider,cfg['requested_model'])==(BASE_URL,PROVIDER,MODEL)
    import run_stage_g_teacher_pilot_v4_1_recovery as legacy
    import stage_g_teacher_stream_diagnostics_v1 as diagnostic
    legacy.call=diagnostic.call
    started_utc=now();start=time.monotonic();active=[j['job_id'] for j in trial_jobs]
    state=trial_progress(snapshot,active[:],started_utc);update_docs(state)

    def worker(j):
        diagnostic.set_context(OUT/'stream_diagnostics',j['job_id'])
        try:
            cp=job_result(j,cfg,prompts[j['task']],schemas[j['task']],base,provider,key,OUT,CONFIG)
            return {'job_id':j['job_id'],'passed':True,'seconds':cp.get('seconds'),'response_id':cp.get('response_id')}
        except Exception as e:
            return {'job_id':j['job_id'],'passed':False,'error_type':type(e).__name__}

    results=[]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures={pool.submit(worker,j):j for j in trial_jobs}
        for future in as_completed(futures):
            r=future.result();results.append(r);active.remove(r['job_id'])
            trial_progress(snapshot,active[:],started_utc)
            print(json.dumps(r),flush=True)
    elapsed=round(time.monotonic()-start,3)
    # All writers have drained before offline validation and shared-state reads.
    for r in results:
        if r['passed']:
            j=next(j for j in trial_jobs if j['job_id']==r['job_id'])
            verify_checkpoint(j,cfg,OUT,CONFIG)
    for rel,sha in previous.items(): assert file_sha(ROOT/rel)==sha
    assert file_sha(CONFIG)==frozen_cfg_sha
    from datetime import datetime
    timings=[]; events=[]; http=[];times_by_task=defaultdict(list);tier=Counter();effort=Counter();usage=Counter()
    for j in trial_jobs:
        p=OUT/'stream_diagnostics'/(j['job_id']+'.json')
        if p.exists():
            d=read(p);timings.append(d);http.append(d.get('http_status'))
            events.extend([(datetime.fromisoformat(d['started_utc']).timestamp(),1),
                           (datetime.fromisoformat(d['finished_utc']).timestamp(),-1)])
            times_by_task[j['task']].append(d['elapsed_seconds'])
        rawp=OUT/'raw_responses'/(j['job_id']+'.json')
        if rawp.exists():
            raw=read(rawp);tier[str(raw.get('service_tier'))]+=1;effort[str((raw.get('reasoning') or {}).get('effort'))]+=1
            u=raw.get('usage') or {};usage['input_tokens']+=u.get('input_tokens',0);usage['output_tokens']+=u.get('output_tokens',0)
            usage['reasoning_tokens']+=u.get('output_tokens_details',{}).get('reasoning_tokens',0)
    running=peak=0
    for _,delta in sorted(events,key=lambda e:(e[0],e[1])):running+=delta;peak=max(peak,running)
    success=sum(r['passed'] for r in results)
    snapshot,accepted,complete=progress(jobs,businesses,imports,[],'concurrency10_trial_completed',
         None if success==10 else 'Ten-request trial had failure; no automatic retry',changed_jobs=trial_jobs)
    snapshot.update(status='TRIAL_COMPLETED_PENDING_RETURN_TO_FROZEN_FOUR' if success==10 else 'STOPPED_REQUIRES_REVIEW',
                    maximum_concurrency=4,current_jobs=[])
    atomic(OUT/'execution_progress.json',snapshot);update_docs(snapshot)
    actual_attempts=len([j for j in trial_jobs if (OUT/'request_intents'/(j['job_id']+'.json')).exists()])
    report={'status':'CONCURRENCY10_SINGLE_BATCH_TECHNICAL_PASS' if success==10 else 'CONCURRENCY10_TRIAL_FAILURE_STOPPED',
            'planned_new_requests':10,'actual_new_request_intents':actual_attempts,'passed':success,'failed':10-success,
            'batch_wall_seconds':elapsed,'observed_transport_overlap_peak':peak,'max_workers':10,
            'http_statuses':dict(Counter(str(x) for x in http)),
            'HTTP429_or_5xx':sum(x==429 or (isinstance(x,int) and x>=500) for x in http),
            'request_seconds':numeric_stats([d['elapsed_seconds'] for d in timings]),
            'seconds_by_task':{k:numeric_stats(v) for k,v in times_by_task.items()},
            'task_mix':dict(Counter(j['task'] for j in trial_jobs)),
            'successful_requests_per_minute':60*success/elapsed,'baseline_recent_four':baseline,
            'throughput_ratio_vs_unmatched_recent_four':(60*success/elapsed)/baseline['pooled_requests_per_minute'] if baseline['pooled_requests_per_minute'] else None,
            'comparison_limitation':'Different request/business/task lengths and transient provider load; one batch cannot establish sustained stability or a causal speedup.',
            'raw_response_service_tiers':dict(tier),'raw_response_reasoning_efforts':dict(effort),'usage':dict(usage),
            'all_previous_successful_artifacts_unchanged':True,'config_prompts_schema_sources_and_payload_settings_unchanged':True,
            'all_successful_trial_requests_independently_schema_source_validated':True,
            'target_content_origin':'teacher_API_only_frozen_serialization','human_factual_quality_certified':False,
            'businesses_complete_after_trial':len(complete),'targets_complete_after_trial':3*len(complete),
            'results':results,'automatic_retries':0,'production_concurrency10_frozen':False,
            'next':'resume_existing_four_worker_production_without_duplicate_calls' if success==10 else 'review_preserved_failures_before_resuming'}
    immutable(TRIAL/'report.json',report)
    with (OUT/'batch_log.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'finished_utc':now(),'phase':'concurrency10_bounded_trial','results':results})+'\n')
    note=(f"**D-042 concurrency10 trial result:** {success}/10 technical PASS, peak simultaneous transport={peak}, "
          f"wall={elapsed}s, HTTP429/5xx={report['HTTP429_or_5xx']}; {len(complete)}/2200 complete businesses. "
          "Default service tier/medium reasoning and all frozen D040 science settings unchanged; no manual target rewriting or duplicate calls. "
          "One batch does not prove sustained10-worker reliability; report at outputs/stage_g/teacher_production_d042_v1/execution_control_concurrency10_trial_v1/report.json. "
          + ('Return to approved four-worker production pending researcher review of trial.' if success==10 else 'Failure preserved; stop without automatic retry.')+'\n')
    for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md','DECISIONS.md']:
        p=ROOT/rel;text=p.read_text(encoding='utf-8');first,rest=text.split('\n',1)
        p.write_text(first+'\n\n'+note+'\n'+rest,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','baseline_recent_four')},indent=2),flush=True)
    if success==10:
        # Lock must be released by process exit before the normal runner resumes.
        atomic(TRIAL/'return_to_four_ready.json',{'approved_existing_production_resume':True,'trial_report_sha256':file_sha(TRIAL/'report.json')})


if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic(TRIAL/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(e).__name__,'updated_utc':now(),'automatic_retries':0})
        p=OUT/'execution_progress.json'
        if p.exists():
            state=read(p);state.update(status='STOPPED_REQUIRES_REVIEW',current_jobs=[],stop_reason='Operational trial runner failed: '+type(e).__name__,updated_utc=now())
            atomic(p,state);update_docs(state)
        print(json.dumps({'status':'STOPPED_REQUIRES_REVIEW','error_type':type(e).__name__}),flush=True)
        raise SystemExit(1)
