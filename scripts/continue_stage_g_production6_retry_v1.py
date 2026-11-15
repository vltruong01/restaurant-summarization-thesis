"""User-approved bounded retries; unchanged teacher payloads and strict validation."""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import stage_g_production_d042 as frozen
import continue_stage_g_production6_deferred_v1 as parent
import stage_g_business_id_guard_v1 as guard
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import stage_g_teacher_stream_diagnostics_v1 as transport
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL, KEY_VARIABLE

ROOT=frozen.ROOT
CONFIG=ROOT/'configs/stage_g_production6_retry_v1.json'
RUN=frozen.FOLDER/'guarded_production6_retry_v1'
OUT=RUN/'execution_v1'
SCRIPT=Path(__file__).resolve()
PASS='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'
MAX_RETRIES=2
CACHE=None
INDEX=None
TARGETS=None
ATTEMPTS=defaultdict(list)
USAGE=Counter()
SEEN_RAW=set()
COMPLETE=set()


def read_index():
    global INDEX
    if INDEX is None: INDEX=frozen.read(RUN/'reuse_index.json')
    return INDEX


def historical_verify(job,entry):
    return parent.parent.historical_verify(job,entry)


def prepare():
    assert not (RUN/'manifest.json').exists(),'Already prepared: use --run'
    previous,jobs,businesses,imports=parent.checked_inputs()
    cfg=frozen.read(CONFIG)
    assert all(cfg[k]==previous[k] for k in frozen.FIXED_KEYS)
    assert cfg['identifier_guard']==previous['identifier_guard']
    state=frozen.read(parent.OUT/'execution_progress.json')
    assert state['status']!='RUNNING' and not state['current_jobs'],'Previous batch must drain before transition'
    index=dict(parent.read_index());failures={};hashes={}
    for job in jobs:
        directory=parent.location(job,imports)
        path=directory/'checkpoints'/(job['job_id']+'.json')
        if path.exists():
            cp=frozen.read(path)
            if cp['status']==PASS:
                parent.verify_checkpoint(job,previous,directory,parent.CONFIG)
                if job['job_id'] not in index:
                    index[job['job_id']]={'directory':directory.relative_to(ROOT).as_posix(),
                        'config_path':parent.CONFIG.relative_to(ROOT).as_posix(),'request_schema':'business_id_enum_v1'}
            else:
                failures[job['job_id']]={'directory':directory.relative_to(ROOT).as_posix(),
                    'config_path':parent.CONFIG.relative_to(ROOT).as_posix(),'historical_attempts_counted':1}
        elif (directory/'request_intents'/path.name).exists():
            failures[job['job_id']]={'directory':directory.relative_to(ROOT).as_posix(),
                'config_path':parent.CONFIG.relative_to(ROOT).as_posix(),'historical_attempts_counted':1}
    assert len(index)==state['accepted_tasks']
    targets=dict(frozen.read(parent.RUN/'reuse_targets.json'))
    for path in (parent.OUT/'targets').glob('*.json'):
        assert path.stem not in targets;targets[path.stem]=path.relative_to(ROOT).as_posix()
    assert len(targets)==state['businesses_completed']
    for path in parent.OUT.rglob('*'):
        if path.is_file():hashes[path.relative_to(ROOT).as_posix()]=frozen.file_sha(path)
    for rel in targets.values():hashes[rel]=frozen.file_sha(ROOT/rel)
    frozen.immutable(RUN/'reuse_index.json',index)
    frozen.immutable(RUN/'reuse_targets.json',targets)
    frozen.immutable(RUN/'historical_failed_tasks.json',failures)
    auth=frozen.read(parent.RUN/'authorization.json')
    auth.update(config_sha256=frozen.file_sha(CONFIG),maximum_concurrency=6,
        maximum_automatic_retries=MAX_RETRIES,authorized_by='user requests retry failed requests during generation')
    frozen.immutable(RUN/'authorization.json',auth)
    for name in ('small_batch_manifest.json','small_batch_validation.json'):
        path=RUN/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write((parent.RUN/name).read_bytes())
    frozen.immutable(RUN/'manifest.json',{'approved_by':'user','scientific_settings_changed':False,
        'initial_complete_businesses':len(targets),'initial_reused_components':len(index),
        'initial_failed_tasks':len(failures),'maximum_concurrency':6,'maximum_automatic_retries':MAX_RETRIES,
        'retry_backoff_seconds':[30,60],'unknown_intent_without_checkpoint':'defer_no_automatic_resend',
        'config_sha256':frozen.file_sha(CONFIG),'reuse_index_sha256':frozen.file_sha(RUN/'reuse_index.json'),
        'reuse_targets_sha256':frozen.file_sha(RUN/'reuse_targets.json'),
        'historical_failures_sha256':frozen.file_sha(RUN/'historical_failed_tasks.json'),
        'code_sha256':{p.relative_to(ROOT).as_posix():frozen.file_sha(p) for p in (SCRIPT,ROOT/'scripts/stage_g_business_id_guard_v1.py')},
        'preserved_artifact_sha256':hashes,'prepared_utc':frozen.now(),'Codex_configuration_or_auth_access':False})
    print(json.dumps({'offline_preflight':'PASS','reused_complete_businesses':len(targets),
        'reused_components':len(index),'failed_tasks_eligible_for_bounded_review':len(failures),'API_calls':0}),flush=True)


def checked_inputs():
    global CACHE
    m=frozen.read(RUN/'manifest.json')
    for p,key in ((CONFIG,'config_sha256'),(RUN/'reuse_index.json','reuse_index_sha256'),
                  (RUN/'reuse_targets.json','reuse_targets_sha256'),(RUN/'historical_failed_tasks.json','historical_failures_sha256')):
        assert frozen.file_sha(p)==m[key]
    for rel,sha in m['code_sha256'].items():assert frozen.file_sha(ROOT/rel)==sha,rel
    if CACHE is None:
        previous,jobs,businesses,imports=parent.checked_inputs();cfg=frozen.read(CONFIG)
        assert all(cfg[k]==previous[k] for k in frozen.FIXED_KEYS)
        assert cfg['identifier_guard']==previous['identifier_guard']
        for rel,sha in m['preserved_artifact_sha256'].items():assert frozen.file_sha(ROOT/rel)==sha,rel
        for j in jobs:
            if j['job_id'] in read_index():historical_verify(j,read_index()[j['job_id']])
        CACHE=cfg,jobs,businesses,imports
    return CACHE


def load_attempts(jobs):
    ATTEMPTS.clear()
    known={j['job_id'] for j in jobs}
    for jid,entry in frozen.read(RUN/'historical_failed_tasks.json').items():
        ATTEMPTS[jid].append({'directory':ROOT/entry['directory'],'config':ROOT/entry['config_path'],'historical':True})
    for directory in sorted((OUT/'attempts').glob('attempt_*')):
        names={p.stem for sub in ('checkpoints','request_intents') for p in (directory/sub).glob('*.json')}
        assert names<=known
        for jid in names:
            ATTEMPTS[jid].append({'directory':directory,'config':CONFIG,'historical':False})


def checkpoint(jid,attempt):
    p=attempt['directory']/'checkpoints'/(jid+'.json')
    return frozen.read(p) if p.exists() else None


def retry_decision(jid,attempts,epoch=None):
    """Pure scheduler decision: failed CP required; never resend unknown outcomes."""
    if not attempts:return 'ready',0
    last=attempts[-1];cp=checkpoint(jid,last)
    if cp is None:return 'unresolved',0
    if cp['status']==PASS:return 'accepted',0
    if cp.get('http_status') in (401,402,403):return 'fatal',0
    if cp.get('local_validation_error')=='Reported model differs from requested alias; review provider provenance':return 'fatal',0
    if len(attempts)>=1+MAX_RETRIES:return 'exhausted',0
    failed_at=datetime.fromisoformat(cp['failed_utc']).timestamp()
    remaining=failed_at+[30,60][min(len(attempts)-1,1)]-(time.time() if epoch is None else epoch)
    return ('ready',0) if remaining<=0 else ('cooldown',remaining)


def location(job,imports):
    jid=job['job_id'];entry=read_index().get(jid)
    if entry:return ROOT/entry['directory']
    for attempt in ATTEMPTS[jid]:
        cp=checkpoint(jid,attempt)
        if cp and cp['status']==PASS:return attempt['directory']
    return ATTEMPTS[jid][-1]['directory'] if ATTEMPTS[jid] else OUT/'attempts/attempt_01'


def verify_checkpoint(job,cfg,directory,config_path,exact_request=True):
    entry=read_index().get(job['job_id'])
    if entry:return historical_verify(job,entry)
    return guard.verify_checkpoint(job,cfg,Path(directory),CONFIG)


def save_business(b,grouped,imports):
    global TARGETS
    content={}
    for j in grouped[b['business_id']]:
        p=location(j,imports)/'checkpoints'/(j['job_id']+'.json')
        if not p.exists():return None
        cp=frozen.read(p)
        if cp['status']!=PASS:return None
        content[j['condition']]=cp.get('resolved_output',cp['output'])
    target=frozen.build_target(b,content)
    if TARGETS is None:TARGETS=frozen.read(RUN/'reuse_targets.json')
    if b['business_id'] in TARGETS:
        p=ROOT/TARGETS[b['business_id']];assert frozen.read(p)==target;return p
    p=OUT/'targets'/(b['business_id']+'.json');frozen.immutable(p,target);return p


def update_docs(state):
    frozen.atomic(OUT/'execution_progress.json',state)
    frozen.atomic(ROOT/'outputs/stage_g/active_production_execution_pointer.json',{
        'status':state['status'],'provider':PROVIDER,'requested_model':MODEL,
        'production_directory':RUN.relative_to(ROOT).as_posix(),'progress_path':(OUT/'execution_progress.json').relative_to(ROOT).as_posix(),
        'maximum_concurrency':6,'maximum_retries_per_failed_task':MAX_RETRIES,
        'execution_entrypoint':SCRIPT.relative_to(ROOT).as_posix(),'pause_file':(RUN/'STOP_REQUESTED.json').relative_to(ROOT).as_posix(),
        'updated_utc':state['updated_utc']})
    path=ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md';marker='**D-042 live production status:**'
    lines=[l for l in path.read_text(encoding='utf-8').splitlines(keepends=True) if not l.startswith(marker)]
    note=(f"{marker} {state['businesses_completed']}/2200 businesses; {state['targets_completed']} targets; "
        f"status={state['status']}; max6 requests; retry pending={state['retry_pending_count']}; "
        f"unresolved/exhausted={state['deferred_task_count']}; max2 retries with30/60s cooldown. "
        "Every attempt preserved separately; failed outputs excluded; frozen payloads and strict evidence validation unchanged. "
        "TeacherAPI-only content; no held-out teacher targets, Codex auth changes or LED training. "
        "Pointer: outputs/stage_g/active_production_execution_pointer.json; structural PASS is not human factual certification.\n")
    lines.insert(1,'\n'+note+'\n');path.write_text(''.join(lines),encoding='utf-8')


def progress(jobs,businesses,imports,phase,changed=None,initial=False):
    inspect=jobs if initial else (changed or [])
    accepted=set(read_index())
    for jid,attempts in ATTEMPTS.items():
        if any((checkpoint(jid,a) or {}).get('status')==PASS for a in attempts):accepted.add(jid)
    grouped=defaultdict(list)
    for j in jobs:grouped[j['business_id']].append(j)
    affected={j['business_id'] for j in inspect}
    for b in businesses:
        if b['business_id'] in affected and save_business(b,grouped,imports):COMPLETE.add(b['business_id'])
    for j in inspect:
        paths=[location(j,imports)/'raw_responses'/(j['job_id']+'.json')]
        paths += [a['directory']/'raw_responses'/(j['job_id']+'.json') for a in ATTEMPTS[j['job_id']]]
        for p in paths:
            if not p.exists() or p in SEEN_RAW:continue
            SEEN_RAW.add(p);u=frozen.read(p).get('usage') or {}
            if u:
                USAGE['responses_with_usage']+=1;USAGE['input_tokens']+=u.get('input_tokens',0)
                USAGE['output_tokens']+=u.get('output_tokens',0)
                USAGE['reasoning_tokens']+=u.get('output_tokens_details',{}).get('reasoning_tokens',0)
    retry=[];deferred=[]
    for j in jobs:
        jid=j['job_id']
        if jid in accepted or not ATTEMPTS[jid]:continue
        decision,wait=retry_decision(jid,ATTEMPTS[jid])
        item={'job_id':jid,'state':decision,'recorded_attempts':len(ATTEMPTS[jid]),'cooldown_remaining_seconds':round(wait,2)}
        (retry if decision in ('ready','cooldown') else deferred).append(item)
    manifest=frozen.read(RUN/'manifest.json')
    api_attempts=sum(1 for p in (OUT/'attempts').glob('attempt_*/request_intents/*.json'))
    state={'status':'RUNNING','phase':phase,'updated_utc':frozen.now(),'pid':os.getpid(),
        'businesses_completed':len(COMPLETE),'businesses_planned':2200,'targets_completed':3*len(COMPLETE),
        'accepted_tasks':len(accepted),'tasks_planned':6600,'current_jobs':[],'maximum_concurrency':6,
        'initial_reused_complete_businesses':manifest['initial_complete_businesses'],
        'initial_reused_components':manifest['initial_reused_components'],'new_API_attempts':api_attempts,
        'retry_pending_count':len(retry),'deferred_task_count':len(deferred),'failed_or_unresolved':retry+deferred,
        'maximum_automatic_retries':MAX_RETRIES,'usage_including_imported_and_failed_attempts':dict(USAGE),
        'held_out_teacher_targets':0,'LED_training_started':False,'stop_reason':None}
    frozen.atomic(OUT/'retry_queue.json',{'updated_utc':state['updated_utc'],'pending':retry,'deferred':deferred})
    update_docs(state);print(json.dumps(state),flush=True)
    return state,accepted


def worker(job,cfg,prompt,schema,base,provider,key,directory):
    jid=job['job_id']
    if not job.get('skip_api_empty_captions'):transport.set_context(directory/'stream_diagnostics',jid)
    try:
        cp=legacy.job_result(job,cfg,prompt,guard.schema_for_job(job,schema),base,provider,key,directory,CONFIG)
        return {'job_id':jid,'passed':True,'attempt_directory':directory.relative_to(ROOT).as_posix(),
                'seconds':cp.get('seconds'),'response_id':cp.get('response_id')}
    except Exception as error:
        recorded=(directory/'checkpoints'/(jid+'.json')).exists() or (directory/'request_intents'/(jid+'.json')).exists()
        return {'job_id':jid,'passed':False,'attempt_directory':directory.relative_to(ROOT).as_posix(),
                'error_type':type(error).__name__,'local_integrity_failure':not recorded}


def audit_run():
    cfg,jobs,businesses,imports=checked_inputs();load_attempts(jobs)
    import audit_stage_g_production_d042 as audit
    audit.CONFIG=CONFIG;audit.FOLDER=RUN;audit.OUT=OUT;audit.check_inputs=checked_inputs
    audit.location=location;audit.verify_checkpoint=verify_checkpoint;audit.save_completed_business=save_business
    import run_stage_g_production_d042 as original
    def audit_docs(state):
        state.setdefault('retry_pending_count',0);state.setdefault('deferred_task_count',0);update_docs(state)
    original.update_docs=audit_docs;audit.run(smoke=False)


def run():
    cfg,jobs,businesses,imports=checked_inputs();load_attempts(jobs)
    for j in jobs:
        p=location(j,imports)/'checkpoints'/(j['job_id']+'.json')
        if p.exists() and frozen.read(p)['status']==PASS:verify_checkpoint(j,cfg,p.parent.parent,CONFIG)
    state,accepted=progress(jobs,businesses,imports,'resume_and_retry_check',initial=True)
    assert not os.environ.get(KEY_VARIABLE),'Private teacher credentials only'
    base,provider,key=credentials();assert (base,provider,cfg['requested_model'])==(BASE_URL,PROVIDER,MODEL)
    schemas=frozen.read(ROOT/cfg['schema']);prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    legacy.call=transport.call;failed_batches=0
    with ThreadPoolExecutor(max_workers=6) as pool:
        while len(accepted)<6600:
            if (RUN/'STOP_REQUESTED.json').exists():
                state.update(status='STOPPED_REQUIRES_REVIEW',stop_reason='Pause requested; in-flight batch drained');update_docs(state);return
            ready=[];waits=[];fatal=[]
            for j in jobs:
                jid=j['job_id']
                if jid in accepted:continue
                decision,wait=retry_decision(jid,ATTEMPTS[jid])
                if decision=='ready':ready.append(j)
                elif decision=='cooldown':waits.append(wait)
                elif decision=='fatal':fatal.append(jid)
            if fatal:
                state.update(status='STOPPED_REQUIRES_REVIEW',stop_reason='Account authentication/quota or model-alias failure');update_docs(state);return
            if not ready:
                if waits:
                    state['phase']='retry_cooldown';state['updated_utc']=frozen.now();update_docs(state)
                    time.sleep(min(min(waits),10));continue
                state.update(status='GENERATION_SWEEP_FINISHED_PENDING_RECOVERY',phase='retry_limit_or_unresolved_intents',current_jobs=[])
                update_docs(state);return
            # Recover known failures first, without exceeding the same six-call global pool.
            ready.sort(key=lambda j:(not bool(ATTEMPTS[j['job_id']]),j['job_id']))
            batch=ready[:6];submitted=[]
            for j in batch:
                jid=j['job_id'];new_count=sum(not a['historical'] for a in ATTEMPTS[jid])
                directory=OUT/'attempts'/f'attempt_{new_count+1:02d}'
                assert not (directory/'checkpoints'/(jid+'.json')).exists()
                assert not (directory/'request_intents'/(jid+'.json')).exists()
                if not j.get('skip_api_empty_captions'):
                    body=frozen.canonical(guard.payload_for_job(j,cfg,prompts[j['task']],schemas[j['task']]))
                    for old in ATTEMPTS[jid]:
                        p=old['directory']/'raw_requests'/(jid+'.json.gz')
                        if p.exists():assert gzip.decompress(p.read_bytes())==body,'Retry payload changed'
                    p=directory/'raw_requests'/(jid+'.json.gz');p.parent.mkdir(parents=True,exist_ok=True)
                    if p.exists():assert gzip.decompress(p.read_bytes())==body
                    else:
                        with p.open('xb') as f:f.write(gzip.compress(body,mtime=0))
                ATTEMPTS[jid].append({'directory':directory,'config':CONFIG,'historical':False})
                submitted.append((j,directory))
            running=dict(state,status='RUNNING',phase='production_with_bounded_retries',current_jobs=[j['job_id'] for j in batch],updated_utc=frozen.now())
            update_docs(running)
            futures={pool.submit(worker,j,cfg,prompts[j['task']],schemas[j['task']],base,provider,key,d):j for j,d in submitted}
            results=[]
            for future in as_completed(futures):
                result=future.result();results.append(result);running['current_jobs'].remove(result['job_id'])
                running['updated_utc']=frozen.now();frozen.atomic(OUT/'execution_progress.json',running)
                print(json.dumps(result),flush=True)
            with (OUT/'batch_log.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'finished_utc':frozen.now(),'phase':'production_with_bounded_retries','results':results})+'\n')
            state,accepted=progress(jobs,businesses,imports,'production_with_bounded_retries',changed=batch)
            failed_batches=failed_batches+1 if all(not r['passed'] for r in results) else 0
            if any(r.get('local_integrity_failure') for r in results) or failed_batches>=3:
                state.update(status='STOPPED_REQUIRES_REVIEW',stop_reason='Local failure or three entirely failed batches; attempts retained');update_docs(state);return
    assert len(COMPLETE)==2200
    frozen.atomic(OUT/'generation_completion.json',dict(state,status='GENERATION_COMPLETE_PENDING_FINAL_VALIDATION'))
    subprocess.run([sys.executable,str(SCRIPT),'--audit'],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ('prepare','run','prepare-and-run','audit'):parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args();assert sum(vars(args).values())==1
    lock=None if args.audit else parent.parent.parent.restored.shared_lock()
    if args.prepare or args.prepare_and_run:prepare()
    if args.prepare:return
    if args.audit:audit_run()
    else:run()


if __name__=='__main__':
    try:main()
    except Exception as error:
        frozen.atomic(RUN/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(error).__name__,'updated_utc':frozen.now()})
        p=OUT/'execution_progress.json'
        if p.exists():
            state=frozen.read(p);state.update(status='STOPPED_REQUIRES_REVIEW',current_jobs=[],stop_reason='Local preflight/runner failure: '+type(error).__name__,updated_utc=frozen.now());update_docs(state)
        print(json.dumps({'status':'STOPPED_REQUIRES_REVIEW','error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
