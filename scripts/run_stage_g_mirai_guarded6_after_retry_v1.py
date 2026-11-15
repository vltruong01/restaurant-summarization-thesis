"""Six Mirai requests only; identical frozen payloads, isolated provider provenance."""
import gzip
import json
import os
import threading
import time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path

import stage_g_production_d042 as frozen
import continue_stage_g_production6_retry_v1 as parent
import stage_g_mirai_guard_v1 as guard
import stage_g_business_id_guard_v1 as original_guard
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import stage_g_teacher_stream_diagnostics_v1 as transport
import restore_stage_g_d042_runner_v1 as restored
from stage_g_teacher_credentials import key_file,KEY_VARIABLE,MODEL

ROOT=frozen.ROOT
CONFIG=ROOT/'configs/stage_g_mirai_guarded6_after_retry_v1.json'
TRIAL=frozen.FOLDER/'mirai_guarded6_after_retry_v1'
OUT=TRIAL/'execution_v1'
BASE_URL='https://api.miraiapi.com/v1'
PROVIDER='miraiapi'
SCRIPT=Path(__file__).resolve()


def note(text):
    path=ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md'
    with path.open('a',encoding='utf-8') as stream:stream.write('\n\n'+text+'\n')


def select_businesses(jobs,businesses,imports):
    groups=defaultdict(list)
    for j in jobs:groups[j['business_id']].append(j)
    selected=[]
    for b in businesses:
        js=groups[b['business_id']]
        if len(js)!=3 or b['split'] not in ('train','validation'):continue
        if any(j.get('skip_api_empty_captions') for j in js):continue
        if any(j['job_id'] in parent.read_index() for j in js):continue
        if any((parent.location(j,imports)/sub/(j['job_id']+'.json')).exists()
               for j in js for sub in ('checkpoints','request_intents')):continue
        selected.append(b)
        if len(selected)==2:break
    assert len(selected)==2
    return selected,[j for b in selected for j in groups[b['business_id']]]


def main():
    lock=restored.shared_lock()
    assert not (TRIAL/'manifest.json').exists(),'Preserve prior trial; no automatic rerun'
    previous,jobs,businesses,imports=parent.checked_inputs()
    parent.load_attempts(jobs)
    state=frozen.read(parent.OUT/'execution_progress.json')
    assert state['status']=='STOPPED_REQUIRES_REVIEW' and not state['current_jobs']
    cfg=frozen.read(CONFIG)
    assert all(cfg[k]==previous[k] for k in frozen.FIXED_KEYS)
    assert cfg['identifier_guard']==previous['identifier_guard']
    assert (cfg['execution']['base_url'],cfg['execution']['provider'],cfg['requested_model'])==(BASE_URL,PROVIDER,MODEL)
    assert cfg['operational_provider_trial']['maximum_calls']==cfg['execution']['maximum_concurrency']==6
    assert not cfg['full_generation_enabled']
    selected_businesses,selected=select_businesses(jobs,businesses,imports)
    assert len(selected)==len({j['job_id'] for j in selected})==6
    preserved={p.relative_to(ROOT).as_posix():frozen.file_sha(p) for p in parent.OUT.rglob('*') if p.is_file()}
    for p in (parent.CONFIG,Path(parent.__file__).resolve(),ROOT/'scripts/stage_g_teacher_credentials.py',
              parent.RUN/'manifest.json',ROOT/'outputs/stage_g/active_production_execution_pointer.json'):
        preserved[p.relative_to(ROOT).as_posix()]=frozen.file_sha(p)
    snapshot=TRIAL/'coletoken_restore_snapshot'
    for p in (parent.CONFIG,Path(parent.__file__).resolve(),parent.OUT/'execution_progress.json',
              ROOT/'outputs/stage_g/active_production_execution_pointer.json'):
        dest=snapshot/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(p.read_bytes())
    frozen.immutable(snapshot/'RESTORE_STATE.json',{'provider':'coletoken',
        'config_path':parent.CONFIG.relative_to(ROOT).as_posix(),'runner':Path(parent.__file__).resolve().relative_to(ROOT).as_posix(),
        'production_directory':parent.RUN.relative_to(ROOT).as_posix(),
        'businesses_complete':state['businesses_completed'],'targets_complete':state['targets_completed'],
        'accepted_tasks':state['accepted_tasks'],'status':state['status'],'stop_reason':state['stop_reason'],
        'provider_secret_unchanged':True,'resume_requires_resolving_billing_failures':True,
        'preserved_artifact_sha256':preserved,'saved_utc':frozen.now()})
    schemas=frozen.read(ROOT/cfg['schema']);prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    for j in selected:
        body=frozen.canonical(guard.payload_for_job(j,cfg,prompts[j['task']],schemas[j['task']]))
        assert body==frozen.canonical(original_guard.payload_for_job(j,previous,prompts[j['task']],schemas[j['task']]))
        p=OUT/'raw_requests'/(j['job_id']+'.json.gz');p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as stream:stream.write(gzip.compress(body,mtime=0))
    frozen.immutable(TRIAL/'manifest.json',{'approved_by':'user','authorization':'Switch teacher endpoint/key to Mirai for six concurrent trial requests only',
        'provider':PROVIDER,'base_url':BASE_URL,'requested_model':MODEL,'maximum_calls':6,'maximum_concurrency':6,
        'automatic_retries_in_bounded_trial':0,'production_retry_configuration_unchanged':True,
        'config_sha256':frozen.file_sha(CONFIG),'parent_config_sha256':frozen.file_sha(parent.CONFIG),
        'runner_sha256':frozen.file_sha(SCRIPT),'provider_guard_sha256':frozen.file_sha(ROOT/'scripts/stage_g_mirai_guard_v1.py'),
        'business_ids':[b['business_id'] for b in selected_businesses],'job_ids':[j['job_id'] for j in selected],
        'request_payloads_byte_identical_to_coletoken':True,'validation_change':'expected_endpoint_provider_binding_only',
        'original_transport_unchanged':True,'production_outputs_unchanged':True,'production_integration_approved':False,
        'coletoken_restore_snapshot':snapshot.relative_to(ROOT).as_posix(),'prepared_utc':frozen.now(),
        'Codex_configuration_or_auth_access':False})
    assert not os.environ.get(KEY_VARIABLE),'Private provider-specific teacher key only'
    secret=key_file().with_name(cfg['execution']['dedicated_secret_filename'])
    assert not secret.resolve().is_relative_to(ROOT.resolve())
    assert not secret.resolve().is_relative_to((Path.home()/'.codex').resolve())
    key=secret.read_text(encoding='utf-8-sig').strip();assert key and not any(c.isspace() for c in key)
    legacy.call=transport.call
    note('User-authorized isolated Mirai six-request teacher trial prepared after Coletoken billing stop: '
         f"preserved{state['businesses_completed']} complete businesses/{state['targets_completed']} targets and{state['accepted_tasks']} accepted components. "
         'Coletoken configuration, key and complete execution history remain unchanged; restore snapshot saved in '
         'outputs/stage_g/teacher_production_d042_v1/mirai_guarded6_after_retry_v1/coletoken_restore_snapshot. '
         'Mirai endpoint https://api.miraiapi.com/v1, alias gpt-6.1-sol, original medium reasoning/stream/store/max-output/strict schema/ID/evidence checks unchanged. '
         'Six byte-identical request payloads for two untouched train/validation businesses; endpoint/key and expected provider binding only differ. '
         'No Codex configuration/auth/key access, no automatic trial retries, full production integration, held-out targets or LED training. '
         'Trial uses active_teacher_trial_pointer.json; active production pointer remains the preserved Coletoken state.')
    print(json.dumps({'offline_preflight':'PASS','planned_calls':6,'businesses':[b['business_id'] for b in selected_businesses],
                      'payloads_identical':True,'preserved_complete_businesses':state['businesses_completed']}),flush=True)
    barrier=threading.Barrier(6);mutex=threading.RLock();active=set();results=[];peak=[0]
    def publish(status='RUNNING_ISOLATED_MIRAI_SIX_REQUEST_TRIAL'):
        with mutex:
            current={'status':status,'provider':PROVIDER,'requested_model':MODEL,'pid':os.getpid(),
                'current_jobs':sorted(active),'completed_requests':len(results),'maximum_concurrency':6,
                'worker_overlap_peak':peak[0],'planned_calls':6,'results':results[:],'production_unchanged':True,'updated_utc':frozen.now()}
            frozen.atomic(TRIAL/'execution_progress.json',current)
            frozen.atomic(ROOT/'outputs/stage_g/active_teacher_trial_pointer.json',{
                'status':status,'provider':PROVIDER,'progress_path':(TRIAL/'execution_progress.json').relative_to(ROOT).as_posix(),
                'trial_directory':TRIAL.relative_to(ROOT).as_posix(),'maximum_concurrency':6,'production_resume_after_trial':False,'updated_utc':current['updated_utc']})
    def worker(j):
        transport.set_context(OUT/'stream_diagnostics',j['job_id']);barrier.wait(timeout=30)
        with mutex:active.add(j['job_id']);peak[0]=max(peak[0],len(active));publish()
        try:
            cp=legacy.job_result(j,cfg,prompts[j['task']],guard.schema_for_job(j,schemas[j['task']]),BASE_URL,PROVIDER,key,OUT,CONFIG)
            guard.verify_checkpoint(j,cfg,OUT,CONFIG)
            return {'job_id':j['job_id'],'business_id':j['business_id'],'passed':True,'seconds':cp.get('seconds'),'response_id':cp.get('response_id')}
        except Exception as error:
            return {'job_id':j['job_id'],'business_id':j['business_id'],'passed':False,'error_type':type(error).__name__}
        finally:
            with mutex:active.discard(j['job_id']);publish()
    started=time.monotonic();publish()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(worker,j) for j in selected]
        for future in as_completed(futures):
            result=future.result()
            with mutex:results.append(result);publish()
            print(json.dumps(result),flush=True)
    seconds=round(time.monotonic()-started,3);usage=Counter();models=Counter();tiers=Counter();events=[];failures=[]
    for j in selected:
        jid=j['job_id'];diag_path=OUT/'stream_diagnostics'/(jid+'.json');cp_path=OUT/'checkpoints'/(jid+'.json')
        diag=frozen.read(diag_path) if diag_path.exists() else {};cp=frozen.read(cp_path) if cp_path.exists() else {}
        if diag.get('started_utc') and diag.get('finished_utc'):
            events += [(datetime.fromisoformat(diag['started_utc']).timestamp(),1),(datetime.fromisoformat(diag['finished_utc']).timestamp(),-1)]
        if cp.get('status')=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':guard.verify_checkpoint(j,cfg,OUT,CONFIG)
        else:failures.append({'job_id':jid,'checkpoint':cp,'transport':diag})
        p=OUT/'raw_responses'/(jid+'.json')
        if p.exists():
            raw=frozen.read(p);models[raw.get('model','missing')]+=1;tiers[raw.get('service_tier','missing')]+=1
            u=raw.get('usage') or {};usage['responses_with_usage']+=bool(u)
            usage['input_tokens']+=u.get('input_tokens',0);usage['output_tokens']+=u.get('output_tokens',0)
            usage['reasoning_tokens']+=(u.get('output_tokens_details') or {}).get('reasoning_tokens',0)
    live=overlap=0
    for epoch,delta in sorted(events,key=lambda e:(e[0],e[1])):live+=delta;overlap=max(overlap,live)
    complete=[]
    for b in selected_businesses:
        js=[j for j in selected if j['business_id']==b['business_id']]
        cps=[frozen.read(OUT/'checkpoints'/(j['job_id']+'.json')) for j in js]
        if not all(cp['status']=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for cp in cps):continue
        content={j['condition']:cp.get('resolved_output',cp['output']) for j,cp in zip(js,cps)}
        frozen.immutable(OUT/'targets'/(b['business_id']+'.json'),frozen.build_target(b,content));complete.append(b['business_id'])
    for rel,sha in preserved.items():assert frozen.file_sha(ROOT/rel)==sha,rel
    calls=len(list((OUT/'request_intents').glob('*.json')));assert calls<=6
    passed=sum(r['passed'] for r in results)
    report={'status':'MIRAI_GUARDED_SIX_REQUEST_TRIAL_PASS' if passed==6 else 'MIRAI_GUARDED_SIX_REQUEST_TRIAL_WITH_FAILURES',
        'passed':passed,'failed':6-passed,'paid_calls':calls,'batch_seconds':seconds,'observed_client_transport_overlap_peak':overlap,
        'results':results,'failures':failures,'returned_models':dict(models),'returned_service_tiers':dict(tiers),
        'usage_returned':dict(usage),'complete_businesses':complete,'condition_aligned_targets':len(complete)*3,
        'coletoken_artifacts_unchanged':True,'payloads_identical':True,'validation_change':'expected_endpoint_provider_binding_only',
        'teacher_content_rewritten':False,'production_integration_approved':False,'automatic_retries_in_bounded_trial':0,
        'held_out_teacher_targets':0,'LED_training_started':False,'Codex_configuration_or_auth_access':False,
        'snapshot_attested':False,'client_overlap_is_not_proof_of_provider_compute_parallelism':True,'completed_utc':frozen.now()}
    frozen.immutable(TRIAL/'report.json',report);publish(report['status'])
    (TRIAL/'TRIAL_REPORT.md').write_text('# Mirai teacher trial with preserved Coletoken runtime\n\n'+
        f"{passed}/6 schema/model/evidence PASS; client overlap={overlap}; wall={seconds}s; complete businesses={len(complete)}.\n\n"+
        'Only endpoint, private provider key and provider provenance binding changed. Frozen API payloads and content validators remain identical. '
        'Existing639-business Coletoken dataset and configuration remain unchanged; restore snapshot is saved separately. '
        'All Mirai results, including failures, are isolated. No production integration, held-out teacher targets or LED training. '
        'Structural PASS is not independent human factual validation; identical model alias is not an attested backend snapshot.\n',encoding='utf-8')
    note(f"Mirai isolated trial result: {passed}/6 strict model/schema/source PASS; overlap={overlap}; wall={seconds}s; "
         f"{len(complete)} complete businesses saved separately. Coletoken639-business runtime unchanged; no integration or automatic scale-up. "
         'Report: outputs/stage_g/teacher_production_d042_v1/mirai_guarded6_after_retry_v1/report.json.')
    print(json.dumps({k:report[k] for k in ('status','passed','failed','paid_calls','batch_seconds','returned_models','complete_businesses')}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        frozen.atomic(TRIAL/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(error).__name__,'updated_utc':frozen.now()})
        print(json.dumps({'runner_error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
