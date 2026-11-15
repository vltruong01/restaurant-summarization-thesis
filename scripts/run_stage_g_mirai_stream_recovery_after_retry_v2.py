"""Bounded Mirai SSE recovery trial; content validation never overrides provenance."""
import gzip
import json
import os
import threading
import time
import urllib.request
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path

import stage_g_production_d042 as frozen
import run_stage_g_mirai_guarded6_after_retry_v1 as prior
import stage_g_mirai_guard_v1 as guard
import stage_g_mirai_stream_adapter_v5 as transport
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import restore_stage_g_d042_runner_v1 as restored
from stage_g_canonical_review_ids import resolve_review,citation_diagnostics
from stage_g_teacher_credentials import key_file,KEY_VARIABLE,MODEL

ROOT=frozen.ROOT
CONFIG=ROOT/'configs/stage_g_mirai_stream_recovery_after_retry_v2.json'
RUN=frozen.FOLDER/'mirai_stream_recovery_after_retry_v2'
OUT=RUN/'execution_v1'
SCRIPT=Path(__file__).resolve()
BASE_URL=prior.BASE_URL
PROVIDER=prior.PROVIDER


def note(value):
    with (ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md').open('a',encoding='utf-8') as stream:stream.write('\n\n'+value+'\n')


def validate_content(job,cfg,response):
    if response.get('status')!='completed':raise ValueError('Provider response is not complete')
    output=json.loads(legacy.response_text(response))
    schema=frozen.read(ROOT/cfg['schema'])[job['task']]
    validation=guard.validate_output(job,output,schema)
    resolved,evidence=resolve_review(job,output) if job['task']=='review_consensus' else (output,[])
    return {'output':output,'resolved_output':resolved,'validation':validation,'evidence_resolution':evidence,
        'citation_metadata_diagnostics':citation_diagnostics(job,output) if job['task']=='review_consensus' else []}


def model_state(response,diagnostic,requested):
    returned=response.get('model')
    observations=diagnostic.get('provider_model_observations',[])
    values={o['value'] for o in observations}
    if returned is not None and returned!=requested:return 'MODEL_MISMATCH',False
    if any(v!=requested for v in values):return 'MODEL_OBSERVATION_CONFLICT',False
    if returned==requested:return 'MODEL_FIELD_MATCHES_REQUEST',True
    # Header/event observations are preserved but do not silently relax the frozen response.model requirement.
    return ('MODEL_OBSERVED_OUTSIDE_FINAL_RESPONSE_REQUIRES_REVIEW' if values else 'MODEL_METADATA_MISSING'),False


def main():
    lock=restored.shared_lock()
    assert not (RUN/'manifest.json').exists(),'Versioned attempt already exists; no automatic rerun'
    base,jobs,businesses,imports=frozen.check_inputs()
    cfg=frozen.read(CONFIG);previous=frozen.read(prior.CONFIG);manifest=frozen.read(prior.TRIAL/'manifest.json')
    assert all(cfg[k]==base[k]==previous[k] for k in frozen.FIXED_KEYS)
    assert cfg['identifier_guard']==previous['identifier_guard']
    assert cfg['execution']['maximum_concurrency']==cfg['operational_provider_trial']['maximum_calls']==6
    assert not cfg['full_generation_enabled']
    assert frozen.file_sha(prior.CONFIG)==manifest['config_sha256']
    by_id={j['job_id']:j for j in jobs};selected=[by_id[jid] for jid in manifest['job_ids']]
    assert len(selected)==len({j['job_id'] for j in selected})==6
    assert all(j['split'] in ('train','validation') and not j.get('skip_api_empty_captions') for j in selected)
    current=frozen.read(prior.parent.OUT/'execution_progress.json')
    assert current['status']=='STOPPED_REQUIRES_REVIEW' and not current['current_jobs']
    preserved=frozen.read(prior.TRIAL/'coletoken_restore_snapshot/RESTORE_STATE.json')['preserved_artifact_sha256']
    for p in prior.TRIAL.rglob('*'):
        if p.is_file():preserved[p.relative_to(ROOT).as_posix()]=frozen.file_sha(p)
    for rel,sha in preserved.items():assert frozen.file_sha(ROOT/rel)==sha,rel
    schemas=frozen.read(ROOT/cfg['schema']);prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    for j in selected:
        body=frozen.canonical(guard.payload_for_job(j,cfg,prompts[j['task']],schemas[j['task']]))
        assert body==gzip.decompress((prior.OUT/'raw_requests'/(j['job_id']+'.json.gz')).read_bytes())
        p=OUT/'raw_requests'/(j['job_id']+'.json.gz');p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as stream:stream.write(gzip.compress(body,mtime=0))
    frozen.immutable(RUN/'manifest.json',{'approved_by':'user','authorization':'Handle Mirai SSE compatibility after failed six-request trial',
        'maximum_calls':6,'maximum_concurrency':6,'automatic_retries':0,'job_ids':manifest['job_ids'],
        'business_ids':manifest['business_ids'],'scientific_settings_changed':False,'identical_prior_requests':True,
        'config_sha256':frozen.file_sha(CONFIG),'code_sha256':{p.relative_to(ROOT).as_posix():frozen.file_sha(p) for p in
            (SCRIPT,ROOT/'scripts/stage_g_mirai_stream_adapter_v5.py',ROOT/'scripts/stage_g_mirai_guard_v1.py')},
        'preserved_artifact_sha256':preserved,'model_metadata_policy':'no_fabrication_no_automatic_production_acceptance_if_missing',
        'overall_deadline_seconds':180,'socket_idle_seconds':120,'prepared_utc':frozen.now(),
        'production_integration_approved':False,'Codex_config_or_auth_access':False})
    assert not os.environ.get(KEY_VARIABLE),'Private teacher secret only'
    key=key_file().with_name(cfg['execution']['dedicated_secret_filename']).read_text(encoding='utf-8-sig').strip()
    assert key and not any(c.isspace() for c in key)
    note('User-authorized Mirai SSE compatibility refinement, isolated six-call recovery: reuse the same six prior jobs and byte-identical '
         'D040/D043 payloads; preserve all existing trials and the stopped639-business Coletoken state. New adapter archives the full SSE '
         'wire and reconstructs only its output envelope from exact completed provider text/items, with delta/item/final consistency checks '
         'and replay verification. Frozen120s idle/180s overall deadlines unchanged. No claim/ID/quotation alias repair or manual target editing. '
         'Missing/mismatched model metadata cannot pass the frozen production guard; content validation and provenance are separately reported. '
         'Candidate targets, if created, remain non-production. Namespace: outputs/stage_g/teacher_production_d042_v1/mirai_stream_recovery_after_retry_v2. '
         'No full production, held-out targets, Codex auth/config changes or LED training.')
    print(json.dumps({'offline_preflight':'PASS','planned_calls':6,'same_jobs_same_payload':True,
        'teacher_model':MODEL,'original_production_businesses':current['businesses_completed']}),flush=True)
    active=set();results=[];mutex=threading.RLock();barrier=threading.Barrier(6)
    def publish(status='RUNNING_MIRAI_SSE_RECOVERY_TRIAL'):
        with mutex:
            state={'status':status,'provider':PROVIDER,'requested_model':MODEL,'current_jobs':sorted(active),
                'completed_requests':len(results),'maximum_concurrency':6,'pid':os.getpid(),'results':results[:],
                'production_unchanged':True,'updated_utc':frozen.now()}
            frozen.atomic(RUN/'execution_progress.json',state)
            frozen.atomic(ROOT/'outputs/stage_g/active_teacher_trial_pointer.json',{'status':status,'provider':PROVIDER,
                'trial_directory':RUN.relative_to(ROOT).as_posix(),'progress_path':(RUN/'execution_progress.json').relative_to(ROOT).as_posix(),
                'maximum_concurrency':6,'production_resume_after_trial':False,'updated_utc':state['updated_utc']})
    def worker(j):
        jid=j['job_id'];directory=OUT/'stream_diagnostics'
        transport.set_context(directory,jid);barrier.wait(timeout=30)
        with mutex:active.add(jid);publish()
        binding={'job_sha256':frozen.digest(frozen.canonical(j)),'config_sha256':frozen.file_sha(CONFIG),
                 'prompt_sha256':j['prompt_sha256'],'schema_sha256':frozen.digest(frozen.canonical(guard.schema_for_job(j,schemas[j['task']]))),
                 'base_url':BASE_URL,'provider':PROVIDER,'requested_model':MODEL}
        try:
            body=guard.payload_for_job(j,cfg,prompts[j['task']],schemas[j['task']])
            frozen.immutable(OUT/'request_intents'/(jid+'.json'),{'started_utc':frozen.now(),'binding':binding,
                'request_sha256':frozen.digest(frozen.canonical(body)),'automatic_retries':0})
            response=transport.call(BASE_URL,key,body)
            response_path=OUT/'projected_responses'/(jid+'.json');frozen.immutable(response_path,response)
            wire=frozen.read(directory/(jid+'.completed_wire.json'))
            replay,lineage=transport.replay(directory/(jid+'.sse.txt.gz'),wire)
            assert replay==response,'Stored projected response differs from exact provider stream'
            diagnostic=frozen.read(directory/(jid+'.json'))
            provenance,model_ok=model_state(response,diagnostic,MODEL)
            try:
                content=validate_content(j,cfg,response)
            except Exception as error:
                frozen.immutable(OUT/'checkpoints'/(jid+'.json'),{'status':'CONTENT_VALIDATION_FAILED_NOT_PRODUCTION',
                    'binding':binding,'content_valid':False,'model_provenance':provenance,'reported_model':response.get('model'),
                    'error_type':type(error).__name__,'local_validation_error':str(error) if isinstance(error,ValueError) else None,
                    'source_response':response_path.relative_to(ROOT).as_posix(),'source_response_sha256':frozen.file_sha(response_path),
                    'SSE_replay_match':True,'production_accepted':False,'teacher_text_rewritten':False})
                return {'job_id':jid,'content_valid':False,'model_provenance':provenance,'production_accepted':False}
            status='CONTENT_AND_MODEL_VALID_ISOLATED_NOT_PRODUCTION' if model_ok else 'CONTENT_VALID_MODEL_PROVENANCE_PENDING_NOT_PRODUCTION'
            frozen.immutable(OUT/'checkpoints'/(jid+'.json'),{'status':status,'binding':binding,'content_valid':True,
                'model_provenance':provenance,'model_guard_satisfied':model_ok,'reported_model':response.get('model'),
                'provider_model_observations':diagnostic.get('provider_model_observations',[]),**content,
                'source_response':response_path.relative_to(ROOT).as_posix(),'source_response_sha256':frozen.file_sha(response_path),
                'SSE_replay_match':True,'output_projection':lineage,'production_accepted':False,'teacher_text_rewritten':False,
                'usage':response.get('usage'),'response_id':response.get('id')})
            return {'job_id':jid,'content_valid':True,'model_guard_satisfied':model_ok,'model_provenance':provenance,
                'seconds':diagnostic['elapsed_seconds'],'production_accepted':False}
        except Exception as error:
            frozen.immutable(OUT/'checkpoints'/(jid+'.json'),{'status':'TRANSPORT_OR_REPLAY_FAILED_NOT_PRODUCTION',
                'binding':binding,'content_valid':False,'error_type':type(error).__name__,'production_accepted':False})
            return {'job_id':jid,'content_valid':False,'error_type':type(error).__name__,'production_accepted':False}
        finally:
            with mutex:active.discard(jid);publish()
    started=time.monotonic();publish()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(worker,j) for j in selected]):
            result=future.result()
            with mutex:results.append(result);publish()
            print(json.dumps(result),flush=True)
    candidates=[];usage=Counter();metadata=Counter();events=[];details=[]
    for j in selected:
        jid=j['job_id'];cp=frozen.read(OUT/'checkpoints'/(jid+'.json'))
        details.append({'job_id':jid,'status':cp['status'],'model_provenance':cp.get('model_provenance'),
            'local_validation_error':cp.get('local_validation_error'),'reported_model':cp.get('reported_model')})
        metadata[cp.get('model_provenance','transport_failed')]+=1
        d=OUT/'stream_diagnostics'/(jid+'.json')
        if d.exists():
            diagnostic=frozen.read(d)
            events += [(datetime.fromisoformat(diagnostic['started_utc']).timestamp(),1),
                       (datetime.fromisoformat(diagnostic['finished_utc']).timestamp(),-1)]
        p=OUT/'projected_responses'/(jid+'.json')
        if p.exists():
            u=frozen.read(p).get('usage') or {};usage['responses_with_usage']+=bool(u)
            usage['input_tokens']+=u.get('input_tokens',0);usage['output_tokens']+=u.get('output_tokens',0)
            usage['reasoning_tokens']+=(u.get('output_tokens_details') or {}).get('reasoning_tokens',0)
    for b in businesses:
        if b['business_id'] not in manifest['business_ids']:continue
        js=[j for j in selected if j['business_id']==b['business_id']]
        cps=[frozen.read(OUT/'checkpoints'/(j['job_id']+'.json')) for j in js]
        if not all(cp.get('content_valid') for cp in cps):continue
        target=frozen.build_target(b,{j['condition']:cp['resolved_output'] for j,cp in zip(js,cps)})
        target.update(status='ISOLATED_MIRAI_REVIEW_CANDIDATE_NOT_PRODUCTION',production_accepted=False,
            model_provenance={j['condition']:cp['model_provenance'] for j,cp in zip(js,cps)},
            independent_human_verified=False)
        frozen.immutable(OUT/'review_candidates'/(b['business_id']+'.json'),target);candidates.append(b['business_id'])
    for rel,sha in preserved.items():assert frozen.file_sha(ROOT/rel)==sha,rel
    live=peak=0
    for epoch,delta in sorted(events,key=lambda e:(e[0],e[1])):live+=delta;peak=max(peak,live)
    count=sum(r.get('content_valid',False) for r in results);model_count=sum(r.get('model_guard_satisfied',False) for r in results)
    report={'status':'MIRAI_SSE_RECOVERY_CONTENT_VALID_PENDING_RESEARCH_REVIEW' if count==6 else 'MIRAI_SSE_RECOVERY_WITH_CONTENT_OR_TRANSPORT_FAILURES',
        'API_calls':len(list((OUT/'request_intents').glob('*.json'))),'content_valid_components':count,
        'content_or_transport_failures':6-count,'model_guard_satisfied_components':model_count,
        'model_provenance_distribution':dict(metadata),'candidate_businesses':candidates,'production_accepted_components':0,
        'batch_seconds':round(time.monotonic()-started,3),'observed_client_transport_overlap_peak':peak,
        'usage_returned':dict(usage),'results':results,'component_details':details,
        'same_jobs_same_payloads':True,'teacher_text_rewritten':False,'all_previous_artifacts_unchanged':True,
        'frozen_model_requirement_relaxed':False,'independent_human_factual_validation':False,'Codex_config_or_auth_access':False,
        'held_out_teacher_targets':0,'LED_training_started':False,'completed_utc':frozen.now()}
    frozen.immutable(RUN/'report.json',report);publish(report['status'])
    text='# Mirai SSE recovery trial\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n\n'
    text+='Exact provider text is reconstructed only from recorded SSE events; original wire response and projection are separate. '
    text+='Content/schema/evidence PASS is separate from model provenance and independent factual assessment. '
    text+='Missing model metadata is never invented; no component or candidate target has been admitted to production. '
    text+='All previous639-business Coletoken outputs and original Mirai trial remain unchanged. No training or full production started.\n'
    (RUN/'TRIAL_REPORT.md').write_text(text,encoding='utf-8')
    note(f"Mirai SSE recovery result: {count}/6 content/schema/evidence valid; {model_count}/6 satisfy frozen returned-model requirement; "
         f"{len(candidates)} isolated candidate businesses;0 production integrations. Model provenance: {dict(metadata)}. "
         'Previous Coletoken639-business state and prior trials unchanged. Report: outputs/stage_g/teacher_production_d042_v1/mirai_stream_recovery_after_retry_v2/report.json.')
    print(json.dumps({k:report[k] for k in ('status','API_calls','content_valid_components','model_guard_satisfied_components','batch_seconds','candidate_businesses')}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        frozen.atomic(RUN/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(error).__name__,'updated_utc':frozen.now()})
        print(json.dumps({'runner_error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
