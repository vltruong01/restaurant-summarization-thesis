"""D043 metadata constraint; original D042 runner/transport and frozen target logic."""
import argparse
import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import stage_g_production_d042 as frozen
import stage_g_coletoken_production_v1 as history
import restore_stage_g_d042_runner_v1 as restored
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import stage_g_teacher_stream_diagnostics_v1 as transport
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL
import stage_g_business_id_guard_v1 as guard

ROOT = frozen.ROOT
CONFIG = ROOT / 'configs/stage_g_business_id_guard_production_v1.json'
RUN = frozen.FOLDER / 'business_id_guard_v1'
OUT = RUN / 'execution_v1'
SCRIPT = Path(__file__).resolve()
CHECKED_INPUTS = None


def prepare():
    lock = restored.shared_lock()
    cfg, jobs, businesses, imports = restored.checked_inputs()
    old_state = frozen.read(restored.OUT / 'execution_progress.json')
    assert not old_state['current_jobs'] and old_state.get('stop_reason') == 'User pause flag; checkpoints retained'
    assert not old_state['failed_or_unresolved']
    new_cfg = frozen.read(CONFIG)
    assert all(new_cfg[k] == cfg[k] for k in frozen.FIXED_KEYS)
    prior_index = frozen.read(restored.RUN / 'reuse_index.json')
    index = {}; hashes = {}; groups = defaultdict(list)
    for j in jobs:
        jid = j['job_id']; groups[j['business_id']].append(jid)
        origin = ROOT / prior_index[jid] if jid in prior_index else restored.OUT
        p = origin / 'checkpoints' / (jid + '.json')
        if not p.exists():
            assert not (origin / 'request_intents' / p.name).exists(), 'Unresolved existing request'
            continue
        config = frozen.PILOT_CONFIG if jid in imports else frozen.CONFIG
        frozen.verify_checkpoint(j, cfg, origin, config)
        index[jid] = {'directory':origin.relative_to(ROOT).as_posix(),
                      'config_path':config.relative_to(ROOT).as_posix(), 'request_schema':'original_base_v3'}
        paths = [p]
        if frozen.read(p).get('api_called'):
            paths += [origin/'raw_responses'/p.name, origin/'request_intents'/p.name,
                      origin/'raw_requests'/(jid+'.json.gz')]
        for item in paths: hashes[item.relative_to(ROOT).as_posix()] = frozen.file_sha(item)
    pending = [j for j in jobs if j['job_id'] not in index and not j.get('skip_api_empty_captions')]
    assert len(pending) >= 4
    initial_complete = sum(all(jid in index for jid in ids) for ids in groups.values())
    assert initial_complete == old_state['businesses_completed']
    frozen.immutable(RUN/'reuse_index.json', index)
    frozen.immutable(RUN/'canary_manifest.json', {'job_ids':[j['job_id'] for j in pending[:4]],
        'selection':'first_four_pending_nonempty_jobs_in_frozen_order', 'new_API_calls':4, 'regenerate_completed':False})
    auth = frozen.read(frozen.FOLDER/'authorization.json')
    auth.update(config_sha256=frozen.file_sha(CONFIG), business_id_guard_authorized_by='user_current_request',
                parent_config_sha256=frozen.file_sha(frozen.CONFIG))
    frozen.immutable(RUN/'authorization.json', auth)
    for name in ('small_batch_manifest.json','small_batch_validation.json'):
        destination = RUN/name; destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((frozen.FOLDER/name).read_bytes())
    frozen.immutable(RUN/'manifest.json', {
        'decision':'D-043', 'approved_by':'user', 'config_sha256':frozen.file_sha(CONFIG),
        'reuse_index_sha256':frozen.file_sha(RUN/'reuse_index.json'),
        'initial_complete_businesses':initial_complete, 'initial_reused_components':len(index),
        'maximum_concurrency':4, 'guard_only':'top_level_business_id_single_value_enum',
        'old_requests_verified_under_original_schema':True, 'manual_output_rewriting':False,
        'code_sha256':{rel:frozen.file_sha(ROOT/rel) for rel in (
            'scripts/stage_g_business_id_guard_v1.py','scripts/run_stage_g_business_id_guard_production_v1.py')},
        'preserved_artifact_sha256':hashes, 'prepared_utc':frozen.now()})
    print({'preflight':'PASS','reused_businesses':initial_complete,'reused_components':len(index),'API_calls':0},flush=True)


def checked_inputs():
    global CHECKED_INPUTS
    if CHECKED_INPUTS is not None:
        # One full frozen-input audit per process. Canary and runner use the same
        # verified in-memory jobs; new requests still verify every original photo.
        m = frozen.read(RUN/'manifest.json')
        assert frozen.file_sha(CONFIG) == m['config_sha256']
        assert frozen.file_sha(RUN/'reuse_index.json') == m['reuse_index_sha256']
        for rel, sha in m['code_sha256'].items():
            assert frozen.file_sha(ROOT/rel) == sha, rel
        return CHECKED_INPUTS
    old_cfg, jobs, businesses, imports = restored.checked_inputs()
    cfg = frozen.read(CONFIG); m = frozen.read(RUN/'manifest.json')
    assert all(cfg[k] == old_cfg[k] for k in frozen.FIXED_KEYS)
    assert frozen.file_sha(CONFIG) == m['config_sha256']
    assert frozen.file_sha(RUN/'reuse_index.json') == m['reuse_index_sha256']
    for rel, sha in {**m['code_sha256'], **m['preserved_artifact_sha256']}.items():
        assert frozen.file_sha(ROOT/rel) == sha, rel
    CHECKED_INPUTS = cfg, jobs, businesses, imports
    return CHECKED_INPUTS


def configure():
    import run_stage_g_production_d042 as runner
    index = frozen.read(RUN/'reuse_index.json')

    def location(job, imports):
        return ROOT/index[job['job_id']]['directory'] if job['job_id'] in index else OUT

    def verify(job, cfg, directory, config_path, exact_request=True):
        if job['job_id'] in index:
            return frozen.verify_checkpoint(job,cfg,directory,ROOT/index[job['job_id']]['config_path'],exact_request)
        assert Path(directory) == OUT
        return guard.verify_checkpoint(job,cfg,directory,CONFIG)

    def result(job,cfg,prompt,schema,base,provider,key,directory,config_path):
        return legacy.job_result(job,cfg,prompt,guard.schema_for_job(job,schema),base,provider,key,directory,CONFIG)

    def save_business(b, grouped, imports):
        content = {}
        for j in grouped[b['business_id']]:
            p=location(j,imports)/'checkpoints'/(j['job_id']+'.json')
            if not p.exists(): return None
            cp=frozen.read(p)
            if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW': return None
            content[j['condition']]=cp.get('resolved_output',cp['output'])
        target=frozen.build_target(b,content)
        for directory in (frozen.PILOT_OUT/'provisional_targets',history.ORIGINAL_EXEC/'targets',history.EXEC/'targets',restored.OUT/'targets'):
            p=directory/(b['business_id']+'.json')
            if p.exists(): assert frozen.read(p)==target; return p
        p=OUT/'targets'/(b['business_id']+'.json'); frozen.immutable(p,target); return p

    def docs(state):
        frozen.atomic(history.POINTER, {'status':state['status'],'provider':'coletoken','requested_model':MODEL,
            'production_directory':RUN.relative_to(ROOT).as_posix(),
            'progress_path':(OUT/'execution_progress.json').relative_to(ROOT).as_posix(),
            'maximum_concurrency':4,'business_id_guard':'single_value_enum_v1',
            'execution_entrypoint':SCRIPT.relative_to(ROOT).as_posix(),'updated_utc':state['updated_utc'],
            'pause_file':(RUN/'STOP_REQUESTED.json').relative_to(ROOT).as_posix()})
        p=ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md'; marker='**D-042 live production status:**'
        lines=[s for s in p.read_text(encoding='utf-8').splitlines(keepends=True) if not s.startswith(marker)]
        initial=frozen.read(RUN/'manifest.json')['initial_complete_businesses']
        note=(f"{marker} {state['businesses_completed']}/2200 businesses, {state['targets_completed']} targets; "
              f"status={state['status']}; stop={state.get('stop_reason')}; D043 exact-business-ID request enum guard. "
              f"{initial} completed businesses reused under original schema; original D042 four-request batches/Coletoken transport. "
              "Authoritative pointer: outputs/stage_g/active_production_execution_pointer.json. "
              "No teacher text rewriting, no held-out targets or LED training; structural checks are not factual certification.\n")
        lines.insert(1,'\n'+note+'\n'); p.write_text(''.join(lines),encoding='utf-8')

    runner.CONFIG=CONFIG; runner.FOLDER=RUN; runner.OUT=OUT
    runner.check_inputs=checked_inputs; runner.location=location; runner.verify_checkpoint=verify
    runner.save_completed_business=save_business; runner.job_result=result; runner.payload=guard.payload_for_job
    runner.update_docs=docs; runner.runner_lock=restored.shared_lock
    old_run=subprocess.run
    def redirect_audit(command,*args,**kwargs):
        if len(command)>1 and Path(command[1]).name=='audit_stage_g_production_d042.py':
            command=[sys.executable,str(SCRIPT),'--audit']+command[2:]
        return old_run(command,*args,**kwargs)
    runner.subprocess.run=redirect_audit
    return runner, location, verify, save_business


def canary(runner):
    lock=restored.shared_lock()
    cfg,jobs,businesses,imports=checked_inputs()
    selected_ids=frozen.read(RUN/'canary_manifest.json')['job_ids']
    selected=[next(j for j in jobs if j['job_id']==jid) for jid in selected_ids]
    report_path=RUN/'guard_small_batch_report.json'
    if report_path.exists():
        assert frozen.read(report_path)['status']=='STRICT_BUSINESS_ID_ENUM_CANARY_PASS'
        for j in selected: guard.verify_checkpoint(j,cfg,OUT,CONFIG)
        return
    assert not (RUN/'STOP_REQUESTED.json').exists()
    schemas=frozen.read(ROOT/cfg['schema']); prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    accepted=frozen.read(RUN/'reuse_index.json'); initial=frozen.read(RUN/'manifest.json')['initial_complete_businesses']
    state={'status':'RUNNING_GUARD_CANARY','phase':'business_id_enum_canary','pid':os.getpid(),
           'businesses_completed':initial,'targets_completed':initial*3,'accepted_tasks':len(accepted),
           'businesses_planned':2200,'current_jobs':selected_ids[:],'maximum_concurrency':4,
           'new_API_attempts':0,'failed_or_unresolved':[],'stop_reason':None,'updated_utc':frozen.now()}
    frozen.atomic(OUT/'execution_progress.json',state); runner.update_docs(state)
    base,provider,key=credentials(); assert (base,provider,cfg['requested_model'])==(BASE_URL,PROVIDER,MODEL)
    legacy.call=transport.call
    def component(j):
        cp=OUT/'checkpoints'/(j['job_id']+'.json')
        if cp.exists(): return guard.verify_checkpoint(j,cfg,OUT,CONFIG)
        assert not (OUT/'request_intents'/cp.name).exists(), 'Unresolved guard canary; no automatic resend'
        schema=schemas[j['task']]; prompt=prompts[j['task']]
        data=frozen.canonical(guard.payload_for_job(j,cfg,prompt,schema))
        p=OUT/'raw_requests'/(j['job_id']+'.json.gz'); p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists(): assert gzip.decompress(p.read_bytes())==data
        else: p.write_bytes(gzip.compress(data,mtime=0))
        transport.set_context(OUT/'stream_diagnostics',j['job_id'])
        legacy.job_result(j,cfg,prompt,guard.schema_for_job(j,schema),base,provider,key,OUT,CONFIG)
        return guard.verify_checkpoint(j,cfg,OUT,CONFIG)
    results=[]; failed=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        running={pool.submit(component,j):j for j in selected}
        for f in as_completed(running):
            j=running[f]
            try:
                cp=f.result(); results.append({'job_id':j['job_id'],'expected_business_id':j['business_id'],
                    'returned_business_id':cp['output']['business_id'],'request_schema_sha256':cp['binding']['schema_sha256'],
                    'response_id':cp.get('response_id'),'reported_model':cp.get('reported_model'),'usage':cp.get('usage')})
            except Exception as e: failed.append({'job_id':j['job_id'],'error_type':type(e).__name__})
            state.update(current_jobs=[x for x in state['current_jobs'] if x!=j['job_id']],
                accepted_tasks=len(accepted)+len(results),failed_or_unresolved=failed[:],
                new_API_attempts=len(list((OUT/'request_intents').glob('*.json'))),updated_utc=frozen.now())
            frozen.atomic(OUT/'execution_progress.json',state); runner.update_docs(state)
    report={'status':'GUARD_CANARY_FAILED_REQUIRES_REVIEW' if failed else 'STRICT_BUSINESS_ID_ENUM_CANARY_PASS',
            'passed':len(results),'failed':len(failed),'results':results,'failures':failed,
            'schema_guard_field':'business_id','base_schema_changed':False,'manual_output_rewrite':False,
            'provider_support_observed_for_validated_calls_only':True,
            'does_not_guarantee_future_provider_behavior_or_factual_accuracy':True,'completed_utc':frozen.now()}
    frozen.immutable(report_path,report)
    assert not failed, 'Guard canary failed: do not silently downgrade the schema'
    print(json.dumps({'guard_canary':'PASS','API_calls':len(results),'exact_ID_checks':len(results)}),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--audit',action='store_true')
    args=p.parse_args()
    if args.prepare: prepare(); return
    runner,location,verify,save_business=configure()
    try:
        if args.audit:
            import audit_stage_g_production_d042 as audit
            audit.CONFIG=CONFIG; audit.FOLDER=RUN; audit.OUT=OUT; audit.check_inputs=checked_inputs
            audit.location=location; audit.verify_checkpoint=verify; audit.save_completed_business=save_business
            audit.run(smoke=False)
        else:
            canary(runner)
            runner.main()
    except Exception as e:
        frozen.atomic(RUN/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(e).__name__,
                      'updated_utc':frozen.now(),'automatic_retries':0})
        p=OUT/'execution_progress.json'
        if p.exists():
            s=frozen.read(p); s.update(status='STOPPED_REQUIRES_REVIEW',current_jobs=[],
                 stop_reason='Guard/protocol failure: '+type(e).__name__,updated_utc=frozen.now())
            frozen.atomic(p,s); runner.update_docs(s)
        raise


if __name__=='__main__': main()
