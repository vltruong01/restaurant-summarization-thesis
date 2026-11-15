"""Resume frozen teacher production; defer individual failures without editing outputs."""
import argparse
import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import stage_g_production_d042 as frozen
import continue_stage_g_guarded_production6_v1 as parent
import stage_g_business_id_guard_v1 as guard
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
from stage_g_teacher_credentials import BASE_URL, PROVIDER, MODEL, KEY_VARIABLE

ROOT = frozen.ROOT
CONFIG = ROOT/'configs/stage_g_production6_deferred_v1.json'
RUN = frozen.FOLDER/'guarded_production6_deferred_v1'
OUT = RUN/'execution_v1'
SCRIPT = Path(__file__).resolve()
REVIEWED_RETRY = 'train_1mLyYz0skNlRGP6wXBEmHQ_review_consensus'
CACHE = None
INDEX = None
TARGETS = None


def read_index():
    global INDEX
    if INDEX is None: INDEX = frozen.read(RUN/'reuse_index.json')
    return INDEX


def check_science(cfg, previous):
    assert all(cfg[k] == previous[k] for k in frozen.FIXED_KEYS)
    assert {k:v for k,v in cfg['identifier_guard'].items() if k!='provider_ignores_guard'} == {
        k:v for k,v in previous['identifier_guard'].items() if k!='provider_ignores_guard'}
    assert cfg['execution']['maximum_concurrency'] == 6
    assert cfg['execution']['automatic_retries'] == 0


def prepare():
    assert not (RUN/'manifest.json').exists(), 'Prepared already: use --run'
    previous, jobs, businesses, imports = parent.checked_inputs()
    cfg = frozen.read(CONFIG); check_science(cfg, previous)
    state = frozen.read(parent.OUT/'execution_progress.json')
    assert state['status']=='STOPPED_REQUIRES_REVIEW' and not state['current_jobs']
    index = dict(parent.read_index()); hashes = {}; failed = []
    for j in jobs:
        directory = parent.location(j, imports)
        path = directory/'checkpoints'/(j['job_id']+'.json')
        if not path.exists(): continue
        cp = frozen.read(path)
        if cp['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            assert j['job_id']==REVIEWED_RETRY
            failed.append(j['job_id']); continue
        parent.verify_checkpoint(j, previous, directory, parent.CONFIG)
        if j['job_id'] not in index:
            index[j['job_id']] = {'directory': directory.relative_to(ROOT).as_posix(),
                'config_path': parent.CONFIG.relative_to(ROOT).as_posix(), 'request_schema':'business_id_enum_v1'}
    assert failed==[REVIEWED_RETRY] and len(index)==state['accepted_tasks']==508
    for directory in (parent.OUT, parent.RECOVERY):
        for path in directory.rglob('*'):
            if path.is_file(): hashes[path.relative_to(ROOT).as_posix()] = frozen.file_sha(path)
    targets = dict(frozen.read(parent.RUN/'reuse_targets.json'))
    for path in (parent.OUT/'targets').glob('*.json'):
        assert path.stem not in targets
        targets[path.stem] = path.relative_to(ROOT).as_posix()
    assert len(targets)==state['businesses_completed']==169
    for rel in targets.values(): hashes[rel] = frozen.file_sha(ROOT/rel)
    # The one reviewed recovery repeats the exact original API payload in a new namespace.
    job = next(j for j in jobs if j['job_id']==REVIEWED_RETRY)
    prompt = (ROOT/cfg['prompts'][job['task']]).read_text(encoding='utf-8')
    schema = frozen.read(ROOT/cfg['schema'])[job['task']]
    original = parent.OUT/'raw_requests'/(REVIEWED_RETRY+'.json.gz')
    assert gzip.decompress(original.read_bytes()) == frozen.canonical(guard.payload_for_job(job,cfg,prompt,schema))
    frozen.immutable(RUN/'reuse_index.json',index)
    frozen.immutable(RUN/'reuse_targets.json',targets)
    frozen.immutable(RUN/'reviewed_retry_authorization.json', {'job_id':REVIEWED_RETRY,
        'maximum_new_attempts':1,'same_payload':True,'original_failure_preserved':True,
        'reason':'Wrong-review evidence span: REV_6_SEG_2 grouped under REV_1; strict validator rejected',
        'authorized_by':'user instruction fix and continue; defer subsequent request failures',
        'targets_or_labels_manually_edited':False})
    auth = frozen.read(parent.RUN/'authorization.json')
    auth.update(config_sha256=frozen.file_sha(CONFIG), maximum_concurrency=6,
        reviewed_retry_job=REVIEWED_RETRY,maximum_reviewed_retry_calls=1,
        individual_failure_policy='defer_continue_others')
    frozen.immutable(RUN/'authorization.json',auth)
    for name in ('small_batch_manifest.json','small_batch_validation.json'):
        path=RUN/name; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write((parent.RUN/name).read_bytes())
    code=[SCRIPT,ROOT/'scripts/run_stage_g_production6_deferred_v1.py',ROOT/'scripts/stage_g_business_id_guard_v1.py']
    frozen.immutable(RUN/'manifest.json', {'approved_by':'user','scientific_settings_changed':False,
        'initial_complete_businesses':169,'initial_reused_components':508,'automatic_retries':0,
        'maximum_concurrency':6,'failure_policy':'defer_individual_task_continue_others',
        'reviewed_retry_job':REVIEWED_RETRY,'maximum_reviewed_retry_calls':1,
        'config_sha256':frozen.file_sha(CONFIG),'reuse_index_sha256':frozen.file_sha(RUN/'reuse_index.json'),
        'reuse_targets_sha256':frozen.file_sha(RUN/'reuse_targets.json'),
        'code_sha256':{p.relative_to(ROOT).as_posix():frozen.file_sha(p) for p in code},
        'preserved_artifact_sha256':hashes,'prepared_utc':frozen.now(),
        'Codex_configuration_or_auth_access':False})
    print(json.dumps({'offline_preflight':'PASS','reused_components':508,'reused_complete_businesses':169,'API_calls':0}),flush=True)


def checked_inputs():
    global CACHE
    manifest = frozen.read(RUN/'manifest.json')
    for path,key in ((CONFIG,'config_sha256'),(RUN/'reuse_index.json','reuse_index_sha256'),(RUN/'reuse_targets.json','reuse_targets_sha256')):
        assert frozen.file_sha(path)==manifest[key]
    for rel,sha in manifest['code_sha256'].items(): assert frozen.file_sha(ROOT/rel)==sha,rel
    if CACHE is None:
        old,jobs,businesses,imports=parent.checked_inputs()
        cfg=frozen.read(CONFIG);check_science(cfg,old)
        for rel,sha in manifest['preserved_artifact_sha256'].items(): assert frozen.file_sha(ROOT/rel)==sha,rel
        for j in jobs:
            if j['job_id'] in read_index(): parent.historical_verify(j,read_index()[j['job_id']])
        CACHE=cfg,jobs,businesses,imports
    return CACHE


def location(job,imports):
    entry=read_index().get(job['job_id'])
    return ROOT/entry['directory'] if entry else OUT


def verify_checkpoint(job,cfg,directory,config_path,exact_request=True):
    entry=read_index().get(job['job_id'])
    if entry: return parent.historical_verify(job,entry)
    assert Path(directory)==OUT
    return guard.verify_checkpoint(job,cfg,OUT,CONFIG)


def save_business(b,grouped,imports):
    global TARGETS
    content={}
    for j in grouped[b['business_id']]:
        path=location(j,imports)/'checkpoints'/(j['job_id']+'.json')
        if not path.exists(): return None
        cp=frozen.read(path)
        if cp['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW': return None
        content[j['condition']]=cp.get('resolved_output',cp['output'])
    target=frozen.build_target(b,content)
    if TARGETS is None: TARGETS=frozen.read(RUN/'reuse_targets.json')
    if b['business_id'] in TARGETS:
        path=ROOT/TARGETS[b['business_id']];assert frozen.read(path)==target;return path
    path=OUT/'targets'/(b['business_id']+'.json')
    frozen.immutable(path,target);return path


def update_docs(state):
    state.update(initial_reused_complete_businesses=169,initial_reused_components=508,
        deferred_task_count=len(state.get('failed_or_unresolved',[])),failure_policy='defer_individual_task_continue_others')
    frozen.atomic(OUT/'execution_progress.json',state)
    frozen.atomic(ROOT/'outputs/stage_g/active_production_execution_pointer.json',{
        'status':state['status'],'provider':PROVIDER,'requested_model':MODEL,
        'production_directory':RUN.relative_to(ROOT).as_posix(),
        'progress_path':(OUT/'execution_progress.json').relative_to(ROOT).as_posix(),
        'deferred_tasks_path':(OUT/'deferred_tasks.json').relative_to(ROOT).as_posix(),
        'maximum_concurrency':6,'execution_entrypoint':SCRIPT.relative_to(ROOT).as_posix(),
        'pause_file':(RUN/'STOP_REQUESTED.json').relative_to(ROOT).as_posix(),'updated_utc':state['updated_utc']})
    marker='**D-042 live production status:**'
    path=ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md'
    lines=[l for l in path.read_text(encoding='utf-8').splitlines(keepends=True) if not l.startswith(marker)]
    note=(f"{marker} {state['businesses_completed']}/2200 businesses; {state['targets_completed']} targets; "
        f"status={state['status']}; phase={state['phase']}; deferred tasks={state['deferred_task_count']}; "
        "max6 requests, failed/unresolved requests quarantined without automatic resend; other requests continue. "
        "169 businesses/508 accepted components reused; one reviewed same-payload API retry for1mLy spans. "
        "D040/D043/frozen science unchanged; API-only targets, no held-out teacher targets or LED training. "
        "Authoritative pointer: outputs/stage_g/active_production_execution_pointer.json. Structural PASS is not human factual certification.\n")
    lines.insert(1,'\n'+note+'\n');path.write_text(''.join(lines),encoding='utf-8')


def configure():
    import run_stage_g_production6_deferred_v1 as runner
    def result(job,cfg,prompt,schema,base,provider,key,directory,config_path):
        cp=legacy.job_result(job,cfg,prompt,guard.schema_for_job(job,schema),base,provider,key,directory,CONFIG)
        return cp
    runner.CONFIG=CONFIG;runner.FOLDER=RUN;runner.OUT=OUT
    runner.check_inputs=checked_inputs;runner.location=location;runner.verify_checkpoint=verify_checkpoint
    runner.save_completed_business=save_business;runner.job_result=result;runner.payload=guard.payload_for_job
    runner.update_docs=update_docs
    old_run=subprocess.run
    def redirect(command,*args,**kwargs):
        if len(command)>1 and Path(command[1]).name=='audit_stage_g_production_d042.py':
            command=[sys.executable,str(SCRIPT),'--audit']+command[2:]
        return old_run(command,*args,**kwargs)
    runner.subprocess.run=redirect
    return runner


def main():
    parser=argparse.ArgumentParser()
    for name in ('prepare','run','prepare-and-run','audit'): parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args();assert sum(vars(args).values())==1
    lock=None if args.audit else parent.parent.restored.shared_lock()
    if args.prepare or args.prepare_and_run: prepare()
    if args.prepare:return
    runner=configure()
    if args.audit:
        import audit_stage_g_production_d042 as audit
        audit.CONFIG=CONFIG;audit.FOLDER=RUN;audit.OUT=OUT;audit.check_inputs=checked_inputs
        audit.location=location;audit.verify_checkpoint=verify_checkpoint;audit.save_completed_business=save_business
        import run_stage_g_production_d042 as original
        original.update_docs=update_docs
        # The spawning process holds the OS lock throughout final audit.
        audit.run(smoke=False);return
    assert not os.environ.get(KEY_VARIABLE),'Private teacher credentials only'
    runner.runner_lock=lambda:lock
    runner.main()


if __name__=='__main__':
    try:main()
    except Exception as error:
        frozen.atomic(RUN/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(error).__name__,
            'updated_utc':frozen.now(),'automatic_retries':0})
        path=OUT/'execution_progress.json'
        if path.exists():
            state=frozen.read(path);state.update(status='STOPPED_REQUIRES_REVIEW',current_jobs=[],
                stop_reason='Local integrity/preflight/runner failure: '+type(error).__name__,updated_utc=frozen.now())
            update_docs(state)
        print(json.dumps({'status':'STOPPED_REQUIRES_REVIEW','error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
