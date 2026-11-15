"""Versioned D042 six-worker drained batches; deferred individual failures, frozen validation, zero automatic retries."""
import gzip
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from stage_g_production_d042 import (
    ROOT, CONFIG, FOLDER, OUT, PILOT_OUT, read, file_sha, canonical, immutable, atomic,
    check_inputs, location, verify_checkpoint, save_completed_business, now,
)
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL
from run_stage_g_teacher_pilot_v4_1_recovery import job_result, payload


def runner_lock():
    """OS lock is released on exit/crash; concurrent resume cannot duplicate requests."""
    OUT.mkdir(parents=True, exist_ok=True)
    handle = (OUT / 'runner.lock').open('a+b')
    if handle.tell() == 0: handle.write(b'0'); handle.flush()
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


STATE = None


def progress(jobs, businesses, imports, active, phase, stop=None, changed_jobs=None):
    global STATE
    first = STATE is None
    if first:
        STATE = {'accepted':set(), 'failures':{}, 'complete':set(), 'usage':Counter(), 'seen_raw':set()}
    accepted=STATE['accepted']; failure_map=STATE['failures']; usage=STATE['usage']; complete=STATE['complete']
    by_business = defaultdict(list)
    for j in jobs:
        by_business[j['business_id']].append(j)
    inspect = jobs if first else (changed_jobs or [])
    for j in inspect:
        directory = location(j, imports)
        p = directory / 'checkpoints' / (j['job_id'] + '.json')
        if p.exists():
            cp = read(p)
            if cp['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                accepted.add(j['job_id']); failure_map.pop(j['job_id'], None)
            else: failure_map[j['job_id']]={'job_id': j['job_id'], 'status': cp['status'], 'error_type': cp.get('error_type')}
        elif (directory / 'request_intents' / p.name).exists() and j['job_id'] not in active:
            failure_map[j['job_id']]={'job_id': j['job_id'], 'status': 'UNRESOLVED_REQUEST_NO_AUTOMATIC_RESEND'}
    # Read completed records only at drained boundaries, never partial worker writes.
    if not active:
        affected={j['business_id'] for j in inspect}
        for b in businesses:
            if b['business_id'] in affected and save_completed_business(b, by_business, imports): complete.add(b['business_id'])
    # CP scans are only called before submissions/after full drain. Active transition
    # updates use a saved snapshot and do not call this function while workers write.
    for j in inspect:
        directory=location(j,imports)
        p=directory/'raw_responses'/(j['job_id']+'.json')
        if p.exists() and str(p) not in STATE['seen_raw']:
            STATE['seen_raw'].add(str(p))
            raw = read(p); u = raw.get('usage') or {}
            if u:
                usage['responses_with_usage'] += 1
                usage['input_tokens'] += u.get('input_tokens', 0)
                usage['output_tokens'] += u.get('output_tokens', 0)
                usage['reasoning_tokens'] += u.get('output_tokens_details', {}).get('reasoning_tokens', 0)
    report = {'status': 'RUNNING' if not stop else 'STOPPED_REQUIRES_REVIEW', 'phase': phase,
              'updated_utc': now(), 'pid': os.getpid(), 'businesses_completed': len(complete),
              'businesses_planned': 2200, 'reused_businesses': 100, 'new_businesses_completed': len(complete)-100,
              'targets_completed': 3*len(complete), 'accepted_tasks': len(accepted), 'tasks_planned': 6600,
              'current_jobs': active, 'maximum_concurrency': 6, 'failure_policy': 'defer_individual_task_continue_others',
              'new_API_attempts': len(list((OUT / 'request_intents').glob('*.json'))),
              'imported_complete_API_responses': 292, 'historical_pilot_failed_attempts': 3,
              'failed_or_unresolved': list(failure_map.values()), 'stop_reason': stop, 'usage_including_imported': dict(usage),
              'automatic_retries': 0, 'target_content_origin': 'teacher_API_only_frozen_serialization',
              'held_out_teacher_targets': 0, 'decoder_limit_frozen': False, 'LED_training_started': False}
    atomic(OUT / 'deferred_tasks.json', {'updated_utc': now(), 'tasks': list(failure_map.values()), 'automatic_resends': 0})
    atomic(OUT / 'execution_progress.json', report)
    print(json.dumps(report), flush=True)
    return report, set(accepted), sorted(complete)


def update_docs(report):
    marker = '**D-042 live production status:**'
    note = (f"{marker} {report['businesses_completed']}/2200 complete businesses, "
            f"{report['targets_completed']} targets; imported100 unchanged; new API attempts={report['new_API_attempts']}; "
            f"status={report['status']}; phase={report['phase']}; stop reason={report.get('stop_reason')}. "
            "Authoritative resumable state: outputs/stage_g/teacher_production_d042_v1/execution_v1/execution_progress.json. "
            "TeacherAPI-only content, no held-out targets, no LED training; structural checks are not factual certification.\n")
    p = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
    text = p.read_text(encoding='utf-8'); lines = text.splitlines(keepends=True)
    lines = [l for l in lines if not l.startswith(marker)]
    lines.insert(1, '\n' + note + '\n')
    p.write_text(''.join(lines), encoding='utf-8')


def pending_jobs(jobs, accepted, imports, failures):
    deferred = {x['job_id'] for x in failures}
    return [j for j in jobs if j['job_id'] not in accepted and j['job_id'] not in imports and j['job_id'] not in deferred]


def main():
    lock = runner_lock()
    cfg, jobs, businesses, imports = check_inputs()
    auth = read(FOLDER / 'authorization.json')
    assert auth['approved'] and auth['paid_calls_authorized'] and auth['config_sha256'] == file_sha(CONFIG)
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    for j in jobs:
        directory = location(j, imports)
        cp = directory / 'checkpoints' / (j['job_id']+'.json')
        if cp.exists() and read(cp)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            # All imported CPs were independently checked during preflight. Re-check
            # newly generated CPs on every resume before accessing credentials.
            if j['job_id'] not in imports: verify_checkpoint(j, cfg, directory, CONFIG)
    report, accepted, complete = progress(jobs, businesses, imports, [], 'offline_resume_check')
    if len(complete) == 2200:
        subprocess.run([sys.executable, str(ROOT/'scripts/audit_stage_g_production_d042.py')], cwd=ROOT, check=True)
        return
    base, provider, key = credentials()  # Isolated teacher runtime; no Codex config/auth.
    assert (base,provider,cfg['requested_model']) == (BASE_URL,PROVIDER,MODEL)
    import run_stage_g_teacher_pilot_v4_1_recovery as legacy
    import stage_g_teacher_stream_diagnostics_v1 as diagnostic
    legacy.call = diagnostic.call
    smoke_ids = set(read(FOLDER / 'small_batch_manifest.json')['business_ids'])
    smoke_pass = FOLDER / 'small_batch_validation.json'
    phase = 'remaining_production' if smoke_pass.exists() else 'small_production_batch'
    update_docs(report)
    batches = OUT / 'batch_log.jsonl'

    def worker(j):
        if not j.get('skip_api_empty_captions'):
            diagnostic.set_context(OUT/'stream_diagnostics', j['job_id'])
        try:
            result = job_result(j,cfg,prompts[j['task']],schemas[j['task']],base,provider,key,OUT,CONFIG)
            return {'job_id': j['job_id'], 'passed': True, 'seconds': result.get('seconds'), 'response_id': result.get('response_id')}
        except Exception as e:
            # Exception contents can include provider headers; log classes only.
            attempted = (OUT/'request_intents'/(j['job_id']+'.json')).exists()
            recorded = (OUT/'checkpoints'/(j['job_id']+'.json')).exists()
            return {'job_id': j['job_id'], 'passed': False, 'error_type': type(e).__name__,
                    'local_integrity_failure': not attempted and not recorded}

    consecutive_failed_batches = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        while True:
            pending = pending_jobs(jobs, accepted, imports, report['failed_or_unresolved'])
            if not smoke_pass.exists(): pending = [j for j in pending if j['business_id'] in smoke_ids]
            if not pending:
                if not smoke_pass.exists():
                    for j in jobs:
                        if j['business_id'] in smoke_ids: verify_checkpoint(j,cfg,OUT,CONFIG)
                    assert smoke_ids <= set(complete)
                    subprocess.run([sys.executable, str(ROOT/'scripts/audit_stage_g_production_d042.py'), '--smoke'], cwd=ROOT, check=True)
                    assert smoke_pass.exists(); phase = 'remaining_production'
                    continue
                break
            if (FOLDER/'STOP_REQUESTED.json').exists():
                report,accepted,complete=progress(jobs,businesses,imports,[],phase,'User pause flag; checkpoints retained')
                update_docs(report); return
            batch = pending[:6]
            for j in batch:
                assert not (OUT/'checkpoints'/(j['job_id']+'.json')).exists()
                assert not (OUT/'request_intents'/(j['job_id']+'.json')).exists()
                if not j.get('skip_api_empty_captions'):
                    body = canonical(payload(j,cfg,prompts[j['task']],schemas[j['task']]))
                    path = OUT/'raw_requests'/(j['job_id']+'.json.gz'); path.parent.mkdir(parents=True,exist_ok=True)
                    compressed = gzip.compress(body,mtime=0)
                    if path.exists(): assert path.read_bytes() == compressed
                    else: path.write_bytes(compressed)
            running = dict(report, current_jobs=[j['job_id'] for j in batch], updated_utc=now(), phase=phase, status='RUNNING')
            atomic(OUT/'execution_progress.json',running)
            futures = {pool.submit(worker,j): j for j in batch}; results=[]
            for future in as_completed(futures):
                item=future.result();results.append(item)
                running['current_jobs'].remove(item['job_id']);running['updated_utc']=now()
                atomic(OUT/'execution_progress.json',running)
                print(json.dumps(item),flush=True)
            with batches.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'finished_utc':now(),'phase':phase,'results':results})+'\n')
            report,accepted,complete=progress(jobs,businesses,imports,[],phase,changed_jobs=batch)
            update_docs(report)
            # Failure is quarantined; no retry and no acceptance of invalid output.
            # Stop only for account-wide auth failures, backend alias mismatch,
            # or three entirely failed batches (avoid a provider-wide failure flood).
            terminal = False
            for item in results:
                if item['passed']: continue
                terminal |= item.get('local_integrity_failure', False)
                cp_path = OUT/'checkpoints'/(item['job_id']+'.json')
                cp = read(cp_path) if cp_path.exists() else {}
                terminal |= cp.get('http_status') in (401, 402, 403)
                terminal |= cp.get('local_validation_error') == 'Reported model differs from requested alias; review provider provenance'
            consecutive_failed_batches = consecutive_failed_batches + 1 if all(not r['passed'] for r in results) else 0
            if terminal or consecutive_failed_batches >= 3:
                report,accepted,complete=progress(jobs,businesses,imports,[],phase,'Local integrity or provider-wide safety stop; deferred requests preserved, no automatic resend')
                update_docs(report); return
    if report['failed_or_unresolved']:
        report.update(status='GENERATION_SWEEP_FINISHED_PENDING_RECOVERY', phase='deferred_recovery_required', current_jobs=[])
        atomic(OUT/'generation_sweep_report.json', report)
        update_docs(report)
        return
    assert len(complete)==2200
    atomic(OUT/'generation_completion.json',dict(report,status='GENERATION_COMPLETE_PENDING_FINAL_VALIDATION',phase='final_validation'))
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_stage_g_production_d042.py')],cwd=ROOT,check=True)
    # lock and isolated key remain process-local and are released at process exit.


if __name__ == '__main__':
    try: main()
    except Exception as e:
        atomic(OUT/'runner_failure.json',{'status':'STOPPED_REQUIRES_REVIEW','error_type':type(e).__name__,'updated_utc':now(),
              'outputs_not_rewritten':True,'automatic_retries':0})
        p=OUT/'execution_progress.json'
        if p.exists():
            state=read(p);state.update(status='STOPPED_REQUIRES_REVIEW',current_jobs=[],stop_reason='Runner/offline validation failure: '+type(e).__name__,updated_utc=now())
            atomic(p,state);update_docs(state)
        print(json.dumps({'status':'STOPPED_REQUIRES_REVIEW','error_type':type(e).__name__}),flush=True)
        raise SystemExit(1)
