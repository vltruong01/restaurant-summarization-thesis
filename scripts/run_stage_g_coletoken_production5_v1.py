"""Resumable frozen Stage G production: two business slots, five requests, stop on new failure."""
import argparse
import gzip
import json
import os
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import stage_g_coletoken_production_v1 as ledger
from stage_g_coletoken_production_v1 import (
    ROOT, RUN, EXEC, OVERLAY, CONFIG, POINTER, read, canonical, file_sha,
    immutable, atomic, now, checked_inputs, reuse_index, location,
    verify_checkpoint, save_completed_business, update_docs,
)
from run_stage_g_production_d042 import runner_lock
from run_stage_g_teacher_pilot_v4_1_recovery import payload, job_result
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL, KEY_VARIABLE
import stage_g_teacher_sse_transport_v1 as transport


def main():
    lock = runner_lock()
    cfg, jobs, businesses, imports = checked_inputs()
    manifest = ledger.verify_manifest()
    auth = read(ledger.ORIGINAL / 'authorization.json')
    assert auth['approved'] and auth['paid_calls_authorized'] and auth['config_sha256'] == file_sha(CONFIG)
    assert read(ledger.ORIGINAL / 'small_batch_validation.json')['status'] == 'D042_SMALL_BATCH_AUTOMATED_PASS'
    grouped = defaultdict(list)
    for j in jobs: grouped[j['business_id']].append(j)
    accepted = set(reuse_index()); completed = set(); failures = []; usage = Counter(); seen_usage = set()
    by_id = {j['job_id']: j for j in jobs}
    for j in jobs:
        directory = location(j, imports); cp_path = directory / 'checkpoints' / (j['job_id'] + '.json')
        if j['job_id'] not in accepted:
            if cp_path.exists():
                cp = read(cp_path)
                if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                    failures.append({'job_id': j['job_id'], 'status': cp['status']}); continue
                verify_checkpoint(j, cfg, directory, CONFIG); accepted.add(j['job_id'])
            elif (directory / 'request_intents' / cp_path.name).exists():
                failures.append({'job_id': j['job_id'], 'status': 'UNRESOLVED_INTENT_NO_AUTOMATIC_RESEND'})
    for b in businesses:
        if all(j['job_id'] in accepted for j in grouped[b['business_id']]):
            assert save_completed_business(b, grouped, imports); completed.add(b['business_id'])

    def count_usage(j):
        directory = location(j, imports)
        p = directory / 'raw_responses' / (j['job_id'] + '.json')
        if not p.exists() or str(p) in seen_usage: return
        seen_usage.add(str(p)); u = read(p).get('usage') or {}
        if u:
            usage['responses_with_usage'] += 1
            usage['input_tokens'] += u.get('input_tokens', 0); usage['output_tokens'] += u.get('output_tokens', 0)
            usage['reasoning_tokens'] += (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
    for jid in accepted: count_usage(by_id[jid])
    mutex = threading.RLock(); stop = threading.Event(); active = set(); active_businesses = set()
    peak = {'requests': 0, 'businesses': 0}; phase = ['remaining_production']; status = ['RUNNING']; reason = [None]

    def publish():
        with mutex:
            state = {'status': status[0], 'phase': phase[0], 'updated_utc': now(), 'pid': os.getpid(),
                'provider': PROVIDER, 'base_url': BASE_URL, 'requested_model': MODEL,
                'businesses_completed': len(completed), 'businesses_planned': 2200,
                'targets_completed': 3*len(completed), 'accepted_tasks': len(accepted), 'tasks_planned': 6600,
                'reused_D041_businesses': 100, 'initial_reused_complete_businesses': 134,
                'current_jobs': sorted(active), 'active_businesses': sorted(active_businesses),
                'maximum_concurrency': 5, 'maximum_active_businesses': 2,
                'observed_peak_requests': peak['requests'], 'observed_peak_businesses': peak['businesses'],
                'new_API_attempts': len(list((EXEC / 'request_intents').glob('*.json'))),
                'failed_or_unresolved': failures[:], 'stop_reason': reason[0],
                'usage_for_accepted_responses_including_reuse': dict(usage),
                'usage_note': 'Historical/failed attempts remain in preserved namespaces; provider dashboard is monetary authority',
                'automatic_retries': 0, 'held_out_teacher_targets': 0, 'LED_training_started': False,
                'decoder_limit_frozen': False, 'target_content_origin': 'teacher_API_only_frozen_serialization'}
            atomic(EXEC / 'execution_progress.json', state); update_docs(state)
            return state

    pointer = {'status': 'RUNNING_COLETOKEN_PRODUCTION', 'provider': PROVIDER, 'requested_model': MODEL,
        'production_directory': RUN.relative_to(ROOT).as_posix(),
        'progress_path': (EXEC / 'execution_progress.json').relative_to(ROOT).as_posix(),
        'overlay_config': OVERLAY.relative_to(ROOT).as_posix(),
        'maximum_concurrency': 5, 'maximum_active_businesses': 2,
        'pause_file': (RUN / 'STOP_REQUESTED.json').relative_to(ROOT).as_posix(),
        'manifest_sha256': file_sha(RUN / 'manifest.json'), 'updated_utc': now()}
    atomic(POINTER, pointer)
    if failures:
        status[0] = 'STOPPED_REQUIRES_REVIEW'; reason[0] = 'Prior failed/unresolved continuation attempt preserved'
        publish(); return
    if (RUN / 'STOP_REQUESTED.json').exists():
        status[0] = 'STOPPED_BY_USER'; reason[0] = 'Continuation pause flag'; publish(); return
    assert not os.environ.get(KEY_VARIABLE)
    base, provider, key = credentials()
    assert (base, provider, cfg['requested_model']) == (BASE_URL, PROVIDER, MODEL)
    legacy.call = transport.call
    schemas = read(ROOT / cfg['schema']); prompts = {k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    if not (RUN / 'execution_started.json').exists():
        immutable(RUN / 'execution_started.json', {'started_utc': now(), 'manifest_sha256': file_sha(RUN / 'manifest.json')})
    publish()

    def component(j):
        jid = j['job_id']
        with mutex:
            if jid in accepted: return True
            if stop.is_set() or (RUN / 'STOP_REQUESTED.json').exists():
                stop.set(); return False
            if not j.get('skip_api_empty_captions'):
                assert len(list((EXEC / 'request_intents').glob('*.json'))) < manifest['maximum_new_API_calls']
                encoded = canonical(payload(j, cfg, prompts[j['task']], schemas[j['task']]))
                p = EXEC / 'raw_requests' / (jid + '.json.gz'); p.parent.mkdir(parents=True, exist_ok=True)
                if p.exists(): assert gzip.decompress(p.read_bytes()) == encoded
                else:
                    with p.open('xb') as f: f.write(gzip.compress(encoded, mtime=0))
                active.add(jid); peak['requests'] = max(peak['requests'], len(active))
                assert len(active) <= 5
                transport.set_context(EXEC / 'stream_diagnostics', jid)
            publish()
        try:
            cp = job_result(j, cfg, prompts[j['task']], schemas[j['task']], base, provider, key, EXEC, CONFIG)
            with mutex:
                accepted.add(jid); count_usage(j)
                if all(x['job_id'] in accepted for x in grouped[j['business_id']]):
                    b = next(b for b in businesses if b['business_id'] == j['business_id'])
                    assert save_completed_business(b, grouped, imports); completed.add(b['business_id'])
                print(json.dumps({'job_id':jid, 'passed':True, 'seconds':cp.get('seconds')}), flush=True)
            return True
        except Exception as error:
            with mutex:
                stop.set(); failures.append({'job_id':jid, 'error_type':type(error).__name__})
                reason[0] = 'New request failure; drain submitted requests without automatic retry'
                print(json.dumps({'job_id':jid, 'passed':False, 'error_type':type(error).__name__}), flush=True)
            return False
        finally:
            with mutex: active.discard(jid); publish()

    pending_businesses = iter(b for b in businesses if b['business_id'] not in completed)
    with ThreadPoolExecutor(max_workers=5) as requests:
        def business(b):
            bid = b['business_id']
            with mutex:
                active_businesses.add(bid); peak['businesses'] = max(peak['businesses'], len(active_businesses))
                assert len(active_businesses) <= 2; publish()
            try:
                futures = [requests.submit(component, j) for j in grouped[bid] if j['job_id'] not in accepted]
                return all([f.result() for f in futures])
            finally:
                with mutex: active_businesses.remove(bid); publish()

        with ThreadPoolExecutor(max_workers=2) as slots:
            running = {}
            def fill():
                while len(running) < 2 and not stop.is_set() and not (RUN / 'STOP_REQUESTED.json').exists():
                    b = next(pending_businesses, None)
                    if b is None: break
                    running[slots.submit(business, b)] = b['business_id']
            fill()
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    if not future.result(): stop.set()
                    del running[future]
                fill()
    if stop.is_set() or (RUN / 'STOP_REQUESTED.json').exists():
        status[0] = 'STOPPED_REQUIRES_REVIEW' if failures else 'STOPPED_BY_USER'
        reason[0] = reason[0] or 'Continuation pause flag; completed outputs retained'
        publish(); return
    assert len(completed) == 2200 and len(accepted) == 6600
    phase[0] = 'final_validation'; status[0] = 'GENERATION_COMPLETE_PENDING_FINAL_VALIDATION'
    state = publish(); immutable(EXEC / 'generation_completion.json', state)
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_stage_g_coletoken_production5_v1.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true')
    args = p.parse_args(); assert args.prepare != args.run
    try:
        if args.prepare:
            lock = runner_lock(); ledger.prepare()
        else: main()
    except Exception as error:
        atomic(RUN / 'runner_failure.json', {'status':'STOPPED_REQUIRES_REVIEW','error_type':type(error).__name__,'utc':now(),'automatic_retries':0})
        path = EXEC / 'execution_progress.json'
        if path.exists():
            state = read(path); state.update(status='STOPPED_REQUIRES_REVIEW', current_jobs=[], active_businesses=[],
                stop_reason='Runner/offline validation failure: '+type(error).__name__,updated_utc=now())
            atomic(path,state); update_docs(state)
        print(json.dumps({'runner_error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
