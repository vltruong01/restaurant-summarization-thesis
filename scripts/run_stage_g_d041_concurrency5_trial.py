"""User-approved five-request scheduling trial, then resume the original serial pilot.

Exact frozen request bodies, existing cohort/checkpoints, zero automatic retry.
Only task scheduling differs; all target content comes from the teacher API.
"""
import gzip
import json
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from run_stage_g_extended_pilot_d041_dashboard import preflight
from run_stage_g_extended_pilot_d041 import (
    ROOT, FOLDER, CONFIG, read, file_sha, canonical, immutable, credentials,
    BASE_URL, PROVIDER, MODEL, job_result, payload, now, build_target, usage_so_far,
)

TRIAL = FOLDER / 'concurrency5_trial_v1'
OUT = FOLDER / 'execution_v1'


def rebuild_targets(jobs):
    content = {}; failures = []
    for job in jobs:
        p = OUT / 'checkpoints' / (job['job_id'] + '.json')
        if not p.exists(): continue
        cp = read(p)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            failures.append({'job_id': job['job_id'], 'business_id': job['business_id'], 'failure': cp})
            continue
        content.setdefault(job['business_id'], {})[job['condition']] = cp.get('resolved_output', cp['output'])
    complete = []
    for business in read(FOLDER / 'pilot_businesses.json'):
        record = build_target(business, content.get(business['business_id'], {}))
        if record is not None:
            immutable(OUT / 'provisional_targets' / (business['business_id'] + '.json'), record)
            complete.append(business)
    return complete, failures


def main():
    cfg, pre, jobs, auth = preflight()
    assert read(FOLDER / 'concurrency5_serial_handoff.json')['active_requests_at_stop'] == []
    assert not (OUT / 'completion.json').exists()
    assert not (TRIAL / 'manifest.json').exists(), 'Trial already started; no automatic rerun'
    unresolved = [p.stem for p in (OUT / 'request_intents').glob('*.json')
                  if not (OUT / 'checkpoints' / p.name).exists()]
    assert not unresolved, 'Unresolved paid request; operator review required'
    selected = [j for j in jobs if not j.get('skip_api_empty_captions')
                and not (OUT / 'checkpoints' / (j['job_id'] + '.json')).exists()
                and not (OUT / 'request_intents' / (j['job_id'] + '.json')).exists()][:5]
    assert len(selected) == 5
    assert len(list((OUT / 'request_intents').glob('*.json'))) + 5 <= pre['max_new_API_calls']
    immutable(TRIAL / 'authorization.json', {
        'approved_by': 'user', 'approved': True, 'approval_date': '2026-10-11',
        'authorization_basis': 'User asked to try five requests in parallel.',
        'maximum_new_calls': 5, 'maximum_in_flight': 5,
        'cohort_maximum_initial_calls': 292, 'automatic_retries': 0,
        'after_success': 'resume_previously_authorized_serial_100_business_pilot',
        'production_authorized': False,
    })
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    request_hashes = {}
    for job in selected:
        request = canonical(payload(job, cfg, prompts[job['task']], schemas[job['task']]))
        destination = OUT / 'raw_requests' / (job['job_id'] + '.json.gz')
        if destination.exists(): assert gzip.decompress(destination.read_bytes()) == request
        else: destination.write_bytes(gzip.compress(request, mtime=0))
        import hashlib
        request_hashes[job['job_id']] = hashlib.sha256(request).hexdigest()
    baseline = {}
    job_types = {j['job_id']: j['task'] + ':' + j['condition'] for j in jobs}
    for p in (OUT / 'checkpoints').glob('*.json'):
        cp = read(p)
        if cp.get('api_called') and cp.get('seconds') is not None:
            baseline.setdefault(job_types[p.stem], []).append(cp['seconds'])
    immutable(TRIAL / 'manifest.json', {
        'started_utc': now(), 'status': 'USER_APPROVED_RUNTIME_SCHEDULING_TRIAL_NOT_PROMPT_CHANGE',
        'runner_sha256': file_sha(ROOT / 'scripts/run_stage_g_d041_concurrency5_trial.py'),
        'config_sha256': file_sha(CONFIG), 'requested_model': MODEL,
        'original_serial_concurrency': 1, 'effective_trial_concurrency': 5,
        'selected_jobs': [j['job_id'] for j in selected],
        'selection_rule': 'first_five_unstarted_nonempty_jobs_in_unchanged_frozen_order',
        'request_sha256': request_hashes,
        'inference_payload_settings_changed': False, 'automatic_retries': 0,
        'baseline_seconds_by_task': baseline,
        'Codex_configuration_or_authentication_access': False,
    })
    base, provider, key = credentials()
    assert base == BASE_URL and provider == PROVIDER
    lock = threading.Lock(); active = 0; peak = 0; events = []
    def request_job(job):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
            events.append({'event': 'worker_started', 'job_id': job['job_id'], 'utc': now(), 'active_workers': active})
        try:
            cp = job_result(job, cfg, prompts[job['task']], schemas[job['task']], base, provider, key, OUT, CONFIG)
            return {'job_id': job['job_id'], 'status': cp['status'], 'seconds': cp.get('seconds'),
                    'reported_model': cp.get('reported_model'), 'usage': cp.get('usage'), 'fatal': False}
        except (RuntimeError, ValueError):
            path = OUT / 'checkpoints' / (job['job_id'] + '.json')
            failed = read(path) if path.exists() else {'status': 'UNRESOLVED_REQUEST_NO_RESEND'}
            raw_path = OUT / 'raw_responses' / path.name
            fatal = failed.get('http_status') in [401, 403, 429] or not raw_path.exists()
            if raw_path.exists() and read(raw_path).get('model') != MODEL: fatal = True
            return {'job_id': job['job_id'], 'status': failed['status'], 'failure': failed, 'fatal': fatal}
        finally:
            with lock:
                active -= 1
                events.append({'event': 'worker_finished', 'job_id': job['job_id'], 'utc': now(), 'active_workers': active})
    started = time.monotonic(); results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(request_job, job) for job in selected]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps({'completed_trial_requests': len(results), 'job': result['job_id'],
                              'status': result['status'], 'seconds': result.get('seconds') }), flush=True)
    wall = time.monotonic() - started
    complete, failures = rebuild_targets(jobs)
    check_inputs = preflight()  # Frozen sources, hashes and prep exports still match.
    equivalent_serial_estimate = sum(statistics.mean(baseline[job_types[j['job_id']]]) for j in selected)
    fatal = any(r['fatal'] for r in results)
    report = {
        'status': 'TRIAL_STOP_FOR_ACCOUNT_NETWORK_OR_PROVENANCE_REVIEW' if fatal else
                  ('TRIAL_COMPLETED_WITH_PRESERVED_VALIDATION_FAILURES' if failures else 'TRIAL_TECHNICAL_PASS_NOT_FACTUAL_CERTIFICATE'),
        'trial_initial_requests': 5, 'maximum_concurrent_workers': peak,
        'wall_seconds': round(wall, 3), 'results': results, 'worker_events': events,
        'previous_serial_task_mix_estimate_seconds': round(equivalent_serial_estimate, 3),
        'estimated_throughput_ratio_vs_previous_task_means': round(equivalent_serial_estimate / wall, 3),
        'comparison_warning': 'Different inputs; task-mean comparison is approximate, not a repeated-input controlled speed benchmark.',
        'automatic_retries': 0, 'businesses_complete_after_trial': len(complete),
        'cohort_request_intents': len(list((OUT / 'request_intents').glob('*.json'))),
        'target_content_origin': 'teacher_API_only_frozen_serialization',
        'after_trial': 'stop_for_review' if fatal else 'resume_original_serial_worker',
        'methodological_inputs_and_inference_payload_unchanged': True,
    }
    immutable(TRIAL / 'report.json', report)
    print(json.dumps(report), flush=True)
    if fatal:
        raw = [read(p) for p in (OUT / 'raw_responses').glob('*.json')]
        immutable(OUT / 'completion.json', {
            'status': 'D041_STOPPED_FOR_REVIEW', 'businesses_planned': 100,
            'businesses_completed': len(complete), 'target_texts': 3 * len(complete),
            'businesses_by_split': dict(Counter(b['split'] for b in complete)),
            'new_API_request_intents': report['cohort_request_intents'], 'empty_tasks_planned': pre['empty_tasks'],
            'usage': usage_so_far(OUT), 'responses_without_usage': sum(not bool(r.get('usage')) for r in raw),
            'reported_model_counts': dict(Counter(r.get('model') for r in raw)),
            'cost_monitoring': auth['cost_monitoring'], 'provider_invoice_cost': None,
            'runtime_seconds_this_execution': round(wall, 3), 'failed_tasks': failures,
            'stop_reason': 'Parallel trial account/network/provenance failure; all already started requests drained and preserved; no retries',
            'automatic_retries': 0, 'human_verified_businesses': 0, 'production_protocol_frozen': False,
            'held_out_teacher_targets': 0, 'remaining_2100_generation_started': False, 'LED_training_started': False,
        })
        return
    print('Trial finished; resuming the previously approved serial D041 pilot.', flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/run_stage_g_extended_pilot_d041_dashboard.py'), '--execute-pilot'],
                   cwd=ROOT, check=True)


if __name__ == '__main__': main()
