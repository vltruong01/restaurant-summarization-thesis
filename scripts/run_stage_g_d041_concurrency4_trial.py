"""Bounded user-authorized four-request trial; then resume serial D041 work.

No frozen inference inputs are changed. Failed or started tasks are never resent.
"""
import gzip
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import run_stage_g_d041_continuation_v2 as continuation
from run_stage_g_d041_continuation_v2 import (
    ROOT, FOLDER, OUT, CONFIG, read, file_sha, canonical, digest, immutable,
    credentials, BASE_URL, PROVIDER, MODEL, payload, job_result, now,
    check_inputs, saved_state,
)

TRIAL = FOLDER / 'concurrency4_trial_v1'
PAUSE = FOLDER / 'STOP_REQUESTED.json'


def final_report(completion):
    """Use the frozen audit; add scheduling provenance to the final package."""
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_stage_g_d041_continuation_v2.py'), '--audit-results'], cwd=ROOT, check=True)
    validation = read(FOLDER / 'validation_report.json')
    immutable(FOLDER / 'execution_report.json', {
        'execution': completion, 'validation_status': validation['status'],
        'source_namespace_preserved': True, '100_and_manual25_manifests_unchanged': True,
        'previous_failure_preserved_not_retried': True,
        'effective_continuation_concurrency': 'serial1_with_one_bounded_parallel4_batch',
        'scheduling_history': read(FOLDER / 'runtime_scheduling_history.json'),
        'all_target_content_from_teacher_API': True, 'independent_factual_accuracy': None,
        'researcher_review_pending': True, 'production_or_training_started': False,
    })
    marker = '**D041 continuation_v2 result including bounded concurrency4 trial (2026-10-11):**'
    note = (f"\n{marker} {completion['businesses_completed']}/100 complete businesses; "
            f"{completion['target_texts']} provisional target texts; "
            f"{completion['cumulative_cohort_API_request_intents']} cumulative cohort attempts. "
            f"Failures={len(completion['failed_tasks'])}; automatic retries=0. Scheduling: serial1, "
            f"one user-authorized four-request batch, then serial1. Validation={validation['status']}; "
            "automated validation does not establish independent factual accuracy. Frozen inputs, "
            "previous pilot reports and fixed25 review sample unchanged. No production/test targets/training.\n")
    for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md', 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
        path = ROOT / rel
        text = path.read_text(encoding='utf-8')
        if marker not in text:
            first, rest = text.split('\n', 1)
            path.write_text(first + '\n' + note + '\n' + rest, encoding='utf-8')
    exporter = ROOT / 'scripts/export_stage_g_d041_continuation_v2_review.py'
    source = exporter.read_text(encoding='utf-8')
    anchor = "    files['additional_flagged_cases.json']=original(FOLDER/'technical_flags.json')"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + "\n    for relative in ['runtime_scheduling_history.json','concurrency4_trial_v1/authorization.json','concurrency4_trial_v1/manifest.json','concurrency4_trial_v1/report.json']:\n        files['reports/'+relative]=original(FOLDER/relative)\n    files['protocol/runtime/run_stage_g_d041_concurrency4_trial.py']=original(ROOT/'scripts/run_stage_g_d041_concurrency4_trial.py')")
    versioned_exporter = ROOT / 'scripts/export_stage_g_d041_continuation_v2_concurrency4_review.py'
    content = '# Frozen base exporter SHA256: ' + file_sha(exporter) + '\n' + source
    if versioned_exporter.exists():
        assert versioned_exporter.read_text(encoding='utf-8') == content
    else:
        versioned_exporter.write_text(content, encoding='utf-8')
    subprocess.run([sys.executable, str(versioned_exporter)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/execute_stage_g_d041_status_cell.py')], cwd=ROOT, check=True)
    immutable(FOLDER / 'postprocessing_status.json', {'status': 'COMPLETED', 'API_calls': 0,
              'businesses_complete': completion['businesses_completed'], 'factual_review_pending': True})


def main():
    cfg, pre, jobs = check_inputs(verify_checkpoints=True)
    assert PAUSE.exists(), 'Serial worker must be gracefully paused first'
    assert not (OUT / 'completion.json').exists()
    assert not (TRIAL / 'manifest.json').exists(), 'Trial already started; no automatic rerun'
    assert not [p for p in (OUT / 'request_intents').glob('*.json')
                if not (OUT / 'checkpoints' / p.name).exists()], 'Unresolved paid request'
    selected = [j for j in jobs if not j.get('skip_api_empty_captions')
                and not (OUT / 'checkpoints' / (j['job_id'] + '.json')).exists()
                and not (OUT / 'request_intents' / (j['job_id'] + '.json')).exists()][:4]
    assert len(selected) == 4
    assert len(list((OUT / 'request_intents').glob('*.json'))) + 4 <= 292
    immutable(TRIAL / 'authorization.json', {
        'approved_by': 'user', 'approved': True, 'approved_utc': now(),
        'authorization_basis': 'User requested trying four parallel requests instead of five.',
        'maximum_new_calls': 4, 'maximum_in_flight': 4,
        'cohort_maximum_initial_calls': 292, 'automatic_retries': 0,
        'after_success': 'resume_previously_authorized_serial100_pilot',
        'production_authorized': False,
    })
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    request_hashes = {}
    for job in selected:
        request = canonical(payload(job, cfg, prompts[job['task']], schemas[job['task']]))
        path = OUT / 'raw_requests' / (job['job_id'] + '.json.gz')
        if path.exists():
            assert gzip.decompress(path.read_bytes()) == request
        else:
            path.write_bytes(gzip.compress(request, mtime=0))
        request_hashes[job['job_id']] = digest(request)
    job_types = {j['job_id']: j['task'] + ':' + j['condition'] for j in jobs}
    baseline = {}
    imported = read(FOLDER / 'continuation_manifest.json')['copied_files_sha256']
    for p in (OUT / 'checkpoints').glob('*.json'):
        cp = read(p)
        # Only this continuation's genuinely serial responses form the baseline.
        if 'execution_v1/checkpoints/' + p.name not in imported and cp.get('api_called') and cp.get('seconds') is not None:
            baseline.setdefault(job_types[p.stem], []).append(cp['seconds'])
    immutable(TRIAL / 'manifest.json', {
        'started_utc': now(), 'runner_sha256': file_sha(ROOT / 'scripts/run_stage_g_d041_concurrency4_trial.py'),
        'config_sha256': file_sha(CONFIG), 'requested_model': MODEL,
        'original_serial_concurrency': 1, 'effective_trial_concurrency': 4,
        'selected_jobs': [j['job_id'] for j in selected],
        'selection_rule': 'first_four_unstarted_nonempty_jobs_in_unchanged_frozen_order',
        'request_sha256': request_hashes, 'inference_payload_settings_changed': False,
        'automatic_retries': 0, 'baseline_seconds_by_task': baseline,
        'Codex_configuration_or_authentication_access': False,
    })
    base, provider, key = credentials()
    assert base == BASE_URL and provider == PROVIDER
    lock = threading.Lock()
    active = 0
    peak = 0
    events = []

    def request_job(job):
        nonlocal active, peak
        started = time.monotonic()
        with lock:
            active += 1
            peak = max(peak, active)
            events.append({'event': 'worker_started', 'job_id': job['job_id'], 'utc': now(), 'active_workers': active})
        try:
            cp = job_result(job, cfg, prompts[job['task']], schemas[job['task']], base, provider, key, OUT, CONFIG)
            return {'job_id': job['job_id'], 'status': cp['status'], 'seconds': cp.get('seconds'),
                    'reported_model': cp.get('reported_model'), 'usage': cp.get('usage'), 'failed': False}
        except Exception as error:
            path = OUT / 'checkpoints' / (job['job_id'] + '.json')
            failure = read(path) if path.exists() else {'status': 'UNRESOLVED_REQUEST_NO_RESEND', 'error_type': type(error).__name__}
            return {'job_id': job['job_id'], 'status': failure['status'], 'failure': failure,
                    'seconds': round(time.monotonic() - started, 3), 'failed': True}
        finally:
            with lock:
                active -= 1
                events.append({'event': 'worker_finished', 'job_id': job['job_id'], 'utc': now(), 'active_workers': active})

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(request_job, j) for j in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({'trial_completed': len(results), 'job_id': result['job_id'],
                              'status': result['status'], 'seconds': result.get('seconds')}), flush=True)
    wall = time.monotonic() - started
    check_inputs(verify_checkpoints=True)
    _, failures, complete = saved_state(jobs)
    estimate = sum(statistics.mean(baseline[job_types[j['job_id']]]) for j in selected) if all(job_types[j['job_id']] in baseline for j in selected) else None
    failed = any(r['failed'] for r in results)
    report = {'status': 'TRIAL_STOP_FOR_REVIEW' if failed else 'TRIAL_TECHNICAL_PASS_NOT_FACTUAL_CERTIFICATE',
              'trial_initial_requests': 4, 'maximum_concurrent_workers': peak,
              'wall_seconds': round(wall, 3), 'results': results, 'worker_events': events,
              'previous_serial_task_mix_estimate_seconds': round(estimate, 3) if estimate else None,
              'comparison_warning': 'Different inputs, approximate task means; a single batch does not establish reliability or causal speedup.',
              'automatic_retries': 0, 'businesses_complete_after_trial': len(complete),
              'cohort_request_intents': len(list((OUT / 'request_intents').glob('*.json'))),
              'existing_and_new_failed_tasks': len(failures),
              'methodological_inputs_and_inference_payload_unchanged': True,
              'after_trial': 'stop_for_review' if failed else 'resume_serial1'}
    immutable(TRIAL / 'report.json', report)
    immutable(FOLDER / 'runtime_scheduling_history.json', {
        'baseline': 'serial1', 'bounded_trial_concurrency': 4, 'trial_request_count': 4,
        'trial_manifest_sha256': file_sha(TRIAL / 'manifest.json'),
        'trial_report_sha256': file_sha(TRIAL / 'report.json'),
        'subsequent_schedule': 'stopped' if failed else 'serial1',
        'frozen_config_concurrency1_is_baseline_not_complete_runtime_history': True,
    })
    print(json.dumps(report), flush=True)
    if failed:
        print('All four requests drained; stopped for review without retry or further calls.', flush=True)
        return
    PAUSE.unlink()
    print('Four-request trial passed technical checks; resuming serial pilot.', flush=True)
    completion = continuation.execute()
    if completion is not None:
        final_report(completion)


if __name__ == '__main__':
    main()
