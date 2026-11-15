"""User-approved one serial HTTP520 recovery then unstarted D041 tasks; instrumented transport and zero automatic retries."""
import gzip
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from stage_g_d041_recovery_v4 import (
    ROOT, FOLDER, OUT, CONFIG, read, file_sha, canonical, digest, immutable,
    credentials, BASE_URL, PROVIDER, MODEL, payload, job_result, now,
    check_inputs, saved_state, usage_so_far,
)

PHASE = FOLDER / 'execution_control_v1'
PAUSE = FOLDER / 'STOP_REQUESTED.json'


def status(jobs, complete, failures, current, accepted, usage):
    recovering = not (FOLDER / 'recovery_attempts.json').exists()
    report = {'updated_utc': now(), 'current_job': current[0] if current else None,
              'current_jobs': current, 'businesses_completed': len(complete),
              'target_texts': 3 * len(complete),
              'API_request_intents': len(list((OUT / 'request_intents').glob('*.json'))) + 3,
              'accepted_tasks': accepted, 'tasks_planned': len(jobs), 'failed_tasks': len(failures),
              'usage': usage, 'effective_concurrency': 1 if recovering else 4, 'active_task_workers': len(current),
              'scheduler': 'serial_operator_recovery' if recovering else 'four_request_batches',
              'historical_failed_attempts_preserved': 3, 'imported_API_request_intents': 237,
              'automatic_retries': 0, 'human_verified': False,
              'target_content_origin': 'teacher_API_only_frozen_serialization'}
    path = OUT / 'execution_progress.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
    print(json.dumps(report), flush=True)


def finalize(completion):
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_stage_g_d041_recovery_v4.py'), '--audit-results'], cwd=ROOT, check=True)
    validation = read(FOLDER / 'validation_report.json')
    immutable(FOLDER / 'execution_report.json', {
        'execution': completion, 'validation_status': validation['status'],
        'source_namespace_preserved': True, '100_and_manual25_manifests_unchanged': True,
        'previous_failed_attempts_preserved': True,
        'explicit_operator_recovery': read(FOLDER / 'recovery_attempts.json'),
        'subsequent_maximum_concurrency': 4, 'all_target_content_from_teacher_API': True,
        'independent_factual_accuracy': None, 'researcher_review_pending': True,
        'production_or_training_started': False,
    })
    marker = '**D041 recovery_v4 execution result (2026-10-11):**'
    note = (f"{marker} {completion['businesses_completed']}/100 complete businesses; "
            f"{completion['target_texts']} provisional target texts; "
            f"{completion['cumulative_cohort_API_request_intents']} total cohort attempts including three preserved previous failed attempts. "
            f"Current failures={len(completion['failed_tasks'])}; stop reason={completion['stop_reason']}. "
            "One operator-approved serial HTTP520 recovery attempt then four-request batches; zero automatic retries. "
            f"Validation={validation['status']}; automated validation does not establish factual accuracy. "
            "Previous79 complete business targets and old raw outputs remain unchanged. Fixed25 ZIP/page are under recovery_v4. "
            "No test targets/production teacher freeze/full2200 generation/LED training.\n")
    for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md', 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
        p = ROOT / rel
        text = p.read_text(encoding='utf-8')
        if marker not in text:
            first, rest = text.split('\n', 1)
            p.write_text(first + '\n\n' + note + '\n' + rest, encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_stage_g_d041_recovery_v4_review.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/execute_stage_g_d041_status_cell.py')], cwd=ROOT, check=True)
    immutable(FOLDER / 'postprocessing_status.json', {'status': 'COMPLETED', 'API_calls': 0,
              'businesses_complete': completion['businesses_completed'], 'factual_review_pending': True})


def main():
    cfg, pre, jobs = check_inputs(verify_checkpoints=True)
    assert PAUSE.exists(), 'Previous serial worker must have paused at a saved-response boundary'
    assert not (OUT / 'completion.json').exists()
    assert not (PHASE / 'manifest.json').exists(), 'Phase already started; no automatic restart'
    recovery_manifest = read(FOLDER / 'continuation_manifest.json')
    retry_ids = recovery_manifest['operator_retry_job_ids']
    assert len(retry_ids) == 1
    import run_stage_g_teacher_pilot_v4_1_recovery as legacy
    import stage_g_teacher_stream_diagnostics_v1 as diagnostic
    legacy.call = diagnostic.call  # Runtime transport instrumentation; frozen source file unchanged.
    pending = [j for j in jobs if not (OUT / 'checkpoints' / (j['job_id'] + '.json')).exists()]
    previous_calls = len(list((OUT / 'request_intents').glob('*.json')))
    remaining_calls = sum(not j.get('skip_api_empty_captions') for j in pending)
    assert previous_calls + remaining_calls <= 292
    immutable(PHASE / 'manifest.json', {
        'approved': True, 'approved_by': 'user', 'approved_utc': now(),
        'authorization_basis': 'User approved one serial HTTP520 operator retry then unstarted pilot work with four concurrent requests.',
        'runner_sha256': file_sha(ROOT / 'scripts/run_stage_g_d041_recovery_v4.py'),
        'config_sha256': file_sha(CONFIG), 'sampling_manifest_sha256': file_sha(FOLDER / 'sampling_manifest.json'),
        'manual25_manifest_sha256': file_sha(FOLDER / 'manual_review_25_manifest.json'),
        'requested_model': MODEL, 'baseline_config_concurrency': 1, 'effective_concurrency': 4,
        'scheduler': 'four_request_batches_in_frozen_job_order',
        'initial_cohort_attempts': previous_calls + 3, 'maximum_remaining_initial_calls': remaining_calls,
        'maximum_total_cohort_attempts_including_operator_retries': 295, 'namespace_unique_job_call_ceiling': 292, 'pending_job_ids': [j['job_id'] for j in pending],
        'inference_payload_settings_changed': False, 'automatic_retries': 0,
        'failure_policy': 'stop_new_batches_on_any_new_failure_and_drain_started_requests',
        'target_content_origin': 'teacher_API_only_frozen_serialization',
        'Codex_configuration_or_authentication_access': False, 'production_authorized': False,
    })
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    base, provider, key = credentials()
    assert base == BASE_URL and provider == PROVIDER and cfg['requested_model'] == MODEL
    PAUSE.unlink()
    started = time.monotonic()
    stopped = None
    batches = []
    _, failures, complete = saved_state(jobs)
    accepted = sum(read(p)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for p in (OUT / 'checkpoints').glob('*.json'))
    usage = usage_so_far(OUT)

    def call(job):
        diagnostic.set_context(OUT / 'stream_diagnostics', job['job_id'])
        begin = time.monotonic()
        try:
            cp = job_result(job, cfg, prompts[job['task']], schemas[job['task']], base, provider, key, OUT, CONFIG)
            return {'job_id': job['job_id'], 'status': cp['status'], 'seconds': cp.get('seconds'),
                    'reported_model': cp.get('reported_model'), 'api_called': cp.get('api_called'), 'failed': False}
        except Exception as error:
            path = OUT / 'checkpoints' / (job['job_id'] + '.json')
            failure = read(path) if path.exists() else {'status': 'UNRESOLVED_REQUEST_NO_RESEND', 'error_type': type(error).__name__}
            return {'job_id': job['job_id'], 'status': failure['status'], 'failure': failure,
                    'seconds': round(time.monotonic() - begin, 3), 'failed': True}

    recovery_results = []
    by_id = {j['job_id']: j for j in pending}
    for retry_id in retry_ids:
        job = by_id[retry_id]
        body = canonical(payload(job, cfg, prompts[job['task']], schemas[job['task']]))
        path = OUT / 'raw_requests' / (job['job_id'] + '.json.gz')
        assert path.exists() and gzip.decompress(path.read_bytes()) == body
        assert not (OUT / 'request_intents' / (retry_id + '.json')).exists()
        status(jobs, complete, failures, [retry_id], accepted, usage)
        result = call(job)
        recovery_results.append(result)
        _, failures, complete = saved_state(jobs)
        accepted = sum(read(p)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for p in (OUT / 'checkpoints').glob('*.json'))
        usage = usage_so_far(OUT)
        status(jobs, complete, failures, [], accepted, usage)
        print(json.dumps({'operator_recovery_result': result}), flush=True)
        if result['failed']:
            stopped = 'Explicit operator recovery failed; no further retries or unstarted calls'
            break
    immutable(FOLDER / 'recovery_attempts.json', {
        'approved_operator_retry_job_ids': retry_ids, 'attempt_results': recovery_results,
        'automatic_retries': 0, 'serial_recovery_concurrency': 1,
        'all_recoveries_passed': len(recovery_results) == 1 and not any(r['failed'] for r in recovery_results),
        'previous_failed_attempts_preserved': True,
    })
    (PHASE / 'batch_results.jsonl').touch(exist_ok=False)
    pending = [j for j in pending if j['job_id'] not in retry_ids]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index in range(0, len(pending), 4):
            if stopped:
                break
            if PAUSE.exists():
                status(jobs, complete, failures, [], accepted, usage)
                print('Operator pause after draining current batch; no request interrupted.', flush=True)
                return
            batch = pending[index:index + 4]
            for job in batch:
                assert not (OUT / 'checkpoints' / (job['job_id'] + '.json')).exists()
                assert not (OUT / 'request_intents' / (job['job_id'] + '.json')).exists()
                if not job.get('skip_api_empty_captions'):
                    body = canonical(payload(job, cfg, prompts[job['task']], schemas[job['task']]))
                    path = OUT / 'raw_requests' / (job['job_id'] + '.json.gz')
                    if path.exists():
                        assert gzip.decompress(path.read_bytes()) == body
                    else:
                        path.write_bytes(gzip.compress(body, mtime=0))
            assert len(list((OUT / 'request_intents').glob('*.json'))) + sum(not j.get('skip_api_empty_captions') for j in batch) <= 292
            batch_start = time.monotonic()
            futures = {pool.submit(call, j): j['job_id'] for j in batch}
            active = list(futures.values())
            status(jobs, complete, failures, active, accepted, usage)
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                active.remove(result['job_id'])
                if not result['failed']:
                    accepted += 1
                print(json.dumps({'completed_batch_job': result}), flush=True)
                status(jobs, complete, failures, active, accepted, usage)
                # Disk-wide counters are refreshed only once all writers drain.
            _, failures, complete = saved_state(jobs)
            accepted = sum(read(p)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for p in (OUT / 'checkpoints').glob('*.json'))
            usage = usage_so_far(OUT)
            record = {'batch': len(batches) + 1, 'finished_utc': now(),
                      'wall_seconds': round(time.monotonic() - batch_start, 3),
                      'job_ids': [j['job_id'] for j in batch], 'results': results}
            batches.append(record)
            with (PHASE / 'batch_results.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(record) + '\n')
            status(jobs, complete, failures, [], accepted, usage)
            if any(r['failed'] for r in results):
                stopped = 'New request failure; all already submitted calls drained; stopped without retry'
                break
    check_inputs(verify_checkpoints=True)
    raw = [read(p) for p in (OUT / 'raw_responses').glob('*.json')]
    total = len(list((OUT / 'request_intents').glob('*.json')))
    completion = {
        'status': 'D041_STOPPED_FOR_REVIEW' if stopped else ('D041_ATTEMPTED_WITH_FAILURES' if failures else 'D041_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE'),
        'businesses_planned': 100, 'businesses_completed': len(complete), 'target_texts': 3 * len(complete),
        'businesses_by_split': dict(Counter(b['split'] for b in complete.values())),
        'new_API_request_intents': total + 3, 'legacy_call_count_field_is_cumulative': True, 'namespace_unique_job_request_intents': total,
        'cumulative_cohort_API_request_intents': total + 3, 'imported_API_request_intents': 237,
        'new_initial_API_calls_this_continuation': total - previous_calls, 'explicit_operator_recovery_calls': len(recovery_results), 'historical_failed_attempts_preserved': 3,
        'new_initial_API_calls_this_concurrency4_phase': total - previous_calls,
        'empty_tasks_planned': pre['empty_tasks'], 'usage': usage,
        'reported_usage_includes_imported_responses': True,
        'responses_without_usage': sum(not bool(r.get('usage')) for r in raw),
        'request_intents_without_complete_raw_response': [p.stem for p in (OUT / 'request_intents').glob('*.json') if not (OUT / 'raw_responses' / p.name).exists()],
        'reported_model_counts': dict(Counter(r.get('model') for r in raw)),
        'failed_tasks': failures, 'stop_reason': stopped, 'automatic_retries': 0,
        'effective_concurrency_this_continuation': 4,
        'scheduling_history_note': 'Previous namespaces and failures preserved; one user-approved serial HTTP520 recovery, then unstarted tasks in four-request batches.',
        'runtime_seconds_this_concurrency4_phase': round(time.monotonic() - started, 3),
        'cost_monitoring': 'researcher_Coletoken_dashboard', 'provider_invoice_cost': None,
        'human_verified_businesses': 0, 'production_protocol_frozen': False,
        'held_out_teacher_targets': 0, 'remaining_2100_generation_started': False, 'LED_training_started': False,
    }
    immutable(PHASE / 'report.json', {'completion': completion, 'batches_completed': len(batches),
                                   'manifest_sha256': file_sha(PHASE / 'manifest.json'),
                                   'batch_log_sha256': file_sha(PHASE / 'batch_results.jsonl')})
    immutable(OUT / 'completion.json', completion)
    try:
        finalize(completion)
    except Exception as error:
        immutable(FOLDER / 'postprocessing_status.json', {'status': 'FAILED_PRESERVED_FOR_REVIEW',
                  'error_type': type(error).__name__, 'API_calls': 0})
        raise


if __name__ == '__main__':
    main()
