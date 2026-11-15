"""Derive recovery scheduling from the preserved four-request runner, offline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'scripts/run_stage_g_d041_concurrency4_continuous.py').read_text(encoding='utf-8')


def replace(old, new):
    global source
    assert source.count(old) == 1, 'Unexpected scheduler source anchor'
    source = source.replace(old, new)


replace('from run_stage_g_d041_continuation_v2 import (', 'from stage_g_d041_recovery_v3 import (')
replace("PHASE = FOLDER / 'concurrency4_continuous_v1'", "PHASE = FOLDER / 'execution_control_v1'")
replace("    assert not (PHASE / 'manifest.json').exists(), 'Phase already started; no automatic restart'", "    assert not (PHASE / 'manifest.json').exists(), 'Phase already started; no automatic restart'\n    recovery_manifest = read(FOLDER / 'continuation_manifest.json')\n    retry_ids = recovery_manifest['operator_retry_job_ids']\n    assert len(retry_ids) == 2\n    import run_stage_g_teacher_pilot_v4_1_recovery as legacy\n    import stage_g_teacher_stream_diagnostics_v1 as diagnostic\n    legacy.call = diagnostic.call  # Runtime transport instrumentation; frozen source file unchanged.")
replace("        'API_request_intents': len(list((OUT / 'request_intents').glob('*.json'))),", "        'API_request_intents': len(list((OUT / 'request_intents').glob('*.json'))) + 2,")
source = source.replace("'imported_API_request_intents': 51", "'imported_API_request_intents': 152")
replace("'maximum_cohort_initial_calls': 292, 'pending_job_ids'", "'maximum_total_cohort_attempts_including_operator_retries': 294, 'namespace_unique_job_call_ceiling': 292, 'pending_job_ids'")
replace("'authorization_basis': 'User clarified that remaining pilot work should continue with four concurrent requests.'", "'authorization_basis': 'User approved two serial operator retries then unstarted pilot work with four concurrent requests.'")
replace("'initial_cohort_attempts': previous_calls", "'initial_cohort_attempts': previous_calls + 2")
replace("'runner_sha256': file_sha(ROOT / 'scripts/run_stage_g_d041_concurrency4_continuous.py')", "'runner_sha256': file_sha(ROOT / 'scripts/run_stage_g_d041_recovery_v3.py')")
replace("    def call(job):\n        begin = time.monotonic()", "    def call(job):\n        diagnostic.set_context(OUT / 'stream_diagnostics', job['job_id'])\n        begin = time.monotonic()")
anchor = '    with ThreadPoolExecutor(max_workers=4) as pool:'
recovery = '''    recovery_results = []
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
        'all_two_passed': len(recovery_results) == 2 and not any(r['failed'] for r in recovery_results),
        'previous_failed_attempts_preserved': True,
    })
    (PHASE / 'batch_results.jsonl').touch(exist_ok=False)
    pending = [j for j in pending if j['job_id'] not in retry_ids]
    with ThreadPoolExecutor(max_workers=4) as pool:'''
replace(anchor, recovery)
replace('        for index in range(0, len(pending), 4):', '        for index in range(0, len(pending), 4):\n            if stopped:\n                break')
replace("'new_API_request_intents': total, 'legacy_call_count_field_is_cumulative': True", "'new_API_request_intents': total + 2, 'legacy_call_count_field_is_cumulative': True, 'namespace_unique_job_request_intents': total")
replace("'cumulative_cohort_API_request_intents': total", "'cumulative_cohort_API_request_intents': total + 2")
replace("'new_initial_API_calls_this_continuation': total - 51", "'new_initial_API_calls_this_continuation': total - previous_calls, 'explicit_operator_recovery_calls': len(recovery_results), 'historical_failed_attempts_preserved': 2")
replace("'scheduling_history_note': 'Earlier serial1 and bounded4 trial preserved; remaining work uses four-request batches.'", "'scheduling_history_note': 'Previous namespaces and failures preserved; two user-approved serial recoveries, then unstarted tasks in four-request batches.'")
replace("'status': 'D041_STOPPED_FOR_REVIEW' if stopped else ('D041_ATTEMPTED_WITH_FAILURES' if failures else 'D041_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE')", "'status': 'D041_STOPPED_FOR_REVIEW' if stopped else ('D041_ATTEMPTED_WITH_FAILURES' if failures else 'D041_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE')")
start = source.index('def finalize(completion):')
end = source.index('\n\ndef main():', start)
finalizer = '''def finalize(completion):
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_stage_g_d041_recovery_v3.py'), '--audit-results'], cwd=ROOT, check=True)
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
    marker = '**D041 recovery_v3 execution result (2026-10-11):**'
    note = (f"{marker} {completion['businesses_completed']}/100 complete businesses; "
            f"{completion['target_texts']} provisional target texts; "
            f"{completion['cumulative_cohort_API_request_intents']} total cohort attempts including two preserved previous failures. "
            f"Current failures={len(completion['failed_tasks'])}; stop reason={completion['stop_reason']}. "
            "Two operator-approved serial recovery attempts then four-request batches; zero automatic retries. "
            f"Validation={validation['status']}; automated validation does not establish factual accuracy. "
            "Previous51 complete business targets and old raw outputs remain unchanged. Fixed25 ZIP/page are under recovery_v3. "
            "No test targets/production teacher freeze/full2200 generation/LED training.\\n")
    for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md', 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
        p = ROOT / rel
        text = p.read_text(encoding='utf-8')
        if marker not in text:
            first, rest = text.split('\\n', 1)
            p.write_text(first + '\\n\\n' + note + '\\n' + rest, encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_stage_g_d041_recovery_v3_review.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/execute_stage_g_d041_status_cell.py')], cwd=ROOT, check=True)
    immutable(FOLDER / 'postprocessing_status.json', {'status': 'COMPLETED', 'API_calls': 0,
              'businesses_complete': completion['businesses_completed'], 'factual_review_pending': True})
'''
source = source[:start] + finalizer + source[end:]
source = source.replace('Run remaining frozen D041 jobs in four-request batches; no automatic retries.', 'User-approved two serial recoveries then unstarted D041 tasks; instrumented transport and zero automatic retries.')
target = ROOT / 'scripts/run_stage_g_d041_recovery_v3.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
compile(source, str(target), 'exec')
print('Recovery v3 runner derived and syntax checked; no API/key access.')
