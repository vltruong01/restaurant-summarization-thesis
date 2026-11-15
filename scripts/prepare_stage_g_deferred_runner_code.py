"""Create a new execution runner; preserve all historically bound source files."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
source = root / 'scripts/run_stage_g_production_d042_six_worker_v1.py'
destination = root / 'scripts/run_stage_g_production6_deferred_v1.py'
assert not destination.exists()
text = source.read_text(encoding='utf-8')
text = text.replace('frozen target logic and zero automatic retries.', 'deferred individual failures, frozen validation, zero automatic retries.')
text = text.replace("accepted.add(j['job_id'])", "accepted.add(j['job_id']); failure_map.pop(j['job_id'], None)")
text = text.replace("'maximum_concurrency': 6,", "'maximum_concurrency': 6, 'failure_policy': 'defer_individual_task_continue_others',")
text = text.replace("    atomic(OUT / 'execution_progress.json', report)", "    atomic(OUT / 'deferred_tasks.json', {'updated_utc': now(), 'tasks': list(failure_map.values()), 'automatic_resends': 0})\n    atomic(OUT / 'execution_progress.json', report)")
start = text.index("    if report['failed_or_unresolved']:\n")
end = text.index('    if len(complete) == 2200:', start)
text = text[:start] + text[end:]
text = text.replace("pending = [j for j in jobs if j['job_id'] not in accepted and j['job_id'] not in imports]", "pending = pending_jobs(jobs, accepted, imports, report['failed_or_unresolved'])")
start = text.index('            if any(not r[\'passed\'] for r in results):')
end = text.index('    assert len(complete)==2200', start)
text = text[:start] + '''            # Failure is quarantined; no retry and no acceptance of invalid output.
            # Stop only for account-wide auth failures, backend alias mismatch,
            # or three entirely failed batches (avoid a provider-wide failure flood).
            terminal = False
            for item in results:
                if item['passed']: continue
                cp_path = OUT/'checkpoints'/(item['job_id']+'.json')
                cp = read(cp_path) if cp_path.exists() else {}
                terminal |= cp.get('http_status') in (401, 402, 403)
                terminal |= cp.get('local_validation_error') == 'Reported model differs from requested alias; review provider provenance'
            consecutive_failed_batches = consecutive_failed_batches + 1 if all(not r['passed'] for r in results) else 0
            if terminal or consecutive_failed_batches >= 3:
                report,accepted,complete=progress(jobs,businesses,imports,[],phase,'Provider-wide safety stop; deferred requests preserved, no automatic resend')
                update_docs(report); return
    if report['failed_or_unresolved']:
        report.update(status='GENERATION_SWEEP_FINISHED_PENDING_RECOVERY', phase='deferred_recovery_required', current_jobs=[])
        atomic(OUT/'generation_sweep_report.json', report)
        update_docs(report)
        return
''' + text[end:]
text = text.replace('    with ThreadPoolExecutor(max_workers=6) as pool:', '    consecutive_failed_batches = 0\n    with ThreadPoolExecutor(max_workers=6) as pool:')
at = text.index('\ndef main():')
text = text[:at] + '''
def pending_jobs(jobs, accepted, imports, failures):
    deferred = {x['job_id'] for x in failures}
    return [j for j in jobs if j['job_id'] not in accepted and j['job_id'] not in imports and j['job_id'] not in deferred]

''' + text[at:]
destination.write_text(text, encoding='utf-8')
config = json.loads((root/'configs/stage_g_guarded_production6_v1.json').read_text(encoding='utf-8'))
config['status'] = 'USER_APPROVED_DEFER_INDIVIDUAL_FAILURES_CONTINUE_SIX_REQUESTS'
config['output_directory'] = 'outputs/stage_g/teacher_production_d042_v1/guarded_production6_deferred_v1'
config['production_retry_policy'] = 'zero_automatic_retries_defer_failed_or_unresolved_tasks_continue_other_tasks_review_before_resend'
config['identifier_guard']['provider_ignores_guard'] = 'preserve_invalid_output_defer_request_no_silent_downgrade'
config['production_scheduling']['small_batch_gate'] = 'prior_six_request_trial_and_production_PASS_reused'
config['production_scheduling']['individual_failure_policy'] = 'defer_continue_other_tasks'
config['execution']['runner'] = 'guarded_six_request_deferred_failure_v1'
config['execution']['parent_config'] = 'configs/stage_g_guarded_production6_v1.json'
config['execution']['reviewed_retry_job'] = 'train_1mLyYz0skNlRGP6wXBEmHQ_review_consensus'
config['execution']['maximum_reviewed_retry_calls'] = 1
config['execution']['provider_outage_policy'] = 'stop_on_auth_or_alias_mismatch_or_three_all_failed_batches'
target = root/'configs/stage_g_production6_deferred_v1.json'
assert not target.exists()
target.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
print('New versioned runner and execution configuration created; old files unchanged')
