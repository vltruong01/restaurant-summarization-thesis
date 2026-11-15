"""Finite offline completion audit for the running recovery; never calls the API."""
import json
import time
from pathlib import Path

import continue_stage_g_pending_recovery_20261014_v2 as runner
import validate_stage_g_pending_recovery_20261014_v2 as validation


def main():
    read = runner.parent.frozen.read
    deadline = time.monotonic() + 6 * 60 * 60
    while time.monotonic() < deadline:
        report_path = runner.OVERLAY / 'recovery_report.json'
        failure = runner.OVERLAY / 'runner_failure.json'
        if report_path.exists() and runner.AUTH.exists():
            recovery = read(report_path)
            state = read(runner.parent.OUT / 'execution_progress.json')
            owned = state.get('execution_policy_path') == runner.POLICY.relative_to(runner.ROOT).as_posix()
            clear = recovery['recovered'] == recovery['authorized_pending_tasks']
            stopped = owned and state['status'] != 'RUNNING' and not state.get('current_jobs')
            if clear or stopped:
                validation.main()
                report = read(runner.OVERLAY / 'recovery_validation_report.json')
                heading = '### D-042 recovery result before fresh production (2026-10-14, v2)'
                progress = runner.ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
                assert heading not in progress.read_text(encoding='utf-8')
                audit = report['preservation_checks']
                with progress.open('a', encoding='utf-8') as out:
                    out.write(
                        '\n\n' + heading + '\n'
                        'The researcher explicitly requested clearing the existing failed requests before further fresh production. '
                        'Fresh production was paused after its current batch drained, before offline preflight. '
                        f"The immutable authorization contained {report['authorized_failed_requests']} confirmed failures. "
                        f"Recovery obtained {report['recovered_requests']} structurally validated components in "
                        f"{report['recovery_API_calls']} additional dispatched API attempts; "
                        f"{len(report['unresolved_tasks'])} authorized jobs remain unresolved. "
                        'The versioned scheduler admits at most three recovery requests per batch and at most two additional '
                        'attempts per authorized failure. Failed/exhausted or unknown-outcome recovery jobs do not unlock fresh '
                        'production. Only after every authorized job has a validated PASS may fresh production resume, with '
                        'a maximum of five requests in flight. New production failures retain the existing defer policy. '
                        'Seventeen offline scheduler safety tests passed, including checks that exhausted/unknown recoveries '
                        'cannot unlock fresh work, single generation availability gating, and no new dispatch after HTTP503. All recovered components independently passed the unchanged schema/evidence '
                        'validator, raw API output binding and byte-identical request replay. '
                        f"All {audit['existing_complete_business_target_files_unchanged']} pre-recovery complete-business "
                        f"target files and {audit['original_failed_attempt_artifacts_unchanged']} old failed-attempt artifacts "
                        'remained unchanged. Raw attempts, returned usage and response metadata remain preserved. '
                        'Structural PASS is not independent human factual certification. No teacher text or verification '
                        'label was manually rewritten. D040/D043 prompts, provider/model/settings, schemas, Stage F, '
                        'K=5, selected reviews and 5000/4000/750 budgets remain unchanged. No Codex config/auth access, '
                        'held-out teacher generation or LED training. Validation status: ' + report['status'] + '. '
                        'Immutable report: outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/'
                        'pending_recovery_20261014_v2/recovery_validation_report.json.\n')
                runner.parent.frozen.immutable(runner.OVERLAY / 'offline_completion_audit_status.json', {
                    'status': report['status'], 'updated_utc': runner.parent.frozen.now(),
                    'API_calls_by_auditor': 0, 'progress_log_updated': True,
                    'recovered': report['recovered_requests'], 'unresolved': len(report['unresolved_tasks']),
                })
                return
        elif failure.exists():
            runner.parent.frozen.immutable(runner.OVERLAY / 'offline_completion_audit_status.json', {
                'status': 'PREFLIGHT_OR_RUNNER_FAILURE_BEFORE_RECOVERY_REPORT',
                'updated_utc': runner.parent.frozen.now(), 'API_calls_by_auditor': 0,
                'runner_failure': read(failure),
            })
            return
        time.sleep(20)
    runner.parent.frozen.immutable(runner.OVERLAY / 'offline_completion_audit_status.json', {
        'status': 'OFFLINE_AUDIT_WAIT_TIMEOUT_NO_PRODUCER_INTERVENTION',
        'updated_utc': runner.parent.frozen.now(), 'API_calls_by_auditor': 0,
    })


if __name__ == '__main__':
    main()
