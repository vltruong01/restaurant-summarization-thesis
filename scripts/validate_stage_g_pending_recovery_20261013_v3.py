"""Offline preservation and raw-API binding audit; no requests or target edits."""
import gzip
import json
from collections import Counter
from pathlib import Path

import continue_stage_g_pending_recovery_20261013_v3 as runner
import stage_g_business_id_guard_v1 as guard


def main():
    root = runner.ROOT
    folder = runner.OVERLAY
    read = runner.parent.frozen.read
    sha = runner.parent.frozen.file_sha
    immutable = runner.parent.frozen.immutable
    auth = read(runner.AUTH)
    recovery = read(folder / 'recovery_report.json')
    pending = auth['pending_tasks']
    assert not any(recovery[key] for key in ('blocked', 'pending', 'in_flight_or_unknown'))
    assert recovery['quota_renewal_gate_passed']
    for path, expected in ((runner.SCRIPT, auth['runner_sha256']),
                           (runner.POLICY, auth['policy_sha256']),
                           (root / 'scripts/stage_g_teacher_recovery_wait_v1.py', auth['transport_adapter_sha256']),
                           (runner.parent.CONFIG, auth['frozen_request_config_sha256'])):
        assert sha(path) == expected, str(path)
    protected = auth['preserved_artifact_sha256']
    old_targets = set(read(runner.parent.RUN / 'reuse_targets.json').values())
    prefix = (runner.parent.OUT / 'targets').relative_to(root).as_posix() + '/'
    old_targets.update(rel for rel in protected if rel.startswith(prefix))
    assert len(old_targets) == auth['initial_businesses_completed']
    old_failures = {rel for rel in protected if any('/' + jid + '.' in rel for jid in pending)}
    for rel in old_targets | old_failures:
        assert sha(root / rel) == protected[rel], rel
    preservation = {
        'status': 'PASS', 'checked_utc': runner.parent.frozen.now(),
        'existing_complete_business_target_files_unchanged': len(old_targets),
        'original_failed_attempt_artifacts_unchanged': len(old_failures),
        'runner_policy_transport_and_frozen_request_hashes': 'PASS',
        'API_calls': 0, 'Codex_configuration_or_auth_access': False,
    }
    immutable(folder / 'preserved_artifact_validation.json', preservation)
    jobs = {job['job_id']: job for job in read(root / 'outputs/stage_g/teacher_production_d042_v1/production_jobs.json')}
    cfg = read(runner.parent.CONFIG)
    usage = Counter()
    errors = Counter()
    details = []
    unresolved = []
    api_calls = 0
    for item in recovery['tasks']:
        jid = item['job_id']
        job = jobs[jid]
        assert job['split'] in ('train', 'validation')
        previous = (root / pending[jid]['checkpoint_path']).parent.parent
        additional = item['additional_attempts']
        assert 1 <= additional <= 2
        calls = 0
        after_old = False
        for attempt in sorted((runner.parent.OUT / 'attempts').glob('attempt_*')):
            if attempt == previous:
                after_old = True
                continue
            if not after_old or not (attempt / 'request_intents' / (jid + '.json')).exists():
                continue
            calls += 1
            raw_path = attempt / 'raw_responses' / (jid + '.json')
            if raw_path.exists():
                raw_usage = read(raw_path).get('usage') or {}
                if raw_usage:
                    usage['responses_with_usage'] += 1
                    usage['input_tokens'] += raw_usage.get('input_tokens') or 0
                    usage['output_tokens'] += raw_usage.get('output_tokens') or 0
                    usage['reasoning_tokens'] += (raw_usage.get('output_tokens_details') or {}).get('reasoning_tokens') or 0
        assert calls == additional
        api_calls += calls
        if item['recovery_status'] != 'RECOVERED_STRUCTURAL_PASS':
            assert item['recovery_status'] == 'RECOVERY_FAILED_DEFERRED'
            unresolved.append({key: item.get(key) for key in (
                'job_id', 'recovery_status', 'additional_attempts', 'latest_checkpoint', 'latest_error')})
            continue
        errors[pending[jid].get('provider_failure_code') or pending[jid].get('reason') or 'unspecified'] += 1
        latest = root / item['latest_checkpoint']
        directory = latest.parent.parent
        cp = guard.verify_checkpoint(job, cfg, directory, runner.parent.CONFIG)
        assert cp['api_called'] is True
        basename = jid + '.json.gz'
        assert gzip.decompress((previous / 'raw_requests' / basename).read_bytes()) == gzip.decompress((directory / 'raw_requests' / basename).read_bytes())
        details.append({
            'job_id': jid, 'business_id': job['business_id'], 'split': job['split'],
            'condition': job['condition'], 'checkpoint': item['latest_checkpoint'],
            'checkpoint_sha256': sha(latest), 'response_id': cp['response_id'],
            'additional_API_calls': calls, 'structural_validation': 'PASS',
            'request_body_unchanged': True, 'raw_API_content_binding': 'PASS',
            'old_failure_reason': pending[jid].get('reason'),
            'old_provider_failure_code': pending[jid].get('provider_failure_code'),
        })
    state = read(runner.parent.OUT / 'execution_progress.json')
    deferred = read(runner.parent.OUT / 'deferred_tasks_no_retry.json')
    report = {
        'status': 'PASS' if not unresolved else 'PARTIAL_RECOVERY_UNRESOLVED_DEFERRED',
        'validated_utc': runner.parent.frozen.now(),
        'authorized_failed_requests': len(pending), 'recovered_requests': len(details),
        'recovery_API_calls': api_calls,
        'all_recovered_on_first_additional_attempt': all(x['additional_API_calls'] == 1 for x in details),
        'independent_schema_evidence_request_and_raw_response_binding_replays': len(details),
        'recovered_original_error_counts': dict(errors),
        'provider': 'coletoken', 'requested_model_alias': 'gpt-6.1-sol',
        'generation_prompts_settings_and_scientific_inputs_unchanged': True,
        'usage_returned_by_provider': dict(usage), 'preservation_checks': preservation,
        'current_runtime': {key: state.get(key) for key in (
            'status', 'phase', 'updated_utc', 'pid', 'businesses_completed',
            'targets_completed', 'maximum_concurrency', 'current_jobs')},
        'new_deferred_task_count_at_validation': len(deferred['tasks']),
        'structural_validation_is_not_independent_human_factual_certification': True,
        'Codex_configuration_or_auth_access': False,
        'held_out_teacher_targets': 0, 'LED_training_started': False, 'tasks': details,
        'unresolved_tasks': unresolved,
    }
    immutable(folder / 'recovery_validation_report.json', report)
    print(json.dumps({key: report[key] for key in (
        'status', 'authorized_failed_requests', 'recovered_requests', 'recovery_API_calls',
        'all_recovered_on_first_additional_attempt', 'current_runtime', 'new_deferred_task_count_at_validation')}))


if __name__ == '__main__':
    main()
