"""Execution-only overlay: untouched jobs first, no failed-request resend."""
import argparse
import json
import os
from pathlib import Path

import continue_stage_g_production6_retry_v1 as parent

ROOT = parent.ROOT
OVERLAY = parent.RUN / 'defer_errors_20261012_v1'
POLICY = ROOT / 'configs/stage_g_production6_defer_errors_20261012_v1.json'
SCRIPT = Path(__file__).resolve()
ORIGINAL_UPDATE = parent.update_docs
OWNS_LOCK = False


def defer_decision(jid, attempts, epoch=None):
    if not attempts:
        return 'ready', 0
    cp = parent.checkpoint(jid, attempts[-1])
    if cp is None:
        return 'unresolved', 0
    if cp['status'] == parent.PASS:
        return 'accepted', 0
    if cp.get('http_status') in (401, 402, 403):
        return 'fatal', 0
    if cp.get('local_validation_error') == 'Reported model differs from requested alias; review provider provenance':
        return 'fatal', 0
    return 'deferred', 0


def error_details(item):
    jid = item['job_id']
    attempts = parent.ATTEMPTS[jid]
    entry = dict(item, automatic_resend=False)
    if not attempts:
        return entry
    directory = attempts[-1]['directory']
    cp_path = directory / 'checkpoints' / (jid + '.json')
    diagnostic = directory / 'stream_diagnostics' / (jid + '.json')
    cp = parent.frozen.read(cp_path) if cp_path.exists() else {}
    entry.update(checkpoint_path=cp_path.relative_to(ROOT).as_posix() if cp_path.exists() else None,
                 diagnostic_path=diagnostic.relative_to(ROOT).as_posix() if diagnostic.exists() else None,
                 reason=cp.get('local_validation_error') or cp.get('error_type') or 'unknown_request_outcome',
                 http_status=cp.get('http_status'),
                 failed_utc=cp.get('failed_utc'),
                 original_attempts_preserved=True)
    if diagnostic.exists():
        d = parent.frozen.read(diagnostic)
        # Only the existing credential-redacted diagnostic fields are exported.
        entry['transport_failure_kind'] = d.get('failure_kind')
        entry['provider_error_redacted'] = d.get('provider_error') or d.get('provider_http_error_body_redacted')
    return entry


def update_docs(state):
    state.update(maximum_automatic_retries=0, retry_pending_count=0,
                 failure_policy='record_defer_continue_untouched_jobs',
                 execution_policy_path=POLICY.relative_to(ROOT).as_posix())
    if state.get('phase') in ('production_with_bounded_retries', 'resume_and_retry_check', 'retry_limit_or_unresolved_intents'):
        state['phase'] = 'production_without_retry' if state['status'] == 'RUNNING' else 'deferred_errors_pending_recovery'
    ORIGINAL_UPDATE(state)
    pointer_path = ROOT / 'outputs/stage_g/active_production_execution_pointer.json'
    pointer = parent.frozen.read(pointer_path)
    pointer.update(execution_policy_path=POLICY.relative_to(ROOT).as_posix(),
                   deferred_tasks_path=(parent.OUT / 'deferred_tasks_no_retry.json').relative_to(ROOT).as_posix())
    parent.frozen.atomic(pointer_path, pointer)
    deferred = [error_details(item) for item in state.get('failed_or_unresolved', [])]
    parent.frozen.atomic(parent.OUT / 'deferred_tasks_no_retry.json', {
        'updated_utc': state['updated_utc'], 'automatic_retries': 0,
        'recovery_requires_later_researcher_instruction': True, 'tasks': deferred})
    # Replace only the scheduler status line. Preserve all scientific and audit notes.
    path = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
    text = path.read_text(encoding='utf-8')
    text = text.replace('max2 retries with30/60s cooldown.',
                        'no automatic retries; failed/unresolved tasks recorded for later recovery.')
    path.write_text(text, encoding='utf-8')


def configure():
    parent.retry_decision = defer_decision
    parent.MAX_RETRIES = 0
    parent.SCRIPT = SCRIPT
    parent.update_docs = update_docs


def prepare():
    assert not (OVERLAY / 'authorization.json').exists(), 'Already prepared: use --run'
    state = parent.frozen.read(parent.OUT / 'execution_progress.json')
    assert state['status'] == 'STOPPED_REQUIRES_REVIEW' and not state['current_jobs'], 'Previous batch must drain'
    pause = parent.RUN / 'STOP_REQUESTED.json'
    assert pause.exists() and parent.frozen.read(pause).get('transition') == 'defer_errors_20261012_v1'
    print(json.dumps({'offline_preflight': 'VERIFYING_FROZEN_INPUTS', 'API_calls': 0}), flush=True)
    cfg, jobs, businesses, imports = parent.checked_inputs()
    policy = parent.frozen.read(POLICY)
    assert policy['maximum_concurrency'] == 6 and policy['automatic_retries'] == 0
    assert policy['frozen_request_config'] == parent.CONFIG.relative_to(ROOT).as_posix()
    assert not policy['scientific_payload_or_validation_changed']
    parent.load_attempts(jobs)
    protected = {}
    target_paths = set(parent.frozen.read(parent.RUN / 'reuse_targets.json').values())
    target_paths.update(p.relative_to(ROOT).as_posix() for p in (parent.OUT / 'targets').glob('*.json'))
    for rel in target_paths:
        protected[rel] = parent.frozen.file_sha(ROOT / rel)
    accepted_count = 0
    for job in jobs:
        path = parent.location(job, imports) / 'checkpoints' / (job['job_id'] + '.json')
        if path.exists() and parent.frozen.read(path)['status'] == parent.PASS:
            parent.verify_checkpoint(job, cfg, path.parent.parent, parent.CONFIG)
            protected[path.relative_to(ROOT).as_posix()] = parent.frozen.file_sha(path)
            accepted_count += 1
            if accepted_count % 500 == 0:
                print(json.dumps({'offline_preflight_verified_components': accepted_count, 'API_calls': 0}), flush=True)
    assert len(target_paths) == state['businesses_completed']
    assert accepted_count == state['accepted_tasks']
    for path in (parent.CONFIG, POLICY, SCRIPT, parent.SCRIPT,
                 parent.OUT / 'execution_progress.json', parent.OUT / 'retry_queue.json',
                 parent.OUT / 'batch_log.jsonl', pause,
                 ROOT / 'outputs/stage_g/active_production_execution_pointer.json'):
        destination = OVERLAY / 'before_transition' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(path.read_bytes())
    parent.frozen.immutable(OVERLAY / 'authorization.json', {
        'authorized_by': policy['authorization'], 'created_utc': parent.frozen.now(),
        'initial_businesses': state['businesses_completed'], 'initial_accepted_components': accepted_count,
        'effective_policy_sha256': parent.frozen.file_sha(POLICY),
        'runner_sha256': parent.frozen.file_sha(SCRIPT),
        'frozen_request_config_sha256': parent.frozen.file_sha(parent.CONFIG),
        'preserved_artifact_sha256': protected,
        'request_payloads_validation_provider_key_and_scientific_settings_unchanged': True,
        'maximum_concurrency': 6, 'automatic_retries': 0, 'API_calls_during_preflight': 0,
        'Codex_configuration_or_auth_access': False})
    with (ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n### D-042 execution refinement — defer errors until later recovery (2026-10-12)\n'
            'The researcher requested prioritizing untouched production jobs, recording failures and recovering them later. '
            'The prior batch was drained before activating the versioned execution-only overlay. Successful components and '
            'complete targets are re-used without regeneration; every failed/unknown outcome is preserved and receives no '
            'automatic resend during this sweep. Maximum concurrency remains six. Recent gateway_concurrency_limit errors '
            'are provider-side admission failures; no guarantee of error-free API execution is made. Authentication/billing, '
            'model-alias/provenance, local integrity and sustained all-failed-batch stops remain in force. '
            'Incomplete businesses remain incomplete; a finished sweep with deferred tasks is not Stage G completion. '
            'Policy: configs/stage_g_production6_defer_errors_20261012_v1.json; runner: '
            'scripts/continue_stage_g_production6_defer_errors_20261012_v1.py; authorization/snapshots: '
            'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/defer_errors_20261012_v1. '
            'Deferred task details: execution_v1/deferred_tasks_no_retry.json. Frozen request config remains a historical '
            'payload/validation binding; the effective scheduler policy is the separate overlay. D040/D043 prompts, model, '
            'schemas, evidence checks, Stage F, K5, budgets, teacher-only credentials and existing human status are unchanged. '
            'No test teacher targets, manual output edits, Codex settings changes or LED training.\n')
    # Remove only our checked transition signal, after preserving its exact bytes.
    assert pause.resolve().parent == parent.RUN.resolve()
    pause.unlink()
    print(json.dumps({'offline_preflight': 'PASS', 'businesses_reused': len(target_paths),
                      'accepted_components_reused': accepted_count, 'automatic_retries': 0}), flush=True)


def verify_overlay():
    m = parent.frozen.read(OVERLAY / 'authorization.json')
    assert parent.frozen.file_sha(POLICY) == m['effective_policy_sha256']
    assert parent.frozen.file_sha(SCRIPT) == m['runner_sha256']
    assert parent.frozen.file_sha(parent.CONFIG) == m['frozen_request_config_sha256']
    for rel, sha in m['preserved_artifact_sha256'].items():
        assert parent.frozen.file_sha(ROOT / rel) == sha, rel


def main():
    global OWNS_LOCK
    parser = argparse.ArgumentParser()
    for name in ('prepare-and-run', 'run', 'audit'):
        parser.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    assert sum(vars(args).values()) == 1
    lock = None if args.audit else parent.parent.parent.parent.restored.shared_lock()
    OWNS_LOCK = not args.audit
    if args.prepare_and_run:
        prepare()
    verify_overlay()
    configure()
    if args.audit:
        parent.audit_run()
    else:
        assert not (parent.RUN / 'STOP_REQUESTED.json').exists(), 'Respect pause request'
        parent.run()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        parent.frozen.atomic(OVERLAY / 'runner_failure.json', {'status': 'STOPPED_REQUIRES_REVIEW',
            'error_type': type(error).__name__, 'updated_utc': parent.frozen.now()})
        path = parent.OUT / 'execution_progress.json'
        if path.exists() and OWNS_LOCK:
            state = parent.frozen.read(path)
            state.update(status='STOPPED_REQUIRES_REVIEW', current_jobs=[],
                stop_reason='Deferred-error overlay preflight/runner failure: ' + type(error).__name__,
                updated_utc=parent.frozen.now())
            update_docs(state)
        print(json.dumps({'status': 'STOPPED_REQUIRES_REVIEW', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
