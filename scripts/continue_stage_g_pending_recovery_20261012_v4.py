"""One researcher-authorized recovery pass, then the unchanged no-retry sweep."""
import argparse
import json
import os
import time
from pathlib import Path

import continue_stage_g_production6_retry_v1 as parent
import continue_stage_g_production6_defer_errors_20261012_v1 as defer
import stage_g_teacher_recovery_wait_v1 as recovery_transport

ROOT = parent.ROOT
SCRIPT = Path(__file__).resolve()
POLICY = ROOT / 'configs/stage_g_pending_recovery_20261012_v4.json'
OVERLAY = parent.RUN / 'pending_recovery_20261012_v4'
AUTH = OVERLAY / 'authorization.json'
ORIGINAL_UPDATE = parent.update_docs
ALLOW = {}
OWNS_LOCK = False


def remaining_recoveries():
    return sorted(jid for jid, item in ALLOW.items()
                  if len(parent.ATTEMPTS[jid]) == item['recorded_attempts'])


def recovery_decision(jid, attempts, epoch=None):
    decision, wait = defer.defer_decision(jid, attempts, epoch)
    if decision in ('accepted', 'fatal', 'unresolved'):
        return decision, wait
    pending = remaining_recoveries()
    if jid in pending:
        return ('ready', 0) if jid in pending[:3] else ('deferred', 0)
    if pending and not attempts:
        return 'deferred', 0
    return decision, wait


def recovery_report():
    entries = []
    for jid, old in sorted(ALLOW.items()):
        attempts = parent.ATTEMPTS[jid]
        attempted = len(attempts) > old['recorded_attempts']
        cp = parent.checkpoint(jid, attempts[-1]) if attempts else None
        status = ('RECOVERED_STRUCTURAL_PASS' if attempted and cp and cp['status'] == parent.PASS
                  else 'RECOVERY_FAILED_DEFERRED' if attempted and cp
                  else 'RECOVERY_IN_FLIGHT_OR_UNKNOWN' if attempted else 'PENDING')
        entries.append(dict(old, job_id=jid, recovery_status=status,
                            additional_attempts=len(attempts)-old['recorded_attempts'],
                            latest_checkpoint=(attempts[-1]['directory'] / 'checkpoints' / (jid+'.json'))
                            .relative_to(ROOT).as_posix() if attempts else None,
                            latest_error=(cp or {}).get('local_validation_error') or (cp or {}).get('error_type')))
    report = {'updated_utc': parent.frozen.now(), 'authorized_pending_tasks': len(ALLOW),
              'recovered': sum(x['recovery_status'] == 'RECOVERED_STRUCTURAL_PASS' for x in entries),
              'failed_again': sum(x['recovery_status'] == 'RECOVERY_FAILED_DEFERRED' for x in entries),
              'pending': sum(x['recovery_status'] == 'PENDING' for x in entries),
              'in_flight_or_unknown': sum(x['recovery_status'] == 'RECOVERY_IN_FLIGHT_OR_UNKNOWN' for x in entries),
              'maximum_additional_attempts_per_task': 1, 'request_payloads_unchanged': True,
              'structural_validation_is_not_human_factual_certification': True, 'tasks': entries}
    parent.frozen.atomic(OVERLAY / 'recovery_report.json', report)
    return report


def update_docs(state):
    report = recovery_report()
    state.update(maximum_automatic_retries=0,
                 failure_policy='one_authorized_pending_recovery_then_defer_new_failures',
                 execution_policy_path=POLICY.relative_to(ROOT).as_posix(),
                 pending_recovery_report=(OVERLAY / 'recovery_report.json').relative_to(ROOT).as_posix(),
                 recovered_pending_tasks=report['recovered'], recovery_failed_tasks=report['failed_again'])
    recovery_active = report['pending'] or report['in_flight_or_unknown']
    if state['status'] == 'RUNNING':
        state['phase'] = 'authorized_pending_recovery' if recovery_active else 'production_without_retry'
    ORIGINAL_UPDATE(state)
    pointer_path = ROOT / 'outputs/stage_g/active_production_execution_pointer.json'
    pointer = parent.frozen.read(pointer_path)
    pointer.update(execution_policy_path=POLICY.relative_to(ROOT).as_posix(),
                   pending_recovery_report=state['pending_recovery_report'])
    parent.frozen.atomic(pointer_path, pointer)
    deferred = [defer.error_details(x) for x in state.get('failed_or_unresolved', [])
                if x['state'] not in ('ready', 'cooldown')]
    parent.frozen.atomic(parent.OUT / 'deferred_tasks_no_retry.json', {
        'updated_utc': state['updated_utc'], 'automatic_retries': 0,
        'recovery_requires_later_researcher_instruction': True, 'tasks': deferred})
    path = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
    text = path.read_text(encoding='utf-8').replace('max2 retries with30/60s cooldown.',
        'one explicitly authorized pending-failure recovery; new failures deferred without automatic retry.')
    path.write_text(text, encoding='utf-8')


def configure():
    recovery_transport.EXTENDED_JOBS = frozenset(ALLOW)
    parent.transport = recovery_transport
    parent.retry_decision = recovery_decision
    parent.MAX_RETRIES = 0
    parent.SCRIPT = SCRIPT
    parent.update_docs = update_docs


def prepare_handoff():
    global OWNS_LOCK, ALLOW
    assert not AUTH.exists(), 'Already prepared; use --run'
    policy = parent.frozen.read(POLICY)
    assert policy['maximum_recovery_concurrency'] == 3
    assert policy['maximum_production_concurrency'] == 6
    assert policy['additional_attempts_per_confirmed_pending_failure'] == 1
    assert not policy['request_payload_and_validation_changed']
    assert policy['frozen_request_config'] == parent.CONFIG.relative_to(ROOT).as_posix()
    pause = parent.RUN / 'STOP_REQUESTED.json'
    assert not pause.exists(), 'Respect an existing pause request'
    print(json.dumps({'offline_preflight': 'VERIFYING_FROZEN_INPUTS_WHILE_PRODUCTION_CONTINUES', 'API_calls': 0}), flush=True)
    # Preflight remains read-only and does not own the production lock or call the API.
    parent.checked_inputs()
    print(json.dumps({'offline_preflight': 'PASS', 'handoff': 'DRAIN_CURRENT_BATCH', 'API_calls': 0}), flush=True)
    parent.frozen.immutable(pause, {'transition': 'pending_recovery_20261012_v4',
                                  'requested_utc': parent.frozen.now(), 'reason': policy['authorization']})
    deadline = time.monotonic() + 600
    lock = None
    while lock is None:
        assert time.monotonic() < deadline, 'Timed out draining previous producer; do not start competing calls'
        try:
            lock = parent.parent.parent.parent.restored.shared_lock()
        except OSError:
            time.sleep(1)
    OWNS_LOCK = True
    assert parent.frozen.read(pause)['transition'] == 'pending_recovery_20261012_v4'
    state = parent.frozen.read(parent.OUT / 'execution_progress.json')
    assert state['status'] == 'STOPPED_REQUIRES_REVIEW' and not state['current_jobs']
    cfg, jobs, businesses, imports = parent.checked_inputs()
    parent.load_attempts(jobs)
    candidates = {}
    protected = {}
    for job in jobs:
        jid = job['job_id']
        attempts = parent.ATTEMPTS[jid]
        if jid in parent.read_index() or any((parent.checkpoint(jid,a) or {}).get('status') == parent.PASS for a in attempts):
            continue
        if attempts and defer.defer_decision(jid, attempts)[0] == 'deferred':
            item = defer.error_details({'job_id': jid, 'state': 'deferred', 'recorded_attempts': len(attempts)})
            candidates[jid] = item
    for directory in (parent.OUT / 'attempts').glob('attempt_*'):
        for sub in ('checkpoints', 'raw_requests', 'raw_responses', 'request_intents', 'stream_diagnostics'):
            for path in (directory / sub).glob('*'):
                if path.is_file():
                    protected[path.relative_to(ROOT).as_posix()] = parent.frozen.file_sha(path)
    target_paths = set(parent.frozen.read(parent.RUN / 'reuse_targets.json').values())
    target_paths.update(p.relative_to(ROOT).as_posix() for p in (parent.OUT / 'targets').glob('*.json'))
    assert len(target_paths) == state['businesses_completed']
    for rel in target_paths:
        protected[rel] = parent.frozen.file_sha(ROOT / rel)
    for path in (parent.OUT / 'execution_progress.json', parent.OUT / 'deferred_tasks_no_retry.json',
                 parent.OUT / 'batch_log.jsonl', ROOT / 'outputs/stage_g/active_production_execution_pointer.json', pause):
        dest = OVERLAY / 'before_handoff' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(path.read_bytes())
    parent.frozen.immutable(AUTH, {'created_utc': parent.frozen.now(), 'authorization': policy['authorization'],
        'pending_tasks': candidates, 'initial_businesses_completed': state['businesses_completed'],
        'initial_accepted_components': state['accepted_tasks'], 'runner_sha256': parent.frozen.file_sha(SCRIPT),
        'policy_sha256': parent.frozen.file_sha(POLICY),
        'transport_adapter_sha256': parent.frozen.file_sha(ROOT / 'scripts/stage_g_teacher_recovery_wait_v1.py'), 'frozen_request_config_sha256': parent.frozen.file_sha(parent.CONFIG),
        'preserved_artifact_sha256': protected, 'API_calls_during_preflight': 0,
        'Codex_configuration_or_auth_access': False})
    ALLOW = candidates
    with (ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n### D-042 execution-only pending-request recovery (2026-10-12)\n'
            f'The researcher explicitly requested handling pending errors and continuing production. After read-only '
            f'preflight and a drained single-producer handoff, {len(candidates)} confirmed failed components were '
            'frozen for one additional identical-payload API attempt each, in groups of at most three. Only this '
            'confirmed recovery queue uses a socket wait of 300 seconds and wall-clock check of 360 seconds '
            'to accommodate delayed responses. Original SSE interpretation, request headers, strict span and output '
            'validation are unchanged. This is an execution wait adjustment, not a teacher-generation or research '
            'method amendment. Unknown outcomes, successful components and new failures are not resent. Normal '
            'fresh production delegates to the original 120/180-second transport and maximum six-call core. All '
            'prior failures and targets, D040/D043, model/settings, Stage F, K5 and budgets are preserved. Content '
            'is Teacher API-only; no Codex settings access, test teacher generation or LED training. This is one '
            'operator-authorized extra attempt for exhausted timing failures, with every earlier attempt preserved. '
            'Policy: configs/stage_g_pending_recovery_20261012_v4.json; authorization and reports: '
            'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/pending_recovery_20261012_v4/.\n')
    assert pause.resolve().parent == parent.RUN.resolve()
    pause.unlink()
    print(json.dumps({'handoff': 'PASS', 'authorized_pending_failures': len(candidates),
                      'businesses_preserved': len(target_paths), 'maximum_recovery_concurrency': 3}), flush=True)
    return lock


def verify_overlay():
    global ALLOW
    auth = parent.frozen.read(AUTH)
    assert parent.frozen.file_sha(SCRIPT) == auth['runner_sha256']
    assert parent.frozen.file_sha(POLICY) == auth['policy_sha256']
    assert parent.frozen.file_sha(ROOT / 'scripts/stage_g_teacher_recovery_wait_v1.py') == auth['transport_adapter_sha256']
    assert parent.frozen.file_sha(parent.CONFIG) == auth['frozen_request_config_sha256']
    for rel, sha in auth['preserved_artifact_sha256'].items():
        assert parent.frozen.file_sha(ROOT / rel) == sha, rel
    ALLOW = auth['pending_tasks']


def main():
    global OWNS_LOCK
    parser = argparse.ArgumentParser()
    for name in ('prepare-and-run', 'run', 'audit'):
        parser.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    assert sum(vars(args).values()) == 1
    if args.prepare_and_run:
        lock = prepare_handoff()
    elif args.run:
        lock = parent.parent.parent.parent.restored.shared_lock()
        OWNS_LOCK = True
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
            'error_type': type(error).__name__, 'updated_utc': parent.frozen.now(), 'owns_production_lock': OWNS_LOCK})
        if OWNS_LOCK:
            state = parent.frozen.read(parent.OUT / 'execution_progress.json')
            state.update(status='STOPPED_REQUIRES_REVIEW', current_jobs=[],
                         stop_reason='Pending recovery overlay failure: ' + type(error).__name__,
                         updated_utc=parent.frozen.now())
            update_docs(state)
        print(json.dumps({'status': 'STOPPED_REQUIRES_REVIEW', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
