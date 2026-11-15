"""Researcher-authorized bounded recovery, then the unchanged no-retry sweep."""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import continue_stage_g_production6_retry_v1 as parent
import continue_stage_g_production6_defer_errors_20261012_v1 as defer
import stage_g_teacher_recovery_wait_v1 as recovery_transport

ROOT = parent.ROOT
SCRIPT = Path(__file__).resolve()
POLICY = ROOT / 'configs/stage_g_pending_recovery_20261013_v3.json'
OVERLAY = parent.RUN / 'pending_recovery_20261013_v3'
AUTH = OVERLAY / 'authorization.json'
ORIGINAL_UPDATE = parent.update_docs
ALLOW = {}
OWNS_LOCK = False
FRESH_JOB_ORDER = ()
IMPORTED_IDS = frozenset()
FRESH_CACHE = None
PENDING_CACHE = None
ERROR_CODE_CACHE = {}
QUOTA_GATE_JOB = None


def provider_error_code(value):
    """Read explicit provider codes; HTTP429 alone is not a quota diagnosis."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, dict):
        return None
    error = value.get('error')
    if isinstance(error, dict):
        value = error
    return value.get('code') or value.get('type')


def failure_code(jid, attempt):
    cache_key = (jid, str(attempt['directory']))
    if cache_key not in ERROR_CODE_CACHE:
        diagnostic = attempt['directory'] / 'stream_diagnostics' / (jid + '.json')
        if not diagnostic.exists():
            return None
        data = parent.frozen.read(diagnostic)
        ERROR_CODE_CACHE[cache_key] = provider_error_code(
            data.get('provider_error') or data.get('provider_http_error_body_redacted'))
    return ERROR_CODE_CACHE[cache_key]


def confirmed_old_quota_failure(jid, attempts):
    old = ALLOW.get(jid)
    if not old or len(attempts) != old['recorded_attempts']:
        return False
    checkpoint_path = (attempts[-1]['directory'] / 'checkpoints' / (jid + '.json')).relative_to(ROOT).as_posix()
    return old.get('provider_failure_code') == 'insufficient_quota' and checkpoint_path == old['checkpoint_path']


def guarded_defer_decision(jid, attempts, epoch=None):
    decision, wait = defer.defer_decision(jid, attempts, epoch)
    if decision in ('accepted', 'fatal', 'unresolved') or not attempts:
        return decision, wait
    if failure_code(jid, attempts[-1]) == 'insufficient_quota' and not confirmed_old_quota_failure(jid, attempts):
        return 'fatal', 0
    return decision, wait


def quota_gate_passed():
    if QUOTA_GATE_JOB is None:
        return True
    attempts = parent.ATTEMPTS[QUOTA_GATE_JOB]
    old = ALLOW[QUOTA_GATE_JOB]
    cp = parent.checkpoint(QUOTA_GATE_JOB, attempts[-1]) if attempts else None
    return bool(len(attempts) > old['recorded_attempts'] and cp and cp['status'] == parent.PASS)


def remaining_recoveries():
    global PENDING_CACHE
    signature = tuple((jid, len(parent.ATTEMPTS[jid])) for jid in sorted(ALLOW))
    if PENDING_CACHE is not None and PENDING_CACHE[0] == signature:
        return PENDING_CACHE[1]
    pending = []
    for jid, item in ALLOW.items():
        attempts = parent.ATTEMPTS[jid]
        cp = parent.checkpoint(jid, attempts[-1]) if attempts else None
        if (len(attempts) < item['maximum_recorded_attempts'] and cp
                and guarded_defer_decision(jid, attempts)[0] == 'deferred'):
            pending.append(jid)
    result = sorted(pending)
    PENDING_CACHE = (signature, result)
    return result


def select_fresh_window(job_order, attempts, imported_ids, limit=5):
    return tuple(jid for jid in job_order if jid not in imported_ids and not attempts.get(jid))[:limit]


def fresh_window():
    global FRESH_CACHE
    assert FRESH_JOB_ORDER, 'Frozen job list must be initialized before scheduling'
    if FRESH_CACHE is None:
        FRESH_CACHE = frozenset(select_fresh_window(FRESH_JOB_ORDER, parent.ATTEMPTS, IMPORTED_IDS))
    return FRESH_CACHE


def recovery_decision(jid, attempts, epoch=None):
    decision, wait = guarded_defer_decision(jid, attempts, epoch)
    if decision in ('accepted', 'fatal', 'unresolved'):
        return decision, wait
    pending = remaining_recoveries()
    if not quota_gate_passed():
        if jid != QUOTA_GATE_JOB:
            return 'deferred', 0
        if jid not in pending:
            # No usable generation response within the authorized bound: stop.
            return 'fatal', 0
        admitted = (QUOTA_GATE_JOB,)
    else:
        admitted = pending[:3]
    if jid in pending:
        if jid not in admitted:
            return 'deferred', 0
        cp = parent.checkpoint(jid, attempts[-1])
        cooldown = 30 if len(attempts) == 1 else 60
        remaining = datetime.fromisoformat(cp['failed_utc']).timestamp() + cooldown - (time.time() if epoch is None else epoch)
        return ('ready', 0) if remaining <= 0 else ('cooldown', remaining)
    if not attempts:
        if pending:
            return 'deferred', 0
        return ('ready', 0) if jid in fresh_window() else ('deferred', 0)
    return decision, wait


def recovery_report():
    entries = []
    for jid, old in sorted(ALLOW.items()):
        attempts = parent.ATTEMPTS[jid]
        attempted = len(attempts) > old['recorded_attempts']
        cp = parent.checkpoint(jid, attempts[-1]) if attempts else None
        blocked = guarded_defer_decision(jid, attempts)[0] == 'fatal'
        status = ('RECOVERED_STRUCTURAL_PASS' if attempted and cp and cp['status'] == parent.PASS
                  else 'RECOVERY_BLOCKED_ACCOUNT_OR_PROVENANCE' if blocked
                  else 'PENDING' if cp and len(attempts) < old['maximum_recorded_attempts']
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
              'blocked': sum(x['recovery_status'] == 'RECOVERY_BLOCKED_ACCOUNT_OR_PROVENANCE' for x in entries),
              'pending': sum(x['recovery_status'] == 'PENDING' for x in entries),
              'in_flight_or_unknown': sum(x['recovery_status'] == 'RECOVERY_IN_FLIGHT_OR_UNKNOWN' for x in entries),
              'maximum_additional_attempts_per_task': 2, 'recorded_attempt_bound': 'handoff_recorded_attempts_plus_two', 'request_payloads_unchanged': True,
              'quota_renewal_gate_job': QUOTA_GATE_JOB, 'quota_renewal_gate_passed': quota_gate_passed(),
              'new_insufficient_quota_stops_after_batch_drain': True,
              'structural_validation_is_not_human_factual_certification': True, 'tasks': entries}
    parent.frozen.atomic(OVERLAY / 'recovery_report.json', report)
    return report


def update_docs(state):
    global FRESH_CACHE, PENDING_CACHE
    FRESH_CACHE = None
    PENDING_CACHE = None
    state['maximum_concurrency'] = 5
    report = recovery_report()
    state.update(maximum_automatic_retries=0,
                 failure_policy='bounded_authorized_pending_recovery_then_defer_new_failures',
                 execution_policy_path=POLICY.relative_to(ROOT).as_posix(),
                 pending_recovery_report=(OVERLAY / 'recovery_report.json').relative_to(ROOT).as_posix(),
                 recovered_pending_tasks=report['recovered'], recovery_failed_tasks=report['failed_again'])
    recovery_active = report['pending'] or report['in_flight_or_unknown']
    if state['status'] == 'RUNNING':
        state['phase'] = ('quota_renewal_generation_check' if not quota_gate_passed()
                          else 'authorized_pending_recovery' if recovery_active else 'production_without_retry')
    ORIGINAL_UPDATE(state)
    pointer_path = ROOT / 'outputs/stage_g/active_production_execution_pointer.json'
    pointer = parent.frozen.read(pointer_path)
    pointer.update(maximum_concurrency=5, execution_policy_path=POLICY.relative_to(ROOT).as_posix(),
                   pending_recovery_report=state['pending_recovery_report'])
    parent.frozen.atomic(pointer_path, pointer)
    deferred = [defer.error_details(x) for x in state.get('failed_or_unresolved', [])
                if x['state'] not in ('ready', 'cooldown')]
    parent.frozen.atomic(parent.OUT / 'deferred_tasks_no_retry.json', {
        'updated_utc': state['updated_utc'], 'automatic_retries': 0,
        'recovery_requires_later_researcher_instruction': True, 'tasks': deferred})
    path = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
    text = path.read_text(encoding='utf-8').replace('max2 retries with30/60s cooldown.',
        'bounded explicitly authorized pending-failure recovery; new failures deferred without automatic retry.')
    text = '\n'.join(line.replace('max6 requests;', 'max5 requests;') if line.startswith('**D-042 live production status:**') else line for line in text.split('\n'))
    path.write_text(text, encoding='utf-8')


def configure():
    global FRESH_JOB_ORDER, IMPORTED_IDS
    _, jobs, _, _ = parent.checked_inputs()
    assert len(jobs) == 6600
    FRESH_JOB_ORDER = tuple(sorted(j['job_id'] for j in jobs))
    IMPORTED_IDS = frozenset(parent.read_index())
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
    assert policy['maximum_production_concurrency'] == 5
    assert policy['additional_attempts_per_confirmed_pending_failure'] == 2
    assert not policy['request_payload_and_validation_changed']
    assert policy['frozen_request_config'] == parent.CONFIG.relative_to(ROOT).as_posix()
    pause = parent.RUN / 'STOP_REQUESTED.json'
    assert not pause.exists(), 'Respect an existing pause request'
    print(json.dumps({'offline_preflight': 'VERIFYING_FROZEN_INPUTS_WHILE_PRODUCTION_CONTINUES', 'API_calls': 0}), flush=True)
    # Preflight remains read-only and does not own the production lock or call the API.
    parent.checked_inputs()
    print(json.dumps({'offline_preflight': 'PASS', 'handoff': 'DRAIN_CURRENT_BATCH', 'API_calls': 0}), flush=True)
    parent.frozen.immutable(pause, {'transition': 'pending_recovery_20261013_v3',
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
    assert parent.frozen.read(pause)['transition'] == 'pending_recovery_20261013_v3'
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
        if attempts and guarded_defer_decision(jid, attempts)[0] == 'deferred':
            item = defer.error_details({'job_id': jid, 'state': 'deferred', 'recorded_attempts': len(attempts)})
            item['provider_failure_code'] = failure_code(jid, attempts[-1])
            item['maximum_recorded_attempts'] = len(attempts) + 2
            candidates[jid] = item
    assert set(candidates) <= {j['job_id'] for j in jobs}
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
    with (ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n### D-042 execution-only pending request recovery (2026-10-13)\n'
            f'The researcher requested recovery of pending errors and continued production. After read-only '
            f'preflight and a drained single-producer handoff, {len(candidates)} confirmed failed components were '
            'frozen for at most two additional identical-payload Teacher API calls each, with 30/60-second '
            'cooldowns and at most three recovery requests per batch. Known recovery jobs retain the existing '
            '300-second socket wait and 360-second wall-clock check. Raw responses, failed attempts and '
            'successful targets remain immutable; caption/photo order and verbatim-caption quotation checks '
            'are unchanged. No target or evidence label is manually rewritten. New insufficient_quota '
            'errors stop after the current batch drains; this request does not reopen any quota failure. '
            'Authentication/model-provenance stops and unknown-outcome no-resend safeguards remain unchanged. '
            'Fresh production resumes with at most five requests per drained batch under the existing '
            '120/180-second transport; new nonfatal failures are recorded and deferred for later recovery. '
            'This is execution recovery only. D040/D043 prompts, model/settings, selected reviews, Stage F, '
            'K5 and 5000/4000/750 budgets remain unchanged. Content is Teacher API-only; no Codex settings '
            'access, held-out teacher targets or LED training. Policy: '
            'configs/stage_g_pending_recovery_20261013_v3.json; authorization/reports: '
            'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/pending_recovery_20261013_v3/.\n')
    assert pause.resolve().parent == parent.RUN.resolve()
    pause.unlink()
    print(json.dumps({'handoff': 'PASS', 'authorized_pending_failures': len(candidates),
                      'businesses_preserved': len(target_paths), 'maximum_recovery_concurrency': 3}), flush=True)
    return lock


def verify_overlay():
    global ALLOW, QUOTA_GATE_JOB
    auth = parent.frozen.read(AUTH)
    assert parent.frozen.file_sha(SCRIPT) == auth['runner_sha256']
    assert parent.frozen.file_sha(POLICY) == auth['policy_sha256']
    assert parent.frozen.file_sha(ROOT / 'scripts/stage_g_teacher_recovery_wait_v1.py') == auth['transport_adapter_sha256']
    assert parent.frozen.file_sha(parent.CONFIG) == auth['frozen_request_config_sha256']
    for rel, sha in auth['preserved_artifact_sha256'].items():
        assert parent.frozen.file_sha(ROOT / rel) == sha, rel
    ALLOW = auth['pending_tasks']
    assert all(item.get('provider_failure_code') != 'insufficient_quota' for item in ALLOW.values()), 'Quota failures are not reopened by this authorization'
    quota_jobs = sorted(jid for jid, item in ALLOW.items() if item.get('provider_failure_code') == 'insufficient_quota')
    QUOTA_GATE_JOB = quota_jobs[0] if quota_jobs else None


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
