"""User-authorized billing recovery; frozen Coletoken payloads/validators unchanged."""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import continue_stage_g_production6_retry_v1 as parent

ROOT = parent.ROOT
RUN = parent.RUN / 'billing_resume_20261012_v1'
SCRIPT = Path(__file__).resolve()
ORIGINAL_DECISION = parent.retry_decision
AUTHORIZED = {}


def billing_resume_decision(jid, attempts, epoch=None):
    result = ORIGINAL_DECISION(jid, attempts, epoch)
    entry = AUTHORIZED.get(jid)
    if result[0] != 'fatal' or not entry or not attempts:
        return result
    path = attempts[-1]['directory'] / 'checkpoints' / (jid + '.json')
    if path != ROOT / entry['checkpoint_path'] or parent.frozen.file_sha(path) != entry['checkpoint_sha256']:
        return result
    if len(attempts) >= 1 + parent.MAX_RETRIES:
        return 'exhausted', 0
    cp = parent.frozen.read(path)
    assert cp['http_status'] == 403 and cp['error_type'] == 'HTTPError'
    assert cp['binding']['provider'] == 'coletoken'
    failed = datetime.fromisoformat(cp['failed_utc']).timestamp()
    remaining = failed + (30, 60)[min(len(attempts)-1, 1)] - (time.time() if epoch is None else epoch)
    return ('ready', 0) if remaining <= 0 else ('cooldown', remaining)


def main(audit=False):
    if audit:
        parent.audit_run()
        return
    lock = parent.parent.parent.parent.restored.shared_lock()
    assert not (RUN / 'authorization.json').exists(), 'Resume activation already recorded; no blind duplicate activation'
    state = parent.frozen.read(parent.OUT / 'execution_progress.json')
    assert state['status'] == 'STOPPED_REQUIRES_REVIEW' and not state['current_jobs']
    assert state['businesses_completed'] == 639
    assert not (parent.RUN / 'STOP_REQUESTED.json').exists(), 'Respect any explicit pause request'
    cfg = parent.frozen.read(parent.CONFIG)
    assert cfg['execution']['provider'] == 'coletoken' and cfg['execution']['maximum_concurrency'] == 6
    assert not os.environ.get(parent.KEY_VARIABLE), 'Use the unchanged teacher-only private key file'
    base, provider, key = parent.credentials()
    assert (base, provider, cfg['requested_model']) == (parent.BASE_URL, 'coletoken', parent.MODEL)
    assert key and not any(c.isspace() for c in key)
    # Secret remains private; never save it or read Codex settings.
    del key
    jobs = parent.frozen.read(parent.frozen.FOLDER / 'production_jobs.json')
    parent.load_attempts(jobs)
    entries = {}
    for failure in state['failed_or_unresolved']:
        if failure['state'] != 'fatal':
            continue
        jid = failure['job_id']
        latest = parent.ATTEMPTS[jid][-1]['directory']
        cp_path = latest / 'checkpoints' / (jid + '.json')
        diagnostic = parent.frozen.read(latest / 'stream_diagnostics' / (jid + '.json'))
        cp = parent.frozen.read(cp_path)
        error = json.loads(diagnostic.get('provider_http_error_body_redacted', '{}')).get('error', {})
        assert cp['http_status'] == 403 and cp['error_type'] == 'HTTPError'
        assert cp['binding']['provider'] == 'coletoken' and cp['binding']['base_url'] == parent.BASE_URL
        assert error.get('type') == 'billing_error' and error.get('message', '').lower() == 'insufficient balance'
        assert len(parent.ATTEMPTS[jid]) < 1 + parent.MAX_RETRIES
        entries[jid] = {'checkpoint_path': cp_path.relative_to(ROOT).as_posix(),
                        'checkpoint_sha256': parent.frozen.file_sha(cp_path),
                        'diagnostic_sha256': parent.frozen.file_sha(latest / 'stream_diagnostics' / (jid + '.json')),
                        'recorded_attempts_before_resume': len(parent.ATTEMPTS[jid])}
    assert len(entries) == 2, 'Only the two confirmed historical insufficient-balance failures may be reopened'
    for path in (parent.CONFIG, ROOT / 'outputs/stage_g/active_production_execution_pointer.json',
                 parent.OUT / 'execution_progress.json', parent.OUT / 'retry_queue.json'):
        destination = RUN / 'before_resume' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(path.read_bytes())
    authorization = {'authorized_by': 'researcher: return to previous Coletoken teacher key; balance topped up',
                     'billing_recovery_scope': entries, 'maximum_concurrency': 6,
                     'existing_retry_limit_unchanged': True, 'initial_businesses': 639, 'initial_targets': 1917,
                     'initial_accepted_components': 1925, 'config_path': parent.CONFIG.relative_to(ROOT).as_posix(),
                     'config_sha256': parent.frozen.file_sha(parent.CONFIG), 'runner_sha256': parent.frozen.file_sha(SCRIPT),
                     'original_runner_sha256': parent.frozen.file_sha(Path(parent.__file__).resolve()),
                     'new_failure_policy': 'original_auth_billing_stop_applies_to_every_new_attempt',
                     'exhausted_review_failure_remains_deferred': True, 'Mirai_outputs_imported': False,
                     'scientific_settings_changed': False, 'Codex_configuration_or_auth_access': False,
                     'authorized_utc': parent.frozen.now()}
    parent.frozen.immutable(RUN / 'authorization.json', authorization)
    AUTHORIZED.update(entries)
    parent.retry_decision = billing_resume_decision
    parent.SCRIPT = SCRIPT  # Completion audit uses this wrapper's --audit route.
    with (ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\nUser-authorized Coletoken resume after balance top-up (2026-10-12): return to the unchanged '
                     'teacher-only Coletoken key/endpoint and original D040/D043 generation/validation settings, max6 requests. '
                     'Reuse639 complete businesses/1917 targets/1925 accepted components. A hash-bound execution overlay '
                     'reopens only the two recorded403 billing_error/insufficient balance checkpoints within their existing '
                     'remaining retry allowance. Any new authentication/billing failure still stops; no counter reset or '
                     'unlimited retries. The exhausted72H2 review remains deferred. Mirai outputs/trials preserved separately '
                     'and not imported. Before-resume snapshots and authorization: '
                     'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/billing_resume_20261012_v1. '
                     'No teacher targets for held-out test, Codex account/config changes, manual target rewrites or LED training.\n')
    print(json.dumps({'billing_resume_authorization': 'PASS', 'reopened_historical_billing_tasks': 2,
                      'existing_businesses': 639, 'maximum_concurrency': 6}), flush=True)
    parent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    assert args.run != args.audit
    try:
        main(audit=args.audit)
    except Exception as error:
        parent.frozen.atomic(RUN / 'resume_failure.json', {'status': 'STOPPED_REQUIRES_REVIEW',
                            'error_type': type(error).__name__, 'updated_utc': parent.frozen.now()})
        p = parent.OUT / 'execution_progress.json'
        if p.exists():
            state = parent.frozen.read(p)
            state.update(status='STOPPED_REQUIRES_REVIEW', current_jobs=[],
                         stop_reason='Billing resume preflight/runner failure: ' + type(error).__name__,
                         updated_utc=parent.frozen.now())
            parent.update_docs(state)
        print(json.dumps({'status': 'STOPPED_REQUIRES_REVIEW', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
