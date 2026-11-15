"""Six-worker diagnostic trial: reuse five successes, retry recorded upstream failures."""
import gzip
import argparse
import json
import os
import threading
import time
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import run_stage_g_mirai_stream_recovery_after_retry_v2 as old
import stage_g_mirai_retry_policy_v2 as policy
from stage_g_teacher_credentials import KEY_VARIABLE, MODEL, key_file

frozen = old.frozen
ROOT = old.ROOT
CONFIG = ROOT / 'configs/stage_g_mirai_retry6_trial_v4.json'
RUN = frozen.FOLDER / 'mirai_retry6_trial_v4'
OUT = RUN / 'execution_v1'
SCRIPT = Path(__file__).resolve()
PROVIDER_RECOVERED = False
HISTORICAL_FAILED_ATTEMPTS = 2


def select_jobs(jobs, businesses, imports, previous):
    """One failed job + five untouched jobs; no successful request regenerated."""
    by_id = {j['job_id']: j for j in jobs}
    successes = [r['job_id'] for r in previous['results'] if r.get('content_valid')]
    failures = [r['job_id'] for r in previous['results'] if not r.get('content_valid')]
    assert len(successes) == 5 and len(failures) == 1
    failed = failures[0]
    latest = frozen.FOLDER / 'mirai_retry6_trial_v3/execution_v1/attempts/attempt_01'
    diagnostic = frozen.read(latest / 'stream_diagnostics' / (failed + '.json'))
    plan = policy.retry_plan(diagnostic, HISTORICAL_FAILED_ATTEMPTS)
    if plan['reason'] == 'upstream_credentials_unavailable':
        assert PROVIDER_RECOVERED, 'Provider upstream repair must be explicitly confirmed before a bounded probe'
        plan = {'action': 'retry', 'reason': 'user_confirmed_provider_repair_bounded_probe', 'wait_seconds': 0}
    assert plan['action'] == 'retry', 'Latest failure is not safely retryable'
    parent = old.prior.parent
    parent.load_attempts(jobs)
    index = parent.read_index()
    groups = defaultdict(list)
    for job in jobs:
        groups[job['business_id']].append(job)
    excluded = {by_id[jid]['business_id'] for jid in successes + failures}
    prior_trial = frozen.read(frozen.FOLDER / 'mirai_retry6_trial_v3/manifest.json')
    excluded.update(by_id[jid]['business_id'] for jid in prior_trial['job_ids'] if jid != failed)
    untouched = []
    for business in businesses:
        if business['business_id'] in excluded or business['split'] not in ('train', 'validation'):
            continue
        group = groups[business['business_id']]
        if any(j.get('skip_api_empty_captions') or j['job_id'] in index for j in group):
            continue
        if any((parent.location(j, imports) / sub / (j['job_id'] + '.json')).exists()
               for j in group for sub in ('checkpoints', 'request_intents')):
            continue
        untouched += group
        if len(untouched) >= 5:
            break
    assert len(untouched) >= 5
    return [by_id[failed]] + untouched[:5], successes, plan


def main():
    assert PROVIDER_RECOVERED, 'Mirai reports no active upstream credentials; confirm external repair before a new paid probe'
    lock = old.restored.shared_lock()
    assert not (RUN / 'manifest.json').exists(), 'Preserve prior attempts; no blind restart/resend'
    base, jobs, businesses, imports = frozen.check_inputs()
    cfg = frozen.read(CONFIG)
    previous = frozen.read(old.RUN / 'report.json')
    manifest = frozen.read(old.RUN / 'manifest.json')
    assert all(cfg[k] == base[k] == frozen.read(old.CONFIG)[k] for k in frozen.FIXED_KEYS)
    assert cfg['identifier_guard'] == frozen.read(old.CONFIG)['identifier_guard']
    assert cfg['execution']['maximum_concurrency'] == 6 and not cfg['full_generation_enabled']
    assert cfg['execution']['base_url'] == old.BASE_URL and cfg['execution']['provider'] == old.PROVIDER
    assert frozen.file_sha(old.CONFIG) == manifest['config_sha256']
    for rel, sha in manifest['code_sha256'].items():
        assert frozen.file_sha(ROOT / rel) == sha, rel
    preserved = dict(manifest['preserved_artifact_sha256'])
    for previous_dir in (old.RUN, frozen.FOLDER / 'mirai_retry6_trial_v3'):
        for path in previous_dir.rglob('*'):
            if path.is_file():
                preserved[path.relative_to(ROOT).as_posix()] = frozen.file_sha(path)
    for rel, sha in preserved.items():
        assert frozen.file_sha(ROOT / rel) == sha, rel
    current = frozen.read(old.prior.parent.OUT / 'execution_progress.json')
    assert current['status'] == 'STOPPED_REQUIRES_REVIEW' and not current['current_jobs']
    selected, successes, historical_plan = select_jobs(jobs, businesses, imports, previous)
    schemas = frozen.read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    payloads = {}
    for job in selected:
        jid = job['job_id']
        body = old.guard.payload_for_job(job, cfg, prompts[job['task']], schemas[job['task']])
        payloads[jid] = body
        if jid == selected[0]['job_id']:
            assert frozen.canonical(body) == gzip.decompress((old.OUT / 'raw_requests' / (jid + '.json.gz')).read_bytes())
    accepted = {}
    by_id = {j['job_id']: j for j in jobs}
    for jid in successes:
        cp = frozen.read(old.OUT / 'checkpoints' / (jid + '.json'))
        response = frozen.read(ROOT / cp['source_response'])
        assert frozen.file_sha(ROOT / cp['source_response']) == cp['source_response_sha256']
        assert old.validate_content(by_id[jid], cfg, response)['output'] == cp['output']
        directory = old.OUT / 'stream_diagnostics'
        replay, _ = old.transport.replay(directory / (jid + '.sse.txt.gz'),
                                         frozen.read(directory / (jid + '.completed_wire.json')))
        assert replay == response
        accepted[jid] = cp
    assert not os.environ.get(KEY_VARIABLE), 'Use teacher-only private provider file'
    key = key_file().with_name(cfg['execution']['dedicated_secret_filename']).read_text(encoding='utf-8-sig').strip()
    assert key and not any(c.isspace() for c in key)
    frozen.immutable(RUN / 'manifest.json', {
        'authorized_by': 'user requests recovering upstream failure with six concurrent requests',
        'job_ids': [j['job_id'] for j in selected], 'reused_content_valid_job_ids': successes,
        'historically_failed_job_id': selected[0]['job_id'], 'maximum_concurrency': 6,
        'maximum_total_attempts_per_job_including_historical_failed_attempt': 3,
        'maximum_new_API_calls': 16, 'scientific_settings_changed': False,
        'historical_failed_attempts_for_recovery_job': HISTORICAL_FAILED_ATTEMPTS,
        'external_provider_repair_confirmed_by_operator': PROVIDER_RECOVERED,
        'config_sha256': frozen.file_sha(CONFIG), 'production_integration_approved': False,
        'preserved_artifact_sha256': preserved,
        'code_sha256': {p.relative_to(ROOT).as_posix(): frozen.file_sha(p) for p in
                       (SCRIPT, ROOT / 'scripts/stage_g_mirai_retry_policy_v2.py',
                        ROOT / 'scripts/stage_g_mirai_stream_adapter_v5.py')},
        'prepared_utc': frozen.now(), 'Codex_config_or_auth_access': False})
    old.note('User-authorized Mirai upstream recovery trial v3: retain five exact prior content-valid outputs; '
             'retry the one explicit upstream404/403 balance/reset failure plus five untouched train/validation tasks '
             'to verify six concurrent requests. Initial+two retries maximum including the previous failed attempt; '
             'provider reset/Retry-After honored,30/60-second minimum backoff, separate immutable attempts. '
             'Account auth/billing stops further retries; ambiguous timeouts/unknown routes are deferred, not resent. '
             'No content/schema/evidence relaxation or fabricated model metadata. Full production still blocked on '
             'missing model provenance;639 Coletoken businesses preserved. No Codex config/test targets/training. '
             'Namespace: outputs/stage_g/teacher_production_d042_v1/mirai_retry6_trial_v4.')
    print(json.dumps({'offline_preflight': 'PASS', 'reused_content_valid': 5, 'first_wave_calls': 6}), flush=True)
    started = time.monotonic()
    active = set()
    results = []
    attempts = []
    mutex = threading.RLock()
    stop = threading.Event()
    barrier = threading.Barrier(6)

    def publish(status='RUNNING_MIRAI_RETRY6_TRIAL'):
        with mutex:
            state = {'status': status, 'provider': old.PROVIDER, 'maximum_concurrency': 6,
                     'current_jobs': sorted(active), 'completed_jobs': len(results), 'results': results[:],
                     'attempts': attempts[:], 'reused_content_valid': 5, 'production_unchanged': True,
                     'pid': os.getpid(), 'updated_utc': frozen.now()}
            frozen.atomic(RUN / 'execution_progress.json', state)
            frozen.atomic(ROOT / 'outputs/stage_g/active_teacher_trial_pointer.json', {
                'status': status, 'provider': old.PROVIDER, 'maximum_concurrency': 6,
                'progress_path': (RUN / 'execution_progress.json').relative_to(ROOT).as_posix(),
                'trial_directory': RUN.relative_to(ROOT).as_posix(), 'production_resume_after_trial': False,
                'updated_utc': state['updated_utc']})

    def wait_cooldown(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if stop.wait(min(10, deadline - time.monotonic())):
                return False
        return not stop.is_set()

    def worker(job):
        jid = job['job_id']
        historical = jid == selected[0]['job_id']
        if historical and not wait_cooldown(historical_plan['wait_seconds']):
            return {'job_id': jid, 'content_valid': False, 'status': 'STOPPED_BEFORE_CALL'}
        barrier.wait(timeout=600)
        maximum_new = 3 - HISTORICAL_FAILED_ATTEMPTS if historical else 3
        historical_count = HISTORICAL_FAILED_ATTEMPTS if historical else 0
        for number in range(1, maximum_new + 1):
            if stop.is_set():
                return {'job_id': jid, 'content_valid': False, 'status': 'STOPPED_BEFORE_RETRY'}
            attempt = OUT / 'attempts' / ('attempt_%02d' % number)
            directory = attempt / 'stream_diagnostics'
            body = payloads[jid]
            path = attempt / 'raw_requests' / (jid + '.json.gz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(gzip.compress(frozen.canonical(body), mtime=0))
            frozen.immutable(attempt / 'request_intents' / (jid + '.json'), {
                'job_id': jid, 'recorded_attempt_number_including_history': number + historical_count,
                'request_sha256': frozen.digest(frozen.canonical(body)), 'config_sha256': frozen.file_sha(CONFIG),
                'requested_model': MODEL, 'provider': old.PROVIDER, 'base_url': old.BASE_URL,
                'started_utc': frozen.now()})
            old.transport.set_context(directory, jid)
            with mutex:
                active.add(jid)
                publish()
            retry_after = None
            try:
                response = old.transport.call(old.BASE_URL, key, body)
                projected = attempt / 'projected_responses' / (jid + '.json')
                frozen.immutable(projected, response)
                replay, lineage = old.transport.replay(directory / (jid + '.sse.txt.gz'),
                    frozen.read(directory / (jid + '.completed_wire.json')))
                assert replay == response
                diagnostic = frozen.read(directory / (jid + '.json'))
                provenance, model_ok = old.model_state(response, diagnostic, MODEL)
                content = old.validate_content(job, cfg, response)
                cp = {'status': 'CONTENT_VALID_NOT_PRODUCTION', 'content_valid': True, **content,
                      'model_provenance': provenance, 'model_guard_satisfied': model_ok,
                      'reported_model': response.get('model'), 'response_id': response.get('id'),
                      'usage': response.get('usage'), 'SSE_replay_match': True, 'output_projection': lineage,
                      'source_response': projected.relative_to(ROOT).as_posix(),
                      'source_response_sha256': frozen.file_sha(projected), 'production_accepted': False,
                      'teacher_text_rewritten': False}
                if provenance in ('MODEL_MISMATCH', 'MODEL_OBSERVATION_CONFLICT'):
                    stop.set()
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError):
                    retry_after = error.headers.get('Retry-After')
                cp = {'status': 'FAILED_RECORDED_NOT_PRODUCTION', 'content_valid': False,
                      'error_type': type(error).__name__, 'production_accepted': False}
                dpath = directory / (jid + '.json')
                diagnostic = frozen.read(dpath) if dpath.exists() else {}
                cp['http_status'] = diagnostic.get('http_status')
                cp['retry_plan'] = policy.retry_plan(diagnostic, number + historical_count, retry_after)
                if cp['retry_plan']['action'] == 'stop':
                    stop.set()
                if isinstance(error, ValueError):
                    # Error messages are not target corrections. Preserve exact API output separately.
                    cp['local_validation_error'] = str(error).replace(key, '[REDACTED]')
            finally:
                with mutex:
                    active.discard(jid)
            cp.update(job_id=jid, new_attempt=number, recorded_attempts_including_history=number + historical_count,
                      finished_utc=frozen.now())
            frozen.immutable(attempt / 'checkpoints' / (jid + '.json'), cp)
            with mutex:
                item = {'job_id': jid, 'new_attempt': number, 'content_valid': cp['content_valid'],
                        'http_status': cp.get('http_status'), 'retry_plan': cp.get('retry_plan')}
                attempts.append(item)
                publish()
            print(json.dumps(item), flush=True)
            if cp['content_valid']:
                with mutex:
                    accepted[jid] = cp
                return {'job_id': jid, 'content_valid': True, 'model_guard_satisfied': model_ok,
                        'model_provenance': provenance, 'new_attempts': number}
            plan = cp['retry_plan']
            if plan['action'] != 'retry' or not wait_cooldown(plan['wait_seconds']):
                return {'job_id': jid, 'content_valid': False, 'status': cp['status'], 'reason': plan['reason']}
        return {'job_id': jid, 'content_valid': False, 'reason': 'attempt_limit'}

    publish()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(worker, job) for job in selected]):
            result = future.result()
            with mutex:
                results.append(result)
                publish()
    candidates = []
    for business in businesses:
        group = [j for j in jobs if j['business_id'] == business['business_id']]
        if not all(j['job_id'] in accepted for j in group):
            continue
        target = frozen.build_target(business, {j['condition']: accepted[j['job_id']]['resolved_output'] for j in group})
        target.update(status='ISOLATED_MIRAI_REVIEW_CANDIDATE_NOT_PRODUCTION', production_accepted=False,
                      independent_human_verified=False,
                      model_provenance={j['condition']: accepted[j['job_id']]['model_provenance'] for j in group})
        frozen.immutable(OUT / 'review_candidates' / (business['business_id'] + '.json'), target)
        candidates.append(business['business_id'])
    usage = Counter()
    timeline = []
    failures = []
    for path in (OUT / 'attempts').glob('*/stream_diagnostics/*.json'):
        if path.name.endswith('.completed_wire.json'):
            continue
        diagnostic = frozen.read(path)
        timeline += [(datetime.fromisoformat(diagnostic[k]).timestamp(), d) for k, d in
                     (('started_utc', 1), ('finished_utc', -1))]
        u = diagnostic.get('response_usage') or {}
        usage['responses_with_usage'] += bool(u)
        usage['input_tokens'] += u.get('input_tokens', 0)
        usage['output_tokens'] += u.get('output_tokens', 0)
        usage['reasoning_tokens'] += (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        if diagnostic['status'] != 'COMPLETE_PROVIDER_STREAM_PROJECTED':
            failures.append({'diagnostic_path': path.relative_to(ROOT).as_posix(),
                             'http_status': diagnostic.get('http_status'),
                             'provider_http_error_body_redacted': diagnostic.get('provider_http_error_body_redacted')})
    live = peak = 0
    for _, delta in sorted(timeline, key=lambda x: (x[0], x[1])):
        live += delta
        peak = max(peak, live)
    for rel, sha in preserved.items():
        assert frozen.file_sha(ROOT / rel) == sha, rel
    valid = sum(r.get('content_valid', False) for r in results)
    report = {'status': 'MIRAI_RETRY_TRIAL_FINISHED_MODEL_REVIEW_REQUIRED' if valid == 6 else 'MIRAI_RETRY_TRIAL_WITH_UNRESOLVED_FAILURES',
              'new_jobs_content_valid': valid, 'new_jobs_failed': 6 - valid, 'reused_content_valid': 5,
              'total_content_valid_components': len(accepted), 'API_calls': len(attempts),
              'observed_client_transport_overlap_peak': peak, 'wall_seconds': round(time.monotonic() - started, 3),
              'all_prior_artifacts_unchanged': True, 'usage_returned_for_new_calls': dict(usage),
              'candidate_businesses': candidates, 'results': results, 'attempts': attempts, 'transport_failures': failures,
              'model_provenance_distribution': dict(Counter(cp['model_provenance'] for cp in accepted.values())),
              'production_accepted_components': 0, 'frozen_model_guard_relaxed': False,
              'teacher_text_rewritten': False, 'held_out_targets': 0, 'LED_training_started': False,
              'Codex_config_or_auth_access': False, 'completed_utc': frozen.now()}
    frozen.immutable(RUN / 'report.json', report)
    with (RUN / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('# Mirai bounded six-worker retry trial\n\n' + json.dumps(report, ensure_ascii=False, indent=2)
                     + '\n\nContent checks do not verify backend model identity or independent factual quality. '
                     'All targets remain isolated; no production admission.\n')
    publish(report['status'])
    old.note('Mirai bounded retry trial result: ' + json.dumps({k: report[k] for k in
             ('API_calls', 'new_jobs_content_valid', 'new_jobs_failed', 'reused_content_valid',
              'observed_client_transport_overlap_peak', 'wall_seconds', 'model_provenance_distribution')})
             + '. No model-guard relaxation or production import. Prior639-business Coletoken state preserved. '
             'Report: outputs/stage_g/teacher_production_d042_v1/mirai_retry6_trial_v4/REPORT.md.')
    print(json.dumps({k: report[k] for k in ('status', 'API_calls', 'new_jobs_content_valid', 'new_jobs_failed', 'wall_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bounded Mirai probe only after external upstream repair')
    parser.add_argument('--provider-recovered', action='store_true',
                        help='Use only after the researcher confirms Mirai repaired its upstream credentials')
    PROVIDER_RECOVERED = parser.parse_args().provider_recovered
    try:
        main()
    except Exception as error:
        frozen.atomic(RUN / 'runner_failure.json', {'status': 'STOPPED_REQUIRES_REVIEW',
                       'error_type': type(error).__name__, 'updated_utc': frozen.now()})
        print(json.dumps({'runner_error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
