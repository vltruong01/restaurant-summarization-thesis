"""One explicitly authorized credential-rotation trial; preserve all prior attempts."""
import argparse
import gzip
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from stage_g_production_d042 import (
    ROOT, CONFIG, FOLDER, OUT, PILOT_OUT, PILOT_CONFIG, read, file_sha,
    canonical, immutable, atomic, check_inputs, location, verify_checkpoint, now,
)
from run_stage_g_production_d042 import runner_lock, update_docs
from run_stage_g_extended_pilot_d041 import build_target
from run_stage_g_teacher_pilot_v4_1_recovery import job_result, payload
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL

PREVIOUS = FOLDER / 'execution_control_concurrency10_trial_v1'
TRIAL = FOLDER / 'execution_control_concurrency10_new_key_trial_v2'
ATTEMPT = TRIAL / 'execution_v1'
PASS = 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'


def stats(values):
    return {'n': len(values), 'min': min(values) if values else None,
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'max': max(values) if values else None}


def preserved_files():
    paths = []
    for name in ('checkpoints', 'raw_responses', 'raw_requests', 'targets',
                 'request_intents', 'stream_diagnostics'):
        paths.extend(p for p in (OUT / name).rglob('*') if p.is_file())
    paths.extend(p for p in PREVIOUS.rglob('*') if p.is_file())
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}


def assert_preserved():
    for rel, sha in read(TRIAL / 'prior_artifact_sha256.json').items():
        assert file_sha(ROOT / rel) == sha, rel


def prepare():
    lock = runner_lock()
    assert (FOLDER / 'STOP_REQUESTED.json').exists()
    assert not (TRIAL / 'manifest.json').exists(), 'Already prepared; no automatic repeat'
    cfg, jobs, businesses, imports = check_inputs()
    index = {j['job_id']: j for j in jobs}
    failures = read(PREVIOUS / 'failure_analysis.json')['errors']
    retry_ids = [r['job_id'] for r in failures]
    assert len(retry_ids) == 5 and len(set(retry_ids)) == 5
    for jid in retry_ids:
        assert read(OUT / 'checkpoints' / (jid + '.json'))['status'] != PASS
        assert (OUT / 'request_intents' / (jid + '.json')).exists()
    pending = [j for j in jobs if j['job_id'] not in imports
               and j['job_id'] not in retry_ids and not j.get('skip_api_empty_captions')
               and not (OUT / 'checkpoints' / (j['job_id'] + '.json')).exists()
               and not (OUT / 'request_intents' / (j['job_id'] + '.json')).exists()]
    selected = [index[jid] for jid in retry_ids] + pending[:5]
    assert len(selected) == 10 and len({j['job_id'] for j in selected}) == 10
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    request_hashes = {}
    for j in selected:
        assert j['split'] in ('train', 'validation') and not j.get('skip_api_empty_captions')
        body = canonical(payload(j, cfg, prompts[j['task']], schemas[j['task']]))
        if j['job_id'] in retry_ids:
            assert gzip.decompress((OUT / 'raw_requests' / (j['job_id'] + '.json.gz')).read_bytes()) == body
        p = ATTEMPT / 'raw_requests' / (j['job_id'] + '.json.gz')
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as stream:
            stream.write(gzip.compress(body, mtime=0))
        request_hashes[j['job_id']] = file_sha(p)
    immutable(TRIAL / 'prior_artifact_sha256.json', preserved_files())
    immutable(TRIAL / 'prior_progress.json', read(OUT / 'execution_progress.json'))
    manifest = {
        'approved_by': 'user', 'authorization': 'New teacher key and another bounded ten-concurrent-request trial',
        'retry_job_ids': retry_ids, 'new_job_ids': [j['job_id'] for j in pending[:5]],
        'job_ids': [j['job_id'] for j in selected], 'maximum_calls': 10,
        'maximum_concurrency': 10, 'automatic_retries': 0,
        'config_sha256': file_sha(CONFIG), 'jobs_sha256': file_sha(FOLDER / 'production_jobs.json'),
        'runner_sha256': file_sha(Path(__file__)), 'request_file_sha256': request_hashes,
        'credential_rotation': 'user_supplied_teacher_key_rotation_v2',
        'credential_values_or_fingerprints_recorded': False,
        'Codex_config_or_auth_access': False, 'scientific_protocol_changed': False,
        'retry_payloads_identical_to_previous_attempt': True,
        'test_requests': 0, 'continuous_concurrency10_approved': False,
        'previous_attempts_overwritten': False, 'API_calls_in_preflight': 0,
    }
    immutable(TRIAL / 'manifest.json', manifest)
    print(json.dumps({'offline_preflight': 'PASS', 'retry_tasks': 5,
                      'previously_unstarted_tasks': 5, 'paid_calls': 0}), flush=True)


def write_status(snapshot, active, status, phase, **extra):
    state = dict(snapshot, status=status, phase=phase, current_jobs=active,
                 pid=os.getpid(), maximum_concurrency=10 if active else 4,
                 updated_utc=now(), operational_trial_only=True,
                 attempt_namespace=ATTEMPT.relative_to(ROOT).as_posix(), **extra)
    atomic(OUT / 'execution_progress.json', state)
    atomic(TRIAL / 'execution_progress.json', state)
    return state


def run():
    lock = runner_lock()
    manifest = read(TRIAL / 'manifest.json')
    assert manifest['runner_sha256'] == file_sha(Path(__file__))
    assert not (TRIAL / 'execution_started.json').exists(), 'No automatic repeat of paid calls'
    assert (FOLDER / 'STOP_REQUESTED.json').exists()
    cfg, jobs, businesses, imports = check_inputs()
    assert manifest['config_sha256'] == file_sha(CONFIG)
    assert manifest['jobs_sha256'] == file_sha(FOLDER / 'production_jobs.json')
    assert_preserved()
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    by_id = {j['job_id']: j for j in jobs}
    selected = [by_id[jid] for jid in manifest['job_ids']]
    for j in selected:
        p = ATTEMPT / 'raw_requests' / (j['job_id'] + '.json.gz')
        assert file_sha(p) == manifest['request_file_sha256'][j['job_id']]
        assert gzip.decompress(p.read_bytes()) == canonical(payload(j, cfg, prompts[j['task']], schemas[j['task']]))
        for name in ('checkpoints', 'request_intents', 'raw_responses', 'stream_diagnostics'):
            assert not (ATTEMPT / name / (j['job_id'] + '.json')).exists()
    base, provider, key = credentials()
    assert (base, provider, cfg['requested_model']) == (BASE_URL, PROVIDER, MODEL)
    import run_stage_g_teacher_pilot_v4_1_recovery as legacy
    import stage_g_teacher_stream_diagnostics_v1 as diagnostic
    legacy.call = diagnostic.call
    snapshot = read(TRIAL / 'prior_progress.json')
    immutable(TRIAL / 'execution_started.json', {'started_utc': now(), 'maximum_paid_calls': 10,
               'manifest_sha256': file_sha(TRIAL / 'manifest.json'), 'automatic_retries': 0})
    active = manifest['job_ids'][:]
    state = write_status(snapshot, active[:], 'RUNNING', 'new_key_concurrency10_bounded_trial', stop_reason=None)
    update_docs(state)

    def worker(j):
        diagnostic.set_context(ATTEMPT / 'stream_diagnostics', j['job_id'])
        try:
            cp = job_result(j, cfg, prompts[j['task']], schemas[j['task']], base, provider, key, ATTEMPT, CONFIG)
            return {'job_id': j['job_id'], 'passed': True, 'seconds': cp.get('seconds'),
                    'response_id': cp.get('response_id')}
        except Exception as error:
            return {'job_id': j['job_id'], 'passed': False, 'error_type': type(error).__name__}

    start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(worker, j): j for j in selected}
        for future in as_completed(futures):
            result = future.result(); results.append(result); active.remove(result['job_id'])
            write_status(snapshot, active[:], 'RUNNING', 'new_key_concurrency10_bounded_trial', stop_reason=None)
            print(json.dumps(result), flush=True)
    elapsed = time.monotonic() - start
    successes = {r['job_id'] for r in results if r['passed']}
    for jid in successes:
        verify_checkpoint(by_id[jid], cfg, ATTEMPT, CONFIG)
    assert_preserved()
    assert file_sha(CONFIG) == manifest['config_sha256']

    timings = []; events = []; errors = []; tiers = Counter(); models = Counter()
    effort = Counter(); usage = Counter(); types = defaultdict(list)
    for j in selected:
        jid = j['job_id']; d = read(ATTEMPT / 'stream_diagnostics' / (jid + '.json'))
        timings.append(d); types[j['task']].append(d['elapsed_seconds'])
        events.extend([(datetime.fromisoformat(d['started_utc']).timestamp(), 1),
                       (datetime.fromisoformat(d['finished_utc']).timestamp(), -1)])
        rawp = ATTEMPT / 'raw_responses' / (jid + '.json')
        if rawp.exists():
            raw = read(rawp); tiers[str(raw.get('service_tier'))] += 1
            models[str(raw.get('model'))] += 1; effort[str((raw.get('reasoning') or {}).get('effort'))] += 1
            u = raw.get('usage') or {}; usage['responses_with_usage'] += bool(u)
            usage['input_tokens'] += u.get('input_tokens', 0); usage['output_tokens'] += u.get('output_tokens', 0)
            usage['reasoning_tokens'] += (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        if jid not in successes:
            cp = read(ATTEMPT / 'checkpoints' / (jid + '.json'))
            errors.append({'job_id': jid, 'http_status': d.get('http_status'),
                           'provider_error': d.get('provider_error'),
                           'provider_http_error_body_redacted': d.get('provider_http_error_body_redacted'),
                           'failure_kind': d.get('failure_kind'),
                           'local_validation_error': cp.get('local_validation_error'),
                           'error_type': cp.get('error_type')})
    running = peak = 0
    for _, delta in sorted(events, key=lambda pair: (pair[0], pair[1])):
        running += delta; peak = max(peak, running)

    # Versioned location overlay; failed original attempts are never overwritten.
    overrides = {jid: (ATTEMPT / 'checkpoints' / (jid + '.json')).relative_to(ROOT).as_posix()
                 for jid in sorted(successes)}
    immutable(TRIAL / 'successful_checkpoint_overrides.json', {
        'overrides': overrides, 'original_attempts_preserved': True,
        'production_runner_integration_pending': True, 'requires_versioned_resume': True,
        'manifest_sha256': file_sha(TRIAL / 'manifest.json')})
    grouped = defaultdict(list)
    for j in jobs: grouped[j['business_id']].append(j)
    complete = []; newly_complete = []
    for business in businesses:
        content = {}
        for j in grouped[business['business_id']]:
            directory = ATTEMPT if j['job_id'] in successes else location(j, imports)
            p = directory / 'checkpoints' / (j['job_id'] + '.json')
            if not p.exists() or read(p)['status'] != PASS: break
            cp = read(p); content[j['condition']] = cp.get('resolved_output', cp['output'])
        if len(content) != 3: continue
        complete.append(business['business_id'])
        if any(j['job_id'] in successes for j in grouped[business['business_id']]):
            target = build_target(business, content)
            immutable(ATTEMPT / 'targets' / (business['business_id'] + '.json'), target)
            newly_complete.append(business['business_id'])
    report = {
        'status': 'NEW_KEY_CONCURRENCY10_TECHNICAL_PASS' if len(successes) == 10 else 'NEW_KEY_CONCURRENCY10_FAILURE_STOPPED',
        'planned_calls': 10, 'actual_request_intents': len(list((ATTEMPT / 'request_intents').glob('*.json'))),
        'passed': len(successes), 'failed': 10 - len(successes),
        'retry_tasks_passed': len(successes & set(manifest['retry_job_ids'])),
        'new_tasks_passed': len(successes & set(manifest['new_job_ids'])),
        'batch_wall_seconds': round(elapsed, 3), 'observed_transport_overlap_peak': peak,
        'successful_requests_per_minute': 60 * len(successes) / elapsed,
        'request_seconds': stats([d['elapsed_seconds'] for d in timings]),
        'seconds_by_task': {k: stats(v) for k, v in types.items()},
        'task_mix': dict(Counter(j['task'] for j in selected)),
        'http_statuses': dict(Counter(str(d.get('http_status')) for d in timings)),
        'gateway_concurrency_limit_failures': sum('gateway_concurrency_limit' in str(e.get('provider_error')) for e in errors),
        'returned_models': dict(models), 'returned_service_tiers': dict(tiers),
        'returned_reasoning_efforts': dict(effort), 'usage_returned_by_provider': dict(usage),
        'usage_of_failed_responses_not_assumed_zero': True,
        'errors': errors, 'results': results, 'automatic_retries': 0,
        'previous_artifacts_unchanged': True, 'scientific_settings_unchanged': True,
        'retry_payloads_identical': True, 'Codex_config_or_auth_access': False,
        'all_successes_offline_schema_source_payload_validated': True,
        'teacher_API_only_target_content': True, 'human_factual_certification': False,
        'businesses_complete_with_versioned_overrides': len(complete), 'targets_complete': 3 * len(complete),
        'newly_completed_business_ids': newly_complete, 'new_target_namespace': ATTEMPT.relative_to(ROOT).as_posix(),
        'prior_trial_passed': read(PREVIOUS / 'report.json')['passed'],
        'benchmark_limitation': 'Different job mix, regenerated responses and transient gateway load; not a matched or sustained benchmark.',
        'production_concurrency10_adopted': False, 'production_resume_started': False,
        'next': 'Researcher review; retain versioned checkpoint overrides for future resumable production.',
    }
    immutable(TRIAL / 'report.json', report)
    assert_preserved()
    unresolved = [r for r in snapshot['failed_or_unresolved'] if r['job_id'] not in successes]
    unresolved += [{'job_id': r['job_id'], 'status': 'FAILED_NEW_KEY_TRIAL_NO_AUTOMATIC_RETRY'}
                   for r in results if not r['passed'] and r['job_id'] not in {x['job_id'] for x in unresolved}]
    total_usage = Counter(snapshot.get('usage_including_imported') or {}); total_usage.update(usage)
    state = write_status(snapshot, [], 'TRIAL_COMPLETED_AWAITING_REVIEW' if len(successes) == 10 else 'STOPPED_REQUIRES_REVIEW',
                         'new_key_concurrency10_trial_completed',
                         businesses_completed=len(complete), targets_completed=3 * len(complete),
                         new_businesses_completed=len(complete) - 100,
                         accepted_tasks=snapshot['accepted_tasks'] + len(successes),
                         new_API_attempts=snapshot['new_API_attempts'] + report['actual_request_intents'],
                         failed_or_unresolved=unresolved, historical_previous_trial_failures=5,
                         usage_including_imported=dict(total_usage),
                         checkpoint_override_manifest=(TRIAL / 'successful_checkpoint_overrides.json').relative_to(ROOT).as_posix(),
                         production_resume_requires_versioned_override_support=True,
                         stop_reason='Bounded new-key trial finished; production remains paused for researcher review')
    update_docs(state)
    note = (f"**D-042 teacher-only credential rotation / bounded ten-request trial v2:** {len(successes)}/10 technical PASS; "
            f"peak client transport overlap={peak}; wall={elapsed:.3f}s; provider gateway concurrency failures={report['gateway_concurrency_limit_failures']}. "
            f"{len(complete)}/2200 complete businesses with versioned successful checkpoint overrides. "
            "Five explicitly authorized failed-task retries plus five previously unstarted tasks; old attempts/checkpoints/raw outputs preserved byte-for-byte. "
            "Same model, prompts, medium reasoning, request payloads and default requested service behavior; no Codex configuration/authentication access. "
            "Targets exclusively Teacher API; automated validation is not human factual confirmation. "
            "Continuous concurrency10 not adopted; production remains paused and requires a versioned override-aware resume. "
            "Report: outputs/stage_g/teacher_production_d042_v1/execution_control_concurrency10_new_key_trial_v2/report.json.\n")
    for rel in ('thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md'):
        p = ROOT / rel; text = p.read_text(encoding='utf-8'); first, rest = text.split('\n', 1)
        p.write_text(first + '\n\n' + note + '\n' + rest, encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    assert args.prepare != args.run, 'Choose offline prepare OR one bounded execution'
    try:
        prepare() if args.prepare else run()
    except Exception as error:
        # Do not print exception messages that may contain provider text/secrets.
        atomic(TRIAL / 'runner_failure.json', {'error_type': type(error).__name__,
               'status': 'STOPPED_REQUIRES_REVIEW', 'automatic_retries': 0, 'updated_utc': now()})
        print(json.dumps({'runner_error_type': type(error).__name__, 'automatic_retries': 0}), flush=True)
        raise SystemExit(1)
