"""Two business slots, three independent frozen components each, at most six API calls.

Bounded repair + smoke run. Prior successful API components are reused after replay
and source validation. Failed/uncertain attempts are retained, never auto-resubmitted.
"""
import argparse
import gzip
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from stage_g_production_d042 import (
    ROOT, FOLDER, FIXED_KEYS, CONFIG as PARENT_CONFIG, read, canonical,
    file_sha, immutable, atomic, check_inputs, now,
)
from prepare_stage_g_teacher_pilot import digest
from run_stage_g_production_d042 import runner_lock
from run_stage_g_teacher_pilot_v4_1_recovery import payload, response_text, validate_output
from stage_g_canonical_review_ids import resolve_review, citation_diagnostics
from run_stage_g_extended_pilot_d041 import build_target
from stage_g_teacher_credentials import key_file, KEY_VARIABLE
from stage_g_teacher_wire_fields_v1 import project, self_check
import stage_g_mirai_stream_adapter_v3 as transport

CONFIG = ROOT / 'configs/stage_g_teacher_mirai_grouped6_v1.json'
RUN = FOLDER / 'mirai_grouped6_recovery_v1'
EXEC = RUN / 'execution_v1'
PRIOR_NAMES = ('mirai_concurrency6_stream_trial_v1', 'mirai_concurrency9_stream_trial_v1')


def context():
    parent, jobs, businesses, _ = check_inputs()
    cfg = read(CONFIG)
    assert all(cfg[k] == parent[k] for k in FIXED_KEYS)
    assert cfg['operational']['provider'] == 'miraiapi'
    assert cfg['operational']['base_url'] == 'https://api.miraiapi.com/v1'
    assert cfg['operational']['maximum_businesses_active'] == 2
    assert cfg['operational']['maximum_concurrency'] == 6
    assert not cfg['full_generation_enabled']
    bids = sum(cfg['operational']['phases'], [])
    selected = [j for j in jobs if j['business_id'] in bids]
    assert len(selected) == 3 * len(bids) == 12
    assert all(j['split'] in ('train', 'validation') for j in selected)
    return cfg, selected, {b['business_id']: b for b in businesses if b['business_id'] in bids}, read(ROOT / cfg['schema'])


def verify_response(job, cfg, schema, raw):
    assert raw['status'] == 'completed'
    reported = raw.get('model')
    if reported and reported != cfg['requested_model']:
        raise ValueError('Returned model alias mismatch')
    original = json.loads(response_text(raw))
    output, changes = project(job, original)
    validation = validate_output(job, output, schema[job['task']])
    resolved, evidence = resolve_review(job, output) if job['task'] == 'review_consensus' else (output, [])
    return {'status': 'CONTENT_PASS_PROVENANCE_PENDING' if not reported else 'CONTENT_AND_ALIAS_PASS',
            'api_output_original': original, 'output': output, 'wire_field_projection': changes,
            'resolved_output': resolved, 'evidence_resolution': evidence, 'validation': validation,
            'citation_metadata_diagnostics': citation_diagnostics(job, output) if job['task'] == 'review_consensus' else [],
            'reported_model': reported, 'requested_model': cfg['requested_model'],
            'response_id': raw.get('id'), 'usage': raw.get('usage'),
            'human_verified': False, 'production_accepted': False,
            'target_content_origin': 'teacher_API_only_exact_text_no_target_rewriting'}


def replay(directory, jid, raw):
    projector = transport.StreamProjection(); wire = None
    with gzip.open(directory / 'stream_diagnostics' / (jid + '.sse.jsonl.gz'), 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip() == '[DONE]': continue
            event = json.loads(line); projector.consume(event)
            if event.get('type') == 'response.completed': wire = event['response']
    assert wire is not None and projector.project(wire)[0] == raw
    assert wire == read(directory / 'stream_diagnostics' / (jid + '.completed_wire.json'))


def prepare():
    lock = runner_lock()
    cfg, jobs, businesses, schema = context()
    assert not (RUN / 'manifest.json').exists()
    preserved = read(FOLDER / PRIOR_NAMES[0] / 'prior_artifact_sha256.json')
    preserved.update({p.relative_to(ROOT).as_posix(): file_sha(p)
                      for p in (FOLDER / PRIOR_NAMES[0]).rglob('*') if p.is_file()})
    cached = {}; requests = {}
    for j in jobs:
        jid = j['job_id']
        prompt = (ROOT / cfg['prompts'][j['task']]).read_text(encoding='utf-8')
        body = canonical(payload(j, cfg, prompt, schema[j['task']]))
        reqpath = EXEC / 'raw_requests' / (jid + '.json.gz')
        reqpath.parent.mkdir(parents=True, exist_ok=True)
        with reqpath.open('xb') as f: f.write(gzip.compress(body, mtime=0))
        requests[jid] = file_sha(reqpath)
        for namespace in PRIOR_NAMES:
            directory = FOLDER / namespace / 'execution_v1'
            path = directory / 'projected_responses' / (jid + '.json')
            if not path.exists(): continue
            assert gzip.decompress((directory / 'raw_requests' / reqpath.name).read_bytes()) == body
            raw = read(path)
            try:
                cp = verify_response(j, cfg, schema, raw)
            except (ValueError, AssertionError):
                continue
            replay(directory, jid, raw)
            cp.update(job_sha256=digest(canonical(j)), source_response=path.relative_to(ROOT).as_posix(),
                      source_response_sha256=file_sha(path), reused_without_API_call=True)
            immutable(EXEC / 'checkpoints' / (jid + '.json'), cp)
            cached[jid] = cp['source_response']; break
    assert len(cached) == 4, 'Expected three successful visual components and one metadata-alias recovery'
    manifest = {'created_utc': now(), 'approved_by': 'user',
                'authorization': 'Fix request failures and run six requests as two complete businesses concurrently; bounded repair and verification before scaling',
                'phases': cfg['operational']['phases'], 'maximum_calls': 8,
                'maximum_businesses_active': 2, 'maximum_concurrency': 6, 'automatic_retries': 0,
                'job_ids': [j['job_id'] for j in jobs], 'cached_jobs': cached,
                'config_sha256': file_sha(CONFIG),
                'code_sha256': {rel: file_sha(ROOT / rel) for rel in (
                    'scripts/run_stage_g_mirai_grouped6_v1.py',
                    'scripts/stage_g_mirai_stream_adapter_v3.py', 'scripts/stage_g_teacher_wire_fields_v1.py')},
                'request_sha256': requests, 'prior_artifact_sha256': preserved,
                'offline_checks': {'wire_fields': self_check(), 'SSE_projection': transport.offline_self_check()},
                'scientific_settings_changed': False, 'original_Codex_configuration_accessed': False,
                'model_metadata_missing_not_fabricated': True, 'production_integration': False}
    immutable(RUN / 'manifest.json', manifest)
    print(json.dumps({'preflight': 'PASS', 'cached_components': len(cached), 'paid_calls_needed': 8,
                      'maximum_parallel_requests': 6, 'maximum_parallel_businesses': 2}), flush=True)


def run():
    lock = runner_lock()
    cfg, jobs, businesses, schema = context()
    manifest = read(RUN / 'manifest.json')
    assert manifest['config_sha256'] == file_sha(CONFIG)
    for rel, sha in manifest['code_sha256'].items(): assert file_sha(ROOT / rel) == sha
    for rel, sha in manifest['prior_artifact_sha256'].items(): assert file_sha(ROOT / rel) == sha, rel
    assert not os.environ.get(KEY_VARIABLE)
    key = key_file().with_name('mirai_api_key.txt').read_text(encoding='utf-8-sig').strip()
    assert key and not any(c.isspace() for c in key)
    if not (RUN / 'execution_started.json').exists():
        immutable(RUN / 'execution_started.json', {'started_utc': now(), 'manifest_sha256': file_sha(RUN / 'manifest.json')})
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    guard = threading.RLock(); active = set(); active_businesses = set(); complete = set()
    peak = [0, 0]; events = []; phase = ['recovery']; started = time.monotonic()

    def progress():
        with guard:
            atomic(RUN / 'execution_progress.json', {'status': 'RUNNING_GROUPED', 'phase': phase[0],
                'current_jobs': sorted(active), 'active_businesses': sorted(active_businesses),
                'completed_businesses': sorted(complete), 'maximum_concurrency': 6,
                'maximum_businesses_active': 2, 'observed_peak_requests': peak[0],
                'observed_peak_businesses': peak[1], 'pid': os.getpid(), 'updated_utc': now(),
                'production_unchanged': True})

    def component(j):
        jid = j['job_id']; cp_path = EXEC / 'checkpoints' / (jid + '.json')
        if cp_path.exists():
            cp = read(cp_path)
            if not cp['status'].startswith('CONTENT_'): return False
            raw_path = ROOT / cp['source_response']
            assert file_sha(raw_path) == cp['source_response_sha256']
            fresh = verify_response(j, cfg, schema, read(raw_path))
            for k, value in fresh.items(): assert cp[k] == value
            assert cp['job_sha256'] == digest(canonical(j))
            return True
        body = payload(j, cfg, prompts[j['task']], schema[j['task']])
        req = EXEC / 'raw_requests' / (jid + '.json.gz')
        assert file_sha(req) == manifest['request_sha256'][jid]
        assert gzip.decompress(req.read_bytes()) == canonical(body)
        intent = EXEC / 'request_intents' / (jid + '.json')
        if intent.exists(): raise ValueError('Unresolved prior intent; never silently resend')
        with guard:
            assert len(list((EXEC / 'request_intents').glob('*.json'))) < manifest['maximum_calls']
            immutable(intent, {'started_utc': now(), 'job_sha256': digest(canonical(j)),
                              'request_sha256': digest(canonical(body)), 'provider': 'miraiapi',
                              'requested_model': cfg['requested_model'], 'automatic_retries': 0})
            active.add(jid); peak[0] = max(peak[0], len(active)); events.append({'job_id': jid, 'event': 'start', 'utc': now()})
            assert len(active) <= 6; progress()
        transport.set_context(EXEC / 'stream_diagnostics', jid)
        try:
            raw = transport.call(cfg['operational']['base_url'], key, body)
            raw_path = EXEC / 'projected_responses' / (jid + '.json'); immutable(raw_path, raw)
            cp = verify_response(j, cfg, schema, raw)
            cp.update(job_sha256=digest(canonical(j)), source_response=raw_path.relative_to(ROOT).as_posix(),
                      source_response_sha256=file_sha(raw_path), reused_without_API_call=False)
            immutable(cp_path, cp)
            print(json.dumps({'job_id': jid, 'status': cp['status']}), flush=True)
            return True
        except Exception as error:
            immutable(cp_path, {'status': 'FAILED_PRESERVED_NO_AUTOMATIC_RETRY',
                'job_sha256': digest(canonical(j)), 'error_type': type(error).__name__,
                'error': str(error).replace(key, '[REDACTED]')[:2500], 'failed_utc': now()})
            print(json.dumps({'job_id': jid, 'status': 'FAILED', 'error_type': type(error).__name__}), flush=True)
            return False
        finally:
            with guard:
                active.remove(jid); events.append({'job_id': jid, 'event': 'finish', 'utc': now()}); progress()

    with ThreadPoolExecutor(max_workers=6) as requests:
        def business(bid):
            with guard:
                active_businesses.add(bid); peak[1] = max(peak[1], len(active_businesses))
                assert len(active_businesses) <= 2; progress()
            try:
                group = [j for j in jobs if j['business_id'] == bid]
                futures = [requests.submit(component, j) for j in group]
                results = [f.result() for f in futures]
                if not all(results): return False
                content = {j['condition']: read(EXEC / 'checkpoints' / (j['job_id'] + '.json'))['resolved_output'] for j in group}
                target = build_target(businesses[bid], content)
                target.update(status='MIRAI_GROUPED_CONTENT_VALID_PENDING_PROVENANCE_NOT_HUMAN_VERIFIED', production_accepted=False)
                immutable(EXEC / 'targets' / (bid + '.json'), target)
                with guard: complete.add(bid)
                return True
            finally:
                with guard: active_businesses.remove(bid); progress()

        for pi, pair in enumerate(cfg['operational']['phases']):
            phase[0] = 'recovery' if pi == 0 else 'six_live_request_verification'
            with ThreadPoolExecutor(max_workers=2) as slots:
                results = list(slots.map(business, pair))
            if not all(results): break  # Drain these two businesses; do not scale after failure.
    reports = []; failures = []; usage = Counter()
    for j in jobs:
        p = EXEC / 'checkpoints' / (j['job_id'] + '.json')
        if not p.exists(): continue
        cp = read(p)
        if not cp['status'].startswith('CONTENT_'):
            failures.append({'job_id': j['job_id'], 'checkpoint': cp}); continue
        raw = read(ROOT / cp['source_response']); verified = verify_response(j, cfg, schema, raw)
        for k, value in verified.items(): assert cp[k] == value
        source_directory = (ROOT / cp['source_response']).parent.parent
        replay(source_directory, j['job_id'], raw)
        reports.append({'job_id': j['job_id'], 'reused': cp['reused_without_API_call'],
                        'alias_fields_projected': len(cp['wire_field_projection']), 'reported_model': cp['reported_model']})
        if not cp['reused_without_API_call']:
            u = cp['usage'] or {}; usage['responses_with_usage'] += bool(u)
            usage['input_tokens'] += u.get('input_tokens', 0); usage['output_tokens'] += u.get('output_tokens', 0)
    for rel, sha in manifest['prior_artifact_sha256'].items(): assert file_sha(ROOT / rel) == sha, rel
    for p in RUN.rglob('*'):
        if p.is_file():
            data = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
            assert key.encode() not in data
    report = {'status': 'GROUPED_REPAIR_AND_SIX_REQUEST_CHECK_PASS' if len(complete) == 4 else 'GROUPED_RUN_STOPPED_WITH_FAILURES',
              'businesses_completed': sorted(complete), 'targets': 3*len(complete), 'component_checks': reports,
              'failures': failures, 'maximum_businesses_active': 2, 'maximum_concurrency': 6,
              'observed_peak_requests': peak[0], 'observed_peak_businesses': peak[1],
              'new_API_calls': len(list((EXEC / 'request_intents').glob('*.json'))),
              'reused_components': sum(r['reused'] for r in reports),
              'elapsed_seconds': round(time.monotonic()-started, 3), 'new_call_usage': dict(usage),
              'automatic_retries': 0, 'prior_artifacts_unchanged': True,
              'SSE_replay_and_source_validation': True, 'teacher_text_rewritten': False,
              'scientific_settings_changed': False, 'Codex_config_auth_key_accessed': False,
              'original_production_businesses': 131, 'production_integration': False,
              'missing_model_metadata_not_fabricated': True, 'human_factual_quality_certified': False}
    immutable(RUN / 'request_timeline.json', events)
    immutable(RUN / 'report.json', report)
    atomic(RUN / 'execution_progress.json', dict(report, current_jobs=[], active_businesses=[], updated_utc=now()))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); assert args.prepare != args.run
    try:
        prepare() if args.prepare else run()
    except Exception as error:
        atomic(RUN / 'runner_failure.json', {'error_type': type(error).__name__, 'error': str(error)[:1000], 'utc': now()})
        print(json.dumps({'runner_error': type(error).__name__}), flush=True)
        raise SystemExit(1)
