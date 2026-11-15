"""Nine user-authorized Mirai requests; content diagnostics stay outside production."""
import argparse
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from stage_g_production_d042 import (
    ROOT, FOLDER, OUT, CONFIG as PRODUCTION_CONFIG, FIXED_KEYS,
    read, file_sha, immutable, atomic, canonical, check_inputs, now,
)
from prepare_stage_g_teacher_pilot import digest
from run_stage_g_production_d042 import runner_lock
from run_stage_g_teacher_pilot_v4_1_recovery import payload, response_text, validate_output
from stage_g_canonical_review_ids import resolve_review, citation_diagnostics
from run_stage_g_extended_pilot_d041 import build_target
from stage_g_teacher_credentials import key_file, KEY_VARIABLE
from run_stage_g_d042_concurrency10_new_key_trial_v2 import preserved_files, stats
import stage_g_mirai_stream_adapter_v2 as transport

CONFIG = ROOT / 'configs/stage_g_teacher_mirai_concurrency9_v1.json'
TRIAL = FOLDER / 'mirai_concurrency9_stream_trial_v1'
ATTEMPT = TRIAL / 'execution_v1'
PRIOR = FOLDER / 'mirai_concurrency10_trial_v1'


def inputs():
    parent, jobs, businesses, _ = check_inputs()
    cfg = read(CONFIG); provider = cfg['operational_provider_trial']
    assert all(cfg[k] == parent[k] for k in FIXED_KEYS)
    assert file_sha(PRODUCTION_CONFIG) == provider['parent_config_sha256']
    assert file_sha(ROOT / provider['transport_adapter']) == provider['transport_adapter_sha256']
    assert (provider['provider'], provider['base_url']) == ('miraiapi', 'https://api.miraiapi.com/v1')
    assert provider['maximum_concurrency'] == provider['maximum_calls'] == 9
    assert not cfg['full_generation_enabled'] and not provider['production_integration_approved']
    index = {j['job_id']: j for j in jobs}
    selected = [index[jid] for jid in read(PRIOR / 'manifest.json')['job_ids'][:9]]
    assert len(selected) == len({j['job_id'] for j in selected}) == 9
    assert all(j['split'] in ('train', 'validation') for j in selected)
    schema = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    return cfg, selected, businesses, schema, prompts


def preservation_check():
    for rel, sha in read(TRIAL / 'prior_artifact_sha256.json').items():
        assert file_sha(ROOT / rel) == sha, rel


def prepare():
    lock = runner_lock()
    assert (FOLDER / 'STOP_REQUESTED.json').exists()
    assert not (TRIAL / 'manifest.json').exists()
    cfg, jobs, businesses, schema, prompts = inputs()
    prior = preserved_files()
    for name in ('execution_control_concurrency10_new_key_trial_v2', 'mirai_concurrency10_trial_v1'):
        prior.update({p.relative_to(ROOT).as_posix(): file_sha(p)
                      for p in (FOLDER / name).rglob('*') if p.is_file()})
    immutable(TRIAL / 'prior_artifact_sha256.json', prior)
    hashes = {}
    for j in jobs:
        body = canonical(payload(j, cfg, prompts[j['task']], schema[j['task']]))
        p = ATTEMPT / 'raw_requests' / (j['job_id'] + '.json.gz')
        assert gzip.decompress((PRIOR / 'execution_v1/raw_requests' / p.name).read_bytes()) == body
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f: f.write(gzip.compress(body, mtime=0))
        hashes[j['job_id']] = file_sha(p)
    immutable(TRIAL / 'manifest.json', {
        'approved_by': 'user', 'authorization': 'User explicitly requested nine concurrent teacher requests after the Mirai compatibility report',
        'maximum_paid_calls': 9, 'maximum_concurrency': 9, 'automatic_retries': 0,
        'job_ids': [j['job_id'] for j in jobs],
        'selection': 'first9_prior_manifest_jobs_in_frozen_order_including_previous_timeout_not_selected_by_outcome',
        'config_sha256': file_sha(CONFIG), 'runner_sha256': file_sha(Path(__file__)),
        'adapter_sha256': file_sha(ROOT / cfg['operational_provider_trial']['transport_adapter']),
        'request_file_sha256': hashes, 'request_payloads_identical': True,
        'provenance_policy': 'Missing returned model remains pending; content validity separately audited without model fabrication',
        'production_integration_approved': False, 'Codex_config_or_auth_access': False,
        'offline_transport_checks': transport.offline_self_check(), 'preflight_API_calls': 0,
    })
    print(json.dumps({'preflight': 'PASS', 'concurrency': 9, 'requests': 9, 'API_calls': 0}), flush=True)


def run():
    lock = runner_lock()
    assert (FOLDER / 'STOP_REQUESTED.json').exists()
    assert not (TRIAL / 'execution_started.json').exists(), 'No automatic rerun'
    manifest_path = TRIAL / 'execution_manifest_v2.json'
    if not manifest_path.exists(): manifest_path = TRIAL / 'manifest.json'
    m = read(manifest_path)
    assert m['runner_sha256'] == file_sha(Path(__file__))
    cfg, jobs, businesses, schema, prompts = inputs()
    assert file_sha(CONFIG) == m['config_sha256']
    preservation_check()
    for j in jobs:
        p = ATTEMPT / 'raw_requests' / (j['job_id'] + '.json.gz')
        assert file_sha(p) == m['request_file_sha256'][j['job_id']]
        assert gzip.decompress(p.read_bytes()) == canonical(payload(j, cfg, prompts[j['task']], schema[j['task']]))
        assert not (ATTEMPT / 'request_intents' / (j['job_id'] + '.json')).exists()
    assert not os.environ.get(KEY_VARIABLE)
    provider = cfg['operational_provider_trial']
    key = key_file().with_name(provider['dedicated_secret_filename']).read_text(encoding='utf-8-sig').strip()
    assert key and not any(c.isspace() for c in key)
    immutable(TRIAL / 'execution_started.json', {'started_utc': now(), 'maximum_paid_calls': 9,
              'manifest_sha256': file_sha(manifest_path), 'automatic_retries': 0})
    active = m['job_ids'][:]

    def progress():
        atomic(TRIAL / 'execution_progress.json', {'status': 'RUNNING_ISOLATED_TRIAL' if active else 'DRAINED_OFFLINE_AUDIT',
               'current_jobs': active[:], 'maximum_concurrency': 9, 'updated_utc': now(),
               'pid': os.getpid(), 'production_unchanged': True, 'provider': provider['provider']})

    def worker(j):
        jid = j['job_id']; transport.set_context(ATTEMPT / 'stream_diagnostics', jid)
        binding = {'job_sha256': digest(canonical(j)), 'config_sha256': file_sha(CONFIG),
                   'prompt_sha256': j['prompt_sha256'], 'schema_sha256': j['schema_sha256'],
                   'base_url': provider['base_url'], 'provider': provider['provider'],
                   'requested_model': cfg['requested_model']}
        body = payload(j, cfg, prompts[j['task']], schema[j['task']])
        immutable(ATTEMPT / 'request_intents' / (jid + '.json'), {
            'status': 'REQUEST_STARTED', 'started_utc': now(), 'binding': binding,
            'request_sha256': digest(canonical(body)), 'request_images': j['images']})
        start = time.monotonic()
        try:
            raw = transport.call(provider['base_url'], key, body)
            projected_path = ATTEMPT / 'projected_responses' / (jid + '.json')
            immutable(projected_path, raw)
            assert raw['status'] == 'completed'
            output = json.loads(response_text(raw))
            validation = validate_output(j, output, schema[j['task']])
            resolved, evidence = resolve_review(j, output) if j['task'] == 'review_consensus' else (output, [])
            reported_model = raw.get('model')
            provenance = ('MODEL_ALIAS_MATCH' if reported_model == cfg['requested_model']
                          else 'MODEL_METADATA_MISSING_PENDING' if not reported_model
                          else 'MODEL_ALIAS_MISMATCH_REJECTED')
            cp = {'status': 'ISOLATED_CONTENT_PASS_PENDING_MODEL_PROVENANCE' if provenance != 'MODEL_ALIAS_MATCH'
                            else 'ISOLATED_CONTENT_AND_MODEL_ALIAS_PASS_NOT_PRODUCTION_APPROVED',
                  'binding': binding, 'output': output, 'resolved_output': resolved,
                  'evidence_resolution': evidence, 'validation': validation,
                  'citation_metadata_diagnostics': citation_diagnostics(j, output) if j['task'] == 'review_consensus' else [],
                  'reported_model': reported_model, 'model_provenance': provenance,
                  'projected_response_sha256': file_sha(projected_path),
                  'response_id': raw.get('id'), 'usage': raw.get('usage'),
                  'seconds': round(time.monotonic() - start, 3), 'completed_utc': now(),
                  'production_accepted': False, 'human_verified': False,
                  'target_content_origin': 'teacher_API_only_exact_stream_projection'}
            if provenance == 'MODEL_ALIAS_MISMATCH_REJECTED':
                cp['status'] = 'ISOLATED_CONTENT_PASS_MODEL_ALIAS_MISMATCH_REJECTED'
            immutable(ATTEMPT / 'checkpoints' / (jid + '.json'), cp)
            return {'job_id': jid, 'content_pass': True, 'model_provenance': provenance,
                    'seconds': cp['seconds'], 'response_id': raw.get('id')}
        except Exception as error:
            # Source-validation errors contain no authorization material; redact defensively.
            immutable(ATTEMPT / 'checkpoints' / (jid + '.json'), {
                'status': 'FAILED_NO_AUTOMATIC_RETRY', 'binding': binding,
                'error_type': type(error).__name__, 'failed_utc': now(),
                'validation_error': str(error).replace(key, '[REDACTED]')[:2500] if isinstance(error, ValueError) else None})
            return {'job_id': jid, 'content_pass': False, 'error_type': type(error).__name__}

    progress(); start = time.monotonic(); results = []
    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(worker, j): j for j in jobs}
        for future in as_completed(futures):
            r = future.result(); results.append(r); active.remove(r['job_id'])
            progress(); print(json.dumps(r), flush=True)
    elapsed = time.monotonic() - start
    events = []; timings = defaultdict(list); usage = Counter(); model_states = Counter(); http = Counter(); errors = []
    valid = {r['job_id'] for r in results if r['content_pass']}
    for j in jobs:
        jid = j['job_id']; d = read(ATTEMPT / 'stream_diagnostics' / (jid + '.json'))
        timings[j['task']].append(d['elapsed_seconds']); http[str(d.get('http_status'))] += 1
        events.extend([(datetime.fromisoformat(d['started_utc']).timestamp(), 1),
                       (datetime.fromisoformat(d['finished_utc']).timestamp(), -1)])
        cp = read(ATTEMPT / 'checkpoints' / (jid + '.json'))
        projected_path = ATTEMPT / 'projected_responses' / (jid + '.json')
        if projected_path.exists():
            raw = read(projected_path); u = raw.get('usage') or {}
            usage['responses_with_usage'] += bool(u)
            usage['input_tokens'] += u.get('input_tokens', 0); usage['output_tokens'] += u.get('output_tokens', 0)
            usage['reasoning_tokens'] += (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        if jid in valid:
            raw = read(projected_path)
            assert file_sha(projected_path) == cp['projected_response_sha256']
            assert json.loads(response_text(raw)) == cp['output']
            assert validate_output(j, cp['output'], schema[j['task']]) == cp['validation']
            # Replay preserved SSE to prove the projection came exclusively from API events.
            projector = transport.StreamProjection(); wire = None
            with gzip.open(ATTEMPT / 'stream_diagnostics' / (jid + '.sse.jsonl.gz'), 'rt', encoding='utf-8') as stream:
                for line in stream:
                    if line.strip() == '[DONE]': continue
                    event = json.loads(line); projector.consume(event)
                    if event.get('type') == 'response.completed': wire = event['response']
            assert wire == read(ATTEMPT / 'stream_diagnostics' / (jid + '.completed_wire.json'))
            assert projector.project(wire)[0] == raw
            model_states[cp['model_provenance']] += 1
        else:
            errors.append({'job_id': jid, 'checkpoint': cp, 'transport_failure': d.get('failure_kind'),
                           'request_seconds': d['elapsed_seconds']})
    current = peak = 0
    for _, delta in sorted(events, key=lambda x: (x[0], x[1])):
        current += delta; peak = max(peak, current)
    grouped = defaultdict(list)
    for j in jobs: grouped[j['business_id']].append(j)
    complete_trial_ids = []
    for b in businesses:
        items = grouped[b['business_id']]
        if len(items) != 3 or not all(j['job_id'] in valid for j in items): continue
        content = {}
        for j in items:
            cp = read(ATTEMPT / 'checkpoints' / (j['job_id'] + '.json'))
            if cp['model_provenance'] == 'MODEL_ALIAS_MISMATCH_REJECTED': break
            content[j['condition']] = cp['resolved_output']
        if len(content) != 3: continue
        target = build_target(b, content)
        target['status'] = 'ISOLATED_MIRAI_TRIAL_TARGET_PENDING_MODEL_PROVENANCE_AND_RESEARCHER_REVIEW'
        target['production_accepted'] = False
        immutable(ATTEMPT / 'trial_targets' / (b['business_id'] + '.json'), target)
        complete_trial_ids.append(b['business_id'])
    preservation_check()
    for p in TRIAL.rglob('*'):
        if p.is_file() and p.suffix != '.gz': assert key.encode() not in p.read_bytes()
    report = {'status': 'NINE_REQUEST_CONTENT_AUDIT_PASS_PROVENANCE_PENDING' if len(valid) == 9 else 'NINE_REQUEST_TRIAL_DRAINED_WITH_ERRORS',
              'provider': 'miraiapi', 'requested_model': cfg['requested_model'],
              'actual_paid_request_intents': len(list((ATTEMPT / 'request_intents').glob('*.json'))),
              'content_schema_source_pass': len(valid), 'content_or_transport_fail': 9 - len(valid),
              'model_provenance_counts': dict(model_states), 'production_accepted_targets': 0,
              'batch_wall_seconds': round(elapsed, 3), 'observed_client_overlap_peak': peak,
              'seconds_by_task': {k: stats(v) for k, v in timings.items()}, 'http_statuses': dict(http),
              'usage_returned': dict(usage), 'unknown_usage_not_assumed_zero': True,
              'errors': errors, 'results': results, 'automatic_retries': 0,
              'SSE_projection_replay_checked_for_every_content_pass': True,
              'previous_artifacts_preserved': True, 'request_payloads_unchanged': True,
              'requested_reasoning': 'medium', 'requested_service_tier': None,
              'missing_response_metadata_not_fabricated': True,
              'isolated_complete_trial_business_ids': complete_trial_ids,
              'production_businesses_unchanged': 131, 'production_integration': False,
              'human_factual_quality_certified': False, 'teacher_API_only_target_content': True,
              'Codex_configuration_authentication_or_key_modified': False,
              'continuous_nine_request_production_started': False}
    immutable(TRIAL / 'report.json', report)
    atomic(TRIAL / 'execution_progress.json', {'status': report['status'], 'current_jobs': [],
           'maximum_concurrency': 9, 'content_pass': len(valid), 'failed': 9-len(valid),
           'updated_utc': now(), 'production_unchanged': True})
    note = (f"**D-042 Mirai nine-request stream trial:** {len(valid)}/9 schema/source content PASS, "
            f"wall={elapsed:.3f}s, client overlap peak={peak}; model provenance={dict(model_states)}. "
            "Original SSE, completed wire envelopes, exact-text projected responses and failures saved separately; all content-pass projections replay-checked offline. "
            "No model metadata fabrication or production-validator relaxation; isolated targets are pending provenance/researcher review, not human factual truth. "
            "Same D040 prompt/schema/reviews/captions/photos, requested medium reasoning and omitted service tier. "
            "Original production remains131/2200 businesses,393 targets; no Codex configuration/authentication/key changes, held-out teacher targets or LED training. "
            "Report: outputs/stage_g/teacher_production_d042_v1/mirai_concurrency9_stream_trial_v1/report.json.\n")
    for rel in ('thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md'):
        p = ROOT / rel; first, rest = p.read_text(encoding='utf-8').split('\n', 1)
        p.write_text(first + '\n\n' + note + '\n' + rest, encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    assert args.prepare != args.run
    try:
        prepare() if args.prepare else run()
    except Exception as error:
        atomic(TRIAL / 'runner_failure.json', {'status': 'STOPPED_REQUIRES_REVIEW', 'error_type': type(error).__name__,
               'updated_utc': now(), 'automatic_retries': 0})
        print(json.dumps({'runner_error_type': type(error).__name__, 'automatic_retries': 0}), flush=True)
        raise SystemExit(1)
