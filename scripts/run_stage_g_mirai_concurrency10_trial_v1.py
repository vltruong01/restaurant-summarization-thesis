"""User-authorized Mirai teacher-only compatibility trial, isolated from production."""
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
    ROOT, CONFIG as PRODUCTION_CONFIG, FOLDER, OUT, FIXED_KEYS, read, file_sha,
    canonical, immutable, atomic, check_inputs, now,
)
from prepare_stage_g_teacher_pilot import digest
from run_stage_g_production_d042 import runner_lock
from run_stage_g_teacher_pilot_v4_1_recovery import (
    job_result, payload, validate_output, response_text,
)
from stage_g_canonical_review_ids import resolve_review
from run_stage_g_extended_pilot_d041 import build_target
from stage_g_teacher_credentials import key_file, KEY_VARIABLE
from run_stage_g_d042_concurrency10_new_key_trial_v2 import stats, preserved_files

CONFIG = ROOT / 'configs/stage_g_teacher_mirai_trial_v1.json'
TRIAL = FOLDER / 'mirai_concurrency10_trial_v1'
ATTEMPT = TRIAL / 'execution_v1'
PRIOR = FOLDER / 'execution_control_concurrency10_new_key_trial_v2'
PASS = 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'


def preservation_check():
    for relative, sha in read(TRIAL / 'prior_artifact_sha256.json').items():
        assert file_sha(ROOT / relative) == sha, relative


def setup_inputs():
    parent, jobs, businesses, imports = check_inputs()
    cfg = read(CONFIG)
    assert all(cfg[k] == parent[k] for k in FIXED_KEYS)
    provider = cfg['operational_provider_trial']
    assert provider['parent_config_sha256'] == file_sha(PRODUCTION_CONFIG)
    assert (provider['provider'], provider['base_url'], provider['wire_api']) == (
        'miraiapi', 'https://api.miraiapi.com/v1', 'responses')
    assert not cfg['full_generation_enabled'] and provider['maximum_calls'] == 10
    assert not provider['production_integration_approved']
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    index = {j['job_id']: j for j in jobs}
    selected = [index[jid] for jid in read(PRIOR / 'manifest.json')['job_ids']]
    assert len(selected) == len({j['job_id'] for j in selected}) == 10
    assert all(j['split'] in ('train', 'validation') and not j.get('skip_api_empty_captions') for j in selected)
    return cfg, selected, businesses, schemas, prompts


def prepare():
    lock = runner_lock()
    assert (FOLDER / 'STOP_REQUESTED.json').exists()
    assert not (TRIAL / 'manifest.json').exists()
    cfg, selected, businesses, schemas, prompts = setup_inputs()
    previous = preserved_files()
    previous.update({p.relative_to(ROOT).as_posix(): file_sha(p)
                     for p in PRIOR.rglob('*') if p.is_file()})
    immutable(TRIAL / 'prior_artifact_sha256.json', previous)
    hashes = {}
    for j in selected:
        body = canonical(payload(j, cfg, prompts[j['task']], schemas[j['task']]))
        original = PRIOR / 'execution_v1/raw_requests' / (j['job_id'] + '.json.gz')
        assert gzip.decompress(original.read_bytes()) == body
        path = ATTEMPT / 'raw_requests' / original.name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(gzip.compress(body, mtime=0))
        hashes[j['job_id']] = file_sha(path)
    immutable(TRIAL / 'manifest.json', {
        'approved_by': 'user',
        'authorization': 'User requested new teacher key and ten concurrent requests, then supplied Mirai endpoint/model',
        'job_ids': [j['job_id'] for j in selected], 'maximum_paid_calls': 10,
        'maximum_concurrency': 10, 'automatic_retries': 0,
        'base_url': cfg['operational_provider_trial']['base_url'], 'provider': 'miraiapi',
        'config_sha256': file_sha(CONFIG), 'runner_sha256': file_sha(Path(__file__)),
        'request_file_sha256': hashes, 'request_bodies_identical_to_prior_trial': True,
        'provider_changed': True, 'same_alias_does_not_attest_backend_equivalence': True,
        'scientific_prompt_schema_settings_changed': False,
        'production_targets_or_provider_changed': False,
        'production_integration_approved': False, 'Codex_config_or_auth_access': False,
        'secret_or_fingerprint_recorded': False, 'network_calls_in_preflight': 0,
    })
    print(json.dumps({'offline_preflight': 'PASS', 'requests': 10,
                      'payloads_identical': True, 'production_integration': False}), flush=True)


def verify_trial_checkpoint(job, cfg, schema, prompt):
    cp = read(ATTEMPT / 'checkpoints' / (job['job_id'] + '.json'))
    assert cp['status'] == PASS
    provider = cfg['operational_provider_trial']
    expected = {'job_sha256': digest(canonical(job)), 'config_sha256': file_sha(CONFIG),
                'prompt_sha256': digest(prompt.encode()), 'schema_sha256': digest(canonical(schema)),
                'base_url': provider['base_url'], 'provider': provider['provider'],
                'requested_model': cfg['requested_model']}
    assert cp['binding'] == expected
    assert cp['validation'] == validate_output(job, cp['output'], schema)
    rawp = ATTEMPT / 'raw_responses' / (job['job_id'] + '.json')
    assert file_sha(rawp) == cp['raw_response_sha256']
    raw = read(rawp)
    assert raw['status'] == 'completed' and raw['model'] == cfg['requested_model']
    assert json.loads(response_text(raw)) == cp['output']
    body = canonical(payload(job, cfg, prompt, schema))
    assert gzip.decompress((ATTEMPT / 'raw_requests' / (job['job_id'] + '.json.gz')).read_bytes()) == body
    assert read(ATTEMPT / 'request_intents' / (job['job_id'] + '.json'))['request_sha256'] == digest(body)
    if job['task'] == 'review_consensus':
        resolved, evidence = resolve_review(job, cp['output'])
        assert resolved == cp['resolved_output'] and evidence == cp['evidence_resolution']
    return cp


def run():
    lock = runner_lock()
    assert (FOLDER / 'STOP_REQUESTED.json').exists()
    assert not (TRIAL / 'execution_started.json').exists(), 'No automatic repeat'
    manifest = read(TRIAL / 'manifest.json')
    assert manifest['runner_sha256'] == file_sha(Path(__file__))
    cfg, selected, businesses, schemas, prompts = setup_inputs()
    assert file_sha(CONFIG) == manifest['config_sha256']
    preservation_check()
    for j in selected:
        p = ATTEMPT / 'raw_requests' / (j['job_id'] + '.json.gz')
        assert file_sha(p) == manifest['request_file_sha256'][j['job_id']]
        assert gzip.decompress(p.read_bytes()) == canonical(payload(j, cfg, prompts[j['task']], schemas[j['task']]))
        assert not (ATTEMPT / 'request_intents' / (j['job_id'] + '.json')).exists()
    assert not os.environ.get(KEY_VARIABLE), 'Do not accidentally use a different environment credential'
    secret_path = key_file().with_name(cfg['operational_provider_trial']['dedicated_secret_filename'])
    key = secret_path.read_text(encoding='utf-8-sig').strip()
    assert key and not any(c.isspace() for c in key)
    provider = cfg['operational_provider_trial']
    import run_stage_g_teacher_pilot_v4_1_recovery as legacy
    import stage_g_teacher_stream_diagnostics_v1 as diagnostic
    legacy.call = diagnostic.call
    immutable(TRIAL / 'execution_started.json', {'started_utc': now(),
              'maximum_paid_calls': 10, 'manifest_sha256': file_sha(TRIAL / 'manifest.json')})
    active = manifest['job_ids'][:]

    def status():
        atomic(TRIAL / 'execution_progress.json', {'status': 'RUNNING_TRIAL_ONLY' if active else 'REQUESTS_DRAINED',
               'current_jobs': active[:], 'maximum_concurrency': 10, 'pid': os.getpid(),
               'provider': 'miraiapi', 'updated_utc': now(), 'production_unchanged': True})

    def worker(j):
        diagnostic.set_context(ATTEMPT / 'stream_diagnostics', j['job_id'])
        try:
            cp = job_result(j, cfg, prompts[j['task']], schemas[j['task']], provider['base_url'],
                            provider['provider'], key, ATTEMPT, CONFIG)
            return {'job_id': j['job_id'], 'passed': True, 'seconds': cp.get('seconds'),
                    'response_id': cp.get('response_id')}
        except Exception as error:
            return {'job_id': j['job_id'], 'passed': False, 'error_type': type(error).__name__}

    status(); start = time.monotonic(); results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(worker, j): j for j in selected}
        for future in as_completed(futures):
            result = future.result(); results.append(result); active.remove(result['job_id'])
            status(); print(json.dumps(result), flush=True)
    elapsed = time.monotonic() - start
    successful = {r['job_id'] for r in results if r['passed']}
    for j in selected:
        if j['job_id'] in successful:
            verify_trial_checkpoint(j, cfg, schemas[j['task']], prompts[j['task']])
    preservation_check()

    diagnostics = []; errors = []; events = []; usage = Counter(); models = Counter()
    tiers = Counter(); efforts = Counter(); types = defaultdict(list)
    for j in selected:
        jid = j['job_id']; dp = ATTEMPT / 'stream_diagnostics' / (jid + '.json')
        d = read(dp); diagnostics.append(d); types[j['task']].append(d['elapsed_seconds'])
        events.extend([(datetime.fromisoformat(d['started_utc']).timestamp(), 1),
                       (datetime.fromisoformat(d['finished_utc']).timestamp(), -1)])
        rp = ATTEMPT / 'raw_responses' / (jid + '.json')
        if rp.exists():
            raw = read(rp); models[str(raw.get('model'))] += 1; tiers[str(raw.get('service_tier'))] += 1
            efforts[str((raw.get('reasoning') or {}).get('effort'))] += 1
            u = raw.get('usage') or {}; usage['responses_with_usage'] += bool(u)
            usage['input_tokens'] += u.get('input_tokens', 0); usage['output_tokens'] += u.get('output_tokens', 0)
            usage['reasoning_tokens'] += (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        if jid not in successful:
            cp = read(ATTEMPT / 'checkpoints' / (jid + '.json'))
            errors.append({'job_id': jid, 'http_status': d.get('http_status'),
                          'failure_kind': d.get('failure_kind'), 'provider_error': d.get('provider_error'),
                          'http_error_body_redacted': d.get('provider_http_error_body_redacted'),
                          'local_validation_error': cp.get('local_validation_error')})
    running = peak = 0
    for _, delta in sorted(events, key=lambda x: (x[0], x[1])):
        running += delta; peak = max(peak, running)
    # Only complete targets made entirely from this provider's trial responses.
    grouped = defaultdict(list)
    for j in selected: grouped[j['business_id']].append(j)
    target_ids = []
    for business in businesses:
        items = grouped[business['business_id']]
        if len(items) != 3 or not all(j['job_id'] in successful for j in items): continue
        content = {}
        for j in items:
            cp = read(ATTEMPT / 'checkpoints' / (j['job_id'] + '.json'))
            content[j['condition']] = cp.get('resolved_output', cp['output'])
        target = build_target(business, content)
        immutable(ATTEMPT / 'trial_targets' / (business['business_id'] + '.json'), target)
        target_ids.append(business['business_id'])
    report = {
        'status': 'MIRAI_TEN_REQUEST_TECHNICAL_PASS' if len(successful) == 10 else 'MIRAI_TRIAL_STOPPED_WITH_FAILURES',
        'provider': 'miraiapi', 'base_url': provider['base_url'], 'requested_model': cfg['requested_model'],
        'paid_request_intents': len(list((ATTEMPT / 'request_intents').glob('*.json'))),
        'passed': len(successful), 'failed': 10 - len(successful), 'automatic_retries': 0,
        'batch_wall_seconds': round(elapsed, 3), 'observed_client_transport_overlap_peak': peak,
        'successful_requests_per_minute': 60 * len(successful) / elapsed,
        'request_seconds': stats([d['elapsed_seconds'] for d in diagnostics]),
        'seconds_by_task': {k: stats(v) for k, v in types.items()},
        'http_statuses': dict(Counter(str(d.get('http_status')) for d in diagnostics)),
        'returned_model_aliases': dict(models), 'returned_service_tiers': dict(tiers),
        'returned_reasoning_efforts': dict(efforts), 'usage_returned': dict(usage),
        'usage_missing_for_failed_requests_not_assumed_zero': True,
        'errors': errors, 'results': results,
        'all_successes_offline_schema_evidence_payload_validated': True,
        'preserved_previous_artifacts': True, 'request_payloads_unchanged': True,
        'teacher_API_only_target_content': True, 'human_factual_quality_certified': False,
        'same_alias_does_not_certify_backend_equivalence': True,
        'Coletoken_production_businesses_unchanged': read(OUT / 'execution_progress.json')['businesses_completed'],
        'isolated_complete_trial_business_ids': target_ids,
        'production_data_integration': False, 'continuous_ten_request_schedule_adopted': False,
        'frozen_production_config_unchanged': True, 'Codex_config_or_auth_access': False,
        'comparison_limitation': 'One bounded provider trial; not evidence of sustained reliability or backend equivalence.',
        'report_for_researcher_review': True,
    }
    # Verify the current secret was not persisted in requests, diagnostics or outputs.
    for p in TRIAL.rglob('*'):
        if p.is_file(): assert key.encode() not in p.read_bytes()
    immutable(TRIAL / 'report.json', report)
    preservation_check()
    atomic(TRIAL / 'execution_progress.json', {'status': report['status'], 'current_jobs': [],
           'provider': 'miraiapi', 'updated_utc': now(), 'production_unchanged': True,
           'passed': len(successful), 'failed': 10 - len(successful), 'automatic_retries': 0})
    note = (f"**D-042 isolated Mirai teacher compatibility trial:** {len(successful)}/10 technical PASS; "
            f"wall={elapsed:.3f}s; client transport overlap peak={peak}. User explicitly supplied Mirai endpoint/new teacher key. "
            "D040 prompts/schema/model alias/medium reasoning/request bodies unchanged, but provider provenance changed; backend equivalence is not attested. "
            "Mirai responses remain in a separate trial namespace and are not imported into Coletoken production. "
            "Frozen production still131/2200 businesses and remains paused; no Codex configuration/authentication access, no manual target rewriting, no test targets or LED training. "
            "Report: outputs/stage_g/teacher_production_d042_v1/mirai_concurrency10_trial_v1/report.json.\n")
    for rel in ('thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md'):
        p = ROOT / rel; text = p.read_text(encoding='utf-8'); first, rest = text.split('\n', 1)
        p.write_text(first + '\n\n' + note + '\n' + rest, encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    assert args.prepare != args.run
    try:
        prepare() if args.prepare else run()
    except Exception as error:
        atomic(TRIAL / 'runner_failure.json', {'status': 'STOPPED_REQUIRES_REVIEW',
               'error_type': type(error).__name__, 'automatic_retries': 0, 'updated_utc': now()})
        print(json.dumps({'runner_error_type': type(error).__name__, 'automatic_retries': 0}), flush=True)
        raise SystemExit(1)
