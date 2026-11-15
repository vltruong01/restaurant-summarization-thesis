"""D041 user-authorized bounded execution; monetary usage monitored on provider dashboard."""
import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from run_stage_g_extended_pilot_d041 import (
    check_inputs, build_target, usage_so_far, ROOT, FOLDER, CONFIG, read,
    file_sha, canonical, immutable, credentials, BASE_URL, PROVIDER, MODEL,
    job_result, payload, now,
)

AUTH = FOLDER / 'dashboard_execution_authorization.json'
PREFLIGHT = FOLDER / 'dashboard_execution_preflight.json'


def preflight():
    cfg, pre, jobs = check_inputs()
    auth = read(AUTH)
    expected = {
        'approved': True, 'approved_by': 'user', 'business_count': 100,
        'production_authorized': False, 'maximum_calls': pre['max_new_API_calls'],
        'cost_monitoring': 'researcher_Coletoken_dashboard',
        'sampling_manifest_sha256': file_sha(FOLDER / 'sampling_manifest.json'),
        'manual_review_25_manifest_sha256': file_sha(FOLDER / 'manual_review_25_manifest.json'),
        'config_sha256': file_sha(CONFIG), 'jobs_sha256': pre['jobs_sha256'],
    }
    for k, v in expected.items():
        if auth.get(k) != v:
            raise ValueError('Bounded dashboard authorization mismatch: ' + k)
    businesses = read(FOLDER / 'pilot_businesses.json')
    assert len(businesses) == 100
    assert sum(b['zero_filtered_edge_case'] for b in businesses) == 8
    assert len(jobs) == 300
    assert sum(not j.get('skip_api_empty_captions') for j in jobs) == 292
    assert cfg['requested_model'] == MODEL
    assert cfg['generation_proposal']['concurrency'] == 1
    report = {
        'status': 'OFFLINE_TECHNICAL_PASS_USER_AUTHORIZED_DASHBOARD_MONITORING',
        'authorization_sha256': file_sha(AUTH),
        'runner_sha256': file_sha(Path(__file__)),
        'original_preflight_sha256': file_sha(FOLDER / 'preflight.json'),
        'config_sha256': file_sha(CONFIG), 'jobs_sha256': pre['jobs_sha256'],
        'sampling_manifest_sha256': expected['sampling_manifest_sha256'],
        'manual_review_25_manifest_sha256': expected['manual_review_25_manifest_sha256'],
        'original_protected_files_verified': len(set(pre['input_sha256']) | set(pre['preserved_original_sha256'])),
        'businesses': 100, 'zero_filtered_businesses': 8, 'maximum_initial_API_calls': 292,
        'automatic_retries': 0, 'new_API_calls_during_preflight': 0,
        'credentials_accessed_during_preflight': False,
        'preparation_artifacts_sha256': {
            str(p.relative_to(FOLDER)): file_sha(p) for p in [
                FOLDER / 'chatgpt_stage_g_d041_preparation_bundle.zip',
                FOLDER / 'preparation_review_ui_v1/dist/index.html',
            ]
        },
        'methodology_changed': False, 'pricing_checkpoint_required': False,
    }
    immutable(PREFLIGHT, report)
    return cfg, pre, jobs, auth


def save_progress(out, jobs, completed, failures, current_job=None):
    summary = {
        'updated_utc': now(), 'current_job': current_job,
        'businesses_completed': len(completed), 'target_texts': 3 * len(completed),
        'API_request_intents': len(list((out / 'request_intents').glob('*.json'))),
        'accepted_tasks': sum(read(p).get('status') == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'
                              for p in (out / 'checkpoints').glob('*.json')),
        'tasks_planned': len(jobs), 'failed_tasks': len(failures),
        'usage': usage_so_far(out), 'human_verified': False,
        'target_content_origin': 'teacher_API_only_frozen_serialization',
    }
    path = out / 'execution_progress.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
    print(json.dumps(summary), flush=True)


def execute():
    cfg, pre, jobs, auth = preflight()  # No key/API until offline checks pass.
    out = FOLDER / 'execution_v1'
    if (out / 'completion.json').exists():
        print(json.dumps(read(out / 'completion.json'), indent=2)); return
    base, provider, key = credentials()
    assert base == BASE_URL and provider == PROVIDER and cfg['requested_model'] == MODEL
    manifest = {
        'scope': 'D041_100_BUSINESSES_ONLY_NOT_PRODUCTION',
        'config_sha256': file_sha(CONFIG), 'sampling_manifest_sha256': file_sha(FOLDER / 'sampling_manifest.json'),
        'authorization_sha256': file_sha(AUTH), 'runtime_preflight_sha256': file_sha(PREFLIGHT),
        'requested_model': MODEL, 'snapshot': cfg['snapshot'], 'provider': provider,
        'generation_settings': cfg['generation_proposal'], 'maximum_calls': 292,
        'automatic_retries': 0, 'Codex_configuration_or_authentication_access': False,
        'cost_monitoring': auth['cost_monitoring'], 'pricing_checkpoint_required': False,
    }
    immutable(out / 'run_manifest.json', manifest)
    immutable(out / 'dashboard_authorization_without_secrets.json', auth)
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    businesses = {b['business_id']: b for b in read(FOLDER / 'pilot_businesses.json')}
    started = time.monotonic(); failures = []; outputs = {}; completed = {}; stop_reason = None
    for job in jobs:
        path = out / 'checkpoints' / (job['job_id'] + '.json')
        intent = out / 'request_intents' / path.name
        if path.exists() and read(path)['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            failures.append({'job_id': job['job_id'], 'business_id': job['business_id'], 'failure': read(path)})
            continue
        if intent.exists() and not path.exists():
            failures.append({'job_id': job['job_id'], 'business_id': job['business_id'],
                             'failure': {'status': 'UNRESOLVED_REQUEST_NO_RESEND'}})
            stop_reason = 'Unresolved prior request; no resend'; break
        is_new = not path.exists() and not job.get('skip_api_empty_captions')
        if is_new:
            assert len(list((out / 'request_intents').glob('*.json'))) < 292
            request = canonical(payload(job, cfg, prompts[job['task']], schemas[job['task']]))
            destination = out / 'raw_requests' / (job['job_id'] + '.json.gz')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert gzip.decompress(destination.read_bytes()) == request
            else:
                destination.write_bytes(gzip.compress(request, mtime=0))
        save_progress(out, jobs, completed, failures, job['job_id'])
        try:
            cp = job_result(job, cfg, prompts[job['task']], schemas[job['task']], base, provider, key, out, CONFIG)
        except (RuntimeError, ValueError):
            failed = read(path) if path.exists() else {'status': 'UNRESOLVED_REQUEST_NO_RESEND'}
            failures.append({'job_id': job['job_id'], 'business_id': job['business_id'], 'failure': failed})
            raw_path = out / 'raw_responses' / path.name
            if failed.get('http_status') in [401, 403, 429] or not raw_path.exists():
                stop_reason = 'Account/rate/network failure; preserved without automatic retry'; break
            if read(raw_path).get('model') != MODEL:
                stop_reason = 'Provider model provenance mismatch'; break
            continue
        outputs.setdefault(job['business_id'], {})[job['condition']] = cp.get('resolved_output', cp['output'])
        result = build_target(businesses[job['business_id']], outputs[job['business_id']])
        if result is not None:
            immutable(out / 'provisional_targets' / (job['business_id'] + '.json'), result)
            completed[job['business_id']] = businesses[job['business_id']]
        save_progress(out, jobs, completed, failures)
    raw = [read(p) for p in (out / 'raw_responses').glob('*.json')]
    report = {
        'status': 'D041_STOPPED_FOR_REVIEW' if stop_reason else
                  ('D041_ATTEMPTED_WITH_FAILURES' if failures else 'D041_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE'),
        'businesses_planned': 100, 'businesses_completed': len(completed), 'target_texts': 3 * len(completed),
        'businesses_by_split': dict(Counter(b['split'] for b in completed.values())),
        'new_API_request_intents': len(list((out / 'request_intents').glob('*.json'))),
        'empty_tasks_planned': pre['empty_tasks'], 'usage': usage_so_far(out),
        'responses_without_usage': sum(not bool(r.get('usage')) for r in raw),
        'reported_model_counts': dict(Counter(r.get('model') for r in raw)),
        'cost_monitoring': auth['cost_monitoring'], 'provider_invoice_cost': None,
        'runtime_seconds_this_execution': round(time.monotonic() - started, 3),
        'failed_tasks': failures, 'stop_reason': stop_reason, 'automatic_retries': 0,
        'human_verified_businesses': 0, 'production_protocol_frozen': False,
        'held_out_teacher_targets': 0, 'remaining_2100_generation_started': False, 'LED_training_started': False,
    }
    save_progress(out, jobs, completed, failures)
    immutable(out / 'completion.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-pilot', action='store_true')
    args = parser.parse_args()
    if args.execute_pilot:
        execute()
    else:
        preflight()
        print('D041 offline technical preflight PASS; no API/key access.')
