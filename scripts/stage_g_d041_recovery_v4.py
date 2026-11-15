"""Versioned, user-approved recovery of one HTTP520 failure plus unstarted D041 work."""
import argparse
import gzip
import json
import shutil
from pathlib import Path
from stage_g_d041_recovery_v3 import check_inputs as previous_check
from stage_g_d041_recovery_v3 import FOLDER as PREVIOUS
from run_stage_g_d041_continuation_v2 import (
    ROOT, ORIGINAL, CONFIG, read, file_sha, canonical, digest, immutable,
    credentials, BASE_URL, PROVIDER, MODEL, payload, job_result, now, build_target, usage_so_far,
)

FOLDER = ORIGINAL / 'recovery_v4'
OUT = FOLDER / 'execution_v1'
HISTORICAL_FAILED_ATTEMPTS = 3


def copy_exact(source, destination, hashes):
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    shutil.copy2(source, destination)
    assert file_sha(source) == file_sha(destination)
    hashes[destination.relative_to(FOLDER).as_posix()] = file_sha(destination)


def check_inputs(verify_checkpoints=False):
    cfg, pre, jobs = previous_check(verify_checkpoints=True)
    manifest = read(FOLDER / 'continuation_manifest.json')
    assert file_sha(CONFIG) == manifest['config_sha256']
    for rel, expected in manifest['previous_namespace_preserved_sha256'].items():
        assert file_sha(ROOT / rel) == expected, 'Previous namespace changed: ' + rel
    for rel, expected in manifest['copied_files_sha256'].items():
        assert file_sha(FOLDER / rel) == expected, 'Imported checkpoint/source changed: ' + rel
    for rel, expected in manifest['runtime_sha256'].items():
        assert file_sha(ROOT / rel) == expected, 'Recovery runtime changed: ' + rel
    assert read(FOLDER / 'pilot_jobs.json') == jobs
    if verify_checkpoints:
        schemas = read(ROOT / cfg['schema'])
        prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
        for j in jobs:
            cp = OUT / 'checkpoints' / (j['job_id'] + '.json')
            intent = OUT / 'request_intents' / cp.name
            if intent.exists():
                assert cp.exists(), 'Unresolved attempt; no automatic resend'
                body = canonical(payload(j, cfg, prompts[j['task']], schemas[j['task']]))
                assert gzip.decompress((OUT / 'raw_requests' / (j['job_id'] + '.json.gz')).read_bytes()) == body
                assert read(intent)['request_sha256'] == digest(body)
            if cp.exists() and read(cp)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                job_result(j, cfg, prompts[j['task']], schemas[j['task']], BASE_URL, PROVIDER, None, OUT, CONFIG)
    return cfg, pre, jobs


def saved_state(jobs):
    outputs = {}; failures = []; complete = {}
    for j in jobs:
        path = OUT / 'checkpoints' / (j['job_id'] + '.json')
        if not path.exists():
            continue
        cp = read(path)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            failures.append({'job_id': j['job_id'], 'business_id': j['business_id'], 'failure': cp})
            continue
        outputs.setdefault(j['business_id'], {})[j['condition']] = cp.get('resolved_output', cp['output'])
    for b in read(FOLDER / 'pilot_businesses.json'):
        result = build_target(b, outputs.get(b['business_id'], {}))
        if result is not None:
            immutable(OUT / 'provisional_targets' / (b['business_id'] + '.json'), result)
            complete[b['business_id']] = b
    return outputs, failures, complete


def adapters():
    audit = (ROOT / 'scripts/audit_stage_g_d041_continuation_v2.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_stage_g_d041_continuation_v2 import', 'from stage_g_d041_recovery_v4 import')
    (ROOT / 'scripts/audit_stage_g_d041_recovery_v4.py').write_text(audit, encoding='utf-8')
    exporter = (ROOT / 'scripts/export_stage_g_d041_continuation_v2_review.py').read_text(encoding='utf-8')
    exporter = exporter.replace('from run_stage_g_d041_continuation_v2 import', 'from stage_g_d041_recovery_v4 import')
    exporter = exporter.replace('chatgpt_stage_g_d041_random25_review_continuation_v2', 'chatgpt_stage_g_d041_random25_review_recovery_v4')
    anchor = "    for name,data in files.items():assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}',data),'Possible secret in export: '+name"
    assert exporter.count(anchor) == 1
    exporter = exporter.replace(anchor, "    from stage_g_review_export_secret_guard import check_export_bytes\n    for name,data in files.items():check_export_bytes(data,name)")
    anchor = "    files['additional_flagged_cases.json']=original(FOLDER/'technical_flags.json')"
    extra = "\n    for rel in ['recovery_authorization.json','recovery_attempts.json','execution_control_v1/manifest.json','execution_control_v1/report.json','execution_control_v1/batch_results.jsonl']:\n        files['reports/'+rel]=original(FOLDER/rel)\n    for p in (FOLDER/'previous_failed_attempts').rglob('*'):\n        if p.is_file():files['previous_failed_attempts/'+p.relative_to(FOLDER/'previous_failed_attempts').as_posix()]=original(p)\n    for p in (FOLDER/'execution_v1/stream_diagnostics').glob('*.json'):\n        files['stream_diagnostics/'+p.name]=original(p)\n    for runtime in ['run_stage_g_d041_recovery_v4.py','stage_g_d041_recovery_v4.py','stage_g_teacher_stream_diagnostics_v1.py','stage_g_review_export_secret_guard.py']:\n        files['protocol/runtime/'+runtime]=original(ROOT/'scripts'/runtime)"
    assert exporter.count(anchor) == 1
    exporter = exporter.replace(anchor, anchor + extra)
    (ROOT / 'scripts/export_stage_g_d041_recovery_v4_review.py').write_text(exporter, encoding='utf-8')


def prepare():
    cfg, pre, jobs = previous_check(verify_checkpoints=True)
    assert read(PREVIOUS / 'execution_v1/completion.json')['status'] == 'D041_STOPPED_FOR_REVIEW'
    assert not FOLDER.exists(), 'Recovery already prepared; use existing version'
    failures = [j['job_id'] for j in jobs if (PREVIOUS / 'execution_v1/checkpoints' / (j['job_id'] + '.json')).exists()
                and read(PREVIOUS / 'execution_v1/checkpoints' / (j['job_id'] + '.json'))['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW']
    assert len(failures) == 1
    assert not [p for p in (PREVIOUS / 'execution_v1/request_intents').glob('*.json') if not (PREVIOUS / 'execution_v1/checkpoints' / p.name).exists()]
    hashes = {}
    for name in ['preflight.json', 'sampling_manifest.json', 'pilot_businesses.json', 'sampling_candidate_features.json',
                 'manual_review_25_manifest.json', 'pilot_jobs.json', 'frozen_source_witness.json', 'pilot_usage_estimate.json',
                 'runtime_scheduling_history.json']:
        copy_exact(PREVIOUS / name, FOLDER / name, hashes)
    for branch in ['protocol_snapshot', 'concurrency4_trial_v1', 'concurrency4_continuous_v1', 'previous_failed_attempts']:
        for p in (PREVIOUS / branch).rglob('*'):
            if p.is_file():
                copy_exact(p, FOLDER / p.relative_to(PREVIOUS), hashes)
    for branch in ['checkpoints', 'raw_responses', 'raw_requests', 'request_intents', 'provisional_targets', 'stream_diagnostics']:
        for p in (PREVIOUS / 'execution_v1' / branch).glob('*'):
            if not p.is_file():
                continue
            if p.stem in failures and branch in ['checkpoints', 'request_intents', 'stream_diagnostics']:
                copy_exact(p, FOLDER / 'previous_failed_attempts' / branch / p.name, hashes)
            else:
                copy_exact(p, OUT / branch / p.name, hashes)
    adapters()
    protected = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in PREVIOUS.rglob('*') if p.is_file()}
    pending = [j for j in jobs if not (OUT / 'checkpoints' / (j['job_id'] + '.json')).exists()]
    paid = sum(not j.get('skip_api_empty_captions') for j in pending)
    assert paid == 58 and len(pending) == 59
    runtime = ['scripts/stage_g_d041_recovery_v4.py', 'scripts/run_stage_g_d041_recovery_v4.py',
               'scripts/stage_g_teacher_stream_diagnostics_v1.py', 'scripts/audit_stage_g_d041_recovery_v4.py',
               'scripts/export_stage_g_d041_recovery_v4_review.py', 'scripts/stage_g_review_export_secret_guard.py']
    authorization = {'approved': True, 'approved_by': 'user', 'approved_utc': now(),
                     'basis': 'User explicitly requested fixing the HTTP520 request and continuing; one serial operator retry then57 unstarted paid calls if recovery passes.',
                     'operator_retry_job_ids': failures, 'maximum_operator_retries': 1,
                     'maximum_unstarted_paid_calls': 57, 'maximum_new_paid_calls': 58,
                     'maximum_total_cohort_attempts_including_previous_failures': 295,
                     'automatic_retries': 0, 'recovery_concurrency': 1, 'subsequent_concurrency': 4,
                     'production_authorized': False, 'cost_monitoring': 'researcher_Coletoken_dashboard'}
    immutable(FOLDER / 'recovery_authorization.json', authorization)
    immutable(FOLDER / 'continuation_manifest.json', {
        'created_utc': now(), 'status': 'USER_APPROVED_ONE_SERIAL_HTTP520_RECOVERY_THEN_UNSTARTED_CONCURRENCY4',
        'source_namespace': PREVIOUS.relative_to(ROOT).as_posix(),
        'config_sha256': file_sha(CONFIG), 'runtime_sha256': {p: file_sha(ROOT / p) for p in runtime},
        'copied_files_sha256': hashes, 'previous_namespace_preserved_sha256': protected,
        'imported_accepted_tasks': 241, 'imported_complete_businesses': 79, 'imported_complete_API_responses': 234,
        'previous_total_API_attempts': 237, 'previous_failed_attempts_preserved': failures,
        'operator_retry_job_ids': failures, 'maximum_new_paid_calls': 58, 'automatic_retries': 0,
        'inference_payload_settings_changed': False, 'Codex_configuration_or_authentication_access': False,
        'production_authorized': False,
    })
    check_inputs(verify_checkpoints=True)
    immutable(FOLDER / 'continuation_preflight.json', {'status': 'OFFLINE_RECOVERY_PREFLIGHT_PASS',
              'API_calls': 0, 'credential_access': False, 'imported_complete_businesses': 79,
              'imported_accepted_tasks': 241, 'operator_retries_authorized': 1, 'unstarted_paid_calls': 57,
              'cohort100_and_manual25_unchanged': True, 'previous_files_unchanged': True})
    (FOLDER / 'STOP_REQUESTED.json').write_text('{"prepared_recovery_handshake":true}\n', encoding='utf-8')
    pointer = {'current_namespace': FOLDER.relative_to(ROOT).as_posix(), 'previous_namespace_preserved': PREVIOUS.relative_to(ROOT).as_posix(),
               'status': 'RECOVERY_V3_PREPARED', 'previous_failures_preserved': True, 'operator_retries_authorized': 1}
    (ORIGINAL / 'active_execution_pointer.json').write_text(json.dumps(pointer, indent=2) + '\n', encoding='utf-8')
    print('Recovery v3 preflight PASS; no API calls or credential access.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        check_inputs(verify_checkpoints=True)
        print('Offline recovery checks PASS.')
