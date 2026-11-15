"""One user-approved five-request Coletoken recovery trial; no automatic repeats."""
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from stage_g_production_d042 import (
    ROOT, CONFIG, FOLDER, OUT, read, canonical, file_sha, immutable, atomic,
    check_inputs, verify_checkpoint, now,
)
from run_stage_g_production_d042 import runner_lock
from run_stage_g_teacher_pilot_v4_1_recovery import payload, job_result
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL, KEY_VARIABLE
from run_stage_g_d042_concurrency10_new_key_trial_v2 import preserved_files, stats
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import stage_g_teacher_sse_transport_v1 as transport

TRIAL = FOLDER / 'coletoken_concurrency5_recovery_v1'
EXEC = TRIAL / 'execution_v1'


def main():
    lock = runner_lock()
    assert not (TRIAL / 'execution_started.json').exists(), 'No automatic rerun'
    assert (FOLDER / 'STOP_REQUESTED.json').exists()
    cfg, jobs, businesses, imports = check_inputs()
    old_report = read(FOLDER / 'execution_control_concurrency10_trial_v1/report.json')
    selected_ids = [r['job_id'] for r in old_report['results'] if not r['passed']]
    selected = [j for j in jobs if j['job_id'] in selected_ids]
    assert len(selected) == len(set(selected_ids)) == 5
    assert all(j['split'] in ('train', 'validation') for j in selected)
    assert all(read(OUT / 'checkpoints' / (j['job_id'] + '.json'))['status'].startswith('FAILED') for j in selected)
    prior = preserved_files()
    for name in ('mirai_concurrency6_stream_trial_v1', 'mirai_grouped6_recovery_v1', 'mirai_grouped6_recovery_v2'):
        prior.update({p.relative_to(ROOT).as_posix(): file_sha(p) for p in (FOLDER / name).rglob('*') if p.is_file()})
    immutable(TRIAL / 'prior_artifact_sha256.json', prior)
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    for j in selected:
        encoded = canonical(payload(j, cfg, prompts[j['task']], schemas[j['task']]))
        assert encoded == gzip.decompress((OUT / 'raw_requests' / (j['job_id'] + '.json.gz')).read_bytes())
        p = EXEC / 'raw_requests' / (j['job_id'] + '.json.gz'); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f: f.write(gzip.compress(encoded, mtime=0))
    immutable(TRIAL / 'manifest.json', {
        'approved_by': 'user', 'authorization': 'Return teacher to old Coletoken key and test five concurrent requests',
        'provider': PROVIDER, 'base_url': BASE_URL, 'requested_model': MODEL,
        'maximum_concurrency': 5, 'maximum_calls': 5, 'automatic_retries': 0,
        'job_ids': [j['job_id'] for j in selected], 'selection': 'five preserved failures from original Coletoken ten-request trial',
        'successful_Coletoken_outputs_regenerated': 0, 'Mirai_outputs_preserved_not_imported': True,
        'config_sha256': file_sha(CONFIG), 'runner_sha256': file_sha(Path(__file__)),
        'transport_sha256': file_sha(ROOT / 'scripts/stage_g_teacher_sse_transport_v1.py'),
        'request_payloads_identical_to_previous_Coletoken_attempts': True,
        'reasoning': 'medium', 'service_tier': 'omitted_provider_default',
        'socket_idle_timeout_seconds': 120, 'overall_request_deadline_seconds': 600,
        'strict_model_schema_and_source_validation_unchanged': True,
        'offline_transport_fixture': transport.offline_self_check(),
        'full_production_concurrency5_started': False, 'Codex_config_auth_key_accessed': False})
    assert not os.environ.get(KEY_VARIABLE), 'Teacher env override must not bypass restored private key'
    base, provider, key = credentials()
    assert (base, provider, cfg['requested_model']) == (BASE_URL, PROVIDER, MODEL)
    immutable(TRIAL / 'execution_started.json', {'started_utc': now(), 'maximum_calls': 5})
    legacy.call = transport.call  # Only transport; never invoke legacy.credentials().
    active = [j['job_id'] for j in selected]; results = []; started = time.monotonic()

    def progress():
        atomic(TRIAL / 'execution_progress.json', {'status': 'RUNNING_COLETOKEN_FIVE_REQUEST_TRIAL',
            'provider': provider, 'current_jobs': active[:], 'maximum_concurrency': 5,
            'results': results[:], 'pid': os.getpid(), 'updated_utc': now(),
            'production_unchanged': True, 'automatic_retries': 0})

    def worker(j):
        transport.set_context(EXEC / 'stream_diagnostics', j['job_id'])
        try:
            cp = job_result(j, cfg, prompts[j['task']], schemas[j['task']], base, provider, key, EXEC, CONFIG)
            return {'job_id': j['job_id'], 'passed': True, 'seconds': cp['seconds'], 'response_id': cp['response_id']}
        except Exception as error:
            return {'job_id': j['job_id'], 'passed': False, 'error_type': type(error).__name__}

    progress()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(worker, j): j for j in selected}
        for future in as_completed(futures):
            r = future.result(); results.append(r); active.remove(r['job_id']); progress()
            print(json.dumps(r), flush=True)
    usage = Counter(); model = Counter(); tiers = Counter(); efforts = Counter(); timing = defaultdict(list); events = []; failures = []
    for j in selected:
        jid = j['job_id']; d = read(EXEC / 'stream_diagnostics' / (jid + '.json'))
        timing[j['task']].append(d['elapsed_seconds'])
        events.extend([(datetime.fromisoformat(d['started_utc']).timestamp(), 1),
                       (datetime.fromisoformat(d['finished_utc']).timestamp(), -1)])
        cp = read(EXEC / 'checkpoints' / (jid + '.json'))
        if cp['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            verify_checkpoint(j, cfg, EXEC, CONFIG)
        else:
            failures.append({'job_id': jid, 'checkpoint': cp, 'transport': d})
        rawp = EXEC / 'raw_responses' / (jid + '.json')
        if rawp.exists():
            raw = read(rawp); model[str(raw.get('model'))] += 1
            tiers[str(raw.get('service_tier'))] += 1; efforts[str((raw.get('reasoning') or {}).get('effort'))] += 1
            u = raw.get('usage') or {}; usage['responses_with_usage'] += bool(u)
            usage['input_tokens'] += u.get('input_tokens', 0); usage['output_tokens'] += u.get('output_tokens', 0)
    live = peak = 0
    for _, delta in sorted(events, key=lambda x: (x[0], x[1])):
        live += delta; peak = max(peak, live)
    for rel, sha in prior.items(): assert file_sha(ROOT / rel) == sha, rel
    for p in TRIAL.rglob('*'):
        if p.is_file():
            data = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
            assert key.encode() not in data
    passed = sum(r['passed'] for r in results)
    report = {'status': 'COLETOKEN_FIVE_REQUEST_TRIAL_PASS' if passed == 5 else 'COLETOKEN_FIVE_REQUEST_TRIAL_WITH_FAILURES',
        'provider': provider, 'base_url': base, 'requested_model': cfg['requested_model'],
        'passed': passed, 'failed': 5-passed, 'maximum_concurrency': 5, 'observed_transport_overlap_peak': peak,
        'paid_calls': len(list((EXEC / 'request_intents').glob('*.json'))),
        'batch_seconds': round(time.monotonic()-started, 3), 'seconds_by_task': {k: stats(v) for k,v in timing.items()},
        'returned_models': dict(model), 'returned_service_tiers': dict(tiers), 'returned_reasoning_efforts': dict(efforts),
        'usage_returned': dict(usage), 'results': results, 'failures': failures,
        'strict_schema_model_and_source_checks': True, 'scientific_API_payloads_unchanged': True,
        'prior_artifacts_unchanged': True, 'Codex_config_auth_key_accessed': False, 'teacher_content_manually_rewritten': False,
        'automatic_retries': 0, 'human_factual_ground_truth_certified': False,
        'Mirai_stopped': True, 'continuous_production_five_workers_started': False}
    immutable(TRIAL / 'report.json', report)
    atomic(TRIAL / 'execution_progress.json', dict(report, current_jobs=[], updated_utc=now()))
    note = (f'**Coletoken five-request recovery trial:** {passed}/5 strict model/schema/source PASS; '
            f'overlap={peak}, wall={report["batch_seconds"]}s. Original key restored in teacher-only private store; Mirai stopped. '
            'Original D040 scientific payloads unchanged; old failures and all Mirai outputs preserved. '
            'No Codex config/auth/key access, held-out teacher targets, target rewriting or LED training. '
            'One bounded trial does not establish sustained provider capacity. '
            'Report: outputs/stage_g/teacher_production_d042_v1/coletoken_concurrency5_recovery_v1/report.json.\n')
    for rel in ('thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md'):
        p = ROOT / rel; first, rest = p.read_text(encoding='utf-8').split('\n', 1)
        p.write_text(first+'\n\n'+note+'\n'+rest, encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        atomic(TRIAL / 'runner_failure.json', {'status': 'STOPPED', 'error_type': type(error).__name__, 'utc': now()})
        print(json.dumps({'runner_error': type(error).__name__}), flush=True)
        raise SystemExit(1)
