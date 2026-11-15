"""Continue unstarted D041 tasks serially; preserve the failed task and prior reports.

No automatic retry, no new cohort, no target writing outside frozen API serialization.
"""
import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from run_stage_g_extended_pilot_d041_dashboard import preflight as original_preflight
from run_stage_g_extended_pilot_d041 import (
    ROOT, FOLDER as ORIGINAL, CONFIG, read, file_sha, canonical, digest, immutable,
    credentials, BASE_URL, PROVIDER, MODEL, job_result, payload, now, build_target, usage_so_far,
)

FOLDER = ORIGINAL / 'continuation_v2'
OUT = FOLDER / 'execution_v1'
PREPARED_FILES = [
    'preflight.json', 'sampling_manifest.json', 'pilot_businesses.json',
    'sampling_candidate_features.json', 'manual_review_25_manifest.json',
    'pilot_jobs.json', 'frozen_source_witness.json', 'pilot_usage_estimate.json',
]
ARTIFACT_DIRS = ['checkpoints', 'raw_responses', 'raw_requests', 'request_intents', 'provisional_targets']


def copy_exact(source, destination):
    h = file_sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): assert file_sha(destination) == h, 'Existing copy changed: ' + str(destination)
    else: shutil.copy2(source, destination)
    assert file_sha(source) == file_sha(destination) == h
    return h


def create_adapters():
    name = 'run_stage_g_d041_continuation_v2'
    audit = ROOT / 'scripts/audit_stage_g_extended_pilot_d041.py'
    source = audit.read_text(encoding='utf-8')
    source = source.replace('from prepare_stage_g_extended_pilot_d041 import FOLDER, CONFIG, choose, rank',
                            'from prepare_stage_g_extended_pilot_d041 import CONFIG, choose, rank\nfrom ' + name + ' import FOLDER')
    source = source.replace('from run_stage_g_extended_pilot_d041 import check_inputs, build_target',
                            'from ' + name + ' import check_inputs, build_target')
    adapters = {'audit_stage_g_d041_continuation_v2.py': '# Original audit SHA256: ' + file_sha(audit) + '\n' + source}
    exporter = ROOT / 'scripts/export_stage_g_extended_pilot_d041_review.py'
    source = exporter.read_text(encoding='utf-8')
    source = source.replace('from prepare_stage_g_extended_pilot_d041 import FOLDER, CONFIG',
                            'from prepare_stage_g_extended_pilot_d041 import CONFIG\nfrom ' + name + ' import FOLDER')
    source = source.replace('from run_stage_g_extended_pilot_d041 import check_inputs',
                            'from ' + name + ' import check_inputs')
    source = source.replace("PREFIX='chatgpt_stage_g_d041_random25_review/'", "PREFIX='chatgpt_stage_g_d041_random25_review_continuation_v2/'")
    source = source.replace("ZIP=FOLDER/'chatgpt_stage_g_d041_random25_review.zip'", "ZIP=FOLDER/'chatgpt_stage_g_d041_random25_review_continuation_v2.zip'")
    marker = "    files['additional_flagged_cases.json']=original(FOLDER/'technical_flags.json')"
    assert source.count(marker) == 1
    extras = marker + "\n    for report in ['continuation_manifest.json','continuation_preflight.json','execution_report.json','execution_v1/completion.json']:\n        files['reports/'+report]=original(FOLDER/report)\n    files['protocol/runtime/run_stage_g_d041_continuation_v2.py']=original(ROOT/'scripts/run_stage_g_d041_continuation_v2.py')"
    source = source.replace(marker, extras)
    adapters['export_stage_g_d041_continuation_v2_review.py'] = '# Original exporter SHA256: ' + file_sha(exporter) + '\n' + source
    for filename, source in adapters.items():
        path = ROOT / 'scripts' / filename
        if path.exists(): assert path.read_text(encoding='utf-8') == source
        else: path.write_text(source, encoding='utf-8')
    return {filename: file_sha(ROOT / 'scripts' / filename) for filename in adapters}


def prepare():
    cfg, pre, jobs, auth = original_preflight()
    previous = read(ORIGINAL / 'execution_v1/completion.json')
    assert previous['status'] == 'D041_STOPPED_FOR_REVIEW'
    assert previous['new_API_request_intents'] == 51
    assert previous['businesses_completed'] == 17
    assert not FOLDER.exists(), 'Continuation already prepared; use its offline status/execute command'
    assert not [p for p in (ORIGINAL / 'execution_v1/request_intents').glob('*.json')
                if not (ORIGINAL / 'execution_v1/checkpoints' / p.name).exists()]
    copied = {}
    for filename in PREPARED_FILES:
        copied[filename] = copy_exact(ORIGINAL / filename, FOLDER / filename)
    for p in (ORIGINAL / 'protocol_snapshot').rglob('*'):
        if p.is_file():
            rel = p.relative_to(ORIGINAL).as_posix()
            copied[rel] = copy_exact(p, FOLDER / rel)
    for directory in ARTIFACT_DIRS:
        for p in (ORIGINAL / 'execution_v1' / directory).glob('*'):
            if p.is_file():
                rel = 'execution_v1/' + directory + '/' + p.name
                copied[rel] = copy_exact(p, FOLDER / rel)
    preserved = {}
    for branch in ['execution_v1', 'protocol_snapshot', 'concurrency5_trial_v1', 'human_review_25_ui', 'preparation_review_ui_v1']:
        for p in (ORIGINAL / branch).rglob('*'):
            if p.is_file(): preserved[p.relative_to(ROOT).as_posix()] = file_sha(p)
    for p in ORIGINAL.iterdir():
        if p.is_file() and p.name != 'active_execution_pointer.json':
            preserved[p.relative_to(ROOT).as_posix()] = file_sha(p)
    failed_ids = [p.stem for p in (OUT / 'checkpoints').glob('*.json')
                  if read(p)['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW']
    pending = [j for j in jobs if not (OUT / 'checkpoints' / (j['job_id'] + '.json')).exists()]
    maximum_new = sum(not j.get('skip_api_empty_captions') for j in pending)
    assert maximum_new == 241 and len(pending) == 246 and len(failed_ids) == 1
    adapters = create_adapters()
    manifest = {
        'status': 'USER_AUTHORIZED_UNSTARTED_TASK_CONTINUATION_SERIAL1', 'created_utc': now(),
        'authorization_basis': 'Researcher instructed continue after the failed five-request trial.',
        'requested_model': MODEL, 'effective_concurrency': 1, 'automatic_retries': 0,
        'source_namespace': ORIGINAL.relative_to(ROOT).as_posix(),
        'output_namespace': FOLDER.relative_to(ROOT).as_posix(),
        'config_sha256': file_sha(CONFIG), 'sampling_manifest_sha256': file_sha(FOLDER / 'sampling_manifest.json'),
        'manual_review_25_manifest_sha256': file_sha(FOLDER / 'manual_review_25_manifest.json'),
        'imported_API_request_intents': 51, 'maximum_new_initial_API_calls': maximum_new,
        'cohort_initial_call_ceiling': 292, 'pending_task_slots': len(pending),
        'failed_tasks_preserved_not_retried': failed_ids,
        'copied_files_sha256': copied, 'previous_namespace_preserved_sha256': preserved,
        'adapter_sha256': adapters,
        'runner_sha256': file_sha(Path(__file__)),
        'Codex_configuration_or_authentication_access': False,
        'monetary_usage_monitoring': 'researcher_Coletoken_dashboard',
        'teacher_content_origin': 'API_only_frozen_serialization', 'production_authorized': False,
    }
    immutable(FOLDER / 'continuation_manifest.json', manifest)
    check_inputs(verify_checkpoints=True)
    immutable(FOLDER / 'continuation_preflight.json', {
        'status': 'OFFLINE_CONTINUATION_PREFLIGHT_PASS', 'new_API_calls': 0,
        'credential_access': False, 'imported_paid_intents': 51, 'imported_checkpoints': 54,
        'imported_complete_raw_responses': 50, 'imported_complete_businesses': 17,
        'failed_tasks_not_retried': failed_ids, 'unstarted_paid_tasks': 241,
        'cohort100_and_manual25_unchanged': True, 'inference_payloads_and_sources_unchanged': True,
        'concurrency': 1, 'automatic_retries': 0,
        'manifest_sha256': file_sha(FOLDER / 'continuation_manifest.json'),
    })
    pointer = {'current_namespace': FOLDER.relative_to(ROOT).as_posix(),
               'previous_namespace_preserved': ORIGINAL.relative_to(ROOT).as_posix(),
               'status': 'CONTINUATION_PREPARED_SERIAL1', 'failed_request_not_retried': True}
    (ORIGINAL / 'active_execution_pointer.json').write_text(json.dumps(pointer, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(read(FOLDER / 'continuation_preflight.json')), flush=True)


def check_inputs(verify_checkpoints=False):
    cfg, pre, jobs, auth = original_preflight()
    manifest = read(FOLDER / 'continuation_manifest.json')
    assert file_sha(CONFIG) == manifest['config_sha256']
    assert file_sha(ROOT / 'scripts/run_stage_g_d041_continuation_v2.py') == manifest['runner_sha256']
    for name, h in manifest['adapter_sha256'].items(): assert file_sha(ROOT / 'scripts' / name) == h
    for rel, h in manifest['copied_files_sha256'].items(): assert file_sha(FOLDER / rel) == h
    for rel, h in manifest['previous_namespace_preserved_sha256'].items(): assert file_sha(ROOT / rel) == h
    assert read(FOLDER / 'pilot_jobs.json') == jobs
    if verify_checkpoints:
        schemas = read(ROOT / cfg['schema'])
        prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
        for job in jobs:
            cp_path = OUT / 'checkpoints' / (job['job_id'] + '.json')
            request_path = OUT / 'raw_requests' / (job['job_id'] + '.json.gz')
            intent_path = OUT / 'request_intents' / cp_path.name
            if intent_path.exists():
                assert cp_path.exists(), 'Unresolved paid request; no automatic resend'
                body = canonical(payload(job, cfg, prompts[job['task']], schemas[job['task']]))
                assert gzip.decompress(request_path.read_bytes()) == body
                assert read(intent_path)['request_sha256'] == digest(body)
            if cp_path.exists() and read(cp_path)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                # Existing checkpoint path ensures job_result never calls the API.
                job_result(job, cfg, prompts[job['task']], schemas[job['task']], BASE_URL, PROVIDER, None, OUT, CONFIG)
    return cfg, pre, jobs


def saved_state(jobs):
    outputs = {}; failures = []; complete = {}
    for job in jobs:
        p = OUT / 'checkpoints' / (job['job_id'] + '.json')
        if not p.exists(): continue
        cp = read(p)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            failures.append({'job_id': job['job_id'], 'business_id': job['business_id'], 'failure': cp}); continue
        outputs.setdefault(job['business_id'], {})[job['condition']] = cp.get('resolved_output', cp['output'])
    for b in read(FOLDER / 'pilot_businesses.json'):
        result = build_target(b, outputs.get(b['business_id'], {}))
        if result is not None:
            immutable(OUT / 'provisional_targets' / (b['business_id'] + '.json'), result)
            complete[b['business_id']] = b
    return outputs, failures, complete


def progress(jobs, complete, failures, current=None):
    r = {'updated_utc': now(), 'current_job': current, 'businesses_completed': len(complete),
         'target_texts': 3 * len(complete), 'API_request_intents': len(list((OUT / 'request_intents').glob('*.json'))),
         'accepted_tasks': sum(read(p)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for p in (OUT / 'checkpoints').glob('*.json')),
         'tasks_planned': 300, 'failed_tasks': len(failures), 'usage': usage_so_far(OUT),
         'effective_concurrency': 1, 'imported_API_request_intents': 51, 'automatic_retries': 0,
         'human_verified': False, 'target_content_origin': 'teacher_API_only_frozen_serialization'}
    path = OUT / 'execution_progress.json'; temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8'); temp.replace(path)
    print(json.dumps(r), flush=True)


def execute():
    cfg, pre, jobs = check_inputs(verify_checkpoints=True)
    if (OUT / 'completion.json').exists(): return read(OUT / 'completion.json')
    base, provider, key = credentials()
    assert base == BASE_URL and provider == PROVIDER and cfg['requested_model'] == MODEL
    immutable(OUT / 'run_manifest.json', {'continuation_manifest_sha256': file_sha(FOLDER / 'continuation_manifest.json'),
        'config_sha256': file_sha(CONFIG), 'requested_model': MODEL, 'effective_concurrency': 1,
        'generation_settings': cfg['generation_proposal'], 'maximum_new_initial_API_calls': 241,
        'imported_API_request_intents': 51, 'automatic_retries': 0,
        'Codex_configuration_or_authentication_access': False})
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    businesses = {b['business_id']: b for b in read(FOLDER / 'pilot_businesses.json')}
    outputs, failures, complete = saved_state(jobs)
    started = time.monotonic(); stop = None
    for job in jobs:
        cp_path = OUT / 'checkpoints' / (job['job_id'] + '.json')
        if cp_path.exists(): continue  # All existing successful/failed checkpoints preserved, never rebilled.
        if (FOLDER / 'STOP_REQUESTED.json').exists():
            progress(jobs, complete, failures)
            print('Operator pause at saved-response boundary; no active call interrupted.', flush=True)
            return None
        if (OUT / 'request_intents' / cp_path.name).exists():
            stop = 'Unresolved prior request; do not resend'; break
        if not job.get('skip_api_empty_captions'):
            assert len(list((OUT / 'request_intents').glob('*.json'))) < 292
            body = canonical(payload(job, cfg, prompts[job['task']], schemas[job['task']]))
            p = OUT / 'raw_requests' / (job['job_id'] + '.json.gz')
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists(): assert gzip.decompress(p.read_bytes()) == body
            else: p.write_bytes(gzip.compress(body, mtime=0))
        progress(jobs, complete, failures, job['job_id'])
        try:
            cp = job_result(job, cfg, prompts[job['task']], schemas[job['task']], base, provider, key, OUT, CONFIG)
        except (RuntimeError, ValueError):
            failed = read(cp_path) if cp_path.exists() else {'status': 'UNRESOLVED_REQUEST_NO_RESEND'}
            failures.append({'job_id': job['job_id'], 'business_id': job['business_id'], 'failure': failed})
            raw = OUT / 'raw_responses' / cp_path.name
            if failed.get('http_status') in [401, 403, 429] or not raw.exists():
                stop = 'New account/rate/network/stream failure; stopped without automatic retry'; break
            if read(raw).get('model') != MODEL:
                stop = 'Provider model provenance mismatch'; break
            continue
        outputs.setdefault(job['business_id'], {})[job['condition']] = cp.get('resolved_output', cp['output'])
        result = build_target(businesses[job['business_id']], outputs[job['business_id']])
        if result is not None:
            immutable(OUT / 'provisional_targets' / (job['business_id'] + '.json'), result)
            complete[job['business_id']] = businesses[job['business_id']]
        progress(jobs, complete, failures)
    total_intents = len(list((OUT / 'request_intents').glob('*.json')))
    raw = [read(p) for p in (OUT / 'raw_responses').glob('*.json')]
    report = {'status': 'D041_STOPPED_FOR_REVIEW' if stop else ('D041_ATTEMPTED_WITH_FAILURES' if failures else 'D041_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE'),
        'businesses_planned': 100, 'businesses_completed': len(complete), 'target_texts': 3 * len(complete),
        'businesses_by_split': dict(Counter(b['split'] for b in complete.values())),
        'new_API_request_intents': total_intents, 'legacy_call_count_field_is_cumulative': True,
        'cumulative_cohort_API_request_intents': total_intents, 'imported_API_request_intents': 51,
        'new_initial_API_calls_this_continuation': total_intents - 51,
        'empty_tasks_planned': pre['empty_tasks'], 'usage': usage_so_far(OUT),
        'reported_usage_includes_imported_responses': True,
        'responses_without_usage': sum(not bool(r.get('usage')) for r in raw),
        'request_intents_without_complete_raw_response': [p.stem for p in (OUT / 'request_intents').glob('*.json') if not (OUT / 'raw_responses' / p.name).exists()],
        'reported_model_counts': dict(Counter(r.get('model') for r in raw)),
        'failed_tasks': failures, 'stop_reason': stop, 'automatic_retries': 0,
        'effective_concurrency_this_continuation': 1, 'runtime_seconds_this_continuation': round(time.monotonic() - started, 3),
        'cost_monitoring': 'researcher_Coletoken_dashboard', 'provider_invoice_cost': None,
        'human_verified_businesses': 0, 'production_protocol_frozen': False,
        'held_out_teacher_targets': 0, 'remaining_2100_generation_started': False, 'LED_training_started': False}
    progress(jobs, complete, failures)
    check_inputs()
    immutable(OUT / 'completion.json', report)
    return report


def finalize(completion):
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_stage_g_d041_continuation_v2.py'), '--audit-results'], cwd=ROOT, check=True)
    validation = read(FOLDER / 'validation_report.json')
    immutable(FOLDER / 'execution_report.json', {'execution': completion, 'validation_status': validation['status'],
        'source_namespace_preserved': True, '100_and_manual25_manifests_unchanged': True,
        'previous_failure_preserved_not_retried': True, 'effective_continuation_concurrency': 1,
        'all_target_content_from_teacher_API': True, 'independent_factual_accuracy': None,
        'researcher_review_pending': True, 'production_or_training_started': False})
    marker = '**D041 continuation_v2 result (2026-10-11):**'
    note = f"\n{marker} {completion['businesses_completed']}/100 complete businesses, {completion['target_texts']} provisional target texts; {completion['new_initial_API_calls_this_continuation']} new initial calls in this continuation and {completion['cumulative_cohort_API_request_intents']} cumulative cohort attempts. Existing failed response remains preserved, not retried; failures={len(completion['failed_tasks'])}; stop reason={completion['stop_reason']}. Actual concurrency1. Validation={validation['status']}; schema/source identity PASS does not establish independent factual accuracy. Updated fixed25 ZIP/page and reports are under `outputs/stage_g/teacher_extended_pilot_d041_v1/continuation_v2/`; prior partial reports/ZIPs remain unchanged. No production teacher freeze/full-cohort generation/test targets/training.\n"
    for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md', 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
        path = ROOT / rel; text = path.read_text(encoding='utf-8')
        if marker not in text:
            first, rest = text.split('\n', 1); path.write_text(first + '\n' + note + '\n' + rest, encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_stage_g_d041_continuation_v2_review.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/execute_stage_g_d041_status_cell.py')], cwd=ROOT, check=True)
    immutable(FOLDER / 'postprocessing_status.json', {'status': 'COMPLETED', 'API_calls': 0,
        'businesses_complete': completion['businesses_completed'], 'factual_review_pending': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute-pilot', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.execute_pilot:
        result = execute()
        if result is not None:
            try: finalize(result)
            except Exception as error:
                immutable(FOLDER / 'postprocessing_status.json', {'status': 'FAILED_PRESERVED_FOR_REVIEW', 'error_type': type(error).__name__, 'API_calls': 0})
                raise
    else:
        cfg, pre, jobs = check_inputs(verify_checkpoints=True)
        print('Offline continuation checks PASS; no credential access/API call.')
