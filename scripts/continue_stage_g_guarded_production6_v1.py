"""User-authorized six-call production continuation with one reviewed source recovery.

All content comes from the unchanged GPT Teacher API. Old attempts and successful
components are preserved under their original schema/configuration bindings.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import stage_g_production_d042 as frozen
import run_stage_g_business_id_guard_production_v1 as parent
import stage_g_business_id_guard_v1 as guard
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import stage_g_teacher_stream_diagnostics_v1 as transport
from prepare_stage_g_review_evidence_ids import evidence_spans
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL, KEY_VARIABLE

ROOT = frozen.ROOT
CONFIG = ROOT / 'configs/stage_g_guarded_production6_v1.json'
RUN = frozen.FOLDER / 'guarded_production6_v1'
OUT = RUN / 'execution_v1'
RECOVERY = RUN / 'operator_reviewed_source_retry_v1' / 'execution_v1'
TRIAL = frozen.FOLDER / 'coletoken_guarded_concurrency6_trial_v1'
TRIAL_CONFIG = ROOT / 'configs/stage_g_coletoken_guarded_concurrency6_trial_v1.json'
SCRIPT = Path(__file__).resolve()
FAILED_JOB = 'train_16tjKOvMw_nOgNViq2LwOQ_review_consensus'
CACHE = None
INDEX_CACHE = None
TARGET_CACHE = None


def read_index():
    global INDEX_CACHE
    if INDEX_CACHE is None: INDEX_CACHE = frozen.read(RUN / 'reuse_index.json')
    return INDEX_CACHE


def historical_verify(job, entry):
    directory = ROOT / entry['directory']
    config = ROOT / entry['config_path']
    cfg = frozen.read(config)
    verify = guard.verify_checkpoint if entry['request_schema'] == 'business_id_enum_v1' else frozen.verify_checkpoint
    return verify(job, cfg, directory, config)


def source_failure_audit(job):
    raw = frozen.read(parent.OUT / 'raw_responses' / (job['job_id'] + '.json'))
    output = json.loads(legacy.response_text(raw))
    allowed = {r['review_marker']: {s['evidence_span_id'] for s in evidence_spans(r)}
               for r in job['evidence']['reviews']}
    issues = []
    for ci, claim in enumerate(output['claims']):
        for si, support in enumerate(claim['support']):
            ids = support['evidence_span_ids']
            valid = allowed.get(support['review_marker'], set())
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= valid:
                issues.append({'claim_index': ci, 'support_index': si,
                    'review_marker': support['review_marker'], 'reported_span_ids': ids,
                    'invalid_span_ids': sorted(set(ids) - valid)})
    assert issues and output['business_id'] == job['business_id']
    return {'job_id': job['job_id'], 'issue_count': len(issues), 'issues': issues,
        'cause': 'Evidence spans from other REV markers grouped under one support entry',
        'incorrect_output_accepted': False, 'output_rewritten': False,
        'action': 'one_user_authorized_API_regeneration_same_payload_then_strict_validation',
        'automatic_retries': 0}


def prepare():
    lock = parent.restored.shared_lock()
    assert not (RUN / 'manifest.json').exists(), 'Use --run for an already prepared execution'
    old_cfg, jobs, businesses, imports = parent.checked_inputs()
    cfg = frozen.read(CONFIG)
    assert all(cfg[k] == old_cfg[k] for k in frozen.FIXED_KEYS)
    assert cfg['identifier_guard'] == old_cfg['identifier_guard']
    assert cfg['execution']['maximum_concurrency'] == 6
    state = frozen.read(parent.OUT / 'execution_progress.json')
    trial_state = frozen.read(TRIAL / 'execution_progress.json')
    trial_report = frozen.read(TRIAL / 'report.json')
    assert state['status'] == 'STOPPED_REQUIRES_REVIEW' and not state['current_jobs']
    assert trial_report['passed'] == 6 and trial_report['paid_calls'] == 6
    assert trial_report['observed_client_transport_overlap_peak'] == 6 and not trial_state['current_jobs']
    assert frozen.file_sha(TRIAL_CONFIG) == frozen.read(TRIAL / 'manifest.json')['config_sha256']
    for rel, sha in frozen.read(TRIAL / 'manifest.json')['previous_execution_sha256'].items():
        assert frozen.file_sha(ROOT / rel) == sha, rel
    index = dict(frozen.read(parent.RUN / 'reuse_index.json'))
    by_id = {j['job_id']: j for j in jobs}
    failed = []
    for directory, config in ((parent.OUT, parent.CONFIG), (TRIAL / 'execution_v1', TRIAL_CONFIG)):
        for path in (directory / 'checkpoints').glob('*.json'):
            job = by_id[path.stem]
            cp = frozen.read(path)
            if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                assert directory == parent.OUT and job['job_id'] == FAILED_JOB
                assert cp['local_validation_error'] == 'Invalid, duplicate, empty or wrong-review source spans'
                failed.append(job['job_id'])
                continue
            assert job['job_id'] not in index, 'Duplicate accepted task'
            index[job['job_id']] = {'directory': directory.relative_to(ROOT).as_posix(),
                'config_path': config.relative_to(ROOT).as_posix(), 'request_schema': 'business_id_enum_v1'}
    assert failed == [FAILED_JOB]
    hashes = {}
    for jid, entry in index.items():
        cp = historical_verify(by_id[jid], entry)
        directory = ROOT / entry['directory']
        paths = [directory / 'checkpoints' / (jid + '.json')]
        if cp.get('api_called'):
            paths += [directory / 'raw_responses' / (jid + '.json'),
                directory / 'request_intents' / (jid + '.json'), directory / 'raw_requests' / (jid + '.json.gz')]
        for path in paths: hashes[path.relative_to(ROOT).as_posix()] = frozen.file_sha(path)
    # Snapshot every attempt of the previous guarded executions, including failures.
    for directory in (parent.OUT, TRIAL / 'execution_v1'):
        for path in directory.rglob('*'):
            if path.is_file(): hashes[path.relative_to(ROOT).as_posix()] = frozen.file_sha(path)
    grouped = defaultdict(list)
    for job in jobs: grouped[job['business_id']].append(job)
    target_index = {}
    for b in businesses:
        js = grouped[b['business_id']]
        if not all(j['job_id'] in index for j in js): continue
        content = {j['condition']: frozen.read(ROOT / index[j['job_id']]['directory'] / 'checkpoints' /
            (j['job_id'] + '.json')) for j in js}
        target = frozen.build_target(b, {k: cp.get('resolved_output', cp['output']) for k, cp in content.items()})
        origins = {ROOT / index[j['job_id']]['directory'] for j in js}
        candidates = [d / sub / (b['business_id'] + '.json') for d in origins
                      for sub in ('targets', 'provisional_targets')]
        # Some earlier continuations assembled targets in a new namespace while
        # retaining every component in older namespaces. Resolve the existing
        # exact frozen target rather than assuming it sits next to a checkpoint.
        if not any(p.exists() for p in candidates):
            candidates += sorted(p for p in frozen.FOLDER.rglob(b['business_id'] + '.json')
                if p.parent.name in ('targets', 'provisional_targets'))
        path = next(p for p in candidates if p.exists() and frozen.read(p) == target)
        assert frozen.read(path) == target
        target_index[b['business_id']] = path.relative_to(ROOT).as_posix()
        hashes[path.relative_to(ROOT).as_posix()] = frozen.file_sha(path)
    assert len(target_index) == 155 and len(index) == 466
    job = by_id[FAILED_JOB]
    schema = frozen.read(ROOT / cfg['schema'])[job['task']]
    prompt = (ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
    exact = gzip.decompress((parent.OUT / 'raw_requests' / (FAILED_JOB + '.json.gz')).read_bytes())
    assert exact == frozen.canonical(guard.payload_for_job(job, cfg, prompt, schema))
    frozen.immutable(RUN / 'reuse_index.json', index)
    frozen.immutable(RUN / 'reuse_targets.json', target_index)
    frozen.immutable(RUN / 'source_failure_review.json', source_failure_audit(job))
    auth = frozen.read(frozen.FOLDER / 'authorization.json')
    auth.update(config_sha256=frozen.file_sha(CONFIG), continuation_authorized_by='user',
        maximum_concurrency=6, reviewed_retry_job=FAILED_JOB, maximum_reviewed_retry_calls=1)
    frozen.immutable(RUN / 'authorization.json', auth)
    for name in ('small_batch_manifest.json', 'small_batch_validation.json'):
        destination = RUN / name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((frozen.FOLDER / name).read_bytes())
    frozen.immutable(RUN / 'manifest.json', {'approved_by': 'user',
        'authorization': 'Continue target generation after successful six-request trial',
        'maximum_concurrency': 6, 'scheduler': 'six_request_drained_batches',
        'automatic_retries': 0, 'reviewed_retry_job': FAILED_JOB, 'maximum_reviewed_retry_calls': 1,
        'original_failed_attempt_preserved': True, 'successful_components_regenerated': False,
        'initial_complete_businesses': len(target_index), 'initial_reused_components': len(index),
        'config_sha256': frozen.file_sha(CONFIG),
        'reuse_index_sha256': frozen.file_sha(RUN / 'reuse_index.json'),
        'reuse_targets_sha256': frozen.file_sha(RUN / 'reuse_targets.json'),
        'code_sha256': {p.relative_to(ROOT).as_posix(): frozen.file_sha(p) for p in (
            SCRIPT, ROOT / 'scripts/run_stage_g_production_d042_six_worker_v1.py',
            ROOT / 'scripts/stage_g_business_id_guard_v1.py')},
        'preserved_artifact_sha256': hashes, 'scientific_settings_changed': False,
        'Codex_configuration_or_auth_access': False, 'prepared_utc': frozen.now()})
    print(json.dumps({'offline_preflight': 'PASS', 'complete_businesses_reused': 155,
        'components_reused': 466, 'maximum_concurrency': 6, 'API_calls': 0}), flush=True)


def checked_inputs():
    global CACHE
    manifest = frozen.read(RUN / 'manifest.json')
    assert frozen.file_sha(CONFIG) == manifest['config_sha256']
    assert frozen.file_sha(RUN / 'reuse_index.json') == manifest['reuse_index_sha256']
    assert frozen.file_sha(RUN / 'reuse_targets.json') == manifest['reuse_targets_sha256']
    for rel, sha in manifest['code_sha256'].items(): assert frozen.file_sha(ROOT / rel) == sha, rel
    if CACHE is None:
        old_cfg, jobs, businesses, imports = parent.checked_inputs()
        cfg = frozen.read(CONFIG)
        assert all(cfg[k] == old_cfg[k] for k in frozen.FIXED_KEYS)
        assert cfg['identifier_guard'] == old_cfg['identifier_guard']
        for rel, sha in manifest['preserved_artifact_sha256'].items():
            assert frozen.file_sha(ROOT / rel) == sha, rel
        index = read_index()
        for job in jobs:
            if job['job_id'] in index: historical_verify(job, index[job['job_id']])
        CACHE = cfg, jobs, businesses, imports
    return CACHE


def location(job, imports):
    index = read_index()
    if job['job_id'] in index: return ROOT / index[job['job_id']]['directory']
    return RECOVERY if job['job_id'] == FAILED_JOB else OUT


def verify_checkpoint(job, cfg, directory, config_path, exact_request=True):
    index = read_index()
    if job['job_id'] in index: return historical_verify(job, index[job['job_id']])
    assert Path(directory) == location(job, None)
    return guard.verify_checkpoint(job, cfg, directory, CONFIG)


def save_business(b, grouped, imports):
    global TARGET_CACHE
    content = {}
    for j in grouped[b['business_id']]:
        path = location(j, imports) / 'checkpoints' / (j['job_id'] + '.json')
        if not path.exists(): return None
        cp = frozen.read(path)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW': return None
        content[j['condition']] = cp.get('resolved_output', cp['output'])
    target = frozen.build_target(b, content)
    if TARGET_CACHE is None: TARGET_CACHE = frozen.read(RUN / 'reuse_targets.json')
    reused = TARGET_CACHE
    if b['business_id'] in reused:
        path = ROOT / reused[b['business_id']]; assert frozen.read(path) == target; return path
    path = OUT / 'targets' / (b['business_id'] + '.json')
    if path.exists(): assert frozen.read(path) == target
    else: frozen.immutable(path, target)
    return path


def update_docs(state):
    state['operator_retry_API_attempts'] = len(list((RECOVERY / 'request_intents').glob('*.json')))
    state['initial_reused_complete_businesses'] = 155
    state['initial_reused_components'] = 466
    state['total_new_execution_API_attempts'] = state.get('new_API_attempts', 0) + state['operator_retry_API_attempts']
    frozen.atomic(OUT / 'execution_progress.json', state)
    frozen.atomic(ROOT / 'outputs/stage_g/active_production_execution_pointer.json', {
        'status': state['status'], 'provider': PROVIDER, 'requested_model': MODEL,
        'production_directory': RUN.relative_to(ROOT).as_posix(),
        'progress_path': (OUT / 'execution_progress.json').relative_to(ROOT).as_posix(),
        'maximum_concurrency': 6, 'business_id_guard': 'single_value_enum_v1',
        'execution_entrypoint': SCRIPT.relative_to(ROOT).as_posix(),
        'pause_file': (RUN / 'STOP_REQUESTED.json').relative_to(ROOT).as_posix(),
        'updated_utc': state['updated_utc']})
    p = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'; marker = '**D-042 live production status:**'
    lines = [s for s in p.read_text(encoding='utf-8').splitlines(keepends=True) if not s.startswith(marker)]
    note = (f"{marker} {state['businesses_completed']}/2200 businesses, {state['targets_completed']} targets; "
        f"status={state['status']}; phase={state['phase']}; stop={state.get('stop_reason')}; "
        "maximum6 requests in drained batches after six-call trial PASS; 155 completed businesses/466 components reused. "
        "D043 ID guard and frozen D040 teacher protocol unchanged. One explicitly reviewed API retry for16tj spans; "
        "original failure preserved, no automatic retries. Authoritative pointer: outputs/stage_g/active_production_execution_pointer.json. "
        "TeacherAPI-only content, no held-out targets or LED training; structural checks are not factual certification.\n")
    lines.insert(1, '\n' + note + '\n'); p.write_text(''.join(lines), encoding='utf-8')


def recover_source_once():
    cfg, jobs, businesses, imports = checked_inputs()
    job = next(j for j in jobs if j['job_id'] == FAILED_JOB)
    cp = RECOVERY / 'checkpoints' / (FAILED_JOB + '.json')
    if cp.exists():
        guard.verify_checkpoint(job, cfg, RECOVERY, CONFIG); return
    assert not (RECOVERY / 'request_intents' / cp.name).exists(), 'Unresolved recovery; no automatic resend'
    assert not (RUN / 'STOP_REQUESTED.json').exists()
    assert not os.environ.get(KEY_VARIABLE), 'Private teacher credentials only'
    schema = frozen.read(ROOT / cfg['schema'])[job['task']]
    prompt = (ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
    body = frozen.canonical(guard.payload_for_job(job, cfg, prompt, schema))
    path = RECOVERY / 'raw_requests' / (FAILED_JOB + '.json.gz'); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): assert gzip.decompress(path.read_bytes()) == body
    else:
        with path.open('xb') as stream: stream.write(gzip.compress(body, mtime=0))
    state = {'status': 'RUNNING_REVIEWED_SOURCE_RECOVERY', 'phase': 'single_reviewed_source_retry',
        'businesses_completed': 155, 'targets_completed': 465, 'businesses_planned': 2200,
        'accepted_tasks': 466, 'current_jobs': [FAILED_JOB], 'maximum_concurrency': 6,
        'pid': os.getpid(), 'failed_or_unresolved': [], 'automatic_retries': 0,
        'updated_utc': frozen.now(), 'stop_reason': None}
    frozen.atomic(OUT / 'execution_progress.json', state); update_docs(state)
    base, provider, key = credentials(); assert (base, provider, cfg['requested_model']) == (BASE_URL, PROVIDER, MODEL)
    legacy.call = transport.call; transport.set_context(RECOVERY / 'stream_diagnostics', FAILED_JOB)
    result = legacy.job_result(job, cfg, prompt, guard.schema_for_job(job, schema), base, provider, key, RECOVERY, CONFIG)
    guard.verify_checkpoint(job, cfg, RECOVERY, CONFIG)
    frozen.immutable(RUN / 'source_recovery_report.json', {'status': 'SOURCE_REVIEWED_RETRY_STRUCTURAL_PASS',
        'job_id': FAILED_JOB, 'API_calls': 1, 'seconds': result.get('seconds'),
        'response_id': result.get('response_id'), 'reported_model': result.get('reported_model'),
        'usage': result.get('usage'), 'original_failure_preserved': True, 'visual_components_regenerated': False,
        'same_payload_prompt_model': True, 'output_rewritten': False, 'automatic_retries': 0,
        'independent_factual_certification': False, 'completed_utc': frozen.now()})
    print('Reviewed source-span API recovery PASS; unchanged strict evidence checks', flush=True)


def configure():
    import run_stage_g_production_d042_six_worker_v1 as runner
    def result(job, cfg, prompt, schema, base, provider, key, directory, config_path):
        return legacy.job_result(job, cfg, prompt, guard.schema_for_job(job, schema), base, provider, key, directory, CONFIG)
    runner.CONFIG = CONFIG; runner.FOLDER = RUN; runner.OUT = OUT
    runner.check_inputs = checked_inputs; runner.location = location; runner.verify_checkpoint = verify_checkpoint
    runner.save_completed_business = save_business; runner.job_result = result; runner.payload = guard.payload_for_job
    runner.update_docs = update_docs; runner.runner_lock = parent.restored.shared_lock
    old_run = subprocess.run
    def redirect(command, *args, **kwargs):
        if len(command) > 1 and Path(command[1]).name == 'audit_stage_g_production_d042.py':
            command = [sys.executable, str(SCRIPT), '--audit'] + command[2:]
        return old_run(command, *args, **kwargs)
    runner.subprocess.run = redirect
    return runner


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true'); parser.add_argument('--audit', action='store_true')
    parser.add_argument('--prepare-and-run', action='store_true')
    args = parser.parse_args(); assert sum((args.prepare, args.run, args.audit, args.prepare_and_run)) == 1
    if args.prepare: prepare(); return
    if args.prepare_and_run: prepare()
    runner = configure()
    if args.audit:
        import audit_stage_g_production_d042 as audit
        audit.CONFIG = CONFIG; audit.FOLDER = RUN; audit.OUT = OUT; audit.check_inputs = checked_inputs
        audit.location = location; audit.verify_checkpoint = verify_checkpoint; audit.save_completed_business = save_business
        # Frozen final audit updates its runner-global docs callback at completion.
        import run_stage_g_production_d042 as frozen_runner
        frozen_runner.update_docs = update_docs
        audit.run(smoke=False); return
    lock = parent.restored.shared_lock()
    recover_source_once()
    # Retain one OS lock from the reviewed retry through the whole production run.
    runner.runner_lock = lambda: lock
    runner.main()


if __name__ == '__main__':
    try: main()
    except Exception as error:
        frozen.atomic(RUN / 'runner_failure.json', {'status': 'STOPPED_REQUIRES_REVIEW',
            'error_type': type(error).__name__, 'updated_utc': frozen.now(), 'automatic_retries': 0})
        p = OUT / 'execution_progress.json'
        if p.exists():
            state = frozen.read(p); state.update(status='STOPPED_REQUIRES_REVIEW', current_jobs=[],
                stop_reason='Reviewed recovery or production failed; attempts preserved: ' + type(error).__name__,
                updated_utc=frozen.now())
            frozen.atomic(p, state); update_docs(state)
        print(json.dumps({'status': 'STOPPED_REQUIRES_REVIEW', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
