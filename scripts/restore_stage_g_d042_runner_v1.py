"""Restore unchanged D042 four-worker runner/transport while reusing later valid outputs."""
import argparse
import gzip
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import stage_g_coletoken_production_v1 as previous
import stage_g_production_d042 as frozen

ROOT = frozen.ROOT
RUN = frozen.FOLDER / 'restored_d042_runner_v1'
OUT = RUN / 'execution_v1'
SCRIPT = Path(__file__).resolve()
MANIFEST = RUN / 'restoration_manifest.json'


def shared_lock():
    p = previous.ORIGINAL_EXEC / 'runner.lock'
    h = p.open('a+b')
    if h.tell() == 0: h.write(b'0'); h.flush()
    h.seek(0)
    import msvcrt
    msvcrt.locking(h.fileno(), msvcrt.LK_NBLCK, 1)
    return h


def prepare():
    lock = shared_lock()
    previous.verify_manifest()
    state = frozen.read(previous.EXEC / 'execution_progress.json')
    assert not state['current_jobs'] and state['status'] == 'STOPPED_BY_USER'
    cfg, jobs, businesses, imports = previous.checked_inputs()
    index = {}; hashes = {}; groups = defaultdict(list)
    for j in jobs:
        groups[j['business_id']].append(j['job_id'])
        origin = previous.location(j, imports)
        p = origin / 'checkpoints' / (j['job_id'] + '.json')
        if not p.exists():
            assert not (origin / 'request_intents' / p.name).exists(), 'Unresolved in-flight request'
            continue
        cp = frozen.read(p)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            assert j['job_id'] == 'train_0mLgF2T8wVDj7NFtoRtpQQ_review_consensus'
            assert cp['local_validation_error'] == 'Teacher returned another business ID'
            continue
        config = frozen.PILOT_CONFIG if j['job_id'] in imports else frozen.CONFIG
        frozen.verify_checkpoint(j, cfg, origin, config)
        index[j['job_id']] = origin.relative_to(ROOT).as_posix()
        paths = [p]
        if cp.get('api_called'):
            paths += [origin / 'raw_responses' / p.name, origin / 'request_intents' / p.name,
                      origin / 'raw_requests' / (j['job_id'] + '.json.gz')]
        for item in paths: hashes[item.relative_to(ROOT).as_posix()] = frozen.file_sha(item)
    for directory in (previous.EXEC, previous.RUN / 'operator_review_retry_v1'):
        for p in directory.rglob('*'):
            if p.is_file() and ('0mLgF2T8wVDj7NFtoRtpQQ' in p.name or directory.name == 'operator_review_retry_v1'):
                hashes[p.relative_to(ROOT).as_posix()] = frozen.file_sha(p)
    completed = sum(all(jid in index for jid in js) for js in groups.values())
    assert completed == 145 and len(index) == 437
    assert not MANIFEST.exists()
    frozen.immutable(RUN / 'reuse_index.json', index)
    for name in ('authorization.json', 'small_batch_manifest.json', 'small_batch_validation.json'):
        source = frozen.FOLDER / name
        destination = RUN / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    frozen.immutable(MANIFEST, {
        'approved_by': 'user', 'authorization': 'Restore previously working D042 runner',
        'runner': 'scripts/run_stage_g_production_d042.py',
        'transport': 'scripts/stage_g_teacher_stream_diagnostics_v1.py',
        'maximum_concurrency': 4, 'scheduler': 'original_D042_drained_four_request_batches',
        'teacher_model': 'gpt-6.1-sol', 'provider': 'coletoken', 'automatic_retries': 0,
        'initial_complete_businesses': completed, 'reused_components': len(index),
        'all_2200_businesses_remain_scheduled': True, 'deferred_business_exclusion_removed': True,
        'explicit_new_attempt_for_failed_job': 'train_0mLgF2T8wVDj7NFtoRtpQQ_review_consensus',
        'original_failed_attempts_preserved': True, 'scientific_settings_changed': False,
        'Codex_configuration_or_auth_access': False, 'config_sha256': frozen.file_sha(frozen.CONFIG),
        'wrapper_sha256': frozen.file_sha(SCRIPT), 'reuse_index_sha256': frozen.file_sha(RUN / 'reuse_index.json'),
        'preserved_artifact_sha256': hashes, 'prepared_utc': frozen.now()})
    print({'offline_preflight': 'PASS', 'complete_businesses': completed, 'reused_components': len(index),
           'restored_maximum_concurrency': 4, 'API_calls': 0}, flush=True)


def checked_inputs():
    previous.verify_manifest()
    manifest = frozen.read(MANIFEST)
    assert frozen.file_sha(SCRIPT) == manifest['wrapper_sha256']
    assert frozen.file_sha(frozen.CONFIG) == manifest['config_sha256']
    assert frozen.file_sha(RUN / 'reuse_index.json') == manifest['reuse_index_sha256']
    for rel, sha in manifest['preserved_artifact_sha256'].items():
        assert frozen.file_sha(ROOT / rel) == sha, rel
    return previous.checked_inputs()


def configure():
    import run_stage_g_production_d042 as runner
    index = frozen.read(RUN / 'reuse_index.json')
    def location(job, imports):
        return ROOT / index[job['job_id']] if job['job_id'] in index else OUT

    def save_business(b, grouped, imports):
        content = {}
        for j in grouped[b['business_id']]:
            p = location(j, imports) / 'checkpoints' / (j['job_id'] + '.json')
            if not p.exists(): return None
            cp = frozen.read(p)
            if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW': return None
            content[j['condition']] = cp.get('resolved_output', cp['output'])
        target = frozen.build_target(b, content)
        for directory in (frozen.PILOT_OUT / 'provisional_targets', previous.ORIGINAL_EXEC / 'targets', previous.EXEC / 'targets'):
            p = directory / (b['business_id'] + '.json')
            if p.exists():
                assert frozen.read(p) == target
                return p
        p = OUT / 'targets' / (b['business_id'] + '.json')
        frozen.immutable(p, target)
        return p

    def docs(state):
        frozen.atomic(previous.POINTER, {
            'status': state['status'], 'provider': 'coletoken', 'requested_model': 'gpt-6.1-sol',
            'production_directory': RUN.relative_to(ROOT).as_posix(),
            'progress_path': (OUT / 'execution_progress.json').relative_to(ROOT).as_posix(),
            'maximum_concurrency': 4, 'restored_original_D042_runner': True,
            'restored_original_D042_transport': True, 'deferred_businesses': [],
            'execution_entrypoint': SCRIPT.relative_to(ROOT).as_posix(), 'updated_utc': state['updated_utc'],
            'pause_file': (RUN / 'STOP_REQUESTED.json').relative_to(ROOT).as_posix()})
        p = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
        marker = '**D-042 live production status:**'
        lines = [line for line in p.read_text(encoding='utf-8').splitlines(keepends=True) if not line.startswith(marker)]
        note = (f"{marker} {state['businesses_completed']}/2200 complete businesses, {state['targets_completed']} targets; "
                f"status={state['status']}; original D042 runner and Coletoken transport restored, maximum4 requests in drained batches. "
                "145 businesses/437 components reused; all2200 businesses scheduled, no deferred exclusion. "
                "Failed business-ID outputs are preserved and not manually corrected. "
                "Authoritative progress comes from outputs/stage_g/active_production_execution_pointer.json. "
                "Frozen protocol/source unchanged; teacher-API-only content; no held-out targets or LED training.\n")
        lines.insert(1, '\n'+note+'\n'); p.write_text(''.join(lines), encoding='utf-8')

    runner.FOLDER = RUN; runner.OUT = OUT
    runner.check_inputs = checked_inputs; runner.location = location
    runner.save_completed_business = save_business; runner.update_docs = docs; runner.runner_lock = shared_lock
    original_subprocess = subprocess.run
    def redirect_audit(command, *args, **kwargs):
        if len(command) > 1 and Path(command[1]).name == 'audit_stage_g_production_d042.py':
            command = [sys.executable, str(SCRIPT), '--audit'] + command[2:]
        return original_subprocess(command, *args, **kwargs)
    runner.subprocess.run = redirect_audit
    return runner, location, save_business


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepare', action='store_true'); p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    if args.prepare: prepare(); return
    runner, location, save_business = configure()
    try:
        if args.audit:
            import audit_stage_g_production_d042 as audit
            audit.FOLDER = RUN; audit.OUT = OUT; audit.check_inputs = checked_inputs
            audit.location = location; audit.save_completed_business = save_business
            audit.run(smoke=False)
        else: runner.main()
    except Exception as error:
        frozen.atomic(RUN / 'runner_failure.json', {'status':'STOPPED_REQUIRES_REVIEW',
            'error_type':type(error).__name__, 'updated_utc':frozen.now(), 'automatic_retries':0})
        if (OUT / 'execution_progress.json').exists():
            state=frozen.read(OUT / 'execution_progress.json')
            state.update(status='STOPPED_REQUIRES_REVIEW',current_jobs=[],updated_utc=frozen.now(),
                         stop_reason='Restored runner failed: '+type(error).__name__)
            frozen.atomic(OUT / 'execution_progress.json',state); runner.update_docs(state)
        raise


if __name__ == '__main__': main()
