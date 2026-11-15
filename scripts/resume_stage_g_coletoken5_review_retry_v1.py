"""Explicit single-job operator recovery; preserve failed response and frozen protocol."""
import argparse
import gzip
import subprocess
import sys
from pathlib import Path
import stage_g_coletoken_production_v1 as ledger
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
import stage_g_teacher_sse_transport_v1 as transport
from stage_g_teacher_credentials import credentials, BASE_URL, PROVIDER, MODEL

JOB = 'train_0mLgF2T8wVDj7NFtoRtpQQ_review_consensus'
RECOVERY = ledger.RUN / 'operator_review_retry_v1'
OUT = RECOVERY / 'execution_v1'
SCRIPT = Path(__file__).resolve()


def preflight():
    ledger.verify_manifest()
    cfg, jobs, businesses, imports = ledger.checked_inputs()
    job = next(j for j in jobs if j['job_id'] == JOB)
    failed = ledger.read(ledger.EXEC / 'checkpoints' / (JOB + '.json'))
    assert failed['status'] == 'FAILED_STOP_NO_AUTOMATIC_RETRY'
    assert failed['local_validation_error'] == 'Teacher returned another business ID'
    assert job['split'] in {'train', 'validation'} and job['task'] == 'review_consensus'
    schema = ledger.read(ledger.ROOT / cfg['schema'])[job['task']]
    prompt = (ledger.ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
    body = ledger.canonical(legacy.payload(job, cfg, prompt, schema))
    assert gzip.decompress((ledger.EXEC / 'raw_requests' / (JOB + '.json.gz')).read_bytes()) == body
    preserved = {p.relative_to(ledger.ROOT).as_posix(): ledger.file_sha(p)
                 for p in ledger.EXEC.rglob('*') if p.is_file() and job['business_id'] in p.name}
    manifest_path = RECOVERY / 'authorization_manifest.json'
    if not manifest_path.exists():
        ledger.immutable(manifest_path, {
            'approved_by': 'user', 'authorization': 'Resume approved production; isolated retry of wrong-business-ID response',
            'job_id': JOB, 'maximum_retry_API_calls': 1, 'automatic_retries': 0,
            'request_sha256': legacy.digest(body), 'config_sha256': ledger.file_sha(ledger.CONFIG),
            'runner_sha256': ledger.file_sha(SCRIPT), 'preserved_artifact_sha256': preserved,
            'original_failure_overwritten': False, 'scientific_settings_changed': False,
            'target_content_origin': 'configured_teacher_API_only', 'created_utc': ledger.now()})
    manifest = ledger.read(manifest_path)
    assert manifest['runner_sha256'] == ledger.file_sha(SCRIPT)
    assert manifest['config_sha256'] == ledger.file_sha(ledger.CONFIG)
    assert manifest['request_sha256'] == legacy.digest(body)
    for rel, sha in manifest['preserved_artifact_sha256'].items():
        assert ledger.file_sha(ledger.ROOT / rel) == sha, rel
    # Audit all accepted native continuation components before any retry spending.
    for j in jobs:
        p = ledger.EXEC / 'checkpoints' / (j['job_id'] + '.json')
        if p.exists() and j['job_id'] != JOB:
            assert ledger.read(p)['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'
            ledger.verify_checkpoint(j, cfg, ledger.EXEC, ledger.CONFIG)
    return cfg, job, schema, prompt, body


def retry():
    from run_stage_g_production_d042 import runner_lock
    lock = runner_lock()
    cfg, job, schema, prompt, body = preflight()
    cp_path = OUT / 'checkpoints' / (JOB + '.json')
    if cp_path.exists():
        ledger.verify_checkpoint(job, cfg, OUT, ledger.CONFIG)
        return
    assert not (OUT / 'request_intents' / (JOB + '.json')).exists(), 'Unresolved retry: do not resend'
    assert not (ledger.RUN / 'STOP_REQUESTED.json').exists()
    base, provider, key = credentials()
    assert (base, provider, cfg['requested_model']) == (BASE_URL, PROVIDER, MODEL)
    p = OUT / 'raw_requests' / (JOB + '.json.gz')
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists(): p.write_bytes(gzip.compress(body, mtime=0))
    assert gzip.decompress(p.read_bytes()) == body
    legacy.call = transport.call
    transport.set_context(OUT / 'stream_diagnostics', JOB)
    cp = legacy.job_result(job, cfg, prompt, schema, base, provider, key, OUT, ledger.CONFIG)
    ledger.verify_checkpoint(job, cfg, OUT, ledger.CONFIG)
    ledger.immutable(RECOVERY / 'recovery_report.json', {
        'status': 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW', 'job_id': JOB,
        'response_id': cp.get('response_id'), 'reported_model': cp.get('reported_model'),
        'seconds': cp.get('seconds'), 'usage': cp.get('usage'), 'API_calls': 1,
        'previous_failed_attempt_preserved': True, 'visual_components_regenerated': False,
        'factual_quality_independently_confirmed': False, 'completed_utc': ledger.now()})


def connect_recovery():
    cfg, job, schema, prompt, body = preflight()
    ledger.verify_checkpoint(job, cfg, OUT, ledger.CONFIG)
    old_location = ledger.location
    ledger.location = lambda j, imports: OUT if j['job_id'] == JOB else old_location(j, imports)
    old_docs = ledger.update_docs

    def update_docs(state):
        state['operator_retry_API_attempts'] = len(list((OUT / 'request_intents').glob('*.json')))
        state['total_continuation_API_attempts'] = state['new_API_attempts'] + state['operator_retry_API_attempts']
        state['operator_recovery_manifest'] = (RECOVERY / 'authorization_manifest.json').relative_to(ledger.ROOT).as_posix()
        ledger.atomic(ledger.EXEC / 'execution_progress.json', state)
        old_docs(state)
        pointer = ledger.read(ledger.POINTER)
        pointer.update(status=state['status'], updated_utc=state['updated_utc'],
                       recovery_checkpoint_directory=OUT.relative_to(ledger.ROOT).as_posix(),
                       recovery_job_id=JOB, execution_entrypoint=SCRIPT.relative_to(ledger.ROOT).as_posix())
        ledger.atomic(ledger.POINTER, pointer)

    ledger.update_docs = update_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(); print('Offline preflight PASS; API calls=0', flush=True); return
    if args.audit:
        connect_recovery()
        import audit_stage_g_coletoken_production5_v1 as audit
        audit.main(); return
    try:
        retry()
        connect_recovery()
        import run_stage_g_coletoken_production5_v1 as runner
        # Keep the inherited full audit while supplying the recovered checkpoint mapping.
        original_run = subprocess.run

        def audit_run(command, *positional, **keyword):
            if len(command) >= 2 and Path(command[1]).name == 'audit_stage_g_coletoken_production5_v1.py':
                command = [sys.executable, str(SCRIPT), '--audit']
            return original_run(command, *positional, **keyword)

        runner.subprocess.run = audit_run
        runner.main()
    except Exception as error:
        ledger.atomic(RECOVERY / 'runner_failure.json', {
            'status': 'STOPPED_REQUIRES_REVIEW', 'error_type': type(error).__name__,
            'utc': ledger.now(), 'automatic_retries': 0})
        state = ledger.read(ledger.EXEC / 'execution_progress.json')
        state.update(status='STOPPED_REQUIRES_REVIEW', current_jobs=[], active_businesses=[],
                     stop_reason='Operator recovery/continuation failure: ' + type(error).__name__, updated_utc=ledger.now())
        ledger.atomic(ledger.EXEC / 'execution_progress.json', state)
        ledger.update_docs(state)
        raise


if __name__ == '__main__': main()
