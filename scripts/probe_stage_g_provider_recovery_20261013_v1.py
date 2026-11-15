"""One real failed-job retry before bulk recovery; private teacher runtime only."""
import gzip
import json
import os
from pathlib import Path

import continue_stage_g_production6_retry_v1 as core
import continue_stage_g_pending_recovery_20261013_v4 as prior
import stage_g_teacher_recovery_wait_v1 as transport

ROOT = core.ROOT
FOLDER = core.RUN / 'provider_recovery_probe_20261013_v1'
SCRIPT = Path(__file__).resolve()


def main():
    # Keep the shared lock alive until the single request has finished.
    lock = core.parent.parent.parent.restored.shared_lock()
    assert not (core.RUN / 'STOP_REQUESTED.json').exists(), 'Respect producer pause'
    state = core.frozen.read(core.OUT / 'execution_progress.json')
    assert state['status'] == 'STOPPED_REQUIRES_REVIEW' and not state['current_jobs']
    assert not (FOLDER / 'authorization.json').exists(), 'Probe is single-use; never resend unknown outcome'
    previous_report = core.frozen.read(prior.OVERLAY / 'recovery_report.json')
    selected = next(x for x in previous_report['tasks']
                    if x['recovery_status'] != 'RECOVERED_STRUCTURAL_PASS')
    jid = selected['job_id']
    # Targeted preflight binds the exact job, protocol, original photos and all
    # old requests. The full cohort preflight is still required before bulk work.
    initial = core.frozen.read(core.frozen.FOLDER / 'preflight.json')['protected_sha256']
    jobs_path = core.frozen.FOLDER / 'production_jobs.json'
    assert core.frozen.file_sha(jobs_path) == initial[jobs_path.relative_to(ROOT).as_posix()]
    cfg = core.frozen.read(core.CONFIG)
    old_auth = core.frozen.read(prior.AUTH)
    assert core.frozen.file_sha(core.CONFIG) == old_auth['frozen_request_config_sha256']
    for rel, expected in initial.items():
        if rel.startswith(('scripts/', 'prompts/', 'schemas/')):
            assert core.frozen.file_sha(ROOT / rel) == expected, rel
    for rel, expected in core.frozen.read(core.RUN / 'manifest.json')['code_sha256'].items():
        assert core.frozen.file_sha(ROOT / rel) == expected, rel
    jobs = core.frozen.read(jobs_path)
    core.load_attempts(jobs)
    job = next(j for j in jobs if j['job_id'] == jid)
    assert job['split'] in ('train', 'validation') and not job['skip_api_empty_captions']
    assert jid not in core.read_index()
    attempts = core.ATTEMPTS[jid]
    assert attempts and all(core.checkpoint(jid, a) is not None for a in attempts)
    assert not any(core.checkpoint(jid, a)['status'] == core.PASS for a in attempts)
    assert prior.guarded_defer_decision(jid, attempts)[0] == 'deferred'
    prompt = (ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
    schema = core.frozen.read(ROOT / cfg['schema'])[job['task']]
    body = core.frozen.canonical(core.guard.payload_for_job(job, cfg, prompt, schema))
    expected_binding = dict(job_sha256=core.frozen.digest(core.frozen.canonical(job)),
        config_sha256=core.frozen.file_sha(core.CONFIG),
        prompt_sha256=core.frozen.digest(prompt.encode()),
        schema_sha256=core.frozen.digest(core.frozen.canonical(core.guard.schema_for_job(job, schema))),
        base_url=core.BASE_URL, provider=core.PROVIDER, requested_model=core.MODEL)
    protected = {}
    for attempt in attempts:
        assert core.checkpoint(jid, attempt)['binding'] == expected_binding
        old_request = attempt['directory'] / 'raw_requests' / (jid + '.json.gz')
        assert gzip.decompress(old_request.read_bytes()) == body, 'Retry request changed'
        for sub in ('checkpoints', 'raw_requests', 'raw_responses', 'request_intents', 'stream_diagnostics'):
            for path in (attempt['directory'] / sub).glob(jid + '.*'):
                protected[path.relative_to(ROOT).as_posix()] = core.frozen.file_sha(path)
    number = sum(not a['historical'] for a in attempts) + 1
    directory = core.OUT / 'attempts' / f'attempt_{number:02d}'
    assert not (directory / 'request_intents' / (jid + '.json')).exists()
    assert not (directory / 'checkpoints' / (jid + '.json')).exists()
    assert not os.environ.get(core.KEY_VARIABLE), 'Private teacher credential store only'
    base, provider, key = core.credentials()
    assert (base, provider, cfg['requested_model']) == (core.BASE_URL, core.PROVIDER, core.MODEL)
    core.frozen.immutable(FOLDER / 'authorization.json', {
        'authorization': 'Researcher requested fixing failures and rerunning on 2026-10-13',
        'created_utc': core.frozen.now(), 'job_id': jid, 'maximum_API_calls': 1,
        'targeted_preflight': 'PASS', 'full_cohort_preflight_required_before_bulk': True,
        'request_sha256': core.frozen.digest(body), 'request_payload_changed': False,
        'preserved_attempt_sha256': protected, 'runner_sha256': core.frozen.file_sha(SCRIPT),
        'attempt_directory': directory.relative_to(ROOT).as_posix(),
        'Codex_configuration_or_auth_access': False,
    })
    archive = directory / 'raw_requests' / (jid + '.json.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as out:
        out.write(gzip.compress(body, mtime=0))
    core.frozen.atomic(FOLDER / 'status.json', {
        'status': 'SINGLE_GENERATION_PROBE_RUNNING', 'updated_utc': core.frozen.now(),
        'pid': os.getpid(), 'job_id': jid, 'maximum_concurrency': 1,
    })
    print(json.dumps({'preflight': 'TARGETED_PASS', 'job_id': jid, 'API_calls_planned': 1}), flush=True)
    transport.EXTENDED_JOBS = frozenset([jid])
    core.transport = transport
    core.legacy.call = transport.call
    result = core.worker(job, cfg, prompt, schema, base, provider, key, directory)
    cp = core.frozen.read(directory / 'checkpoints' / (jid + '.json'))
    if result['passed']:
        core.guard.verify_checkpoint(job, cfg, directory, core.CONFIG)
    for rel, expected in protected.items():
        assert core.frozen.file_sha(ROOT / rel) == expected, rel
    report = dict(result, status='PROBE_STRUCTURAL_PASS' if result['passed'] else 'PROBE_FAILED_NO_BULK_CALLS',
        updated_utc=core.frozen.now(), http_status=cp.get('http_status'),
        old_attempts_unchanged=True, request_payload_unchanged=True,
        strict_schema_and_evidence_validation_unchanged=True,
        factual_quality_human_certified=False, API_calls=1,
        raw_output_and_checkpoint_preserved=True, Codex_configuration_or_auth_access=False)
    core.frozen.immutable(FOLDER / 'report.json', report)
    core.frozen.atomic(FOLDER / 'status.json', report)
    with (ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md').open('a', encoding='utf-8') as out:
        out.write('\n\n### D-042 provider availability generation probe (2026-10-13, v1)\n'
            'After the v4 recovery stopped on repeated provider HTTP503, the researcher authorized another recovery. '
            'One confirmed failed job was retried using its byte-identical frozen request after targeted input/photo/protocol '
            'preflight, under the shared producer lock. Existing target text and old attempts were not changed. '
            'Result: ' + report['status'] + '; HTTP status: ' + str(report['http_status']) + '. '
            'No bulk requests are made by this probe. Full cohort preflight remains required before bulk recovery. '
            'D040/D043 protocol, provider/model/settings, Stage F and budgets remain unchanged; no Codex config/auth '
            'access, test targets or LED training. Report: ' + (FOLDER / 'report.json').relative_to(ROOT).as_posix() + '.\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        core.frozen.atomic(FOLDER / 'failure.json', {'status': 'PROBE_STOPPED_REQUIRES_REVIEW',
            'updated_utc': core.frozen.now(), 'error_type': type(error).__name__})
        print(json.dumps({'status': 'PROBE_STOPPED_REQUIRES_REVIEW', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
