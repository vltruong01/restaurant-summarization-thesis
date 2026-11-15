"""Two predeclared review-only cases; no image calls or production targets."""
from pathlib import Path
import argparse
import json
import shutil
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, canonical, digest, immutable
from run_stage_g_teacher_pilot_v3_network_recovery import credentials, job_result


def run():
    config = ROOT / 'configs/stage_g_teacher_pilot_v3.json'
    cfg = read(config)
    root = ROOT / cfg['output_directory']
    prepared = read(root / 'preparation_report.json')
    for relative, expected in prepared['input_sha256'].items():
        if file_sha(ROOT / relative) != expected:
            raise ValueError('Prepared source/protocol changed')
    jobs = read(root / 'pilot_jobs.json')
    if digest(canonical(jobs)) != prepared['jobs_sha256']:
        raise ValueError('Preselected pilot jobs changed')
    selected = read(root / 'pilot_businesses.json')[:2]
    if any(b['split'] != 'train' for b in selected):
        raise ValueError('Unexpected regression cohort')
    out = root / 'review_regression_v1'
    first_job = next(j for j in jobs if j['business_id'] == selected[0]['business_id'] and j['task'] == 'review_consensus')
    for category in ['checkpoints', 'raw_responses']:
        src = root / 'execution_v1' / category / (first_job['job_id'] + '.json')
        dst = out / category / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.read_bytes() != src.read_bytes():
                raise ValueError('Existing copied successful review checkpoint changed')
        else:
            shutil.copyfile(src, dst)
    base, provider, key = credentials()
    immutable(out / 'run_manifest.json', {
        'scope': 'TWO_PREDECLARED_REVIEW_CASES_ONLY_NO_VISUAL_CALLS',
        'businesses': selected, 'config_sha256': file_sha(config),
        'runner_sha256': file_sha(Path(__file__)),
        'task_helper_sha256': file_sha(ROOT / 'scripts/run_stage_g_teacher_pilot_v3_network_recovery.py'),
        'requested_model': cfg['requested_model'], 'provider': provider, 'base_url': base,
        'network_socket_timeout_seconds': 120, 'automatic_retries': 0,
        'first_case_reused_byte_identically': True, 'full_generation': False})
    prompt = (ROOT / cfg['prompts']['review_consensus']).read_text(encoding='utf-8')
    schema = read(ROOT / cfg['schema'])['review_consensus']
    results = []
    for business in selected:
        job = next(j for j in jobs if j['business_id'] == business['business_id'] and j['task'] == 'review_consensus')
        if job['images']:
            raise ValueError('Review-only regression must not send images')
        results.append(job_result(job, cfg, prompt, schema, base, provider, key, out, config))
    report = {'status': 'REVIEW_EVIDENCE_ID_REGRESSION_PASS_HUMAN_GROUNDING_PENDING',
        'businesses': [b['business_id'] for b in selected], 'review_tasks_completed': len(results),
        'first_task_reused': True, 'new_API_calls': 1,
        'requested_model': cfg['requested_model'],
        'reported_models': sorted({r['reported_model'] for r in results}),
        'review_claims': [len(r['output']['claims']) for r in results],
        'source_spans_resolved': [len(r['evidence_resolution']) for r in results],
        'exact_quotes_copied_from_original_source': True, 'claim_text_modified': False,
        'visual_targets_completed': False, 'human_verified': False,
        'full_generation_started': False, 'held_out_teacher_targets': 0}
    immutable(out / 'regression_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-regression', action='store_true')
    args = parser.parse_args()
    if not args.execute_regression:
        parser.error('Explicit execution flag required; no automatic API calls')
    print(json.dumps(run(), indent=2), flush=True)
