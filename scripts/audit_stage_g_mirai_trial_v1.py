"""Offline diagnostic audit only; never relabel, repair, import or resend trial outputs."""
import gzip
import json
from collections import Counter
from run_stage_g_mirai_concurrency10_trial_v1 import (
    ROOT, CONFIG, TRIAL, ATTEMPT, read, immutable, file_sha, canonical, payload,
    validate_output, response_text, preservation_check, stats,
)
from prepare_stage_g_teacher_pilot import digest
from stage_g_teacher_credentials import key_file


def main():
    report = read(TRIAL / 'report.json')
    cfg = read(CONFIG); schema = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    jobs = {j['job_id']: j for j in read(TRIAL.parent / 'production_jobs.json')}
    manifest = read(TRIAL / 'manifest.json')
    cases = []; counts = Counter(); latency = []; token_rates = []
    for jid in manifest['job_ids']:
        j = jobs[jid]
        request_bytes = gzip.decompress((ATTEMPT / 'raw_requests' / (jid + '.json.gz')).read_bytes())
        assert request_bytes == canonical(payload(j, cfg, prompts[j['task']], schema[j['task']]))
        assert read(ATTEMPT / 'request_intents' / (jid + '.json'))['request_sha256'] == digest(request_bytes)
        cp = read(ATTEMPT / 'checkpoints' / (jid + '.json'))
        diagnostic = read(ATTEMPT / 'stream_diagnostics' / (jid + '.json'))
        p = ATTEMPT / 'raw_responses' / (jid + '.json')
        case = {'job_id': jid, 'task': j['task'], 'business_id': j['business_id'],
                'checkpoint_status': cp['status'], 'http_status': diagnostic.get('http_status'),
                'seconds': diagnostic['elapsed_seconds'], 'original_checkpoint_error': cp.get('local_validation_error'),
                'raw_completed_response_present': p.exists(), 'target_imported': False}
        if p.exists():
            raw = read(p); counts['complete_raw_responses'] += raw.get('status') == 'completed'
            case.update(raw_response_sha256=file_sha(p), requested_model=cfg['requested_model'],
                        returned_model_present='model' in raw, returned_model=raw.get('model'),
                        returned_reasoning=raw.get('reasoning'), returned_service_tier=raw.get('service_tier'),
                        metadata_top_level_keys=sorted(raw), response_id=raw.get('id'),
                        usage=raw.get('usage'))
            if not raw.get('model'): counts['missing_returned_model'] += 1
            if raw.get('model') and raw.get('model') != cfg['requested_model']: counts['different_returned_model'] += 1
            try:
                output = json.loads(response_text(raw))
                validation = validate_output(j, output, schema[j['task']])
                case['offline_schema_source_validation'] = 'PASS'
                case['offline_validation_details'] = validation
                counts['offline_schema_source_pass'] += 1
            except Exception as error:
                case['offline_schema_source_validation'] = 'FAIL'
                case['offline_validation_error_type'] = type(error).__name__
                case['offline_validation_error'] = str(error)[:2500]
                counts['offline_schema_source_fail'] += 1
            latency.append(diagnostic['elapsed_seconds'])
            u = raw.get('usage') or {}
            if u.get('output_tokens') is not None:
                # Includes reasoning/setup/queue time: NOT a streaming decode-speed measurement.
                token_rates.append(u['output_tokens'] / diagnostic['elapsed_seconds'])
        cases.append(case)
    preservation_check()
    counts['paid_attempts'] = report['paid_request_intents']
    counts['full_protocol_technical_pass'] = report['passed']
    result = {'status': 'OFFLINE_DIAGNOSTIC_ONLY_NO_PRODUCTION_ADOPTION', 'counts': dict(counts),
              'batch_wall_seconds': report['batch_wall_seconds'],
              'client_transport_overlap_peak': report['observed_client_transport_overlap_peak'],
              'complete_API_response_rate_per_minute': 60 * counts['complete_raw_responses'] / report['batch_wall_seconds'],
              'complete_response_latency_seconds': stats(latency),
              'seconds_by_task': report['seconds_by_task'],
              'total_output_tokens_per_request_wall_second': stats(token_rates),
              'token_rate_limitation': 'End-to-end request output-token/wall-second ratio, including reasoning and queue/setup; not pure decode speed.',
              'model_provenance_supported_by_request_only_if_response_model_missing': True,
              'requested_medium_reasoning_or_default_tier_not_confirmed_when_response_metadata_missing': True,
              'original_failed_checkpoints_unchanged': True, 'human_factual_verification': False,
              'no_target_changes_or_imports': True, 'cases': cases}
    key = key_file().with_name('mirai_api_key.txt').read_text(encoding='utf-8-sig').strip()
    assert key not in json.dumps(result)
    immutable(TRIAL / 'offline_diagnostic_audit.json', result)
    text = ("# Mirai teacher-only ten-request trial\n\n"
            f"API returned {counts['complete_raw_responses']}/10 complete responses in {report['batch_wall_seconds']:.3f}s, "
            f"with peak client transport overlap {report['observed_client_transport_overlap_peak']}. "
            f"Frozen full-protocol validation PASS: {report['passed']}/10.\n\n"
            f"Offline content schema/source validation (diagnostic only): {counts['offline_schema_source_pass']} PASS, "
            f"{counts['offline_schema_source_fail']} FAIL. Returned `model` missing: {counts['missing_returned_model']}; "
            f"different returned model: {counts['different_returned_model']}. The request used `gpt-6.1-sol`, "
            "but missing response model metadata cannot independently confirm the served model. "
            "No metadata fields were fabricated, failed checkpoints changed, targets repaired, or production outputs imported.\n\n"
            "Latency by task:\n\n| Task | Count | Min seconds | Mean seconds | Max seconds |\n"
            "|---|---:|---:|---:|---:|\n")
    for task, s in report['seconds_by_task'].items():
        text += f"| {task} | {s['n']} | {s['min']:.3f} | {s['mean']:.3f} | {s['max']:.3f} |\n"
    text += ("\nAll previous Coletoken and failed key-trial artifacts remain byte-identical. "
             "Frozen production stays paused at 131/2200 completed businesses (393 targets). "
             "Mirai results are isolated; this one batch does not establish sustained concurrency stability, backend equivalence or human factual quality. "
             "Provider/model provenance must be resolved and any production provider amendment reviewed before integration. "
             "No additional API calls were made by this audit. Codex configuration, authentication, model catalog and keys were not accessed or modified.\n")
    with (TRIAL / 'TRIAL_REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    note = (f"**D-042 Mirai trial offline interpretation:** {counts['complete_raw_responses']}/10 complete API responses, "
            f"{report['passed']}/10 frozen full-protocol PASS; {counts['missing_returned_model']} responses missing model metadata. "
            f"Offline schema/source diagnostic: {counts['offline_schema_source_pass']} PASS, {counts['offline_schema_source_fail']} FAIL; "
            "diagnostic PASS does not change failed checkpoints or certify the actual backend/model or factual grounding. "
            "Provider trial remains isolated, production unchanged at131/2200 businesses. "
            "See outputs/stage_g/teacher_production_d042_v1/mirai_concurrency10_trial_v1/TRIAL_REPORT.md and offline_diagnostic_audit.json.\n")
    for rel in ('thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md'):
        p = ROOT / rel; first, rest = p.read_text(encoding='utf-8').split('\n', 1)
        p.write_text(first + '\n\n' + note + '\n' + rest, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__': main()
