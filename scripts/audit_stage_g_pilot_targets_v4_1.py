"""Offline integrity/token audit of complete pilot target sets; never calls an API."""
from pathlib import Path
from collections import Counter
import argparse
import json
import math
from tokenizers import Tokenizer
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from run_stage_g_teacher_pilot_v4_1 import validate_output, response_text, payload
from stage_g_canonical_review_ids import resolve_review, citation_diagnostics
from stage_g_visual_target_serialization import visual_claims


def stats(values):
    if not values:
        return {'count': 0}
    values = sorted(values)
    def percentile(p):
        at = (len(values) - 1) * p
        low, high = math.floor(at), math.ceil(at)
        return values[low] + (values[high] - values[low]) * (at - low)
    return {'count': len(values), 'min': values[0], 'mean': sum(values) / len(values),
            'median': percentile(.5), 'p95': percentile(.95), 'max': values[-1]}


def audit(require_complete=True, config='configs/stage_g_teacher_pilot_v4_1_grounding.json'):
    config_path = ROOT / config
    cfg = read(config_path)
    folder = ROOT / cfg['output_directory']
    execution = folder / 'execution_v1'
    run_manifest = read(execution / 'run_manifest.json')
    for field, relative in {
            'config_sha256': config,
            'runner_sha256': 'scripts/run_stage_g_teacher_pilot_v4_1.py',
            'resolver_sha256': 'scripts/stage_g_canonical_review_ids.py',
            'visual_serializer_sha256': 'scripts/stage_g_visual_target_serialization.py',
            'segmenter_sha256': 'scripts/prepare_stage_g_review_evidence_ids.py'}.items():
        if file_sha(ROOT / relative) != run_manifest[field]:
            raise ValueError('Executed implementation/config changed: ' + relative)
    prepared = read(folder / 'preparation_report.json')
    for relative, expected in prepared['input_sha256'].items():
        if file_sha(ROOT / relative) != expected:
            raise ValueError('Source/protocol changed: ' + relative)
    jobs = read(folder / 'pilot_jobs.json')
    if digest(canonical(jobs)) != prepared['jobs_sha256']:
        raise ValueError('Prepared pilot job identity changed')
    businesses = read(folder / 'pilot_businesses.json')
    schema = read(ROOT / cfg['schema'])
    token_manifest = read(ROOT / 'assets/tokenizers/led-base-16384.manifest.json')
    token_folder = ROOT / token_manifest['relative_directory']
    for name, expected in token_manifest['sha256'].items():
        if file_sha(token_folder / name) != expected:
            raise ValueError('Pinned LED tokenizer changed')
    tokenizer = Tokenizer.from_file(str(token_folder / 'tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    led_lengths = {c: [] for c in ['reviews_only', 'unfiltered', 'filtered']}
    words = {c: [] for c in led_lengths}
    rows = []
    models = Counter()
    usage = Counter()
    verdicts = Counter()
    durations = []
    verified_tasks = 0
    all_exact_source_spans = 0
    quote_support_counts = []
    total_visual_duplicates_merged = 0
    empty_visual = Counter()
    artifact_sha = {}
    citation_id_diagnostics = []
    request_hash_checks = 0
    slow_target_checks = 0
    prior_target_text_checks = 0
    slow_tokenizer = None
    if require_complete:
        from transformers import LEDTokenizer
        slow_tokenizer = LEDTokenizer.from_pretrained(str(token_folder), local_files_only=True)
    for business in businesses:
        bid = business['business_id']
        target_path = execution / 'provisional_targets' / (bid + '.json')
        if not target_path.exists():
            continue
        target = read(target_path)
        if target['human_verified']:
            raise ValueError('Pilot target unexpectedly claims full human verification')
        if target['split'] not in {'train', 'validation'} or target['split'] != business['split']:
            raise ValueError('Unexpected target split')
        review = None
        source_span_count = 0
        task_outputs = {}
        for job in [j for j in jobs if j['business_id'] == bid]:
            checkpoint_path = execution / 'checkpoints' / (job['job_id'] + '.json')
            cp = read(checkpoint_path)
            if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                raise ValueError('Target has a failed underlying task')
            validate_output(job, cp['output'], schema[job['task']])
            prompt = (ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
            expected_binding = {'job_sha256': digest(canonical(job)),
                'config_sha256': file_sha(config_path), 'prompt_sha256': digest(prompt.encode()),
                'schema_sha256': digest(canonical(schema[job['task']])),
                'base_url': run_manifest['base_url'], 'provider': run_manifest['provider'],
                'requested_model': cfg['requested_model']}
            if cp['binding'] != expected_binding:
                raise ValueError('Checkpoint prompt/config/source binding changed')
            if cp['api_called']:
                request_sha = digest(canonical(payload(job, cfg, prompt, schema[job['task']])))
                if cp.get('imported_completed_response'):
                    lineage = cp['import_lineage']
                    if request_sha != lineage['identical_API_request_sha256']:
                        raise ValueError('Imported response request differs from current inputs')
                    if file_sha(ROOT / lineage['source_checkpoint']) != lineage['source_checkpoint_sha256']:
                        raise ValueError('Historical parent checkpoint changed')
                    if file_sha(ROOT / lineage['source_raw_response']) != cp['raw_response_sha256']:
                        raise ValueError('Imported raw response differs from its parent')
                else:
                    intent = read(execution / 'request_intents' / (job['job_id'] + '.json'))
                    if request_sha != intent['request_sha256'] or intent['binding'] != expected_binding:
                        raise ValueError('Actual request differs from current inputs')
                request_hash_checks += 1
                raw_path = execution / 'raw_responses' / (job['job_id'] + '.json')
                raw = read(raw_path)
                if file_sha(raw_path) != cp['raw_response_sha256']:
                    raise ValueError('Raw response hash changed')
                if json.loads(response_text(raw)) != cp['output']:
                    raise ValueError('Model output changed after receipt')
                if raw['model'] != cp['reported_model'] or raw['model'] != cfg['requested_model']:
                    raise ValueError('Teacher model mismatch')
                models[raw['model']] += 1
                usage['input_tokens'] += cp['usage'].get('input_tokens', 0)
                usage['output_tokens'] += cp['usage'].get('output_tokens', 0)
                usage['reasoning_tokens'] += cp['usage'].get('output_tokens_details', {}).get('reasoning_tokens', 0)
                if cp.get('seconds') is not None:
                    durations.append(cp['seconds'])
                artifact_sha[raw_path.relative_to(ROOT).as_posix()] = file_sha(raw_path)
            if job['task'] == 'review_consensus':
                citation_id_diagnostics.extend(dict(d, business_id=bid, job_id=job['job_id'])
                                               for d in citation_diagnostics(job, cp['output']))
                resolved, provenance = resolve_review(job, cp['output'])
                if (resolved, provenance) != (cp['resolved_output'], cp['evidence_resolution']):
                    raise ValueError('Source resolution changed')
                all_exact_source_spans += len(provenance)
                source_span_count = len(provenance)
                for claim in resolved['claims']:
                    quote_support_counts.append(len({s['review_id'] for s in claim['support']}))
                review = '[REVIEW_CONSENSUS]\n' + ' '.join(c['text'] for c in resolved['claims'])
            task_outputs[job['condition']] = cp.get('resolved_output', cp['output'])
            verified_tasks += 1
            artifact_sha[checkpoint_path.relative_to(ROOT).as_posix()] = file_sha(checkpoint_path)
        if len(task_outputs) != 3 or target['targets']['reviews_only'] != review:
            raise ValueError('Review consensus/condition task mismatch')
        for condition in ['unfiltered', 'filtered']:
            assessments = task_outputs[condition]['assessments']
            groups = visual_claims(assessments)
            if groups != target['visual_target_claims_with_all_evidence'][condition]:
                raise ValueError('Visual support records changed')
            for assessment in assessments:
                verdicts.update(c['verdict'] for c in assessment['claims'])
            count = sum(len(c['support']) for c in groups)
            total_visual_duplicates_merged += count - len(groups)
            if not groups:
                empty_visual[condition] += 1
            expected = review + '\n\n[VISUAL_EVIDENCE]\n' + ' '.join(c['text'] for c in groups)
            if target['targets'][condition] != expected:
                raise ValueError('Source-grounded visual target serialization changed')
        row = {'business_id': bid, 'split': business['split'],
               'review_claims': len(task_outputs['shared']['claims']),
               'source_spans': source_span_count,
               'human_verified_business': False}
        for condition, text in target['targets'].items():
            token_ids = tokenizer.encode(text, add_special_tokens=True).ids
            if slow_tokenizer is not None:
                if slow_tokenizer.encode(text, add_special_tokens=True, truncation=False) != token_ids:
                    raise ValueError('Slow/native target token IDs differ')
                slow_target_checks += 1
            length = len(token_ids)
            led_lengths[condition].append(length)
            words[condition].append(len(text.split()))
            row[condition + '_LED_tokens'] = length
            row[condition + '_over512_diagnostic'] = length > 512
        rows.append(row)
        artifact_sha[target_path.relative_to(ROOT).as_posix()] = file_sha(target_path)
    complete = len(rows) == len(businesses)
    if require_complete and not complete:
        raise ValueError(f'Pilot incomplete: {len(rows)}/{len(businesses)} businesses; use --partial for progress only')
    feedback_path = execution / 'human_claim_verifications.json'
    feedback = read(feedback_path) if feedback_path.exists() else {'records': []}
    prior_feedback = ROOT / 'outputs/stage_g/teacher_pilot_v3_visual_merge_v1/execution_v1/human_claim_verifications.json'
    if feedback_path.exists():
        if file_sha(feedback_path) != file_sha(prior_feedback):
            raise ValueError('Existing user-confirmed claim record changed')
        artifact_sha[feedback_path.relative_to(ROOT).as_posix()] = file_sha(feedback_path)
    report = {'status': 'PILOT_INTEGRITY_PASS_HUMAN_REVIEW_PENDING' if complete else 'PARTIAL_PROGRESS_NOT_PILOT_COMPLETION',
              'businesses_complete': len(rows), 'businesses_planned': len(businesses),
              'businesses_by_split': dict(Counter(r['split'] for r in rows)),
              'condition_target_texts': len(rows) * 3, 'checkpoints_verified': verified_tasks,
              'reported_models': dict(models), 'teacher_usage_complete_tasks': dict(usage),
              'exact_source_spans_resolved': all_exact_source_spans,
              'distinct_review_supports_per_claim': stats(quote_support_counts),
              'visual_teacher_verdict_counts': dict(verdicts),
              'visual_identical_statements_merged': total_visual_duplicates_merged,
              'empty_visual_targets': dict(empty_visual),
              'LED_target_tokens_with_special_tokens': {k: stats(v) for k,v in led_lengths.items()},
              'target_words': {k: stats(v) for k,v in words.items()},
              'over512_counts_diagnostic_only_not_frozen_target_cap': {k:sum(v>512 for v in lengths) for k,lengths in led_lengths.items()},
              'API_duration_seconds_new_records_only': stats(durations),
              'raw_model_outputs_unchanged': True, 'shared_consensus_identical': True,
              'reconstructed_actual_request_SHA_checks': request_hash_checks,
              'slow_LED_target_token_ID_checks': slow_target_checks,
              'baseline_target_text_identity_required': False,
              'all_source_hashes_unchanged': True, 'teacher_claims_not_rewritten': True,
              'redundant_reported_review_ID_mismatches': citation_id_diagnostics,
              'redundant_reported_review_ID_mismatch_count': len(citation_id_diagnostics),
              'human_confirmed_claims': 0, 'historical_user_confirmed_claims': sum(r['human_verified'] for r in feedback['records']),
              'human_verified_businesses': 0, 'held_out_teacher_targets': 0,
              'full_production_method_frozen': False,
              'note': 'Integrity/source IDs/quote fidelity are not semantic or factual-grounding certification.512is a planning diagnostic only; no target truncation or claim removal.',
              'business_details': rows, 'artifact_sha256': artifact_sha}
    path = execution / ('pilot_integrity_and_tokens_report.json' if complete else 'partial_integrity_and_tokens_report.json')
    if complete:
        immutable(path, report)
    else:
        path.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--config', default='configs/stage_g_teacher_pilot_v4_1_grounding.json')
    args = parser.parse_args()
    result = audit(require_complete=not args.partial, config=args.config)
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_sha256','business_details']}, indent=2))
