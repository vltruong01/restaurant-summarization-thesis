"""Explicit, bounded teacher pilot; resumable checkpoints, never full production."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import json
import os
import re
import time
import tomllib
import urllib.request
import urllib.error

from prepare_stage_g_teacher_pilot import (
    ROOT, CONFIG, read, digest, canonical, file_sha, local_path, immutable,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def schema_check(value, schema, location='output'):
    types = {'object': dict, 'array': list, 'string': str}
    if not isinstance(value, types[schema['type']]):
        raise ValueError(f'{location}: wrong JSON type')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{location}: invalid enum value')
    if isinstance(value, dict):
        if set(schema.get('required', [])) - set(value):
            raise ValueError(f'{location}: missing required fields')
        if schema.get('additionalProperties') is False and set(value) - set(schema['properties']):
            raise ValueError(f'{location}: unexpected fields')
        for key, item in value.items():
            schema_check(item, schema['properties'][key], location + '.' + key)
    if isinstance(value, list):
        for i, item in enumerate(value):
            schema_check(item, schema['items'], f'{location}[{i}]')


def validate_output(job, output, schema):
    schema_check(output, schema)
    if output['business_id'] != job['business_id']:
        raise ValueError('Teacher returned another business ID')
    all_text = []
    if job['task'] == 'review_consensus':
        reviews = {r['review_marker']: r for r in job['evidence']['reviews']}
        for claim in output['claims']:
            ids = set()
            for support in claim['support']:
                review = reviews.get(support['review_marker'])
                if review is None or review['review_id'] != support['review_id']:
                    raise ValueError('Invalid REV marker/original review ID mapping')
                if not support['quote'].strip() or support['quote'] not in review['text']:
                    raise ValueError('Review support quote is not verbatim source evidence')
                ids.add(support['review_id'])
            if len(ids) < 2:
                raise ValueError('Recurring claim violates proposed two-review pilot support guard')
            all_text.append(claim['text'])
    else:
        if output['condition'] != job['condition']:
            raise ValueError('Teacher returned another caption condition')
        captions = job['evidence']['captions']
        if len(output['assessments']) != len(captions):
            raise ValueError('Every provided caption must have an assessment')
        for assessment, caption in zip(output['assessments'], captions):
            if (assessment['caption_id'], assessment['photo_id']) != (caption['caption_id'], caption['photo_id']):
                raise ValueError('Caption/photo ID or order mismatch')
            for claim in assessment['claims']:
                if not claim['caption_quote'].strip() or claim['caption_quote'] not in caption['caption']:
                    raise ValueError('Visual support quote is not verbatim caption evidence')
                if claim['verdict'] == 'verified':
                    all_text.append(claim['text'])
    for text in all_text:
        if not text.strip():
            raise ValueError('Empty claim text')
        if re.search(r'\[(?:REV|IMG_CAP|PHOTO)_\d+\]', text):
            raise ValueError('Evidence marker leaked into LED target text')
    normalized = [' '.join(t.lower().split()) for t in all_text]
    if len(normalized) != len(set(normalized)):
        raise ValueError('Repeated identical target claim')
    return {'structural_and_evidence_binding_valid': True,
            'semantic_caption_support_human_verified': False,
            'photo_grounding_human_verified': False,
            'review_consensus_grounding_human_verified': False,
            'empty_review_claims_flag': job['task'] == 'review_consensus' and not all_text}


def credentials():
    home = Path(os.environ.get('CODEX_HOME') or (Path.home() / '.codex'))
    cfg = tomllib.loads((home / 'config.toml').read_text(encoding='utf-8'))
    provider = cfg.get('model_provider')
    base = cfg.get('model_providers', {}).get(provider, {}).get('base_url', '').rstrip('/')
    if base != 'https://www.coletoken.de/v1' or cfg.get('model') != 'gpt-6.1-sol':
        raise ValueError('Credential destination/model differs from researcher-authorized setup')
    auth = read(home / 'auth.json')
    key = auth.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('No API key in authorized personal credential store')
    return base, provider, key


def payload(job, cfg, prompt, schema):
    # Exact block remains in the offline witness; avoid sending each review twice.
    evidence = {k: v for k, v in job['evidence'].items() if k != 'exact_review_block'}
    content = [{'type': 'input_text', 'text': json.dumps(evidence, ensure_ascii=False)}]
    for image in job['images']:
        path = local_path(image['relative_path'])
        if file_sha(path) != image['raw_sha256']:
            raise ValueError('Original photo changed before upload')
        content.append({'type': 'input_text', 'text': f"[{image['photo_marker']}] photo_id={image['photo_id']}"})
        # Original bytes, no resizing, no synthetic replacement, detail omitted (API default).
        content.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' +
                        base64.b64encode(path.read_bytes()).decode('ascii')})
    generation = cfg['generation_proposal']
    return {'model': cfg['requested_model'], 'reasoning': {'effort': generation['reasoning_effort']},
            'store': generation['store'], 'stream': generation['stream'],
            'max_output_tokens': generation['max_output_tokens'],
            'input': [{'role': 'developer', 'content': prompt}, {'role': 'user', 'content': content}],
            'text': {'format': {'type': 'json_schema', 'name': 'stage_g_' + job['task'],
                                'strict': True, 'schema': schema}}}


def call(base, key, body):
    req = urllib.request.Request(base + '/responses', data=canonical(body), method='POST',
                                 headers={'Authorization': 'Bearer ' + key,
                                          'Content-Type': 'application/json',
                                          'Accept': 'text/event-stream',
                                          'User-Agent': 'codex-cli/0.149.0'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=45) as stream:
        for line in stream:
            if time.monotonic() - started > 180:
                raise TimeoutError('Pilot request exceeded wall-clock deadline')
            line = line.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            event = json.loads(data)
            kind = event.get('type')
            if kind == 'response.completed':
                return event['response']
            if kind in {'error', 'response.failed', 'response.incomplete'}:
                raise RuntimeError('Response failed or incomplete; no automatic retry')
    raise RuntimeError('Stream ended without a completed response')


def response_text(response):
    text = []
    for item in response.get('output', []):
        if item.get('type') != 'message':
            continue
        for part in item.get('content', []):
            if part.get('type') == 'refusal':
                raise ValueError('Teacher refusal; no automatic prompt substitution')
            if part.get('type') == 'output_text':
                text.append(part['text'])
    if not text:
        raise ValueError('Completed response has no text output')
    return ''.join(text)


def job_result(job, cfg, prompt, schema, base, provider, key, out, config_path=CONFIG):
    task_path = out / 'checkpoints' / (job['job_id'] + '.json')
    binding = {'job_sha256': digest(canonical(job)), 'config_sha256': file_sha(config_path),
               'prompt_sha256': digest(prompt.encode()), 'schema_sha256': digest(canonical(schema)),
               'base_url': base, 'provider': provider, 'requested_model': cfg['requested_model']}
    if task_path.exists():
        checkpoint = read(task_path)
        if checkpoint['binding'] != binding:
            raise ValueError('Checkpoint input/provider/prompt/settings changed; new version required')
        if checkpoint['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            raise ValueError('Existing task failed/started; explicit reviewed recovery required, no automatic retry')
        validate_output(job, checkpoint['output'], schema)
        print('Reused checkpoint: ' + job['job_id'], flush=True)
        return checkpoint
    if job.get('skip_api_empty_captions'):
        output = {'business_id': job['business_id'], 'condition': job['condition'], 'assessments': []}
        result = {'status': 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW', 'binding': binding,
                  'api_called': False, 'creation': 'deterministic_empty_visual_condition',
                  'output': output, 'validation': validate_output(job, output, schema), 'usage': None}
        immutable(task_path, result)
        return result
    body = payload(job, cfg, prompt, schema)
    intent_path = out / 'request_intents' / (job['job_id'] + '.json')
    # Never resend a task with an uncertain outcome after a crash/network error.
    if intent_path.exists():
        raise ValueError('Request intent exists without a validated checkpoint; review before retry')
    immutable(intent_path, {'status': 'REQUEST_STARTED', 'started_utc': now(), 'binding': binding,
                            'request_sha256': digest(canonical(body)),
                            'image_detail': 'omitted_API_default', 'request_images': job['images']})
    print('Calling ' + cfg['requested_model'] + ': ' + job['job_id'], flush=True)
    started = time.monotonic()
    try:
        response = call(base, key, body)
        # Preserve the returned response before schema validation, including invalid outputs.
        raw_path = out / 'raw_responses' / (job['job_id'] + '.json')
        immutable(raw_path, response)
        if response.get('model') != cfg['requested_model']:
            raise ValueError('Reported model differs from requested alias; review provider provenance')
        if response.get('status') != 'completed':
            raise ValueError('Teacher response is not complete')
        output = json.loads(response_text(response))
        validation = validate_output(job, output, schema)
        result = {'status': 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW', 'binding': binding,
                  'api_called': True, 'completed_utc': now(), 'seconds': round(time.monotonic() - started, 3),
                  'reported_model': response.get('model'), 'response_id': response.get('id'),
                  'raw_response_sha256': file_sha(raw_path), 'output': output,
                  'validation': validation, 'usage': response.get('usage')}
        immutable(task_path, result)
        print('Structural/evidence-binding PASS: ' + job['job_id'], flush=True)
        return result
    except Exception as error:
        # Never persist HTTP response bodies or credentials in failure messages.
        failure = {'status': 'FAILED_STOP_NO_AUTOMATIC_RETRY', 'binding': binding,
                   'error_type': type(error).__name__, 'failed_utc': now(),
                   'http_status': error.code if isinstance(error, urllib.error.HTTPError) else None}
        if isinstance(error, ValueError):
            failure['local_validation_error'] = str(error)
        immutable(task_path, failure)
        raise RuntimeError('Teacher task failed; see checkpoint diagnostics (credentials suppressed).') from None


def run(limit, config_path=CONFIG):
    cfg = read(config_path)
    out_root = local_path(cfg['output_directory'])
    prepared = read(out_root / 'preparation_report.json')
    for relative, expected in prepared['input_sha256'].items():
        if file_sha(local_path(relative)) != expected:
            raise ValueError('Prepared protocol/source binding changed before execution')
    jobs = read(out_root / 'pilot_jobs.json')
    if digest(canonical(jobs)) != prepared['jobs_sha256']:
        raise ValueError('Predeclared pilot jobs changed')
    businesses = read(out_root / 'pilot_businesses.json')
    if limit < 1 or limit > len(businesses):
        raise ValueError('Execution is limited to the predeclared pilot, never all production')
    selected = businesses[:limit]
    if any(x['split'] not in {'train', 'validation'} for x in selected):
        raise ValueError('Held-out teacher execution forbidden')
    base, provider, key = credentials()
    schemas = read(local_path(cfg['schema']))
    prompts = {task: local_path(path).read_text(encoding='utf-8') for task, path in cfg['prompts'].items()}
    out = out_root / 'execution_v1'
    immutable(out / 'run_manifest.json', {
        'status': 'BOUNDED_PILOT_NOT_PRODUCTION_FREEZE', 'protocol_version': cfg['protocol_version'],
        'config_sha256': file_sha(config_path), 'runner_sha256': file_sha(Path(__file__)),
        'prepared_jobs_sha256': prepared['jobs_sha256'], 'provider': provider, 'base_url': base,
        'requested_model': cfg['requested_model'], 'image_detail': 'omitted_API_default',
        'automatic_retries': 0, 'original_image_bytes_preserved': True,
        'full_generation_authorized': False, 'human_grounding_verification_required': True})
    results = []
    for business in selected:
        bid = business['business_id']
        tasks = [j for j in jobs if j['business_id'] == bid and j['split'] == business['split']]
        if len(tasks) != 3:
            raise ValueError('Expected shared review and both visual tasks')
        outputs = {}
        for job in tasks:
            task = job['task']
            result = job_result(job, cfg, prompts[task], schemas[task], base, provider, key, out, config_path)
            results.append(result)
            outputs[job['condition']] = result['output']
        consensus = ' '.join(c['text'] for c in outputs['shared']['claims'])
        targets = {'reviews_only': '[REVIEW_CONSENSUS]\n' + consensus}
        for condition in ['unfiltered', 'filtered']:
            visual = ' '.join(c['text'] for a in outputs[condition]['assessments']
                              for c in a['claims'] if c['verdict'] == 'verified')
            targets[condition] = targets['reviews_only'] + '\n\n[VISUAL_EVIDENCE]\n' + visual
            if targets[condition].split('\n\n[VISUAL_EVIDENCE]\n')[0] != targets['reviews_only']:
                raise ValueError('Shared consensus differs across conditions')
        immutable(out / 'provisional_targets' / (bid + '.json'), {
            'status': 'PROVISIONAL_PILOT_TARGETS_NOT_HUMAN_VERIFIED_NOT_PRODUCTION',
            'business_id': bid, 'split': business['split'], 'targets': targets,
            'evidence_records': outputs, 'human_verified': False,
            'empty_review_consensus_flag': not consensus,
            'review_IDs_text_order_shared': True,
            'zero_filtered_edge_case': business['zero_filtered_edge_case']})
    api_results = [r for r in results if r['api_called']]
    report = {'status': 'PILOT_STRUCTURAL_PASS_AWAITING_HUMAN_GROUNDING_REVIEW',
              'businesses_completed': len(selected), 'business_ids': [b['business_id'] for b in selected],
              'requested_model': cfg['requested_model'], 'reported_models': sorted({r['reported_model'] for r in api_results}),
              'api_tasks_completed': len(api_results), 'empty_visual_tasks_without_API': len(results) - len(api_results),
              'input_tokens_reported': sum(r['usage'].get('input_tokens', 0) for r in api_results),
              'output_tokens_reported': sum(r['usage'].get('output_tokens', 0) for r in api_results),
              'reasoning_tokens_reported': sum(r['usage'].get('output_tokens_details', {}).get('reasoning_tokens', 0) for r in api_results),
              'human_verified_businesses': 0, 'held_out_teacher_targets': 0,
              'production_method_frozen': False, 'LED_training_started': False,
              'target_files': [(out / 'provisional_targets' / (b['business_id'] + '.json')).relative_to(ROOT).as_posix() for b in selected]}
    immutable(out / f'completion_{limit:02d}_businesses.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-pilot', action='store_true', help='Explicitly issue bounded pilot API requests')
    parser.add_argument('--limit-businesses', type=int, default=1)
    parser.add_argument('--config', default=CONFIG.relative_to(ROOT).as_posix())
    args = parser.parse_args()
    if not args.execute_pilot:
        parser.error('No API call without explicit --execute-pilot; default is not Run All execution')
    print(json.dumps(run(args.limit_businesses, local_path(args.config)), indent=2), flush=True)
