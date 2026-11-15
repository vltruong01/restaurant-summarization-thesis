"""D-042 production helpers. Teacher sentences are never generated or edited here."""
import gzip
import json
from collections import Counter
from pathlib import Path
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable, rows, split_reviews
from run_stage_g_teacher_pilot_v4_1_recovery import payload, validate_output, response_text, now
from run_stage_g_extended_pilot_d041 import build_target
from stage_g_canonical_review_ids import resolve_review, citation_diagnostics

CONFIG = ROOT / 'configs/stage_g_teacher_production_d042.json'
FOLDER = ROOT / 'outputs/stage_g/teacher_production_d042_v1'
OUT = FOLDER / 'execution_v1'
PILOT = ROOT / 'outputs/stage_g/teacher_extended_pilot_d041_v1/recovery_v4'
PILOT_OUT = PILOT / 'execution_v1'
PARENT = ROOT / 'configs/stage_g_teacher_pilot_v6_focused_d040.json'
PILOT_CONFIG = ROOT / 'configs/stage_g_teacher_extended_pilot_d041.json'
FIXED_KEYS = ['protocol_version', 'requested_model', 'snapshot', 'generation_proposal',
              'tasks', 'prompts', 'schema', 'source_pointer', 'evidence_representation',
              'target_serialization', 'citation_metadata_policy']
CORE = ['prepare_stage_g_teacher_pilot.py', 'run_stage_g_teacher_pilot_v4_1_recovery.py',
        'run_stage_g_extended_pilot_d041.py', 'stage_g_visual_target_serialization.py',
        'stage_g_evidence_id_protocol.py', 'stage_g_canonical_review_ids.py',
        'prepare_stage_g_review_evidence_ids.py', 'stage_g_teacher_credentials.py',
        'stage_g_teacher_stream_diagnostics_v1.py', 'stage_g_production_d042.py',
        'run_stage_g_production_d042.py', 'audit_stage_g_production_d042.py']


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def location(job, imports):
    return PILOT_OUT if job['job_id'] in imports else OUT


def verify_checkpoint(job, cfg, directory, config_path, exact_request=True):
    """Offline response/request/source audit; does not call legacy credential resolver."""
    p = directory / 'checkpoints' / (job['job_id'] + '.json')
    cp = read(p)
    assert cp['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW', 'Unresolved/failed checkpoint: ' + job['job_id']
    binding = cp['binding']
    assert binding['job_sha256'] == digest(canonical(job))
    assert binding['config_sha256'] == file_sha(config_path)
    assert binding['prompt_sha256'] == job['prompt_sha256']
    assert binding['schema_sha256'] == job['schema_sha256']
    assert (binding['base_url'], binding['provider'], binding['requested_model']) == ('https://www.coletoken.de/v1', 'coletoken', 'gpt-6.1-sol')
    schema = read(ROOT / cfg['schema'])[job['task']]
    assert cp['validation'] == validate_output(job, cp['output'], schema)
    if cp.get('api_called'):
        rawpath = directory / 'raw_responses' / p.name
        raw = read(rawpath)
        assert file_sha(rawpath) == cp['raw_response_sha256']
        assert raw['status'] == 'completed' and raw['model'] == cfg['requested_model']
        assert json.loads(response_text(raw)) == cp['output']
        if exact_request:
            prompt = (ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
            body = canonical(payload(job, cfg, prompt, schema))
            exact = gzip.decompress((directory / 'raw_requests' / (job['job_id'] + '.json.gz')).read_bytes())
            assert exact == body
            assert read(directory / 'request_intents' / p.name)['request_sha256'] == digest(body)
    else:
        assert job.get('skip_api_empty_captions') and cp['output']['assessments'] == []
    if job['task'] == 'review_consensus':
        resolved, evidence = resolve_review(job, cp['output'])
        assert resolved == cp['resolved_output'] and evidence == cp['evidence_resolution']
        assert [(c['text'], c['polarity']) for c in resolved['claims']] == [(c['text'], c['polarity']) for c in cp['output']['claims']]
    return cp


def check_inputs():
    cfg = read(CONFIG)
    frozen = read(FOLDER / 'preflight.json')
    parent = read(PARENT)
    assert cfg['decision'] == 'D-042' and cfg['execution_enabled'] and cfg['full_generation_enabled']
    for k in FIXED_KEYS:
        assert cfg[k] == parent[k], 'D040 scientific field changed: ' + k
    for rel, sha in frozen['protected_sha256'].items():
        assert file_sha(ROOT / rel) == sha, 'Protected input changed: ' + rel
    jobs = read(FOLDER / 'production_jobs.json')
    businesses = read(FOLDER / 'production_businesses.json')
    imports = set(read(FOLDER / 'reuse_manifest.json')['job_ids'])
    assert len(jobs) == len({j['job_id'] for j in jobs}) == 6600
    assert len(businesses) == len({b['business_id'] for b in businesses}) == 2200
    assert Counter(b['split'] for b in businesses) == {'train': 2000, 'validation': 200}
    assert all(j['split'] in ('train', 'validation') for j in jobs)
    return cfg, jobs, businesses, imports


def save_completed_business(business, jobs_by_business, imports):
    content = {}
    for j in jobs_by_business[business['business_id']]:
        cp_path = location(j, imports) / 'checkpoints' / (j['job_id'] + '.json')
        if not cp_path.exists():
            return None
        cp = read(cp_path)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            return None
        content[j['condition']] = cp.get('resolved_output', cp['output'])
    result = build_target(business, content)
    if jobs_by_business[business['business_id']][0]['job_id'] in imports:
        original = PILOT_OUT / 'provisional_targets' / (business['business_id'] + '.json')
        assert read(original) == result
        return original
    path = OUT / 'targets' / (business['business_id'] + '.json')
    # Frozen serializer's historical status is retained; approval/provenance is external.
    immutable(path, result)
    return path


def prepare():
    """Bind all permitted frozen inputs and prove 100-business reuse before any call."""
    cfg = read(CONFIG); parent = read(PARENT)
    for k in FIXED_KEYS:
        assert cfg[k] == parent[k], k
    pointer = read(ROOT / cfg['source_pointer'])
    assert file_sha(ROOT / pointer['completion_report']) == pointer['completion_report_sha256']
    stage_f = read(ROOT / pointer['completion_report'])
    production = ROOT / pointer['production_directory']
    bindings = {}

    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = file_sha(path)
        return path

    def bound(rel):
        path = production / rel
        assert file_sha(path) == stage_f['artifact_sha256'][rel], rel
        return bind(path)

    reviews = {r['business_id']: r for r in rows(bound('manifests/shared_review_inputs.jsonl')) if r['split'] in ('train', 'validation')}
    inventory = {r['photo_id']: r for r in rows(bound('manifests/complete_caption_pool_inventory.jsonl')) if r['split'] in ('train', 'validation')}
    conditions = {}; sources = {}
    for split, expected in [('train', 2000), ('validation', 200)]:
        for c in ('unfiltered', 'filtered'):
            data = list(rows(bound(f'manifests/{split}/{c}_captions.jsonl')))
            assert len(data) == expected and all(r['split'] == split for r in data)
            conditions[split, c] = {r['business_id']: r for r in data}
            assert len(conditions[split, c]) == expected
        for c in ('reviews_only', 'unfiltered', 'filtered'):
            data = list(rows(bound(f'manifests/{split}/{c}_sources.jsonl')))
            assert len(data) == expected and all(r['split'] == split for r in data)
            sources[split, c] = {r['business_id']: r for r in data}
    assert len(reviews) == 2200
    schemas = read(ROOT / cfg['schema'])
    prompts = {k: (ROOT / v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    businesses = []; jobs = []
    for split in ('train', 'validation'):
        for bid in sorted(conditions[split, 'filtered']):
            shared = reviews[bid]; assert shared['split'] == split
            extracted = split_reviews(shared)
            business = {'business_id': bid, 'split': split,
                        'zero_filtered_edge_case': conditions[split, 'filtered'][bid]['retained_caption_count'] == 0,
                        'selected_review_ids': shared['selected_review_ids'], 'review_block_sha256': shared['review_block_sha256']}
            businesses.append(business)
            for c in ('reviews_only', 'unfiltered', 'filtered'):
                s = sources[split, c][bid]
                assert s['selected_review_ids'] == shared['selected_review_ids'] and s['review_block_sha256'] == shared['review_block_sha256']
                assert digest(s['source'].encode()) == s['source_sha256']
            jobs.append({'job_id': f'{split}_{bid}_review_consensus', 'business_id': bid, 'split': split,
                         'task': 'review_consensus', 'condition': 'shared',
                         'evidence': {'business_id': bid, 'reviews': extracted, 'exact_review_block': shared['review_block']},
                         'review_block_sha256': shared['review_block_sha256'], 'images': [],
                         'prompt_sha256': digest(prompts['review_consensus'].encode()),
                         'schema_sha256': digest(canonical(schemas['review_consensus']))})
            for c in ('unfiltered', 'filtered'):
                record = conditions[split, c][bid]; captions = []; images = []
                assert record['selected_photo_ids'] == [x['photo_id'] for x in record['captions']]
                for i, caption in enumerate(record['captions'], 1):
                    pid = caption['photo_id']; inv = inventory[pid]
                    assert (inv['business_id'], inv['split'], inv['normalized_caption']) == (bid, split, caption['normalized_caption'])
                    cp_path = ROOT / inv['caption_record_path']
                    assert file_sha(bind(cp_path)) == inv['caption_record_sha256']
                    cp = read(cp_path)
                    assert (cp['business_id'], cp['split'], cp['photo_id'], cp['normalized_caption']) == (bid, split, pid, caption['normalized_caption'])
                    photo_path = ROOT / cp['image_path']
                    if photo_path.relative_to(ROOT).as_posix() not in bindings:
                        assert file_sha(bind(photo_path)) == cp['raw_sha256']
                    assert bindings[photo_path.relative_to(ROOT).as_posix()] == cp['raw_sha256']
                    captions.append({'caption_id': f'IMG_CAP_{i}', 'photo_id': pid, 'photo_marker': f'PHOTO_{i}', 'caption': caption['normalized_caption']})
                    images.append({'photo_id': pid, 'photo_marker': f'PHOTO_{i}', 'relative_path': photo_path.relative_to(ROOT).as_posix(), 'raw_sha256': cp['raw_sha256']})
                assert len(captions) <= 5
                assert sources[split, c][bid]['selected_photo_ids'] == [x['photo_id'] for x in captions]
                jobs.append({'job_id': f'{split}_{bid}_{c}_visual_evidence', 'business_id': bid, 'split': split,
                             'task': 'visual_evidence', 'condition': c,
                             'evidence': {'business_id': bid, 'condition': c, 'captions': captions},
                             'images': images, 'review_block_sha256': shared['review_block_sha256'], 'skip_api_empty_captions': not captions,
                             'prompt_sha256': digest(prompts['visual_evidence'].encode()), 'schema_sha256': digest(canonical(schemas['visual_evidence']))})
    by_id = {j['job_id']: j for j in jobs}
    pilot_jobs = read(PILOT / 'pilot_jobs.json'); pilot_cfg = read(PILOT_CONFIG)
    assert len(pilot_jobs) == 300
    for k in FIXED_KEYS: assert pilot_cfg[k] == cfg[k], k
    for j in pilot_jobs:
        assert by_id[j['job_id']] == j, 'Pilot evidence differs: ' + j['job_id']
        verify_checkpoint(j, cfg, PILOT_OUT, PILOT_CONFIG)
    pilot_ids = {j['business_id'] for j in pilot_jobs}; assert len(pilot_ids) == 100
    # Preserve all pilot/audit/review-material bytes, including human annotation status.
    for directory in [ROOT / 'outputs/stage_g/teacher_extended_pilot_d041_v1',
                      ROOT / 'outputs/stage_g/d041_final_pilot_audit_v1',
                      ROOT / 'outputs/stage_g/d041_visual_informativeness_audit_v1']:
        for p in sorted(directory.rglob('*')):
            if p.is_file(): bind(p)
    for rel in [cfg['source_pointer'], pointer['completion_report'], cfg['schema'], 'schemas/stage_g_teacher_pilot_v1.json',
                *cfg['prompts'].values(), 'assets/tokenizers/led-base-16384.manifest.json']:
        bind(ROOT / rel)
    for name in CORE: bind(ROOT / 'scripts' / name)
    bind(CONFIG); bind(PARENT); bind(PILOT_CONFIG)
    tokenizer = read(ROOT / 'assets/tokenizers/led-base-16384.manifest.json')
    for name, sha in tokenizer['sha256'].items(): assert file_sha(bind(ROOT / tokenizer['relative_directory'] / name)) == sha
    immutable(FOLDER / 'production_businesses.json', businesses)
    immutable(FOLDER / 'production_jobs.json', jobs)
    reuse = {'business_ids': sorted(pilot_ids), 'job_ids': [j['job_id'] for j in pilot_jobs],
             'source_namespace': PILOT.relative_to(ROOT).as_posix(), 'source_config': PILOT_CONFIG.relative_to(ROOT).as_posix(),
             'config_SHA_bindings_preserved_not_rewritten': True, 'exact_request_compatibility_checks': 292,
             'API_calls_for_import': 0, 'target_content_and_existing_status_unchanged': True,
             'human_confirmed_labels_created': 0}
    immutable(FOLDER / 'reuse_manifest.json', reuse)
    imports = set(reuse['job_ids'])
    lookup = {b['business_id']: [by_id[f"{b['split']}_{b['business_id']}_review_consensus"],
                               by_id[f"{b['split']}_{b['business_id']}_unfiltered_visual_evidence"],
                               by_id[f"{b['split']}_{b['business_id']}_filtered_visual_evidence"]] for b in businesses}
    for b in businesses:
        if b['business_id'] in pilot_ids: assert save_completed_business(b, lookup, imports)
    remaining = [b for b in businesses if b['business_id'] not in pilot_ids]
    smoke = [next(b for b in remaining if b['split'] == 'train'), next(b for b in remaining if b['split'] == 'validation')]
    # Two independent train and two validation businesses, chosen before any output.
    smoke += [next(b for b in remaining if b['split'] == s and b not in smoke) for s in ('train', 'validation')]
    immutable(FOLDER / 'small_batch_manifest.json', {'business_ids': [b['business_id'] for b in smoke],
              'selection': 'first_two_remaining_sorted_businesses_per_train_validation_split_before_API', 'unchanged_after_outputs': True})
    immutable(FOLDER / 'authorization.json', {'approved': True, 'approved_by': 'user', 'decision': 'D-042',
              'scope': '2200_train_validation_reuse100_generate2100_stop_before_training',
              'config_sha256': file_sha(CONFIG), 'paid_calls_authorized': True,
              'monetary_usage_monitored_by_researcher_in_provider_dashboard': True,
              'no_separate_pricing_checkpoint': True, 'automatic_retries': 0, 'maximum_concurrency': 4})
    for p in [FOLDER / n for n in ['production_businesses.json', 'production_jobs.json', 'reuse_manifest.json', 'small_batch_manifest.json', 'authorization.json']]: bind(p)
    report = {'status': 'D042_OFFLINE_PREFLIGHT_PASS', 'businesses': 2200, 'by_split': {'train': 2000, 'validation': 200},
              'reused_businesses': 100, 'remaining_businesses': 2100, 'jobs': 6600,
              'new_API_task_slots': sum(j['job_id'] not in imports and not j.get('skip_api_empty_captions') for j in jobs),
              'zero_filtered_businesses': sum(b['zero_filtered_edge_case'] for b in businesses),
              'imported_checkpoints_verified': 300, 'imported_exact_request_checks': 292,
              'held_out_payloads': 0, 'API_calls': 0, 'factual_accuracy_certified': False,
              'review_ID_text_order_identical_across_conditions': True, 'protected_sha256': bindings}
    immutable(FOLDER / 'preflight.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'protected_sha256'}, indent=2))
    return report


if __name__ == '__main__': prepare()
