"""Prepare immutable Stage G pilot inputs offline; never calls an API or reads keys."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/stage_g_teacher_pilot.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def local_path(relative):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError('Input/output path escapes project')
    return p


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing output differs; use a new version namespace: {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def split_reviews(record):
    block = record['review_block']
    matches = list(re.finditer(r'(?m)^\[REV_(\d+)\] stars=(\d+) date=([^\n]+)\n', block))
    ids = record['selected_review_ids']
    if len(matches) != len(ids):
        raise ValueError('Review marker/ID count differs')
    result = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() - 2 if i + 1 < len(matches) else len(block)
        if int(match[1]) != i + 1:
            raise ValueError('Review markers out of order')
        text = block[match.end():end]
        if digest(text.encode('utf-8')) != record['review_text_sha256_in_order'][i]:
            raise ValueError('Complete original review text changed during extraction')
        result.append({'review_marker': f'REV_{i+1}', 'review_id': ids[i],
                       'stars': int(match[2]), 'date': match[3], 'text': text})
    if digest(block.encode('utf-8')) != record['review_block_sha256']:
        raise ValueError('Review block checksum mismatch')
    return result


def prepare(config_path=CONFIG):
    cfg = read(config_path)
    if cfg['execution_enabled'] or cfg['full_generation_enabled']:
        raise ValueError('This preparation version is draft/offline-only')
    pointer = read(local_path(cfg['source_pointer']))
    production = local_path(pointer['production_directory'])
    report_path = local_path(pointer['completion_report'])
    if file_sha(report_path) != pointer['completion_report_sha256']:
        raise ValueError('Stage F completion report changed')
    stage_f = read(report_path)
    bindings = {}

    def bound(relative):
        path = production / relative
        actual = file_sha(path)
        if actual != stage_f['artifact_sha256'][relative]:
            raise ValueError(f'Frozen Stage F input changed: {relative}')
        bindings[path.relative_to(ROOT).as_posix()] = actual
        return path

    reviews = {r['business_id']: r for r in rows(bound('manifests/shared_review_inputs.jsonl'))
               if r['split'] in ('train', 'validation')}
    conditions = {}
    source_rows = {}
    for split, expected in [('train', 2000), ('validation', 200)]:
        for condition in ['unfiltered', 'filtered']:
            data = list(rows(bound(f'manifests/{split}/{condition}_captions.jsonl')))
            if len(data) != expected or any(r['split'] != split for r in data):
                raise ValueError('Unexpected caption split membership/count')
            conditions[split, condition] = {r['business_id']: r for r in data}
        for condition in ['reviews_only', 'unfiltered', 'filtered']:
            source_rows[split, condition] = {
                r['business_id']: r for r in rows(bound(f'manifests/{split}/{condition}_sources.jsonl'))}
    if len(reviews) != 2200:
        raise ValueError('Expected 2200 permitted train/validation review inputs')
    inventory = {r['photo_id']: r for r in rows(bound('manifests/complete_caption_pool_inventory.jsonl'))
                 if r['split'] in ('train', 'validation')}
    chosen = []
    for split, count in cfg['pilot']['businesses_per_split'].items():
        if split not in ('train', 'validation'):
            raise ValueError('Held-out teacher pilot forbidden')
        pool = conditions[split, 'filtered']
        order = sorted(pool, key=lambda bid: digest(f"{cfg['pilot']['seed']}|StageG|{split}|{bid}".encode()))
        edge = next((bid for bid in order if pool[bid]['retained_caption_count'] == 0), None)
        selected = ([edge] if edge else []) + [bid for bid in order if bid != edge][:count - bool(edge)]
        for bid in selected:
            chosen.append({'business_id': bid, 'split': split,
                           'zero_filtered_edge_case': pool[bid]['retained_caption_count'] == 0})
    prompts = {k: local_path(v).read_text(encoding='utf-8') for k, v in cfg['prompts'].items()}
    schemas = read(local_path(cfg['schema']))
    for relative in [config_path.relative_to(ROOT).as_posix(), cfg['schema'], *cfg['prompts'].values()]:
        bindings[relative] = file_sha(local_path(relative))
    jobs = []
    images_checked = {}
    for item in chosen:
        bid, split = item['business_id'], item['split']
        shared = reviews[bid]
        if shared['split'] != split:
            raise ValueError('Review split mismatch')
        extracted = split_reviews(shared)
        for condition in ['reviews_only', 'unfiltered', 'filtered']:
            source = source_rows[split, condition][bid]
            if source['selected_review_ids'] != shared['selected_review_ids'] or source['review_block_sha256'] != shared['review_block_sha256']:
                raise ValueError('Review ID/order/block differs across conditions')
            if digest(source['source'].encode()) != source['source_sha256']:
                raise ValueError('Source serialization checksum changed')
        jobs.append({'job_id': f'{split}_{bid}_review_consensus', 'business_id': bid,
                     'split': split, 'task': 'review_consensus', 'condition': 'shared',
                     'evidence': {'business_id': bid, 'reviews': extracted,
                                  'exact_review_block': shared['review_block']},
                     'review_block_sha256': shared['review_block_sha256'], 'images': [],
                     'prompt_sha256': digest(prompts['review_consensus'].encode()),
                     'schema_sha256': digest(canonical(schemas['review_consensus']))})
        for condition in ['unfiltered', 'filtered']:
            record = conditions[split, condition][bid]
            if record['selected_photo_ids'] != [c['photo_id'] for c in record['captions']]:
                raise ValueError('Caption order mismatch')
            captions, images = [], []
            for i, caption in enumerate(record['captions'], 1):
                pid = caption['photo_id']
                inv = inventory[pid]
                if (inv['business_id'], inv['split'], inv['normalized_caption']) != (bid, split, caption['normalized_caption']):
                    raise ValueError('Caption inventory binding changed')
                checkpoint_path = local_path(inv['caption_record_path'])
                if file_sha(checkpoint_path) != inv['caption_record_sha256']:
                    raise ValueError('Caption checkpoint changed')
                checkpoint = read(checkpoint_path)
                if (checkpoint['business_id'], checkpoint['split']) != (bid, split):
                    raise ValueError('Caption checkpoint belongs to another business/split')
                if checkpoint['photo_id'] != pid or checkpoint['normalized_caption'] != caption['normalized_caption']:
                    raise ValueError('Caption checkpoint contents differ')
                image_path = local_path(checkpoint['image_path'])
                if pid not in images_checked:
                    if file_sha(image_path) != checkpoint['raw_sha256']:
                        raise ValueError('Original image checksum changed')
                    images_checked[pid] = checkpoint['raw_sha256']
                pair = {'caption_id': f'IMG_CAP_{i}', 'photo_id': pid,
                        'photo_marker': f'PHOTO_{i}', 'caption': caption['normalized_caption']}
                captions.append(pair)
                images.append({'photo_id': pid, 'photo_marker': f'PHOTO_{i}',
                               'relative_path': image_path.relative_to(ROOT).as_posix(),
                               'raw_sha256': checkpoint['raw_sha256']})
            jobs.append({'job_id': f'{split}_{bid}_{condition}_visual_evidence', 'business_id': bid,
                         'split': split, 'task': 'visual_evidence', 'condition': condition,
                         'evidence': {'business_id': bid, 'condition': condition, 'captions': captions},
                         'images': images, 'review_block_sha256': shared['review_block_sha256'],
                         'skip_api_empty_captions': not captions,
                         'prompt_sha256': digest(prompts['visual_evidence'].encode()),
                         'schema_sha256': digest(canonical(schemas['visual_evidence']))})
    output = local_path(cfg['output_directory'])
    summary = {'status': 'OFFLINE_PREPARATION_PASS_DRAFT_NOT_TEACHER_OUTPUT',
               'protocol_version': cfg['protocol_version'], 'requested_model': cfg['requested_model'],
               'snapshot': cfg['snapshot'], 'business_count': len(chosen),
               'businesses_by_split': dict(Counter(c['split'] for c in chosen)),
               'zero_filtered_pilot_businesses': sum(c['zero_filtered_edge_case'] for c in chosen),
               'jobs': len(jobs), 'proposed_api_calls': sum(not j.get('skip_api_empty_captions', False) for j in jobs),
               'unique_original_images_verified': len(images_checked),
               'review_ID_text_order_identical_across_conditions': True,
               'selected_caption_text_order_unchanged': True,
               'held_out_teacher_payloads': 0, 'teacher_API_calls': 0, 'teacher_targets': 0,
               'Stage_F_outputs_modified': False, 'execution_enabled': False,
               'input_sha256': bindings, 'jobs_sha256': digest(canonical(jobs)),
               'unresolved_before_execution': ['review_draft_pilot_protocol_and_prompts'],
               'unresolved_before_production': ['model_version_policy', 'target_length_policy',
                   'retry_policy', 'spending_limit', 'random_manual_inspection_sample_size',
                   'pilot_grounding_and_gateway_provenance_review']}
    immutable(output / 'pilot_businesses.json', chosen)
    immutable(output / 'pilot_jobs.json', jobs)
    immutable(output / 'preparation_report.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG.relative_to(ROOT).as_posix())
    args = parser.parse_args()
    result = prepare(local_path(args.config))
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))
