"""Export an immutable Stage B source cohort, gated on the full raw audit."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import time

import orjson
import yaml
from data_inspection import ACCOMMODATION_CATEGORIES, is_restaurant, photo_inventory, resolve_data_root

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')
ELIGIBLE_FIELDS = ['business_id', 'raw_review_records', 'local_photo_records']
SPLIT_FIELDS = ['business_id', 'split', 'selection_index', 'raw_review_records', 'local_photo_records']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_stat(path):
    stat = Path(path).stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def copy_photo(record, assignment, filename, source_directory, staging):
    """Independent file task; never changes metadata, selection or image bytes."""
    require(Path(filename).name == filename, 'Unsafe image filename')
    source = source_directory / filename
    destination = staging / 'photos' / filename
    require(not destination.exists(), 'Duplicate photo path')
    before = source_stat(source)
    shutil.copyfile(source, destination)
    original_hash = sha256(source)
    require(original_hash == sha256(destination), f'Image byte mismatch: {record["photo_id"]}')
    require(source_stat(source) == before, f'Original image changed during export: {record["photo_id"]}')
    row = {'photo_id': record['photo_id'], 'business_id': record['business_id'],
           'split': assignment['split'], 'label': record.get('label'),
           'relative_path': f'photos/{filename}', 'bytes': before['bytes'], 'sha256': original_hash}
    return record, row


def hashed_records(path, hashes):
    """Hash every original byte while streaming all JSON objects; fail closed."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            record = orjson.loads(line)
            require(isinstance(record, dict), f'Non-object JSON at {path}:{number}')
            yield record
    hashes[Path(path).name] = digest.hexdigest()


def select_cohort(eligible, seed, split_counts):
    rows = {row['business_id']: row for row in eligible}
    require(len(rows) == len(eligible), 'Duplicate eligible IDs')
    sampled = random.Random(seed).sample(sorted(rows), sum(split_counts.values()))
    assignments = {}
    cursor = 0
    for split in SPLITS:
        for index in range(cursor, cursor + split_counts[split]):
            bid = sampled[index]
            assignments[bid] = {
                'business_id': bid, 'split': split, 'selection_index': index,
                'raw_review_records': int(rows[bid]['raw_review_records']),
                'local_photo_records': int(rows[bid]['local_photo_records']),
            }
        cursor += split_counts[split]
    return assignments


def main():
    started = time.perf_counter()
    project = yaml.safe_load((ROOT / 'configs/project.yaml').read_text(encoding='utf-8'))
    cfg = yaml.safe_load((ROOT / 'configs/cohort.yaml').read_text(encoding='utf-8'))
    require(cfg['seed'] == project['reproducibility']['seed'] == 42, 'Seed mismatch')
    require(cfg['split_counts'] == project['study']['split_counts'] == dict(train=2000, validation=200, test=100), 'Split configuration mismatch')
    require(set(cfg['eligibility']['excluded_categories']) == ACCOMMODATION_CATEGORIES, 'Exclusion mismatch')
    audit_dir = ROOT / 'outputs/cohort'
    audit = json.loads((audit_dir / 'eligibility_audit.json').read_text(encoding='utf-8'))
    verified = json.loads((audit_dir / 'eligibility_verified.json').read_text(encoding='utf-8'))
    require(verified['gate_passed'] and verified['eligible_count'] == cfg['expected_eligible_businesses'] == 10197,
            'Full raw audit did not pass the 10,197 gate')
    require(sha256(audit_dir / 'eligibility_audit.json') == verified['baseline_audit_sha256'], 'Baseline audit changed')
    require(sha256(audit_dir / 'verified_eligible_businesses.csv') == verified['verified_eligible_csv_sha256'], 'Verified eligible CSV changed')
    eligible = read_csv(audit_dir / 'verified_eligible_businesses.csv')
    require(len(eligible) == 10197, 'Eligible CSV size mismatch')
    require(all(int(row['raw_review_records']) >= 10 and int(row['local_photo_records']) >= 5 for row in eligible), 'Invalid eligibility thresholds')
    eligible_ids = {row['business_id'] for row in eligible}
    require(all(row['business_id'] not in eligible_ids for row in verified['duplicate_details']),
            'This source exporter requires no conflicting photo metadata inside the eligible pool')
    output = ROOT / cfg['output_directory']
    staging = output.parent / f'.{output.name}.building'
    canonical = ROOT / cfg['canonical_split_directory']
    split_staging = canonical.parent / f'.splits.{output.name}.building'
    for path in (output, staging, canonical, split_staging):
        require(not path.exists(), f'Refusing to overwrite existing artifact: {path}')
    require(shutil.disk_usage(output.parent).free >= 15 * 1024**3, 'Need >=15 GiB free before export')
    data = resolve_data_root(ROOT, project)
    sources = {kind: data / project['paths'][f'{kind}_jsonl'] for kind in ('business', 'review', 'photo')}
    source_before = {kind: source_stat(path) for kind, path in sources.items()}
    require(source_before == verified['source_stats_at_verification'], 'Raw source metadata changed after verification')
    require(all(source_before[kind]['bytes'] == audit['sources'][kind]['file_bytes'] for kind in sources), 'Raw source size changed after audit')
    filenames, inventory = photo_inventory(data / project['paths']['photo_directory'])
    require(inventory == audit['image_inventory'], 'Local photo inventory changed after audit')
    assignments = select_cohort(eligible, cfg['seed'], cfg['split_counts'])
    remaining = [row for row in eligible if row['business_id'] not in assignments]
    require(len(remaining) == 7897, 'Remaining pool mismatch')
    staging.mkdir()
    for child in ('reviews', 'photo_metadata', 'photos', 'configs', 'provenance'):
        (staging / child).mkdir()
    split_rows = [assignments[bid] for bid in sorted(assignments)]
    write_csv(staging / 'business_split.csv', SPLIT_FIELDS, split_rows)
    write_csv(staging / 'eligible_businesses.csv', ELIGIBLE_FIELDS, eligible)
    write_csv(staging / 'remaining_eligible_businesses.csv', ELIGIBLE_FIELDS, remaining)
    for filename in ('project.yaml', 'cohort.yaml', 'research_pending.yaml'):
        shutil.copyfile(ROOT / 'configs' / filename, staging / 'configs' / filename)
    shutil.copyfile(ROOT / 'requirements/preprocessing.lock.txt', staging / 'provenance/preprocessing.lock.txt')
    shutil.copyfile(ROOT / 'specs/002_yelp_cohort_split.md', staging / 'provenance/stage_b_spec_at_build.md')
    shutil.copyfile(audit_dir / 'eligibility_audit.json', staging / 'provenance/eligibility_audit.json')
    shutil.copyfile(audit_dir / 'eligibility_verified.json', staging / 'provenance/eligibility_verified.json')
    shutil.copyfile(audit_dir / 'photo_duplicate_investigation.json', staging / 'provenance/photo_duplicate_investigation.json')
    source_hashes = {}
    businesses, review_counts, photo_counts = {}, Counter(), Counter()
    summary = {split: {'businesses': cfg['split_counts'][split], 'raw_reviews': 0, 'local_photos': 0} for split in SPLITS}
    print('Exporting selected business records...', flush=True)
    with (staging / 'businesses.jsonl').open('wb') as stream:
        for record in hashed_records(sources['business'], source_hashes):
            bid = record.get('business_id')
            if bid not in assignments:
                continue
            require(bid not in businesses and is_restaurant(record), 'Selected business invalid/duplicated')
            businesses[bid] = record
            stream.write(orjson.dumps(record) + b'\n')
    require(len(businesses) == 2300, 'Missing selected businesses')
    business_rows = []
    for bid in sorted(assignments):
        record = businesses[bid]
        business_rows.append({**assignments[bid], **{field: record.get(field) for field in ('name', 'city', 'state')}})
    write_csv(staging / 'cohort_businesses.csv', SPLIT_FIELDS + ['name', 'city', 'state'], business_rows)
    print('Streaming all original reviews; keeping every review of the 2,300 businesses...', flush=True)
    with ExitStack() as stack:
        streams = {split: stack.enter_context((staging / 'reviews' / f'{split}.jsonl').open('wb')) for split in SPLITS}
        for index, record in enumerate(hashed_records(sources['review'], source_hashes), 1):
            if index % 1000000 == 0:
                print(f'Review records scanned: {index:,}', flush=True)
            bid = record.get('business_id')
            if bid in assignments:
                split = assignments[bid]['split']
                streams[split].write(orjson.dumps(record) + b'\n')
                review_counts[bid] += 1
                summary[split]['raw_reviews'] += 1
    print('Exporting every locally available associated photo; copying original bytes...', flush=True)
    missing_selected_photos, image_bytes, copied = 0, 0, 0
    workers = cfg['exports']['photo_copy_workers']
    require(isinstance(workers, int) and 1 <= workers <= 32, 'Invalid photo worker count')
    photo_fields = ['photo_id', 'business_id', 'split', 'label', 'relative_path', 'bytes', 'sha256']
    with ExitStack() as stack:
        streams = {split: stack.enter_context((staging / 'photo_metadata' / f'{split}.jsonl').open('wb')) for split in SPLITS}
        photo_csv = stack.enter_context((staging / 'photo_manifest.csv').open('w', encoding='utf-8', newline=''))
        writer = csv.DictWriter(photo_csv, fieldnames=photo_fields)
        writer.writeheader()
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        pending = deque()
        seen_photo_ids = set()

        def finish(future):
            nonlocal copied, image_bytes
            record, row = future.result()
            bid, split = row['business_id'], row['split']
            writer.writerow(row)
            streams[split].write(orjson.dumps(record) + b'\n')
            photo_counts[bid] += 1
            summary[split]['local_photos'] += 1
            copied += 1
            image_bytes += row['bytes']
            if copied % 5000 == 0:
                print(f'Original photos copied and hash-verified: {copied:,}', flush=True)

        for record in hashed_records(sources['photo'], source_hashes):
            bid, pid = record.get('business_id'), record.get('photo_id')
            if bid not in assignments:
                continue
            filename = filenames.get(pid)
            if filename is None:
                missing_selected_photos += 1
                continue
            require(pid not in seen_photo_ids, f'Duplicate selected photo ID: {pid}')
            seen_photo_ids.add(pid)
            pending.append(executor.submit(copy_photo, record, assignments[bid], filename,
                                           data / project['paths']['photo_directory'], staging))
            # Bounded queue, metadata output remains in original source order.
            if len(pending) >= 2 * workers:
                finish(pending.popleft())
        while pending:
            finish(pending.popleft())
    for bid, row in assignments.items():
        require(review_counts[bid] == row['raw_review_records'], f'Review count mismatch: {bid}')
        require(photo_counts[bid] == row['local_photo_records'], f'Photo count mismatch: {bid}')
    require(all(source_stat(path) == source_before[kind] for kind, path in sources.items()), 'Original source metadata changed during export')
    split_hash = sha256(staging / 'business_split.csv')
    manifest = {
        'schema_version': 1, 'dataset_version': cfg['dataset_version'], 'stage': 'B_source_pool',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_export_complete': True,
        'final_led_inputs_ready': False, 'english_filter_applied': False, 'review_selection_applied': False,
        'image_integrity_and_deduplication_applied': False,
        'eligible_businesses': 10197, 'cohort_businesses': 2300, 'remaining_eligible_businesses': 7897,
        'seed': cfg['seed'], 'sampling_algorithm': cfg['sampling_algorithm'], 'python_version': platform.python_version(),
        'eligibility': cfg['eligibility'], 'split_summary': summary, 'business_split_sha256': split_hash,
        'raw_reviews': sum(review_counts.values()), 'local_photos': copied, 'photo_bytes': image_bytes,
        'selected_photo_metadata_without_local_file': missing_selected_photos,
        'raw_sources': {kind: {'relative_path_from_data_root': project['paths'][f'{kind}_jsonl'],
                              **source_before[kind], 'sha256': source_hashes[path.name]} for kind, path in sources.items()},
        'implementation_sha256': {name: sha256(ROOT / 'scripts' / name) for name in ('build_yelp_cohort.py', 'audit_yelp_eligibility.py', 'verify_yelp_photo_ids.py', 'data_inspection.py')},
        'notes': [
            'Raw photo metadata has two repeated photo IDs with label variants, both outside the eligible pool. Unique-photo verification confirms eligible IDs and counts unchanged; no label preference is applied.',
            'One example is one business; review JSONL rows are the complete original source pool, not selected training examples.',
            'Stage C must freeze English reviews, representative selection (<=30), chronological order and shared token budget.',
            'Minimum five images is eligibility only; all associated local images are exported without decoding/filtering.',
            'Original photo captions are Yelp metadata, not BLIP-2 outputs. Human/external evaluation decisions remain pending.',
            'Stage B alone does not satisfy the local B/C milestone for starting the GPU handoff.'
        ],
    }
    write_json(staging / 'dataset_manifest.json', manifest)
    (staging / 'README.md').write_text(
        '# Yelp 2,300-business source dataset (Stage B)\n\n'
        'Eligibility: Restaurants, accommodation exclusions, >=10 original reviews and >=5 local associated photos. '
        'Full audit: 10,197 eligible; sampled 2,300 with seed 42; train/validation/test = 2,000/200/100.\n\n'
        '`business_split.csv` is immutable and has 0-based selection positions. `businesses.jsonl` preserves original business records. '
        '`reviews/{split}.jsonl` contains ALL original reviews for the businesses in that split, including reviews not yet language-filtered. '
        '`photo_metadata/{split}.jsonl` preserves original Yelp metadata for ALL associated local photos; `photos/` preserves original bytes. '
        '`photo_manifest.csv` maps photo/business/split IDs to relative paths, sizes and SHA-256. `cohort_businesses.csv` provides names and source counts.\n\n'
        'One experimental example = one business. This source pool is not yet the final LED dataset. Stage C English-only representative '
        'selection (at most 30 reviews), ordering and token budgets are pending; Stage D image validation/deduplication, '
        'BLIP-2 captions, teacher targets, human-sample design and external evaluation are pending. No GPU instance has been rented.\n\n'
        'Keep the split and source IDs fixed. The remaining eligible businesses are listed separately. '
        '`dataset_manifest.json` records raw-source hashes, runtime, counts and configuration; `checksums.json` covers every other package file, including original images. '
        'Preserve the source Yelp agreements and observe their terms when using/transferring data.\n', encoding='utf-8')
    checksums = {}
    print('Hashing the complete package...', flush=True)
    for path in sorted(staging.rglob('*')):
        if path.is_file():
            checksums[path.relative_to(staging).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    write_json(staging / 'checksums.json', checksums)
    split_staging.mkdir()
    shutil.copyfile(staging / 'business_split.csv', split_staging / 'business_split.csv')
    write_json(split_staging / 'split_summary.json', {
        'dataset_version': cfg['dataset_version'], 'seed': 42, 'eligible_businesses': 10197,
        'cohort_businesses': 2300, 'remaining_eligible_businesses': 7897,
        'business_split_sha256': split_hash, 'split_summary': summary,
        'business_overlap': 0, 'source_pool_not_final_led_dataset': True,
    })
    # No existing output may be replaced. These paths have been checked above.
    require(not output.exists() and not canonical.exists(), 'Outputs appeared during build')
    staging.rename(output)
    split_staging.rename(canonical)
    report = {
        'build_passed': True, 'dataset_directory': str(output), 'canonical_split_directory': str(canonical),
        'split_summary': summary, 'raw_reviews': manifest['raw_reviews'], 'local_photos': copied,
        'photo_bytes': image_bytes, 'business_split_sha256': split_hash,
        'photo_copy_workers': workers,
        'checksum_inventory_sha256': sha256(output / 'checksums.json'),
        'raw_source_stats_unchanged': True, 'original_images_hash_verified': copied,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
        'independent_validation_required': True,
    }
    write_json(audit_dir / 'build_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
