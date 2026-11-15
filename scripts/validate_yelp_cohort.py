"""Independent read-only acceptance of the saved Stage B source package."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import random
import time

import orjson
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def records(path, source_hashes=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                record = orjson.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f'Non-object JSON: {path}')
                yield record
    if source_hashes is not None:
        source_hashes[path.name] = digest.hexdigest()


def canonical(record):
    return orjson.dumps(record, option=orjson.OPT_SORT_KEYS) + b'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/yelp_2300_v1')
    args = parser.parse_args()
    package = args.dataset.resolve()
    started = time.perf_counter()
    checks = []
    report_path = ROOT / 'outputs/cohort/validation_report.json'

    def check(condition, name):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise RuntimeError(name)

    report = {'dataset_directory': str(package), 'validation_passed': False, 'checks': checks}
    try:
        manifest = json.loads((package / 'dataset_manifest.json').read_text(encoding='utf-8'))
        inventory = json.loads((package / 'checksums.json').read_text(encoding='utf-8'))
        actual_files = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
        check(actual_files == set(inventory) | {'checksums.json'}, 'Complete checksum inventory with no extra/missing files')
        print(f'Validating hashes of {len(inventory):,} package files...', flush=True)
        for index, (relative, entry) in enumerate(inventory.items(), 1):
            path = (package / relative).resolve()
            if not path.is_relative_to(package) or path.stat().st_size != entry['bytes'] or digest_file(path) != entry['sha256']:
                raise RuntimeError(f'Invalid path/size/hash: {relative}')
            if index % 5000 == 0:
                print(f'Package hashes verified: {index:,}', flush=True)
        check(True, 'Every package file SHA-256 and size verified')
        check(manifest['stage'] == 'B_source_pool' and not manifest['final_led_inputs_ready']
              and not manifest['english_filter_applied'] and not manifest['review_selection_applied'], 'Stage B scope accurately declared')
        eligible_rows = csv_rows(package / 'eligible_businesses.csv')
        eligible = {row['business_id']: (int(row['raw_review_records']), int(row['local_photo_records'])) for row in eligible_rows}
        audit_eligible_rows = csv_rows(ROOT / 'outputs/cohort/verified_eligible_businesses.csv')
        check(len(eligible_rows) == len(eligible) == 10197 and eligible_rows == audit_eligible_rows, 'Eligible pool equals full audit: 10,197 unique IDs')
        check(all(reviews >= 10 and photos >= 5 for reviews, photos in eligible.values()), 'Every eligible business satisfies original thresholds')
        rows = csv_rows(package / 'business_split.csv')
        assignment = {row['business_id']: row for row in rows}
        split_ids = {split: {bid for bid, row in assignment.items() if row['split'] == split} for split in ('train', 'validation', 'test')}
        expected_sizes = {'train': 2000, 'validation': 200, 'test': 100}
        check(len(rows) == len(assignment) == 2300 and {key: len(value) for key, value in split_ids.items()} == expected_sizes,
              'Exactly 2,300 unique businesses split 2,000/200/100')
        check(all(not split_ids[left] & split_ids[right] for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))), 'Zero business overlap')
        check(set(assignment) <= set(eligible), 'All selected businesses belong to eligible pool')
        check(all((int(row['raw_review_records']), int(row['local_photo_records'])) == eligible[bid] for bid, row in assignment.items()), 'Saved cohort counts equal audited counts')
        draw = random.Random(42).sample(sorted(eligible), 2300)
        position_rows = sorted(rows, key=lambda row: int(row['selection_index']))
        check([int(row['selection_index']) for row in position_rows] == list(range(2300))
              and [row['business_id'] for row in position_rows] == draw
              and all(row['split'] == ('train' if index < 2000 else 'validation' if index < 2200 else 'test') for index, row in enumerate(position_rows)),
              'Saved membership/positions reproduce fresh seed-42 sampling')
        remaining_rows = csv_rows(package / 'remaining_eligible_businesses.csv')
        remaining = {row['business_id'] for row in remaining_rows}
        check(len(remaining_rows) == len(remaining) == 7897 and not remaining & set(assignment)
              and remaining | set(assignment) == set(eligible), 'Remaining 7,897 forms exact disjoint partition')
        split_hash = digest_file(package / 'business_split.csv')
        canonical_split = ROOT / 'data/processed/splits/business_split.csv'
        split_summary = json.loads((canonical_split.parent / 'split_summary.json').read_text(encoding='utf-8'))
        check(split_hash == manifest['business_split_sha256'] == digest_file(canonical_split) == split_summary['business_split_sha256'], 'Canonical/package split hashes identical')
        business_records = list(records(package / 'businesses.jsonl'))
        business_content = {row['business_id']: canonical(row) for row in business_records}
        check(len(business_records) == len(business_content) == 2300 and set(business_content) == set(assignment), 'Business JSONL IDs exactly match cohort')
        review_counts = Counter()
        review_ids = set()
        review_content = {bid: hashlib.sha256() for bid in assignment}
        review_quality = Counter()
        observed_summary = {}
        for split in expected_sizes:
            count = 0
            for row in records(package / 'reviews' / f'{split}.jsonl'):
                bid, rid = row.get('business_id'), row.get('review_id')
                if bid not in split_ids[split] or rid in review_ids or not isinstance(rid, str) or not rid:
                    raise RuntimeError(f'Invalid/duplicate review ID or split join: {rid}')
                review_ids.add(rid)
                review_counts[bid] += 1
                review_content[bid].update(canonical(row))
                review_quality['empty_text'] += not isinstance(row.get('text'), str) or not row['text'].strip()
                stars = row.get('stars')
                review_quality['invalid_stars'] += isinstance(stars, bool) or stars not in (1, 2, 3, 4, 5)
                try:
                    datetime.fromisoformat(row.get('date'))
                except (ValueError, TypeError):
                    review_quality['invalid_date'] += 1
                count += 1
            observed_summary[split] = {'businesses': len(split_ids[split]), 'raw_reviews': count, 'local_photos': 0}
        check(all(review_counts[bid] == eligible[bid][0] for bid in assignment), 'Every exported raw review count equals audit; review IDs unique; split joins valid')
        report['review_quality_flags'] = dict(review_quality)
        # Raw quality issues are recorded, not silently removed from the source pool.
        photos = csv_rows(package / 'photo_manifest.csv')
        photo_rows = {row['photo_id']: row for row in photos}
        photo_counts, image_bytes = Counter(), 0
        check(len(photos) == len(photo_rows), 'Photo manifest IDs unique')
        for row in photos:
            bid, split, pid = row['business_id'], row['split'], row['photo_id']
            if split not in split_ids or bid not in split_ids[split]:
                raise RuntimeError(f'Invalid photo split join: {pid}')
            relative = row['relative_path']
            if Path(relative).parts[0] != 'photos' or Path(relative).stem != pid:
                raise RuntimeError(f'Invalid photo path: {relative}')
            entry = inventory[relative]
            if entry['sha256'] != row['sha256'] or entry['bytes'] != int(row['bytes']):
                raise RuntimeError(f'Photo checksum disagrees with inventory: {pid}')
            photo_counts[bid] += 1
            observed_summary[split]['local_photos'] += 1
            image_bytes += int(row['bytes'])
        check(all(photo_counts[bid] == eligible[bid][1] for bid in assignment), 'All local associated photos exported; per-business photo counts equal audit')
        photo_content = {}
        for split in expected_sizes:
            for row in records(package / 'photo_metadata' / f'{split}.jsonl'):
                pid, bid = row.get('photo_id'), row.get('business_id')
                if pid not in photo_rows or bid not in split_ids[split] or pid in photo_content:
                    raise RuntimeError(f'Invalid photo metadata ID/join: {pid}')
                if str(row.get('label') or '') != photo_rows[pid]['label'] or bid != photo_rows[pid]['business_id']:
                    raise RuntimeError(f'Photo metadata mismatch: {pid}')
                photo_content[pid] = canonical(row)
        check(set(photo_content) == set(photo_rows), 'Photo metadata and manifest form exact one-to-one join')
        check(observed_summary == manifest['split_summary'] == split_summary['split_summary']
              and sum(review_counts.values()) == manifest['raw_reviews']
              and len(photos) == manifest['local_photos'] and image_bytes == manifest['photo_bytes'], 'Actual source record/photo totals equal manifest and canonical summary')

        # Independently compare every exported record and original image to the raw sources.
        config = yaml.safe_load((ROOT / 'configs/project.yaml').read_text(encoding='utf-8'))
        configured_root = Path(os.environ.get('THESIS_DATA_ROOT', config['paths']['data_root']))
        data_root = configured_root if configured_root.is_absolute() else ROOT / configured_root
        raw_hashes = {}
        raw_businesses = set()
        for row in records(data_root / config['paths']['business_jsonl'], raw_hashes):
            bid = row.get('business_id')
            if bid in assignment:
                if canonical(row) != business_content[bid]:
                    raise RuntimeError(f'Business record differs from original: {bid}')
                raw_businesses.add(bid)
        check(raw_businesses == set(assignment), 'All business fields exactly preserved from raw source')
        print('Independently comparing all selected review contents with the full raw source...', flush=True)
        raw_review_content = {bid: hashlib.sha256() for bid in assignment}
        raw_counts = Counter()
        for index, row in enumerate(records(data_root / config['paths']['review_jsonl'], raw_hashes), 1):
            bid = row.get('business_id')
            if bid in assignment:
                raw_review_content[bid].update(canonical(row))
                raw_counts[bid] += 1
            if index % 1000000 == 0:
                print(f'Raw review records independently scanned: {index:,}', flush=True)
        check(raw_counts == review_counts and all(raw_review_content[bid].digest() == review_content[bid].digest() for bid in assignment),
              'All raw review fields/content/order preserved; no missing or added source reviews')
        raw_photo_ids = set()
        for row in records(data_root / config['paths']['photo_jsonl'], raw_hashes):
            pid = row.get('photo_id')
            if pid in photo_rows:
                if canonical(row) != photo_content[pid]:
                    raise RuntimeError(f'Photo metadata differs from original: {pid}')
                raw_photo_ids.add(pid)
        check(raw_photo_ids == set(photo_rows), 'All photo metadata fields exactly preserved from raw source')
        for kind, original in manifest['raw_sources'].items():
            path = data_root / original['relative_path_from_data_root']
            stat = path.stat()
            if raw_hashes[path.name] != original['sha256'] or stat.st_size != original['bytes'] or stat.st_mtime_ns != original['mtime_ns']:
                raise RuntimeError(f'Original source changed: {kind}')
        check(True, 'Original raw JSONL hashes/sizes/mtime unchanged')
        print('Verifying exported photo hashes against original local images...', flush=True)
        for row in photos:
            original = data_root / config['paths']['photo_directory'] / Path(row['relative_path']).name
            if digest_file(original) != row['sha256']:
                raise RuntimeError(f'Original image differs from exported photo: {row["photo_id"]}')
        check(True, 'Every exported photo equals original source bytes by SHA-256')
        report.update({'validation_passed': True, 'eligible_businesses': 10197, 'cohort_businesses': 2300,
                       'remaining_eligible_businesses': 7897, 'business_overlap': 0, 'split_summary': observed_summary,
                       'raw_reviews': sum(review_counts.values()), 'local_photos': len(photos),
                       'business_split_sha256': split_hash, 'checksum_inventory_sha256': digest_file(package / 'checksums.json'),
                       'raw_source_hashes_verified': True, 'original_images_hash_verified': len(photos)})
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
