"""Resolve raw metadata multiplicity without choosing labels or filtering images."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json

import yaml
from data_inspection import is_restaurant, photo_inventory, resolve_data_root, stream_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'outputs/cohort'
    final = output / 'eligibility_verified.json'
    if final.exists():
        raise FileExistsError(final)
    audit = json.loads((output / 'eligibility_audit.json').read_text(encoding='utf-8'))
    quality = audit['quality_flags']
    other_failures = {key: value for key, value in quality.items() if value and key != 'photo_duplicate_id_occurrences'}
    if (audit['eligible_count'] != 10197 or audit['parse_errors'] or other_failures
            or audit['review_duplicate_id_occurrences'] != 0
            or not all(info['complete_scan'] for info in audit['sources'].values())
            or audit['image_inventory']['duplicate_stems']):
        raise RuntimeError('The baseline has unresolved failures beyond photo metadata multiplicity')
    config = yaml.safe_load((ROOT / 'configs/project.yaml').read_text(encoding='utf-8'))
    data = resolve_data_root(ROOT, config)
    for kind in ('business', 'review', 'photo'):
        if (data / config['paths'][f'{kind}_jsonl']).stat().st_size != audit['sources'][kind]['file_bytes']:
            raise RuntimeError('Source size changed')
    restaurant_ids = {row['business_id'] for row in stream_jsonl(data / config['paths']['business_jsonl']) if is_restaurant(row)}
    filenames, inventory = photo_inventory(data / config['paths']['photo_directory'])
    if inventory != audit['image_inventory']:
        raise RuntimeError('Photo inventory changed')
    records, duplicate_rows, raw_counts, unique_counts = {}, [], Counter(), Counter()
    for row in stream_jsonl(data / config['paths']['photo_jsonl']):
        pid, bid = row['photo_id'], row['business_id']
        if bid in restaurant_ids and pid in filenames:
            raw_counts[bid] += 1
        if pid in records:
            first = records[pid]
            if first['business_id'] != bid or {k: v for k, v in first.items() if k != 'label'} != {k: v for k, v in row.items() if k != 'label'}:
                raise RuntimeError(f'Conflicting photo ownership/content metadata: {pid}')
            duplicate_rows.append({'photo_id': pid, 'business_id': bid, 'original_records': [first, row],
                                   'resolution': 'one physical image; preserve all original metadata rows and label variants'})
        else:
            records[pid] = row
            if bid in restaurant_ids and pid in filenames:
                unique_counts[bid] += 1
    if len(duplicate_rows) != quality['photo_duplicate_id_occurrences']:
        raise RuntimeError('Duplicate occurrence count changed')
    with (output / 'eligible_businesses.csv').open(encoding='utf-8', newline='') as stream:
        original = list(csv.DictReader(stream))
    # Removing metadata repetitions can only reduce photo counts, so cannot add
    # a business that failed the baseline gate. Rechecking all baseline eligible
    # IDs is therefore sufficient to establish the full unique-photo pool.
    eligible = []
    for row in original:
        bid = row['business_id']
        if int(row['local_photo_records']) != raw_counts[bid]:
            raise RuntimeError(f'Baseline photo counts changed: {bid}')
        if unique_counts[bid] >= 5:
            eligible.append({'business_id': bid, 'raw_review_records': int(row['raw_review_records']),
                             'local_photo_records': unique_counts[bid]})
    final_csv = output / 'verified_eligible_businesses.csv'
    with final_csv.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['business_id', 'raw_review_records', 'local_photo_records'])
        writer.writeheader()
        writer.writerows(eligible)
    report = {
        'gate_passed': len(eligible) == 10197, 'eligible_count': len(eligible),
        'eligible_ids_unchanged_from_baseline': {r['business_id'] for r in eligible} == {r['business_id'] for r in original},
        'raw_review_records': sum(r['raw_review_records'] for r in eligible),
        'local_unique_photos': sum(r['local_photo_records'] for r in eligible),
        'local_photo_metadata_records': sum(raw_counts[r['business_id']] for r in eligible),
        'photo_metadata_records': audit['sources']['photo']['records'], 'unique_local_photo_ids': len(filenames),
        'duplicate_metadata_occurrences': len(duplicate_rows), 'duplicate_details': duplicate_rows,
        'verified_eligible_csv_sha256': hashlib.sha256(final_csv.read_bytes()).hexdigest(),
        'baseline_audit_sha256': hashlib.sha256((output / 'eligibility_audit.json').read_bytes()).hexdigest(),
        'policy': 'Count unique local photo_id for >=5 eligibility; preserve all raw metadata/label variants; no image-content deduplication or label preference.',
        'source_stats_at_verification': {kind: {'bytes': (p := data / config['paths'][f'{kind}_jsonl']).stat().st_size,
                                              'mtime_ns': p.stat().st_mtime_ns} for kind in ('business', 'review', 'photo')},
    }
    final.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['gate_passed']:
        raise RuntimeError('Unique-photo eligibility differs from 10,197')


if __name__ == '__main__':
    main()
