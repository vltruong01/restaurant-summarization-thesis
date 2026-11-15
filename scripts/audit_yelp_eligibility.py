"""Read-only full audit; never creates a cohort or split."""
from pathlib import Path
import csv
import json
import sys

import yaml
from data_inspection import full_core_audit, resolve_data_root

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = yaml.safe_load((ROOT / 'configs/project.yaml').read_text(encoding='utf-8'))
    data = resolve_data_root(ROOT, config)
    paths = config['paths']
    output = ROOT / 'outputs/cohort'
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'eligibility_audit.json'
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite {report_path}')
    print('Starting full raw-source audit, including exact review-ID duplicate check.', flush=True)
    report = full_core_audit(
        data / paths['business_jsonl'], data / paths['review_jsonl'],
        data / paths['photo_jsonl'], data / paths['photo_directory'],
        min_reviews=10, min_photos=5, track_review_duplicates=True, progress=False,
    )
    eligible = report.pop('eligible_businesses')
    with (output / 'eligible_businesses.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['business_id', 'raw_review_records', 'local_photo_records'])
        writer.writeheader()
        writer.writerows(eligible)
    historical_path = data / 'processed/language_stats/restaurant_business_language_stats.csv'
    if historical_path.exists():
        with historical_path.open(encoding='utf-8-sig', newline='') as stream:
            historical = list(csv.DictReader(stream))
        historical_ids = {row['business_id'] for row in historical}
        current_ids = {row['business_id'] for row in eligible}
        report['historical_comparison'] = {
            'path': str(historical_path), 'rows': len(historical),
            'same_business_id_set': historical_ids == current_ids,
            'only_current_count': len(current_ids - historical_ids),
            'only_historical_count': len(historical_ids - current_ids),
            'columns': list(historical[0]) if historical else [],
        }
    report['eligible_totals'] = {
        'raw_reviews': sum(row['raw_review_records'] for row in eligible),
        'local_photos': sum(row['local_photo_records'] for row in eligible),
    }
    report['gate_passed'] = (
        report['eligible_count'] == 10197 and not report['counts_need_review']
        and report['review_duplicate_id_occurrences'] == 0
        and report['image_inventory']['duplicate_stems'] == 0
        and all(info['complete_scan'] for info in report['sources'].values())
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['gate_passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
