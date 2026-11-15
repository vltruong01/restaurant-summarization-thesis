"""Independently verify qualification evidence and post-gate sampling."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import random
import sqlite3
import time

import orjson
import yaml
from data_inspection import resolve_data_root

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    started = time.perf_counter()
    out = ROOT / 'outputs/english_eligibility'
    report = json.loads((out / 'eligibility_report.json').read_text(encoding='utf-8'))
    decisions = read_csv(out / 'business_english_decisions.csv')
    baseline = read_csv(ROOT / 'outputs/cohort/verified_eligible_businesses.csv')
    assert len(decisions) == len(baseline) == 10197
    by_id = {r['business_id']: r for r in decisions}
    assert len(by_id) == len(decisions)
    assert set(by_id) == {r['business_id'] for r in baseline}
    for row in baseline:
        current = by_id[row['business_id']]
        assert current['raw_review_records'] == row['raw_review_records']
        assert current['local_photo_records'] == row['local_photo_records']
    db_path = out / 'review_language_cache.sqlite'
    connection = sqlite3.connect(f'file:{db_path.as_posix()}?mode=ro', uri=True)
    cache = {r[0]: r for r in connection.execute('SELECT * FROM cache')}
    connection.close()
    cache_assessed = Counter(r[1] for r in cache.values())
    cache_english = Counter(r[1] for r in cache.values() if r[2] == 'en')
    for bid, row in by_id.items():
        assert int(row['reviews_language_assessed']) == cache_assessed[bid]
        assert int(row['english_review_count_lower_bound']) == cache_english[bid]
        assert int(row['unassessed_reviews']) == int(row['raw_review_records']) - cache_assessed[bid]
        eligible = cache_english[bid] >= 10 and int(row['local_photo_records']) >= 5
        assert (row['eligible_english'] == 'True') == eligible
        assert (row['english_count_is_exact'] == 'True') == (cache_assessed[bid] == int(row['raw_review_records']))
        if not eligible:
            assert row['english_count_is_exact'] == 'True'
    config = yaml.safe_load((ROOT / 'configs/project.yaml').read_text(encoding='utf-8'))
    data = resolve_data_root(ROOT, config)
    digest, found, actual_raw = hashlib.sha256(), set(), Counter()
    print('Checking every cached review ID/text/owner against the full raw source...', flush=True)
    with (data / config['paths']['review_jsonl']).open('rb') as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = orjson.loads(line)
            bid, rid = row['business_id'], row['review_id']
            if bid in by_id:
                actual_raw[bid] += 1
            if rid in cache:
                cached = cache[rid]
                assert cached[1] == bid
                assert cached[4] == hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
                assert rid not in found
                found.add(rid)
    assert digest.hexdigest() == report['identity']['source_review_sha256']
    assert found == set(cache)
    assert all(actual_raw[bid] == int(row['raw_review_records']) for bid, row in by_id.items())
    proposed = read_csv(out / 'proposed_business_split.csv')
    qualified = sorted(bid for bid, row in by_id.items() if row['eligible_english'] == 'True')
    expected = random.Random(42).sample(qualified, 2300)
    ordered = sorted(proposed, key=lambda r: int(r['selection_index']))
    assert [r['business_id'] for r in ordered] == expected
    assert [int(r['selection_index']) for r in ordered] == list(range(2300))
    assert len({r['business_id'] for r in proposed}) == 2300
    assert Counter(r['split'] for r in proposed) == {'train': 2000, 'validation': 200, 'test': 100}
    assert all(r['split'] == ('train' if i < 2000 else 'validation' if i < 2200 else 'test') for i, r in enumerate(ordered))
    old = read_csv(data / 'processed/yelp_2300_v1/business_split.csv')
    old_membership = {r['business_id']: (r['split'], int(r['selection_index'])) for r in old}
    new_membership = {r['business_id']: (r['split'], int(r['selection_index'])) for r in proposed}
    for name, expected_hash in report['artifact_sha256'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected_hash
    result = {'validation_passed': True, 'baseline_businesses': len(by_id),
              'english_eligible_businesses': len(qualified), 'english_ineligible_businesses': len(by_id) - len(qualified),
              'language_cache_records_verified_against_raw': len(found), 'full_raw_review_hash_verified': True,
              'all_business_counts_and_evidence_verified': True, 'proposed_sampling_reproduced': True,
              'same_business_ids_split_and_selection_positions_as_v1': old_membership == new_membership,
              'elapsed_seconds': round(time.perf_counter() - started, 3)}
    (out / 'validation_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
