"""Independent Stage C1 source/export/tokenization acceptance."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import random
import time

import orjson
import yaml
from tokenizers import Tokenizer
from english_eligibility import _initialize, classify_text, file_hash

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def canonical(row):
    return orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b'\n'


def main():
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT / 'configs/review_diagnostics.yaml').read_text(encoding='utf-8'))
    source, out, package = (ROOT / cfg[k] for k in ('dataset_directory', 'output_directory', 'english_pool_directory'))
    report = json.loads((out / 'completion_report.json').read_text(encoding='utf-8'))
    for name, expected in report['diagnostic_artifact_sha256'].items():
        assert file_hash(out / name) == expected, name
    for name, expected in report['english_pool_artifact_sha256'].items():
        assert file_hash(package / name) == expected, name
    meta_rows = csv_rows(out / 'review_language_token_metadata.csv')
    meta = {row['review_id']: row for row in meta_rows}
    assert len(meta_rows) == len(meta) == 601493 == report['raw_reviews']
    del meta_rows
    business = {row['business_id']: row for row in csv_rows(out / 'business_review_statistics.csv')}
    split_rows = csv_rows(source / 'business_split.csv')
    membership = {row['business_id']: row['split'] for row in split_rows}
    assert len(business) == len(membership) == 2300 and set(business) == set(membership)
    assert file_hash(source / 'business_split.csv') == file_hash(package / 'business_split.csv')
    raw_counts, english_counts, unknown_counts, nonenglish_counts = Counter(), Counter(), Counter(), Counter()
    language_counts, star_counts, year_counts, business_stars = Counter(), Counter(), Counter(), Counter()
    business_token_sums, business_token_max = Counter(), Counter()
    english_ids, encountered = set(), set()
    split_counts = {s: Counter() for s in ('train', 'validation', 'test')}
    source_english_digest = {s: hashlib.sha256() for s in split_counts}
    output_english_digest = {s: hashlib.sha256() for s in split_counts}
    source_manifest = json.loads((source / 'dataset_manifest.json').read_text(encoding='utf-8'))
    source_hashes = json.loads((source / 'checksums.json').read_text(encoding='utf-8'))
    token_identity = json.loads((out / 'scientific_identity.json').read_text(encoding='utf-8'))['identity']
    tokenizer_path = ROOT / 'assets/tokenizers/led-base-16384/tokenizer.json'
    assert file_hash(tokenizer_path) == token_identity['tokenizer_assets_sha256']['tokenizer.json']
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation(); tokenizer.no_padding()
    _initialize(token_identity['language_policy'])
    # Deterministic train/validation token samples; test is only used for integrity joins/counts.
    allowed = sorted(rid for rid, row in meta.items() if row['split'] in ('train', 'validation'))
    token_sample_ids = set(random.Random(42).sample(allowed, 256))
    language_sample_ids = set(sorted(token_sample_ids)[:32])
    sample_tokens_checked, sample_languages_checked = 0, 0
    print('Comparing all review metadata and English exports with immutable source records...', flush=True)
    for split in split_counts:
        digest = hashlib.sha256()
        with (source / 'reviews' / f'{split}.jsonl').open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip(): continue
                row = orjson.loads(line)
                rid, bid = row['review_id'], row['business_id']
                assert rid in meta and rid not in encountered
                encountered.add(rid)
                m = meta[rid]
                assert m['business_id'] == bid and m['split'] == membership[bid] == split
                assert int(m['stars']) == row['stars'] and m['date'] == row['date']
                assert m['text_sha256'] == hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
                assert int(m['led_tokens_with_special']) == int(m['led_tokens_without_special']) + report['special_tokens_per_single_review']
                assert int(m['led_tokens_without_special']) >= 0
                raw_counts[bid] += 1; split_counts[split]['raw_reviews'] += 1
                language_counts[(split, m['language'])] += 1
                if m['language'] == 'en':
                    english_counts[bid] += 1; split_counts[split]['english_reviews'] += 1
                    english_ids.add(rid); source_english_digest[split].update(canonical(row))
                    star_counts[(split, int(row['stars']))] += 1
                    business_stars[(bid, int(row['stars']))] += 1
                    year_counts[(split, int(row['date'][:4]))] += 1
                    token_count = int(m['led_tokens_without_special'])
                    business_token_sums[bid] += token_count
                    business_token_max[bid] = max(business_token_max[bid], token_count)
                elif m['language'] == 'unknown':
                    unknown_counts[bid] += 1; split_counts[split]['unknown_reviews'] += 1
                else:
                    nonenglish_counts[bid] += 1; split_counts[split]['non_english_reviews'] += 1
                if rid in token_sample_ids:
                    assert len(tokenizer.encode(row['text'], add_special_tokens=False).ids) == int(m['led_tokens_without_special'])
                    assert len(tokenizer.encode(row['text'], add_special_tokens=True).ids) == int(m['led_tokens_with_special'])
                    sample_tokens_checked += 1
                if rid in language_sample_ids:
                    assert classify_text(row['text']) == (m['language'], m['language_reason'])
                    sample_languages_checked += 1
        assert digest.hexdigest() == source_hashes[f'reviews/{split}.jsonl']['sha256']
    assert encountered == set(meta)
    assert sample_tokens_checked == 256 and sample_languages_checked == 32
    exported_ids = set()
    for split in split_counts:
        export_count = 0
        with (package / 'reviews' / f'{split}.jsonl').open('rb') as stream:
            for line in stream:
                if not line.strip(): continue
                row = orjson.loads(line); rid = row['review_id']; bid = row['business_id']
                assert rid in english_ids and rid not in exported_ids and membership[bid] == split
                exported_ids.add(rid); export_count += 1
                output_english_digest[split].update(canonical(row))
        assert source_english_digest[split].digest() == output_english_digest[split].digest()
        assert export_count == split_counts[split]['english_reviews']
        expected = report['split_summary'][split]
        for key in ('raw_reviews', 'english_reviews', 'unknown_reviews', 'non_english_reviews'):
            assert split_counts[split][key] == expected[key]
    assert exported_ids == english_ids
    for bid, b in business.items():
        assert int(b['raw_reviews']) == raw_counts[bid]
        assert int(b['english_reviews']) == english_counts[bid] >= 10
        assert int(b['unknown_reviews']) == unknown_counts[bid]
        assert int(b['non_english_reviews']) == nonenglish_counts[bid]
        for star in range(1, 6):
            assert int(b[f'english_star_{star}']) == business_stars[(bid, star)]
        assert int(b['english_token_sum_individual_reviews']) == business_token_sums[bid]
        assert int(b['english_review_token_max']) == business_token_max[bid]
    for filename, field, counter, cast in (
        ('language_counts_by_split.csv', 'language', language_counts, str),
        ('english_star_counts_by_split.csv', 'stars', star_counts, int),
        ('english_year_counts_by_split.csv', 'year', year_counts, int),
    ):
        rows = csv_rows(out / filename)
        observed = {(row['split'], cast(row[field])): int(row['reviews']) for row in rows}
        assert len(observed) == len(rows) and observed == dict(counter), filename
    for row in csv_rows(out / 'english_token_summary_by_split.csv'):
        assert int(float(row['count'])) == split_counts[row['split']]['english_reviews']
    totals = {'raw_reviews': sum(raw_counts.values()), 'english_reviews': sum(english_counts.values()),
              'unknown_reviews': sum(unknown_counts.values()), 'non_english_reviews': sum(nonenglish_counts.values())}
    assert totals['raw_reviews'] == totals['english_reviews'] + totals['unknown_reviews'] + totals['non_english_reviews']
    for key, count in totals.items(): assert report[key] == count
    result = {'validation_passed': True, **totals, 'businesses': len(membership),
              'minimum_english_reviews_per_business': min(english_counts.values()),
              'all_original_review_ids_fields_text_and_export_order_verified': True,
              'source_review_hashes_and_artifact_hashes_verified': True,
              'language_star_year_and_token_count_reports_verified': True,
              'token_samples_reencoded': sample_tokens_checked, 'language_samples_reclassified': sample_languages_checked,
              'unchanged_split_sha256': source_manifest['business_split_sha256'],
              'overall_stage_c_complete': False, 'c2_final_selection_pending': True,
              'elapsed_seconds': round(time.perf_counter() - started, 3)}
    (out / 'validation_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
