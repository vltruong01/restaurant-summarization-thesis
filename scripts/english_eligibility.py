"""Certify English eligibility of every original-pool business before sampling."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import multiprocessing as mp
import random
import sqlite3
import time
from importlib.metadata import version

import orjson
import yaml
from data_inspection import photo_inventory, resolve_data_root, stream_jsonl

ROOT = Path(__file__).resolve().parents[1]
_POLICY = None


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    if not rows:
        raise ValueError('Cannot infer columns from an empty table')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _initialize(policy):
    global _POLICY
    from langdetect import DetectorFactory
    DetectorFactory.seed = policy['seed']
    _POLICY = policy


def classify_text(text):
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
    if not isinstance(text, str):
        return 'unknown', 'non_string_text'
    stripped = text.strip()
    if len(stripped) < _POLICY['minimum_stripped_characters']:
        return 'unknown', 'short_text'
    try:
        return detect(stripped[:_POLICY['maximum_probe_characters']]), 'detected'
    except LangDetectException:
        return 'unknown', 'no_language_features'


def _classify(task):
    rid, bid, text, text_hash = task
    language, reason = classify_text(text)
    return rid, bid, language, reason, text_hash


def _stat(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def run_english_gate(project_root=ROOT):
    root = Path(project_root)
    cfg = yaml.safe_load((root / 'configs/english_eligibility.yaml').read_text(encoding='utf-8'))
    project = yaml.safe_load((root / 'configs/project.yaml').read_text(encoding='utf-8'))
    data = resolve_data_root(root, project)
    policy = cfg['language_detector']
    if version('langdetect') != policy['version']:
        raise RuntimeError('Detector version differs from the recorded policy')
    baseline_path = root / 'outputs/cohort/verified_eligible_businesses.csv'
    verified = json.loads((root / 'outputs/cohort/eligibility_verified.json').read_text(encoding='utf-8'))
    if file_hash(baseline_path) != verified['verified_eligible_csv_sha256']:
        raise RuntimeError('Baseline pool changed')
    rows = csv_read(baseline_path)
    baseline = {r['business_id']: r for r in rows}
    if len(rows) != len(baseline):
        raise RuntimeError('Baseline IDs are not unique')
    if len(baseline) != cfg['baseline_businesses']:
        raise RuntimeError('Baseline count mismatch')
    old_manifest = json.loads((data / 'processed/yelp_2300_v1/dataset_manifest.json').read_text(encoding='utf-8'))
    sources = {kind: data / project['paths'][f'{kind}_jsonl'] for kind in ('business', 'review', 'photo')}
    source_stats = {kind: _stat(path) for kind, path in sources.items()}
    if source_stats != verified['source_stats_at_verification']:
        raise RuntimeError('Raw sources changed since baseline verification')
    filenames, inventory = photo_inventory(data / project['paths']['photo_directory'])
    unique_photos, seen_photos = Counter(), set()
    for row in stream_jsonl(sources['photo']):
        pid, bid = row['photo_id'], row['business_id']
        if pid not in seen_photos:
            seen_photos.add(pid)
            if bid in baseline and pid in filenames:
                unique_photos[bid] += 1
    if any(unique_photos[bid] != int(row['local_photo_records']) for bid, row in baseline.items()):
        raise RuntimeError('Baseline local photo counts changed')
    del filenames, seen_photos
    identity = {'source_review_sha256': old_manifest['raw_sources']['review']['sha256'],
                'baseline_sha256': file_hash(baseline_path), 'language_detector': policy,
                'minimum_english_reviews': cfg['minimum_english_reviews'],
                'minimum_local_photos': cfg['minimum_local_photos']}
    signature = hashlib.sha256(orjson.dumps(identity, option=orjson.OPT_SORT_KEYS)).hexdigest()
    out = root / cfg['execution']['cache_directory']
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'eligibility_report.json'
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        if old['identity_sha256'] != signature:
            raise RuntimeError('Existing report uses different settings; use a new cache directory')
        for relative, expected in old['artifact_sha256'].items():
            if file_hash(out / relative) != expected:
                raise RuntimeError(f'Report artifact changed: {relative}')
        print('Reusing completed full-pool eligibility report; source stats/photo counts rechecked.', flush=True)
        return old
    started = time.perf_counter()
    connection = sqlite3.connect(out / 'review_language_cache.sqlite')
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    connection.execute('CREATE TABLE IF NOT EXISTS cache (review_id TEXT PRIMARY KEY, business_id TEXT, language TEXT, reason TEXT, text_sha256 TEXT) WITHOUT ROWID')
    existing = connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if existing and existing[0] != signature:
        connection.close()
        raise RuntimeError('Existing cache has another detector/source identity')
    connection.execute("INSERT OR IGNORE INTO meta VALUES ('identity', ?)", (signature,))
    connection.commit()
    cached = {r[0]: r for r in connection.execute('SELECT * FROM cache')}
    english, assessed, raw_counts, assessed_labels = Counter(), Counter(), Counter(), Counter()
    pending, pending_counts = [], Counter()
    threshold = cfg['minimum_english_reviews']
    if threshold != 10 or cfg['minimum_local_photos'] != 5:
        raise RuntimeError('Eligibility policy differs from the approved amendment')
    workers = cfg['execution']['workers']
    if not isinstance(workers, int) or not 1 <= workers <= 12:
        raise ValueError('workers must be 1..12')
    digest = hashlib.sha256()
    source_records = 0
    classified_new = 0
    print(f'Certifying all {len(baseline):,} businesses; >=10 English reviews, {workers} CPU workers.', flush=True)
    try:
        with mp.get_context('spawn').Pool(workers, initializer=_initialize, initargs=(policy,)) as pool:
            def apply(result):
                rid, bid, language, reason, text_hash = result
                assessed[bid] += 1
                assessed_labels[language] += 1
                english[bid] += language == policy['accepted_language']

            def flush():
                nonlocal classified_new
                if not pending:
                    return
                results = list(pool.imap(_classify, pending, chunksize=8))
                connection.executemany('INSERT INTO cache VALUES (?,?,?,?,?)', results)
                connection.commit()
                for result in results:
                    apply(result)
                classified_new += len(results)
                pending.clear()
                pending_counts.clear()
                status = {'raw_records_scanned': source_records, 'new_reviews_classified': classified_new,
                          'reviews_assessed': sum(assessed.values()), 'businesses_certified': sum(n >= threshold for n in english.values()),
                          'baseline_businesses': len(baseline), 'elapsed_seconds': round(time.perf_counter() - started, 1)}
                write_json(out / 'run_progress.json', status)

            with sources['review'].open('rb') as stream:
                for line in stream:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = orjson.loads(line)
                    source_records += 1
                    bid = row['business_id']
                    if bid not in baseline:
                        continue
                    raw_counts[bid] += 1
                    if english[bid] >= threshold:
                        continue
                    # Resolve pending work before skipping a possibly needed row.
                    if english[bid] + pending_counts[bid] >= threshold:
                        flush()
                        if english[bid] >= threshold:
                            continue
                    rid, text = row['review_id'], row['text']
                    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    if rid in cached:
                        result = cached[rid]
                        if result[1] != bid or result[4] != text_hash:
                            raise RuntimeError('Cached review content/ownership changed')
                        apply(result)
                    else:
                        pending.append((rid, bid, text, text_hash))
                        pending_counts[bid] += 1
                    if len(pending) >= cfg['execution']['batch_size']:
                        flush()
            flush()
    finally:
        connection.close()
    if digest.hexdigest() != identity['source_review_sha256']:
        raise RuntimeError('Full raw review SHA-256 changed')
    if any(raw_counts[bid] != int(row['raw_review_records']) for bid, row in baseline.items()):
        raise RuntimeError('Full raw baseline per-business review counts differ')
    if source_stats != {kind: _stat(path) for kind, path in sources.items()}:
        raise RuntimeError('Source changed during scan')
    decisions = []
    for bid in sorted(baseline):
        passed = english[bid] >= threshold and unique_photos[bid] >= cfg['minimum_local_photos']
        exact = assessed[bid] == raw_counts[bid]
        if not passed and not exact:
            raise RuntimeError('Failed business was not exhaustively assessed')
        decisions.append({'business_id': bid, 'raw_review_records': raw_counts[bid],
                          'local_photo_records': unique_photos[bid], 'reviews_language_assessed': assessed[bid],
                          'english_review_count_lower_bound': english[bid],
                          'unassessed_reviews': raw_counts[bid] - assessed[bid], 'english_count_is_exact': exact,
                          'eligible_english': passed})
    passed_rows = [r for r in decisions if r['eligible_english']]
    failed_rows = [r for r in decisions if not r['eligible_english']]
    csv_write(out / 'business_english_decisions.csv', decisions)
    csv_write(out / 'english_eligible_businesses.csv', passed_rows)
    # Explicit header even if nobody fails.
    with (out / 'english_ineligible_businesses.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
        writer.writeheader(); writer.writerows(failed_rows)
    if len(passed_rows) < cfg['sampling_preview']['businesses']:
        raise RuntimeError('Fewer than 2,300 businesses satisfy English eligibility')
    by_id = {r['business_id']: r for r in decisions}
    old_split = csv_read(data / 'processed/yelp_2300_v1/business_split.csv')
    comparison = [{'business_id': r['business_id'], 'old_split': r['split'],
                   **{k: v for k, v in by_id[r['business_id']].items() if k != 'business_id'}} for r in old_split]
    csv_write(out / 'current_cohort_comparison.csv', comparison)
    selected = random.Random(42).sample(sorted(r['business_id'] for r in passed_rows), 2300)
    proposal = []
    for index, bid in enumerate(selected):
        proposal.append({'business_id': bid, 'split': 'train' if index < 2000 else 'validation' if index < 2200 else 'test',
                         'selection_index': index, **{k: v for k, v in by_id[bid].items() if k != 'business_id'}})
    csv_write(out / 'proposed_business_split.csv', sorted(proposal, key=lambda r: r['business_id']))
    old_ids, new_ids = {r['business_id'] for r in old_split}, set(selected)
    old_membership = {r['business_id']: r['split'] for r in old_split}
    snapshot = out / 'config_at_run.yaml'
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    artifacts = ['business_english_decisions.csv', 'english_eligible_businesses.csv', 'english_ineligible_businesses.csv',
                 'current_cohort_comparison.csv', 'proposed_business_split.csv', 'config_at_run.yaml']
    report = {'identity_sha256': signature, 'identity': identity, 'all_businesses_decided': True,
              'full_raw_review_file_scanned': True, 'full_raw_review_sha256_verified': True,
              'raw_review_records_scanned': source_records, 'baseline_businesses': len(baseline),
              'baseline_raw_reviews': sum(raw_counts.values()), 'reviews_language_assessed': sum(assessed.values()),
              'assessed_label_counts': dict(assessed_labels), 'all_reviews_language_classified': all(r['english_count_is_exact'] for r in decisions),
              'english_eligible_businesses': len(passed_rows), 'english_ineligible_businesses': len(failed_rows),
              'current_cohort_ineligible_businesses': sum(not r['eligible_english'] for r in comparison),
              'proposed_cohort_businesses': len(proposal), 'proposed_split_counts': dict(Counter(r['split'] for r in proposal)),
              'old_cohort_businesses_removed': len(old_ids - new_ids), 'new_cohort_businesses_added': len(new_ids - old_ids),
              'retained_businesses_with_changed_split': sum(r['business_id'] in old_ids and old_membership[r['business_id']] != r['split'] for r in proposal),
              'creates_physical_dataset': False, 'v1_package_and_split_overwritten': False,
              'artifact_sha256': {name: file_hash(out / name) for name in artifacts},
              'elapsed_seconds': round(time.perf_counter() - started, 3),
              'scope_note': 'Counts for certified businesses are lower bounds unless exactness flag is true. Assessed label distribution is not a representative/full-corpus language distribution.'}
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run_english_gate()
