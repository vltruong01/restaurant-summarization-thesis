"""Stage C1: exhaustive cohort language classification and LED length metadata."""
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import time

import orjson
import pandas as pd
import yaml
from english_eligibility import _initialize, classify_text, file_hash

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')
TOKENIZER = None
FIELDS = ['review_id', 'business_id', 'split', 'stars', 'date', 'text_sha256', 'language', 'language_reason', 'led_tokens_without_special']


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_read(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def stats(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def initialize_worker(policy, tokenizer_path):
    global TOKENIZER
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _initialize(policy)
    from tokenizers import Tokenizer
    TOKENIZER = Tokenizer.from_file(tokenizer_path)
    TOKENIZER.no_truncation()
    TOKENIZER.no_padding()


def process_review(task):
    rid, bid, split, stars, date, text, text_hash, prior_language, prior_reason = task
    language, reason = (prior_language, prior_reason) if prior_language else classify_text(text)
    count = len(TOKENIZER.encode(text, add_special_tokens=False).ids)
    return rid, bid, split, stars, date, text_hash, language, reason, count


def run_diagnostics(root=ROOT):
    root = Path(root)
    config_path = root / 'configs/review_diagnostics.yaml'
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    project = yaml.safe_load((root / 'configs/project.yaml').read_text(encoding='utf-8'))
    policy_cfg = yaml.safe_load((root / cfg['language_policy_source']).read_text(encoding='utf-8'))
    policy = policy_cfg['language_detector']
    if version('langdetect') != policy['version']:
        raise RuntimeError('Language detector version mismatch')
    source = root / cfg['dataset_directory']
    out = root / cfg['output_directory']
    package = root / cfg['english_pool_directory']
    out.mkdir(parents=True, exist_ok=True)
    package.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source / 'dataset_manifest.json').read_text(encoding='utf-8'))
    source_checksums = json.loads((source / 'checksums.json').read_text(encoding='utf-8'))
    tokenizer_manifest = json.loads((root / 'assets/tokenizers/led-base-16384.manifest.json').read_text(encoding='utf-8'))
    tokenizer_dir = root / project['tokenizer']['assets_directory']
    for name, expected in tokenizer_manifest['sha256'].items():
        if file_hash(tokenizer_dir / name) != expected:
            raise RuntimeError(f'Tokenizer asset mismatch: {name}')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / 'tokenizer.json'))
    special_tokens = tokenizer.num_special_tokens_to_add(is_pair=False)
    input_paths = {split: source / 'reviews' / f'{split}.jsonl' for split in SPLITS}
    before = {split: stats(path) for split, path in input_paths.items()}
    identity = {'review_sources': {split: source_checksums[f'reviews/{split}.jsonl']['sha256'] for split in SPLITS},
                'business_split_sha256': source_manifest['business_split_sha256'], 'language_policy': policy,
                'tokenizer_revision': tokenizer_manifest['revision'], 'tokenizer_assets_sha256': tokenizer_manifest['sha256'],
                'tokenizers_version': version('tokenizers'), 'base_add_special_tokens': False, 'special_tokens_per_single_review': special_tokens,
                'normalization': 'unchanged original text', 'truncation': False, 'padding': False}
    identity_hash = hashlib.sha256(orjson.dumps(identity, option=orjson.OPT_SORT_KEYS)).hexdigest()
    completion = out / 'completion_report.json'
    if completion.exists():
        old = json.loads(completion.read_text(encoding='utf-8'))
        if old['identity_sha256'] != identity_hash or old['source_stats'] != before:
            raise RuntimeError('Finished diagnostics have a different input identity')
        for relative, expected in old['diagnostic_artifact_sha256'].items():
            if file_hash(out / relative) != expected:
                raise RuntimeError(f'Diagnostic artifact changed: {relative}')
        for relative, expected in old['english_pool_artifact_sha256'].items():
            if file_hash(package / relative) != expected:
                raise RuntimeError(f'English pool artifact changed: {relative}')
        print('Using completed exhaustive review diagnostics.', flush=True)
        return old
    if package.exists():
        raise FileExistsError(f'English pool exists without matching completed report: {package}')
    old_cache_path = (root / cfg['threshold_cache']).resolve()
    old_db = sqlite3.connect(f'file:{old_cache_path.as_posix()}?mode=ro', uri=True)
    old_identity = old_db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    gate = json.loads((root / 'outputs/english_eligibility/eligibility_report.json').read_text(encoding='utf-8'))
    if old_identity[0] != gate['identity_sha256'] or gate['identity']['language_detector'] != policy:
        raise RuntimeError('Threshold cache policy identity mismatch')
    business_rows = csv_read(source / 'business_split.csv')
    membership = {row['business_id']: row['split'] for row in business_rows}
    if len(business_rows) != len(membership):
        raise RuntimeError('Business split IDs invalid')
    if len(membership) != 2300 or file_hash(source / 'business_split.csv') != identity['business_split_sha256']:
        raise RuntimeError('Source split changed')
    prior = {row[0]: row for row in old_db.execute('SELECT * FROM cache') if row[1] in membership}
    old_db.close()
    db = sqlite3.connect(out / 'review_metadata.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS reviews (review_id TEXT PRIMARY KEY, business_id TEXT, split TEXT, stars INTEGER, date TEXT, text_sha256 TEXT, language TEXT, language_reason TEXT, led_tokens_without_special INTEGER) WITHOUT ROWID')
    stored_identity = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if stored_identity and stored_identity[0] != identity_hash:
        db.close()
        raise RuntimeError('Review metadata cache identity mismatch')
    db.execute("INSERT OR IGNORE INTO meta VALUES ('identity', ?)", (identity_hash,))
    db.commit()
    already_done = {r[0] for r in db.execute('SELECT review_id FROM reviews')}
    seen, raw_counts = set(), Counter()
    started = time.perf_counter()
    scheduled, prior_reused, resumed, total_done = 0, 0, 0, 0
    pending = []
    workers = cfg['workers']
    if not isinstance(workers, int) or not 1 <= workers <= 12:
        raise ValueError('Invalid workers')
    print(f'Full corpus: {source_manifest["raw_reviews"]:,} reviews; {len(prior):,} prior cohort language labels; {workers} workers.', flush=True)
    try:
        with mp.get_context('spawn').Pool(workers, initializer=initialize_worker,
                                        initargs=(policy, str(tokenizer_dir / 'tokenizer.json'))) as pool:
            def flush():
                nonlocal total_done
                if not pending:
                    return
                results = list(pool.imap(process_review, pending, chunksize=32))
                db.executemany('INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?)', results)
                db.commit()
                total_done += len(results)
                pending.clear()
                progress = {'completed_reviews': total_done, 'expected_reviews': source_manifest['raw_reviews'],
                            'new_tasks_scheduled': scheduled, 'prior_languages_reused': prior_reused,
                            'metadata_rows_resumed': resumed, 'elapsed_seconds': round(time.perf_counter() - started, 1)}
                json_write(out / 'run_progress.json', progress)
                if total_done // 20000 != (total_done - len(results)) // 20000:
                    print(f'Reviews fully processed: {total_done:,}/{source_manifest["raw_reviews"]:,}', flush=True)

            for split, path in input_paths.items():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for line in stream:
                        digest.update(line)
                        if not line.strip():
                            continue
                        row = orjson.loads(line)
                        rid, bid, text = row['review_id'], row['business_id'], row['text']
                        if rid in seen or membership.get(bid) != split or not isinstance(text, str) or not text.strip():
                            raise RuntimeError('Invalid review ID/join/text')
                        if isinstance(row['stars'], bool) or row['stars'] not in (1, 2, 3, 4, 5):
                            raise RuntimeError('Invalid review stars')
                        datetime.fromisoformat(row['date'])
                        seen.add(rid)
                        raw_counts[bid] += 1
                        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                        if rid in already_done:
                            existing = db.execute('SELECT business_id, split, stars, date, text_sha256 FROM reviews WHERE review_id=?', (rid,)).fetchone()
                            if existing != (bid, split, int(row['stars']), row['date'], text_hash):
                                raise RuntimeError('Resumed cached review content mismatch')
                            resumed += 1
                            total_done += 1
                            continue
                        previous = prior.get(rid)
                        if previous and (previous[1] != bid or previous[4] != text_hash):
                            raise RuntimeError('Prior language label content mismatch')
                        prior_reused += previous is not None
                        pending.append((rid, bid, split, int(row['stars']), row['date'], text, text_hash,
                                        previous[2] if previous else None, previous[3] if previous else None))
                        scheduled += 1
                        if len(pending) >= cfg['batch_size']:
                            flush()
                if digest.hexdigest() != identity['review_sources'][split]:
                    raise RuntimeError(f'Source review file hash mismatch: {split}')
            flush()
        if len(seen) != source_manifest['raw_reviews'] or db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0] != len(seen):
            raise RuntimeError('Metadata row coverage mismatch')
        if any(raw_counts[r['business_id']] != int(r['raw_review_records']) for r in business_rows):
            raise RuntimeError('Per-business raw count mismatch')
        if before != {split: stats(path) for split, path in input_paths.items()}:
            raise RuntimeError('Source changed during full classification')
        json_write(out / 'scientific_identity.json', {'identity_sha256': identity_hash, 'identity': identity})
        (out / 'config_at_run.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        frame = pd.read_sql_query('SELECT * FROM reviews ORDER BY split, business_id, date, review_id', db)
        frame['led_tokens_with_special'] = frame['led_tokens_without_special'] + special_tokens
        frame.to_csv(out / 'review_language_token_metadata.csv', index=False)
        frame['is_english'] = frame['language'].eq('en')
        frame['is_unknown'] = frame['language'].eq('unknown')
        frame['year'] = frame['date'].str[:4].astype(int)
        business = pd.DataFrame(business_rows).set_index('business_id')
        business['raw_reviews'] = frame.groupby('business_id').size()
        business['english_reviews'] = frame.groupby('business_id')['is_english'].sum()
        business['unknown_reviews'] = frame.groupby('business_id')['is_unknown'].sum()
        business['non_english_reviews'] = business['raw_reviews'] - business['english_reviews'] - business['unknown_reviews']
        english_frame = frame.loc[frame['is_english']]
        if business['english_reviews'].min() < 10:
            raise RuntimeError('Full classification contradicts the English eligibility gate')
        english_tokens = english_frame.groupby('business_id')['led_tokens_without_special']
        business['english_token_sum_individual_reviews'] = english_tokens.sum()
        business['english_review_token_mean'] = english_tokens.mean()
        business['english_review_token_median'] = english_tokens.median()
        business['english_review_token_max'] = english_tokens.max()
        business['first_english_review_date'] = english_frame.groupby('business_id')['date'].min()
        business['last_english_review_date'] = english_frame.groupby('business_id')['date'].max()
        english_stars = pd.crosstab(english_frame['business_id'], english_frame['stars']).reindex(columns=range(1, 6), fill_value=0)
        for star in range(1, 6):
            business[f'english_star_{star}'] = english_stars[star].reindex(business.index, fill_value=0)
        names = pd.read_csv(source / 'cohort_businesses.csv').set_index('business_id')[['name', 'city', 'state']]
        business = business.join(names)
        business.reset_index().to_csv(out / 'business_review_statistics.csv', index=False)
        frame.groupby(['split', 'language']).size().rename('reviews').to_csv(out / 'language_counts_by_split.csv')
        english_frame.groupby(['split', 'stars']).size().rename('reviews').to_csv(out / 'english_star_counts_by_split.csv')
        english_frame.groupby(['split', 'year']).size().rename('reviews').to_csv(out / 'english_year_counts_by_split.csv')
        token_summary = english_frame.groupby('split')['led_tokens_without_special'].describe(percentiles=[.5, .9, .95, .99])
        token_summary.to_csv(out / 'english_token_summary_by_split.csv')
        staging = package.parent / f'.{package.name}.building'
        if staging.exists():
            raise FileExistsError(f'Previous export staging exists: {staging}')
        staging.mkdir()
        (staging / 'reviews').mkdir()
        shutil.copyfile(source / 'business_split.csv', staging / 'business_split.csv')
        shutil.copyfile(source / 'businesses.jsonl', staging / 'businesses.jsonl')
        decision_lookup = dict(zip(frame['review_id'], frame['language']))
        export_counts = Counter()
        export_by_business = Counter()
        with ExitStack() as stack:
            destinations = {split: stack.enter_context((staging / 'reviews' / f'{split}.jsonl').open('wb')) for split in SPLITS}
            for split, path in input_paths.items():
                with path.open('rb') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = orjson.loads(line)
                        if decision_lookup[row['review_id']] == 'en':
                            destinations[split].write(orjson.dumps(row) + b'\n')
                            export_counts[split] += 1
                            export_by_business[row['business_id']] += 1
        if any(export_by_business[bid] != int(count) for bid, count in business['english_reviews'].items()):
            raise RuntimeError('English export counts disagree with complete metadata')
        counts = {split: {'businesses': int(business['split'].eq(split).sum()),
                          'raw_reviews': int(frame['split'].eq(split).sum()), 'english_reviews': export_counts[split],
                          'unknown_reviews': int((frame['split'].eq(split) & frame['is_unknown']).sum()),
                          'non_english_reviews': int((frame['split'].eq(split) & ~frame['is_english'] & ~frame['is_unknown']).sum())} for split in SPLITS}
        english_manifest = {'stage': 'C1_full_english_review_pool', 'source_dataset': '../yelp_2300_v1',
                            'photos_reference': '../yelp_2300_v1/photos', 'business_split_sha256': identity['business_split_sha256'],
                            'identity_sha256': identity_hash, 'language_policy': policy, 'split_summary': counts,
                            'businesses': 2300, 'english_reviews': sum(export_counts.values()),
                            'all_cohort_reviews_language_classified': True, 'representative_30_review_selection_applied': False,
                            'final_led_inputs_ready': False, 'tokenizer': project['tokenizer']}
        json_write(staging / 'dataset_manifest.json', english_manifest)
        (staging / 'README.md').write_text(
            '# Exhaustive English review pool (Stage C1)\n\n'
            'All source reviews of the unchanged 2,300-business cohort have language/token metadata. '
            'reviews/{split}.jsonl preserves complete original records classified en by the recorded policy. '
            'This is the complete English pool, not the final <=30-review inputs or LED training data. '
            'Business IDs/split match v1; original photos remain in ../yelp_2300_v1/photos. '
            'Per-review token counts exclude BOS/EOS; sums are diagnostics and must not replace exact tokenization of final marked inputs.\n', encoding='utf-8')
        package_hashes = {p.relative_to(staging).as_posix(): file_hash(p) for p in staging.rglob('*') if p.is_file()}
        json_write(staging / 'checksums.json', package_hashes)
        staging.rename(package)
        artifacts = ['review_language_token_metadata.csv', 'business_review_statistics.csv', 'language_counts_by_split.csv',
                     'english_star_counts_by_split.csv', 'english_year_counts_by_split.csv', 'english_token_summary_by_split.csv',
                     'scientific_identity.json', 'config_at_run.yaml']
        report = {'stage': 'C1', 'identity_sha256': identity_hash, 'source_stats': before,
                  'raw_reviews': len(frame), 'english_reviews': int(frame['is_english'].sum()),
                  'unknown_reviews': int(frame['is_unknown'].sum()), 'non_english_reviews': int((~frame['is_english'] & ~frame['is_unknown']).sum()),
                  'businesses': len(business), 'minimum_english_reviews_per_business': int(business['english_reviews'].min()),
                  'all_review_language_counts_exact': True, 'split_summary': counts,
                  'prior_language_labels_reused': prior_reused, 'resumed_metadata_rows': resumed,
                  'english_pool_directory': str(package), 'selection_ready': False, 'overall_stage_c_complete': False,
                  'diagnostic_artifact_sha256': {name: file_hash(out / name) for name in artifacts},
                  'english_pool_artifact_sha256': package_hashes,
                  'special_tokens_per_single_review': special_tokens, 'source_stats_unchanged': True,
                  'elapsed_seconds': round(time.perf_counter() - started, 3)}
        json_write(completion, report)
        json_write(out / 'run_progress.json', {'completed_reviews': len(frame), 'expected_reviews': len(frame),
                                              'stage': 'C1_export_complete_pending_independent_validation'})
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        db.close()


if __name__ == '__main__':
    run_diagnostics()
