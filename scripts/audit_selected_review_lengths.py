"""Read-only length audit and same-cell counterfactuals; never reselect evidence."""
from pathlib import Path
from collections import Counter
import hashlib, html, json, time
import numpy as np
import pandas as pd
import orjson, yaml
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'outputs/stage_c/review_selection_train_validation_v3'
C1 = ROOT / 'outputs/stage_c/review_diagnostics_v1'
SOURCE = ROOT / 'data/processed/yelp_2300_english_v1'
OUT = ROOT / 'outputs/stage_c/review_selection_length_audit_v1'
BUDGETS = [3500, 4000, 4500]
METHODS = ['budget_aware', 'rating_time_delete_only', 'random_delete_only']


def digest(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsd(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    a /= a.sum(); b /= b.sum(); m = (a + b) / 2
    def kl(p):
        mask = p > 0
        return np.sum(p[mask] * np.log2(p[mask] / m[mask]))
    return float((kl(a) + kl(b)) / 2)


def length_hist(lengths):
    return np.bincount(np.searchsorted([20, 50, 100, 200], lengths, side='right'), minlength=5)


def summary(frame, group_columns):
    result = []
    for key, g in frame.groupby(group_columns, sort=True):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(group_columns, key))
        row.update(businesses=len(g), selected_review_occurrences=int(g.selected_count.sum()),
                   source_mean_macro=float(g.source_mean.mean()), selected_mean_macro=float(g.selected_mean.mean()),
                   selected_body_mean_pooled=float(g.selected_body_sum.sum() / g.selected_count.sum()),
                   selected_source_mean_ratio_median=float(g.mean_ratio.median()),
                   selected_source_mean_ratio_mean=float(g.mean_ratio.mean()),
                   mean_ratio_below_0_5=int((g.mean_ratio < .5).sum()),
                   mean_ratio_below_0_75=int((g.mean_ratio < .75).sum()),
                   mean_ratio_below_1=int((g.mean_ratio < 1).sum()),
                   selected_midrank_percentile_macro=float(g.selected_length_percentile.mean()),
                   source_short20_macro=float(g.source_lt20_fraction.mean()),
                   selected_short20_macro=float(g.selected_lt20_fraction.mean()),
                   source_short30_macro=float(g.source_lt30_fraction.mean()),
                   selected_short30_macro=float(g.selected_lt30_fraction.mean()),
                   source_short50_macro=float(g.source_lt50_fraction.mean()),
                   selected_short50_macro=float(g.selected_lt50_fraction.mean()),
                   length_jsd_macro=float(g.length_jsd.mean()),
                   exactly30=int(g.selected_count.eq(30).sum()),
                   exactly30_under1500=int((g.selected_count.eq(30) & g.selected_block_tokens.lt(1500)).sum()),
                   exactly30_under2000=int((g.selected_count.eq(30) & g.selected_block_tokens.lt(2000)).sum()))
        result.append(row)
    return pd.DataFrame(result)


def choose_examples(df):
    x = df.loc[(df.method == 'budget_aware') & (df.review_budget == 3500)]
    ids = x.loc[x.selected_count.eq(30)].sort_values(['selected_block_tokens', 'business_id']).business_id.head(5).tolist()
    labels = {bid: 'lowest_token_exactly30' for bid in ids}
    for label, rank, target in [
        ('low_selected_source_mean_ratio', x.sort_values(['mean_ratio', 'business_id']), 7),
        ('low_retained_count', x.sort_values(['selected_count', 'selected_block_tokens', 'business_id']), 9),
        ('high_ratio_control', x.sort_values(['mean_ratio', 'business_id'], ascending=[False, True]), 10),
    ]:
        for bid in rank.business_id:
            if bid not in ids: ids.append(bid); labels[bid] = label
            if len(ids) == target: break
    assert len(ids) == len(set(ids)) == 10
    return ids, labels


def make_block(ids, records, cfg):
    ids = sorted(ids, key=lambda rid: (records[rid]['date'], rid))
    fmt = cfg['format']
    return fmt['separator'].join(fmt['review_template'].format(
        index=i, stars=int(records[rid]['stars']), date=records[rid]['date'], text=records[rid]['text'])
        for i, rid in enumerate(ids, 1)), ids


def counterfactual(run, pop, rows, tokenizer, cfg):
    records = {r['review_id']: r for r in pop}
    rank = sorted(records, key=lambda rid: (records[rid]['date'], rid)); n = len(rank)
    cells = {}
    for b in range(5):
        for rid in rank[b*n//5:(b+1)*n//5]: cells[rid] = (int(records[rid]['stars']), b)
    lengths = rows.set_index('review_id').led_tokens_without_special.to_dict()
    current = list(run['selected_review_ids']); old = current.copy(); moves = []
    target = float(np.median(list(lengths.values())))
    for removed in sorted(old, key=lambda rid: (lengths[rid], records[rid]['date'], rid))[:5]:
        offers = [rid for rid in rank if rid not in current and cells[rid] == cells[removed]
                  and lengths[rid] >= 2 * lengths[removed] and lengths[rid] > lengths[removed]]
        offers.sort(key=lambda rid: (abs(lengths[rid] - target), abs(rank.index(rid)-rank.index(removed)), rid))
        for added in offers[:20]:
            block, ids = make_block([rid for rid in current if rid != removed] + [added], records, cfg)
            tokens = len(tokenizer.encode(block, add_special_tokens=True).ids)
            if tokens <= run['review_budget']:
                current = ids
                moves.append({'removed_id': removed, 'added_id': added,
                              'removed_body_tokens': int(lengths[removed]), 'added_body_tokens': int(lengths[added]),
                              'star': cells[added][0], 'temporal_bin': cells[added][1], 'tokens_after': tokens})
                break
    block, current = make_block(current, records, cfg)
    before_hist = Counter(cells[rid] for rid in old); after_hist = Counter(cells[rid] for rid in current)
    assert before_hist == after_hist and len(current) == len(old)
    return {'business_id': run['business_id'], 'review_budget': run['review_budget'],
            'status': 'counterfactual_diagnostic_only_not_selected_evidence',
            'original_selected_ids': old, 'counterfactual_ids': current, 'moves': moves,
            'original_block_tokens': run['tokens_after_budget_enforcement'],
            'counterfactual_block_tokens': len(tokenizer.encode(block, add_special_tokens=True).ids),
            'original_body_sum': sum(int(lengths[rid]) for rid in old),
            'counterfactual_body_sum': sum(int(lengths[rid]) for rid in current),
            'retained_count': len(current), 'same_star_time_joint_histogram': True,
            'rating_jsd': run['rating_jsd_after'], 'temporal_jsd': run['temporal_jsd_after'],
            'input_text': block}


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'completion_report.json').exists():
        raise FileExistsError('Completed audit exists: use a new version rather than overwrite accepted evidence.')
    completed = json.loads((EXP / 'completion_report.json').read_text())
    assert json.loads((EXP / 'validation_report.json').read_text())['validation_passed']
    c1 = json.loads((C1 / 'completion_report.json').read_text())
    hashes = {}
    paths = {
        'selection_runs': EXP / 'selection_runs.jsonl', 'selection_diagnostics': EXP / 'selection_diagnostics.csv',
        'businesses': EXP / 'businesses.csv', 'C1_metadata': C1 / 'review_language_token_metadata.csv',
        'expanded_identity': EXP / 'scientific_identity.json',
        'expanded_config': ROOT / 'configs/review_selection_expanded.yaml',
        'tokenizer': ROOT / 'assets/tokenizers/led-base-16384/tokenizer.json',
        'train': SOURCE / 'reviews/train.jsonl', 'validation': SOURCE / 'reviews/validation.jsonl',
        'split': SOURCE / 'business_split.csv', 'pending': ROOT / 'configs/research_pending.yaml',
    }
    identity = json.loads(paths['expanded_identity'].read_text())['identity']
    expected = {'selection_runs': completed['artifact_sha256']['selection_runs.jsonl'],
                'selection_diagnostics': completed['artifact_sha256']['selection_diagnostics.csv'],
                'businesses': completed['artifact_sha256']['businesses.csv'],
                'C1_metadata': c1['diagnostic_artifact_sha256']['review_language_token_metadata.csv'],
                'expanded_config': identity['config_sha256'],
                'tokenizer': identity['tokenizer_manifest']['sha256']['tokenizer.json'],
                'train': identity['sources']['reviews/train.jsonl'],
                'validation': identity['sources']['reviews/validation.jsonl'],
                'split': identity['sources']['business_split.csv']}
    for label, path in paths.items():
        hashes[label] = digest(path)
        if label in expected: assert hashes[label] == expected[label], label
    for name, value in identity['scripts'].items(): assert digest(ROOT / 'scripts' / name) == value, name
    before = {key: (path.stat().st_size, path.stat().st_mtime_ns) for key, path in paths.items()}
    print('Accepted input hashes verified.', flush=True)
    meta = pd.read_csv(paths['C1_metadata'], keep_default_na=False)
    meta = meta.loc[meta.language.eq('en') & meta.split.isin(['train', 'validation'])].copy()
    assert len(meta) == 575749 and not meta.review_id.duplicated().any()
    meta['stars'] = meta.stars.astype(int)
    indexed = meta.set_index('review_id', drop=False)
    indexed.index.name = None
    groups = {bid: g.sort_values(['date', 'review_id']) for bid, g in meta.groupby('business_id', sort=True)}
    businesses = pd.read_csv(paths['businesses'], keep_default_na=False).set_index('business_id')
    assert set(groups) == set(businesses.index) and len(groups) == 2200
    rows = []; runs = {}; selected_ids = set()
    for line in paths['selection_runs'].open('rb'):
        r = orjson.loads(line); bid = r['business_id']; ids = r['selected_review_ids']; g = groups[bid]
        assert r['method'] in METHODS and r['review_budget'] in BUDGETS
        key = (bid, r['method'], r['review_budget']); assert key not in runs
        runs[key] = {k: r[k] for k in ['business_id', 'method', 'review_budget', 'selected_review_ids',
                                      'tokens_after_budget_enforcement', 'rating_jsd_after', 'temporal_jsd_after']}
        assert len(ids) == len(set(ids)) == r['final_retained_review_count'] and len(ids) > 0
        s = indexed.loc[ids]; assert s.business_id.eq(bid).all() and s.split.eq(r['split']).all()
        assert ids == s.sort_values(['date', 'review_id']).review_id.tolist()
        selected_ids.update(ids)
        a = g.led_tokens_without_special.to_numpy(); b = s.led_tokens_without_special.to_numpy()
        sorted_a = np.sort(a)
        percentiles = (np.searchsorted(sorted_a, b, side='left') + np.searchsorted(sorted_a, b, side='right')) / (2*len(a))
        row = {'business_id': bid, 'name': r['name'], 'split': r['split'], 'volume_band': r['volume_band'],
               'method': r['method'], 'review_budget': r['review_budget'], 'source_count': len(a), 'selected_count': len(b),
               'selected_block_tokens': r['tokens_after_budget_enforcement'],
               'source_mean': float(a.mean()), 'source_median': float(np.median(a)),
               'selected_mean': float(b.mean()), 'selected_median': float(np.median(b)),
               'selected_body_sum': int(b.sum()), 'selected_min': int(b.min()), 'selected_max': int(b.max()),
               'mean_ratio': float(b.mean()/a.mean()), 'selected_length_percentile': float(percentiles.mean()),
               'length_jsd': jsd(length_hist(a), length_hist(b)),
               'rating_jsd': r['rating_jsd_after'], 'temporal_jsd': r['temporal_jsd_after']}
        for cut in [20, 30, 50]:
            row[f'source_lt{cut}_fraction'] = float((a < cut).mean())
            row[f'selected_lt{cut}_fraction'] = float((b < cut).mean())
        rows.append(row)
    assert len(rows) == 19800
    df = pd.DataFrame(rows).sort_values(['business_id', 'method', 'review_budget'])
    example_ids, labels = choose_examples(df)
    populations = {bid: [] for bid in example_ids}; counts = Counter(); encoded = 0; seen = set()
    tokenizer = Tokenizer.from_file(str(paths['tokenizer'])); tokenizer.no_padding(); tokenizer.no_truncation()
    lookup = indexed[['business_id', 'split', 'stars', 'date', 'text_sha256', 'led_tokens_without_special']].to_dict('index')
    print(f'Computed {len(rows)} length profiles; verifying originals and freshly encoding {len(selected_ids)} unique selected reviews.', flush=True)
    for split in ['train', 'validation']:
        for line in paths[split].open('rb'):
            r = orjson.loads(line); rid = r['review_id']; m = lookup[rid]
            assert rid not in seen; seen.add(rid)
            assert (r['business_id'], split, int(r['stars']), r['date']) == (m['business_id'], m['split'], m['stars'], m['date'])
            assert hashlib.sha256(r['text'].encode()).hexdigest() == m['text_sha256']
            counts[r['business_id']] += 1
            if rid in selected_ids:
                assert len(tokenizer.encode(r['text'], add_special_tokens=False).ids) == m['led_tokens_without_special'], rid
                encoded += 1
            if r['business_id'] in populations: populations[r['business_id']].append(r)
        print(f'{split} originals verified.', flush=True)
    assert seen == set(indexed.index) and encoded == len(selected_ids)
    assert all(counts[bid] == len(g) for bid, g in groups.items())
    df.to_csv(OUT / 'business_length_comparisons.csv', index=False)
    summary(df, ['method', 'review_budget']).to_csv(OUT / 'length_summary.csv', index=False)
    summary(df, ['method', 'review_budget', 'split']).to_csv(OUT / 'length_summary_by_split.csv', index=False)
    summary(df, ['method', 'review_budget', 'volume_band']).to_csv(OUT / 'length_summary_by_volume.csv', index=False)
    pairs = []
    for budget in BUDGETS:
        a = df.loc[(df.method == 'budget_aware') & (df.review_budget == budget)].set_index('business_id')
        for baseline in METHODS[1:]:
            b = df.loc[(df.method == baseline) & (df.review_budget == budget)].set_index('business_id').reindex(a.index)
            for same_count in [False, True]:
                mask = a.selected_count.eq(b.selected_count) if same_count else np.ones(len(a), dtype=bool)
                aa = a.loc[mask]; bb = b.loc[mask]
                pairs.append({'review_budget': budget, 'baseline': baseline, 'comparison': 'same_count_only' if same_count else 'all_businesses',
                              'businesses': len(aa), 'new_shorter_mean': int((aa.selected_mean < bb.selected_mean - 1e-12).sum()),
                              'mean_body_length_difference': float((aa.selected_mean-bb.selected_mean).mean()),
                              'mean_ratio_difference': float((aa.mean_ratio-bb.mean_ratio).mean()),
                              'mean_short20_fraction_difference': float((aa.selected_lt20_fraction-bb.selected_lt20_fraction).mean())})
    pd.DataFrame(pairs).to_csv(OUT / 'paired_baseline_comparisons.csv', index=False)
    low = df.loc[df.method.eq('budget_aware') & df.selected_count.eq(30)].sort_values(['review_budget', 'selected_block_tokens', 'business_id'])
    low.to_csv(OUT / 'exactly30_ranked_by_tokens.csv', index=False)
    cfg = yaml.safe_load(paths['expanded_config'].read_text())
    examples = []; alternatives = []
    for bid in example_ids:
        pop = populations[bid]; records = {r['review_id']: r for r in pop}
        case = {'business_id': bid, 'name': businesses.loc[bid, 'name'], 'reason': labels[bid], 'runs': [], 'source_count': len(pop)}
        for budget in BUDGETS:
            r = runs[(bid, 'budget_aware', budget)]
            alternative = counterfactual(r, pop, groups[bid], tokenizer, cfg)
            alternatives.append(alternative)
            case['runs'].append({'review_budget': budget, 'selected_block_tokens': r['tokens_after_budget_enforcement'],
                                'selected': [{**records[rid], 'body_tokens': int(lookup[rid]['led_tokens_without_special'])} for rid in r['selected_review_ids']],
                                'alternative_moves': [{**move, 'removed_text': records[move['removed_id']]['text'],
                                                       'added_text': records[move['added_id']]['text']} for move in alternative['moves']]})
        examples.append(case)
    write_json(OUT / 'inspection_examples.json', examples)
    write_json(OUT / 'counterfactual_alternatives.json', alternatives)
    cf = pd.DataFrame([{k: v for k, v in a.items() if k not in ['original_selected_ids', 'counterfactual_ids', 'moves', 'input_text']}
                       | {'replacements': len(a['moves'])} for a in alternatives])
    cf.to_csv(OUT / 'counterfactual_summary.csv', index=False)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Selected review length inspection</title>',
            '<style>body{font:16px system-ui;max-width:1050px;margin:24px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ccc;padding:8px;margin:8px 0}.added{background:#eef8ee}</style>',
            '<h1>Selected originals and counterfactual longer alternatives</h1><p>Exploratory content inspection. Short does not automatically mean bad. Alternatives are diagnostic only and do not replace accepted selections.</p>']
    for case in examples:
        page.append(f'<h2>{html.escape(case["name"])} | {case["business_id"]}</h2><p>{case["reason"]}; {case["source_count"]} source reviews.</p>')
        for run in case['runs']:
            page.append(f'<details><summary>Budget {run["review_budget"]}: {len(run["selected"])} originals, {run["selected_block_tokens"]} block tokens</summary>')
            for r in run['selected']:
                page.append(f'<details><summary>{r["review_id"]} | {int(r["stars"])} stars | {r["date"]} | {r["body_tokens"]} body tokens</summary><pre>{html.escape(r["text"])}</pre></details>')
            page.append('<h3>Illustrative replacements (not selected evidence)</h3>')
            for move in run['alternative_moves']:
                page.append(f'<details class="added"><summary>{move["removed_id"]} ({move["removed_body_tokens"]}) → {move["added_id"]} ({move["added_body_tokens"]}) | same star/time bin</summary><p>Original selected:</p><pre>{html.escape(move["removed_text"])}</pre><p>Counterfactual original:</p><pre>{html.escape(move["added_text"])}</pre></details>')
            page.append('</details>')
    page.append('</html>')
    html_text = '\n'.join(page)
    for case in examples:
        for run in case['runs']:
            for r in run['selected']: assert html.escape(r['text']) in html_text
    (OUT / 'inspection_examples.html').write_text(html_text, encoding='utf-8')
    assert before == {key: (path.stat().st_size, path.stat().st_mtime_ns) for key, path in paths.items()}
    artifacts = {p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != 'completion_report.json'}
    write_json(OUT / 'completion_report.json', {'status': 'computed_source_and_selected_lengths_verified',
               'businesses': 2200, 'source_reviews_verified': len(meta), 'selection_runs': len(rows),
               'unique_selected_originals_fresh_encoded': encoded, 'test_businesses': 0,
               'input_sha256': hashes, 'input_paths': {k: str(p.relative_to(ROOT)) for k, p in paths.items()},
               'artifact_sha256': artifacts, 'example_business_ids': example_ids,
               'source_metadata_unchanged': True, 'new_length_filter': None, 'semantic_model': None,
               'final_manifest': False, 'final_review_budget': None,
               'script_sha256': digest(Path(__file__)), 'elapsed_seconds': round(time.perf_counter()-started, 3)})
    print('Length audit computed; independent aggregate/counterfactual validation and content inspection pending.', flush=True)


if __name__ == '__main__': main()
