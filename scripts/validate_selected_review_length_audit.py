"""Independent arithmetic, original-text and counterfactual audit checks."""
from pathlib import Path
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from statistics import mean, median
import csv, hashlib, html, json, math, time
import orjson, pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_c/review_selection_length_audit_v1'
EXP = ROOT / 'outputs/stage_c/review_selection_train_validation_v3'


def sha(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def divergence(a, b):
    a = [x / sum(a) for x in a]; b = [x / sum(b) for x in b]
    value = 0
    for x, y in zip(a, b):
        m = (x + y) / 2
        if x: value += .5 * x * math.log2(x / m)
        if y: value += .5 * y * math.log2(y / m)
    return value


def hist(values):
    counts = [0]*5
    for v in values: counts[sum(v >= threshold for threshold in (20, 50, 100, 200))] += 1
    return counts


def check(actual, expected, label):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10), (label, actual, expected)


def main():
    started = time.perf_counter()
    report = json.loads((OUT / 'completion_report.json').read_text())
    for label, value in report['input_sha256'].items():
        assert sha(ROOT / report['input_paths'][label]) == value, label
    assert sha(ROOT / 'scripts/audit_selected_review_lengths.py') == report['script_sha256']
    for name, value in report['artifact_sha256'].items(): assert sha(OUT / name) == value, name
    groups = defaultdict(list); reviews = {}
    with (ROOT / report['input_paths']['C1_metadata']).open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r['language'] != 'en' or r['split'] not in ('train', 'validation'): continue
            rid = r['review_id']; assert rid not in reviews
            r['length'] = int(r['led_tokens_without_special']); r['star'] = int(r['stars'])
            reviews[rid] = r; groups[r['business_id']].append(r)
    assert len(reviews) == 575749 and len(groups) == 2200
    sorted_lengths = {bid: sorted(r['length'] for r in pop) for bid, pop in groups.items()}
    with (OUT / 'business_length_comparisons.csv').open(encoding='utf-8', newline='') as f:
        output = {(r['business_id'], r['method'], int(r['review_budget'])): r for r in csv.DictReader(f)}
    selected_union = set(); runs = {}; count = 0
    for line in (EXP / 'selection_runs.jsonl').open('rb'):
        r = orjson.loads(line); bid = r['business_id']; ids = r['selected_review_ids']
        key = (bid, r['method'], r['review_budget']); runs[key] = r
        a = sorted_lengths[bid]; b = [reviews[rid]['length'] for rid in ids]; row = output[key]
        assert len(ids) == len(set(ids)) == int(row['selected_count'])
        assert all(reviews[rid]['business_id'] == bid and reviews[rid]['split'] == row['split'] for rid in ids)
        assert ids == sorted(ids, key=lambda rid: (reviews[rid]['date'], rid))
        expected = {'source_count': len(a), 'selected_count': len(b), 'selected_block_tokens': r['tokens_after_budget_enforcement'],
                    'source_mean': mean(a), 'source_median': median(a), 'selected_mean': mean(b), 'selected_median': median(b),
                    'selected_body_sum': sum(b), 'selected_min': min(b), 'selected_max': max(b),
                    'mean_ratio': mean(b)/mean(a),
                    'selected_length_percentile': mean((bisect_left(a, v)+bisect_right(a, v))/(2*len(a)) for v in b),
                    'length_jsd': divergence(hist(a), hist(b)),
                    'rating_jsd': r['rating_jsd_after'], 'temporal_jsd': r['temporal_jsd_after']}
        for cut in (20, 30, 50):
            expected[f'source_lt{cut}_fraction'] = sum(v < cut for v in a)/len(a)
            expected[f'selected_lt{cut}_fraction'] = sum(v < cut for v in b)/len(b)
        for column, value in expected.items(): check(row[column], value, (key, column))
        selected_union.update(ids); count += 1
    assert count == len(output) == 19800 and len(selected_union) == report['unique_selected_originals_fresh_encoded']
    print('All 19800 per-business/run arithmetic and IDs verified.', flush=True)
    frame = pd.read_csv(OUT / 'business_length_comparisons.csv')
    for file, columns in [('length_summary.csv', ['method', 'review_budget']),
                          ('length_summary_by_split.csv', ['method', 'review_budget', 'split']),
                          ('length_summary_by_volume.csv', ['method', 'review_budget', 'volume_band'])]:
        summary = pd.read_csv(OUT / file)
        for row in summary.to_dict('records'):
            g = frame
            for col in columns: g = g.loc[g[col].eq(row[col])]
            expected = {'businesses': len(g), 'selected_review_occurrences': g.selected_count.sum(),
                        'source_mean_macro': g.source_mean.mean(), 'selected_mean_macro': g.selected_mean.mean(),
                        'selected_body_mean_pooled': g.selected_body_sum.sum()/g.selected_count.sum(),
                        'selected_source_mean_ratio_median': g.mean_ratio.median(),
                        'selected_source_mean_ratio_mean': g.mean_ratio.mean(),
                        'mean_ratio_below_0_5': (g.mean_ratio < .5).sum(), 'mean_ratio_below_0_75': (g.mean_ratio < .75).sum(),
                        'mean_ratio_below_1': (g.mean_ratio < 1).sum(),
                        'selected_midrank_percentile_macro': g.selected_length_percentile.mean(),
                        'length_jsd_macro': g.length_jsd.mean(), 'exactly30': g.selected_count.eq(30).sum(),
                        'exactly30_under1500': (g.selected_count.eq(30) & g.selected_block_tokens.lt(1500)).sum(),
                        'exactly30_under2000': (g.selected_count.eq(30) & g.selected_block_tokens.lt(2000)).sum()}
            for cut in (20, 30, 50):
                expected[f'source_short{cut}_macro'] = g[f'source_lt{cut}_fraction'].mean()
                expected[f'selected_short{cut}_macro'] = g[f'selected_lt{cut}_fraction'].mean()
            for col, value in expected.items(): check(row[col], value, (file, col))
    paired = pd.read_csv(OUT / 'paired_baseline_comparisons.csv')
    for row in paired.to_dict('records'):
        a = frame.loc[frame.method.eq('budget_aware') & frame.review_budget.eq(row['review_budget'])].set_index('business_id')
        b = frame.loc[frame.method.eq(row['baseline']) & frame.review_budget.eq(row['review_budget'])].set_index('business_id').reindex(a.index)
        if row['comparison'] == 'same_count_only':
            mask = a.selected_count.eq(b.selected_count); a = a.loc[mask]; b = b.loc[mask]
        for col, value in {'businesses': len(a), 'new_shorter_mean': (a.selected_mean < b.selected_mean-1e-12).sum(),
                           'mean_body_length_difference': (a.selected_mean-b.selected_mean).mean(),
                           'mean_ratio_difference': (a.mean_ratio-b.mean_ratio).mean(),
                           'mean_short20_fraction_difference': (a.selected_lt20_fraction-b.selected_lt20_fraction).mean()}.items():
            check(row[col], value, ('paired', col))
    ranked = pd.read_csv(OUT / 'exactly30_ranked_by_tokens.csv')
    expected_rank = frame.loc[frame.method.eq('budget_aware') & frame.selected_count.eq(30)].sort_values(['review_budget','selected_block_tokens','business_id'])
    assert ranked.business_id.tolist() == expected_rank.business_id.tolist()
    examples = json.loads((OUT / 'inspection_examples.json').read_text(encoding='utf-8'))
    cf = json.loads((OUT / 'counterfactual_alternatives.json').read_text(encoding='utf-8'))
    assert len(cf) == 30 and len(examples) == 10 and len({x['business_id'] for x in examples}) == 10
    exids = set(report['example_business_ids']); records = {}
    for split in ('train', 'validation'):
        for line in (ROOT / report['input_paths'][split]).open('rb'):
            r = orjson.loads(line)
            if r['business_id'] in exids: records[r['review_id']] = r
    tokenizer = Tokenizer.from_file(str(ROOT / report['input_paths']['tokenizer']))
    tokenizer.no_truncation(); tokenizer.no_padding()
    full_html = (OUT / 'inspection_examples.html').read_text(encoding='utf-8')
    for case in examples:
        for run in case['runs']:
            accepted = runs[(case['business_id'], 'budget_aware', run['review_budget'])]
            assert [r['review_id'] for r in run['selected']] == accepted['selected_review_ids']
            for r in run['selected']:
                original = records[r['review_id']]
                assert all(r[k] == v for k, v in original.items())
                assert r['body_tokens'] == len(tokenizer.encode(original['text'], add_special_tokens=False).ids)
                assert html.escape(original['text']) in full_html
    annotations = json.loads((OUT / 'qualitative_inspection.json').read_text(encoding='utf-8'))
    assert [c['business_id'] for c in annotations['cases']] == [c['business_id'] for c in examples[:5]]
    annotation_texts = 0
    for note, case in zip(annotations['cases'], examples[:5]):
        selected = {r['review_id'] for r in case['runs'][0]['selected']}
        for field in ['generic_or_low_detail_example_ids', 'useful_short_example_ids']:
            assert set(note[field]) <= selected, (note['name'], field)
        assert note['longer_originals_read'] == [m['added_id'] for m in case['runs'][0]['alternative_moves'][:2]]
        annotation_texts += len(selected)
    assert annotation_texts == 150
    cf_table = pd.read_csv(OUT / 'counterfactual_summary.csv').set_index(['business_id', 'review_budget'])
    for alternative in cf:
        bid = alternative['business_id']; budget = alternative['review_budget']; ids = alternative['counterfactual_ids']
        r = runs[(bid, 'budget_aware', budget)]; old = r['selected_review_ids']
        assert old == alternative['original_selected_ids']
        ordered = sorted(groups[bid], key=lambda r: (r['date'], r['review_id'])); n = len(ordered)
        bins = {r['review_id']: i for i in range(5) for r in ordered[i*n//5:(i+1)*n//5]}
        def joint(seq): return Counter((reviews[rid]['star'], bins[rid]) for rid in seq)
        assert len(ids) == len(set(ids)) == len(old) and joint(old) == joint(ids)
        assert ids == sorted(ids, key=lambda rid: (records[rid]['date'], rid))
        current = old.copy()
        for move in alternative['moves']:
            assert move['removed_id'] in current and move['added_id'] not in current
            current.remove(move['removed_id']); current.append(move['added_id'])
            assert joint(current) == joint(old)
            assert reviews[move['added_id']]['length'] >= 2 * reviews[move['removed_id']]['length']
            assert move['removed_body_tokens'] == reviews[move['removed_id']]['length']
            assert move['added_body_tokens'] == reviews[move['added_id']]['length']
            intermediate = sorted(current, key=lambda rid: (records[rid]['date'], rid))
            text = '\n\n'.join(f'[REV_{i}] stars={int(records[rid]["stars"])} date={records[rid]["date"]}\n{records[rid]["text"]}' for i, rid in enumerate(intermediate, 1))
            assert len(tokenizer.encode(text, add_special_tokens=True).ids) == move['tokens_after'] <= budget
        assert set(current) == set(ids)
        text = '\n\n'.join(f'[REV_{i}] stars={int(records[rid]["stars"])} date={records[rid]["date"]}\n{records[rid]["text"]}' for i, rid in enumerate(ids, 1))
        assert text == alternative['input_text']
        assert len(tokenizer.encode(text, add_special_tokens=True).ids) == alternative['counterfactual_block_tokens'] <= budget
        assert alternative['original_body_sum'] == sum(reviews[rid]['length'] for rid in old)
        assert alternative['counterfactual_body_sum'] == sum(reviews[rid]['length'] for rid in ids)
        star_pop = Counter(x['star'] for x in ordered); time_pop = Counter(bins.values())
        star_sel = Counter(reviews[rid]['star'] for rid in ids); time_sel = Counter(bins[rid] for rid in ids)
        check(alternative['rating_jsd'], divergence([star_pop[i] for i in range(1, 6)], [star_sel[i] for i in range(1, 6)]), 'cf rating')
        check(alternative['temporal_jsd'], divergence([time_pop[i] for i in range(5)], [time_sel[i] for i in range(5)]), 'cf time')
        table = cf_table.loc[(bid, budget)]
        for col in ['original_block_tokens','counterfactual_block_tokens','original_body_sum','counterfactual_body_sum','retained_count','rating_jsd','temporal_jsd']:
            check(table[col], alternative[col], ('cf table', col))
        assert int(table.replacements) == len(alternative['moves'])
    write = {'validation_passed': True, 'businesses': 2200, 'test_businesses': 0,
             'source_metadata_records': len(reviews), 'run_arithmetic_verified': count,
             'unique_selected_originals_fresh_encoded_by_runner': report['unique_selected_originals_fresh_encoded'],
             'counterfactual_sets_verified': len(cf), 'inspection_examples': len(examples),
             'qualitative_inspection_selected_originals': annotation_texts,
             'qualitative_inspection_longer_originals': 10,
             'qualitative_annotation_sha256': sha(OUT / 'qualitative_inspection.json'),
             'html_originals_verified': True, 'input_artifact_hashes_verified': True,
             'final_manifest': False, 'quality_audit_scope': 'selected deterministic examples only, annotations separate',
             'elapsed_seconds': round(time.perf_counter()-started, 3)}
    (OUT / 'validation_report.json').write_text(json.dumps(write, indent=2)+'\n')
    print(json.dumps(write, indent=2), flush=True)


if __name__ == '__main__': main()
