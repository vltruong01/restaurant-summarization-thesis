"""Bounded, explicitly provisional C2 pilot; never changes source or final splits."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import hashlib
import html
import json
import math
import random
import sqlite3
import time

import orjson
import pandas as pd
import yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash

ROOT = Path(__file__).resolve().parents[1]
BANDS = ('10-30', '31-100', '101-300', '301+')
METHODS = ('random', 'rating_time')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def stable_seed(seed, *parts):
    data = orjson.dumps([seed, *parts])
    return int.from_bytes(hashlib.sha256(data).digest()[:16], 'big')


def ordered(reviews):
    return sorted(reviews, key=lambda r: (r['date'], r['review_id']))


def hamilton(counts, sample_size):
    total = sum(counts.values())
    if total <= 0 or not 0 <= sample_size <= total:
        raise ValueError('Invalid Hamilton population/sample size')
    quotas = {star: sample_size * counts.get(star, 0) // total for star in range(1, 6)}
    ranking = sorted(range(1, 6), key=lambda s: (-(sample_size * counts.get(s, 0) % total), s))
    remaining = sample_size - sum(quotas.values())
    for star in ranking:
        if remaining and quotas[star] < counts.get(star, 0):
            quotas[star] += 1
            remaining -= 1
    assert not remaining and sum(quotas.values()) == sample_size
    return quotas


def rank_segments(reviews, count):
    reviews = ordered(reviews)
    if count == 0:
        return []
    if not 1 <= count <= len(reviews):
        raise ValueError('Invalid segment count')
    n = len(reviews)
    return [reviews[i*n//count:(i+1)*n//count] for i in range(count)]


def select_initial(population, method, cfg, bid):
    population = ordered(population)
    size = min(len(population), cfg['selection']['maximum_reviews'])
    if size == len(population):
        return population.copy()
    if method == 'random':
        rng = random.Random(stable_seed(cfg['seed'], cfg['algorithm_version'], bid, method))
        return ordered(rng.sample(population, size))
    if method != 'rating_time':
        raise ValueError(method)
    counts = Counter(int(r['stars']) for r in population)
    quotas = hamilton(counts, size)
    result = []
    for star in range(1, 6):
        group = [r for r in population if int(r['stars']) == star]
        for i, segment in enumerate(rank_segments(group, quotas[star])):
            rng = random.Random(stable_seed(cfg['seed'], cfg['algorithm_version'], bid, method, star, i))
            result.append(segment[rng.randrange(len(segment))])
    return ordered(result)


def jsd(p, q):
    """Jensen-Shannon divergence, log base 2 (not its square-root distance)."""
    if len(p) != len(q) or not p or sum(p) <= 0 or sum(q) <= 0:
        raise ValueError('Invalid probability populations')
    p = [v / sum(p) for v in p]
    q = [v / sum(q) for v in q]
    result = 0.0
    for a, b in zip(p, q):
        mean = (a + b) / 2
        if a: result += .5 * a * math.log2(a / mean)
        if b: result += .5 * b * math.log2(b / mean)
    return max(0.0, result)


def fixed_time_bins(population, count=5):
    segments = rank_segments(population, min(count, len(population)))
    return {r['review_id']: i for i, segment in enumerate(segments) for r in segment}, [len(s) for s in segments]


def formatted_input(selected, cfg):
    fmt = cfg['format']
    return fmt['separator'].join(fmt['review_template'].format(
        index=i, stars=int(r['stars']), date=r['date'], text=r['text']) for i, r in enumerate(ordered(selected), 1))


def token_count(selected, cfg, tokenizer):
    return len(tokenizer.encode(formatted_input(selected, cfg), add_special_tokens=True).ids)


def diagnostics(population, selected, cfg, bins=None):
    if not selected:
        raise ValueError('Cannot describe an empty selected set')
    population, selected = ordered(population), ordered(selected)
    bins, full_time = bins or fixed_time_bins(population, cfg['diagnostics']['temporal_rank_bins'])
    full_stars = Counter(int(r['stars']) for r in population)
    selected_stars = Counter(int(r['stars']) for r in selected)
    selected_time = Counter(bins[r['review_id']] for r in selected)
    full_p = [full_stars[s]/len(population) for s in range(1, 6)]
    selected_p = [selected_stars[s]/len(selected) for s in range(1, 6)]
    span = (datetime.fromisoformat(population[-1]['date']) - datetime.fromisoformat(population[0]['date'])).total_seconds()
    selected_span = (datetime.fromisoformat(selected[-1]['date']) - datetime.fromisoformat(selected[0]['date'])).total_seconds()
    return {
        'total_reviews': len(population), 'selected_reviews': len(selected),
        'full_rating_distribution': {str(s): full_p[s-1] for s in range(1, 6)},
        'selected_rating_distribution': {str(s): selected_p[s-1] for s in range(1, 6)},
        'rating_jsd': jsd(full_p, selected_p),
        'rating_total_variation': sum(abs(a-b) for a,b in zip(full_p, selected_p))/2,
        'full_temporal_distribution': [v/len(population) for v in full_time],
        'selected_temporal_distribution': [selected_time[i]/len(selected) for i in range(len(full_time))],
        'temporal_jsd': jsd(full_time, [selected_time[i] for i in range(len(full_time))]),
        'temporal_bins_covered': len(selected_time), 'temporal_bins_total': len(full_time),
        'earliest_full_date': population[0]['date'], 'latest_full_date': population[-1]['date'],
        'earliest_selected_date': selected[0]['date'], 'latest_selected_date': selected[-1]['date'],
        'date_span_coverage': selected_span/span if span else 1.0,
        'semantic_centroid_similarity': None, 'semantic_nearest_selected_mean': None,
        'semantic_status': 'not_computed_embedding_model_TBD',
    }


def adjust_budget(population, initial, budget, cfg, tokenizer):
    """Greedy provisional removal; callers must supply an explicit scenario budget."""
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError('Budget must be an explicitly configured positive integer')
    selected = ordered(initial)
    bins = fixed_time_bins(population, cfg['diagnostics']['temporal_rank_bins'])
    rule = cfg['budget']
    traces = []
    current_tokens = token_count(selected, cfg, tokenizer)
    while current_tokens > budget and len(selected) > rule['minimum_selected_reviews']:
        candidates = []
        before = diagnostics(population, selected, cfg, bins)
        before_cost = rule['removal_rating_weight']*before['rating_jsd'] + rule['removal_temporal_weight']*before['temporal_jsd']
        for review in selected:
            remainder = [r for r in selected if r['review_id'] != review['review_id']]
            d = diagnostics(population, remainder, cfg, bins)
            tokens = token_count(remainder, cfg, tokenizer)
            cost = rule['removal_rating_weight']*d['rating_jsd'] + rule['removal_temporal_weight']*d['temporal_jsd']
            candidates.append({'review_id': review['review_id'], 'rating_jsd_after': d['rating_jsd'],
                               'temporal_jsd_after': d['temporal_jsd'], 'cost_after': cost,
                               'cost_change': cost-before_cost, 'tokens_after': tokens,
                               'exact_tokens_saved': current_tokens-tokens})
        minimum = min(c['cost_after'] for c in candidates)
        tied = [c for c in candidates if c['cost_after'] <= minimum + rule['numerical_tie_tolerance']]
        choice = min(tied, key=lambda c: (-c['exact_tokens_saved'], c['review_id']))
        traces.append({'step': len(traces)+1, 'tokens_before': current_tokens, 'cost_before': before_cost,
                       'chosen_removal': choice, 'candidate_removals': candidates})
        selected = [r for r in selected if r['review_id'] != choice['review_id']]
        current_tokens = choice['tokens_after']
    return selected, traces, current_tokens <= budget


def choose_pilot(business, cfg):
    frame = business.loc[business['split'].isin(cfg['allowed_splits'])].copy()
    frame['volume_band'] = pd.cut(frame['english_reviews'], [9,30,100,300,float('inf')], labels=BANDS).astype(str)
    frame['positive_fraction'] = frame['english_star_5']/frame['english_reviews']
    frame['negative_fraction'] = (frame['english_star_1']+frame['english_star_2'])/frame['english_reviews']
    def entropy(row):
        p = [row[f'english_star_{s}']/row['english_reviews'] for s in range(1,6)]
        return -sum(v*math.log2(v) for v in p if v)
    frame['rating_entropy'] = frame.apply(entropy, axis=1)
    frame['history_days'] = (pd.to_datetime(frame['last_english_review_date'])-pd.to_datetime(frame['first_english_review_date'])).dt.total_seconds()/86400
    selected, reasons = [], defaultdict(list)
    for band in BANDS:
        subset = frame.loc[frame['volume_band'].eq(band)].copy()
        count = cfg['pilot']['businesses_per_volume_band']
        if len(subset) < count: raise ValueError(f'Insufficient pilot businesses: {band}')
        chosen = []
        def add(bid, reason):
            if bid not in chosen: chosen.append(bid)
            reasons[bid].append(reason)
        targets = [('strong_positive','positive_fraction',False,subset),
                   ('mixed_ratings','rating_entropy',False,subset),
                   ('rare_negative','negative_fraction',True,subset.loc[subset['negative_fraction'].gt(0)]),
                   ('long_reviews','english_review_token_mean',False,subset),
                   ('long_history','history_days',False,subset), ('short_history','history_days',True,subset)]
        for reason, metric, ascending, candidates in targets:
            if len(candidates):
                bid = candidates.sort_values([metric,'business_id'], ascending=[ascending,True]).iloc[0]['business_id']
                add(bid, reason)
        val_ids = sorted(subset.loc[subset['split'].eq('validation'),'business_id'])
        rng = random.Random(stable_seed(cfg['seed'], 'pilot_businesses', band))
        rng.shuffle(val_ids)
        membership = subset.set_index('business_id')['split'].to_dict()
        for bid in val_ids:
            if sum(membership[b]=='validation' for b in chosen) >= cfg['pilot']['minimum_validation_per_band']: break
            add(bid,'validation_coverage')
        if sum(membership[b]=='validation' for b in chosen) < cfg['pilot']['minimum_validation_per_band']:
            raise ValueError('Insufficient validation coverage')
        remaining = sorted(set(subset['business_id'])-set(chosen))
        rng.shuffle(remaining)
        for bid in remaining[:count-len(chosen)]: add(bid,'seeded_fill')
        assert len(chosen)==count
        selected.extend(chosen)
    result = frame.set_index('business_id').loc[selected].reset_index()
    result['pilot_reasons'] = result['business_id'].map(lambda bid: '|'.join(reasons[bid]))
    return result


def save_examples(out, sample, populations, runs, cfg):
    chosen = []
    for band in BANDS:
        group = sample.loc[sample['volume_band'].eq(band)]
        first = group.sort_values(['english_review_token_mean','business_id'],ascending=[False,True]).iloc[0]['business_id']
        rest = group.loc[~group['business_id'].eq(first)].sort_values(['history_days','business_id'],ascending=[False,True])
        chosen.extend([first,rest.iloc[0]['business_id']])
    parts = ['<!doctype html><html lang="vi"><meta charset="utf-8"><title>Draft review selection examples</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 18px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;margin:6px 0;padding:8px}.selected{background:#e9f5ff}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style>',
             '<h1>Pilot examples — chưa khóa methodology</h1><p>8 businesses train/validation; tất cả original English review texts trong từng business có thể mở để đối chiếu. Màu xanh là review được chọn ở phương pháp rating/time ban đầu. Semantic model và final budget có thể còn TBD.</p>']
    by_id = sample.set_index('business_id').to_dict('index')
    for bid in chosen:
        row = by_id[bid]
        business_runs = [r for r in runs if r['business_id']==bid]
        initial = next(r for r in business_runs if r['method']=='rating_time' and r['scenario']=='unadjusted')
        ids = set(initial['selected_review_ids'])
        parts.append(f'<h2>{html.escape(row["name"])} — {html.escape(bid)}</h2><p>{row["split"]}; {row["volume_band"]}; {len(populations[bid])} English reviews; reasons: {html.escape(row["pilot_reasons"])}</p>')
        cols = ['method','scenario','selected_reviews','review_tokens_after_budget_adjustment','rating_jsd','temporal_jsd','date_span_coverage']
        parts.append(pd.DataFrame(business_runs)[cols].to_html(index=False,float_format=lambda v:f'{v:.5f}',escape=True))
        for r in ordered(populations[bid]):
            mark = 'SELECTED' if r['review_id'] in ids else 'not selected'
            css = ' class="selected"' if r['review_id'] in ids else ''
            parts.append(f'<details{css}><summary>{mark} | {int(r["stars"])} stars | {html.escape(r["date"])} | {html.escape(r["review_id"])}</summary><pre>{html.escape(r["text"])}</pre></details>')
    parts.append('</html>')
    (out/'manual_examples.html').write_text('\n'.join(parts),encoding='utf-8')
    write_json(out/'detail_example_business_ids.json',chosen)


def run_pilot(root=ROOT):
    root = Path(root)
    started = time.perf_counter()
    config_path = root/'configs/review_selection_pilot.yaml'
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if cfg['allowed_splits'] != ['train','validation'] or cfg['final_methodology_locked'] or cfg['final_selections_ready']:
        raise ValueError('This runner is draft train/validation only')
    if cfg['semantic']['model_id'] is not None:
        raise ValueError('Embedding diagnostics require a separately specified implementation, not implicit model loading')
    source, c1, out = (root/cfg[k] for k in ('english_pool_directory','c1_directory','output_directory'))
    completion = json.loads((c1/'completion_report.json').read_text(encoding='utf-8'))
    acceptance = json.loads((c1/'validation_report.json').read_text(encoding='utf-8'))
    assert acceptance['validation_passed']
    for name in ('business_review_statistics.csv','scientific_identity.json'):
        assert file_hash(c1/name)==completion['diagnostic_artifact_sha256'][name], name
    pool_hashes = json.loads((source/'checksums.json').read_text(encoding='utf-8'))
    source_paths = ['reviews/train.jsonl','reviews/validation.jsonl','business_split.csv']
    before = {name:{'bytes':(source/name).stat().st_size,'mtime_ns':(source/name).stat().st_mtime_ns} for name in source_paths}
    for name in source_paths: assert file_hash(source/name)==pool_hashes[name], name
    token_manifest = json.loads((root/'assets/tokenizers/led-base-16384.manifest.json').read_text(encoding='utf-8'))
    token_dir = root/'assets/tokenizers/led-base-16384'
    for name,digest in token_manifest['sha256'].items(): assert file_hash(token_dir/name)==digest
    identity = {'config_sha256':file_hash(config_path),'c1_identity':completion['identity_sha256'],
                'source_sha256':{n:pool_hashes[n] for n in source_paths},'tokenizer_revision':token_manifest['revision'],
                'tokenizer_assets_sha256':token_manifest['sha256'],'tokenizers_version':version('tokenizers'),
                'script_sha256':file_hash(Path(__file__))}
    identity_hash = hashlib.sha256(orjson.dumps(identity,option=orjson.OPT_SORT_KEYS)).hexdigest()
    report_path = out/'completion_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['identity_sha256']!=identity_hash: raise RuntimeError('Existing pilot has another identity; version the output directory')
        for name,digest in report['artifact_sha256'].items(): assert file_hash(out/name)==digest, name
        print('Using completed draft review-selection pilot.',flush=True)
        return report
    if out.exists() and any(out.iterdir()): raise FileExistsError('Nonempty unfinished pilot output; inspect before replacing')
    tokenizer = Tokenizer.from_file(str(token_dir/'tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    sample = choose_pilot(pd.read_csv(c1/'business_review_statistics.csv',keep_default_na=False),cfg)
    membership = sample.set_index('business_id')['split'].to_dict()
    populations = {bid:[] for bid in membership}
    print(f'Loading full English populations of {len(sample)} pilot businesses; test excluded.',flush=True)
    for split in cfg['allowed_splits']:
        with (source/'reviews'/f'{split}.jsonl').open('rb') as stream:
            for line in stream:
                row = orjson.loads(line)
                if row['business_id'] in membership:
                    assert membership[row['business_id']]==split
                    populations[row['business_id']].append(row)
    db_path = (c1/'review_metadata.sqlite').resolve()
    db = sqlite3.connect(f'file:{db_path.as_posix()}?mode=ro',uri=True)
    placeholders = ','.join('?' for _ in membership)
    metadata = {r[0]:r[1:] for r in db.execute(f'SELECT review_id,business_id,split,language,text_sha256 FROM reviews WHERE business_id IN ({placeholders})',list(membership))}
    db.close()
    seen = set()
    for bid,population in populations.items():
        expected = int(sample.loc[sample['business_id'].eq(bid),'english_reviews'].iloc[0])
        assert len(population)==expected
        for r in population:
            rid = r['review_id']; assert rid not in seen; seen.add(rid)
            assert metadata[rid]==(bid,membership[bid],'en',hashlib.sha256(r['text'].encode('utf-8')).hexdigest())
    out.mkdir(parents=True,exist_ok=True)
    sample.to_csv(out/'pilot_businesses.csv',index=False)
    write_json(out/'scientific_identity.json',{'identity_sha256':identity_hash,'identity':identity})
    (out/'config_at_run.yaml').write_bytes(config_path.read_bytes())
    with (out/'pilot_population.jsonl').open('wb') as stream:
        for bid in membership:
            for row in ordered(populations[bid]): stream.write(orjson.dumps({'split':membership[bid],'review':row})+b'\n')
    runs,flat,traces,preview = [],[],[],[]
    source_rows = sample.set_index('business_id').to_dict('index')
    for bid,population in populations.items():
        for method in METHODS:
            initial = select_initial(population,method,cfg,bid)
            repeat = select_initial(list(reversed(population)),method,cfg,bid)
            assert [r['review_id'] for r in initial]==[r['review_id'] for r in repeat]
            before_tokens = token_count(initial,cfg,tokenizer)
            options = [('unadjusted',None,initial,[],None)]
            for budget in cfg['budget']['exploratory_review_token_scenarios']:
                selected,trace,feasible = adjust_budget(population,initial,budget,cfg,tokenizer)
                options.append((f'exploratory_{budget}',budget,selected,trace,feasible))
            for scenario,budget,selected,trace,feasible in options:
                metrics = diagnostics(population,selected,cfg)
                after_tokens = token_count(selected,cfg,tokenizer)
                run = {'business_id':bid,'name':source_rows[bid]['name'],'split':membership[bid],
                       'volume_band':source_rows[bid]['volume_band'],'method':method,'scenario':scenario,
                       'initial_selected_reviews':len(initial),**metrics,
                       'review_tokens_before_budget_adjustment':before_tokens,'review_tokens_after_budget_adjustment':after_tokens,
                       'review_token_budget':budget,'budget_feasible':feasible,'final_review_token_budget':None,
                       'initial_selected_review_ids':[r['review_id'] for r in initial],
                       'selected_review_ids':[r['review_id'] for r in selected],
                       'removed_for_token_budget':[t['chosen_removal']['review_id'] for t in trace],
                       'random_seed':cfg['seed'],'algorithm_version':cfg['algorithm_version'],'final_selection':False}
                runs.append(run)
                row = {k:v for k,v in run.items() if not isinstance(v,(list,dict))}
                for star in range(1,6):
                    row[f'full_star_{star}_proportion']=metrics['full_rating_distribution'][str(star)]
                    row[f'selected_star_{star}_proportion']=metrics['selected_rating_distribution'][str(star)]
                flat.append(row)
                traces.append({'business_id':bid,'method':method,'scenario':scenario,'budget':budget,'steps':trace})
                preview.append({'business_id':bid,'method':method,'scenario':scenario,'selected_review_ids':run['selected_review_ids'],
                                'input_text':formatted_input(selected,cfg),'led_tokens_with_special':after_tokens,'final_model_input':False})
    for name,values in (('selection_runs.jsonl',runs),('removal_traces.jsonl',traces),('model_input_previews.jsonl',preview)):
        with (out/name).open('wb') as stream:
            for row in values: stream.write(orjson.dumps(row)+b'\n')
    diagnostics_frame = pd.DataFrame(flat)
    diagnostics_frame.to_csv(out/'selection_diagnostics.csv',index=False)
    aggregate = diagnostics_frame.groupby(['volume_band','method','scenario'],sort=False)[
        ['rating_jsd','temporal_jsd','rating_total_variation','date_span_coverage','selected_reviews','review_tokens_after_budget_adjustment']].agg(['mean','median','max'])
    aggregate.columns = ['_'.join(c) for c in aggregate.columns]
    aggregate.to_csv(out/'diagnostics_by_band_method.csv')
    save_examples(out,sample,populations,runs,cfg)
    assert before=={name:{'bytes':(source/name).stat().st_size,'mtime_ns':(source/name).stat().st_mtime_ns} for name in source_paths}
    (out/'README.md').write_text('# C2 experimental review-selection pilot\n\n48 train/validation stress-sample businesses; no held-out test selection or final manifest. '
        'Rating/time and random methods use identical original English populations. Semantic diagnostics are null/model TBD. '
        'Configured exploratory budgets, if any, are sensitivity scenarios, not final review/caption allocations. '
        'manual_examples.html contains all original review texts for eight inspection examples. '
        'Algorithm integrity PASS is not an empirical quality threshold or final methodology approval.\n',encoding='utf-8')
    artifacts = {p.relative_to(out).as_posix():file_hash(p) for p in out.iterdir() if p.is_file()}
    report = {'stage':'C2_experimental_pilot','identity_sha256':identity_hash,'businesses':len(sample),
              'population_reviews':len(seen),'allowed_splits':cfg['allowed_splits'],'test_businesses':0,
              'selection_runs':len(runs),'baseline':'random','candidate_method':'rating_time',
              'exploratory_budget_scenarios':cfg['budget']['exploratory_review_token_scenarios'],
              'final_review_token_budget':None,'semantic_status':'not_computed_embedding_model_TBD',
              'deterministic_repeat_ids_and_order_match':True,'source_metadata_unchanged':True,
              'artifact_sha256':artifacts,'methodology_locked':False,'overall_stage_c_complete':False,
              'elapsed_seconds':round(time.perf_counter()-started,3)}
    write_json(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)
    return report


if __name__=='__main__':
    run_pilot()
