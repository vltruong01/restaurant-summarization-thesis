"""Offline integrity and frozen-result checks; uses only the Python standard library."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import statistics

BASE = Path(__file__).resolve().parent


def load(name):
    return json.loads((BASE / 'data' / name).read_text(encoding='utf-8'))


def rows(name):
    with (BASE / 'data' / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12), (a, b)


def verify():
    checked = 0
    with (BASE / 'PACKAGE_MANIFEST.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            p = (BASE / r['file']).resolve()
            assert p.is_relative_to(BASE.resolve()) and p.is_file()
            assert p.stat().st_size == int(r['bytes'])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == r['sha256'], r['file']
            checked += 1
    for r in load('package_data_manifest.json'):
        assert hashlib.sha256((BASE / r['file']).read_bytes()).hexdigest() == r['sha256']
    split = rows('business_split.csv')
    assert len(split) == len({r['business_id'] for r in split}) == 2300
    assert Counter(r['split'] for r in split) == {'train':2000, 'validation':200, 'test':100}
    budget = rows('source_budget_ledger.csv')
    assert len(budget) == 6900 and len({(r['business_id'],r['condition']) for r in budget}) == 6900
    assert all(int(r['full_source_tokens']) <= 5000 and int(r['review_block_tokens_with_special']) <= 4000 and int(r['caption_block_increment_tokens']) <= 750 for r in budget)
    metrics = ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']
    for stem, cond in [('test100','condition'), ('validation200_teacher','model')]:
        per = rows(stem+'_per_business.csv')
        agg = rows(stem+'_aggregate.csv')
        group = defaultdict(list)
        for r in per:
            group[r[cond],r['task']].append(r)
        for r in agg:
            part = group[r[cond],r['task']]
            assert len(part) == int(r['n'])
            for metric in metrics:
                close(statistics.mean(float(x[metric]) for x in part),r[metric])
    test = rows('test100_per_business.csv')
    assert len(test) == 700 and len({r['business_id'] for r in test}) == 100
    bindings = {r['business_id']:r for r in load('test100_reference_binding_check.json')}
    assert len(bindings) == 100
    for r in test:
        assert r['business_id'] == r['reference_business_id']
        assert r['reference_sha256'] == bindings[r['business_id']]['content_sha256']
    for r in load('test100_paired.json'):
        vals={(x['business_id'],x['condition']):float(x[r['metric']]) for x in test if x['task']==r['task']}
        ids=sorted({x[0] for x in vals})
        comp=r['contrast'].split('-')[1]
        close(statistics.mean(vals[b,'F']-vals[b,comp] for b in ids),r['mean_difference'])
    ratings = load('test30_ratings_240.json')
    assert len(ratings) == 240 and len({(r['business_id'],r['condition'],r['annotator']) for r in ratings}) == 240
    assert Counter((r['condition'],r['annotator']) for r in ratings) == {(c,a):30 for c in ['RO','U','F','S'] for a in ['A01','A02']}
    criteria = ['review_faithfulness','aspect_coverage','coherence','non_redundancy',
                'usefulness','visual_grounding_and_contribution']
    for r in rows('test30_descriptive.csv'):
        part = [x[r['criterion']] for x in ratings if x['condition']==r['condition']
                and (r['panel']=='COMBINED' or x['annotator']==r['panel'])]
        part = [x for x in part if x != 'N/A']
        assert len(part)==int(r['n_valid_ratings'])
        for k in range(1,6):assert part.count(k)==int(r[f'score_{k}'])
        if part:
            for name,actual in [('mean',statistics.mean(part)),('sample_sd',statistics.stdev(part)),
                                ('median',statistics.median(part))]:close(actual,r[name])
    from verify_extension import qwk
    for r in rows('test30_agreement.csv'):
        if r['condition'] not in ['RO','U','F','S'] or r['criterion'] not in criteria:
            continue
        if r['condition']=='RO' and r['criterion']==criteria[-1]:continue
        part={(x['business_id'],x['annotator']):x[r['criterion']] for x in ratings if x['condition']==r['condition']}
        ids=sorted({k[0] for k in part})
        close(qwk([part[b,'A01'] for b in ids],[part[b,'A02'] for b in ids]),r['qwk'])
    per_h = rows('test30_per_business_criterion.csv')
    assert len(per_h) == 690
    for r in per_h:
        pair = [x[r['criterion']] for x in ratings if x['business_id']==r['business_id'] and x['condition']==r['condition']]
        close(statistics.mean(pair),r['two_rater_average'])
    for r in load('test30_paired.json'):
        vals={(x['business_id'],x['condition']):float(x['two_rater_average']) for x in per_h if x['criterion']==r['criterion']}
        ids=sorted({x[0] for x in vals})
        comp=r['contrast'].split('-')[1]
        close(statistics.mean(vals[b,'F']-vals[b,comp] for b in ids),r['mean_paired_difference'])
    for example in load('illustrative_cases.json'):
        for components in example['case']['models'].values():
            for r in components.values():
                if r:
                    assert hashlib.sha256(r['raw_output_text'].encode('utf-8')).hexdigest() == r['output_sha256']
    book = json.loads((BASE/'THESIS_RESEARCH_OVERVIEW.ipynb').read_text(encoding='utf-8'))
    code = [c for c in book['cells'] if c['cell_type']=='code']
    assert len(code)==25 and all(c['execution_count'] is not None for c in code)
    assert not any(o['output_type']=='error' for c in code for o in c['outputs'])
    assert sum('image/png' in o.get('data',{}) for c in code for o in c['outputs'])>=10
    from verify_extension import verify as external_verify
    external_verify(BASE)
    print(f'PASS: {checked} package members; data hashes; split/budgets; all frozen aggregate means; reference bindings; paired point estimates; 240 unchanged numeric records; complete example hashes; executed Yelp/O/P cells with embedded figures; Stage O/P descriptive, mapping, agreement and paired checks.')
    print('Scope: no model scoring, training, new inference or participant authentication. Stage P inferential replay is separately executed in the notebook.')


if __name__ == '__main__':
    verify()
