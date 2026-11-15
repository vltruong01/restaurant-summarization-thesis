"""Read-only arithmetic checks on bundled, unchanged Stage O/P evidence (stdlib)."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import statistics

CRITERIA = ['review_faithfulness', 'aspect_coverage', 'coherence',
            'non_redundancy', 'usefulness', 'visual_grounding_and_contribution']

def close(x, y):
    if x is None or y is None:
        assert x is None and y is None, (x, y)
    else:
        assert math.isclose(x, y, rel_tol=0, abs_tol=1e-12), (x, y)

def load(base, name):
    return json.loads((Path(base) / name).read_text(encoding='utf-8'))

def qwk(x, y):
    n = len(x)
    a, b = Counter(x), Counter(y)
    observed = sum((u-v)**2 for u, v in zip(x, y))/n
    expected = sum(a[u]*b[v]*(u-v)**2 for u in range(1, 6)
                   for v in range(1, 6))/(n*n)
    return 1-observed/expected if expected else None

def records(base, stage):
    raw = load(base, f'evidence/{stage}/RATINGS_90.json')['records']
    if stage == 'O':
        return [dict(r, model=r['condition']) for r in raw]
    key = {r['alias']: r for r in load(base, 'evidence/P/ASSIGNMENT_KEY_90.json')}
    out = {(r['business_id'], r['system']): r
           for r in load(base, 'evidence/P/FROZEN_OUTPUTS_45.json')}
    assert len(key) == 90 and len(out) == 45
    result = []
    for r in raw:
        k = key[r['alias']]
        assert (r['business_id'], r['annotator'], r['condition'], r['position']) == (
            k['business_id'], k['annotator'], k['neutral_slot'], k['position'])
        f = out[r['business_id'], k['system']]
        for component in ['review', 'visual']:
            digest = hashlib.sha256(f[component+'_text'].encode()).hexdigest()
            assert digest == f[component+'_text_sha256'] == k[component+'_text_sha256']
        result.append(dict(r, model=k['system']))
    return result

def verify(base):
    base = Path(base)
    counts = {}
    all_rows = {}
    for stage, models in [('O', ['RO', 'U', 'F']),
                          ('P', ['LED F original', 'Qwen-Caption', 'Qwen-Image'])]:
        rows = records(base, stage)
        all_rows[stage] = rows
        ids = sorted({r['business_id'] for r in rows})
        cells = {(r['business_id'], r['annotator'], r['model']): r for r in rows}
        assert len(ids) == 15 and len(rows) == len(cells) == 90
        assert Counter((r['model'], r['annotator']) for r in rows) == {
            (m, a): 15 for m in models for a in ['A01', 'A02']}
        assert all(r['status'] == 'SUBMITTED' for r in rows)
        desc = load(base, f'data/final/{stage}_DESCRIPTIVES.json')
        assert len(desc) == 54
        for d in desc:
            values = [r[d['criterion']] for r in rows if r['model'] == d['model']
                      and (d['panel'] == 'COMBINED' or r['annotator'] == d['panel'])]
            values = [v for v in values if v != 'N/A']
            assert len(values) == d['n']
            for k in range(1, 6):
                assert values.count(k) == d[f'score_{k}']
            close(statistics.mean(values) if values else None, d['mean'])
            close(statistics.stdev(values) if values else None, d['sample_SD'])
            close(statistics.median(values) if values else None, d['median'])
        agreement = load(base, f'data/final/{stage}_AGREEMENT.json')
        for d in agreement:
            if d['criterion'] not in CRITERIA or d['model'] not in models:
                continue
            if d['model'] == 'RO' and d['criterion'] == CRITERIA[-1]:
                continue
            x = [cells[b, 'A01', d['model']][d['criterion']] for b in ids]
            y = [cells[b, 'A02', d['model']][d['criterion']] for b in ids]
            close(qwk(x, y), d['quadratic_weighted_kappa'])
            close(sum(a == b for a, b in zip(x, y))/15, d['exact_agreement_fraction'])
            close(statistics.mean(abs(a-b) for a, b in zip(x, y)), d['mean_absolute_difference'])
        paired = load(base, f'data/final/{stage}_PAIRED_COMPARISONS.json')
        for d in paired:
            left, right = ('F', d['contrast'].split('-')[1]) if stage == 'O' else (d['left'], d['right'])
            diffs = [statistics.mean(cells[b, a, left][d['criterion']] -
                                     cells[b, a, right][d['criterion']]
                                     for a in ['A01', 'A02']) for b in ids]
            close(statistics.mean(diffs), d['mean_difference'])
        counts[stage] = {'records': 90, 'restaurants': 15, 'descriptive_rows_checked': 54,
                         'agreement_rows_checked': len(agreement), 'paired_point_estimates_checked': len(paired)}
    assert {r['business_id'] for r in all_rows['O']} == {r['business_id'] for r in all_rows['P']}
    outputs = load(base, 'evidence/P/FROZEN_OUTPUTS_45.json')
    old_outputs = load(base, 'evidence/O/FROZEN_OUTPUTS_45.json')
    assert len(old_outputs)==45
    assert len({(r['business_id'],r['condition']) for r in old_outputs})==45
    for r in old_outputs:
        for component,text in r['component_texts'].items():
            assert hashlib.sha256(text.encode()).hexdigest()==r['component_hashes'][component]
        if r['condition']=='F':
            p=next(x for x in outputs if x['business_id']==r['business_id'] and x['system']=='LED F original')
            assert r['component_texts']['review']==p['review_text']
            assert r['component_texts']['visual']==p['visual_text']
    for b in {r['business_id'] for r in outputs}:
        r = {x['system']: x for x in outputs if x['business_id'] == b}
        assert r['Qwen-Caption']['review_text'] == r['Qwen-Image']['review_text']
    facts = load(base, 'evidence/EFFICIENCY/VERIFIED_FACTS.json')
    close(facts['blip_elapsed_seconds'] + facts['scoring_elapsed_seconds'],
          facts['historical_measured_preparation_subtotal_seconds'])
    close(facts['qcap_review_seconds'] + facts['qimage_visual_seconds'],
          facts['qimage_constructed_generation_seconds'])
    assert facts['complete_pipeline_measured'] is False
    counts['scope'] = 'Existing records, mapping/output hashes, descriptive arithmetic, kappa and paired point estimates. No model execution or participant authentication.'
    return counts

if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
