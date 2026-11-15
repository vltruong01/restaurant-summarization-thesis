"""Independent source, rank/quota, entropy-JSD and exact-token pilot acceptance."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import random
import time
import orjson
import pandas as pd
import yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash

ROOT = Path(__file__).resolve().parents[1]


def read_lines(path):
    with path.open('rb') as stream:
        return [orjson.loads(line) for line in stream if line.strip()]


def entropy_divergence(a,b):
    a=[v/sum(a) for v in a]; b=[v/sum(b) for v in b]
    def h(p): return -sum(v*math.log2(v) for v in p if v)
    return max(0.0,h([(x+y)/2 for x,y in zip(a,b)])-.5*h(a)-.5*h(b))


def seed(seed_value,*parts):
    return int.from_bytes(hashlib.sha256(orjson.dumps([seed_value,*parts])).digest()[:16],'big')


def main():
    started=time.perf_counter()
    cfg=yaml.safe_load((ROOT/'configs/review_selection_pilot.yaml').read_text(encoding='utf-8'))
    out=ROOT/cfg['output_directory']; source=ROOT/cfg['english_pool_directory']
    report=json.loads((out/'completion_report.json').read_text(encoding='utf-8'))
    identity=json.loads((out/'scientific_identity.json').read_text(encoding='utf-8'))['identity']
    assert file_hash(ROOT/'configs/review_selection_pilot.yaml')==identity['config_sha256']
    for name,digest in report['artifact_sha256'].items(): assert file_hash(out/name)==digest,name
    for name,digest in identity['source_sha256'].items(): assert file_hash(source/name)==digest,name
    sample=pd.read_csv(out/'pilot_businesses.csv',keep_default_na=False)
    assert len(sample)==48 and sample['business_id'].is_unique and set(sample['split'])<={'train','validation'}
    for band in ('10-30','31-100','101-300','301+'):
        group=sample.loc[sample['volume_band'].eq(band)]
        assert len(group)==12 and group['split'].eq('validation').sum()>=2
    membership=sample.set_index('business_id')['split'].to_dict()
    populations=defaultdict(dict)
    for row in read_lines(out/'pilot_population.jsonl'):
        r=row['review']; bid=r['business_id']; rid=r['review_id']
        assert bid in membership and row['split']==membership[bid] and rid not in populations[bid]
        populations[bid][rid]=r
    encountered=set()
    for split in ('train','validation'):
        with (source/'reviews'/f'{split}.jsonl').open('rb') as stream:
            for line in stream:
                r=orjson.loads(line); bid=r['business_id']; rid=r['review_id']
                if bid in membership:
                    assert membership[bid]==split and rid not in encountered and populations[bid][rid]==r
                    encountered.add(rid)
    assert len(encountered)==report['population_reviews']==sum(len(v) for v in populations.values())
    for row in sample.to_dict('records'): assert len(populations[row['business_id']])==row['english_reviews']
    tokenizer_path=ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'
    assert file_hash(tokenizer_path)==identity['tokenizer_assets_sha256']['tokenizer.json']
    tokenizer=Tokenizer.from_file(str(tokenizer_path)); tokenizer.no_truncation(); tokenizer.no_padding()
    assert cfg['format']['review_template']=='[REV_{index}] stars={stars} date={date}\n{text}'
    assert cfg['format']['separator']=='\n\n'
    runs=read_lines(out/'selection_runs.jsonl')
    previews={(r['business_id'],r['method'],r['scenario']):r for r in read_lines(out/'model_input_previews.jsonl')}
    traces={(r['business_id'],r['method'],r['scenario']):r for r in read_lines(out/'removal_traces.jsonl')}
    seen_runs=set(); budget_removals_checked=0
    for run in runs:
        bid=run['business_id']; method=run['method']; scenario=run['scenario']; key=(bid,method,scenario)
        assert key not in seen_runs; seen_runs.add(key)
        full=sorted(populations[bid].values(),key=lambda r:(r['date'],r['review_id']))
        selected=[populations[bid][rid] for rid in run['selected_review_ids']]
        assert len(selected)==len(set(run['selected_review_ids']))==run['selected_reviews']
        assert selected==sorted(selected,key=lambda r:(r['date'],r['review_id']))
        m=min(len(full),30)
        initial=[populations[bid][rid] for rid in run['initial_selected_review_ids']]
        assert len(initial)==len({r['review_id'] for r in initial})==m
        counts=Counter(int(r['stars']) for r in full)
        if m==len(full): expected=full
        elif method=='random':
            expected=sorted(random.Random(seed(cfg['seed'],cfg['algorithm_version'],bid,method)).sample(full,m),key=lambda r:(r['date'],r['review_id']))
        else:
            quot={s:m*counts[s]//len(full) for s in range(1,6)}
            ranked=sorted(range(1,6),key=lambda s:(-(m*counts[s]%len(full)),s))
            for s in ranked[:m-sum(quot.values())]: quot[s]+=1
            expected=[]
            for s in range(1,6):
                group=[r for r in full if int(r['stars'])==s]; q=quot[s]
                for i in range(q):
                    segment=group[i*len(group)//q:(i+1)*len(group)//q]
                    rng=random.Random(seed(cfg['seed'],cfg['algorithm_version'],bid,method,s,i))
                    expected.append(segment[rng.randrange(len(segment))])
            expected.sort(key=lambda r:(r['date'],r['review_id']))
        assert initial==expected
        time_members={}
        full_time=[]
        for i in range(5):
            part=full[i*len(full)//5:(i+1)*len(full)//5]; full_time.append(len(part))
            time_members.update({r['review_id']:i for r in part})
        def costs(items):
            sc=Counter(int(r['stars']) for r in items)
            tc=Counter(time_members[r['review_id']] for r in items)
            rating=entropy_divergence([counts[s] for s in range(1,6)],[sc[s] for s in range(1,6)])
            temporal=entropy_divergence(full_time,[tc[i] for i in range(5)])
            return rating,temporal
        def encoded(items):
            text='\n\n'.join(f'[REV_{i}] stars={int(r["stars"])} date={r["date"]}\n{r["text"]}' for i,r in enumerate(items,1))
            return text,len(tokenizer.encode(text,add_special_tokens=True).ids)
        rating,temporal=costs(selected)
        assert abs(rating-run['rating_jsd'])<1e-12 and abs(temporal-run['temporal_jsd'])<1e-12
        text,tokens=encoded(selected)
        assert text==previews[key]['input_text'] and tokens==previews[key]['led_tokens_with_special']==run['review_tokens_after_budget_adjustment']
        assert encoded(initial)[1]==run['review_tokens_before_budget_adjustment']
        assert run['semantic_centroid_similarity'] is None and run['semantic_nearest_selected_mean'] is None and not run['final_selection']
        current=initial.copy()
        for step in traces[key]['steps']:
            candidate_ids={r['review_id'] for r in current}
            assert candidate_ids=={c['review_id'] for c in step['candidate_removals']}
            before_tokens=encoded(current)[1]
            evaluated=[]
            for candidate in step['candidate_removals']:
                remain=[r for r in current if r['review_id']!=candidate['review_id']]
                rating,temporal=costs(remain)
                cost=cfg['budget']['removal_rating_weight']*rating+cfg['budget']['removal_temporal_weight']*temporal
                assert abs(cost-candidate['cost_after'])<1e-12
                after_tokens=encoded(remain)[1]
                assert after_tokens==candidate['tokens_after'] and before_tokens-after_tokens==candidate['exact_tokens_saved']
                evaluated.append(candidate)
            best=min(v['cost_after'] for v in evaluated)
            eligible=[v for v in evaluated if v['cost_after']<=best+cfg['budget']['numerical_tie_tolerance']]
            choice=min(eligible,key=lambda v:(-v['exact_tokens_saved'],v['review_id']))
            assert choice==step['chosen_removal']
            current=[r for r in current if r['review_id']!=choice['review_id']]
            budget_removals_checked+=1
        assert current==selected
        if run['review_token_budget'] is not None:
            assert run['budget_feasible']==(tokens<=run['review_token_budget'])
        else:
            assert scenario=='unadjusted' and selected==initial and run['budget_feasible'] is None
    assert len(runs)==len(previews)==len(traces)==len(sample)*2*(1+len(cfg['budget']['exploratory_review_token_scenarios']))
    csv=pd.read_csv(out/'selection_diagnostics.csv')
    assert len(csv)==len(runs)
    for row,run in zip(csv.to_dict('records'),runs):
        assert row['business_id']==run['business_id'] and row['method']==run['method'] and row['scenario']==run['scenario']
        for name in ('rating_jsd','temporal_jsd','review_tokens_after_budget_adjustment','selected_reviews'):
            assert abs(row[name]-run[name])<1e-12
    examples=json.loads((out/'detail_example_business_ids.json').read_text(encoding='utf-8'))
    assert len(examples)==len(set(examples))==8 and set(examples)<=set(membership)
    result={'validation_passed':True,'businesses':len(sample),'population_reviews_verified':len(encountered),
            'selection_runs_verified':len(runs),'deterministic_ids_and_order_independently_reconstructed':True,
            'all_initial_quotas_and_segments_verified':True,'independent_entropy_jsd_and_exact_tokens_verified':True,
            'original_source_fields_text_verified':True,'budget_removal_steps_verified':budget_removals_checked,
            'test_businesses':0,'semantic_status':'not_computed_embedding_model_TBD',
            'final_review_token_budget':None,'methodology_locked':False,'elapsed_seconds':round(time.perf_counter()-started,3)}
    (out/'validation_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
