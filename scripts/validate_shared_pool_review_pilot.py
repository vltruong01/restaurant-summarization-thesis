"""Independent v4 source/token/entropy checks and all-case fresh rerun."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from bisect import bisect_left, bisect_right
from statistics import mean, median
import argparse, hashlib, json, math, time
import orjson, pandas as pd, yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from run_shared_pool_review_pilot import select_sample, worker
from budget_aware_review_selection_v4_1 import discovery_config

ROOT=Path(__file__).resolve().parents[1]


def divergence(a,b):
    a=[v/sum(a) for v in a];b=[v/sum(b) for v in b]
    entropy=lambda p:-sum(v*math.log2(v) for v in p if v)
    return max(0.,entropy([(x+y)/2 for x,y in zip(a,b)])-(entropy(a)+entropy(b))/2)


def priority(ids,cfg,bid,kind):
    return int.from_bytes(hashlib.sha256(orjson.dumps([cfg['seed'],cfg['sampling_and_tie_algorithm_version'],bid,kind,list(ids)])).digest()[:16],'big')


def select_expected(candidates,cfg,bid,tolerance=None):
    if not candidates:return None
    anchor=min(candidates,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],priority(c['ids'],cfg,bid,'choice'),c['ids']))
    rt=cfg['search']['comparable_rating_jsd_tolerance'] if tolerance is None else tolerance
    tt=cfg['search']['comparable_temporal_jsd_tolerance'] if tolerance is None else tolerance
    eligible=[c for c in candidates if c['rating_jsd']<=anchor['rating_jsd']+rt+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+tt+1e-12]
    return min(eligible,key=lambda c:(-c['count'],c['weighted_cost'],priority(c['ids'],cfg,bid,'choice'),c['ids']))


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)


def fresh(args):
    result=worker(args)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    start=time.perf_counter();cfg=yaml.safe_load((ROOT/'configs/review_selection_shared_pool_pilot.yaml').read_text(encoding='utf-8'))
    out=ROOT/cfg['output_directory'];source=ROOT/cfg['english_pool_directory'];old=ROOT/cfg['accepted_expanded_directory']
    completion=json.loads((out/'completion_report.json').read_text(encoding='utf-8'));identity=json.loads((out/'scientific_identity.json').read_text())['identity']
    for name,digest in completion['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    for name,digest in identity['sources'].items():assert file_hash(source/name)==digest,name
    for name,digest in identity['scripts'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    assert file_hash(ROOT/'configs/review_selection_shared_pool_pilot.yaml')==identity['config_sha256']
    sample=pd.read_csv(out/'pilot_businesses.csv',keep_default_na=False)
    assert sample.to_dict('records')==select_sample(cfg).to_dict('records')
    assert len(sample)==80 and set(sample.split)<={'train','validation'} and sample.business_id.is_unique
    membership=sample.set_index('business_id').split.to_dict();info=sample.set_index('business_id').to_dict('index')
    populations={bid:{} for bid in membership}
    for line in (out/'pilot_population.jsonl').open('rb'):
        r=orjson.loads(line);original=r['review'];bid=original['business_id'];rid=original['review_id']
        assert membership[bid]==r['split'] and rid not in populations[bid];populations[bid][rid]=original
    source_count=0
    for split in ['train','validation']:
        for line in (source/'reviews'/f'{split}.jsonl').open('rb'):
            r=orjson.loads(line);bid=r['business_id']
            if bid in populations:
                assert membership[bid]==split and populations[bid][r['review_id']]==r;source_count+=1
    assert source_count==sum(map(len,populations.values()))==40407
    tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    assert file_hash(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json')==identity['tokenizer_sha256']
    tokenizer.no_truncation();tokenizer.no_padding()
    timebins={};targets={};lengths={};sortedlengths={}
    for bid,pop in populations.items():
        ordered=sorted(pop.values(),key=lambda r:(r['date'],r['review_id']));n=len(ordered)
        timebins[bid]={r['review_id']:i for i in range(5) for r in ordered[i*n//5:(i+1)*n//5]}
        targets[bid]=(Counter(int(r['stars']) for r in ordered),Counter(timebins[bid].values()))
        lengths[bid]={r['review_id']:len(tokenizer.encode(r['text'],add_special_tokens=False).ids) for r in ordered}
        sortedlengths[bid]=sorted(lengths[bid].values())
    encoded={};scored={}
    def exact(bid,ids):
        key=(bid,tuple(ids))
        if key not in encoded:
            assert list(ids)==sorted(ids,key=lambda rid:(populations[bid][rid]['date'],rid))
            text='\n\n'.join(f'[REV_{i}] stars={int(populations[bid][rid]["stars"])} date={populations[bid][rid]["date"]}\n{populations[bid][rid]["text"]}' for i,rid in enumerate(ids,1))
            encoded[key]=(text,len(tokenizer.encode(text,add_special_tokens=True).ids))
        return encoded[key]
    def metrics(bid,ids):
        key=(bid,tuple(ids))
        if key not in scored:
            stars,times=targets[bid];s=Counter(int(populations[bid][rid]['stars']) for rid in ids);t=Counter(timebins[bid][rid] for rid in ids)
            scored[key]=(divergence([stars[i] for i in range(1,6)],[s[i] for i in range(1,6)]),divergence([times[i] for i in range(5)],[t[i] for i in range(5)]))
        return scored[key]
    previews={}
    for line in (out/'model_input_previews.jsonl').open('rb'):
        r=orjson.loads(line);previews[(r['business_id'],r['method'],r['review_budget'])]=r
    caches={};runs=[];frontier_count=0;move_count=0;sens_count=0
    for bid,pop in populations.items():
        path=out/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json';payload=json.loads(path.read_text(encoding='utf-8'));caches[bid]=payload
        prior_path=old/'business_runs'/path.name;assert file_hash(prior_path)==identity['prior_business_cache_sha256'][bid]
        assert file_hash(ROOT/cfg['accepted_v4_directory']/'business_runs'/path.name)==identity['accepted_v4_cache_sha256'][bid]
        prior=json.loads(prior_path.read_text(encoding='utf-8'));prior_runs={(r['method'],r['review_budget']):r for r in prior['runs']}
        for r in payload['runs']:
            ids=r['selected_review_ids'];budget=r['review_budget'];key=(bid,r['method'],budget);runs.append(r)
            assert len(ids)==len(set(ids))==r['final_retained_review_count']<=30 and r['source_count']==len(pop)
            assert r['below_10_retained_flag']==(len(ids)<10) and r['below_5_retained_flag']==(len(ids)<5)
            if ids:
                text,tokens=exact(bid,ids);assert tokens==r['tokens_after_budget_enforcement']<=budget
                assert previews[key]['input_text']==text and previews[key]['selected_review_ids']==ids and not previews[key]['final_model_input']
                rating,temporal=metrics(bid,ids);close(rating,r['rating_jsd_after']);close(temporal,r['temporal_jsd_after'])
                a=sortedlengths[bid];b=[lengths[bid][rid] for rid in ids]
                for col,value in {'source_body_mean':mean(a),'source_body_median':median(a),'selected_body_sum':sum(b),
                                  'selected_body_mean':mean(b),'selected_body_median':median(b),'selected_source_mean_ratio':mean(b)/mean(a),
                                  'selected_length_percentile':mean((bisect_left(a,v)+bisect_right(a,v))/(2*len(a)) for v in b),
                                  'source_lt20_fraction':sum(v<20 for v in a)/len(a),'selected_lt20_fraction':sum(v<20 for v in b)/len(b)}.items():close(r[col],value)
            else:assert r['selection_status']=='budget_infeasible' and key not in previews
            if r['method']=='v4_length_balanced':
                ref=next(z for z in json.loads((ROOT/cfg['accepted_v4_directory']/'business_runs'/path.name).read_text(encoding='utf-8'))['runs'] if z['method']=='v4_length_balanced' and z['review_budget']==budget)
                assert r==ref
            elif r['method']!='v4_1_shared_pool':
                reference=prior_runs[('budget_aware' if r['method']=='v3_budget_aware' else r['method'],budget)]
                for col in ['selected_review_ids','tokens_after_budget_enforcement','rating_jsd_after','temporal_jsd_after','search_provenance']:assert r[col]==reference[col]
            else:
                path_moves=r['search_provenance']['accepted_local_path']
                for move_index,move in enumerate(path_moves):
                    before=move['before_ids'];after=move['after_ids'];operation=move['operation']
                    assert set(after)==(set(before)-set(operation['out']))|set(operation['in'])
                    discovery_budget=r['search_provenance']['discovery_budget']
                    assert exact(bid,before)[1]==move['tokens_before']<=discovery_budget
                    assert exact(bid,after)[1]==move['tokens_after']
                    if move_index==len(path_moves)-1:
                        assert after==ids and move['tokens_after']<=budget
                    else:assert move['tokens_after']<=discovery_budget
                    br,bt=metrics(bid,before);ar,at=metrics(bid,after)
                    close(br,move['rating_jsd_before']);close(bt,move['temporal_jsd_before']);close(ar,move['rating_jsd_after']);close(at,move['temporal_jsd_after']);move_count+=1
        for budget,frontier in payload['frontiers'].items():
            budget=int(budget)
            for c in frontier:
                assert len(c['ids'])==c['count'] and exact(bid,c['ids'])[1]==c['tokens']<=budget
                rating,temporal=metrics(bid,c['ids']);close(c['rating_jsd'],rating);close(c['temporal_jsd'],temporal)
                close(c['weighted_cost'],.5*rating+.5*temporal);frontier_count+=1
            expected=select_expected(frontier,cfg,bid)
            actual=next(r for r in payload['runs'] if r['method']=='v4_1_shared_pool' and r['review_budget']==budget)
            assert list(expected['ids'])==actual['selected_review_ids'] if expected else not actual['selected_review_ids']
            for a in frontier:
                assert not any(b is not a and b['rating_jsd']<=a['rating_jsd']+1e-12 and b['temporal_jsd']<=a['temporal_jsd']+1e-12 and b['count']>=a['count'] and
                               (b['rating_jsd']<a['rating_jsd']-1e-12 or b['temporal_jsd']<a['temporal_jsd']-1e-12 or b['count']>a['count']) for b in frontier)
        for row in payload['sensitivity']:
            expected=select_expected(payload['frontiers'][str(row['review_budget'])],cfg,bid,row['tolerance_per_metric'])
            assert row['selected_review_ids']==(list(expected['ids']) if expected else []);sens_count+=1
    for bid,payload in caches.items():
        shared={tuple(c['ids']):c for c in payload['shared_candidates']}
        assert set(shared)=={tuple(c['ids']) for c in payload['candidate_audit'] if c['origin'].get('variant')!='minimum_cost_singleton_rescue'}
        for c in shared.values():
            assert exact(bid,c['ids'])[1]==c['tokens']
            rr,tt=metrics(bid,c['ids']);close(rr,c['rating_jsd']);close(tt,c['temporal_jsd'])
        for b,f in payload['frontiers'].items():
            b=int(b);pool=[c for c in shared.values() if c['tokens']<=b]
            expected=[a for a in pool if not any(c is not a and c['rating_jsd']<=a['rating_jsd']+1e-12 and c['temporal_jsd']<=a['temporal_jsd']+1e-12 and c['count']>=a['count'] and (c['rating_jsd']<a['rating_jsd']-1e-12 or c['temporal_jsd']<a['temporal_jsd']-1e-12 or c['count']>a['count']) for c in pool)]
            assert {tuple(c['ids']) for c in expected}=={tuple(c['ids']) for c in f}
        v3={z['review_budget']:z for z in payload['runs'] if z['method']=='v3_budget_aware'}
        for b,z in v3.items():assert tuple(z['selected_review_ids']) in shared and shared[tuple(z['selected_review_ids'])]['tokens']==z['tokens_after_budget_enforcement']
    saved=[orjson.loads(line) for line in (out/'selection_runs.jsonl').open('rb')]
    assert saved==runs and len(saved)==1200
    frame=pd.read_csv(out/'selection_diagnostics.csv')
    expected_frame=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in runs])
    expected_frame=expected_frame.where(expected_frame.notna(),float('nan'))
    pd.testing.assert_frame_equal(frame,expected_frame,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-10)
    paired=pd.read_csv(out/'paired_v4_1_v3_differences.csv')
    assert len(paired)==240 and not paired.duplicated(['business_id','review_budget']).any()
    indexed=frame.set_index(['business_id','review_budget','method'])
    for row in paired.to_dict('records'):
        a=indexed.loc[(row['business_id'],row['review_budget'],'v4_1_shared_pool')];b=indexed.loc[(row['business_id'],row['review_budget'],'v3_budget_aware')]
        for delta,col in {'count_delta':'final_retained_review_count','rating_delta':'rating_jsd_after','time_delta':'temporal_jsd_after','body_mean_delta':'selected_body_mean','mean_ratio_delta':'selected_source_mean_ratio'}.items():close(row[delta],a[col]-b[col])
    summaries=pd.read_csv(out/'budget_method_summary.csv')
    for row in summaries.to_dict('records'):
        g=frame.loc[frame.method.eq(row['method'])&frame.review_budget.eq(row['review_budget'])];valid=g[g.selection_status.eq('feasible')]
        numeric={'businesses':len(g),'retained_mean':g.final_retained_review_count.mean(),'retained_min':g.final_retained_review_count.min(),
                 'retained_max':g.final_retained_review_count.max(),'below10':g.below_10_retained_flag.sum(),'below5':g.below_5_retained_flag.sum(),
                 'infeasible':len(g)-len(valid),'tokens_min':valid.tokens_after_budget_enforcement.min(),'tokens_max':valid.tokens_after_budget_enforcement.max(),
                 'rating_jsd_mean':valid.rating_jsd_after.mean(),'temporal_jsd_mean':valid.temporal_jsd_after.mean(),
                 'selected_body_mean_macro':valid.selected_body_mean.mean(),'source_body_mean_macro':valid.source_body_mean.mean(),
                 'median_mean_ratio':valid.selected_source_mean_ratio.median(),'mean_length_percentile':valid.selected_length_percentile.mean(),
                 'source_short20_macro':valid.source_lt20_fraction.mean(),'selected_short20_macro':valid.selected_lt20_fraction.mean(),
                 'exactly30_under1500':(g.final_retained_review_count.eq(30)&g.tokens_after_budget_enforcement.lt(1500)).sum(),
                 'rescue_used':g.minimum_cost_singleton_rescue_used.sum()}
        for label,lo,hi in [('25-30',25,30),('20-24',20,24),('15-19',15,19),('10-14',10,14),('5-9',5,9),('<5',0,4)]:numeric['band_'+label]=g.final_retained_review_count.between(lo,hi).sum()
        for col,value in numeric.items():close(row[col],value)
    assert pd.read_csv(out/'below_ten_cases.csv').shape[0]==int(frame.below_10_retained_flag.sum())
    print(f'Independent {len(runs)} runs/{frontier_count} frontier checks PASS. Fresh reversed-source rerun of all80 cases.',flush=True)
    arguments=[(bid,list(reversed(list(pop.values()))),cfg,info[bid],str(old/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json')) for bid,pop in populations.items()]
    fresh_count=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for payload in pool.map(fresh,arguments):
            cached=dict(caches[payload['business_id']]);cached.pop('identity_sha256')
            options=orjson.OPT_SORT_KEYS|orjson.OPT_NON_STR_KEYS
            assert orjson.dumps(payload,option=options)==orjson.dumps(cached,option=options),payload['business_id']
            fresh_count+=1
            if fresh_count%10==0:print(f'Fresh {fresh_count}/80 PASS',flush=True)
    report={'validation_passed':True,'businesses':80,'source_originals_verified':source_count,'shared_candidates_verified':sum(len(x['shared_candidates']) for x in caches.values()),'runs_verified':len(runs),
            'frontier_candidates_verified':frontier_count,'local_moves_verified':move_count,'sensitivity_choices_verified':sens_count,
            'fresh_businesses_verified':fresh_count,'fresh_v4_1_runs_verified':fresh_count*3,'full_fresh_payloads_equal':True,
            'all_body_lengths_fresh_encoded':True,'test_businesses':0,'final_manifest':False,'final_review_budget':None,
            'elapsed_seconds':round(time.perf_counter()-start,3)}
    (out/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
