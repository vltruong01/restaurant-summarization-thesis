"""Independent source/entropy/token/aggregate validation for expanded C2."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,csv,hashlib,json,math,os,random,time
import orjson,pandas as pd,yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from run_budget_review_pilot import run_business

ROOT=Path(__file__).resolve().parents[1]


def divergence(a,b):
    a=[v/sum(a) for v in a];b=[v/sum(b) for v in b]
    entropy=lambda p:-sum(v*math.log2(v) for v in p if v)
    return max(0.,entropy([(x+y)/2 for x,y in zip(a,b)])-(entropy(a)+entropy(b))/2)


def validate_business(arguments):
    bid,population,cfg,body_lengths,fresh=arguments
    os.environ['TOKENIZERS_PARALLELISM']='false'
    out=ROOT/cfg['output_directory']
    payload=json.loads((out/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json').read_text(encoding='utf-8'))
    tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    tokenizer.no_truncation();tokenizer.no_padding()
    full=sorted(population,key=lambda r:(r['date'],r['review_id']));by_id={r['review_id']:r for r in full};n=len(full)
    assert n>=10 and n==len(by_id)==payload['length_profile']['english_reviews']
    times={};target_time=[]
    for i in range(5):
        group=full[i*n//5:(i+1)*n//5];target_time.append(len(group));times.update({r['review_id']:i for r in group})
    target_stars=Counter(int(r['stars']) for r in full)
    def canonical(ids):return sorted(ids,key=lambda rid:(by_id[rid]['date'],rid))
    exact_cache={}
    def exact(ids):
        assert ids==canonical(ids) and len(ids)==len(set(ids))
        key=tuple(ids)
        if key in exact_cache:return exact_cache[key]
        text='\n\n'.join(f'[REV_{i}] stars={int(by_id[rid]["stars"])} date={by_id[rid]["date"]}\n{by_id[rid]["text"]}' for i,rid in enumerate(ids,1))
        value=(text,len(tokenizer.encode(text,add_special_tokens=True).ids));exact_cache[key]=value
        return value
    def metrics(ids):
        stars=Counter(int(by_id[rid]['stars']) for rid in ids);tm=Counter(times[rid] for rid in ids)
        return divergence([target_stars[i] for i in range(1,6)],[stars[i] for i in range(1,6)]),divergence(target_time,[tm[i] for i in range(5)])
    profile=payload['length_profile']
    assert set(body_lengths)==set(by_id)
    assert profile['individual_body_tokens_min']==min(body_lengths.values())
    assert profile['individual_body_tokens_max']==max(body_lengths.values())
    assert profile['individual_body_tokens_sum']==sum(body_lengths.values())
    costs={rid:exact([rid])[1] for rid in by_id}
    assert profile['complete_singleton_tokens_min']==min(costs.values()) and profile['complete_singleton_tokens_max']==max(costs.values())
    probe=canonical(sorted(costs,key=lambda rid:(costs[rid],by_id[rid]['date'],rid))[:10])
    assert probe==profile['ten_cheapest_singleton_probe_ids'] and len(probe)==10
    probe_tokens=exact(probe)[1];assert probe_tokens==profile['ten_cheapest_singleton_probe_tokens']
    assert not profile['ten_review_impossibility_proven']
    previews={(r['method'],r['review_budget']):r for r in payload['previews']};seen=set();moves=0;frontier_count=0
    for row in payload['runs']:
        key=(row['method'],row['review_budget']);assert key not in seen;seen.add(key)
        ids=row['selected_review_ids'];initial=row['initial_reference_review_ids'];budget=row['review_budget']
        assert row['business_id']==bid and row['split'] in ['train','validation']
        assert row['initial_eligible_review_count']==row['candidate_pool_count']==n
        assert len(initial)==row['initial_candidate_count']==min(n,30)
        assert ids==canonical(ids) and len(ids)==row['final_retained_review_count']<=30
        assert row['below_10_retained_flag']==(len(ids)<10) and row['below_5_retained_flag']==(len(ids)<5)
        assert row['tokens_before_budget_enforcement']==exact(initial)[1]
        rb,tb=metrics(initial);assert abs(rb-row['rating_jsd_before'])<1e-12 and abs(tb-row['temporal_jsd_before'])<1e-12
        assert set(row['removed_vs_reference_ids'])==set(initial)-set(ids) and set(row['added_vs_reference_ids'])==set(ids)-set(initial)
        if ids:
            text,tokens=exact(ids);assert tokens==row['tokens_after_budget_enforcement']<=budget
            ra,ta=metrics(ids);assert abs(ra-row['rating_jsd_after'])<1e-12 and abs(ta-row['temporal_jsd_after'])<1e-12
            assert row['selection_status']=='feasible' and previews[key]['input_text']==text
            assert previews[key]['selected_review_ids']==ids and previews[key]['led_tokens_with_special']==tokens and not previews[key]['final_model_input']
        else:
            assert row['selection_status']=='budget_infeasible' and key not in previews and row['tokens_after_budget_enforcement'] is None
        assert row['ten_complete_reviews_demonstrably_fit']==(probe_tokens<=budget)
        expected=('not_below_ten' if len(ids)>=10 else 'ten_reviews_fit_count_not_forced_by_length' if probe_tokens<=budget else
                  'ten_cheapest_probe_exceeds_budget_observed_length_pressure_not_impossibility_proof')
        assert row['below_ten_length_diagnostic']==expected and not row['ten_review_impossibility_proven']
        assert row['semantic_centroid_similarity'] is None and row['final_review_budget'] is None and not row['methodology_locked']
        if row['method']=='budget_aware':
            path=row['search_provenance']['accepted_local_path']
            for index,step in enumerate(path):
                before=step['before_ids'];after=step['after_ids'];op=step['operation']
                assert set(after)==(set(before)-set(op['out']))|set(op['in'])
                assert exact(before)[1]==step['tokens_before'] and exact(after)[1]==step['tokens_after']<=budget
                if index:assert path[index-1]['after_ids']==before
                r,t=metrics(after);assert abs(r-step['rating_jsd_after'])<1e-12 and abs(t-step['temporal_jsd_after'])<1e-12
                moves+=1
            if path:assert path[-1]['after_ids']==ids
    assert len(seen)==len(payload['runs'])==9
    for budget_key,frontier in payload['frontiers'].items():
        budget=int(budget_key)
        for c in frontier:
            assert exact(c['ids'])[1]==c['tokens']<=budget and c['count']==len(c['ids'])
            r,t=metrics(c['ids']);assert abs(r-c['rating_jsd'])<1e-12 and abs(t-c['temporal_jsd'])<1e-12
            assert abs(.5*r+.5*t-c['weighted_cost'])<1e-12
            assert not any(a['rating_jsd']<=c['rating_jsd']+1e-12 and a['temporal_jsd']<=c['temporal_jsd']+1e-12 and a['count']>=c['count'] and
                           (a['rating_jsd']<c['rating_jsd']-1e-12 or a['temporal_jsd']<c['temporal_jsd']-1e-12 or a['count']>c['count']) for a in frontier)
            frontier_count+=1
        if frontier:
            anchor=min(frontier,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],c['tokens'],tuple(c['ids'])))
            for s in [s for s in payload['sensitivity'] if s['review_budget']==budget]:
                tol=s['tolerance_per_metric']
                eligible=[c for c in frontier if c['rating_jsd']<=anchor['rating_jsd']+tol+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+tol+1e-12]
                winner=min(eligible,key=lambda c:(-c['count'],c['weighted_cost'],c['tokens'],tuple(c['ids'])))
                assert s['selected_review_ids']==winner['ids'] and s['count']==winner['count'] and s['tokens']==winner['tokens']
                if abs(tol-.001)<1e-12:
                    row=next(r for r in payload['runs'] if r['method']=='budget_aware' and r['review_budget']==budget)
                    assert row['selected_review_ids']==winner['ids']
    for s in payload['long_review_stress']:
        assert len(s['selected_review_ids'])==s['count'] in (7,8) and exact(s['selected_review_ids'])[1]==s['exact_tokens']
    matched=0
    if fresh:
        info={'name':payload['runs'][0]['name'],'split':payload['runs'][0]['split'],'volume_band':payload['runs'][0]['volume_band'],'sample_group':'all_train_validation'}
        recomputed=run_business(bid,list(reversed(population)),cfg,tokenizer,info)['runs']
        for a,b in zip(payload['runs'],recomputed):
            assert (a['method'],a['review_budget'],a['selected_review_ids'],a['tokens_after_budget_enforcement'])==(b['method'],b['review_budget'],b['selected_review_ids'],b['tokens_after_budget_enforcement'])
            matched+=1
    return {'business_id':bid,'runs':9,'moves':moves,'frontier':frontier_count,'fresh_runs':matched}


def main(workers):
    started=time.perf_counter();cfg=yaml.safe_load((ROOT/'configs/review_selection_expanded.yaml').read_text(encoding='utf-8'))
    out=ROOT/cfg['output_directory'];source=ROOT/cfg['english_pool_directory'];c1=ROOT/cfg['c1_directory']
    report=json.loads((out/'completion_report.json').read_text());identity=json.loads((out/'scientific_identity.json').read_text())['identity']
    for name,digest in report['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    for name,digest in identity['sources'].items():assert file_hash(source/name)==digest,name
    for name,digest in identity['scripts'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    for prior in identity.get('prior_partials',[]):
        directory=ROOT/prior['directory']
        old=json.loads((directory/'scientific_identity.json').read_text())
        assert old['identity_sha256']==prior['identity_sha256']
        assert old['identity']['sources']==identity['sources'] and old['identity']['scientific_settings']==identity['scientific_settings']
        assert file_hash(directory/'runner_at_run.py')==old['identity']['scripts']['run_expanded_budget_diagnostics.py']
        assert file_hash(directory/'config_at_run.yaml')==old['identity']['config_sha256']
        for name,digest in prior['cache_sha256'].items():assert file_hash(directory/'business_runs'/name)==digest,name
    assert file_hash(ROOT/'configs/review_selection_expanded.yaml')==identity['config_sha256']
    for name,digest in identity['tokenizer_manifest']['sha256'].items():assert file_hash(ROOT/'assets/tokenizers/led-base-16384'/name)==digest
    split=pd.read_csv(source/'business_split.csv',keep_default_na=False);expected=split.loc[split['split'].isin(['train','validation'])].set_index('business_id')['split'].to_dict()
    sample=pd.read_csv(out/'businesses.csv',keep_default_na=False)
    assert sample.set_index('business_id')['split'].to_dict()==expected and len(sample)==2200 and sample['business_id'].is_unique
    assert sample['split'].value_counts().to_dict()=={'train':2000,'validation':200} and report['test_businesses']==0
    populations=defaultdict(list);seen=set()
    for stage in ['train','validation']:
        with (source/'reviews'/f'{stage}.jsonl').open('rb') as stream:
            for line in stream:
                r=orjson.loads(line);rid=r['review_id'];bid=r['business_id']
                assert expected[bid]==stage and rid not in seen;seen.add(rid);populations[bid].append(r)
    assert len(seen)==report['source_reviews']==575749 and set(populations)==set(expected)
    c1_report=json.loads((c1/'completion_report.json').read_text())
    meta=c1/'review_language_token_metadata.csv'
    assert file_hash(meta)==c1_report['diagnostic_artifact_sha256'][meta.name]
    lengths=defaultdict(dict)
    with meta.open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] in ['train','validation'] and row['language']=='en':lengths[row['business_id']][row['review_id']]=int(row['led_tokens_without_special'])
    frame=pd.read_csv(out/'selection_diagnostics.csv');assert len(frame)==19800
    assert not frame.duplicated(['business_id','method','review_budget']).any() and set(frame['business_id'])==set(expected)
    # Verify the flattened artifacts agree with original per-business caches.
    grouped=frame.set_index(['business_id','method','review_budget']).to_dict('index')
    cached_bid=None;cached=None
    with (out/'selection_runs.jsonl').open('rb') as stream:
        count=0
        for line in stream:
            r=orjson.loads(line)
            if r['business_id']!=cached_bid:
                cached=json.loads((out/'business_runs'/f'{hashlib.sha256(r["business_id"].encode()).hexdigest()}.json').read_text(encoding='utf-8'))
                cached_bid=r['business_id']
            original=next(s for s in cached['runs'] if s['method']==r['method'] and s['review_budget']==r['review_budget']);assert original==r
            row=grouped[(r['business_id'],r['method'],r['review_budget'])]
            for key in ['final_retained_review_count','tokens_before_budget_enforcement','tokens_after_budget_enforcement','below_10_retained_flag','below_5_retained_flag','below_ten_length_diagnostic']:
                assert row[key]==r[key] or (pd.isna(row[key]) and r[key] is None),key
            count+=1
    assert count==19800
    cached_bid=None;preview_keys=set()
    with (out/'model_input_previews.jsonl').open('rb') as stream:
        for line in stream:
            r=orjson.loads(line);key=(r['business_id'],r['method'],r['review_budget'])
            assert key not in preview_keys;preview_keys.add(key)
            if r['business_id']!=cached_bid:
                cached=json.loads((out/'business_runs'/f'{hashlib.sha256(r["business_id"].encode()).hexdigest()}.json').read_text(encoding='utf-8'))
                cached_bid=r['business_id']
            original=next(s for s in cached['previews'] if s['method']==r['method'] and s['review_budget']==r['review_budget'])
            assert original==r
    assert preview_keys==set(frame.loc[frame['selection_status'].eq('feasible'),['business_id','method','review_budget']].itertuples(index=False,name=None))
    for filename,subset in [('below_ten_cases_all_methods.csv',frame.loc[frame['below_10_retained_flag']]),
                            ('below_ten_cases_budget_aware.csv',frame.loc[frame['below_10_retained_flag']&frame['method'].eq('budget_aware')])]:
        saved=pd.read_csv(out/filename)
        assert len(saved)==len(subset)
        assert set(saved[['business_id','method','review_budget']].itertuples(index=False,name=None))==set(subset[['business_id','method','review_budget']].itertuples(index=False,name=None))
    profiles=pd.read_csv(out/'business_length_profiles.csv');assert len(profiles)==2200 and profiles['business_id'].is_unique
    for row in profiles.to_dict('records'):
        cached=json.loads((out/'business_runs'/f'{hashlib.sha256(row["business_id"].encode()).hexdigest()}.json').read_text(encoding='utf-8'))
        for key,value in row.items():assert cached['length_profile'][key]==value,key
    summary=pd.read_csv(out/'budget_method_summary.csv')
    for row in summary.to_dict('records'):
        g=frame.loc[frame['method'].eq(row['method'])&frame['review_budget'].eq(row['review_budget'])];valid=g.loc[g['selection_status'].eq('feasible')]
        assert row['businesses']==len(g)==2200
        for field,series,operation in [('below_10',g['below_10_retained_flag'],'sum'),('below_5',g['below_5_retained_flag'],'sum'),
                                       ('retained_min',g['final_retained_review_count'],'min'),('retained_max',g['final_retained_review_count'],'max'),
                                       ('tokens_min',valid['tokens_after_budget_enforcement'],'min'),('tokens_max',valid['tokens_after_budget_enforcement'],'max'),
                                       ('reference_tokens_min',g['tokens_before_budget_enforcement'],'min'),('reference_tokens_max',g['tokens_before_budget_enforcement'],'max')]:
            assert row[field]==getattr(series,operation)(),field
        for band in ['25-30','20-24','15-19','10-14','5-9','<5']:assert row[f'band_{band}']==g['retained_count_band'].eq(band).sum()
        assert row['below_ten_probe_fits']==(g['below_10_retained_flag']&g['ten_complete_reviews_demonstrably_fit']).sum()
        assert row['below_ten_probe_exceeds']==(g['below_10_retained_flag']&~g['ten_complete_reviews_demonstrably_fit']).sum()
    new=frame.loc[frame['method'].eq('budget_aware')]
    flagged=set(new.loc[new['below_10_retained_flag'],'business_id']);fresh=set(flagged)
    ranked=new.sort_values(['tokens_before_budget_enforcement','business_id'],ascending=[False,True])['business_id'].drop_duplicates().tolist()
    for bid in ranked[:16]:fresh.add(bid)
    remainder=sorted(set(expected)-fresh);random.Random(42).shuffle(remainder)
    fresh.update(remainder[:max(0,32-len(fresh))])
    (out/'fresh_validation_business_ids.json').write_text(json.dumps(sorted(fresh),indent=2)+'\n',encoding='utf-8')
    totals=Counter();requests=[(bid,pop,cfg,lengths[bid],bid in fresh) for bid,pop in populations.items()]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(validate_business,args) for args in requests]
        for index,future in enumerate(as_completed(futures),1):
            result=future.result()
            for key in ['runs','moves','frontier','fresh_runs']:totals[key]+=result[key]
            if index%100==0 or index==2200:print(f'Independent validation {index}/2200',flush=True)
    result={'validation_passed':True,'businesses':2200,'source_reviews_verified':len(seen),'selection_runs_verified':totals['runs'],
            'frontier_candidates_verified':totals['frontier'],'accepted_local_moves_verified':totals['moves'],
            'fresh_businesses_verified':len(fresh),'fresh_runs_verified':totals['fresh_runs'],'all_below_ten_businesses_fresh_verified':flagged<=fresh,
            'ten_review_probes_verified':2200,'original_body_lengths_checked_against_C1':True,'exact_tokens_jsd_flags_min_max_verified':True,
            'test_businesses':0,'final_review_budget':None,'final_manifest':False,'semantic_status':'not_computed_model_TBD',
            'elapsed_seconds':round(time.perf_counter()-started,3)}
    assert totals['runs']==19800 and totals['fresh_runs']==9*len(fresh)
    (out/'validation_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();main(args.workers)
