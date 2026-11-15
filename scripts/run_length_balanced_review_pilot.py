"""Versioned v4 pilot with immutable v3 comparisons and durable caches."""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, hashlib, json, os, random, time
import numpy as np
import pandas as pd
import orjson, yaml
from tokenizers import Tokenizer
from budget_aware_review_selection_v4 import search_population
from review_selection_pilot import ordered, diagnostics, formatted_input, stable_seed
from token_length_cache import TokenLengthCache
from english_eligibility import file_hash

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/review_selection_length_balanced_pilot.yaml'


def write_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(orjson.dumps(obj,option=orjson.OPT_INDENT_2|orjson.OPT_NON_STR_KEYS)+b'\n')
    for attempt in range(10):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(min(.02*2**attempt,.2))


def select_sample(cfg):
    all_stats=pd.read_csv(ROOT/cfg['accepted_expanded_directory']/'businesses.csv',keep_default_na=False).set_index('business_id')
    base=pd.read_csv(ROOT/cfg['accepted_pilot_directory']/'pilot_businesses.csv').business_id.tolist()
    audit=json.loads((ROOT/cfg['length_audit_directory']/'completion_report.json').read_text())['example_business_ids']
    reasons=defaultdict(list)
    for bid in base:reasons[bid].append('accepted_56_stress_pilot')
    for bid in audit:reasons[bid].append('length_audit_example')
    for band in ['10-30','31-100','101-300','301+']:
        candidates=sorted(all_stats.loc[all_stats.volume_band.eq(band)&~all_stats.index.isin(reasons)].index)
        rng=random.Random(stable_seed(cfg['seed'],'v4_pilot_sample',band))
        for bid in rng.sample(candidates,cfg['pilot']['additional_seeded_per_source_count_band']):reasons[bid].append('additional_seeded_'+band)
    sample=all_stats.loc[sorted(reasons)].reset_index()
    sample['pilot_reasons']=sample.business_id.map(lambda bid:';'.join(reasons[bid]))
    assert sample.business_id.is_unique and set(sample.split)<={'train','validation'}
    return sample


def worker(args):
    bid,population,cfg,info,prior_path=args
    os.environ['TOKENIZERS_PARALLELISM']='false'
    tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    tokenizer.no_truncation();tokenizer.no_padding();tokenizer=TokenLengthCache(tokenizer)
    prior=json.loads(Path(prior_path).read_text(encoding='utf-8'));by_id={r['review_id']:r for r in population}
    baselines=[r for r in prior['runs'] if r['method']!='budget_aware']
    extra=[(f'{r["method"]}_{r["review_budget"]}',[by_id[rid] for rid in r['selected_review_ids']]) for r in baselines if r['selected_review_ids']]
    ev,choices,frontiers,sensitivity,stats=search_population(population,cfg,tokenizer,bid,extra)
    body={rid:len(tokenizer.encode(r['text'],add_special_tokens=False).ids) for rid,r in by_id.items()}
    sorted_body=np.sort(list(body.values()));rows=[];previews=[]
    for budget in cfg['budget']['exploratory_review_token_scenarios']:
        prior_budget=[r for r in prior['runs'] if r['review_budget']==budget]
        chosen=choices[budget];ids=list(chosen['ids']) if chosen else []
        initial=next(r for r in prior_budget if r['method']=='budget_aware')['initial_reference_review_ids']
        selected=[by_id[rid] for rid in ids];after=diagnostics(population,selected,cfg) if ids else None
        base={'business_id':bid,'name':info['name'],'split':info['split'],'volume_band':info['volume_band'],
              'method':'v4_length_balanced','review_budget':budget,'source_count':len(population),
              'selected_review_ids':ids,'initial_reference_review_ids':initial,
              'final_retained_review_count':len(ids),'tokens_after_budget_enforcement':chosen['tokens'] if chosen else None,
              'rating_jsd_after':after['rating_jsd'] if after else None,'temporal_jsd_after':after['temporal_jsd'] if after else None,
              'date_span_coverage_after':after['date_span_coverage'] if after else None,
              'selection_status':'feasible' if ids else 'budget_infeasible','below_10_retained_flag':len(ids)<10,'below_5_retained_flag':len(ids)<5,
              'removed_vs_reference_ids':[rid for rid in initial if rid not in ids],'added_vs_reference_ids':[rid for rid in ids if rid not in initial],
              'search_provenance':{'origin':chosen['origin'] if chosen else None,'accepted_local_path':chosen['path'] if chosen else []},
              'minimum_cost_singleton_rescue_used':stats[budget]['rescue_used'],'methodology_locked':False,'final_review_budget':None,
              'semantic_status':'not_computed_model_TBD','semantic_centroid_similarity':None}
        rows.append(base)
        for old in prior_budget:
            rows.append({**base,'method':'v3_budget_aware' if old['method']=='budget_aware' else old['method'],
                         **{k:old[k] for k in ['selected_review_ids','initial_reference_review_ids','final_retained_review_count','tokens_after_budget_enforcement',
                                               'rating_jsd_after','temporal_jsd_after','date_span_coverage_after','selection_status','below_10_retained_flag',
                                               'below_5_retained_flag','removed_vs_reference_ids','added_vs_reference_ids','search_provenance']},
                         'minimum_cost_singleton_rescue_used':False})
    for row in rows:
        vals=np.array([body[rid] for rid in row['selected_review_ids']],dtype=float)
        row.update(source_body_mean=float(sorted_body.mean()),source_body_median=float(np.median(sorted_body)),
                   selected_body_sum=int(vals.sum()),selected_body_mean=float(vals.mean()) if len(vals) else None,
                   selected_body_median=float(np.median(vals)) if len(vals) else None,
                   selected_source_mean_ratio=float(vals.mean()/sorted_body.mean()) if len(vals) else None,
                   selected_length_percentile=float(((np.searchsorted(sorted_body,vals,side='left')+np.searchsorted(sorted_body,vals,side='right'))/(2*len(sorted_body))).mean()) if len(vals) else None,
                   source_lt20_fraction=float((sorted_body<20).mean()),selected_lt20_fraction=float((vals<20).mean()) if len(vals) else None)
        if len(vals):previews.append({'business_id':bid,'method':row['method'],'review_budget':row['review_budget'],
                                    'selected_review_ids':row['selected_review_ids'],'input_text':formatted_input([by_id[rid] for rid in row['selected_review_ids']],cfg),
                                    'led_tokens_with_special':row['tokens_after_budget_enforcement'],'final_model_input':False})
    return {'business_id':bid,'runs':rows,'previews':previews,'frontiers':{str(k):v for k,v in frontiers.items()},
            'sensitivity':sensitivity,'search_statistics':stats,'candidate_audit':ev.audit}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    start=time.perf_counter();cfg=yaml.safe_load(CONFIG.read_text());out=ROOT/cfg['output_directory'];source=ROOT/cfg['english_pool_directory']
    old=ROOT/cfg['accepted_expanded_directory'];accepted=json.loads((old/'completion_report.json').read_text())
    assert json.loads((old/'validation_report.json').read_text())['validation_passed'] and not cfg['final_methodology_locked']
    old_identity=json.loads((old/'scientific_identity.json').read_text())['identity']
    for name,value in old_identity['sources'].items():assert file_hash(source/name)==value,name
    for name,value in old_identity['scripts'].items():assert file_hash(ROOT/'scripts'/name)==value,name
    assert file_hash(ROOT/'configs/review_selection_expanded.yaml')==old_identity['config_sha256']
    sample=select_sample(cfg);sample.to_dict('records');membership=sample.set_index('business_id').split.to_dict()
    info=sample.set_index('business_id').to_dict('index');populations={bid:[] for bid in membership}
    snapshots={str(source/name):((source/name).stat().st_size,(source/name).stat().st_mtime_ns) for name in old_identity['sources']}
    for split in ['train','validation']:
        for line in (source/'reviews'/f'{split}.jsonl').open('rb'):
            r=orjson.loads(line)
            if r['business_id'] in populations:
                assert membership[r['business_id']]==split;populations[r['business_id']].append(r)
    prior_paths={bid:old/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json' for bid in membership}
    prior_hashes={bid:file_hash(path) for bid,path in prior_paths.items()}
    for bid,path in prior_paths.items():assert prior_hashes[bid]==accepted['artifact_sha256'][path.relative_to(old).as_posix()]
    tokenizer_digest=file_hash(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json')
    assert tokenizer_digest==old_identity['tokenizer_manifest']['sha256']['tokenizer.json']
    identity={'config_sha256':file_hash(CONFIG),'accepted_expanded_identity':accepted['identity_sha256'],
              'sources':old_identity['sources'],'tokenizer_sha256':tokenizer_digest,'prior_business_cache_sha256':prior_hashes,
              'scripts':{n:file_hash(ROOT/'scripts'/n) for n in ['run_length_balanced_review_pilot.py','budget_aware_review_selection_v4.py',
                           'budget_aware_review_selection.py','review_selection_pilot.py','token_length_cache.py']}}
    identity_hash=hashlib.sha256(orjson.dumps(identity,option=orjson.OPT_SORT_KEYS)).hexdigest()
    if (out/'completion_report.json').exists():
        report=json.loads((out/'completion_report.json').read_text());assert report['identity_sha256']==identity_hash
        for name,value in report['artifact_sha256'].items():assert file_hash(out/name)==value
        print('Completed v4 pilot already verified; no rerun.');return
    out.mkdir(parents=True,exist_ok=True);cache=out/'business_runs';cache.mkdir(exist_ok=True)
    if (out/'scientific_identity.json').exists():assert json.loads((out/'scientific_identity.json').read_text())['identity_sha256']==identity_hash
    write_json(out/'scientific_identity.json',{'identity_sha256':identity_hash,'identity':identity})
    (out/'config_at_run.yaml').write_bytes(CONFIG.read_bytes());sample.to_csv(out/'pilot_businesses.csv',index=False)
    with (out/'pilot_population.jsonl').open('wb') as stream:
        for bid,pop in populations.items():
            assert len(pop)==int(info[bid]['english_reviews'])
            for r in ordered(pop):stream.write(orjson.dumps({'split':membership[bid],'review':r})+b'\n')
    missing=[];done=0
    for bid,pop in populations.items():
        path=cache/f'{hashlib.sha256(bid.encode()).hexdigest()}.json'
        if path.exists():assert json.loads(path.read_text())['identity_sha256']==identity_hash;done+=1
        else:missing.append((bid,pop,cfg,info[bid],str(prior_paths[bid])))
    print(f'Pilot {len(sample)} businesses / {sum(map(len,populations.values()))} originals; pending {len(missing)}.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,a):a[0] for a in missing}
        try:
            for future in as_completed(futures):
                bid=futures.pop(future);payload=future.result();payload['identity_sha256']=identity_hash
                write_json(cache/f'{hashlib.sha256(bid.encode()).hexdigest()}.json',payload);done+=1
                write_json(out/'run_progress.json',{'completed_businesses':done,'businesses':len(sample),'elapsed_seconds':round(time.perf_counter()-start,1)})
                print(f'{done}/{len(sample)} {bid}: v4 '+str([r['final_retained_review_count'] for r in payload['runs'] if r['method']=='v4_length_balanced']),flush=True)
                del payload,future
        except BaseException:
            for process in pool._processes.values():process.terminate()
            raise
    rows=[];sens=[]
    with (out/'selection_runs.jsonl').open('wb') as runs,(out/'model_input_previews.jsonl').open('wb') as previews:
        for bid in membership:
            payload=json.loads((cache/f'{hashlib.sha256(bid.encode()).hexdigest()}.json').read_text())
            for r in payload['runs']:
                runs.write(orjson.dumps(r)+b'\n');rows.append({k:v for k,v in r.items() if not isinstance(v,(list,dict))})
            for r in payload['previews']:previews.write(orjson.dumps(r)+b'\n')
            sens.extend({'business_id':bid,**r} for r in payload['sensitivity'])
    df=pd.DataFrame(rows);df.to_csv(out/'selection_diagnostics.csv',index=False)
    df.loc[df.below_10_retained_flag].to_csv(out/'below_ten_cases.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'tolerance_sensitivity.csv',index=False)
    summaries=[]
    for (method,budget),g in df.groupby(['method','review_budget']):
        valid=g[g.selection_status.eq('feasible')]
        summaries.append({'method':method,'review_budget':int(budget),'businesses':len(g),'retained_mean':float(g.final_retained_review_count.mean()),
                          'retained_min':int(g.final_retained_review_count.min()),'retained_max':int(g.final_retained_review_count.max()),
                          'below10':int(g.below_10_retained_flag.sum()),'below5':int(g.below_5_retained_flag.sum()),'infeasible':len(g)-len(valid),
                          'tokens_min':int(valid.tokens_after_budget_enforcement.min()) if len(valid) else None,'tokens_max':int(valid.tokens_after_budget_enforcement.max()) if len(valid) else None,
                          'rating_jsd_mean':float(valid.rating_jsd_after.mean()),'temporal_jsd_mean':float(valid.temporal_jsd_after.mean()),
                          'selected_body_mean_macro':float(valid.selected_body_mean.mean()),'source_body_mean_macro':float(valid.source_body_mean.mean()),
                          'median_mean_ratio':float(valid.selected_source_mean_ratio.median()),'mean_length_percentile':float(valid.selected_length_percentile.mean()),
                          'source_short20_macro':float(valid.source_lt20_fraction.mean()),'selected_short20_macro':float(valid.selected_lt20_fraction.mean()),
                          'exactly30_under1500':int((g.final_retained_review_count.eq(30)&g.tokens_after_budget_enforcement.lt(1500)).sum()),
                          'rescue_used':int(g.minimum_cost_singleton_rescue_used.sum()),
                          **{f'band_{label}':int(g.final_retained_review_count.between(lo,hi).sum()) for label,lo,hi in [('25-30',25,30),('20-24',20,24),('15-19',15,19),('10-14',10,14),('5-9',5,9),('<5',0,4)]}})
    pd.DataFrame(summaries).to_csv(out/'budget_method_summary.csv',index=False)
    paired=[]
    for budget in [3500,4000,4500]:
        a=df.loc[df.method.eq('v4_length_balanced')&df.review_budget.eq(budget)].set_index('business_id')
        b=df.loc[df.method.eq('v3_budget_aware')&df.review_budget.eq(budget)].set_index('business_id').reindex(a.index)
        pair=pd.DataFrame({'business_id':a.index,'review_budget':budget,'count_delta':a.final_retained_review_count-b.final_retained_review_count,
                           'rating_delta':a.rating_jsd_after-b.rating_jsd_after,'time_delta':a.temporal_jsd_after-b.temporal_jsd_after,
                           'body_mean_delta':a.selected_body_mean-b.selected_body_mean,'mean_ratio_delta':a.selected_source_mean_ratio-b.selected_source_mean_ratio})
        paired.append(pair.reset_index(drop=True))
    pd.concat(paired,ignore_index=True).to_csv(out/'paired_v4_v3_differences.csv',index=False)
    assert snapshots=={str(source/name):((source/name).stat().st_size,(source/name).stat().st_mtime_ns) for name in old_identity['sources']}
    artifacts={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completion_report.json','run_progress.json']}
    report={'identity_sha256':identity_hash,'businesses':len(sample),'splits':sample.split.value_counts().to_dict(),'population_reviews':sum(map(len,populations.values())),
            'selection_runs':len(rows),'test_businesses':0,'final_manifest':False,'final_review_budget':None,'methodology_locked':False,
            'source_metadata_unchanged':True,'semantic_status':'not_computed_model_TBD','artifact_sha256':artifacts,'elapsed_seconds':round(time.perf_counter()-start,3)}
    write_json(out/'completion_report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
