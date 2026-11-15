"""Full train/validation exploratory C2 diagnostics, preserving accepted search."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
import argparse, copy, hashlib, json, os, time
import orjson
import pandas as pd
import yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from review_selection_pilot import ordered, formatted_input, token_count
from run_budget_review_pilot import run_business
from token_length_cache import TokenLengthCache

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/'configs/review_selection_expanded.yaml'
SCIENTIFIC_KEYS=['algorithm_version','seed','selection','diagnostics','format','budget','search','semantic','final_methodology_locked','final_selections_ready']


def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(orjson.dumps(obj,option=orjson.OPT_INDENT_2|orjson.OPT_NON_STR_KEYS)+b'\n')
    for attempt in range(10):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt==9:raise
            time.sleep(min(.02*2**attempt,.2))


def worker(arguments):
    bid,population,cfg,info,reused,reused_origin=arguments
    os.environ['TOKENIZERS_PARALLELISM']='false'
    tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    tokenizer.no_truncation();tokenizer.no_padding()
    tokenizer=TokenLengthCache(tokenizer)
    started=time.perf_counter()
    if reused:
        payload=json.loads(Path(reused).read_text(encoding='utf-8'))
        for run in payload['runs']:
            run['sample_group']='all_train_validation'
            assert run['split']==info['split'] and run['name']==info['name']
        origin=reused_origin
    else:
        payload=run_business(bid,population,cfg,tokenizer,info)
        origin='fresh_expanded_search'
    if reused and 'length_profile' in payload:
        payload['computation_origin']=origin
        payload['elapsed_seconds']=round(time.perf_counter()-started,3)
        return payload
    costs={r['review_id']:len(tokenizer.encode(formatted_input([r],cfg),add_special_tokens=True).ids) for r in population}
    bodies={r['review_id']:len(tokenizer.encode(r['text'],add_special_tokens=False).ids) for r in population}
    by_id={r['review_id']:r for r in population}
    probe_ids=sorted(costs,key=lambda rid:(costs[rid],by_id[rid]['date'],rid))[:10]
    probe=ordered([by_id[rid] for rid in probe_ids]);probe_tokens=token_count(probe,cfg,tokenizer)
    profile={'business_id':bid,'english_reviews':len(population),'individual_body_tokens_min':min(bodies.values()),
             'individual_body_tokens_max':max(bodies.values()),'individual_body_tokens_sum':sum(bodies.values()),
             'complete_singleton_tokens_min':min(costs.values()),'complete_singleton_tokens_max':max(costs.values()),
             'ten_cheapest_singleton_probe_ids':[r['review_id'] for r in probe],
             'ten_cheapest_singleton_probe_tokens':probe_tokens,'ten_review_impossibility_proven':False}
    for row in payload['runs']:
        row.update({k:v for k,v in profile.items() if k not in ['business_id','english_reviews','ten_cheapest_singleton_probe_ids']})
        row['ten_complete_reviews_demonstrably_fit']=probe_tokens<=row['review_budget']
        row['below_ten_length_diagnostic']=('not_below_ten' if not row['below_10_retained_flag'] else
            'ten_reviews_fit_count_not_forced_by_length' if probe_tokens<=row['review_budget'] else
            'ten_cheapest_probe_exceeds_budget_observed_length_pressure_not_impossibility_proof')
    payload['length_profile']=profile
    payload['computation_origin']=origin
    payload['elapsed_seconds']=round(time.perf_counter()-started,3)
    payload['token_count_cache']={'hits':tokenizer.hits,'exact_encodes':tokenizer.misses}
    return payload


def summarize(frame):
    rows=[]
    for (method,budget),g in frame.groupby(['method','review_budget']):
        valid=g.loc[g['selection_status'].eq('feasible')]
        counts=g['final_retained_review_count']
        tokens=valid['tokens_after_budget_enforcement']
        rows.append({'method':method,'review_budget':int(budget),'businesses':len(g),
                     'below_10':int(g['below_10_retained_flag'].sum()),'below_5':int(g['below_5_retained_flag'].sum()),
                     'below_10_percent':100*g['below_10_retained_flag'].mean(),'below_5_percent':100*g['below_5_retained_flag'].mean(),
                     'infeasible':int(len(g)-len(valid)),'retained_min':int(counts.min()),'retained_max':int(counts.max()),'retained_mean':counts.mean(),
                     'tokens_min':tokens.min(),'tokens_max':tokens.max(),'tokens_mean':tokens.mean(),
                     'tokens_p50':tokens.quantile(.5),'tokens_p90':tokens.quantile(.9),'tokens_p95':tokens.quantile(.95),'tokens_p99':tokens.quantile(.99),
                     'reference_tokens_min':g['tokens_before_budget_enforcement'].min(),'reference_tokens_max':g['tokens_before_budget_enforcement'].max(),
                     'rating_jsd_before_mean':g['rating_jsd_before'].mean(),'rating_jsd_after_mean':valid['rating_jsd_after'].mean(),
                     'temporal_jsd_before_mean':g['temporal_jsd_before'].mean(),'temporal_jsd_after_mean':valid['temporal_jsd_after'].mean(),
                     'below_ten_probe_fits':int((g['below_10_retained_flag'] & g['ten_complete_reviews_demonstrably_fit']).sum()),
                     'below_ten_probe_exceeds':int((g['below_10_retained_flag'] & ~g['ten_complete_reviews_demonstrably_fit']).sum()),
                     **{f'band_{b}':int(g['retained_count_band'].eq(b).sum()) for b in ['25-30','20-24','15-19','10-14','5-9','<5']}})
    return pd.DataFrame(rows)


def run(workers):
    started=time.perf_counter();cfg=yaml.safe_load(CFG_PATH.read_text(encoding='utf-8'))
    out=ROOT/cfg['output_directory'];source=ROOT/cfg['english_pool_directory'];prior=ROOT/cfg['accepted_pilot_directory'];c1=ROOT/cfg['c1_directory']
    assert cfg['allowed_splits']==['train','validation'] and not cfg['final_methodology_locked']
    accepted=json.loads((prior/'completion_report.json').read_text())
    assert json.loads((prior/'validation_report.json').read_text())['validation_passed']
    old_cfg=yaml.safe_load((prior/'config_at_run.yaml').read_text(encoding='utf-8'))
    assert all(cfg[k]==old_cfg[k] for k in SCIENTIFIC_KEYS)
    for name,digest in accepted['artifact_sha256'].items():assert file_hash(prior/name)==digest,name
    for name,digest in json.loads((prior/'scientific_identity.json').read_text())['identity']['scripts'].items():
        assert file_hash(ROOT/'scripts'/name)==digest,name
    checks=json.loads((source/'checksums.json').read_text());source_names=['reviews/train.jsonl','reviews/validation.jsonl','business_split.csv']
    before={name:((source/name).stat().st_size,(source/name).stat().st_mtime_ns) for name in source_names}
    for name in source_names:assert file_hash(source/name)==checks[name]
    manifest=json.loads((ROOT/'assets/tokenizers/led-base-16384.manifest.json').read_text())
    for name,digest in manifest['sha256'].items():assert file_hash(ROOT/'assets/tokenizers/led-base-16384'/name)==digest
    c1_complete=json.loads((c1/'completion_report.json').read_text())
    assert file_hash(c1/'business_review_statistics.csv')==c1_complete['diagnostic_artifact_sha256']['business_review_statistics.csv']
    stats=pd.read_csv(c1/'business_review_statistics.csv',keep_default_na=False)
    sample=stats.loc[stats['split'].isin(cfg['allowed_splits'])].sort_values('business_id').copy()
    split=pd.read_csv(source/'business_split.csv',keep_default_na=False)
    expected=split.loc[split['split'].isin(cfg['allowed_splits'])].set_index('business_id')['split'].to_dict()
    assert len(sample)==len(expected)==2200 and sample['business_id'].is_unique
    assert sample.set_index('business_id')['split'].to_dict()==expected
    assert sample['split'].value_counts().to_dict()=={'train':2000,'validation':200}
    sample['volume_band']=pd.cut(sample['english_reviews'],[9,30,100,300,float('inf')],labels=['10-30','31-100','101-300','301+']).astype(str)
    sample['sample_group']='all_train_validation';info=sample.set_index('business_id').to_dict('index')
    identity={'config_sha256':file_hash(CFG_PATH),'accepted_pilot_identity':accepted['identity_sha256'],
              'scientific_settings':{k:cfg[k] for k in SCIENTIFIC_KEYS},'sources':{n:checks[n] for n in source_names},
              'tokenizer_manifest':manifest,'runtime_versions':{n:version(n) for n in ['tokenizers','pandas','numpy','orjson']},
              'scripts':{n:file_hash(ROOT/'scripts'/n) for n in ['run_expanded_budget_diagnostics.py','run_budget_review_pilot.py','budget_aware_review_selection.py','review_selection_pilot.py','token_length_cache.py']}}
    partial_records={};prior_identities=[]
    for directory in cfg['prior_partial_directories']:
        partial=ROOT/directory
        partial_identity=json.loads((partial/'scientific_identity.json').read_text())
        assert partial_identity['identity']['sources']==identity['sources']
        assert partial_identity['identity']['scientific_settings']==identity['scientific_settings']
        assert file_hash(partial/'runner_at_run.py')==partial_identity['identity']['scripts']['run_expanded_budget_diagnostics.py']
        assert file_hash(partial/'config_at_run.yaml')==partial_identity['identity']['config_sha256']
        partial_hashes={p.name:file_hash(p) for p in (partial/'business_runs').glob('*.json')}
        prior_identities.append({'directory':directory,'identity_sha256':partial_identity['identity_sha256'],'cache_sha256':partial_hashes})
        for name in partial_hashes:partial_records[name]=(partial/'business_runs'/name,partial_identity['identity_sha256'])
    identity['prior_partials']=prior_identities
    identity_hash=hashlib.sha256(orjson.dumps(identity,option=orjson.OPT_SORT_KEYS)).hexdigest()
    if (out/'completion_report.json').exists():
        completed=json.loads((out/'completion_report.json').read_text());assert completed['identity_sha256']==identity_hash
        for name,digest in completed['artifact_sha256'].items():assert file_hash(out/name)==digest,name
        print('Verified completed expanded diagnostics.');return
    out.mkdir(parents=True,exist_ok=True);cache=out/'business_runs';cache.mkdir(exist_ok=True)
    if (out/'scientific_identity.json').exists():assert json.loads((out/'scientific_identity.json').read_text())['identity_sha256']==identity_hash
    write_json(out/'scientific_identity.json',{'identity_sha256':identity_hash,'identity':identity})
    (out/'config_at_run.yaml').write_bytes(CFG_PATH.read_bytes());sample.to_csv(out/'businesses.csv',index=False)
    populations={bid:[] for bid in expected};seen=set()
    for stage in cfg['allowed_splits']:
        with (source/'reviews'/f'{stage}.jsonl').open('rb') as stream:
            for line in stream:
                r=orjson.loads(line);bid=r['business_id'];rid=r['review_id']
                assert expected[bid]==stage and rid not in seen;seen.add(rid);populations[bid].append(r)
    assert len(seen)==575749 and all(len(populations[b])==info[b]['english_reviews']>=10 for b in expected)
    previous=set(pd.read_csv(prior/'pilot_businesses.csv')['business_id'])
    pending=[];completed=0
    for bid,pop in populations.items():
        target=cache/f'{hashlib.sha256(bid.encode()).hexdigest()}.json'
        if target.exists():
            value=json.loads(target.read_text(encoding='utf-8'));assert value['expanded_identity_sha256']==identity_hash and value['business_id']==bid
            completed+=1
        else:
            if target.name in partial_records:
                old_partial,old_identity=partial_records[target.name]
                value=json.loads(old_partial.read_text(encoding='utf-8'))
                assert value['expanded_identity_sha256']==old_identity and value['business_id']==bid
                old=old_partial;origin='identity_verified_prior_expanded_partial'
            else:
                old=prior/'business_runs'/target.name if bid in previous else None
                origin='verified_accepted_56_business_pilot' if old else 'fresh_expanded_search'
            pending.append((bid,pop,cfg,info[bid],str(old) if old else None,origin))
    print(f'Prepared {len(sample)} businesses / {len(seen)} original reviews; resumed {completed}; workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(worker,args):args[0] for args in pending}
        try:
            for future in as_completed(futures):
                bid=futures.pop(future);payload=future.result();payload['expanded_identity_sha256']=identity_hash
                write_json(cache/f'{hashlib.sha256(bid.encode()).hexdigest()}.json',payload)
                completed+=1
                if completed%25==0 or completed==2200:
                    print(f'{completed}/2200 completed; elapsed={time.perf_counter()-started:.1f}s',flush=True)
                write_json(out/'run_progress.json',{'completed_businesses':completed,'expected_businesses':2200,'workers':workers,'elapsed_seconds':round(time.perf_counter()-started,1)})
                del payload
        except BaseException as error:
            print(f'Execution stopped immediately: {type(error).__name__}: {error}',flush=True)
            for process in list(executor._processes.values()):process.terminate()
            raise
    rows=[];profiles=[];sensitivity=[];stress=[];origin_counts=defaultdict(int)
    handles={n:(out/f'{n}.jsonl').open('wb') for n in ['selection_runs','model_input_previews']}
    try:
        for bid in expected:
            payload=json.loads((cache/f'{hashlib.sha256(bid.encode()).hexdigest()}.json').read_text(encoding='utf-8'))
            profiles.append(payload['length_profile']);origin_counts[payload['computation_origin']]+=1
            for r in payload['runs']:
                handles['selection_runs'].write(orjson.dumps(r)+b'\n');rows.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))})
            for r in payload['previews']:handles['model_input_previews'].write(orjson.dumps(r)+b'\n')
            sensitivity.extend([{'business_id':bid,**r} for r in payload['sensitivity']]);stress.extend(payload['long_review_stress'])
    finally:
        for stream in handles.values():stream.close()
    frame=pd.DataFrame(rows);assert len(frame)==19800
    frame['retained_count_band']=pd.cut(frame['final_retained_review_count'],[-1,4,9,14,19,24,30],labels=['<5','5-9','10-14','15-19','20-24','25-30']).astype(str)
    frame.to_csv(out/'selection_diagnostics.csv',index=False);summarize(frame).to_csv(out/'budget_method_summary.csv',index=False)
    frame.loc[frame['below_10_retained_flag']].to_csv(out/'below_ten_cases_all_methods.csv',index=False)
    frame.loc[frame['method'].eq('budget_aware') & frame['below_10_retained_flag']].to_csv(out/'below_ten_cases_budget_aware.csv',index=False)
    pd.DataFrame(profiles).drop(columns=['ten_cheapest_singleton_probe_ids']).to_csv(out/'business_length_profiles.csv',index=False)
    write_json(out/'tolerance_sensitivity.json',sensitivity);write_json(out/'long_review_stress.json',stress)
    assert before=={name:((source/name).stat().st_size,(source/name).stat().st_mtime_ns) for name in source_names}
    artifacts={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completion_report.json','run_progress.json']}
    report={'identity_sha256':identity_hash,'businesses':2200,'splits':{'train':2000,'validation':200},'source_reviews':len(seen),
            'selection_runs':len(frame),'test_businesses':0,'origins':dict(origin_counts),'workers':workers,
            'review_budgets':[3500,4000,4500],'final_review_budget':None,'methodology_locked':False,'final_manifest':False,
            'semantic_status':'not_computed_model_TBD','source_metadata_unchanged':True,'artifact_sha256':artifacts,
            'elapsed_seconds':round(time.perf_counter()-started,3)}
    write_json(out/'completion_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    assert 1<=args.workers<=12
    run(args.workers)
