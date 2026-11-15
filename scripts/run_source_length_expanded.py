"""All2200 v4.2 diagnostics; preserve witness policy and resume by identity."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,copy,hashlib,json,os,shutil,time
import orjson,pandas as pd,yaml
from english_eligibility import file_hash
from run_shared_pool_review_pilot import write_json,worker as worker41
from run_length_balanced_review_pilot import worker as worker4
from run_source_length_tie_pilot import worker as worker42
from review_scoring_memoization import memoized_review_scoring

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/review_selection_source_length_expanded.yaml'


def business_worker(args):
    with memoized_review_scoring():return uncached_business_worker(args)


def uncached_business_worker(args):
    bid,pop,cfg,info,identity=args
    filename=hashlib.sha256(bid.encode()).hexdigest()+'.json';out=ROOT/cfg['output_directory']
    prior=ROOT/cfg['accepted_expanded_directory']/'business_runs'/filename
    support4=ROOT/cfg['accepted_v4_directory']/'business_runs'/filename
    support41=ROOT/cfg['accepted_v4_1_directory']/'business_runs'/filename
    pilot4=ROOT/cfg['original_v4_directory']/'business_runs'/filename
    pilot41=ROOT/cfg['original_v4_1_directory']/'business_runs'/filename
    accepted80=pilot41.is_file()
    for target,original,version in [(support4,pilot4,4),(support41,pilot41,41)]:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            cached=json.loads(target.read_text(encoding='utf-8'))
            if accepted80:assert file_hash(target)==file_hash(original)
            else:assert cached['expanded_identity_sha256'] in cfg.get('_compatible_cache_identities',[identity])
        elif accepted80:shutil.copy2(original,target)
        else:
            legacy=yaml.safe_load((ROOT/('configs/review_selection_length_balanced_pilot.yaml' if version==4 else 'configs/review_selection_shared_pool_pilot.yaml')).read_text(encoding='utf-8'))
            if version==41:legacy['accepted_v4_directory']=cfg['accepted_v4_directory']
            payload=(worker4 if version==4 else worker41)((bid,pop,legacy,info,str(prior)))
            payload['expanded_identity_sha256']=identity;write_json(target,payload)
    payload=worker42((bid,pop,cfg,info))
    if accepted80:
        original=json.loads((ROOT/cfg['accepted_v4_2_pilot_directory']/'business_runs'/filename).read_text(encoding='utf-8'));original.pop('identity_sha256')
        assert orjson.dumps(original,option=orjson.OPT_SORT_KEYS|orjson.OPT_NON_STR_KEYS)==orjson.dumps(payload,option=orjson.OPT_SORT_KEYS|orjson.OPT_NON_STR_KEYS),bid
    payload['identity_sha256']=identity;payload['expanded_witness_origin']='accepted80_witnesses_payload_equal' if accepted80 else 'fresh_v4_v4_1_witnesses'
    return payload


def summarize(frame,columns):
    result=[]
    for keys,g in frame.groupby(columns,dropna=False):
        if not isinstance(keys,tuple):keys=(keys,)
        valid=g[g.selection_status.eq('feasible')]
        result.append({**dict(zip(columns,keys)),'businesses':len(g),'retained_mean':float(g.final_retained_review_count.mean()),'retained_min':int(g.final_retained_review_count.min()),'retained_max':int(g.final_retained_review_count.max()),'below10':int(g.below_10_retained_flag.sum()),'below5':int(g.below_5_retained_flag.sum()),'infeasible':int(g.selection_status.ne('feasible').sum()),'tokens_min':int(valid.tokens_after_budget_enforcement.min()) if len(valid) else None,'tokens_max':int(valid.tokens_after_budget_enforcement.max()) if len(valid) else None,'rating_jsd_mean':float(valid.rating_jsd_after.mean()),'temporal_jsd_mean':float(valid.temporal_jsd_after.mean()),'length_jsd_mean':float(valid.length_jsd.mean()),'median_mean_ratio':float(valid.selected_source_mean_ratio.median()),'source_short20_macro':float(valid.source_lt20_fraction.mean()),'selected_short20_macro':float(valid.selected_lt20_fraction.mean()),'exactly30_under1500':int((g.final_retained_review_count.eq(30)&g.tokens_after_budget_enforcement.lt(1500)).sum()),**{f'band_{name}':int(g.final_retained_review_count.between(lo,hi).sum()) for name,lo,hi in [('25-30',25,30),('20-24',20,24),('15-19',15,19),('10-14',10,14),('5-9',5,9),('<5',0,4)]}})
    return pd.DataFrame(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();started=time.perf_counter()
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];old=ROOT/cfg['accepted_expanded_directory']
    protected={}
    for directory in [old,ROOT/cfg['original_v4_directory'],ROOT/cfg['original_v4_1_directory'],ROOT/cfg['accepted_v4_2_pilot_directory']]:
        assert json.loads((directory/'validation_report.json').read_text(encoding='utf-8'))['validation_passed']
        report=json.loads((directory/'completion_report.json').read_text(encoding='utf-8'))
        for name,digest in report['artifact_sha256'].items():
            p=directory/name;assert file_hash(p)==digest,name;protected[p.relative_to(ROOT).as_posix()]=digest
        for name in ['completion_report.json','validation_report.json']:protected[(directory/name).relative_to(ROOT).as_posix()]=file_hash(directory/name)
    for relative in ['data/processed/yelp_2300_english_v1/reviews/train.jsonl','data/processed/yelp_2300_english_v1/reviews/validation.jsonl','data/processed/yelp_2300_english_v1/business_split.csv','configs/research_pending.yaml','assets/tokenizers/led-base-16384/tokenizer.json']:
        protected[relative]=file_hash(ROOT/relative)
    pilotcfg=yaml.safe_load((ROOT/'configs/review_selection_source_length_tie_pilot.yaml').read_text(encoding='utf-8'))
    for key in ['seed','algorithm_version','sampling_and_tie_algorithm_version','selection','diagnostics','format','budget','search','secondary_length_balance','semantic']:assert cfg[key]==pilotcfg[key],key
    sample=pd.read_csv(old/'businesses.csv',keep_default_na=False);assert len(sample)==2200 and sample.split.value_counts().to_dict()=={'train':2000,'validation':200}
    info=sample.set_index('business_id').to_dict('index');populations={bid:[] for bid in info}
    for split in ['train','validation']:
        for line in (ROOT/cfg['english_pool_directory']/'reviews'/f'{split}.jsonl').open('rb'):
            r=orjson.loads(line);assert r['business_id'] in populations and info[r['business_id']]['split']==split;populations[r['business_id']].append(r)
    assert sum(map(len,populations.values()))==575749
    identity={'config_sha256':file_hash(CONFIG),'protected_sha256':protected,'scripts':{name:file_hash(ROOT/'scripts'/name) for name in ['run_source_length_expanded.py','review_scoring_memoization.py','run_source_length_tie_pilot.py','run_length_balanced_review_pilot.py','run_shared_pool_review_pilot.py','budget_aware_review_selection_v4_2.py','budget_aware_review_selection_v4_1.py','budget_aware_review_selection_v4.py','budget_aware_review_selection.py','review_selection_pilot.py','token_length_cache.py']},'legacy_config_sha256':{name:file_hash(ROOT/'configs'/name) for name in ['review_selection_length_balanced_pilot.yaml','review_selection_shared_pool_pilot.yaml','review_selection_source_length_tie_pilot.yaml']}}
    prior_partial=out/'pre_memoization_run/scientific_identity.json'
    compatible=[]
    if prior_partial.exists():
        previous=json.loads(prior_partial.read_text(encoding='utf-8'))
        assert previous['identity']['config_sha256']==identity['config_sha256'] and previous['identity']['protected_sha256']==identity['protected_sha256'] and previous['identity']['legacy_config_sha256']==identity['legacy_config_sha256']
        for name,digest in previous['identity']['scripts'].items():
            target=out/'pre_memoization_run/scripts'/name if name=='run_source_length_expanded.py' else ROOT/'scripts'/name
            assert file_hash(target)==digest,name
        compatible=[previous['identity_sha256']]
        identity['compatible_pre_memoization_identity']=compatible[0]
        identity['pre_memoization_partial_manifest_sha256']=file_hash(out/'pre_memoization_run/partial_artifact_sha256.json')
    identity_hash=hashlib.sha256(orjson.dumps(identity,option=orjson.OPT_SORT_KEYS)).hexdigest()
    if (out/'completion_report.json').exists():
        report=json.loads((out/'completion_report.json').read_text(encoding='utf-8'));assert report['identity_sha256']==identity_hash
        for name,digest in report['artifact_sha256'].items():assert file_hash(out/name)==digest
        print('Completed expanded computation verified; no rerun.');return
    out.mkdir(parents=True,exist_ok=True);cache=out/'business_runs';cache.mkdir(exist_ok=True)
    if (out/'scientific_identity.json').exists():
        existing=json.loads((out/'scientific_identity.json').read_text(encoding='utf-8'));assert existing['identity_sha256'] in [identity_hash,*compatible]
        if existing['identity_sha256']!=identity_hash:
            for relative,digest in json.loads((out/'pre_memoization_run/partial_artifact_sha256.json').read_text(encoding='utf-8')).items():assert file_hash(out/relative)==digest,relative
    cfg['_compatible_cache_identities']=[identity_hash,*compatible]
    write_json(out/'scientific_identity.json',{'identity_sha256':identity_hash,'identity':identity});(out/'config_at_run.yaml').write_bytes(CONFIG.read_bytes());sample.to_csv(out/'businesses.csv',index=False)
    pending=[];done=0
    for bid,pop in populations.items():
        assert len(pop)==int(info[bid]['english_reviews']);p=cache/(hashlib.sha256(bid.encode()).hexdigest()+'.json')
        if p.exists():assert json.loads(p.read_text(encoding='utf-8'))['identity_sha256'] in [identity_hash,*compatible];done+=1
        else:pending.append((bid,pop,cfg,info[bid],identity_hash))
    print(f'2200 businesses /575749 originals; pending {len(pending)}; workers {args.workers}.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(business_worker,a):a[0] for a in pending}
        for future in as_completed(futures):
            bid=futures.pop(future);payload=future.result();write_json(cache/(hashlib.sha256(bid.encode()).hexdigest()+'.json'),payload);done+=1
            if done%20==0 or done==2200:
                elapsed=round(time.perf_counter()-started,1);write_json(out/'run_progress.json',{'completed_businesses':done,'businesses':2200,'elapsed_seconds':elapsed});print(f'{done}/2200 elapsed {elapsed}s',flush=True)
            del payload,future
    rows=[];sens=[]
    with (out/'selection_runs.jsonl').open('wb') as stream:
        for bid in info:
            payload=json.loads((cache/(hashlib.sha256(bid.encode()).hexdigest()+'.json')).read_text(encoding='utf-8'))
            for row in payload['runs']:
                row['expanded_witness_origin']=payload['expanded_witness_origin'];stream.write(orjson.dumps(row)+b'\n');rows.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
            sens.extend({'business_id':bid,**row} for row in payload['sensitivity'])
    frame=pd.DataFrame(rows);frame.to_csv(out/'selection_diagnostics.csv',index=False);frame[frame.below_10_retained_flag].to_csv(out/'below_ten_cases.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'tolerance_sensitivity.csv',index=False)
    summarize(frame,['method','review_budget']).to_csv(out/'budget_method_summary.csv',index=False)
    for column in ['split','volume_band','expanded_witness_origin']:summarize(frame,['method','review_budget',column]).to_csv(out/f'summary_by_{column}.csv',index=False)
    for p,digest in protected.items():assert file_hash(ROOT/p)==digest,p
    report={'identity_sha256':identity_hash,'businesses':2200,'source_reviews':575749,'splits':sample.split.value_counts().to_dict(),'selection_runs':len(rows),'new_method_runs':19800,'test_businesses':0,'accepted80_payload_equal':80,'fresh_witness_businesses':2120,'final_manifest':False,'final_review_budget':None,'methodology_locked':False,'source_and_accepted_evidence_unchanged':True,'workers':args.workers,'elapsed_seconds':round(time.perf_counter()-started,3)}
    report['artifact_sha256']={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completion_report.json','run_progress.json']}
    write_json(out/'completion_report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
