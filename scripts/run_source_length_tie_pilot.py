"""Resumable same80 fresh discovery + declared incumbent candidate retention."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,os,time
import numpy as np
import pandas as pd
import orjson,yaml
from tokenizers import Tokenizer
from budget_aware_review_selection_v4_2 import search_population,length_profile,length_score
from review_selection_pilot import ordered,diagnostics,formatted_input
from token_length_cache import TokenLengthCache
from english_eligibility import file_hash
from run_shared_pool_review_pilot import write_json

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/review_selection_source_length_tie_pilot.yaml'


def worker(args):
    bid,pop,cfg,info=args
    os.environ['TOKENIZERS_PARALLELISM']='false'
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));tok.no_truncation();tok.no_padding()
    tok=TokenLengthCache(tok)
    filename=hashlib.sha256(bid.encode()).hexdigest()+'.json'
    old4=json.loads((ROOT/cfg['accepted_v4_directory']/'business_runs'/filename).read_text(encoding='utf-8'))
    old41=json.loads((ROOT/cfg['accepted_v4_1_directory']/'business_runs'/filename).read_text(encoding='utf-8'))
    baselines=[r for r in old41['runs'] if r['method'] in ['v3_budget_aware','v4_length_balanced','v4_1_shared_pool']]
    incumbents=[(f"{r['method']}_{r['review_budget']}",r['selected_review_ids']) for r in baselines if r['selected_review_ids']]
    retained={tuple(c['ids']) for c in old41['shared_candidates']}
    retained.update(tuple(c['ids']) for c in old4['candidate_audit'] if c['origin'].get('variant')!='minimum_cost_singleton_rescue')
    result=search_population(pop,cfg,tok,bid,incumbents,sorted(retained))
    byid={r['review_id']:r for r in pop};body=result['body_lengths'];profile=length_profile(body,cfg['secondary_length_balance']['length_bins'])
    source=np.sort(list(body.values()));rows=[];previews=[]
    for budget,methods in result['choices'].items():
        reference=next(r for r in baselines if r['method']=='v3_budget_aware' and r['review_budget']==budget)['initial_reference_review_ids']
        before=diagnostics(pop,[byid[rid] for rid in reference],cfg)
        before_tokens=len(tok.encode(formatted_input([byid[rid] for rid in reference],cfg),add_special_tokens=True).ids)
        control=methods['v4_2_primary_control']
        for method,c in methods.items():
            ids=list(c['ids']) if c else [];selected=[byid[rid] for rid in ids]
            after=diagnostics(pop,selected,cfg) if c else None
            rows.append({'business_id':bid,'name':info['name'],'split':info['split'],'volume_band':info['volume_band'],'method':method,'review_budget':budget,'source_count':len(pop),'initial_eligible_review_count':len(pop),'initial_candidate_count':len(reference),'initial_reference_review_ids':reference,'selected_review_ids':ids,'final_retained_review_count':len(ids),'tokens_before_budget_enforcement':before_tokens,'tokens_after_budget_enforcement':c['tokens'] if c else None,'rating_jsd_before':before['rating_jsd'],'temporal_jsd_before':before['temporal_jsd'],'rating_jsd_after':after['rating_jsd'] if c else None,'temporal_jsd_after':after['temporal_jsd'] if c else None,'date_span_coverage_after':after['date_span_coverage'] if c else None,'primary_control_ids':list(control['ids']) if control else [],'equivalent_candidates':len(result['equivalent_groups'][budget]),'selection_status':'feasible' if c else 'budget_infeasible','below_10_retained_flag':len(ids)<10,'below_5_retained_flag':len(ids)<5,'removed_vs_reference_ids':[rid for rid in reference if rid not in ids],'added_vs_reference_ids':[rid for rid in ids if rid not in reference],'removed_vs_primary_control_ids':[rid for rid in control['ids'] if rid not in ids] if control else [],'added_vs_primary_control_ids':[rid for rid in ids if rid not in control['ids']] if control else [],'methodology_locked':False,'final_model_input':False,'final_review_budget':None,'semantic_status':'not_computed_model_TBD','semantic_centroid_similarity':None})
    rows.extend(dict(r) for r in baselines)
    for row in rows:
        ids=row['selected_review_ids'];values=np.array([body[rid] for rid in ids],dtype=float)
        row.update(source_body_mean=float(source.mean()),selected_body_mean=float(values.mean()) if len(values) else None,selected_source_mean_ratio=float(values.mean()/source.mean()) if len(values) else None,source_lt20_fraction=float((source<20).mean()),selected_lt20_fraction=float((values<20).mean()) if len(values) else None,length_jsd=length_score(ids,profile) if ids else None,final_model_input=False)
        if ids:previews.append({'business_id':bid,'method':row['method'],'review_budget':row['review_budget'],'selected_review_ids':ids,'input_text':formatted_input([byid[rid] for rid in ids],cfg),'led_tokens_with_special':row['tokens_after_budget_enforcement'],'final_model_input':False})
    result['choices']={str(b):{m:list(c['ids']) if c else [] for m,c in methods.items()} for b,methods in result['choices'].items()}
    return {'business_id':bid,'runs':rows,'previews':previews,'retained_candidate_ids':[list(ids) for ids in sorted(retained)],'incumbents':incumbents,**result}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    started=time.perf_counter();cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory']
    accepted=ROOT/cfg['accepted_v4_1_directory'];sample=pd.read_csv(accepted/'pilot_businesses.csv',keep_default_na=False)
    assert len(sample)==80 and set(sample.split)=={'train','validation'}
    protected={}
    for directory in [ROOT/cfg['accepted_v4_directory'],accepted]:
        completion=json.loads((directory/'completion_report.json').read_text(encoding='utf-8'))
        assert json.loads((directory/'validation_report.json').read_text(encoding='utf-8'))['validation_passed']
        for name,digest in completion['artifact_sha256'].items():
            p=directory/name;assert file_hash(p)==digest,name;protected[p.relative_to(ROOT).as_posix()]=digest
        for name in ['completion_report.json','validation_report.json']:protected[(directory/name).relative_to(ROOT).as_posix()]=file_hash(directory/name)
    for relative in ['data/processed/yelp_2300_english_v1/reviews/train.jsonl','data/processed/yelp_2300_english_v1/reviews/validation.jsonl','data/processed/yelp_2300_english_v1/business_split.csv','configs/research_pending.yaml','assets/tokenizers/led-base-16384/tokenizer.json','scripts/budget_aware_review_selection_v4.py','scripts/budget_aware_review_selection_v4_1.py','scripts/budget_aware_review_selection.py','scripts/review_selection_pilot.py','scripts/token_length_cache.py']:
        protected[relative]=file_hash(ROOT/relative)
    oldscience=json.loads((accepted/'scientific_identity.json').read_text(encoding='utf-8'))['identity']
    for name,digest in oldscience['sources'].items():assert file_hash(ROOT/cfg['english_pool_directory']/name)==digest
    membership=sample.set_index('business_id').split.to_dict();info=sample.set_index('business_id').to_dict('index');populations={bid:[] for bid in membership}
    for split in ['train','validation']:
        for line in (ROOT/cfg['english_pool_directory']/'reviews'/f'{split}.jsonl').open('rb'):
            review=orjson.loads(line);bid=review['business_id']
            if bid in populations:assert membership[bid]==split;populations[bid].append(review)
    assert sum(map(len,populations.values()))==40407
    identity={'config_sha256':file_hash(CONFIG),'protected_sha256':protected,'scripts':{name:file_hash(ROOT/'scripts'/name) for name in ['run_source_length_tie_pilot.py','budget_aware_review_selection_v4_2.py']}}
    identity_hash=hashlib.sha256(orjson.dumps(identity,option=orjson.OPT_SORT_KEYS)).hexdigest()
    if (out/'completion_report.json').exists():
        report=json.loads((out/'completion_report.json').read_text(encoding='utf-8'));assert report['identity_sha256']==identity_hash
        for name,digest in report['artifact_sha256'].items():assert file_hash(out/name)==digest
        print('Accepted completed computation verified; no rerun.');return
    out.mkdir(parents=True,exist_ok=True);cache=out/'business_runs';cache.mkdir(exist_ok=True)
    if (out/'scientific_identity.json').exists():assert json.loads((out/'scientific_identity.json').read_text(encoding='utf-8'))['identity_sha256']==identity_hash
    write_json(out/'scientific_identity.json',{'identity_sha256':identity_hash,'identity':identity})
    (out/'config_at_run.yaml').write_bytes(CONFIG.read_bytes());sample.to_csv(out/'pilot_businesses.csv',index=False)
    with (out/'pilot_population.jsonl').open('wb') as stream:
        for bid,pop in populations.items():
            assert len(pop)==int(info[bid]['english_reviews'])
            for review in ordered(pop):stream.write(orjson.dumps({'split':membership[bid],'review':review})+b'\n')
    pending=[];done=0
    for bid,pop in populations.items():
        p=cache/(hashlib.sha256(bid.encode()).hexdigest()+'.json')
        if p.exists():assert json.loads(p.read_text(encoding='utf-8'))['identity_sha256']==identity_hash;done+=1
        else:pending.append((bid,pop,cfg,info[bid]))
    print(f'{len(sample)} businesses /40407 originals; {len(pending)} pending.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,a):a[0] for a in pending}
        for future in as_completed(futures):
            payload=future.result();payload['identity_sha256']=identity_hash
            write_json(cache/(hashlib.sha256(payload['business_id'].encode()).hexdigest()+'.json'),payload);done+=1
            print(f'{done}/80 '+payload['business_id'],flush=True)
    rows=[];sens=[]
    with (out/'selection_runs.jsonl').open('wb') as runs,(out/'model_input_previews.jsonl').open('wb') as previews:
        for bid in membership:
            payload=json.loads((cache/(hashlib.sha256(bid.encode()).hexdigest()+'.json')).read_text(encoding='utf-8'))
            for row in payload['runs']:runs.write(orjson.dumps(row)+b'\n');rows.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
            for preview in payload['previews']:previews.write(orjson.dumps(preview)+b'\n')
            sens.extend({'business_id':bid,**row} for row in payload['sensitivity'])
    frame=pd.DataFrame(rows);frame.to_csv(out/'selection_diagnostics.csv',index=False);frame[frame.below_10_retained_flag].to_csv(out/'below_ten_cases.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'tolerance_sensitivity.csv',index=False)
    summaries=[]
    for (method,budget),g in frame.groupby(['method','review_budget']):
        summaries.append({'method':method,'review_budget':int(budget),'businesses':len(g),'retained_mean':float(g.final_retained_review_count.mean()),'retained_min':int(g.final_retained_review_count.min()),'retained_max':int(g.final_retained_review_count.max()),'below10':int(g.below_10_retained_flag.sum()),'below5':int(g.below_5_retained_flag.sum()),'infeasible':int(g.selection_status.ne('feasible').sum()),'tokens_min':int(g.tokens_after_budget_enforcement.min()),'tokens_max':int(g.tokens_after_budget_enforcement.max()),'rating_jsd_mean':float(g.rating_jsd_after.mean()),'temporal_jsd_mean':float(g.temporal_jsd_after.mean()),'length_jsd_mean':float(g.length_jsd.mean()),'median_mean_ratio':float(g.selected_source_mean_ratio.median()),'source_short20_macro':float(g.source_lt20_fraction.mean()),'selected_short20_macro':float(g.selected_lt20_fraction.mean()),'exactly30_under1500':int((g.final_retained_review_count.eq(30)&g.tokens_after_budget_enforcement.lt(1500)).sum()),**{f'band_{name}':int(g.final_retained_review_count.between(lo,hi).sum()) for name,lo,hi in [('25-30',25,30),('20-24',20,24),('15-19',15,19),('10-14',10,14),('5-9',5,9),('<5',0,4)]}})
    pd.DataFrame(summaries).to_csv(out/'budget_method_summary.csv',index=False)
    for path,digest in protected.items():assert file_hash(ROOT/path)==digest,path
    report={'identity_sha256':identity_hash,'businesses':80,'population_reviews':40407,'selection_runs':len(rows),'new_method_runs':720,'tolerance_diagnostic_choices':len(sens),'test_businesses':0,'final_manifest':False,'final_review_budget':None,'methodology_locked':False,'protected_sources_and_accepted_results_unchanged':True,'fresh_discovery_with_declared_candidate_retention':True,'elapsed_seconds':round(time.perf_counter()-started,3)}
    report['artifact_sha256']={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completion_report.json','run_progress.json']}
    write_json(out/'completion_report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
