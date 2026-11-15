"""Independent entropy/token/source checks plus fresh deterministic C2 search rerun."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import math
import sqlite3
import time
import orjson
import pandas as pd
import yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from run_budget_review_pilot import run_business

ROOT=Path(__file__).resolve().parents[1]


def fresh_business(arguments):
    bid,population,cfg,info=arguments
    tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    tokenizer.no_truncation();tokenizer.no_padding()
    runs=run_business(bid,population,cfg,tokenizer,info)['runs']
    return [{k:r[k] for k in ['method','review_budget','selected_review_ids','tokens_after_budget_enforcement']} for r in runs]


def lines(path):
    with path.open('rb') as stream:return [orjson.loads(line) for line in stream if line.strip()]


def divergence(a,b):
    a=[v/sum(a) for v in a];b=[v/sum(b) for v in b]
    entropy=lambda p:-sum(v*math.log2(v) for v in p if v)
    return max(0.,entropy([(v+w)/2 for v,w in zip(a,b)])-(entropy(a)+entropy(b))/2)


def main():
    started=time.perf_counter()
    cfg=yaml.safe_load((ROOT/'configs/review_selection_budget_pilot.yaml').read_text(encoding='utf-8'))
    out=ROOT/cfg['output_directory']; source=ROOT/cfg['english_pool_directory']
    report=json.loads((out/'completion_report.json').read_text())
    identity=json.loads((out/'scientific_identity.json').read_text())['identity']
    for name,digest in report['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    for name,digest in identity['sources'].items():assert file_hash(source/name)==digest,name
    for name,digest in identity['scripts'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    assert file_hash(ROOT/'configs/review_selection_budget_pilot.yaml')==identity['config_sha256']
    sample=pd.read_csv(out/'pilot_businesses.csv',keep_default_na=False)
    assert len(sample)==56 and sample['business_id'].is_unique and set(sample['split'])<={'train','validation'}
    old=pd.read_csv(ROOT/cfg['original_pilot_directory']/'pilot_businesses.csv')
    assert set(sample.loc[sample['sample_group'].eq('original_48'),'business_id'])==set(old['business_id'])
    assert sample['sample_group'].eq('additional_long_review_8').sum()==8
    c1=ROOT/cfg['c1_directory']
    full_stats=pd.read_csv(c1/'business_review_statistics.csv')
    allowed=full_stats.loc[full_stats['split'].isin(['train','validation'])].copy()
    lengths=defaultdict(list)
    db=sqlite3.connect(f'file:{(c1/"review_metadata.sqlite").resolve().as_posix()}?mode=ro',uri=True)
    for bid,length in db.execute("SELECT business_id,led_tokens_without_special FROM reviews WHERE language='en' AND split IN ('train','validation')"):
        lengths[bid].append(length)
    db.close()
    allowed['top8']=allowed['business_id'].map(lambda bid:sum(sorted(lengths[bid],reverse=True)[:8]))
    extras=allowed.loc[allowed['english_reviews'].le(100)&~allowed['business_id'].isin(old['business_id'])].sort_values(['top8','business_id'],ascending=[False,True]).head(8)
    assert set(extras['business_id'])==set(sample.loc[sample['sample_group'].eq('additional_long_review_8'),'business_id'])
    membership=sample.set_index('business_id')['split'].to_dict()
    populations=defaultdict(dict);seen=set()
    for row in lines(out/'pilot_population.jsonl'):
        r=row['review'];rid=r['review_id'];bid=r['business_id']
        assert rid not in seen and membership[bid]==row['split'];seen.add(rid);populations[bid][rid]=r
    source_seen=set()
    for split in cfg['allowed_splits']:
        with (source/'reviews'/f'{split}.jsonl').open('rb') as stream:
            for line in stream:
                r=orjson.loads(line);bid=r['business_id'];rid=r['review_id']
                if bid in membership:
                    assert membership[bid]==split and populations[bid][rid]==r and rid not in source_seen
                    source_seen.add(rid)
    assert source_seen==seen and len(seen)==report['population_reviews']
    tokenizer_path=ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'
    assert file_hash(tokenizer_path)==identity['tokenizer_manifest']['sha256']['tokenizer.json']
    tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer.no_truncation();tokenizer.no_padding()
    assert cfg['format']['review_template']=='[REV_{index}] stars={stars} date={date}\n{text}' and cfg['format']['separator']=='\n\n'
    runs=lines(out/'selection_runs.jsonl')
    previews={(r['business_id'],r['method'],r['review_budget']):r for r in lines(out/'model_input_previews.jsonl')}
    time_members={};targets={}
    for bid,items in populations.items():
        full=sorted(items.values(),key=lambda r:(r['date'],r['review_id']))
        times={}; counts=[]
        for i in range(5):
            group=full[i*len(full)//5:(i+1)*len(full)//5];counts.append(len(group));times.update({r['review_id']:i for r in group})
        time_members[bid]=times;targets[bid]=(Counter(int(r['stars']) for r in full),counts)
    def ordered_ids(bid,ids):return sorted(ids,key=lambda rid:(populations[bid][rid]['date'],rid))
    def exact(bid,ids):
        text='\n\n'.join(f'[REV_{i}] stars={int(populations[bid][rid]["stars"])} date={populations[bid][rid]["date"]}\n{populations[bid][rid]["text"]}' for i,rid in enumerate(ids,1))
        return text,len(tokenizer.encode(text,add_special_tokens=True).ids)
    def metrics(bid,ids):
        star_target,time_target=targets[bid]
        s=Counter(int(populations[bid][rid]['stars']) for rid in ids);t=Counter(time_members[bid][rid] for rid in ids)
        return divergence([star_target[i] for i in range(1,6)],[s[i] for i in range(1,6)]),divergence(time_target,[t[i] for i in range(5)])
    seen_runs=set(); move_count=0
    for run in runs:
        bid=run['business_id'];budget=run['review_budget'];key=(bid,run['method'],budget)
        assert key not in seen_runs;seen_runs.add(key)
        ids=run['selected_review_ids'];initial=run['initial_reference_review_ids'];n=len(populations[bid])
        assert run['initial_eligible_review_count']==run['candidate_pool_count']==n
        assert len(initial)==run['initial_candidate_count']==min(n,30)
        assert ids==ordered_ids(bid,ids) and len(ids)==len(set(ids))==run['final_retained_review_count']<=30
        assert run['below_10_retained_flag']==(len(ids)<10) and run['below_5_retained_flag']==(len(ids)<5)
        assert exact(bid,initial)[1]==run['tokens_before_budget_enforcement']
        rb,tb=metrics(bid,initial)
        assert abs(rb-run['rating_jsd_before'])<1e-12 and abs(tb-run['temporal_jsd_before'])<1e-12
        assert set(run['removed_vs_reference_ids'])==set(initial)-set(ids)
        assert set(run['added_vs_reference_ids'])==set(ids)-set(initial)
        if ids:
            text,tokens=exact(bid,ids)
            assert 1<=len(ids)<=min(n,30) and tokens<=budget and tokens==run['tokens_after_budget_enforcement']
            assert text==previews[key]['input_text'] and ids==previews[key]['selected_review_ids'] and not previews[key]['final_model_input']
            ra,ta=metrics(bid,ids)
            assert abs(ra-run['rating_jsd_after'])<1e-12 and abs(ta-run['temporal_jsd_after'])<1e-12
            assert run['selection_status']=='feasible'
        else:
            assert run['selection_status']=='budget_infeasible' and key not in previews and run['tokens_after_budget_enforcement'] is None
        assert run['semantic_centroid_similarity'] is None and run['final_review_budget'] is None and not run['methodology_locked']
        if run['method']=='budget_aware':
            path=run['search_provenance']['accepted_local_path']
            for index,step in enumerate(path):
                before=step['before_ids'];after=step['after_ids'];op=step['operation']
                assert set(after)==(set(before)-set(op['out']))|set(op['in'])
                assert exact(bid,before)[1]==step['tokens_before']
                assert exact(bid,after)[1]==step['tokens_after']<=budget
                if index: assert path[index-1]['after_ids']==before
                r,t=metrics(bid,after)
                assert abs(r-step['rating_jsd_after'])<1e-12 and abs(t-step['temporal_jsd_after'])<1e-12
                move_count+=1
            if path:assert path[-1]['after_ids']==ids
    assert len(runs)==56*3*3==report['selection_runs']
    frame=pd.read_csv(out/'selection_diagnostics.csv')
    assert len(frame)==len(runs)
    for row,run in zip(frame.to_dict('records'),runs):
        for key in ['business_id','method','review_budget','final_retained_review_count','below_10_retained_flag','below_5_retained_flag']:
            assert row[key]==run[key],key
    summary=pd.read_csv(out/'budget_method_summary.csv')
    for row in summary.to_dict('records'):
        group=frame.loc[frame['method'].eq(row['method'])&frame['review_budget'].eq(row['review_budget'])]
        assert row['businesses']==len(group)==56
        assert row['below_10']==int(group['below_10_retained_flag'].sum()) and row['below_5']==int(group['below_5_retained_flag'].sum())
        assert row['infeasible']==int(group['selection_status'].eq('budget_infeasible').sum())
        for band in ['25-30','20-24','15-19','10-14','5-9','<5']:assert row[f'band_{band}']==int(group['retained_count_band'].eq(band).sum())
        assert sum(row[f'band_{band}'] for band in ['25-30','20-24','15-19','10-14','5-9','<5'])==56
    sensitivity=json.loads((out/'tolerance_sensitivity.json').read_text())
    info=sample.set_index('business_id').to_dict('index'); fresh_matches=0;frontier_entries=0;fresh_requests=[]
    for index,(bid,pop) in enumerate(populations.items(),1):
        cache_path=out/'business_runs'/f'{__import__("hashlib").sha256(bid.encode()).hexdigest()}.json'
        payload=json.loads(cache_path.read_text(encoding='utf-8'))
        for budget_key,candidates in payload['frontiers'].items():
            budget=int(budget_key)
            for c in candidates:
                assert c['ids']==ordered_ids(bid,c['ids']) and exact(bid,c['ids'])[1]==c['tokens']<=budget
                r,t=metrics(bid,c['ids']);assert abs(r-c['rating_jsd'])<1e-12 and abs(t-c['temporal_jsd'])<1e-12
                assert abs(cfg['search']['rating_weight']*r+cfg['search']['temporal_weight']*t-c['weighted_cost'])<1e-12
                assert c['count']==len(c['ids']);frontier_entries+=1
                assert not any(a['rating_jsd']<=c['rating_jsd']+1e-12 and a['temporal_jsd']<=c['temporal_jsd']+1e-12 and a['count']>=c['count'] and (a['rating_jsd']<c['rating_jsd']-1e-12 or a['temporal_jsd']<c['temporal_jsd']-1e-12 or a['count']>c['count']) for a in candidates)
            if candidates:
                anchor=min(candidates,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],c['tokens'],tuple(c['ids'])))
                for item in [s for s in sensitivity if s['business_id']==bid and s['review_budget']==budget]:
                    tol=item['tolerance_per_metric']
                    eligible=[c for c in candidates if c['rating_jsd']<=anchor['rating_jsd']+tol+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+tol+1e-12]
                    winner=min(eligible,key=lambda c:(-c['count'],c['weighted_cost'],c['tokens'],tuple(c['ids'])))
                    assert item['selected_review_ids']==winner['ids']
                    if abs(tol-cfg['search']['comparable_rating_jsd_tolerance'])<1e-12:
                        selected=next(row for row in runs if row['business_id']==bid and row['review_budget']==budget and row['method']=='budget_aware')
                        assert selected['selected_review_ids']==winner['ids']
        fresh_requests.append((bid,list(reversed(list(pop.values()))),cfg,info[bid]))
    # Independent businesses can be recomputed concurrently. Each worker loads
    # a fresh tokenizer and runs the search without completed-run caches.
    with ProcessPoolExecutor(max_workers=4) as executor:
        for index,(arguments,fresh) in enumerate(zip(fresh_requests,executor.map(fresh_business,fresh_requests)),1):
            bid=arguments[0]
            cache_path=out/'business_runs'/f'{__import__("hashlib").sha256(bid.encode()).hexdigest()}.json'
            original=json.loads(cache_path.read_text(encoding='utf-8'))['runs']
            assert len(original)==len(fresh)==9
            for a,b in zip(original,fresh):
                assert a['method']==b['method'] and a['review_budget']==b['review_budget'] and a['selected_review_ids']==b['selected_review_ids']
                assert a['tokens_after_budget_enforcement']==b['tokens_after_budget_enforcement']
                fresh_matches+=1
            print(f'Fresh deterministic verification {index}/56',flush=True)
    for row in lines(out/'long_review_stress.jsonl'):
        assert row['count']==len(row['selected_review_ids']) in (7,8) and exact(row['business_id'],row['selected_review_ids'])[1]==row['exact_tokens']
    example_ids=json.loads((out/'detail_example_business_ids.json').read_text())
    assert len(example_ids)==len(set(example_ids))==10 and set(example_ids)<=set(membership)
    result={'validation_passed':True,'businesses':56,'source_reviews_verified':len(seen),'selection_runs_verified':len(runs),
            'fresh_deterministic_runs_verified':fresh_matches,'frontier_candidates_verified':frontier_entries,
            'accepted_local_moves_verified':move_count,'exact_tokens_jsd_flags_histograms_verified':True,
            'test_businesses':0,'final_review_budget':None,'semantic_status':'not_computed_model_TBD','methodology_locked':False,
            'elapsed_seconds':round(time.perf_counter()-started,3)}
    (out/'validation_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
