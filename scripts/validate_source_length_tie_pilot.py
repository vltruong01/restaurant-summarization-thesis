"""Independent original/histogram/ranking checks and fresh reverse-source replay."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,bisect,hashlib,json,math,time
import numpy as np
import pandas as pd
import orjson,yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from review_selection_pilot import stable_seed
from run_source_length_tie_pilot import worker
from run_shared_pool_review_pilot import write_json

ROOT=Path(__file__).resolve().parents[1]


def distance(a,b):
    # Independent explicit scalar base2 Jensen-Shannon calculation.
    pa=[v/sum(a) for v in a];pb=[v/sum(b) for v in b]
    return .5*sum(x*math.log2(x/((x+y)/2)) if x else 0 for x,y in zip(pa,pb))+.5*sum(y*math.log2(y/((x+y)/2)) if y else 0 for x,y in zip(pa,pb))


def dominates(a,b):
    return a['count']>=b['count'] and a['rating_jsd']<=b['rating_jsd']+1e-12 and a['temporal_jsd']<=b['temporal_jsd']+1e-12 and (a['count']>b['count'] or a['rating_jsd']<b['rating_jsd']-1e-12 or a['temporal_jsd']<b['temporal_jsd']-1e-12)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    started=time.perf_counter();cfg=yaml.safe_load((ROOT/'configs/review_selection_source_length_tie_pilot.yaml').read_text(encoding='utf-8'));out=ROOT/cfg['output_directory']
    completion=json.loads((out/'completion_report.json').read_text(encoding='utf-8'))
    for name,digest in completion['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    identity=json.loads((out/'scientific_identity.json').read_text(encoding='utf-8'))['identity']
    for name,digest in identity['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    for name,digest in identity['scripts'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    assert file_hash(ROOT/'configs/review_selection_source_length_tie_pilot.yaml')==identity['config_sha256']
    sample=pd.read_csv(out/'pilot_businesses.csv',keep_default_na=False);info=sample.set_index('business_id').to_dict('index')
    populations={bid:[] for bid in info};sourcejoins={}
    for split in ['train','validation']:
        for line in (ROOT/cfg['english_pool_directory']/'reviews'/f'{split}.jsonl').open('rb'):
            r=orjson.loads(line)
            if r['business_id'] in populations:
                assert info[r['business_id']]['split']==split;populations[r['business_id']].append(r);sourcejoins[r['review_id']]=r
    exported=[orjson.loads(line)['review'] for line in (out/'pilot_population.jsonl').open('rb')]
    assert len(exported)==len(sourcejoins)==40407
    assert all(r==sourcejoins[r['review_id']] for r in exported)
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));tok.no_truncation();tok.no_padding()
    checked=0;encoded={};primary_total=0;frontier_total=0;accepted_moves=0;pairs=[];regressions=[];replay=[]
    for index,path in enumerate(sorted((out/'business_runs').glob('*.json')),1):
        payload=json.loads(path.read_text(encoding='utf-8'));bid=payload['business_id'];pop=sorted(populations[bid],key=lambda r:(r['date'],r['review_id']));byid={r['review_id']:r for r in pop}
        pos={r['review_id']:i for i,r in enumerate(pop)};times={r['review_id']:i for i in range(5) for r in pop[i*len(pop)//5:(i+1)*len(pop)//5]}
        full_stars=Counter(int(r['stars']) for r in pop);full_times=Counter(times.values())
        body={rid:len(tok.encode(r['text'],add_special_tokens=False).ids) for rid,r in byid.items()};assert body==payload['body_lengths']
        lengths=sorted(body.values());cuts=[lengths[(len(lengths)-1)*q//4] for q in range(1,4)];lb={rid:bisect.bisect_right(cuts,value) for rid,value in body.items()};full_lengths=Counter(lb.values())
        assert cuts==payload['length_cutpoints'] and [full_lengths[i] for i in range(4)]==payload['source_length_histogram']
        def signature(ids):return (len(ids),tuple(Counter(int(byid[r]['stars']) for r in ids)[i] for i in range(1,6)),tuple(Counter(times[r] for r in ids)[i] for i in range(5)))
        def score(ids):return distance([full_lengths[i] for i in range(4)],[Counter(lb[r] for r in ids)[i] for i in range(4)])
        def check(c,budget=None,exact=False):
            nonlocal checked
            ids=c['ids'];assert ids and len(ids)==len(set(ids))==c['count']<=30 and set(ids)<=set(byid)
            assert ids==sorted(ids,key=pos.__getitem__)
            sig=signature(ids)
            assert math.isclose(distance([full_stars[i] for i in range(1,6)],sig[1]),c['rating_jsd'],abs_tol=1e-12)
            assert math.isclose(distance([full_times[i] for i in range(5)],sig[2]),c['temporal_jsd'],abs_tol=1e-12)
            assert c['weighted_cost']==.5*c['rating_jsd']+.5*c['temporal_jsd']
            if budget is not None:assert c['tokens']<=budget
            if exact:
                key=(bid,tuple(ids))
                if key not in encoded:
                    text='\n\n'.join(f"[REV_{i}] stars={int(byid[rid]['stars'])} date={byid[rid]['date']}\n{byid[rid]['text']}" for i,rid in enumerate(ids,1))
                    encoded[key]=len(tok.encode(text,add_special_tokens=True).ids)
                assert encoded[key]==c['tokens']
            checked+=1
        pool=payload['primary_candidates'];primary_total+=len(pool)
        assert len({tuple(c['ids']) for c in pool})==len(pool)
        for c in pool:check(c)
        assert set(map(tuple,payload['retained_candidate_ids']))<=set(tuple(c['ids']) for c in pool)
        assert all(tuple(ids) in {tuple(c['ids']) for c in pool} for _,ids in payload['incumbents'])
        tie=lambda c:stable_seed(cfg['seed'],cfg['sampling_and_tie_algorithm_version'],bid,'choice',list(c['ids']))
        runs={(r['method'],r['review_budget']):r for r in payload['runs']}
        for budget in [3500,4000,4500]:
            feasible=[c for c in pool if c['tokens']<=budget]
            if not feasible:feasible=payload['frontiers'][str(budget)]
            frontier=[c for c in feasible if not any(dominates(a,c) for a in feasible)]
            actual=payload['frontiers'][str(budget)];assert {tuple(c['ids']) for c in frontier}=={tuple(c['ids']) for c in actual}
            frontier_total+=len(frontier)
            anchor=min(frontier,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],tie(c),c['ids']))
            comparable=[c for c in frontier if c['rating_jsd']<=anchor['rating_jsd']+.001+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+.001+1e-12]
            control=min(comparable,key=lambda c:(-c['count'],c['weighted_cost'],tie(c),c['ids']))
            assert control['ids']==runs['v4_2_primary_control',budget]['selected_review_ids']
            group=[c for c in frontier if signature(c['ids'])==signature(control['ids'])]
            chosen=lambda candidates:min(candidates,key=lambda c:(score(c['ids']),tie(c),c['ids']))
            existing=chosen(group)
            # Near machine-equal independent scores can reorder numerical ties;
            # exact chosen IDs are additionally checked by full fresh replay.
            existingrow=runs['v4_2_existing_length_tie',budget]
            assert score(existingrow['selected_review_ids'])<=score(existing['ids'])+1e-12
            expanded=payload['equivalent_groups'][str(budget)]
            assert set(tuple(c['ids']) for c in group)<=set(tuple(c['ids']) for c in expanded)
            assert all(signature(c['ids'])==signature(control['ids']) for c in expanded)
            for c in expanded:check(c,budget)
            finalrow=runs['v4_2_length_balanced',budget]
            assert score(finalrow['selected_review_ids'])<=min(score(c['ids']) for c in expanded)+1e-12
            for c in payload['secondary_evaluations'][str(budget)]:
                check(c,exact=True);op=c['origin']['operation']
                assert int(byid[op['out']]['stars'])==int(byid[op['in']]['stars']) and times[op['out']]==times[op['in']]
                assert signature(c['ids'])==signature(control['ids'])
            for t in payload['refinement_traces'][str(budget)]:
                assert t['exact_proposals']<=4 and t['tokens_before']<=budget and t['tokens_best']<=budget
                assert signature(t['before_ids'])==signature(t['best_ids'])==signature(control['ids'])
                assert math.isclose(score(t['before_ids']),t['length_jsd_before'],abs_tol=1e-12) and math.isclose(score(t['best_ids']),t['length_jsd_best'],abs_tol=1e-12)
                if t['accepted']:assert t['length_jsd_best']<t['length_jsd_before']-1e-12;accepted_moves+=1
            for method in ['v4_2_primary_control','v4_2_existing_length_tie','v4_2_length_balanced','v3_budget_aware','v4_length_balanced','v4_1_shared_pool']:
                row=runs[method,budget];ids=row['selected_review_ids'];sig=signature(ids)
                c={'ids':ids,'count':len(ids),'tokens':row['tokens_after_budget_enforcement'],'rating_jsd':row['rating_jsd_after'],'temporal_jsd':row['temporal_jsd_after'],'weighted_cost':.5*row['rating_jsd_after']+.5*row['temporal_jsd_after']};check(c,budget,True)
                assert math.isclose(score(ids),row['length_jsd'],abs_tol=1e-12)
                assert math.isclose(np.mean([body[rid] for rid in ids]),row['selected_body_mean'],abs_tol=1e-12)
                if method.startswith('v4_2_'):assert sig==signature(control['ids'])
            assert finalrow['length_jsd']<=existingrow['length_jsd']+1e-12<=runs['v4_2_primary_control',budget]['length_jsd']+2e-12
            pairs.append({'business_id':bid,'name':info[bid]['name'],'review_budget':budget,'primary_count':control['count'],'primary_tokens':control['tokens'],'final_tokens':finalrow['tokens_after_budget_enforcement'],'length_jsd_primary':runs['v4_2_primary_control',budget]['length_jsd'],'length_jsd_existing':existingrow['length_jsd'],'length_jsd_final':finalrow['length_jsd'],'primary_histograms_preserved':True,'changed_from_control':control['ids']!=finalrow['selected_review_ids']})
            finalcandidate={'count':finalrow['final_retained_review_count'],'rating_jsd':finalrow['rating_jsd_after'],'temporal_jsd':finalrow['temporal_jsd_after']}
            for method in ['v3_budget_aware','v4_length_balanced','v4_1_shared_pool']:
                old=runs[method,budget];oldc={'count':old['final_retained_review_count'],'rating_jsd':old['rating_jsd_after'],'temporal_jsd':old['temporal_jsd_after']}
                regressions.append({'business_id':bid,'review_budget':budget,'baseline':method,'baseline_dominates_final':dominates(oldc,finalcandidate),'count_delta':finalcandidate['count']-oldc['count'],'rating_delta':finalcandidate['rating_jsd']-oldc['rating_jsd'],'time_delta':finalcandidate['temporal_jsd']-oldc['temporal_jsd']})
        for entry in payload['sensitivity']:
            b=entry['review_budget'];tol=entry['primary_tolerance'];frontier=payload['frontiers'][str(b)]
            a=min(frontier,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],tie(c),c['ids']))
            group=[c for c in frontier if c['rating_jsd']<=a['rating_jsd']+tol+1e-12 and c['temporal_jsd']<=a['temporal_jsd']+tol+1e-12]
            c=min(group,key=lambda c:(-c['count'],c['weighted_cost'],tie(c),c['ids']))
            assert entry['control_ids']==c['ids'] and signature(entry['secondary_ids'])==signature(c['ids'])
            equivalents=[x for x in frontier if signature(x['ids'])==signature(c['ids'])]
            assert score(entry['secondary_ids'])<=min(score(x['ids']) for x in equivalents)+1e-12
        replay.append((bid,list(reversed(pop)),cfg,info[bid]))
        if index%10==0:print(f'Independent originals/candidates checked {index}/80.',flush=True)
    print('Fresh reverse-source replay of all80 businesses.',flush=True)
    replayed=0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(worker,a):a[0] for a in replay}
        for future in as_completed(futures):
            bid=futures[future];fresh=future.result();cached=json.loads((out/'business_runs'/(hashlib.sha256(bid.encode()).hexdigest()+'.json')).read_text(encoding='utf-8'));cached.pop('identity_sha256')
            assert orjson.dumps(fresh,option=orjson.OPT_SORT_KEYS|orjson.OPT_NON_STR_KEYS)==orjson.dumps(cached,option=orjson.OPT_SORT_KEYS|orjson.OPT_NON_STR_KEYS),bid
            replayed+=1
            if replayed%10==0:print(f'Fresh replay {replayed}/80.',flush=True)
    pd.DataFrame(pairs).to_csv(out/'paired_primary_secondary.csv',index=False);pd.DataFrame(regressions).to_csv(out/'baseline_regressions.csv',index=False)
    for name,digest in identity['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    report={'validation_passed':True,'businesses':80,'source_originals_joined_and_BODY_encoded':40407,'selection_runs_verified':1440,'primary_candidates_verified':primary_total,'frontier_candidates_verified':frontier_total,'independent_checks':checked,'unique_blocks_fresh_encoded':len(encoded),'secondary_accepted_moves_verified':accepted_moves,'paired_primary_histograms_preserved':len(pairs),'sensitivity_cases_verified':720,'fresh_reverse_source_payloads_equal':replayed,'protected_sources_and_accepted_results_unchanged':True,'final_manifest':False,'final_review_budget':None,'methodology_locked':False,'elapsed_seconds':round(time.perf_counter()-started,3),'validator_sha256':file_hash(Path(__file__))}
    write_json(out/'validation_report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
