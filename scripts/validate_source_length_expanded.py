"""Parallel independent checks from all originals; precommitted fresh32 replay."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed,wait,FIRST_COMPLETED
import argparse,bisect,hashlib,json,math,time
import numpy as np,pandas as pd,orjson,yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from review_selection_pilot import stable_seed
from run_shared_pool_review_pilot import write_json
from run_source_length_expanded import business_worker

ROOT=Path(__file__).resolve().parents[1]


def distance(a,b):
    a=[x/sum(a) for x in a];b=[x/sum(b) for x in b]
    return .5*sum(x*math.log2(x/((x+y)/2)) if x else 0 for x,y in zip(a,b))+.5*sum(y*math.log2(y/((x+y)/2)) if y else 0 for x,y in zip(a,b))


def dominates(a,b):
    return a['count']>=b['count'] and a['rating_jsd']<=b['rating_jsd']+1e-12 and a['temporal_jsd']<=b['temporal_jsd']+1e-12 and (a['count']>b['count'] or a['rating_jsd']<b['rating_jsd']-1e-12 or a['temporal_jsd']<b['temporal_jsd']-1e-12)


def validate_business(args):
    bid,pop,cfg,info=args;out=ROOT/cfg['output_directory'];filename=hashlib.sha256(bid.encode()).hexdigest()+'.json'
    payload=json.loads((out/'business_runs'/filename).read_text(encoding='utf-8'))
    pop=sorted(pop,key=lambda r:(r['date'],r['review_id']));byid={r['review_id']:r for r in pop};pos={r['review_id']:i for i,r in enumerate(pop)}
    assert len(pop)==len(byid)==int(info['english_reviews'])
    times={r['review_id']:i for i in range(5) for r in pop[i*len(pop)//5:(i+1)*len(pop)//5]}
    full_stars=Counter(int(r['stars']) for r in pop);full_times=Counter(times.values())
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));tok.no_truncation();tok.no_padding()
    body={rid:len(tok.encode(r['text'],add_special_tokens=False).ids) for rid,r in byid.items()};assert body==payload['body_lengths']
    lengths=sorted(body.values());cuts=[lengths[(len(lengths)-1)*q//4] for q in range(1,4)];lb={rid:bisect.bisect_right(cuts,value) for rid,value in body.items()};full_lengths=Counter(lb.values())
    assert cuts==payload['length_cutpoints'] and [full_lengths[i] for i in range(4)]==payload['source_length_histogram']
    metadata={};encoded={};checks=0;accepted=0
    def metrics(ids):
        key=tuple(ids)
        if key not in metadata:
            assert len(ids)==len(set(ids))<=30 and set(ids)<=set(byid) and list(ids)==sorted(ids,key=pos.__getitem__)
            s=Counter(int(byid[r]['stars']) for r in ids);t=Counter(times[r] for r in ids);l=Counter(lb[r] for r in ids)
            sig=(len(ids),tuple(s[i] for i in range(1,6)),tuple(t[i] for i in range(5)))
            metadata[key]={'signature':sig,'rating':distance([full_stars[i] for i in range(1,6)],sig[1]),'time':distance([full_times[i] for i in range(5)],sig[2]),'length':distance([full_lengths[i] for i in range(4)],[l[i] for i in range(4)]),'body_mean':float(np.mean([body[r] for r in ids])),'short20':sum(body[r]<20 for r in ids)/len(ids)}
        return metadata[key]
    def text(ids):return '\n\n'.join(f"[REV_{i}] stars={int(byid[r]['stars'])} date={byid[r]['date']}\n{byid[r]['text']}" for i,r in enumerate(ids,1))
    def tokens(ids):
        key=tuple(ids)
        if key not in encoded:encoded[key]=len(tok.encode(text(ids),add_special_tokens=True).ids)
        return encoded[key]
    def check(c,budget=None,exact=False):
        nonlocal checks
        m=metrics(c['ids']);assert c['count']==m['signature'][0]
        assert math.isclose(c['rating_jsd'],m['rating'],abs_tol=1e-12) and math.isclose(c['temporal_jsd'],m['time'],abs_tol=1e-12)
        assert c['weighted_cost']==.5*c['rating_jsd']+.5*c['temporal_jsd']
        if budget is not None:assert c['tokens']<=budget
        if exact:assert c['tokens']==tokens(c['ids'])
        checks+=1
    pool=payload['primary_candidates'];lookup={tuple(c['ids']):c for c in pool};assert len(lookup)==len(pool)
    for c in pool:check(c)
    assert set(map(tuple,payload['retained_candidate_ids']))<=set(lookup)
    assert all(tuple(ids) in lookup for _,ids in payload['incumbents'])
    old41=json.loads((ROOT/cfg['accepted_v4_1_directory']/'business_runs'/filename).read_text(encoding='utf-8'))
    old4=json.loads((ROOT/cfg['accepted_v4_directory']/'business_runs'/filename).read_text(encoding='utf-8'))
    expected={tuple(c['ids']) for c in old41['shared_candidates']}
    expected.update(tuple(c['ids']) for c in old4['candidate_audit'] if c['origin'].get('variant')!='minimum_cost_singleton_rescue')
    assert expected==set(map(tuple,payload['retained_candidate_ids']))
    tie=lambda c:stable_seed(cfg['seed'],cfg['sampling_and_tie_algorithm_version'],bid,'choice',list(c['ids']))
    runs={(r['method'],r['review_budget']):r for r in payload['runs']};assert len(runs)==18
    pairs=[];regressions=[];flagged=[];frontier_total=0
    for budget in [3500,4000,4500]:
        feasible=[c for c in pool if c['tokens']<=budget]
        if not feasible:feasible=payload['frontiers'][str(budget)]
        if not feasible:
            assert all(not row['selected_review_ids'] and row['selection_status']=='budget_infeasible' for (method,b),row in runs.items() if b==budget and method.startswith('v4_2_'));continue
        vals=np.array([[c['rating_jsd'],c['temporal_jsd'],c['count']] for c in feasible])
        a=vals[:,None,:];b=vals[None,:,:]
        no_worse=(a[:,:,0]<=b[:,:,0]+1e-12)&(a[:,:,1]<=b[:,:,1]+1e-12)&(a[:,:,2]>=b[:,:,2])
        better=(a[:,:,0]<b[:,:,0]-1e-12)|(a[:,:,1]<b[:,:,1]-1e-12)|(a[:,:,2]>b[:,:,2])
        indices=np.flatnonzero(~(no_worse&better).any(axis=0));frontier=[feasible[i] for i in indices]
        assert {tuple(c['ids']) for c in frontier}=={tuple(c['ids']) for c in payload['frontiers'][str(budget)]};frontier_total+=len(frontier)
        anchor=min(frontier,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],tie(c),c['ids']))
        compatible=[c for c in frontier if c['rating_jsd']<=anchor['rating_jsd']+.001+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+.001+1e-12]
        control=min(compatible,key=lambda c:(-c['count'],c['weighted_cost'],tie(c),c['ids']));sig=metrics(control['ids'])['signature']
        assert control['ids']==runs['v4_2_primary_control',budget]['selected_review_ids']
        group=[c for c in frontier if metrics(c['ids'])['signature']==sig];expanded=payload['equivalent_groups'][str(budget)]
        assert {tuple(c['ids']) for c in group}<={tuple(c['ids']) for c in expanded}
        assert all(metrics(c['ids'])['signature']==sig for c in expanded)
        for c in expanded:check(c,budget)
        existing=runs['v4_2_existing_length_tie',budget];final=runs['v4_2_length_balanced',budget]
        assert metrics(existing['selected_review_ids'])['length']<=min(metrics(c['ids'])['length'] for c in group)+1e-12
        assert metrics(final['selected_review_ids'])['length']<=min(metrics(c['ids'])['length'] for c in expanded)+1e-12
        for c in payload['secondary_evaluations'][str(budget)]:
            check(c,exact=True);op=c['origin']['operation'];assert int(byid[op['out']]['stars'])==int(byid[op['in']]['stars']) and times[op['out']]==times[op['in']]
            assert metrics(c['ids'])['signature']==sig
        for trace in payload['refinement_traces'][str(budget)]:
            assert trace['exact_proposals']<=4 and trace['tokens_before']==tokens(trace['before_ids'])<=budget and trace['tokens_best']==tokens(trace['best_ids'])<=budget
            assert metrics(trace['before_ids'])['signature']==metrics(trace['best_ids'])['signature']==sig
            assert math.isclose(trace['length_jsd_before'],metrics(trace['before_ids'])['length'],abs_tol=1e-12) and math.isclose(trace['length_jsd_best'],metrics(trace['best_ids'])['length'],abs_tol=1e-12)
            if trace['accepted']:assert trace['length_jsd_best']<trace['length_jsd_before']-1e-12;accepted+=1
        for (method,b),row in runs.items():
            if b!=budget:continue
            ids=row['selected_review_ids'];c={'ids':ids,'count':len(ids),'tokens':row['tokens_after_budget_enforcement'],'rating_jsd':row['rating_jsd_after'],'temporal_jsd':row['temporal_jsd_after'],'weighted_cost':.5*row['rating_jsd_after']+.5*row['temporal_jsd_after']};check(c,budget,True);m=metrics(ids)
            assert math.isclose(row['length_jsd'],m['length'],abs_tol=1e-12) and math.isclose(row['selected_body_mean'],m['body_mean'],abs_tol=1e-12)
            assert row['final_retained_review_count']==len(ids) and row['below_10_retained_flag']==(len(ids)<10) and row['below_5_retained_flag']==(len(ids)<5)
            if method.startswith('v4_2_'):
                assert m['signature']==sig
                assert row['tokens_before_budget_enforcement']==tokens(row['initial_reference_review_ids'])
                assert set(row['removed_vs_reference_ids'])==set(row['initial_reference_review_ids'])-set(ids)
                assert set(row['added_vs_reference_ids'])==set(ids)-set(row['initial_reference_review_ids'])
            if method=='v4_2_length_balanced' and len(ids)<10:
                shortest=sorted(pop,key=lambda r:(body[r['review_id']],r['date'],r['review_id']))[:10]
                probe_ids=sorted([r['review_id'] for r in shortest],key=pos.__getitem__);probe_tokens=tokens(probe_ids)
                flagged.append({'business_id':bid,'name':info['name'],'review_budget':budget,'source_count':len(pop),'selected_count':len(ids),'selected_tokens':c['tokens'],'selected_body_min':min(body[r] for r in ids),'selected_body_max':max(body[r] for r in ids),'ten_shortest_BODY_probe_tokens':probe_tokens,'ten_complete_reviews_demonstrably_fit':probe_tokens<=budget,'attribution':'search_primary_constraints_not_absolute_long_review_infeasibility' if probe_tokens<=budget else 'ten_review_infeasibility_not_proven_by_this_probe','selected_review_ids':ids})
        assert final['length_jsd']<=existing['length_jsd']+1e-12<=runs['v4_2_primary_control',budget]['length_jsd']+2e-12
        pairs.append({'business_id':bid,'name':info['name'],'split':info['split'],'volume_band':info['volume_band'],'review_budget':budget,'primary_count':control['count'],'primary_tokens':control['tokens'],'final_tokens':final['tokens_after_budget_enforcement'],'length_jsd_primary':runs['v4_2_primary_control',budget]['length_jsd'],'length_jsd_existing':existing['length_jsd'],'length_jsd_final':final['length_jsd'],'primary_histograms_preserved':True,'changed_from_control':control['ids']!=final['selected_review_ids']})
        finalc={'count':len(final['selected_review_ids']),'rating_jsd':final['rating_jsd_after'],'temporal_jsd':final['temporal_jsd_after']}
        for method in ['v3_budget_aware','v4_length_balanced','v4_1_shared_pool']:
            old=runs[method,budget];oldc={'count':len(old['selected_review_ids']),'rating_jsd':old['rating_jsd_after'],'temporal_jsd':old['temporal_jsd_after']}
            regressions.append({'business_id':bid,'review_budget':budget,'baseline':method,'baseline_dominates_final':dominates(oldc,finalc),'count_delta':finalc['count']-oldc['count'],'rating_delta':finalc['rating_jsd']-oldc['rating_jsd'],'time_delta':finalc['temporal_jsd']-oldc['temporal_jsd']})
    for preview in payload['previews']:
        assert preview['input_text']==text(preview['selected_review_ids']) and preview['led_tokens_with_special']==tokens(preview['selected_review_ids']) and not preview['final_model_input']
    for entry in payload['sensitivity']:
        frontier=payload['frontiers'][str(entry['review_budget'])]
        if not frontier:assert not entry['control_ids'] and not entry['secondary_ids'];continue
        tol=entry['primary_tolerance'];anchor=min(frontier,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],tie(c),c['ids']))
        eligible=[c for c in frontier if c['rating_jsd']<=anchor['rating_jsd']+tol+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+tol+1e-12]
        chosen=min(eligible,key=lambda c:(-c['count'],c['weighted_cost'],tie(c),c['ids']));assert chosen['ids']==entry['control_ids']
        assert metrics(entry['secondary_ids'])['signature']==metrics(chosen['ids'])['signature']
    drops=[]
    for method in sorted(set(r['method'] for r in payload['runs'])):
        low,mid,high=[runs[method,b] for b in [3500,4000,4500]]
        for a,b in [(low,mid),(mid,high)]:
            if b['final_retained_review_count']<a['final_retained_review_count']:drops.append({'business_id':bid,'name':info['name'],'method':method,'lower_budget':a['review_budget'],'higher_budget':b['review_budget'],'lower_count':a['final_retained_review_count'],'higher_count':b['final_retained_review_count'],'previous_still_fits':a['tokens_after_budget_enforcement']<=b['review_budget']})
    return {'business_id':bid,'source_reviews':len(pop),'checks':checks,'primary_candidates':len(pool),'frontiers':frontier_total,'unique_exact_blocks':len(encoded),'accepted_secondary_moves':accepted,'pairs':pairs,'regressions':regressions,'flagged':flagged,'drops':drops}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);parser.add_argument('--stream',action='store_true');parser.add_argument('--replay-workers',type=int,default=8);args=parser.parse_args();started=time.perf_counter()
    cfg=yaml.safe_load((ROOT/'configs/review_selection_source_length_expanded.yaml').read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];identity_report=json.loads((out/'scientific_identity.json').read_text(encoding='utf-8'));identity=identity_report['identity']
    cfg['_compatible_cache_identities']=[identity_report['identity_sha256']]
    if identity.get('compatible_pre_memoization_identity'):cfg['_compatible_cache_identities'].append(identity['compatible_pre_memoization_identity'])
    if not args.stream:
        completion=json.loads((out/'completion_report.json').read_text(encoding='utf-8'))
        for name,digest in completion['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    for name,digest in identity['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    for name,digest in identity['scripts'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    for name,digest in identity['legacy_config_sha256'].items():assert file_hash(ROOT/'configs'/name)==digest,name
    canonical=json.loads((ROOT/cfg['accepted_v4_2_pilot_directory']/'scientific_identity.json').read_text(encoding='utf-8'))['identity']
    for name,digest in canonical['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    for name,digest in canonical['scripts'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    sample=pd.read_csv(out/'businesses.csv',keep_default_na=False);info=sample.set_index('business_id').to_dict('index');populations={bid:[] for bid in info}
    for split in ['train','validation']:
        for line in (ROOT/cfg['english_pool_directory']/'reviews'/f'{split}.jsonl').open('rb'):
            r=orjson.loads(line);bid=r['business_id'];assert bid in populations and info[bid]['split']==split;populations[bid].append(r)
    results=[];checkpoint=out/'independent_validation_businesses';checkpoint.mkdir(exist_ok=True)
    validator_hash=file_hash(Path(__file__));pending=[]
    for bid,pop in populations.items():
        p=checkpoint/(hashlib.sha256(bid.encode()).hexdigest()+'.json')
        if p.exists():
            r=json.loads(p.read_text(encoding='utf-8'))
            assert r['identity_sha256'] in cfg['_compatible_cache_identities']
            if r['validator_sha256']!=validator_hash:
                old_validator=out/'pre_memoization_run/scripts/validate_source_length_expanded.py'
                assert r['validator_sha256']==file_hash(old_validator)
                # Independent per-business implementation is byte-identical.
                import ast
                oldtree=ast.parse(old_validator.read_text(encoding='utf-8'));newtree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
                oldfunc=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_business');newfunc=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_business')
                assert ast.dump(oldfunc)==ast.dump(newfunc)
            results.append(r)
        else:pending.append((bid,pop,cfg,info[bid]))
    print(f'Independent validation2200; pending {len(pending)}.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={};queue=list(pending)
        while queue or futures:
            slots=args.workers-len(futures)
            ready=[a for a in queue if (out/'business_runs'/(hashlib.sha256(a[0].encode()).hexdigest()+'.json')).exists()][:slots]
            for task in ready:queue.remove(task);futures[pool.submit(validate_business,task)]=task[0]
            if not futures:
                if queue and not args.stream:raise RuntimeError('Missing completed business output')
                time.sleep(5);continue
            finished,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
            for future in finished:
                futures.pop(future);r=future.result();r.update(validator_sha256=validator_hash,identity_sha256=identity_report['identity_sha256']);write_json(checkpoint/(hashlib.sha256(r['business_id'].encode()).hexdigest()+'.json'),r);results.append(r)
                if len(results)%50==0:print(f'Independent checks {len(results)}/2200.',flush=True)
    assert len(results)==2200 and sum(r['source_reviews'] for r in results)==575749
    scope=json.loads((out/'fresh_replay_scope.json').read_text(encoding='utf-8'));assert len(scope['business_ids'])==32
    print('Fresh reverse-source replay of precommitted32 new businesses.',flush=True);done=0
    while not (out/'completion_report.json').exists():time.sleep(5)
    completion=json.loads((out/'completion_report.json').read_text(encoding='utf-8'))
    for name,digest in completion['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    with ProcessPoolExecutor(max_workers=args.replay_workers) as pool:
        futures={pool.submit(business_worker,(bid,list(reversed(populations[bid])),cfg,info[bid],identity_report['identity_sha256'])):bid for bid in scope['business_ids']}
        for future in as_completed(futures):
            bid=futures[future];fresh=future.result();cached=json.loads((out/'business_runs'/(hashlib.sha256(bid.encode()).hexdigest()+'.json')).read_text(encoding='utf-8'))
            fresh.pop('identity_sha256');cached.pop('identity_sha256')
            assert orjson.dumps(fresh,option=orjson.OPT_SORT_KEYS|orjson.OPT_NON_STR_KEYS)==orjson.dumps(cached,option=orjson.OPT_SORT_KEYS|orjson.OPT_NON_STR_KEYS),bid;done+=1
            print(f'Fresh replay {done}/32.',flush=True)
    for field,name in [('pairs','paired_primary_secondary.csv'),('regressions','baseline_regressions.csv'),('drops','cross_budget_count_drops.csv')]:pd.DataFrame([v for r in results for v in r[field]]).to_csv(out/name,index=False)
    flagged=[v for r in results for v in r['flagged']];write_json(out/'below_ten_attribution.json',flagged)
    for name,digest in identity['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    report={'validation_passed':True,'businesses':2200,'source_originals_joined_BODY_encoded':575749,'selection_runs_verified':39600,'new_method_runs':19800,'independent_checks':sum(r['checks'] for r in results),'primary_candidates_verified':sum(r['primary_candidates'] for r in results),'frontier_candidates_verified':sum(r['frontiers'] for r in results),'unique_blocks_fresh_encoded':sum(r['unique_exact_blocks'] for r in results),'accepted_secondary_moves_verified':sum(r['accepted_secondary_moves'] for r in results),'primary_histogram_pairs_preserved':sum(len(r['pairs']) for r in results),'tolerance_diagnostics_verified':19800,'accepted80_payload_equal':80,'additional_fresh_reverse_source_payloads_equal':32,'all2200_fresh_replay_claimed':False,'below_ten_final_cases':len(flagged),'source_and_accepted_evidence_unchanged':True,'final_manifest':False,'final_review_budget':None,'methodology_locked':False,'validator_sha256':validator_hash,'elapsed_seconds':round(time.perf_counter()-started,3)}
    write_json(out/'validation_report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
