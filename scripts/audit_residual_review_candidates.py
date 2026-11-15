"""Cached residual coverage/tie audit; no selector or model inputs are changed."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib,html,json,math,os,time
import numpy as np,pandas as pd,orjson,yaml
from tokenizers import Tokenizer
from audit_review_selection_tradeoffs import independent_choice,candidate,dominates,jsd
from budget_aware_review_selection_v4_1 import discovery_config
from review_selection_pilot import stable_seed

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    started=time.perf_counter();settings=yaml.safe_load((ROOT/'configs/review_selection_residual_audit.yaml').read_text(encoding='utf-8'))
    cfg=discovery_config(yaml.safe_load((ROOT/settings['selector_config']).read_text(encoding='utf-8')))
    source=ROOT/settings['accepted_v4_1_directory'];older=ROOT/settings['accepted_v4_directory'];out=ROOT/settings['output_directory'];out.mkdir(parents=True,exist_ok=True)
    def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    protected={}
    for directory in [source,older]:
        report=json.loads((directory/'completion_report.json').read_text(encoding='utf-8'))
        assert json.loads((directory/'validation_report.json').read_text(encoding='utf-8'))['validation_passed']
        for rel,sha in report['artifact_sha256'].items():assert digest(directory/rel)==sha,rel
        for name in ['completion_report.json','validation_report.json','scientific_identity.json']:protected[(directory/name).relative_to(ROOT).as_posix()]=digest(directory/name)
    identity=json.loads((source/'scientific_identity.json').read_text(encoding='utf-8'))['identity']
    for name,sha in identity['scripts'].items():assert digest(ROOT/'scripts'/name)==sha;protected['scripts/'+name]=sha
    protected[settings['selector_config']]=identity['config_sha256'];assert digest(ROOT/settings['selector_config'])==identity['config_sha256']
    for name,sha in identity['sources'].items():
        rel=(Path('data/processed/yelp_2300_english_v1')/name).as_posix();assert digest(ROOT/rel)==sha;protected[rel]=sha
    populations={};splits={}
    for line in (source/'pilot_population.jsonl').open('rb'):
        row=orjson.loads(line);r=row['review'];populations.setdefault(r['business_id'],[]).append(r);splits[r['business_id']]=row['split']
    assert len(populations)==80 and sum(map(len,populations.values()))==40407 and set(splits.values())<={'train','validation'}
    tokpath=ROOT/'assets/tokenizers/led-base-16384/tokenizer.json';assert digest(tokpath)==identity['tokenizer_sha256']
    os.environ['TOKENIZERS_PARALLELISM']='false';tok=Tokenizer.from_file(str(tokpath));tok.no_truncation();tok.no_padding()
    allrows=[];tie_rows=[];residual=[];counterfactual=[];readrows=[];scopes=[];blocks={};checks=0
    diagnostics=pd.read_csv(source/'selection_diagnostics.csv')
    low=diagnostics.query("method=='v4_1_shared_pool' and review_budget==3500 and final_retained_review_count==30").sort_values(['tokens_after_budget_enforcement','business_id']).head(settings['lowest_exact30_cases']).business_id.tolist()
    residual_ids=['9tXwyr5Yqu4WnuCLOUPtrQ','TIGOlkeBefuEC4TxfueJEA']
    page=['<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1150px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{padding:8px;border:1px solid #ddd}</style><h1>Residual coverage / same-histogram diagnostic originals</h1><p>Counterfactuals only. No final model inputs, quality classifier or formal human evaluation.</p>']
    for case_index,(bid,pop) in enumerate(populations.items()):
        if case_index%10==0:print(f'Independent cached audit {case_index}/80',flush=True)
        pop=sorted(pop,key=lambda r:(r['date'],r['review_id']));byid={r['review_id']:r for r in pop};positions={r['review_id']:i for i,r in enumerate(pop)}
        body={rid:len(tok.encode(r['text'],add_special_tokens=False).ids) for rid,r in byid.items()};source_mean=np.mean(list(body.values()));short_source=np.mean([v<20 for v in body.values()])
        bins={r['review_id']:i for i in range(5) for r in pop[i*len(pop)//5:(i+1)*len(pop)//5]}
        fullstars=Counter(int(r['stars']) for r in pop);fulltimes=Counter(bins.values())
        def histogram(ids):return tuple(Counter(int(byid[rid]['stars']) for rid in ids)[i] for i in range(1,6))+tuple(Counter(bins[rid] for rid in ids)[i] for i in range(5))
        def score(c,budget):
            nonlocal checks
            ids=list(c['ids']);assert len(set(ids))==c['count']<=30 and set(ids)<=set(byid) and ids==sorted(ids,key=positions.__getitem__)
            key=(bid,tuple(ids))
            if key not in blocks:
                text='\n\n'.join(f"[REV_{i}] stars={int(byid[rid]['stars'])} date={byid[rid]['date']}\n{byid[rid]['text']}" for i,rid in enumerate(ids,1))
                hist=histogram(ids);rating=jsd([fullstars[i] for i in range(1,6)],hist[:5]);temporal=jsd([fulltimes[i] for i in range(5)],hist[5:])
                span=(datetime.fromisoformat(byid[ids[-1]]['date'])-datetime.fromisoformat(byid[ids[0]]['date'])).total_seconds()
                fullspan=(datetime.fromisoformat(pop[-1]['date'])-datetime.fromisoformat(pop[0]['date'])).total_seconds()
                blocks[key]={'tokens':len(tok.encode(text,add_special_tokens=True).ids),'rating_jsd':rating,'temporal_jsd':temporal,'histogram':hist,
                             'body_mean':float(np.mean([body[rid] for rid in ids])),'body_ratio':float(np.mean([body[rid] for rid in ids])/source_mean),
                             'short20_fraction':float(np.mean([body[rid]<20 for rid in ids])),'date_span_coverage':span/fullspan if fullspan else 1.}
            s=blocks[key];assert s['tokens']==c['tokens']<=budget
            assert math.isclose(s['rating_jsd'],c['rating_jsd'],abs_tol=1e-12) and math.isclose(s['temporal_jsd'],c['temporal_jsd'],abs_tol=1e-12)
            checks+=1;return s
        cache=hashlib.sha256(bid.encode()).hexdigest()+'.json'
        current=json.loads((source/'business_runs'/cache).read_text(encoding='utf-8'));old=json.loads((older/'business_runs'/cache).read_text(encoding='utf-8'))
        runs={(r['method'],r['review_budget']):r for r in current['runs']}
        shared=current['shared_candidates'];oldchoices=[candidate(runs['v4_length_balanced',b]) for b in [3500,4000,4500]]
        oldfronts=[c for f in old['frontiers'].values() for c in f]
        oldexact=[c for c in old['candidate_audit'] if c['origin'].get('variant')!='minimum_cost_singleton_rescue']
        for budget in [3500,4000,4500]:
            original=candidate(runs['v4_1_shared_pool',budget]);actual,anchor=independent_choice(current['frontiers'][str(budget)],cfg,bid);assert actual['ids']==original['ids'];oldscore=score(original,budget)
            for label,extras in [('plus_v4_incumbents',oldchoices),('plus_v4_frontiers',oldfronts),('plus_all_ordinary_v4_evaluations',oldexact)]:
                pool=[c for c in [*shared,*extras] if c['tokens']<=budget]
                c,a=independent_choice(pool,cfg,bid);s=score(c,budget)
                allrows.append({'business_id':bid,'name':runs['v4_1_shared_pool',budget]['name'],'split':splits[bid],'review_budget':budget,'diagnostic_pool':label,'count':c['count'],
                                'tokens':s['tokens'],'rating_jsd':s['rating_jsd'],'temporal_jsd':s['temporal_jsd'],'body_mean':s['body_mean'],'source_body_mean':float(source_mean),
                                'body_ratio':s['body_ratio'],'short20_fraction':s['short20_fraction'],'source_short20_fraction':float(short_source),'date_span_coverage':s['date_span_coverage'],
                                'changed_from_v4_1':list(c['ids'])!=original['ids'],'selected_review_ids':list(c['ids']),'below10':c['count']<10,'below5':c['count']<5,
                                'v3_dominates':dominates(candidate(runs['v3_budget_aware',budget]),c),'v4_dominates':dominates(candidate(runs['v4_length_balanced',budget]),c),'final_model_input':False})
            group=[c for c in current['frontiers'][str(budget)] if c['count']==original['count'] and histogram(c['ids'])==oldscore['histogram']]
            assert any(c['ids']==original['ids'] for c in group)
            vals=[score(c,budget)['body_mean'] for c in group]
            finalists=[c for c in current['frontiers'][str(budget)] if c['rating_jsd']<=anchor['rating_jsd']+.001+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+.001+1e-12 and c['count']==original['count'] and c['weighted_cost']==actual['weighted_cost']]
            assert any(c['ids']==original['ids'] for c in finalists)
            largest=max(group,key=lambda c:(sum(body[rid] for rid in c['ids']),tuple(c['ids'])))
            tie_rows.append({'business_id':bid,'name':runs['v4_1_shared_pool',budget]['name'],'review_budget':budget,'selected_count':original['count'],'selected_body_mean':oldscore['body_mean'],
                             'exact_histogram_equivalent_sets':len(group),'exact_count_cost_finalists':len(finalists),'group_body_min':min(vals),'group_body_max':max(vals),'group_body_mean':float(np.mean(vals)),
                             'chosen_body_midrank_in_group':(sum(x<oldscore['body_mean'] for x in vals)+.5*sum(x==oldscore['body_mean'] for x in vals))/len(vals),
                             'selected_tokens':original['tokens'],'largest_equivalent_tokens':largest['tokens'],'longer_same_histogram_exists':max(vals)>oldscore['body_mean']+1e-12,'largest_equivalent_review_ids':list(largest['ids'])})
            if bid in residual_ids and budget==4500:
                v4=candidate(runs['v4_length_balanced',budget]);chosen,a=independent_choice([*current['frontiers'][str(budget)],v4],cfg,bid);s=score(chosen,budget)
                residual.append({'business_id':bid,'name':runs['v4_1_shared_pool',budget]['name'],'review_budget':budget,'dominating_v4_in_shared':any(c['ids']==v4['ids'] for c in shared),
                                 'v4_1_count':original['count'],'v4_1_tokens':original['tokens'],'v4_count':v4['count'],'v4_tokens':v4['tokens'],'carried_count':chosen['count'],'carried_tokens':s['tokens'],
                                 'carried_rating_jsd':s['rating_jsd'],'carried_time_jsd':s['temporal_jsd'],'dominated_after_carry':dominates(v4,chosen)})
        if bid in residual_ids:
            a=runs['v4_length_balanced',4500];b=runs['v4_1_shared_pool',4500]
            ids_a=[rid for rid in a['selected_review_ids'] if rid not in b['selected_review_ids']][:3];ids_b=[rid for rid in b['selected_review_ids'] if rid not in a['selected_review_ids']][:3]
            scopes.append({'business_id':bid,'name':a['name'],'reason':'residual dominating v4 candidate','budget':4500,'read_review_ids':ids_a+ids_b,'status':'pending_assistant_read'})
            for label,ids in [('v4_only',ids_a),('v4_1_only',ids_b)]:
                for rid in ids:readrows.append({'version':label,'name':a['name'],'budget':4500,**byid[rid]})
            comparisons=[('accepted_v4',a['selected_review_ids']),('accepted_v4_1',b['selected_review_ids'])]
        elif bid in low:
            before=candidate(runs['v4_1_shared_pool',3500]);before_score=score(before,3500);ids=list(before['ids']);removed=[];added=[]
            qlo,qhi=np.quantile(list(body.values()),settings['diagnostic_source_body_quantiles'])
            for outid in sorted(before['ids'],key=lambda rid:(body[rid],positions[rid])):
                options=[rid for rid in byid if rid not in ids and int(byid[rid]['stars'])==int(byid[outid]['stars']) and bins[rid]==bins[outid] and qlo<=body[rid]<=qhi and body[rid]>body[outid]]
                options.sort(key=lambda rid:(stable_seed(settings['seed'],'same_cell_middle_length_probe',bid,outid,rid),rid))
                for rid in options:
                    newids=sorted([x for x in ids if x!=outid]+[rid],key=positions.__getitem__)
                    text='\n\n'.join(f"[REV_{i}] stars={int(byid[x]['stars'])} date={byid[x]['date']}\n{byid[x]['text']}" for i,x in enumerate(newids,1))
                    tokens=len(tok.encode(text,add_special_tokens=True).ids)
                    if tokens>3500:continue
                    ids=newids;removed.append(outid);added.append(rid);break
                if len(removed)==settings['maximum_same_cell_replacements']:break
            after={'ids':ids,'count':len(ids),'tokens':len(tok.encode('\n\n'.join(f"[REV_{i}] stars={int(byid[x]['stars'])} date={byid[x]['date']}\n{byid[x]['text']}" for i,x in enumerate(ids,1)),add_special_tokens=True).ids),
                   'rating_jsd':before['rating_jsd'],'temporal_jsd':before['temporal_jsd'],'weighted_cost':before['weighted_cost']}
            s=score(after,3500);assert before_score['histogram']==s['histogram'] and len(ids)==30
            counterfactual.append({'business_id':bid,'name':runs['v4_1_shared_pool',3500]['name'],'review_budget':3500,'replacements':len(removed),'removed_ids':removed,'added_ids':added,
                                   'tokens_before':before['tokens'],'tokens_after':s['tokens'],'body_mean_before':before_score['body_mean'],'body_mean_after':s['body_mean'],
                                   'rating_jsd_unchanged':before['rating_jsd'],'time_jsd_unchanged':before['temporal_jsd'],'date_span_before':before_score['date_span_coverage'],'date_span_after':s['date_span_coverage'],
                                   'source_Q25_body':float(qlo),'source_Q75_body':float(qhi),'selected_review_ids':ids,'final_model_input':False})
            scopes.append({'business_id':bid,'name':runs['v4_1_shared_pool',3500]['name'],'reason':'lowest exact30 blocks; same-cell source-middle-length diagnostic','budget':3500,'read_review_ids':removed+added,'status':'pending_assistant_read'})
            for label,which in [('v4_1_removed',removed),('diagnostic_added',added)]:
                for rid in which:readrows.append({'version':label,'name':runs['v4_1_shared_pool',3500]['name'],'budget':3500,**byid[rid]})
            comparisons=[('accepted_v4_1',before['ids']),('diagnostic_same_histogram',ids)]
        else:continue
        page.append('<h2>'+html.escape(runs['v4_1_shared_pool',3500]['name'])+'</h2>')
        for label,ids in comparisons:
            page.append('<h3>'+label+'</h3>')
            for rid in ids:
                rr=byid[rid];page.append('<details><summary>'+html.escape(f"{rid} | {int(rr['stars'])} stars | {rr['date']} | {body[rid]} BODY tokens")+'</summary><pre>'+html.escape(rr['text'])+'</pre></details>')
    data=pd.DataFrame(allrows);summaries=[]
    for (label,budget),g in data.groupby(['diagnostic_pool','review_budget']):
        summaries.append({'diagnostic_pool':label,'review_budget':int(budget),'businesses':len(g),'count_mean':float(g['count'].mean()),'tokens_min':int(g.tokens.min()),'tokens_max':int(g.tokens.max()),
                          'median_body_ratio':float(g.body_ratio.median()),'short20_macro':float(g.short20_fraction.mean()),'rating_jsd_mean':float(g.rating_jsd.mean()),'time_jsd_mean':float(g.temporal_jsd.mean()),
                          'changed':int(g.changed_from_v4_1.sum()),'below10':int(g.below10.sum()),'below5':int(g.below5.sum()),'v3_dominates':int(g.v3_dominates.sum()),'v4_dominates':int(g.v4_dominates.sum())})
    drops={label:int((g.pivot(index='business_id',columns='review_budget',values='count').diff(axis=1).iloc[:,1:]<0).sum().sum()) for label,g in data.groupby('diagnostic_pool')}
    pd.DataFrame([{k:v for k,v in row.items() if k!='selected_review_ids'} for row in allrows]).to_csv(out/'diagnostic_selection_rows.csv',index=False)
    with (out/'diagnostic_selections.jsonl').open('w',encoding='utf-8') as f:
        for row in allrows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    pd.DataFrame(summaries).to_csv(out/'diagnostic_pool_summary.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k!='largest_equivalent_review_ids'} for row in tie_rows]).to_csv(out/'objective_equivalence_audit.csv',index=False)
    write('objective_equivalence_details.json',tie_rows);pd.DataFrame(residual).to_csv(out/'residual_case_diagnostics.csv',index=False)
    write('same_histogram_counterfactuals.json',counterfactual);write('qualitative_read_scope.json',scopes);write('qualitative_reviews_to_read.json',readrows)
    (out/'complete_original_comparisons.html').write_text('\n'.join(page),encoding='utf-8')
    for rel,sha in protected.items():assert digest(ROOT/rel)==sha,rel
    for directory in [source,older]:
        for rel,sha in json.loads((directory/'completion_report.json').read_text(encoding='utf-8'))['artifact_sha256'].items():assert digest(directory/rel)==sha,rel
    validation={'validation_passed':True,'businesses':80,'source_originals':40407,'diagnostic_choices':len(allrows),'independent_original_diagnostic_equivalence_checks':checks,
                'unique_blocks_fresh_encoded':len(blocks),'objective_equivalence_cases':len(tie_rows),'residual_cases':len(residual),'counterfactual_cases':len(counterfactual),
                'reviews_prepared_for_assistant_read':len(readrows),'diagnostic_count_drops':drops,'accepted_science_and_sources_unchanged':True,
                'fresh_selector_run':False,'final_manifest':False,'final_methodology_locked':False,'final_review_budget':None,'script_sha256':digest(Path(__file__)),
                'config_sha256':digest(ROOT/'configs/review_selection_residual_audit.yaml'),'protected_sha256':protected,'elapsed_seconds':round(time.perf_counter()-started,3)}
    validation['artifact_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='validation_report.json'}
    write('validation_report.json',validation);print(json.dumps({k:v for k,v in validation.items() if k not in ['artifact_sha256','protected_sha256']},indent=2))


if __name__=='__main__':main()
