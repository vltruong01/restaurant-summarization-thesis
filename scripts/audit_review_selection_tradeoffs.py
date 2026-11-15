"""Read-only cached v4 tradeoff audit; counterfactuals are never model manifests."""
from pathlib import Path
from collections import Counter
import hashlib, html, json, math, os, time
import numpy as np
import orjson, pandas as pd, yaml
from tokenizers import Tokenizer
from budget_aware_review_selection import nondominated
from budget_aware_review_selection_v4 import choose_candidate

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/stage_c/review_selection_length_balanced_pilot_v1'
OUT=ROOT/'outputs/stage_c/review_selection_tradeoff_audit_v1'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def dominates(a,b):
    weak=a['rating_jsd']<=b['rating_jsd']+1e-12 and a['temporal_jsd']<=b['temporal_jsd']+1e-12 and a['count']>=b['count']
    strict=a['rating_jsd']<b['rating_jsd']-1e-12 or a['temporal_jsd']<b['temporal_jsd']-1e-12 or a['count']>b['count']
    return weak and strict


def independent_choice(pool,cfg,bid):
    unique={tuple(c['ids']):c for c in pool}
    values=list(unique.values())
    frontier=[c for c in values if not any(dominates(a,c) for a in values)]
    def tie(c):
        raw=orjson.dumps([cfg['seed'],cfg['algorithm_version'],bid,'choice',list(c['ids'])])
        return int.from_bytes(hashlib.sha256(raw).digest()[:16],'big')
    anchor=min(frontier,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],tie(c),tuple(c['ids'])))
    comparable=[c for c in frontier if c['rating_jsd']<=anchor['rating_jsd']+.001+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+.001+1e-12]
    chosen=min(comparable,key=lambda c:(-c['count'],c['weighted_cost'],tie(c),tuple(c['ids'])))
    actual=choose_candidate(nondominated(values),cfg,bid)
    assert list(chosen['ids'])==list(actual['ids'])
    return chosen,anchor


def candidate(run):
    return {'ids':run['selected_review_ids'],'count':run['final_retained_review_count'],'tokens':run['tokens_after_budget_enforcement'],
            'rating_jsd':run['rating_jsd_after'],'temporal_jsd':run['temporal_jsd_after'],
            'weighted_cost':.5*(run['rating_jsd_after']+run['temporal_jsd_after'])}


def jsd(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);a/=a.sum();b/=b.sum();m=(a+b)/2
    return float(sum(.5*np.sum(v[v>0]*np.log2(v[v>0]/m[v>0])) for v in (a,b)))


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'configs/review_selection_length_balanced_pilot.yaml').read_text(encoding='utf-8'))
    assert cfg['search']['comparable_rating_jsd_tolerance']==cfg['search']['comparable_temporal_jsd_tolerance']==.001
    completion=json.loads((SOURCE/'completion_report.json').read_text(encoding='utf-8'))
    assert json.loads((SOURCE/'validation_report.json').read_text(encoding='utf-8'))['validation_passed']
    for rel,sha in completion['artifact_sha256'].items():assert digest(SOURCE/rel)==sha,rel
    protected={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'completion_report.json',SOURCE/'validation_report.json',ROOT/'configs/review_selection_length_balanced_pilot.yaml',ROOT/'scripts/budget_aware_review_selection_v4.py']}
    populations={}
    for line in (SOURCE/'pilot_population.jsonl').open('rb'):
        r=orjson.loads(line)['review'];populations.setdefault(r['business_id'],[]).append(r)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    tokpath=ROOT/'assets/tokenizers/led-base-16384/tokenizer.json';tok=Tokenizer.from_file(str(tokpath));tok.no_truncation();tok.no_padding()
    pairs=[];transitions=[];shadow=[];blocks={};payloads={};checks=0
    for path in sorted((SOURCE/'business_runs').glob('*.json')):
        payload=json.loads(path.read_text(encoding='utf-8'));bid=payload['business_id'];payloads[bid]=payload
        runs={(r['review_budget'],r['method']):r for r in payload['runs']}
        frontiers=payload['frontiers'];union={tuple(c['ids']):c for f in frontiers.values() for c in f}
        v3s=[candidate(runs[b,'v3_budget_aware']) for b in [3500,4000,4500]]
        pop=sorted(populations[bid],key=lambda r:(r['date'],r['review_id']));byid={r['review_id']:r for r in pop}
        bins={r['review_id']:i for i in range(5) for r in pop[i*len(pop)//5:(i+1)*len(pop)//5]}
        stars=Counter(int(r['stars']) for r in pop);times=Counter(bins[r['review_id']] for r in pop)
        source_mean=np.mean([len(tok.encode(r['text'],add_special_tokens=False).ids) for r in pop])
        def verify(c,budget):
            nonlocal checks
            key=(bid,tuple(c['ids']));assert len(set(c['ids']))==c['count']<=30 and set(c['ids'])<=set(byid)
            selected=sorted([byid[rid] for rid in c['ids']],key=lambda r:(r['date'],r['review_id']))
            assert [r['review_id'] for r in selected]==list(c['ids'])
            if key not in blocks:
                text='\n\n'.join(f"[REV_{i}] stars={int(r['stars'])} date={r['date']}\n{r['text']}" for i,r in enumerate(selected,1))
                exact=len(tok.encode(text,add_special_tokens=True).ids)
                sr=Counter(int(r['stars']) for r in selected);st=Counter(bins[r['review_id']] for r in selected)
                rating=jsd([stars[i] for i in range(1,6)],[sr[i] for i in range(1,6)])
                temporal=jsd([times[i] for i in range(5)],[st[i] for i in range(5)])
                body=np.mean([len(tok.encode(r['text'],add_special_tokens=False).ids) for r in selected])
                blocks[key]={'tokens':exact,'rating':rating,'temporal':temporal,'body_mean':float(body),'source_body_ratio':float(body/source_mean)}
            info=blocks[key];assert info['tokens']==c['tokens']<=budget
            assert math.isclose(info['rating'],c['rating_jsd'],abs_tol=1e-12) and math.isclose(info['temporal'],c['temporal_jsd'],abs_tol=1e-12)
            checks+=1;return info
        for budget in [3500,4000,4500]:
            original=runs[budget,'v4_length_balanced'];v4=candidate(original);v3=candidate(runs[budget,'v3_budget_aware']);front=frontiers[str(budget)]
            chosen,anchor=independent_choice(front,cfg,bid);assert chosen['ids']==v4['ids'];verify(v4,budget)
            audit_ids={tuple(c['ids']) for c in payload['candidate_audit']}
            pairs.append({'business_id':bid,'name':original['name'],'split':original['split'],'review_budget':budget,
                          'v3_count':v3['count'],'v4_count':v4['count'],'count_delta':v4['count']-v3['count'],
                          'rating_delta':v4['rating_jsd']-v3['rating_jsd'],'time_delta':v4['temporal_jsd']-v3['temporal_jsd'],
                          'v3_pareto_dominates_v4':dominates(v3,v4),'v3_in_any_evaluated_candidates':tuple(v3['ids']) in audit_ids,
                          'v3_in_current_frontier':any(c['ids']==v3['ids'] for c in front),
                          'v3_comparable_to_anchor':v3['rating_jsd']<=anchor['rating_jsd']+.001+1e-12 and v3['temporal_jsd']<=anchor['temporal_jsd']+.001+1e-12})
            pools={'union_budget_frontiers':[c for c in union.values() if c['tokens']<=budget],
                   'current_frontier_plus_v3':front+[v3],
                   'union_frontiers_plus_v3':[c for c in [*union.values(),*v3s] if c['tokens']<=budget]}
            for label,pool in pools.items():
                c,a=independent_choice(pool,cfg,bid);info=verify(c,budget)
                shadow.append({'business_id':bid,'name':original['name'],'split':original['split'],'review_budget':budget,'diagnostic_pool':label,
                               'count':c['count'],'original_v4_count':v4['count'],'tokens':c['tokens'],'rating_jsd':c['rating_jsd'],'temporal_jsd':c['temporal_jsd'],
                               'body_mean':info['body_mean'],'source_body_ratio':info['source_body_ratio'],'selected_review_ids':c['ids'],
                               'changed_from_v4':c['ids']!=v4['ids'],'final_model_input':False})
        for low,high in [(3500,4000),(4000,4500)]:
            prev=candidate(runs[low,'v4_length_balanced']);current=candidate(runs[high,'v4_length_balanced'])
            pool=frontiers[str(high)];carried,anchor=independent_choice(pool+[prev],cfg,bid);verify(carried,high)
            comparable=prev['rating_jsd']<=anchor['rating_jsd']+.001+1e-12 and prev['temporal_jsd']<=anchor['temporal_jsd']+.001+1e-12
            transitions.append({'business_id':bid,'name':runs[low,'v4_length_balanced']['name'],'lower_budget':low,'higher_budget':high,
                                'previous_count':prev['count'],'higher_count':current['count'],'count_decreased':current['count']<prev['count'],
                                'previous_tokens':prev['tokens'],'previous_still_fits':prev['tokens']<=high,
                                'previous_in_higher_frontier':any(c['ids']==prev['ids'] for c in pool),'previous_comparable_after_carry':comparable,
                                'carry_counterfactual_count':carried['count'],'carry_rating_jsd':carried['rating_jsd'],'carry_temporal_jsd':carried['temporal_jsd']})
    pd.DataFrame(pairs).to_csv(OUT/'v3_v4_tradeoffs.csv',index=False)
    pd.DataFrame(transitions).to_csv(OUT/'cross_budget_transitions.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='selected_review_ids'} for r in shadow]).to_csv(OUT/'counterfactual_summary_rows.csv',index=False)
    with (OUT/'counterfactual_selections.jsonl').open('w',encoding='utf-8') as f:
        for r in shadow:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    frame=pd.DataFrame(shadow);summary=[]
    for (label,budget),g in frame.groupby(['diagnostic_pool','review_budget']):
        summary.append({'diagnostic_pool':label,'review_budget':int(budget),'businesses':len(g),'changed':int(g.changed_from_v4.sum()),
                        'count_mean':float(g['count'].mean()),'count_min':int(g['count'].min()),'below10':int(g['count'].lt(10).sum()),
                        'median_source_body_ratio':float(g.source_body_ratio.median()),'rating_jsd_mean':float(g.rating_jsd.mean()),'time_jsd_mean':float(g.temporal_jsd.mean())})
    pd.DataFrame(summary).to_csv(OUT/'counterfactual_budget_summary.csv',index=False)
    # Purposeful content cases, not a formal human-evaluation sample.
    case_ids=['ctHjyadbDQAtUFfkcAFEHw','UcJbErL_KZbzKzxddiaIwQ','RoUP3lcY4wEtdASLNIcmSw','eOWBvwP1MgJx6PrrWBoUuA']
    reasons=['largest rating regression','largest temporal regression','largest count loss / long-review contrast','additional rating/count regression']
    scopes=[];readrows=[];page=['<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{padding:10px;border:1px solid #ddd}</style><h1>Tradeoff audit: full original v3/v4 reviews</h1>']
    for bid,reason in zip(case_ids,reasons):
        budget=4500 if bid=='UcJbErL_KZbzKzxddiaIwQ' else 3500
        payload=payloads[bid];runs={(r['review_budget'],r['method']):r for r in payload['runs']};a=runs[budget,'v3_budget_aware'];b=runs[budget,'v4_length_balanced'];byid={r['review_id']:r for r in populations[bid]}
        only3=[rid for rid in a['selected_review_ids'] if rid not in b['selected_review_ids']][:3]
        only4=[rid for rid in b['selected_review_ids'] if rid not in a['selected_review_ids']][:3]
        scopes.append({'business_id':bid,'name':a['name'],'reason':reason,'review_budget':budget,'v3_only_ids_to_read':only3,'v4_only_ids_to_read':only4,'status':'pending_assistant_read'})
        page.append('<h2>'+html.escape(a['name'])+' | '+str(budget)+'</h2>')
        for version,run in [('v3',a),('v4',b)]:
            page.append(f'<h3>{version}: {run["final_retained_review_count"]} complete reviews, {run["tokens_after_budget_enforcement"]} tokens</h3>')
            for rid in run['selected_review_ids']:
                r=byid[rid];page.append('<details><summary>'+html.escape(f"{rid} | {r['stars']} stars | {r['date']}")+'</summary><pre>'+html.escape(r['text'])+'</pre></details>')
            for rid in only3 if version=='v3' else only4:readrows.append({'business_id':bid,'name':a['name'],'budget':budget,'version':version,**byid[rid]})
    (OUT/'full_original_tradeoff_comparisons.html').write_text('\n'.join(page),encoding='utf-8')
    write('qualitative_read_scope.json',scopes);write('qualitative_reviews_to_read.json',readrows)
    drops={label:int((g.pivot(index='business_id',columns='review_budget',values='count').diff(axis=1).iloc[:,1:]<0).sum().sum()) for label,g in frame.groupby('diagnostic_pool')}
    for rel,sha in completion['artifact_sha256'].items():assert digest(SOURCE/rel)==sha,rel
    for rel,sha in protected.items():assert digest(ROOT/rel)==sha,rel
    report={'validation_passed':True,'businesses':80,'pairs':len(pairs),'adjacent_transitions':len(transitions),'original_count_decreases':sum(r['count_decreased'] for r in transitions),
            'diagnostic_runs':len(shadow),'independent_ranking_and_fresh_token_jsd_checks':checks,'unique_selected_blocks_fresh_encoded':len(blocks),
            'cached_union_count_decreases':drops,'source_and_accepted_results_unchanged':True,'final_manifest':False,'methodology_locked':False,'final_review_budget':None,
            'scientific_identity_sha256':completion['identity_sha256'],'script_sha256':digest(Path(__file__)),'tokenizer_sha256':digest(tokpath),
            'protected_sha256':protected,'elapsed_seconds':round(time.perf_counter()-started,3)}
    report['artifact_sha256']={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='validation_report.json'}
    write('validation_report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
