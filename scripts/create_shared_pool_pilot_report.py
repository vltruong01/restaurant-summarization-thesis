"""Report actual fresh v4.1 outcomes, regressions and complete-original examples."""
from pathlib import Path
import hashlib,html,json
import pandas as pd,orjson
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_c/review_selection_shared_pool_pilot_v1'


def table(frame):
    columns=frame.columns.tolist()
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*['| '+' | '.join(str(v).replace('|','\\|') for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def main():
    validation=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'));assert validation['validation_passed']
    frame=pd.read_csv(OUT/'selection_diagnostics.csv');summary=pd.read_csv(OUT/'budget_method_summary.csv')
    selected=frame[frame.method.eq('v4_1_shared_pool')];comparisons=[];regressions=[]
    for budget in [3500,4000,4500]:
        a=selected[selected.review_budget.eq(budget)].set_index('business_id')
        for method in ['v3_budget_aware','v4_length_balanced']:
            b=frame[frame.method.eq(method)&frame.review_budget.eq(budget)].set_index('business_id').reindex(a.index)
            for bid,x in a.iterrows():
                y=b.loc[bid];dr=x.rating_jsd_after-y.rating_jsd_after;dt=x.temporal_jsd_after-y.temporal_jsd_after;dc=x.final_retained_review_count-y.final_retained_review_count
                dominated=dr>=-1e-12 and dt>=-1e-12 and dc<=0 and (dr>1e-12 or dt>1e-12 or dc<0)
                regressions.append({'business_id':bid,'name':x['name'],'review_budget':budget,'baseline':method,'count_delta':dc,'rating_delta':dr,'time_delta':dt,'baseline_pareto_dominates_v4_1':dominated})
            g=pd.DataFrame(regressions);g=g[g.review_budget.eq(budget)&g.baseline.eq(method)]
            comparisons.append({'budget':budget,'baseline':method,'count_lower':int(g.count_delta.lt(0).sum()),'rating_worse':int(g.rating_delta.gt(1e-12).sum()),'time_worse':int(g.time_delta.gt(1e-12).sum()),'baseline_dominates':int(g.baseline_pareto_dominates_v4_1.sum())})
    pd.DataFrame(regressions).to_csv(OUT/'paired_baseline_comparisons.csv',index=False);pd.DataFrame(comparisons).to_csv(OUT/'regression_summary.csv',index=False)
    piv=selected.pivot(index='business_id',columns='review_budget',values='final_retained_review_count');drops=int((piv.diff(axis=1).iloc[:,1:]<0).sum().sum())
    transitions=[];terminal=[];v4presence={b:0 for b in [3500,4000,4500]}
    caches={}
    for path in sorted((OUT/'business_runs').glob('*.json')):
        p=json.loads(path.read_text(encoding='utf-8'));bid=p['business_id'];caches[bid]=p
        sets={tuple(c['ids']) for c in p['shared_candidates']};rs={(r['method'],r['review_budget']):r for r in p['runs']}
        for b in [3500,4000,4500]:
            v4presence[b]+=tuple(rs['v4_length_balanced',b]['selected_review_ids']) in sets
            r=rs['v4_1_shared_pool',b];moves=r['search_provenance']['accepted_local_path'];db=r['search_provenance']['discovery_budget']
            if moves and moves[-1]['tokens_after']>db:terminal.append({'business_id':bid,'name':r['name'],'final_budget':b,'discovery_budget':db,'final_tokens':moves[-1]['tokens_after'],'interpretation':'terminal proposal failed discovery budget; fits final shared-pool budget'})
        for low,high in [(3500,4000),(4000,4500)]:
            x=rs['v4_1_shared_pool',low];y=rs['v4_1_shared_pool',high]
            transitions.append({'business_id':bid,'name':x['name'],'lower_budget':low,'higher_budget':high,'lower_count':x['final_retained_review_count'],'higher_count':y['final_retained_review_count'],'count_decreased':y['final_retained_review_count']<x['final_retained_review_count']})
    pd.DataFrame(transitions).to_csv(OUT/'cross_budget_transitions.csv',index=False)
    pd.DataFrame(terminal,columns=['business_id','name','final_budget','discovery_budget','final_tokens','interpretation']).to_csv(OUT/'cross_budget_terminal_proposals.csv',index=False)
    case_ids=['RoUP3lcY4wEtdASLNIcmSw','ctHjyadbDQAtUFfkcAFEHw','9PZxjhTIU7OgPIzuGi89Ew','fbkdwbIawWmDhgSrlI8Akw','UcJbErL_KZbzKzxddiaIwQ','9tXwyr5Yqu4WnuCLOUPtrQ','TIGOlkeBefuEC4TxfueJEA']
    pop={bid:{} for bid in case_ids}
    for line in (OUT/'pilot_population.jsonl').open('rb'):
        r=orjson.loads(line)['review']
        if r['business_id'] in pop:pop[r['business_id']][r['review_id']]=r
    page=['<!doctype html><meta charset="utf-8"><title>Fresh v4.1 original comparisons</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:8px}</style><h1>V3/V4/V4.1: complete-original pilot comparisons</h1><p>Pilot only; not final model evidence or formal human evaluation.</p>'];examples=[]
    for bid in case_ids:
        budget=4500 if bid in case_ids[-3:] else 3500
        for method in ['v3_budget_aware','v4_length_balanced','v4_1_shared_pool']:
            row=next(z for z in caches[bid]['runs'] if z['method']==method and z['review_budget']==budget)
            examples.append({k:row[k] for k in ['business_id','name','method','review_budget','final_retained_review_count','tokens_after_budget_enforcement','rating_jsd_after','temporal_jsd_after','selected_body_mean','selected_source_mean_ratio']})
            page.append('<h2>'+html.escape(row['name'])+' / '+method+' / '+str(budget)+'</h2>')
            for rid in row['selected_review_ids']:
                r=pop[bid][rid];page.append('<details><summary>'+html.escape(f"{rid} | {r['stars']} stars | {r['date']}")+'</summary><pre>'+html.escape(r['text'])+'</pre></details>')
    pd.DataFrame(examples).to_csv(OUT/'detailed_examples.csv',index=False)
    (OUT/'complete_original_comparisons.html').write_text('\n'.join(page),encoding='utf-8')
    r=summary[summary.method.isin(['v3_budget_aware','v4_length_balanced','v4_1_shared_pool'])][['method','review_budget','retained_mean','tokens_min','tokens_max','below10','median_mean_ratio','selected_short20_macro','rating_jsd_mean','temporal_jsd_mean','exactly30_under1500']].round(6)
    lines=['# Fresh v4.1 shared-candidate pilot — 2026-10-06',
           '**Fresh pilot implemented and validated. Candidate reuse/incumbent protection fixes the diagnosed count/coverage losses relative to v3, but substantial short-review shift remains. Do not freeze or expand automatically.**',
           'Same80 businesses (69train/11validation),40,407 full English originals;1,200 runs including fresh v4.1 and four immutable comparisons at three budgets. Accepted v4 is comparison only, not discovery input. Freshly generate v4 candidates with original sampling/tie hash version; add all feasible accepted v3 methods/budgets as seeds, discover bounded local candidates per budget, then filter/rerank a shared exact pool. No change to research objectives/seed/tolerances/budgets/cap/complete boundaries. No test tuning/models/API/GPU work.',
           '## Actual fresh results',table(r),
           'At3500 mean retained26.525 equals v3 and exceeds v4=24.0375; at4000/4500 means27.0875/27.1750. Retained10–30 with0<10/<5/infeasible and0 adjacent count decreases. No ten-review floor or global monotonic guarantee. Exact block ranges959–3495/959–3994/959–4485; blocks still exclude pending business metadata/captions.',
           '## Regressions remain visible',table(pd.DataFrame(comparisons)),
           'No accepted v3 incumbent Pareto-dominates v4.1, and count never falls below v3 at the same budget. At4000/4500 one temporal regression versus v3 remains; it trades against another metric/count. At4500 two accepted v4 sets dominate fresh v4.1: Taco Grande and Rhone Rum Bar. Old v4 candidates are not guaranteed present in fresh discovery because v3 seeds change bounded local starts and proposals. This is a remaining coverage limitation, not source corruption.',
           '## Residual short-review bias and difference from previous counterfactual',
           'Median selected/source BODY ratios improve only0.417→0.457 /0.423→0.498 /0.429→0.503 relative to v3. Selected<20 fraction remains11.65%/11.30%/11.09% against source1.05%; v3 was13.53%/13.52%/13.51%. Exactly30 blocks<1500 decrease11→5 per budget. Source-weighted and macro quantities are distinguished; thresholds are descriptions, not quality/exclusion gates.',
           'Previous003g cached-union + v3 counterfactual ratios0.727/0.774/0.845 came from accepted v4 frontiers. This fresh pilot regenerates discovery with incumbent-influenced anchors and does not import those old v4 candidates. Candidate populations/tie choices differ; do not substitute the earlier stronger ratio gain for actual fresh results. Token-neutral ties do not guarantee source-like length distribution. Longer text alone still does not prove semantic coverage.',
           '## Seven detailed original comparisons',table(pd.DataFrame(examples).round(6)),
           'Water Works3500: v4=10/3025, v4.1=30/2960 with v3-equivalent rating/time JSD. El Vez3500: v3=30/1154, v4=30/3420, v4.1=30/1556; rating/time identical across these three3500 sets. Zahav3500: v4=25/3377 with worse ratingJSD; v4.1=30/1514 with v3-equivalent representation. Full unchanged originals for seven cases in complete_original_comparisons.html and notebook viewer. No new formal human evaluation/aspect-recall claim.',
           '## Validation and provenance',
           f"Nine synthetic checks PASS; independent{validation['runs_verified']} run checks, {validation['shared_candidates_verified']} shared candidates, {validation['frontier_candidates_verified']} frontier candidates, {validation['local_moves_verified']} derivation moves, {validation['sensitivity_choices_verified']} tolerance choices. All40,407 original source joins and BODY encodes verified; fresh reversed-source all80 full payloads equal (240 fresh v4.1 choices). Validation{validation['elapsed_seconds']}s; pilot52.441s. Original scripts/config/caches remain unchanged.",
           f'Terminal proposals initially over discovery budget but feasible at final budget: {len(terminal)} final choices. Inherited accepted_local_path denotes candidate derivation, not acceptance of every terminal proposal inside local discovery. Accepted intermediate states fit discovery budget; terminal selected set fits final budget. Initial validator incorrectly required terminal fit at discovery budget; corrected that assertion under clarified003h without changing locked selector/config/results, then validated fresh outputs. Cases recorded in cross_budget_terminal_proposals.csv.',
           '## Stop and next decision',
           'The authorized80-business pilot is complete. Preserve v4.1 as validated diagnostic evidence; final methodology and review/metadata/caption allocation stay null. Do not run all2200 or freeze2300 automatically. Review the two v4-dominates-v4.1 cases and candidate/tie coverage together with remaining short-review shift; retained count/JSD alone are insufficient grounds to call inputs representative in content. A further controlled pilot may retain accepted v4 discovery candidates as comparison/protection candidates, but changing length or semantic objectives requires explicit research-design review. Keep all future dataset/filtering work in notebook00.']
    (OUT/'pilot_review_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    report={'counter_decreases':drops,'regression_summary':comparisons,'accepted_v4_sets_in_fresh_shared_pool':v4presence,'cross_budget_terminal_proposals':len(terminal),'example_businesses':7,'final_manifest':False,'final_methodology_locked':False}
    (OUT/'review_findings.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
