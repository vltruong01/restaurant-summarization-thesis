"""Global expanded statistics, extreme cases and explicit limited reading scope."""
from pathlib import Path
import argparse,hashlib,html,json
import pandas as pd,orjson
from create_source_length_tie_report import markdown_table
from review_selection_pilot import stable_seed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    frame=pd.read_csv(OUT/'selection_diagnostics.csv',keep_default_na=False);final=frame[frame.method.eq('v4_2_length_balanced')];g=final[final.review_budget.eq(3500)]
    businesses=pd.read_csv(OUT/'businesses.csv',keep_default_na=False).set_index('business_id');reasons={}
    def add(bid,reason):reasons.setdefault(bid,[]).append(reason)
    add(g.sort_values(['tokens_after_budget_enforcement','business_id']).iloc[0].business_id,'minimum_final3500_block')
    exact=g[g.final_retained_review_count.eq(30)]
    add(exact.sort_values(['tokens_after_budget_enforcement','business_id']).iloc[0].business_id,'minimum_exact30_final3500_block')
    add(g.sort_values(['tokens_after_budget_enforcement','business_id'],ascending=[False,True]).iloc[0].business_id,'maximum_final3500_block')
    add(businesses.sort_values(['english_review_token_max','business_id'],ascending=[False,True]).index[0],'longest_C1_source_review_stress_case')
    add(g.sort_values(['length_jsd','business_id'],ascending=[False,True]).iloc[0].business_id,'largest_residual3500_lengthJSD')
    add(g.sort_values(['selected_source_mean_ratio','business_id']).iloc[0].business_id,'lowest_final3500_BODY_mean_ratio')
    pairs=pd.read_csv(OUT/'paired_primary_secondary.csv');part=pairs[pairs.review_budget.eq(3500)].copy();part['improvement']=part.length_jsd_primary-part.length_jsd_final
    add(part.sort_values(['improvement','business_id'],ascending=[False,True]).iloc[0].business_id,'largest_secondary3500_JSD_improvement')
    flags=json.loads((OUT/'below_ten_attribution.json').read_text(encoding='utf-8'))
    for item in flags:add(item['business_id'],'flagged_below10')
    drops_path=OUT/'cross_budget_count_drops.csv';drops=pd.read_csv(drops_path) if drops_path.stat().st_size>1 else pd.DataFrame()
    if len(drops):
        new=drops[drops.method.eq('v4_2_length_balanced')]
        if len(new):add(new.iloc[0].business_id,'cross_budget_count_drop')
    for bid in businesses.sort_values(['english_reviews','business_id'],ascending=[False,True]).index:
        if len(reasons)>=8:break
        if bid not in reasons:add(bid,'high_volume_source')
    selected=set(reasons);populations={bid:[] for bid in selected}
    for split in ['train','validation']:
        for line in (ROOT/'data/processed/yelp_2300_english_v1/reviews'/f'{split}.jsonl').open('rb'):
            r=orjson.loads(line)
            if r['business_id'] in selected:populations[r['business_id']].append(r)
    examples=[];reads=[];page=['<!doctype html><meta charset="utf-8"><title>Expanded v4.2 whole reviews</title><style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;margin:.5rem}</style><h1>All2200 expanded diagnostics — example originals</h1><p>Exploratory allocations only; length balance is not semantic quality or final model inputs.</p>']
    for bid,why in reasons.items():
        payload=json.loads((OUT/'business_runs'/(hashlib.sha256(bid.encode()).hexdigest()+'.json')).read_text(encoding='utf-8'));runs={r['method']:r for r in payload['runs'] if r['review_budget']==3500};byid={r['review_id']:r for r in populations[bid]};body=payload['body_lengths']
        control=runs['v4_2_primary_control'];last=runs['v4_2_length_balanced']
        examples.append({'business_id':bid,'name':last['name'],'reasons':';'.join(why),'source_count':len(byid),'source_BODY_min':min(body.values()),'source_BODY_max':max(body.values()),'review_budget':3500,'retained':last['final_retained_review_count'],'tokens_control':control['tokens_after_budget_enforcement'],'tokens_final':last['tokens_after_budget_enforcement'],'rating_jsd':last['rating_jsd_after'],'time_jsd':last['temporal_jsd_after'],'length_jsd_control':control['length_jsd'],'length_jsd_final':last['length_jsd'],'BODY_ratio_control':control['selected_source_mean_ratio'],'BODY_ratio_final':last['selected_source_mean_ratio']})
        page.append('<h2>'+html.escape(last['name'])+' — '+html.escape(';'.join(why))+'</h2>')
        for method in ['v3_budget_aware','v4_2_primary_control','v4_2_length_balanced']:
            row=runs[method];page.append('<h3>'+method+f" | count {row['final_retained_review_count']} | tokens {row['tokens_after_budget_enforcement']}</h3>")
            for rid in row['selected_review_ids']:
                r=byid[rid];page.append('<details><summary>'+html.escape(f"{rid} | {r['stars']} stars | {r['date']} | BODY {body[rid]}")+'</summary><pre>'+html.escape(r['text'])+'</pre></details>')
        difference_count=0
        for label,ids,other in [('control_only',control['selected_review_ids'],set(last['selected_review_ids'])),('final_only',last['selected_review_ids'],set(control['selected_review_ids']))]:
            scope=[rid for rid in ids if rid not in other][:2]
            difference_count+=len(scope)
            for rid in scope:reads.append({'name':last['name'],'version':label,'BODY_tokens':body[rid],**byid[rid]})
        if not difference_count:
            for rid in last['selected_review_ids'][:2]:reads.append({'name':last['name'],'version':'unchanged_selected','BODY_tokens':body[rid],**byid[rid]})
    if args.prepare:
        pd.DataFrame(examples).to_csv(OUT/'example_businesses.csv',index=False)
        (OUT/'complete_original_comparisons.html').write_text('\n'.join(page),encoding='utf-8')
        (OUT/'qualitative_reviews_to_read.json').write_text(json.dumps(reads,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'example_businesses':len(examples),'reviews_prepared':len(reads)},indent=2));return
    validation=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'));assert validation['validation_passed']
    notes=json.loads((OUT/'qualitative_review_notes.json').read_text(encoding='utf-8'));assert notes['read_reviews']==len(reads)
    assert {r['review_id'] for r in reads}=={rid for c in notes['cases'] for rid in c['read_review_ids']} and all(c['status']=='assistant_read_complete' for c in notes['cases'])
    reg=pd.read_csv(OUT/'baseline_regressions.csv');summary=pd.read_csv(OUT/'budget_method_summary.csv')
    before_after=[]
    before_frame=frame[frame.method.str.startswith('v4_2_')].copy()
    for column in ['initial_eligible_review_count','initial_candidate_count','tokens_before_budget_enforcement','tokens_after_budget_enforcement','rating_jsd_before','rating_jsd_after','temporal_jsd_before','temporal_jsd_after']:
        before_frame[column]=pd.to_numeric(before_frame[column],errors='raise')
    for (method,budget),part in before_frame.groupby(['method','review_budget']):
        item={'method':method,'review_budget':int(budget),'businesses':len(part),'initial_eligible_count_mean':float(part.initial_eligible_review_count.mean()),'initial_candidate_count_mean':float(part.initial_candidate_count.mean()),'tokens_before_min':int(part.tokens_before_budget_enforcement.min()),'tokens_before_max':int(part.tokens_before_budget_enforcement.max()),'tokens_after_min':int(part.tokens_after_budget_enforcement.min()),'tokens_after_max':int(part.tokens_after_budget_enforcement.max())}
        for label,prefix in [('rating','rating_jsd'),('temporal','temporal_jsd')]:
            before=part[prefix+'_before'];after=part[prefix+'_after'];delta=after-before
            item.update({label+'_jsd_before_mean':float(before.mean()),label+'_jsd_after_mean':float(after.mean()),label+'_improved_vs_reference':int((delta<-1e-12).sum()),label+'_equal_vs_reference':int((delta.abs()<=1e-12).sum()),label+'_worse_vs_reference':int((delta>1e-12).sum())})
        before_after.append(item)
    pd.DataFrame(before_after).to_csv(OUT/'primary_before_after_summary.csv',index=False)
    drop_details=[]
    if len(drops):
        for item in drops[drops.method.eq('v4_2_length_balanced')].to_dict('records'):
            bid=item['business_id'];payload=json.loads((OUT/'business_runs'/(hashlib.sha256(bid.encode()).hexdigest()+'.json')).read_text(encoding='utf-8'))
            rows={r['review_budget']:r for r in payload['runs'] if r['method']=='v4_2_primary_control'}
            low=rows[item['lower_budget']];high=rows[item['higher_budget']];front=payload['frontiers'][str(item['higher_budget'])]
            anchor=min(front,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],stable_seed(42,'complete_review_search_v4_length_balanced',bid,'choice',list(c['ids'])),c['ids']))
            comparable=low['rating_jsd_after']<=anchor['rating_jsd']+.001+1e-12 and low['temporal_jsd_after']<=anchor['temporal_jsd']+.001+1e-12
            detail={**item,'lower_primary_rating_jsd':low['rating_jsd_after'],'higher_primary_rating_jsd':high['rating_jsd_after'],'lower_primary_time_jsd':low['temporal_jsd_after'],'higher_primary_time_jsd':high['temporal_jsd_after'],'higher_anchor_rating_jsd':anchor['rating_jsd'],'higher_anchor_time_jsd':anchor['temporal_jsd'],'lower_control_in_higher_primary_pool':any(c['ids']==low['selected_review_ids'] for c in payload['primary_candidates']),'lower_control_in_higher_frontier':any(c['ids']==low['selected_review_ids'] for c in front),'lower_control_comparable_to_higher_anchor':comparable,'primary_count_drop_already_present':low['final_retained_review_count']>high['final_retained_review_count'],'automatic_method_change':False}
            assert detail['primary_count_drop_already_present']
            drop_details.append(detail)
    (OUT/'v4_2_count_drop_explanations.json').write_text(json.dumps(drop_details,indent=2)+'\n',encoding='utf-8')
    extremes=[]
    for budget,part in final.groupby('review_budget'):
        for category,column,ascending in [('minimum_block','tokens_after_budget_enforcement',True),('maximum_block','tokens_after_budget_enforcement',False),('minimum_exact30_block','tokens_after_budget_enforcement',True)]:
            pool=part[part.final_retained_review_count.eq(30)] if category=='minimum_exact30_block' else part
            row=pool.sort_values([column,'business_id'],ascending=[ascending,True]).iloc[0]
            extremes.append({'review_budget':int(budget),'extreme':category,'business_id':row.business_id,'name':row['name'],'retained_count':int(row.final_retained_review_count),'block_tokens':int(row.tokens_after_budget_enforcement),'selected_BODY_mean':float(row.selected_body_mean),'source_BODY_mean':float(row.source_body_mean)})
    pd.DataFrame(extremes).to_csv(OUT/'token_extreme_businesses.csv',index=False)
    findings={'businesses':2200,'splits':{'train':2000,'validation':200},'source_reviews':575749,'primary_histogram_pairs_preserved':int(pairs.primary_histograms_preserved.sum()),'secondary_length_jsd_improved_pairs':int((pairs.length_jsd_final<pairs.length_jsd_primary-1e-12).sum()),'secondary_changed_pairs':int(pairs.changed_from_control.sum()),'count_drop_transitions_by_method':drops.groupby('method').size().to_dict() if len(drops) else {},'v4_2_count_drop_transitions':len(drop_details),'baseline_dominates_final':reg.groupby('baseline').baseline_dominates_final.sum().astype(int).to_dict(),'below10_final_cases':len(flags),'below10_demonstrably_ten_fit':sum(r['ten_complete_reviews_demonstrably_fit'] for r in flags),'example_businesses':len(examples),'assistant_read_originals':len(reads),'additional_fresh_replay_businesses':32,'accepted80_expanded_payload_equal':80,'final_manifest':False,'final_review_budget':None,'methodology_locked':False}
    (OUT/'expanded_findings.json').write_text(json.dumps(findings,indent=2)+'\n',encoding='utf-8')
    report=['# Expanded v4.2 train/validation diagnostics','2026-10-06. All2200 businesses (2000train/200validation),575749 complete English originals,39600 comparison runs/19800 new-method runs. Test100 excluded; no final2300 source allocation/manifest or model stage.','## Protocol','All003j primary/secondary/search/seed/whole-review/cap30 settings unchanged. Accepted80 witness caches and exact v4.2 scientific payload equality; newly generated v4/v4.1 witness sets for2120 under frozen rules. Retention policy therefore applies to every business. Primary control stays fixed during exact-histogram-equivalent length-distribution tie/refinement. Fresh current v4.2 replay32 uses precommitted new source-based cases and fixed identity-verified witnesses; do not claim all2200 fresh replay.','## Budget and method summary',markdown_table(summary),'Review budgets3500/4000/4500 are scenarios within planned~5000 TOTALsource; no final allocation selected. BODY tokens exclude structural markers/specials, review-block tokens include them once. Short20 is a diagnostic and macro means weight businesses equally. Length matching is not an individual content-quality/aspect label.','## Matched initial reference and selected primary metrics',markdown_table(pd.DataFrame(before_after)),'Before means the immutable initial candidate reference, not an unconstrained optimum or initial search pool size. Per-business improvements/equalities/regressions are counted separately; average gains do not establish every business improves. All19800 new-method initial-reference diagnostics and6600 immutable v3 comparison rows are checked separately in aggregate_acceptance_checks.json.','## Actual global findings',json.dumps(findings,indent=2),'## Minimum and maximum tokens',markdown_table(pd.DataFrame(extremes)),'## Detailed examples',markdown_table(pd.DataFrame(examples)),'Complete selected originals in `complete_original_comparisons.html`; exact limited reading scope/notes are separate. Short opinions can be useful; larger bodies do not establish quality.','## Cross-budget count drops',json.dumps(drop_details,indent=2),'Secondary length selection preserves its primary control count; count drops already present in control are reported rather than forced away. A lower-budget feasible set may cease to satisfy comparability to the new higher-budget quality anchor. Distinguish that quality tradeoff from missing candidates or long-review infeasibility; do not automatically change tolerance/objectives.','## Below10 attribution',json.dumps(flags,indent=2),'If flags exist, the ten-shortest-BODY block is only a constructive feasibility probe. A fitting probe proves ten complete reviews can fit; a failing probe does not prove no ten-review subset fits because formatting/BPE can affect costs. Do not automatically exclude/truncate/increase budget/change method.','## Independent verification',json.dumps(validation,indent=2),'All exported selections checked from originals; exact candidate/frontier/histogram/trace/tolerance invariants and matched initial-reference diagnostics validated. Compatible119 pre-memoization business checkpoints preserved/independently checked; exact task-local token/candidate/JSD memoization alters computation reuse only, frozen scientific modules unchanged. Three bitwise/safety/restoration tests and canonical full-payload equality probes recorded. Source/accepted evidence hashes unchanged.','## Next step','Expanded measurement complete; review any remaining count/representation/length tradeoffs before final selector decision. Allocate review/metadata/caption space within planned total before final2300 manifest; no automatic source length/budget change, test tuning, final freeze, BLIP-2/teacher/train/rental. Update and continue maintained notebook00.']
    (OUT/'expanded_review_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8');print(json.dumps(findings,indent=2))


if __name__=='__main__':main()
