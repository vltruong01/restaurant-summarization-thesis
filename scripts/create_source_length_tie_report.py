"""Prepare full-original comparisons; finalize only after validation and reading."""
from pathlib import Path
import argparse,hashlib,html,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_c/review_selection_source_length_tie_pilot_v1'


def markdown_table(frame):
    def cell(value):
        if isinstance(value,float):value=f'{value:.6g}'
        return str(value).replace('|','\\|').replace('\n',' ')
    header='| '+' | '.join(map(cell,frame.columns))+' |'
    divider='| '+' | '.join(['---']*len(frame.columns))+' |'
    return '\n'.join([header,divider,*['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    payloads=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'business_runs').glob('*.json'))]
    allruns=[r for p in payloads for r in p['runs']];frame=pd.DataFrame(allruns)
    population={}
    for line in (OUT/'pilot_population.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)['review'];population[r['review_id']]=r
    byname={r['name']:r['business_id'] for r in allruns}
    ids=[byname[name] for name in ['El Vez','Water Works Restaurant & Lounge','Taco Grande','Rhone Rum Bar',"Geno's Steaks",'Parkway Bakery & Tavern','Cochon']]
    longest=max(payloads,key=lambda p:max(p['body_lengths'].values()));longbid=longest['business_id']
    if longbid not in ids:ids.append(longbid)
    example_rows=[];read_reviews=[];parts=['<!doctype html><meta charset="utf-8"><title>v4.2 whole-original comparisons</title><style>body{max-width:1100px;margin:2rem auto;font:16px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;margin:.6rem 0}</style><h1>Exploratory v4.2 complete originals</h1><p>Not final model inputs; length is not an individual quality label. All selected originals shown below; assistant reading scope is a separate purposive subset.</p>']
    for bid in ids:
        budget=4500 if bid in [byname['Taco Grande'],byname['Rhone Rum Bar']] else 3500
        rows={r['method']:r for r in allruns if r['business_id']==bid and r['review_budget']==budget};control=rows['v4_2_primary_control'];final=rows['v4_2_length_balanced']
        example_rows.append({'business_id':bid,'name':final['name'],'review_budget':budget,'source_count':final['source_count'],'source_body_mean':final['source_body_mean'],'long_review_stress_case':bid==longbid,'count':final['final_retained_review_count'],'tokens_control':control['tokens_after_budget_enforcement'],'tokens_final':final['tokens_after_budget_enforcement'],'rating_jsd':final['rating_jsd_after'],'time_jsd':final['temporal_jsd_after'],'length_jsd_control':control['length_jsd'],'length_jsd_final':final['length_jsd'],'body_ratio_control':control['selected_source_mean_ratio'],'body_ratio_final':final['selected_source_mean_ratio'],'short20_control':control['selected_lt20_fraction'],'short20_final':final['selected_lt20_fraction']})
        parts.append('<h2>'+html.escape(final['name'])+f' — budget {budget}</h2>')
        for method in ['v4_2_primary_control','v4_2_existing_length_tie','v4_2_length_balanced']:
            row=rows[method];parts.append('<h3>'+method+f" | count {row['final_retained_review_count']} | tokens {row['tokens_after_budget_enforcement']} | length JSD {row['length_jsd']:.6f}</h3>")
            for rid in row['selected_review_ids']:
                r=population[rid];parts.append('<details><summary>'+html.escape(f"{rid} | {r['stars']} stars | {r['date']}")+'</summary><pre>'+html.escape(r['text'])+'</pre></details>')
        for label,chosen,other in [('control_only',control['selected_review_ids'],set(final['selected_review_ids'])),('final_only',final['selected_review_ids'],set(control['selected_review_ids']))]:
            scope=[rid for rid in chosen if rid not in other][:2]
            if not scope and label=='final_only':scope=chosen[:2]
            for rid in scope:read_reviews.append({'business_id':bid,'name':final['name'],'budget':budget,'version':label,'review_id':rid,'body_tokens':next(p['body_lengths'][rid] for p in payloads if p['business_id']==bid),**population[rid]})
    if args.prepare:
        pd.DataFrame(example_rows).to_csv(OUT/'example_businesses.csv',index=False)
        (OUT/'complete_original_comparisons.html').write_text('\n'.join(parts),encoding='utf-8')
        (OUT/'qualitative_reviews_to_read.json').write_text(json.dumps(read_reviews,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'example_businesses':len(ids),'reviews_prepared':len(read_reviews),'long_review_business':longbid}));return
    validation=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'));assert validation['validation_passed']
    notes=json.loads((OUT/'qualitative_review_notes.json').read_text(encoding='utf-8'))
    assert notes['read_reviews']==len(read_reviews) and {x['review_id'] for x in read_reviews}=={rid for c in notes['cases'] for rid in c['read_review_ids']}
    assert all(c['status']=='assistant_read_complete' for c in notes['cases'])
    summary=pd.read_csv(OUT/'budget_method_summary.csv');paired=pd.read_csv(OUT/'paired_primary_secondary.csv');reg=pd.read_csv(OUT/'baseline_regressions.csv')
    drops={}
    for method,g in frame.groupby('method'):
        drops[method]=int(sum(any(b<a for a,b in zip(counts,counts[1:])) for counts in [part.sort_values('review_budget').final_retained_review_count.tolist() for _,part in g.groupby('business_id')]))
    outcomes={'count_drop_businesses':drops,'baseline_dominates_final':reg.groupby('baseline').baseline_dominates_final.sum().astype(int).to_dict(),'secondary_changed_cases':int(paired.changed_from_control.sum()),'secondary_length_jsd_strictly_improved_cases':int((paired.length_jsd_final<paired.length_jsd_primary-1e-12).sum()),'all_primary_histograms_preserved':bool(paired.primary_histograms_preserved.all()),'example_businesses':len(ids),'long_review_business':longbid,'qualitative_reviews_read':len(read_reviews),'final_manifest':False,'final_review_budget':None,'methodology_locked':False}
    (OUT/'pilot_findings.json').write_text(json.dumps(outcomes,indent=2)+'\n',encoding='utf-8')
    report=['# Source-length distribution tie pilot v4.2','2026-10-06. Same80 train/validation businesses;40407 whole English originals. Fresh discovery with explicit accepted candidate retention;720 new-method runs,720 immutable comparisons. Exploratory only; no final allocation/manifest or model stage.','## Rule and controls','Primary control: fresh v4.1 discovery with v3/v4/v4.1 incumbents plus re-encoded accepted v4 ordinary/v4.1 shared candidates. Existing-length tie: minimize source BODY-length-bin base2 JSD only within exact count/star/time-histogram equivalents. Refined tie: up to six same-cell replacement iterations, four proposals exactly encoded each; strict secondary JSD improvement. Whole reviews,cap30,source/time/rating/count rules unchanged. Four empirical bins keep equal lengths together and use actual source proportions. No longest/fullest/semantic/quality/minimum-length rule. Final parameters remain null.','## Measured summary',markdown_table(summary),'## Paired evidence',json.dumps(outcomes,indent=2),'All240 primary-control/final pairs preserve exact count/star/time histograms and rating/time JSD. Improvements versus old baselines combine candidate retention and secondary selection; use primary control versus existing tie versus refinement to distinguish their contributions. Matching source BODY-bin distribution does not establish individual quality/aspect coverage or better generated summaries. Budget and primary-count constraints can retain residual short-review shift. Higher budgets are scenarios, not selected final allocation.','## Detailed cases',markdown_table(pd.DataFrame(example_rows)),'All originals in `complete_original_comparisons.html`; exact purposive reading IDs/observations in `qualitative_review_notes.json`. Reading is not formal human evaluation or source-wide aspect recall.','## Verification',json.dumps(validation,indent=2),'Seven synthetic tests PASS; before computation a test incorrectly assumed exact25% empirical-bin source masses. Test expectation corrected to actual3/4/4/5 masses; production rule unchanged. Full fresh reverse-source payload replay of all80 confirms deterministic discovery/secondary traces/IDs/tokens. Previous completed outputs and original sources remain unchanged.','## Stop and next stage','Pilot implementation/measurement is complete. Do not freeze all2300, tune on test, increase source length or adopt final methodology automatically. Review this pilot including residual3500 bias before an explicitly authorized expanded train/validation diagnostic run; finalize review/metadata/caption allocation before final manifest. Photo integrity/BLIP-2/teacher/LoRA LED remain future stages. GPU is appropriate for later model stages; this tokenizer/combinatorial pilot used CPU.']
    (OUT/'pilot_review_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8');print(json.dumps(outcomes,indent=2))


if __name__=='__main__':main()
