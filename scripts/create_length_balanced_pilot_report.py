"""Verified v3/v4 tradeoffs and full-original examples for unified notebook."""
from pathlib import Path
import hashlib,html,json
import pandas as pd
import orjson,yaml
ROOT=Path(__file__).resolve().parents[1]


def table(columns,rows):
    def clean(v):return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',
                       *['| '+' | '.join(map(clean,r))+' |' for r in rows]])


def main():
    cfg=yaml.safe_load((ROOT/'configs/review_selection_length_balanced_pilot.yaml').read_text(encoding='utf-8'));out=ROOT/cfg['output_directory']
    completion=json.loads((out/'completion_report.json').read_text(encoding='utf-8'));validation=json.loads((out/'validation_report.json').read_text())
    notebook=json.loads((ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json').read_text())
    assert validation['validation_passed'] and notebook['status']=='PASS'
    sample=pd.read_csv(out/'pilot_businesses.csv');data=pd.read_csv(out/'selection_diagnostics.csv');summary=pd.read_csv(out/'budget_method_summary.csv')
    paired=pd.read_csv(out/'paired_v4_v3_differences.csv');audit=ROOT/cfg['length_audit_directory']
    ids=json.loads((audit/'completion_report.json').read_text())['example_business_ids'][:5]
    for col in ['rating_delta','time_delta']:
        for bid in paired.sort_values([col,'business_id'],ascending=[False,True]).business_id:
            if bid not in ids:ids.append(bid)
            if len(ids)>= (7 if col=='rating_delta' else 9):break
    for bid in data.loc[data.method.eq('v4_length_balanced')].sort_values(['final_retained_review_count','business_id']).business_id:
        if bid not in ids:ids.append(bid)
        if len(ids)==10:break
    assert len(set(ids))==10
    populations={bid:[] for bid in ids}
    for line in (out/'pilot_population.jsonl').open('rb'):
        r=orjson.loads(line)['review']
        if r['business_id'] in populations:populations[r['business_id']].append(r)
    page=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>V3/V4 pilot original reviews</title>',
          '<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:8px;margin:6px 0}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}</style>',
          '<h1>V3 / V4 — full-original review inspection</h1><p>Exploratory pilot only. No frozen training evidence. Compare complete originals; length alone is not quality.</p>']
    sections=['# C2 v4 pilot — giảm ưu tiên shortest reviews',
              '2026-10-05. **Pilot80 businesses đã được kiểm chứng; bản mới giảm rõ thiên về review ngắn nhưng có đánh đổi về count/rating/time. Chưa freeze methodology hoặc manifest.** Notebook chính dùng từ nay: [00_thesis_dataset_filtering_review_selection.ipynb](../../../notebooks/00_thesis_dataset_filtering_review_selection.ipynb).',
              '**Phạm vi:**80 businesses (69train/11validation),40.407 original detector-English reviews,960 runs của bốn methods × ba budgets. Union accepted56 stress cases/ten length-audit cases +16 seeded cases (4/source-count band); overlap cho union64 trước16 thêm. Pilot purposive/stress-plus-stratified, không phải whole-cohort quality estimate. Không dùng test.',
              '## Thay đổi có chủ đích',
              'Giữ primary rating/time JSD, complete reviews, count preference khi representation tương đương, cap30, seed42 và exact LED budget3500/4000/4500. Thay low-cost seed defaults bằng uniform/length-stratum/joint-uniform candidates; incoming offers từ bốn length strata +uniform. Ties sau rating/time/count dùng seed-derived hash không phụ thuộc tokens. Minimum-cost singleton chỉ làm rescue nếu ordinary feasible seeds không có. Các delete-only baselines giữ nguyên từ accepted v3. Không thêm minimum review length, semantic objective hoặc token-filling objective.',
              'Đây là combined revision, chưa ablation từng cơ chế; constraints vẫn có thể ưu tiên short reviews để fit nhiều complete reviews. No global optimum/content-quality guarantee. ~5000 vẫn planned total source target; metadata/caption/final review allocation TBD.',
              '## Kết quả v3/v4 trên cùng80 businesses',
              table(['Method','Budget','Mean complete reviews','Count min–max','<10/<5/infeasible','Median selected/source mean BODY ratio','Mean length percentile','Selected <20 BODY fraction (macro)','Rating JSD mean','Time JSD mean'],
                    [[r.method,int(r.review_budget),f'{r.retained_mean:.2f}',f'{int(r.retained_min)}–{int(r.retained_max)}',f'{int(r.below10)}/{int(r.below5)}/{int(r.infeasible)}',
                      f'{r.median_mean_ratio:.3f}',f'{r.mean_length_percentile:.3f}',f'{100*r.selected_short20_macro:.2f}%',f'{r.rating_jsd_mean:.6f}',f'{r.temporal_jsd_mean:.6f}']
                     for r in summary.loc[summary.method.isin(['v3_budget_aware','v4_length_balanced'])].sort_values(['review_budget','method']).itertuples()]),
              'BODY lengths exclude markers/special tokens; ratios computed within business then summarized. Macro fractions/means give each business equal weight. Selected occurrences across budgets overlap; not independent samples. At3500 median ratio rises0.417→0.808 and short20 fraction drops13.53%→1.35%; source short20 fraction1.05%. Exactly30 inputs under1500: v3=11 per scenario, v4=0. Descriptive thresholds are not quality/exclusion rules.',
              table(['Budget','Count delta mean (v4 − v3)','Businesses lower count','Worse rating JSD','Worse time JSD','Mean BODY length gain'],
                    [[int(b),f'{g.count_delta.mean():.2f}',int(g.count_delta.lt(0).sum()),int(g.rating_delta.gt(1e-12).sum()),int(g.time_delta.gt(1e-12).sum()),f'{g.body_mean_delta.mean():.2f}'] for b,g in paired.groupby('review_budget')]),
              '**Đánh đổi thực tế:** ở3500 mean count26.53→24.04; rating/time means kém hơn v3 dù absolute JSD còn nhỏ. Không gán quality acceptance threshold mới. 31/80 rating regressions và16/80 temporal regressions ở3500; phải xem từng case, không tuyên bố new method tốt hơn mọi mặt. Ở4500 count gap nhỏ hơn nhưng không chọn4500 làm final budget từ pilot này.',
              '## Bins, token min/max và rescue']
    new=summary.loc[summary.method.eq('v4_length_balanced')]
    sections.append(table(['Budget','25–30','20–24','15–19','10–14','5–9','<5','Block tokens min–max','Rescue used'],
                      [[int(r['review_budget']),*[int(r['band_'+b]) for b in ['25-30','20-24','15-19','10-14','5-9','<5']],f'{int(r["tokens_min"])}–{int(r["tokens_max"])}',int(r['rescue_used'])] for r in new.to_dict('records')]))
    sections+=['Không minimum10 input: min10 là observed pilot outcome; synthetic test kiểm tra rare-only fit giữ1 review. Zero rescue ở real pilot không bỏ singleton guarantee. <10/<5 flags không tự loại business hoặc tăng budget.',
               '## El Vez và các cases nguyên văn',
               'El Vez ở3500 vẫn30 originals, tokens1154→3420, BODY mean14.43→89.97; rating/time JSD **giống nhau**. Ví dụ đọc có thêm trải nghiệm cụ thể về portion size, food consistency, service và noise, thay vì chỉ những câu rất ngắn. Đây là illustrative inspection, không semantic/aspect recall score hoặc formal human evaluation.',
               'Assistant đã đọc có chủ đích first3+last2 complete v4 originals ở3500 của El Vez/Mother’s/Zahav (15 reviews): Mother’s có texture/ham/recipe-change/check-in/counter-order details; Zahav có detailed mixed dish/value/wine opinions và outdoor-dining details. Review dài có cả phần ít liên quan; length không chứng minh quality. Toàn bộ selected originals của cả versions/budgets có trong HTML; chưa đọc hết nguồn hoặc toàn pilot.']
    qualitative=[]
    for bid in ids:
        payload=json.loads((out/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json').read_text(encoding='utf-8'))
        by_id={r['review_id']:r for r in populations[bid]};name=next(r['name'] for r in payload['runs'])
        sections+=[f'### {name} — `{bid}`',table(['Version','Budget','Count','Block tokens','Mean BODY','Rating JSD','Time JSD'],
                   [[r['method'],r['review_budget'],r['final_retained_review_count'],r['tokens_after_budget_enforcement'],f'{r["selected_body_mean"]:.2f}',f'{r["rating_jsd_after"]:.6f}',f'{r["temporal_jsd_after"]:.6f}']
                    for r in payload['runs'] if r['method'] in ['v3_budget_aware','v4_length_balanced']])]
        page.append(f'<h2>{html.escape(name)} | {bid}</h2>')
        for budget in [3500,4000,4500]:
            for method in ['v3_budget_aware','v4_length_balanced']:
                r=next(r for r in payload['runs'] if r['method']==method and r['review_budget']==budget)
                page.append(f'<details><summary>{method} budget={budget}: {r["final_retained_review_count"]} complete reviews, {r["tokens_after_budget_enforcement"]} tokens</summary>')
                for rid in r['selected_review_ids']:
                    original=by_id[rid]
                    page.append(f'<details><summary>{rid} | {int(original["stars"])} stars | {original["date"]}</summary><pre>{html.escape(original["text"])}</pre></details>')
                page.append('</details>')
        if bid in ['9PZxjhTIU7OgPIzuGi89Ew','iSRTaT9WngzB8JJ2YKJUig','ctHjyadbDQAtUFfkcAFEHw']:
            r=next(r for r in payload['runs'] if r['method']=='v4_length_balanced' and r['review_budget']==3500)
            qualitative.append({'business_id':bid,'name':name,'review_budget':3500,'read_review_ids':r['selected_review_ids'][:3]+r['selected_review_ids'][-2:],
                                'scope':'Assistant read these15 complete originals in total; purposive illustrative inspection, not human evaluation or exhaustive aspect audit.'})
    page.append('</html>');full='\n'.join(page)
    assert full.count('<h2>')==10
    for bid in ids:
        payload=json.loads((out/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json').read_text(encoding='utf-8'))
        by_id={r['review_id']:r for r in populations[bid]}
        for r in payload['runs']:
            if r['method'] in ['v3_budget_aware','v4_length_balanced']:
                for rid in r['selected_review_ids']:assert html.escape(by_id[rid]['text']) in full
    (out/'review_comparison_examples.html').write_text(full,encoding='utf-8')
    (out/'qualitative_inspection_scope.json').write_text(json.dumps(qualitative,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    progress=json.loads((out/'run_progress.json').read_text())
    sections+=['[Ten full-original comparison cases](review_comparison_examples.html). Notebook00 business viewer also shows raw/English removed/English pool/v3/v4 in one place. Outside the80-business v4 pilot no v4 selection is claimed.',
               '## Limits và bước tiếp theo',
               'Bounded search/candidate pools differ by budget, so count need not increase monotonically with budget. El Vez has30 at3500,29 at4000,30 at4500 despite feasible alternatives; record as candidate-coverage limitation, not a source eligibility failure. Review cross-budget candidate coverage/quality tradeoffs before final methodology. No automatic source-length increase, minimum-length filter or full-cohort rerun.',
               'Review v4 cases and tradeoffs, then decide whether to refine the bounded search or authorize an expanded diagnostics sweep. Keep final allocation/manifest pending; do not create training targets from provisional selected evidence. External/generalization and representative30-business human evaluation plans unchanged.',
               '## Verification và execution history',
               f"Independent validation PASS:40.407 source originals matched;{validation['runs_verified']} exact run token/JSD/length/flags/provenance checks;{validation['frontier_candidates_verified']} frontier candidates;{validation['local_moves_verified']} accepted local moves;{validation['sensitivity_choices_verified']} same-frontier choices. Fresh reversed-source all80 payloads/240 v4 selections match exactly, including baselines/previews/provenance. Six meaningful synthetic tests PASS.",
               f"Unified notebook00 {notebook['executed_code_cells']}/{notebook['expected_code_cells']} cells PASS;{notebook['image_outputs']} plots; sources protected. Local CPU stage; RTX3050 remains available for suitable model stages, no GPU/model/training/rental job.",
               f"Initial80 per-business computations completed in about{progress['elapsed_seconds']:.1f}s, then aggregate read hit Windows default cp1252 decoding on non-ASCII text. Preserve all caches and rerun driver with PYTHONUTF8=1; zero cases recomputed. Warm aggregate/report pass{completion['elapsed_seconds']:.3f}s excludes initial compute; scientific settings/IDs unchanged. Validator uses explicit UTF8 and fresh full rerun; elapsed{validation['elapsed_seconds']:.3f}s. No earlier incomplete aggregates are accepted as evidence.",
               'Artifacts:config/spec003f, versioned v4 implementation/tests/runner/validator, source population/caches/identity, paired differences/bins/flags/sensitivity and notebook00. Prior v3/core scripts/config/artifacts preserved; final review/methodology research fields remain null.']
    (out/'pilot_review_report.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print('v4 report and ten complete-original comparison examples generated.')


if __name__=='__main__':main()
