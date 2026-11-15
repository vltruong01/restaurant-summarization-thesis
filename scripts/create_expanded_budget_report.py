"""Report expanded diagnostics and length attribution without freezing inputs."""
from pathlib import Path
import hashlib,html,json
import orjson,pandas as pd,yaml
from create_budget_pilot_report import table

ROOT=Path(__file__).resolve().parents[1]


def main():
    cfg=yaml.safe_load((ROOT/'configs/review_selection_expanded.yaml').read_text(encoding='utf-8'))
    out=ROOT/cfg['output_directory'];source=ROOT/cfg['english_pool_directory']
    completion=json.loads((out/'completion_report.json').read_text());validation=json.loads((out/'validation_report.json').read_text())
    notebook=json.loads((out/'notebook_execution_report.json').read_text())
    assert validation['validation_passed'] and notebook['status']=='PASS'
    data=pd.read_csv(out/'selection_diagnostics.csv');summary=pd.read_csv(out/'budget_method_summary.csv')
    profiles=pd.read_csv(out/'business_length_profiles.csv');sample=pd.read_csv(out/'businesses.csv',keep_default_na=False)
    new=summary.loc[summary['method'].eq('budget_aware')].sort_values('review_budget')
    selected=data.loc[data['method'].eq('budget_aware')];flags=selected.loc[selected['below_10_retained_flag']]
    stress=pd.DataFrame(json.loads((out/'long_review_stress.json').read_text()))
    sections=['# Expanded C2 diagnostics — 2.200 train/validation businesses',
              '2026-10-05. **Đã kiểm tra đầy đủ 2.000 train + 200 validation; chưa freeze final manifest, không tune 100 test.** '
              'Đây là diagnostics mở rộng, không phải supervised training dataset.',
              f'Nguồn **{completion["source_reviews"]:,} original English reviews**, **{completion["selection_runs"]:,} runs** '
              '(2.200 businesses × ba methods × ba budgets). Giữ nguyên `complete_review_search_v3`, seed 42 và scientific settings của pilot; '
              'tối đa 30 complete originals, không cắt review, không minimum-ten input. Total target vẫn ~5.000; review/caption/metadata allocation cuối TBD. '
              'Ba review budgets là exploratory, semantic diagnostic/model chưa tính.',
              '## Review counts và tokens tối thiểu/tối đa',
              table(['Review budget','Mean retained','Min–max reviews','<10 businesses','<5','Infeasible','Selected tokens min–max','Selected tokens mean'],
                    [[int(r.review_budget),f'{r.retained_mean:.2f}',f'{int(r.retained_min)}–{int(r.retained_max)}',
                      f'{int(r.below_10)}/2200 ({r.below_10_percent:.2f}%)',int(r.below_5),int(r.infeasible),
                      f'{int(r.tokens_min)}–{int(r.tokens_max)}',f'{r.tokens_mean:.2f}'] for r in new.itertuples()]),
              '**Selected tokens** là exact assembled review block gồm headers/stars/date/separators + BOS/EOS một lần; '
              'chưa có business metadata/captions. Min/max xét feasible inputs; infeasible không được coi là valid zero-token input.',
              table(['Budget','Reference tokens min–max (before)','p50 after','p90 after','p95 after','p99 after'],
                    [[int(r.review_budget),f'{int(r.reference_tokens_min)}–{int(r.reference_tokens_max)}',
                      f'{r.tokens_p50:.1f}',f'{r.tokens_p90:.1f}',f'{r.tokens_p95:.1f}',f'{r.tokens_p99:.1f}'] for r in new.itertuples()]),
              f'**Individual original review BODY tokens** (không header/special): min **{int(profiles.individual_body_tokens_min.min())}**, '
              f'max **{int(profiles.individual_body_tokens_max.max())}**, mean '
              f'**{profiles.individual_body_tokens_sum.sum()/completion["source_reviews"]:.2f}**. '
              f'Complete singleton block min/max **{int(profiles.complete_singleton_tokens_min.min())} / {int(profiles.complete_singleton_tokens_max.max())}**. '
              'Individual token sums/business chỉ là diagnostic sums, không đồng nhất với assembled input length.',
              table(['Budget','25–30','20–24','15–19','10–14','5–9','<5'],
                    [[int(r['review_budget']),*[int(r[f'band_{b}']) for b in ['25-30','20-24','15-19','10-14','5-9','<5']]] for r in new.to_dict('records')]),
              '## Bao nhiêu cases <10 có liên quan review dài?',
              f'Có **{flags.business_id.nunique()} unique businesses** dưới 10 ở ít nhất một budget theo bộ chọn mới. '
              'Một business xuất hiện ở nhiều scenarios vẫn chỉ tính một lần ở con số unique này; bảng dưới đếm riêng theo budget.',
              table(['Budget','<10 selected','Ten-review probe fits: count not forced by length','Ten-review probe exceeds: observed length pressure'],
                    [[int(r.review_budget),int(r.below_10),int(r.below_ten_probe_fits),int(r.below_ten_probe_exceeds)] for r in new.itertuples()]),
              'Probe lấy mười complete originals có singleton token cost thấp nhất, sắp chronological rồi encode toàn block. '
              'Probe **fits** là chứng cứ có ít nhất một tập mười reviews vừa budget: selected count thấp không bị độ dài bắt buộc, '
              'có thể do rating/time comparability hoặc bounded search. Probe **exceeds** cho thấy áp lực độ dài trong probe, '
              '**không phải chứng minh mọi tập mười reviews đều không vừa**; chi phí phụ thuộc marker-position/boundary effects. '
              'Không gán mọi <10 case cho long reviews hoặc tự tăng source length. Các cases được giữ để review, không loại business.',
              table(['Business ID','Name','Split','Budget','Source count','Retained','Selected tokens','Ten-review probe tokens','Probe fits'],
                    [[r.business_id,r.name,r.split,int(r.review_budget),int(r.initial_eligible_review_count),int(r.final_retained_review_count),
                      r.tokens_after_budget_enforcement,int(r.ten_cheapest_singleton_probe_tokens),r.ten_complete_reviews_demonstrably_fit]
                     for r in flags.sort_values(['review_budget','final_retained_review_count','business_id']).itertuples()]),
              'Tất cả flags: [below_ten_cases_budget_aware.csv](below_ten_cases_budget_aware.csv); '
              'baselines và toàn bộ methods: [below_ten_cases_all_methods.csv](below_ten_cases_all_methods.csv).',
              table(['Longest complete-review diagnostic','Businesses >5.000','Exact tokens min–max'],
                    [[k,int((stress.loc[stress['count'].eq(k),'exact_tokens']>5000).sum()),
                      f'{int(stress.loc[stress["count"].eq(k),"exact_tokens"].min())}–{int(stress.loc[stress["count"].eq(k),"exact_tokens"].max())}'] for k in [7,8]]),
              'Longest-review stress blocks chỉ để diagnostic, không phải selected evidence.',
              '## Rating/time representativeness và baselines',
              table(['Method','Budget','Mean retained','Rating JSD before → after','Temporal JSD before → after'],
                    [[r.method,int(r.review_budget),f'{r.retained_mean:.2f}',f'{r.rating_jsd_before_mean:.6f} → {r.rating_jsd_after_mean:.6f}',
                      f'{r.temporal_jsd_before_mean:.6f} → {r.temporal_jsd_after_mean:.6f}'] for r in summary.sort_values(['review_budget','method']).itertuples()])]
    paired=[]
    for budget in [3500,4000,4500]:
        a=selected.loc[selected['review_budget'].eq(budget)].set_index('business_id')
        b=data.loc[data['method'].eq('rating_time_delete_only')&data['review_budget'].eq(budget)].set_index('business_id').reindex(a.index)
        paired.append([budget,f'{(a.final_retained_review_count-b.final_retained_review_count).mean():.2f}',
                       int((a.rating_jsd_after>b.rating_jsd_after+1e-12).sum()),int((a.temporal_jsd_after>b.temporal_jsd_after+1e-12).sum())])
    sections.extend([table(['Budget','Mean review count gain vs rating/time deletion','Worse rating JSD vs baseline','Worse temporal JSD vs baseline'],paired),
                     'Scores dùng base-2 JSD và năm full-population time rank bins. Average improvements không chứng minh mọi business tốt hơn hoặc giữ đủ aspects/opinions. '
                     'Draft 0,5/0,5 weights và 0,001 comparability mỗi metric chưa phải final quality thresholds; ranking sensitivity lưu trong `tolerance_sensitivity.json`.',
                     '## Mười cases nguyên văn để inspection'])
    rank=selected.loc[selected['review_budget'].eq(3500)].sort_values(['final_retained_review_count','tokens_before_budget_enforcement','business_id'],ascending=[True,False,True])
    ids=flags.sort_values(['final_retained_review_count','business_id'])['business_id'].drop_duplicates().tolist()[:6]
    for bid in rank['business_id']:
        if bid not in ids:ids.append(bid)
        if len(ids)>=6:break
    for bid in profiles.sort_values(['individual_body_tokens_max','business_id'],ascending=[False,True])['business_id']:
        if bid not in ids:ids.append(bid)
        if len(ids)>=8:break
    for bid in selected.sort_values(['tokens_before_budget_enforcement','business_id'],ascending=[False,True])['business_id']:
        if bid not in ids:ids.append(bid)
        if len(ids)==10:break
    return finalize(out,source,ids,sections,sample,selected,completion,validation,notebook)


def finalize(out,source,ids,sections,sample,selected,completion,validation,notebook):
    populations={bid:[] for bid in ids}
    for split in ['train','validation']:
        with (source/'reviews'/f'{split}.jsonl').open('rb') as stream:
            for line in stream:
                r=orjson.loads(line)
                if r['business_id'] in populations:populations[r['business_id']].append(r)
    page=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>Expanded budget cases</title>',
          '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:8px;margin:6px 0}.kept{background:#e9f5ff}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style>',
          '<h1>Expanded C2 — full original text inspection</h1><p>Budget-aware kept reviews at 3500 tokens are highlighted. Ten-review probe is diagnostic only. Manual aspect/opinion audit pending.</p>']
    for bid in ids:
        payload=json.loads((out/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json').read_text(encoding='utf-8'))
        runs=[r for r in payload['runs'] if r['method']=='budget_aware'];first=runs[0];profile=payload['length_profile']
        sections.extend([f'### {first["name"]} — `{bid}`',
                         f'{first["split"]}; {first["initial_eligible_review_count"]} source reviews; ten-review probe **{profile["ten_cheapest_singleton_probe_tokens"]} tokens**.',
                         table(['Budget','Retained','Exact selected tokens','Rating JSD','Temporal JSD','Probe fits','Removed / added vs reference'],
                               [[r['review_budget'],r['final_retained_review_count'],r['tokens_after_budget_enforcement'],
                                 f'{r["rating_jsd_after"]:.6f}' if r['rating_jsd_after'] is not None else 'NA',
                                 f'{r["temporal_jsd_after"]:.6f}' if r['temporal_jsd_after'] is not None else 'NA',r['ten_complete_reviews_demonstrably_fit'],
                                 f'{len(r["removed_vs_reference_ids"])} / {len(r["added_vs_reference_ids"])}'] for r in runs])])
        page.append(f'<h2>{html.escape(first["name"])} — {html.escape(bid)}</h2>')
        page.append(pd.DataFrame(runs)[['review_budget','final_retained_review_count','tokens_after_budget_enforcement','below_ten_length_diagnostic','ten_cheapest_singleton_probe_tokens','rating_jsd_after','temporal_jsd_after']].to_html(index=False,escape=True))
        kept=set(first['selected_review_ids']);probe=set(profile['ten_cheapest_singleton_probe_ids']);ref=set(first['initial_reference_review_ids'])
        page.append('<details><summary>IDs / actual search provenance</summary><pre>'+html.escape(json.dumps(first,ensure_ascii=False,indent=2))+'</pre></details>')
        for r in sorted(populations[bid],key=lambda r:(r['date'],r['review_id'])):
            rid=r['review_id'];label=('KEPT' if rid in kept else 'not kept')+(' | ten-review probe' if rid in probe else '')+(' | reference' if rid in ref else '')
            css=' class="kept"' if rid in kept else ''
            page.append(f'<details{css}><summary>{label} | {html.escape(rid)} | {int(r["stars"])} stars | {html.escape(r["date"])}</summary><pre>{html.escape(r["text"])}</pre></details>')
    page.append('</html>');full='\n'.join(page)
    assert len(ids)==len(set(ids))==full.count('<h2>')==10
    for pop in populations.values():
        for r in pop:assert html.escape(r['text']) in full and html.escape(r['review_id']) in full
    (out/'manual_examples.html').write_text(full,encoding='utf-8');(out/'detail_example_business_ids.json').write_text(json.dumps(ids,indent=2)+'\n',encoding='utf-8')
    sections.extend(['[Mười cases với toàn bộ original text](manual_examples.html). Tạo/xác minh artifact không thay cho researcher manual aspect/opinion audit. '
                     'Full per-business candidates/frontiers/proposals/local moves/IDs/profiles trong `business_runs/`; reference differences không đồng nhất với actual local edits.',
                     '## Validation, CPU/GPU và điểm dừng',
                     f'Independent validation **PASS**: **{validation["source_reviews_verified"]:,} source reviews**, '
                     f'**{validation["selection_runs_verified"]:,} selected/reference runs**, **{validation["frontier_candidates_verified"]:,} frontier candidates**, '
                     f'**{validation["accepted_local_moves_verified"]:,} actual local moves**. All 2.200 ten-review probes checked; '
                     f'fresh reversed-population re-runs on **{validation["fresh_businesses_verified"]} businesses / {validation["fresh_runs_verified"]} runs**, '
                     'including every budget-aware <10 business; individual body lengths checked against hash-verified exhaustive C1 metadata. '
                     f'Notebook 06 **{notebook["executed_code_cells"]}/{notebook["expected_code_cells"]} cells PASS**, three figures visually checked, source metadata unchanged.',
                     f'Current runner elapsed **{completion["elapsed_seconds"]:.1f}s**, independent validation **{validation["elapsed_seconds"]:.1f}s**. '
                     f'Computation origins: `{completion["origins"]}`. Preliminary v1 saved 742 caches before execution was stopped to release completed futures from RAM; '
                     'v2 added exact count memoization, then encountered a Windows reader lock during checkpoint replacement. Current v3 retries transient atomic-write locks and stops workers promptly on errors; '
                     'it verifies all preserved partial identities/cache hashes and reuses complete results, with exact repeated token-count memoization for the remainder. '
                     'This is a CPU optimization, no scientific setting/seed change. Accepted difficult pilot-case payload matches exactly with memoization; '
                     'independent validation uses the actual tokenizer. Local hardware: RTX 3050 6 GB (6144 MiB, driver 596.36); current tokenizer/search use CPU, no GPU model/job/rental launched.',
                     '**Dừng sau expanded diagnostics.** Không freeze manifest cho 2.300 businesses, không tune 100 test, không tự thay allocations/source length hoặc loại businesses. '
                     'Final research parameters remain null. Review flagged cases/content and metadata/caption space before final shared methodology. '
                     'Semantic model/diagnostics, manual aspect/opinion audit, captions/targets/training and Vast.ai A2 remain pending; thesis core/evaluation plans unchanged.'])
    (out/'expanded_diagnostics_report.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print(f'Created {out/"expanded_diagnostics_report.md"}; ten HTML cases contain {sum(map(len,populations.values()))} full originals.')


if __name__=='__main__':main()
