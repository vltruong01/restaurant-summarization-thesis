"""Write a factual pilot review from accepted outputs; no selection is rerun."""
from pathlib import Path
import json
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    clean = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(clean, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(clean, row)) + ' |' for row in rows]])


def main():
    cfg = yaml.safe_load((ROOT / 'configs/review_selection_budget_pilot.yaml').read_text(encoding='utf-8'))
    out = ROOT / cfg['output_directory']
    completed = json.loads((out / 'completion_report.json').read_text())
    validation = json.loads((out / 'validation_report.json').read_text())
    notebook = json.loads((out / 'notebook_execution_report.json').read_text())
    assert validation['validation_passed'] and notebook['status'] == 'PASS'
    assert completed['test_businesses'] == 0 and completed['final_review_budget'] is None
    runs = [json.loads(line) for line in (out / 'selection_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    frame = pd.DataFrame(runs)
    sample = pd.read_csv(out / 'pilot_businesses.csv', keep_default_na=False)
    summary = pd.read_csv(out / 'budget_method_summary.csv')
    stress = pd.DataFrame([json.loads(line) for line in (out / 'long_review_stress.jsonl').read_text().splitlines()])
    sensitivity = pd.read_json(out / 'tolerance_sensitivity.json')
    new = summary.loc[summary['method'].eq('budget_aware')].sort_values('review_budget')
    sections = [
        '# Báo cáo pilot C2: chọn complete reviews dưới token budget',
        f'Ngày 2026-10-05; algorithm `{cfg["algorithm_version"]}`. **Pilot triển khai và kiểm chứng; final methodology/manifest chưa khóa.**',
        '## Phạm vi và phương pháp',
        f'Mẫu gồm **{completed["businesses"]} businesses**, {int(sample["split"].eq("train").sum())} train / '
        f'{int(sample["split"].eq("validation").sum())} validation, **{completed["population_reviews"]:,} original English reviews**, '
        f'**{completed["selection_runs"]} runs** (3 methods × 3 budgets × 56). '
        'Giữ 48 diagnostic cases v1, thêm 8 low-volume cases (≤100 English reviews) có tổng độ dài tám review dài nhất cao nhất trong train/validation còn lại. '
        'Đây là stress sample có chủ đích, không phải mẫu ngẫu nhiên để ước lượng tỷ lệ toàn cohort. Không tune trên test.',
        'Review-block scenarios **3.500 / 4.000 / 4.500 tokens** là exploratory. Tổng LED source target vẫn **~5.000**; '
        'allocation review/metadata/caption chưa chốt. Đếm nguyên block bằng pinned LED tokenizer, gồm review markers/stars/date/separators và BOS/EOS một lần. '
        'Chưa ghép business metadata/captions, nên chưa chứng minh final multimodal input vừa target. Không cắt text, tối đa 30 review; không đặt minimum 10 input reviews.',
        'Search xét mọi English review làm nguồn, ba dạng seeds cho k=1..min(N,30), cộng minimum-cost singleton để bảo đảm tìm được input nếu có ít nhất một complete review vừa budget. '
        'Local search có additions/deletions/replacements và shortlist theo star/time cells. Giới hạn 3 anchors, 6 iterations, 4 exact proposals/ranking; đây là heuristic có giới hạn, không bảo đảm optimum toàn cục. '
        'Rating/time JSD dùng base 2 và năm time rank bins cố định theo full pool. Anchor có cost 0,5 rating JSD + 0,5 temporal JSD; '
        'trong các candidates không kém anchor quá 0,001 ở từng metric, ưu tiên count lớn hơn. Các số này là draft settings, không phải quality acceptance thresholds. '
        'Semantic centroid chưa tính/model TBD, không tối ưu semantic diversity.',
        '## Retained counts và flags',
        table(['Review budget', 'Mean retained', 'Minimum', '<10', '<5', 'Infeasible'],
              [[int(r.review_budget), f'{r.retained_count_mean:.2f}', int(r.retained_count_min),
                f'{int(r.below_10)}/56', f'{int(r.below_5)}/56', int(r.infeasible)] for r in new.itertuples()]),
        table(['Budget', '25–30', '20–24', '15–19', '10–14', '5–9', '<5'],
              [[int(r['review_budget']), *[int(r[f'band_{b}']) for b in ['25-30','20-24','15-19','10-14','5-9','<5']]] for r in new.to_dict('records')]),
        'Tỷ lệ thấp chỉ áp dụng mẫu này. Kết quả ≥10 không phải ràng buộc hard minimum; synthetic long-review test giữ <5 review vẫn feasible và không loại business. '
        'Nếu pilot thật xuất hiện <10/<5, các flags và diagnostics vẫn được giữ; không tự đổi source length/allocation.',
        '## Baselines và rating/time representativeness',
        'Before là rating/time reference selection (≤30) chưa ràng buộc budget. Nó là mốc so sánh, không phải một thao tác bắt buộc mà search phải thực hiện. '
        'Các delete-only baselines giới hạn vào initial reference; budget-aware search được thêm/thay từ full pool.',
        table(['Budget', 'Method', 'Mean retained', 'Rating JSD after', 'Temporal JSD after'],
              [[int(r.review_budget), r.method, f'{r.retained_count_mean:.2f}', f'{r.rating_jsd_after_mean:.6f}',
                f'{r.temporal_jsd_after_mean:.6f}'] for r in summary.sort_values(['review_budget','method']).itertuples()]),
        table(['Budget', 'Rating before → after', 'Temporal before → after'],
              [[int(r.review_budget), f'{r.rating_jsd_before_mean:.6f} → {r.rating_jsd_after_mean:.6f}',
                f'{r.temporal_jsd_before_mean:.6f} → {r.temporal_jsd_after_mean:.6f}'] for r in new.itertuples()]),
    ]
    rows = []
    for budget in cfg['budget']['exploratory_review_token_scenarios']:
        a = frame.loc[frame['review_budget'].eq(budget) & frame['method'].eq('budget_aware')].set_index('business_id')
        b = frame.loc[frame['review_budget'].eq(budget) & frame['method'].eq('rating_time_delete_only')].set_index('business_id').reindex(a.index)
        rows.append([budget, f'{(a.final_retained_review_count-b.final_retained_review_count).mean():.2f}',
                     int((a.rating_jsd_after > b.rating_jsd_after + 1e-12).sum()),
                     int((a.temporal_jsd_after > b.temporal_jsd_after + 1e-12).sum())])
    sections.extend([table(['Budget', 'Mean count gain vs rating/time delete-only', 'Businesses worse rating JSD', 'Businesses worse time JSD'], rows),
                     'Mean improvement không chứng minh mọi business tốt hơn ở mọi metric; bảng paired regressions ghi rõ các trường hợp kém baseline. Full per-business values nằm trong `selection_diagnostics.csv`.'])
    sections.extend(['## Long-review stress và tolerance sensitivity',
                     table(['Longest complete reviews', 'Businesses >5.000 tokens', 'Maximum exact block tokens'],
                           [[k, f'{int((stress.loc[stress["count"].eq(k), "exact_tokens"] > 5000).sum())}/56',
                             int(stress.loc[stress['count'].eq(k), 'exact_tokens'].max())] for k in [7,8]]),
                     'Đây là diagnostic ghép các review dài nhất, không phải selected model input. Nó xác nhận edge case 7–8 review vượt target có tồn tại. Search có thể chọn complete reviews khác để vừa budget.',
                     table(['Budget', 'Tolerance mỗi metric', 'Mean count', 'Min count'],
                           [[int(b), f'{t:.3f}', f'{g["count"].mean():.2f}', int(g['count'].min())]
                            for (b,t), g in sensitivity.groupby(['review_budget','tolerance_per_metric'])]),
                     'Sensitivity chỉ thay cách rank trên SAME searched frontier; chưa chứng minh search robustness khi thay search strategy hoặc compute budget.',
                     '## Mười ví dụ chi tiết'])
    ids = json.loads((out / 'detail_example_business_ids.json').read_text())
    for bid in ids:
        group = [r for r in runs if r['business_id'] == bid and r['method'] == 'budget_aware']
        reference = group[0]
        longest = stress.loc[stress['business_id'].eq(bid) & stress['count'].eq(8)].iloc[0]
        sections.extend([f'### {reference["name"]} — `{bid}`',
                         f'{reference["split"]}; {reference["sample_group"]}; {reference["initial_eligible_review_count"]} English reviews; '
                         f'initial reference {reference["initial_candidate_count"]} reviews/{reference["tokens_before_budget_enforcement"]} tokens; '
                         f'eight-longest block {int(longest.exact_tokens)} tokens.',
                         table(['Budget', 'Retained', 'Exact tokens', 'Rating JSD', 'Time JSD', 'Removed vs reference', 'Added vs reference', 'Accepted local moves'],
                               [[r['review_budget'],r['final_retained_review_count'],r['tokens_after_budget_enforcement'],
                                 f'{r["rating_jsd_after"]:.6f}',f'{r["temporal_jsd_after"]:.6f}',len(r['removed_vs_reference_ids']),
                                 len(r['added_vs_reference_ids']),len(r['search_provenance']['accepted_local_path'])] for r in group])])
    sections.extend(['Nguyên văn full population, highlights kept reviews (3.500-token scenario), reference differences và accepted paths: '
                     '[manual_examples.html](manual_examples.html). Toàn bộ IDs, texts và provenance của mọi scenario: '
                     '`selection_runs.jsonl`, `model_input_previews.jsonl`, `business_runs/`. Reference-to-final differences là set differences, '
                     'không được đồng nhất với các local edits thực sự đã chạy. **Manual aspect/opinion coverage audit chưa hoàn thành**; tạo ví dụ không có nghĩa đã audit.',
                     '## Kiểm chứng và điểm dừng',
                     f'Independent validation PASS: **{validation["source_reviews_verified"]:,}** original records/IDs/fields/text, '
                     f'**{validation["selection_runs_verified"]}** exact token/JSD/count/order/budget runs, '
                     f'**{validation["fresh_deterministic_runs_verified"]}** fresh no-cache reproducibility comparisons, '
                     f'**{validation["frontier_candidates_verified"]}** frontier candidates và **{validation["accepted_local_moves_verified"]}** accepted local moves. '
                     'Sample/stress ranking, tokenizer/source/script/config identities, flags/histograms và tolerance winners checked. '
                     'Six edge tests PASS, gồm rare-rating-only fit và long-review <5 retained. Notebook 05 chạy 8/8 code cells, bảo toàn source metadata; ba PNG figures được kiểm tra trực quan.',
                     'Evidence: `completion_report.json`, `validation_report.json`, `notebook_execution_report.json`; '
                     '[notebook 05](../../../notebooks/05_budget_constrained_review_selection_pilot.ipynb), `figures/`. '
                     f'Pilot {completed["elapsed_seconds"]:.1f}s; independent validation {validation["elapsed_seconds"]:.1f}s. '
                     'V2 preliminary outputs được giữ làm lịch sử, validation bị dừng trước acceptance do thiếu minimum-cost singleton; V3 là evidence hiện hành.',
                     'Pilot chưa cho thấy cần tự tăng source target ~5.000. Tiếp theo đọc examples kiểm tra coverage của aspects/opinions, '
                     'vì lựa chọn review ngắn hơn trong cùng rating/time group có thể bỏ bớt chi tiết nội dung. Semantic centroid nếu bật chỉ là diagnostic và cần ghi model/config. '
                     'Sau đó kiểm tra metadata/caption space để chốt final shared allocation/methodology trước all-cohort selection. '
                     '**Dừng sau pilot**: chưa freeze selected-review manifest cho 2.300 businesses, chưa thay final pending parameters, source/split/core thesis, chưa training/rental GPU.'])
    (out / 'pilot_review_report.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    print(f'Created {out / "pilot_review_report.md"}')


if __name__ == '__main__':
    main()
