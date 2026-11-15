"""Report verified length shift and bounded assistant content inspection."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_c/review_selection_length_audit_v1'


def table(columns, rows):
    def clean(v): return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |',
                       *['| '+' | '.join(map(clean, r))+' |' for r in rows]])


def main():
    completion = json.loads((OUT/'completion_report.json').read_text())
    validation = json.loads((OUT/'validation_report.json').read_text())
    notebook = json.loads((OUT/'notebook_execution_report.json').read_text())
    assert validation['validation_passed'] and notebook['status']=='PASS'
    data = pd.read_csv(OUT/'business_length_comparisons.csv')
    summary = pd.read_csv(OUT/'length_summary.csv')
    new = summary.loc[summary.method.eq('budget_aware')].sort_values('review_budget')
    examples = json.loads((OUT/'inspection_examples.json').read_text(encoding='utf-8'))
    notes = json.loads((OUT/'qualitative_inspection.json').read_text(encoding='utf-8'))
    cf = pd.read_csv(OUT/'counterfactual_summary.csv')
    sections = ['# C2 — Kiểm tra thiên về review ngắn và chất lượng nội dung',
        '2026-10-05. **Đã xác minh xu hướng chọn review ngắn trên toàn 2.200 train/validation. Chưa nên freeze bộ review được chọn.** Cohort/nguồn và acceptance về integrity/token/rating/time vẫn hợp lệ; chúng không chứng minh đủ thông tin để viết summary.',
        '**Phạm vi:** 2.000 train + 200 validation, 575.749 detector-English originals, 19.800 runs của ba methods × ba budgets. Không tune 100 test. Giữ nguyên accepted expanded v3; không chọn lại model evidence, thêm length gate, đổi budget/source length, chọn semantic model hoặc freeze manifest. Counterfactuals chỉ là diagnostics.',
        '## Kết quả chính — so sánh trong từng business',
        table(['Review budget','Mean body tokens: source → selected (macro)','Median selected/source mean ratio','Businesses có mean ratio <0,5','Giữ đúng 30 và <1.500 block tokens','Giữ đúng 30 và <2.000'],
              [[int(r.review_budget),f'{r.source_mean_macro:.2f} → {r.selected_mean_macro:.2f}',
                f'{r.selected_source_mean_ratio_median:.3f}',f'{int(r.mean_ratio_below_0_5)}/2200 ({100*r.mean_ratio_below_0_5/2200:.2f}%)',
                int(r.exactly30_under1500),int(r.exactly30_under2000)] for r in new.itertuples()]),
        '**Macro**: tính mean review length trong mỗi business rồi trung bình giữa 2.200 businesses, mỗi business cùng trọng số. Vì vậy source macro 126,72 khác pooled source mean 130,21 ở báo cáo trước. Median ratio ~0,383 nghĩa là một nửa businesses có selected mean không quá khoảng 38,3% mean của chính nguồn business đó; không phải tỷ lệ giữa hai macro means. Pooled selected means trong CSV đếm review occurrences riêng từng method/budget; cùng review có thể xuất hiện ở nhiều budgets, không phải independent samples.',
        table(['Budget','Mean within-business length percentile','Source <20 body tokens (macro)','Selected <20 (macro)','Exactly-30 denominator'],
              [[int(r.review_budget),f'{r.selected_midrank_percentile_macro:.3f}',f'{100*r.source_short20_macro:.2f}%',
                f'{100*r.selected_short20_macro:.2f}%',int(r.exactly30)] for r in new.itertuples()]),
        'Length percentile dùng midrank khi bằng nhau. Mean selected percentile ~0,225–0,229 cho thấy review được chọn nằm về phía ngắn của chính source distribution. Các ngưỡng <20/30/50 body tokens, mean ratio <0,5/0,75 và <1.500/2.000 block tokens là exploratory descriptive cutoffs, không phải quality labels hoặc exclusion rules. Length JSD dùng bins <20 / 20–49 / 50–99 / 100–199 / 200+; không phải semantic score.',
        '## Baselines và nguồn có nhiều review',
        table(['Method','Budget','Selected mean BODY tokens (macro)','Median selected/source mean ratio','Mean length percentile'],
              [[r.method,int(r.review_budget),f'{r.selected_mean_macro:.2f}',f'{r.selected_source_mean_ratio_median:.3f}',f'{r.selected_midrank_percentile_macro:.3f}'] for r in summary.itertuples()])]
    pairs = pd.read_csv(OUT/'paired_baseline_comparisons.csv')
    sections += [table(['Budget','Baseline','Comparison','Businesses','New method shorter mean','Mean BODY difference (new − baseline)'],
                       [[int(r.review_budget),r.baseline,r.comparison,int(r.businesses),int(r.new_shorter_mean),f'{r.mean_body_length_difference:.2f}'] for r in pairs.itertuples()]),
        'Same-count comparison chỉ giữ các business mà hai methods có cùng số review; chênh lệch vẫn còn nên không chỉ là phép chia cùng budget cho nhiều review hơn. Đây vẫn là quan sát, không phải random intervention hoặc chứng minh content quality tương đương.',
        table(['Budget','Source count band','Businesses','Median selected/source mean ratio','Mean ratio <0,5'],
              [[int(r.review_budget),r.volume_band,int(r.businesses),f'{r.selected_source_mean_ratio_median:.3f}',int(r.mean_ratio_below_0_5)]
               for r in pd.read_csv(OUT/'length_summary_by_volume.csv').query("method=='budget_aware'").itertuples()]),
        'Ở 3.500 tokens, cả 588 businesses trong nhóm >300 English reviews có selected mean dưới một nửa source mean; median ratio nhóm này ~0,220. Nguồn lớn có nhiều short alternatives hơn; low-volume businesses thường giữ nguyên toàn pool. Tăng budget từ 3.500 lên 4.500 chỉ thay đổi rất ít các length ratios, nên kết quả hiện tại không ủng hộ tăng source length như một cách sửa vấn đề này.',
        '## Đọc năm tập 30-review ít tokens nhất',
        'Assistant đã đọc **150 selected originals** (30/case, budget 3500) và **10 longer original alternatives** (hai/case). Đây là purposive qualitative diagnostic, chưa phải human evaluation, chưa đọc toàn bộ source population, chưa đo aspect recall hoặc inter-rater agreement. Mười examples đầy đủ cho ba budgets được lưu trong JSON/HTML.',
        table(['Business','Source reviews','Selected block tokens','Selected body mean','Source body mean','Same-cell five replacements: block tokens'],
              [[c['name'],c['source_count'],int(data.loc[data.business_id.eq(c['business_id']) & data.method.eq('budget_aware') & data.review_budget.eq(3500),'selected_block_tokens'].iloc[0]),
                f"{data.loc[data.business_id.eq(c['business_id']) & data.method.eq('budget_aware') & data.review_budget.eq(3500),'selected_mean'].iloc[0]:.2f}",
                f"{data.loc[data.business_id.eq(c['business_id']) & data.method.eq('budget_aware') & data.review_budget.eq(3500),'source_mean'].iloc[0]:.2f}",
                int(cf.loc[cf.business_id.eq(c['business_id']) & cf.review_budget.eq(3500),'counterfactual_block_tokens'].iloc[0])]
               for c in examples[:5]])]
    observations_vi = {
        'El Vez': 'Có ý cụ thể về margaritas, guacamole, món ăn, giá và tiếng ồn, nhưng cũng có praise chung chung. Longer alternatives kể chi tiết xử lý phàn nàn và phục vụ dịp sinh nhật; các sự kiện này không được coi là phổ biến chỉ vì xuất hiện trong ví dụ.',
        "Mother's Restaurant": 'Giữ được ý về roast beef/po-boy, giá, vệ sinh, waits và service. Một số review chỉ khuyên nên đến hoặc khen breakfast. Longer alternatives mô tả chicken po-boy nhạt/thiếu condiments và vị cay của jambalaya; cùng star rating vẫn có thể chứa mixed opinions.',
        "Commander's Palace": 'Nhiều lời khen lặp lại về food/service, nhưng review ngắn vẫn có thông tin turtle soup lạnh, reservations, dress code. Longer alternatives thêm món cụ thể, giá và cách đặt bàn.',
        'Zahav': 'Có thông tin hummus/salatim/kibbe, vegetables, reservations và wine price, xen các superlatives/overrated judgments ít giải thích. Longer alternatives mô tả lamb shoulder portions, restaurant-week value và drink options.',
        "Geno's Steaks": 'Review ngắn vẫn có texture, queue và cash-only information. Có một original chỉ là URL (`XGDbX0FXRgTxiMgKek8qTA`) dù detector gắn en; thêm một số nhận xét ít liên quan/thiếu giải thích. Đây là content/automatic-label concern cần review, không tự đổi language filter.'
    }
    for note in notes['cases']:
        sections += [f"### {note['name']} — `{note['business_id']}`", observations_vi[note['name']],
            f"Ví dụ ít thông tin: {', '.join('`'+x+'`' for x in note['generic_or_low_detail_example_ids'])}. "
            f"Ví dụ ngắn nhưng hữu ích: {', '.join('`'+x+'`' for x in note['useful_short_example_ids'])}."]
    sections += ['[Mười examples nguyên văn](inspection_examples.html) · [JSON originals/provenance](inspection_examples.json) · [Qualitative notes](qualitative_inspection.json). Đọc ví dụ cho thấy thông tin khác nhau; chưa chứng minh every longer review tốt hơn hoặc định lượng số aspects đã mất.',
        '## Review ngắn có bị token budget bắt buộc không?',
        'Trong năm low-token cases, thay năm review ngắn bằng longer complete originals cùng star và cùng global temporal rank bin tạo blocks **1.591–1.768 tokens**, đều dưới budget 3.500. Giữ nguyên 30 reviews và joint star/time-bin histogram nên rating/time JSD không đổi. Các alternatives này chứng minh có một số longer sets vừa budget; không chứng minh đây là tập tốt nhất. Finer time coverage, dates và content thay đổi. Search chỉ thử tối đa 20 offers mỗi replacement; không phải exhaustive optimizer.',
        table(['Business','Budget','Replacements','Original block','Counterfactual block','Original body sum','Counterfactual body sum'],
              [[r.business_id,int(r.review_budget),int(r.replacements),int(r.original_block_tokens),int(r.counterfactual_block_tokens),int(r.original_body_sum),int(r.counterfactual_body_sum)] for r in cf.itertuples()]),
        '**Không thay selected evidence:** counterfactuals được lưu riêng và không dùng để train/tạo targets. Kiểm tra 30 sets, gồm cases không có replacement, không coi zero replacements là bằng chứng impossibility.',
        '## Diễn giải và bước tiếp theo',
        'Accepted search có minimum-cost seed, low-cost seeds/candidate offers trong star/time cells và token tie-breaking. Những cơ chế này có thể giúp giữ nhiều reviews nhưng cũng ưu tiên short originals khi count/rating/time tương đương. Observed toàn-corpus shift và same-cell counterfactuals hỗ trợ concern đó; chưa có ablation để định lượng đóng góp riêng của mỗi cơ chế. Semantic diversity không được dùng làm objective.',
        '**Cần review một sửa đổi candidate construction/tie-breaking trước pilot mới**, để tránh chọn shortest reviews mặc định khi budget còn đủ, vẫn giữ complete boundaries, rating/time objectives và count preference. Có thể thử candidate lengths đa dạng hơn và tie-break theo seed thay vì tiếp tục tối thiểu tokens. Đây chỉ là hướng thử nghiệm pending spec/user review; chưa tự áp dụng cutoff, semantic objective, token-filling objective hoặc chọn phương pháp cuối. Không cần lấy lại cohort chỉ vì kết quả này.',
        'Dừng trước final manifest/methodology. Quality concern thuộc selected-review inputs; cohort/source integrity vẫn valid. Metadata/caption allocation và tổng source length cuối vẫn TBD. Không tự loại business hoặc tăng source length; local RTX 3050 6 GB đã ghi nhận, CPU phù hợp cho bước này.',
        '## Xác minh',
        f"Independent validation **PASS**: {validation['run_arithmetic_verified']:,} run aggregates, 575.749 matched source originals; freshly encoded **{completion['unique_selected_originals_fresh_encoded']:,} unique selected original bodies** across methods/budgets. Input/C1/tokenizer/script hashes verified, all 30 counterfactual IDs/texts/order/tokens/star-time histograms/JSD checked, HTML originals checked. Qualitative IDs refer to actual selected/alternative originals.",
        f"Notebook 07 **{notebook['executed_code_cells']}/{notebook['expected_code_cells']} cells PASS**, {notebook['image_outputs']} figures; raw/cohort/split/English sources unchanged. Runner {completion['elapsed_seconds']:.2f}s; independent validator {validation['elapsed_seconds']:.2f}s. No test-business tuning or semantic model/GPU job launched. Semantic model/scores, final review budget and manifest remain null/unfrozen.",
        'Artifacts: [per-business/method/budget metrics](business_length_comparisons.csv), [summary](length_summary.csv), [volume breakdown](length_summary_by_volume.csv), [split breakdown](length_summary_by_split.csv), [paired baselines](paired_baseline_comparisons.csv), [exactly-30 token ranking](exactly30_ranked_by_tokens.csv), [counterfactual provenance](counterfactual_alternatives.json), [validation](validation_report.json).']
    (OUT/'length_content_audit_report.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print('Verified length/content report generated.')


if __name__=='__main__': main()
