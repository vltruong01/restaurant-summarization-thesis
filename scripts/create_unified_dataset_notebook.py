"""Primary maintained notebook: history, current diagnostics, viewer and next steps.
Colabexecutionharness09productionv2is separate; dataset/method/history remainhere.
"""
from pathlib import Path
import json,textwrap
ROOT=Path(__file__).resolve().parents[1];cells=[]


def add(kind,text):
    source=textwrap.dedent(text).strip()+'\n'
    cell={'cell_type':kind,'id':f'dataset-unified-{len(cells):03d}','metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':compile(source,f'cell-{len(cells)}','exec');cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown','''
# Thesis — Dataset, filtering và review selection: notebook tổng hợp

**Đây là notebook chính để xem lại và tiếp tục công việc.** Các phần đã làm, đang làm và sẽ làm nằm chung ở đây; notebooks01–07 giữ làm lịch sử. Notebook08 là harness thử BLIP-2 riêng theo yêu cầu mới, có link ở10.1; dataset/filtering/progress vẫn cập nhật tại đây.

Mở Python3/ipykernel, **Run All mặc định đọc kết quả/cached reports**, không chạy lại raw scan, language detection, review search, model hoặc API. Agent quản lý chạy/kiểm chứng pipeline. Chỉ khi chủ động đổi `RUN_CAPTION_PILOT=True`, cellsE0 mới cài môi trường riêng/tải và chạy model pilot theo spec005. Thay `BUSINESS_ID_TO_VIEW` ở ô cấu hình, hoặc dùng ô chọn business ở phần10. Mọi text review hiển thị nguyên vẹn; phân trang chỉ giới hạn số review trên trang.

1. Tiến độ/phạm vi · 2. Raw/pool/English eligibility · 3. Cohort/splits/reviews/photos · 4. English filtering · 5. LED lengths · 6. Selection history/budgets · 7. Short-review audit · 8. V4 pilot · 9. Review/source/image audits · 10. Review viewer/BLIP-2 pilot · 11. Planned research stages. Bảo trì storage được lưu ở receipts riêng, không nằm trong research notebook.

Cohort2.300 = train2.000/validation200/test100. Test chỉ có thống kê mô tả nguồn; selection/budget pilots dùng train/validation. Review count/rating/time tốt chưa chứng minh giữ đủ aspects/opinions. Final methodology/review-metadata-caption allocation/manifest vẫn pending.
''')
add('code','''
from pathlib import Path
import sys,json,hashlib,html
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from IPython.display import display,HTML
import yaml,orjson
ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'specs/003f_length_balanced_review_selection_pilot.md').is_file())
sys.path.insert(0,str(ROOT/'scripts'))
from dataset_review_notebook_helpers import load_json,checked_csv,stage_status,render_business,optional_viewer
BUSINESS_ID_TO_VIEW='9PZxjhTIU7OgPIzuGi89Ew'
REVIEW_BUDGET_TO_VIEW=3500
VIEW_STAGE_TO_VIEW='v4_2_expanded_selected' # raw / English_pool / English_removed / v3_selected / v4_selected / v4_1_selected / v4_2_selected / v4_2_control_selected / v4_2_expanded_selected / v4_2_expanded_control_selected
VIEW_PAGE=0
PAGE_SIZE=30
SOURCE_METADATA_PROFILE_TO_VIEW='name_categories_location' # provisional audit profile, not final training choice
CAPTION_ALLOWANCE_TO_VIEW=750 # hypothetical entire caption-block allowance, not measured caption tokens
RUN_CAPTION_PILOT=False # cached results by default; enable only to deliberately run/resume E0
ALLOW_CAPTION_SETUP_AND_DOWNLOAD=True # used only when RUN_CAPTION_PILOT=True
FIG=ROOT/'outputs/unified_dataset_inspection/figures';FIG.mkdir(parents=True,exist_ok=True)
COHORT=ROOT/'data/processed/yelp_2300_v1'
C1=ROOT/'outputs/stage_c/review_diagnostics_v1'
EXP=ROOT/'outputs/stage_c/review_selection_train_validation_v3'
AUDIT=ROOT/'outputs/stage_c/review_selection_length_audit_v1'
V4=ROOT/'outputs/stage_c/review_selection_length_balanced_pilot_v1'
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
pd.set_option('display.max_columns',25);pd.set_option('display.max_colwidth',180)
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/name,dpi=150,bbox_inches='tight');plt.show();plt.close(fig)
def accepted(path,key='validation_passed'):
    r=load_json(path);assert r[key],path;return r
cohort_validation=accepted('outputs/cohort/validation_report.json')
eligibility_validation=accepted('outputs/english_eligibility/validation_report.json')
c1_validation=accepted('outputs/stage_c/review_diagnostics_v1/validation_report.json')
expanded_validation=accepted('outputs/stage_c/review_selection_train_validation_v3/validation_report.json')
audit_validation=accepted('outputs/stage_c/review_selection_length_audit_v1/validation_report.json')
v4_validation=accepted('outputs/stage_c/review_selection_length_balanced_pilot_v1/validation_report.json')
c1_completion=load_json('outputs/stage_c/review_diagnostics_v1/completion_report.json')
metadata=checked_csv('outputs/stage_c/review_diagnostics_v1/review_language_token_metadata.csv',c1_completion['diagnostic_artifact_sha256']['review_language_token_metadata.csv'])
c1stats=checked_csv('outputs/stage_c/review_diagnostics_v1/business_review_statistics.csv',c1_completion['diagnostic_artifact_sha256']['business_review_statistics.csv'])
cohort=checked_csv('outputs/cohort_inspection/business_statistics.csv')
assert len(cohort)==2300 and cohort.business_id.is_unique
display(stage_status())
''')
add('markdown','''
## 2. Raw data → restaurant pool → English eligibility → cohort

Business gate chạy **trước sampling**: restaurant policy của StageB, >=10 reviews và >=5 unique local photos, sau đó chứng minh >=10 detector-English reviews. Gate chỉ nhận diện đủ để quyết định eligibility; không phải tổng English của cả raw Yelp. Seed42 tái lập cùng2.300 IDs/split, không phải tạo cohort khác. Toàn raw review file đã scan/hash ở lần kiểm chứng trước; Run All hiện chỉ đọc evidence.
''')
add('code','''
raw=load_json('outputs/cohort/eligibility_audit.json');gate=load_json('outputs/english_eligibility/eligibility_report.json')
display(pd.DataFrame([
    {'step':'Raw businesses','businesses':raw['sources']['business']['records'],'reviews':raw['sources']['review']['records'],'photos':raw['sources']['photo']['records'],'scope':'Full Yelp raw scan'},
    {'step':'Restaurants after accommodation exclusion','businesses':raw['restaurant_business_ids_after_accommodation_exclusion'],'scope':'Category policy'},
    {'step':'>=10 reviews + >=5 local unique photos','businesses':raw['eligible_count'],'scope':'Pool10.197'},
    {'step':'English source eligibility before sampling','businesses':gate['english_eligible_businesses'],'scope':f"Assessed {gate['reviews_language_assessed']:,} reviews; threshold certification, not full language totals"},
    {'step':'Fixed seeded cohort','businesses':2300,'reviews':601493,'photos':30027,'scope':'2.000/200/100'}]))
display(pd.DataFrame([gate['identity']['language_detector']]))
print('English gate assessed102.103 originals; C1 below classifies all601.493 cohort originals. No image-content deduplication/BLIP-2 filtering is claimed.')
''')
add('markdown','''
## 3. Dataset hiện tại: reviews, photos, min/max, split

Nguồn raw cohort ở `data/processed/yelp_2300_v1`; full English pool ở `data/processed/yelp_2300_english_v1`. Ảnh được tham chiếu từ v1, không copy thêm. >=5 local photo files là eligibility, chưa chứng minh mọi ảnh đọc được hoặc khác nội dung. Tables bao gồm test để mô tả cohort; không dùng test quyết định selection.
''')
add('code','''
overview=load_json('outputs/cohort_inspection/overview_summary.json');build=load_json('outputs/cohort/build_report.json')
display(pd.DataFrame(build['split_summary']).T)
display(cohort[['reviews','photos']].describe(percentiles=[.5,.9,.95,.99]).round(2))
for metric in ['reviews','photos']:
    print('MOST',metric);display(cohort.nlargest(5,metric)[['business_id','name','split',metric]])
    print('LEAST',metric);display(cohort.nsmallest(5,metric)[['business_id','name','split',metric]])
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,col,title in zip(axes,['reviews','photos'],['Original reviews/business','Local photos/business']):
    ax.hist(cohort[col],bins=40,color='#2563eb');ax.set(xlabel=title,ylabel='Businesses',yscale='log')
save(fig,'cohort_review_photo_counts.png')
from PIL import Image
photo_manifest=pd.read_csv(COHORT/'photo_manifest.csv')
preview=photo_manifest.loc[photo_manifest.business_id.eq(BUSINESS_ID_TO_VIEW)].sort_values('photo_id').head(3)
if len(preview):
    fig,axes=plt.subplots(1,len(preview),figsize=(10,3.5));axes=np.atleast_1d(axes)
    for ax,r in zip(axes,preview.to_dict('records')):
        path=COHORT/r['relative_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        with Image.open(path) as photo:ax.imshow(photo.copy())
        ax.set_title(str(r['label']));ax.axis('off')
    fig.suptitle('Local photo preview: '+str(cohort.loc[cohort.business_id.eq(BUSINESS_ID_TO_VIEW),'name'].iloc[0]))
    save(fig,'business_photo_preview.png')
    print('Only these preview images were decoded here; not exhaustive image integrity/content-dedup checks.')
''')
add('markdown','''
## 4. English filtering: giữ/loại gì?

langdetect1.0.9 seed42; detector probe strip, tối thiểu20 ký tự, tối đa2.000 ký tự; giữ top label `en`, không confidence cutoff. Unknown và other-language labels không vào English pool. **Review export và token counts vẫn dùng toàn original text**, không cắt theo detector probe. Nhãn tự động không phải language ground truth; URL-only/generic text có thể vẫn được gắn en. Không tự thêm minimum token-length filter.
''')
add('code','''
display(pd.DataFrame([{k:c1_completion[k] for k in ['raw_reviews','english_reviews','non_english_reviews','unknown_reviews','businesses','minimum_english_reviews_per_business']}]))
lang=metadata.assign(group=np.where(metadata.language.eq('en'),'en',np.where(metadata.language.eq('unknown'),'unknown','other')))
display(pd.crosstab(lang.split,lang.group));assert len(metadata)==601493
print('Non-English/unknown businesses to inspect:')
display(metadata.loc[~metadata.language.eq('en'),['business_id','split','language','language_reason','led_tokens_without_special']].head(12))
print('All2300 source-business English counts >=10:',bool(c1stats.english_reviews.ge(10).all()))
fig,axes=plt.subplots(1,2,figsize=(11,4))
metadata.language.value_counts().rename_axis('language').reset_index(name='reviews').head(10).plot.bar(x='language',y='reviews',logy=True,legend=False,ax=axes[0],color='#64748b')
axes[0].set_title('Automatic language labels: full cohort')
en=metadata.loc[metadata.language.eq('en')]
en.stars.astype(int).value_counts().sort_index().plot.bar(ax=axes[1],color='#2563eb');axes[1].set(title='Stars of full English pool',xlabel='Stars',ylabel='Reviews')
save(fig,'language_filter_and_source_stars.png')
removed_examples=metadata.loc[~metadata.language.eq('en') & metadata.split.isin(['train','validation'])].head(5).set_index('review_id')
original_removed={}
for split in removed_examples.split.unique():
    wanted=set(removed_examples.loc[removed_examples.split.eq(split)].index)
    for line in (COHORT/'reviews'/f'{split}.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['review_id'] in wanted:
            assert hashlib.sha256(r['text'].encode()).hexdigest()==removed_examples.loc[r['review_id'],'text_sha256']
            original_removed[r['review_id']]=r
            if wanted<=set(original_removed):break
assert set(original_removed)==set(removed_examples.index)
display(HTML('<h3>Ví dụ nguyên văn không vào English pool (train/validation)</h3>'+''.join(
    '<details open><summary>'+html.escape(f"{rid} | {removed_examples.loc[rid,'language']} | {removed_examples.loc[rid,'language_reason']}")+'</summary><pre style="white-space:pre-wrap">'+html.escape(r['text'])+'</pre></details>'
    for rid,r in original_removed.items())))
''')
add('markdown','''
## 5. LED token lengths và review dài

BODY tokens không gồm markers/BOS/EOS; source blocks gồm markers/stars/date/separators/special tokens, chưa có business metadata/captions. ~5.000 là planned total target;3.500/4.000/4.500 là exploratory review budgets, chưa final allocation. Long-review probes khác selected inputs.
''')
add('code','''
trainval=metadata.loc[metadata.language.eq('en') & metadata.split.isin(['train','validation'])]
assert len(trainval)==575749
display(trainval.led_tokens_without_special.astype(int).describe(percentiles=[.5,.9,.95,.99]))
stress=pd.DataFrame(load_json('outputs/stage_c/review_selection_train_validation_v3/long_review_stress.json'))
display(stress.groupby('count').agg(businesses=('business_id','size'),above5000=('exact_tokens',lambda s:int((s>5000).sum())),tokens_min=('exact_tokens','min'),tokens_max=('exact_tokens','max')))
print('356 businesses:7 longest >5000;548 businesses:8 longest >5000. These are diagnostic blocks, not chosen inputs.')
''')
add('markdown','''
## 6. Lịch sử selection và expanded budget diagnostics

V1 initial rating/time sampling → v3 whole-review search dưới budget → expanded train/validation. Không minimum10 input: eligibility >=10 chỉ áp dụng source business. Không <10 trong dữ liệu thử không phải bằng chứng giữ đủ semantic content. V3 đã được xác nhận thiên về review ngắn ở phần7; giữ lịch sử để so sánh.
''')
add('code','''
display(pd.DataFrame([
    {'version':'initial v1','businesses':48,'runs':96,'status':'Historical initial selection'},
    {'version':'budget pilot v3','businesses':56,'runs':504,'status':'Historical exploratory evidence'},
    {'version':'expanded v3','businesses':2200,'runs':19800,'status':'Validated integrity; short-review bias concern'},
    {'version':'length audit','businesses':2200,'runs':19800,'status':'Validated measurement, selected content quality pending'},
    {'version':'revised v4 pilot','businesses':v4_validation['businesses'],'runs':v4_validation['runs_verified'],'status':'Pilot only; not all2300 frozen'}]))
expanded=pd.read_csv(EXP/'budget_method_summary.csv')
display(expanded.loc[expanded.method.eq('budget_aware'),['review_budget','retained_mean','retained_min','retained_max','below_10','below_5','tokens_min','tokens_max','rating_jsd_after_mean','temporal_jsd_after_mean']].round(6))
''')
add('markdown','''
## 7. Phát hiện thiên về review ngắn trên2.200 businesses

Compare selected/source BODY means trong từng business; median ratio0,383 khác pooled mean. Macro means cho mỗi business cùng trọng số. Thresholds<20/<30/<50 hay ratio<0,5 là mô tả, không quality gates. Source size khiến khả năng chọn cực ngắn khác nhau. Semantic model/scores chưa tính.
''')
add('code','''
audit=pd.read_csv(AUDIT/'length_summary.csv')
display(audit.loc[audit.method.eq('budget_aware'),['review_budget','source_mean_macro','selected_mean_macro','selected_source_mean_ratio_median','mean_ratio_below_0_5','source_short20_macro','selected_short20_macro','exactly30_under1500']].round(4))
display(pd.read_csv(AUDIT/'paired_baseline_comparisons.csv').query("comparison=='same_count_only'").round(4))
notes=load_json('outputs/stage_c/review_selection_length_audit_v1/qualitative_inspection.json')
display(pd.DataFrame(notes['cases'])[['name','observations']]);print(notes['scope']);print(notes['limitations'])
print('Same-star/time-bin longer counterfactuals are diagnostics only; no accepted selected IDs overwritten.')
''')
add('markdown','''
## 8. V4 pilot: giảm shortest-review default

Uniform seeded rating/time candidates + rotating length strata + uniform joint-cell candidates; incoming reviews sampled across length strata. Rating/time/count ties dùng seed-derived hash, không tối thiểu tokens. Minimum-cost singleton chỉ là feasibility rescue. Không minimum review length, semantic objective hoặc ép dùng hết budget. Weights/tolerances/budgets vẫn exploratory như trước.

Pilot80 = accepted56 stress cases ∪ ten audit cases +16 additional seeded cases (4/source-count band),69train/11validation; source40.407 English originals. Purposive/stress sample, không suy rộng thành kết quả toàn2.200. Pilot so sánh combined revision, chưa ablation từng cơ chế.
''')
add('code','''
pilot=pd.read_csv(V4/'budget_method_summary.csv');pdiag=pd.read_csv(V4/'selection_diagnostics.csv')
display(pilot[['method','review_budget','businesses','retained_mean','below10','below5','tokens_min','tokens_max','median_mean_ratio','mean_length_percentile','selected_short20_macro','rating_jsd_mean','temporal_jsd_mean']].round(6))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for method,color in [('v3_budget_aware','#f59e0b'),('v4_length_balanced','#2563eb')]:
    s=pilot.loc[pilot.method.eq(method)].sort_values('review_budget')
    for ax,col,title in zip(axes,['median_mean_ratio','retained_mean','rating_jsd_mean'],['Selected/source BODY ratio (median)','Complete reviews retained (mean)','Rating JSD (mean; lower better)']):
        ax.plot(s.review_budget,s[col],marker='o',color=color,label=method);ax.set(title=title,xlabel='Review budget');ax.set_xticks([3500,4000,4500])
axes[0].legend(fontsize=8);save(fig,'v4_length_count_rating_tradeoffs.png')
''')
add('markdown','''
## 9. Đánh đổi và cases cần xem

Độ dài gần source hơn chưa chứng minh summary tốt hơn. Xem cả count giảm và rating/time regressions. Budget lớn hơn không đảm bảo count monotonic với bounded search + quality/count ranking; chưa global optimum. Những flags không tự loại business/tăng source length.
''')
add('code','''
paired=pd.read_csv(V4/'paired_v4_v3_differences.csv')
display(paired.groupby('review_budget').agg(businesses=('business_id','size'),mean_count_delta=('count_delta','mean'),count_lower=('count_delta',lambda s:int((s<0).sum())),rating_worse=('rating_delta',lambda s:int((s>1e-12).sum())),time_worse=('time_delta',lambda s:int((s>1e-12).sum())),mean_body_delta=('body_mean_delta','mean')).round(6))
display(pd.read_csv(V4/'below_ten_cases.csv'))
print('Worst rating-JSD changes to inspect:');display(paired.nlargest(8,'rating_delta'))
display(pdiag.loc[pdiag.business_id.eq(BUSINESS_ID_TO_VIEW),['method','review_budget','final_retained_review_count','tokens_after_budget_enforcement','selected_body_mean','selected_source_mean_ratio','rating_jsd_after','temporal_jsd_after']].round(6))
''')
add('markdown','''
### 9.1. Review v4: kết luận và đối chiếu ứng viên (2026-10-06)

**Chưa dùng standalone v4 làm final selector.** Tại3.500/4.000/4.500, v3 Pareto-dominates v4 ở28/13/6 cases: v3 không kém rating/time/count và tốt hơn ít nhất một tiêu chí. Đây là vấn đề candidate coverage, không thể giải thích mọi count loss bằng review dài. Ví dụ Water Works: v4=10 reviews/3.025 tokens; v3=30/2.955, rating JSD tốt hơn và time JSD giống. Không tự tăng budget hay lọc bỏ business.

Cả bốn count decreases khi tăng budget có previous set vẫn fit/comparable nhưng vắng ở higher-budget frontier; carry previous set phục hồi count ở cả bốn. Không ép count monotonic làm research objective. Pool toàn các cached budget frontiers rồi filter exact budget cho0 count decreases trong pilot này; không bảo đảm điều đó cho mọi dữ liệu.

**Các bảng counterfactual chỉ dùng chẩn đoán**, không phải fresh v4.1 search hoặc final manifest. Thêm accepted v3 incumbents + union cached frontiers giữ mean count26,53/27,08/27,18 và median selected/source BODY ratio0,727/0,774/0,845 (v3=0,417/0,423/0,429). Đây là cách bảo vệ rating/time/count mà vẫn giữ nhiều ứng viên ít thiên về short reviews; chưa chứng minh content quality hay toàn2.200 businesses.

Đã đọc nguyên văn24 reviews khác nhau giữa v3/v4 ở bốn purposive cases: Zahav, The Lively, Water Works và Narayanni's. V3 cũng có review ngắn chứa thông tin hữu ích; v4 có review dài tập trung vào một visit hoặc ambience. Nhận xét dưới chỉ minh họa, không formal human evaluation/aspect recall. Toàn original v3/v4 của các cases cũng được lưu trong HTML audit; viewer phần10 xem bất kỳ business pilot nào ngay trong notebook.

**Đề xuất tại thời điểm003g:** pilot phiên bản mới với shared feasible candidate pool giữa budgets và accepted v3 incumbent làm ứng viên bảo vệ; giữ nguyên rating/time/count, cap30, whole reviews, seed/tolerances và budgets. Bước đó hiện đã thực hiện ở phần9.2; số liệu003g phía trên vẫn là cached counterfactuals, không thay bằng kết quả fresh.
''')
add('code','''
TRADE=ROOT/'outputs/stage_c/review_selection_tradeoff_audit_v1'
trade_validation=accepted('outputs/stage_c/review_selection_tradeoff_audit_v1/validation_report.json')
trade=checked_csv('outputs/stage_c/review_selection_tradeoff_audit_v1/v3_v4_tradeoffs.csv',trade_validation['artifact_sha256']['v3_v4_tradeoffs.csv'])
transitions=checked_csv('outputs/stage_c/review_selection_tradeoff_audit_v1/cross_budget_transitions.csv',trade_validation['artifact_sha256']['cross_budget_transitions.csv'])
counter=checked_csv('outputs/stage_c/review_selection_tradeoff_audit_v1/counterfactual_budget_summary.csv',trade_validation['artifact_sha256']['counterfactual_budget_summary.csv'])
display(trade.groupby('review_budget').agg(businesses=('business_id','size'),v3_Pareto_dominates_v4=('v3_pareto_dominates_v4','sum'),v3_set_evaluated=('v3_in_any_evaluated_candidates','sum'),v3_comparable_to_v4_anchor=('v3_comparable_to_anchor','sum')))
print('All4 count-decrease transitions and unchanged-rule carry diagnostics:')
display(transitions.loc[transitions.count_decreased])
print('COUNTERFACTUAL ONLY: reused accepted candidates, not fresh selector/model evidence.')
display(counter.round(6));print('Remaining adjacent count decreases:',trade_validation['cached_union_count_decreases'])
qual=load_json('outputs/stage_c/review_selection_tradeoff_audit_v1/qualitative_review_notes.json')
assert qual['read_reviews']==24 and all(c['status']=='assistant_read_complete' for c in qual['cases'])
display(pd.DataFrame(qual['cases'])[['name','review_budget','reason','observations']]);print(qual['sampling']);print(qual['limitations'])
full_read_reviews=load_json('outputs/stage_c/review_selection_tradeoff_audit_v1/qualitative_reviews_to_read.json')
assert len(full_read_reviews)==24
for case in qual['cases']:
    rows=[r for r in full_read_reviews if r['business_id']==case['business_id']]
    assert {r['review_id'] for r in rows}==set(case['read_review_ids'])
    chunks=['<h3>'+html.escape(case['name'])+' — reviews actually read</h3>']
    for r in rows:
        chunks.append('<details><summary>'+html.escape(f"{r['version']} only | {r['review_id']} | {r['stars']} stars | {r['date']}")+'</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape(r['text'])+'</pre></details>')
    display(HTML(''.join(chunks)))
display(pd.DataFrame([{k:v for k,v in trade_validation.items() if k not in ['artifact_sha256','protected_sha256']}]))
''')
add('markdown','''
### 9.2. Fresh pilot v4.1: candidate coverage đã sửa, length shift vẫn còn

**Đã chạy thật trên cùng80 businesses**, không phải rerank cached003g: fresh v4 discovery giữ sampling/tie hash version cũ, thêm accepted v3 incumbents, bounded local discovery ở cả ba budgets, rồi chọn từ shared exact candidate pool. Không nhập accepted v4 candidates vào discovery; v4 chỉ baseline đối chiếu. Settings/rating/time/count/complete boundaries/cap30/seed/budgets/tolerances không đổi.

V4.1 giữ trung bình26,53/27,09/27,18 complete reviews, count10–30;0<10/<5/infeasible và0 adjacent count drops. Không bị v3 Pareto-dominate; vẫn có một time-JSD regression tại4.000/4.500. Hai accepted v4 sets Pareto-dominate v4.1 tại4.500 (Taco Grande/Rhone Rum Bar); coverage chưa hoàn hảo.

**Không kết luận đã giải quyết short-review bias:** median selected/source BODY ratios chỉ0,457/0,498/0,503; selected<20 fractions11,65%/11,30%/11,09% so source1,05%. V3=0,417/0,423/0,429 và~13,53%. Exactly30 blocks<1.500 còn5/budget (v3=11). Fresh gains nhỏ hơn cached counterfactual003g vì incumbent-influenced anchors/proposals tạo pool khác và các old v4 candidates không được nhập lại. Không dùng số0,727/0,774/0,845 trước đó làm actual v4.1 result.

Water Works3.500: v4=10/3.025 → v4.1=30/2.960, rating/time tương đương v3. El Vez3.500: v3=30/1.154 → v4.1=30/1.556, cùng rating/time; accepted v4=30/3.420. Xem nguyên văn cả ba versions ở phần10 bằng VIEW_STAGE_TO_VIEW. Full-original comparisons cho bảy cases nằm trong report folder; chưa có formal content-quality verdict/human evaluation.

Terminal exact proposals vượt discovery budget có thể vừa final budget lớn hơn khi shared reuse; derivation trace ghi rõ, không suy ra tất cả terminal steps đã được local search chấp nhận. Final selected blocks luôn exact-fit configured budget. Chín meaningful tests và independent/fresh all80 validation đã PASS. **Dừng sau pilot:** chưa expand2.200, freeze2.300 hoặc chốt allocation. Bước kế tiếp review hai residual coverage cases cùng bias/tie behavior trước quyết định pilot khác; không tự thêm length/semantic objective.
''')
add('code','''
V41=ROOT/'outputs/stage_c/review_selection_shared_pool_pilot_v1'
v41_validation=accepted('outputs/stage_c/review_selection_shared_pool_pilot_v1/validation_report.json')
v41_completion=load_json('outputs/stage_c/review_selection_shared_pool_pilot_v1/completion_report.json')
v41_summary=checked_csv('outputs/stage_c/review_selection_shared_pool_pilot_v1/budget_method_summary.csv',v41_completion['artifact_sha256']['budget_method_summary.csv'])
v41_diag=checked_csv('outputs/stage_c/review_selection_shared_pool_pilot_v1/selection_diagnostics.csv',v41_completion['artifact_sha256']['selection_diagnostics.csv'])
methods=['v3_budget_aware','v4_length_balanced','v4_1_shared_pool']
display(v41_summary.loc[v41_summary.method.isin(methods),['method','review_budget','retained_mean','retained_min','retained_max','tokens_min','tokens_max','below10','below5','median_mean_ratio','selected_short20_macro','rating_jsd_mean','temporal_jsd_mean','exactly30_under1500']].round(6))
display(pd.read_csv(V41/'regression_summary.csv'))
remaining=pd.read_csv(V41/'paired_baseline_comparisons.csv')
print('Remaining baseline Pareto-dominates-v4.1 cases:');display(remaining.loc[remaining.baseline_pareto_dominates_v4_1])
print('Retained count distribution:');display(v41_summary.loc[v41_summary.method.eq('v4_1_shared_pool')].filter(regex='review_budget|band_'))
print('Detailed examples:');display(pd.read_csv(V41/'detailed_examples.csv').round(6))
findings=load_json('outputs/stage_c/review_selection_shared_pool_pilot_v1/review_findings.json');print('Adjacent count drops:',findings['counter_decreases'])
print('Terminal overflow at discovery, feasible final budget cases:');display(pd.read_csv(V41/'cross_budget_terminal_proposals.csv'))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for method,color,label in [('v3_budget_aware','#f59e0b','v3'),('v4_length_balanced','#64748b','v4'),('v4_1_shared_pool','#2563eb','v4.1 fresh')]:
    s=v41_summary.loc[v41_summary.method.eq(method)].sort_values('review_budget')
    for ax,col,title in zip(axes,['retained_mean','median_mean_ratio','selected_short20_macro'],['Complete reviews retained (mean)','Selected/source BODY ratio (median)','Reviews below20 BODY tokens (macro)']):
        ax.plot(s.review_budget,s[col],marker='o',color=color,label=label);ax.set(title=title,xlabel='Review budget');ax.set_xticks([3500,4000,4500])
axes[2].axhline(v41_summary.loc[v41_summary.method.eq('v4_1_shared_pool'),'source_short20_macro'].iloc[0],color='#16a34a',linestyle='--',label='Source fraction')
axes[0].legend(fontsize=8);axes[2].legend(fontsize=8);save(fig,'v4_1_coverage_and_residual_length_shift.png')
display(pd.DataFrame([v41_validation]));print('Fresh pilot PASS; final input content quality/methodology/allocation remains pending.')
''')
add('markdown','''
### 9.3. Residual coverage và các bộ tương đương rating/time/count

**Hai câu hỏi được kiểm tra riêng:** (1) bỏ sót bộ v4 tốt hơn; (2) primary rating/time/count có đủ để xác định bộ chứa nhiều thông tin hay không. Audit003i giữ accepted v4/v4.1 selections, source, budgets và mọi objectives; không chạy selector mới/freeze/model/API.

Taco Grande/Rhone Rum Bar tại4.500 thiếu accepted v4 dominating sets trong v4.1 shared pool. Thêm lại từng bộ và dùng unchanged rule khôi phục29/30 complete reviews với rating/time tốt hơn. Ba cached pool counterfactuals bên dưới thử giữ v4 incumbents, toàn v4 frontiers hoặc toàn ordinary v4 exact evaluations. Các lựa chọn đó là diagnostics, không fresh v4.2 hoặc final model inputs.

Các sets có **cùng count và exact star/time-bin histograms** có cùng rating/time JSD nhưng có thể chứa các originals khác nhau. Hash tie không phụ thuộc tokens; nó cũng không chấm relevance/aspects/quality. Các range/midrank trong bảng chỉ mô tả những candidates đã tìm thấy, không phải mọi subset có thể có.

Ba low-token exactly30 cases có probe thay nguyên review bằng same-star/same-source-time-bin originals trong middle50% source BODY lengths, sampled bằng seed42; encode cả block và kiểm tra histogram/JSD/count giữ nguyên. Đây là counterfactual để kiểm tra giới hạn objectives, không minimum-length filter, semantic objective hay longest-review rule. Exact date span có thể đổi dù time-bin histogram giống; được báo cáo riêng.

Full originals thực sự được assistant đọc được ghi dưới đây; illustrative/purposive, chưa formal human evaluation/exhaustive source-aspect recall. Việc dài hơn không đủ chứng minh summary tốt hơn. Kết luận/next decision đọc cùng report003i và mọi remaining tradeoffs; không tự thay research design hoặc tăng budget.
''')
add('code','''
RESIDUAL=ROOT/'outputs/stage_c/review_selection_residual_audit_v1'
res_validation=accepted('outputs/stage_c/review_selection_residual_audit_v1/validation_report.json')
res_cases=checked_csv('outputs/stage_c/review_selection_residual_audit_v1/residual_case_diagnostics.csv',res_validation['artifact_sha256']['residual_case_diagnostics.csv'])
res_pools=checked_csv('outputs/stage_c/review_selection_residual_audit_v1/diagnostic_pool_summary.csv',res_validation['artifact_sha256']['diagnostic_pool_summary.csv'])
equivalent=checked_csv('outputs/stage_c/review_selection_residual_audit_v1/objective_equivalence_audit.csv',res_validation['artifact_sha256']['objective_equivalence_audit.csv'])
display(res_cases);print('COUNTERFACTUALS ONLY — no current model evidence replaced.');display(res_pools.round(6))
print('Counterfactual adjacent count drops:',res_validation['diagnostic_count_drops'])
display(equivalent.groupby('review_budget').agg(businesses=('business_id','size'),median_equivalent_sets=('exact_histogram_equivalent_sets','median'),longer_same_histogram_exists=('longer_same_histogram_exists','sum'),median_chosen_midrank=('chosen_body_midrank_in_group','median')))
print('Largest within-found-group BODY spreads:');display(equivalent.assign(body_spread=equivalent.group_body_max-equivalent.group_body_min).nlargest(8,'body_spread'))
cell_probes=load_json('outputs/stage_c/review_selection_residual_audit_v1/same_histogram_counterfactuals.json');display(pd.DataFrame([{k:v for k,v in c.items() if k not in ['selected_review_ids','removed_ids','added_ids']} for c in cell_probes]).round(6))
res_notes=load_json('outputs/stage_c/review_selection_residual_audit_v1/qualitative_review_notes.json')
display(pd.DataFrame(res_notes['cases'])[['name','reason','budget','observations']]);print(res_notes['limitations'])
res_reviews=load_json('outputs/stage_c/review_selection_residual_audit_v1/qualitative_reviews_to_read.json')
assert len(res_reviews)==res_notes['read_reviews']
for case in res_notes['cases']:
    part=[x for x in res_reviews if x['business_id']==case['business_id']]
    assert {x['review_id'] for x in part}==set(case['read_review_ids']) and case['status']=='assistant_read_complete'
    display(HTML('<h3>'+html.escape(case['name'])+'</h3>'+''.join('<details><summary>'+html.escape(f"{x['version']} | {x['review_id']} | {x['stars']} stars | {x['date']}")+'</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape(x['text'])+'</pre></details>' for x in part)))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for label,color in [('plus_v4_incumbents','#2563eb'),('plus_v4_frontiers','#7c3aed'),('plus_all_ordinary_v4_evaluations','#059669')]:
    s=res_pools.loc[res_pools.diagnostic_pool.eq(label)].sort_values('review_budget')
    for ax,col,title in zip(axes,['count_mean','median_body_ratio','short20_macro'],['Complete reviews retained (mean)','Selected/source BODY ratio (median)','Reviews below20 BODY tokens (macro)']):
        ax.plot(s.review_budget,s[col],marker='o',label=label,color=color);ax.set(title=title,xlabel='Review budget');ax.set_xticks([3500,4000,4500])
base=v41_summary.loc[v41_summary.method.eq('v4_1_shared_pool')].sort_values('review_budget')
for ax,col in zip(axes,['retained_mean','median_mean_ratio','selected_short20_macro']):ax.plot(base.review_budget,base[col],linestyle='--',color='#64748b',label='Accepted fresh v4.1')
axes[0].legend(fontsize=6);save(fig,'residual_cached_candidate_pool_diagnostics.png')
display(pd.DataFrame([{k:v for k,v in res_validation.items() if k not in ['artifact_sha256','protected_sha256']}]))
''')
add('markdown',"""
## 9.4. V4.2: secondary source-length distribution pilot

User authorized the secondary rule under spec003j. Same80 train/validation businesses,40407 complete English originals; no test/model/final manifest. First retain accepted candidates and use unchanged primary rating/time/count rules. Only among EXACT same count/star/time-histogram sets, minimize base2 JSD against source BODY-length distribution (four empirical bins, equal lengths together, actual source proportions). This is not choosing longest or filling tokens.

Compare primary control, existing-candidate tie only and bounded same-cell replacement refinement (six iterations/four exact proposals). Refinement preserves primary histograms and complete-review budget, accepts only strict length-JSD improvements. Matching lengths does not classify individual quality or prove aspect preservation. All results below are current fresh discovery with declared accepted-ID retention, not prior cached003i counterfactuals or final methodology.
""")
add('code',"""
V42=ROOT/'outputs/stage_c/review_selection_source_length_tie_pilot_v1'
v42_validation=load_json('outputs/stage_c/review_selection_source_length_tie_pilot_v1/validation_report.json')
assert v42_validation['validation_passed']
v42_summary=checked_csv('outputs/stage_c/review_selection_source_length_tie_pilot_v1/budget_method_summary.csv')
v42_pairs=checked_csv('outputs/stage_c/review_selection_source_length_tie_pilot_v1/paired_primary_secondary.csv')
v42_findings=load_json('outputs/stage_c/review_selection_source_length_tie_pilot_v1/pilot_findings.json')
display(v42_summary);display(pd.DataFrame([v42_findings]));display(v42_pairs)
assert len(v42_pairs)==240 and v42_pairs.primary_histograms_preserved.all()
assert (v42_pairs.length_jsd_final<=v42_pairs.length_jsd_primary+1e-12).all()
display(checked_csv('outputs/stage_c/review_selection_source_length_tie_pilot_v1/example_businesses.csv'))
display(checked_csv('outputs/stage_c/review_selection_source_length_tie_pilot_v1/baseline_regressions.csv'))
v42_notes=load_json('outputs/stage_c/review_selection_source_length_tie_pilot_v1/qualitative_review_notes.json')
v42_reads=load_json('outputs/stage_c/review_selection_source_length_tie_pilot_v1/qualitative_reviews_to_read.json')
assert len(v42_reads)==v42_notes['read_reviews']==28
for case in v42_notes['cases']:
    part=[r for r in v42_reads if r['business_id']==case['business_id']]
    assert set(case['read_review_ids'])=={r['review_id'] for r in part} and case['status']=='assistant_read_complete'
    display(HTML('<h3>'+html.escape(case['name'])+'</h3><p>'+html.escape(case['observations'])+'</p>'+''.join('<details><summary>'+html.escape(f"{r['version']} | {r['review_id']} | BODY {r['body_tokens']}")+'</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape(r['text'])+'</pre></details>' for r in part)))
print(v42_notes['limitations'])
fig,axes=plt.subplots(1,3,figsize=(14,4))
for method,label,color in [('v4_2_primary_control','Primary control','#64748b'),('v4_2_existing_length_tie','Existing-candidate tie','#7c3aed'),('v4_2_length_balanced','Bounded length refinement','#059669'),('v4_1_shared_pool','Accepted v4.1','#2563eb')]:
    g=v42_summary.loc[v42_summary.method.eq(method)].sort_values('review_budget')
    for ax,col,title in zip(axes,['length_jsd_mean','median_mean_ratio','selected_short20_macro'],['Source BODY-length JSD (mean)','Selected/source BODY ratio (median)','Below20 BODY tokens (macro)']):
        ax.plot(g.review_budget,g[col],marker='o',label=label,color=color);ax.set(title=title,xlabel='Review budget');ax.set_xticks([3500,4000,4500])
axes[0].legend(fontsize=7);save(fig,'v4_2_source_length_secondary_rule.png')
display(pd.DataFrame([v42_validation]))
print('Pilot80 only. Review/metadata/caption allocation and final2300 manifest remain pending.')
""")
add('markdown',"""
## 9.5. Expanded v4.2: all2200 train/validation businesses

Spec003k keeps every003j primary/secondary/whole-review/budget setting unchanged. Missing v4/v4.1 witnesses for2120 were generated with frozen rules; accepted80 witness/payload equality is recorded separately. All575749 complete English originals, no test tuning/final2300 manifest/model stage.

Primary control, existing-candidate length tie and bounded refinement are separate ablations. BODY length balance is a statistical criterion, not quality/aspect preservation. Budgets3500/4000/4500 are exploratory review blocks within planned~5000 TOTAL source; final metadata/caption/review allocation remains pending. Exact computation memoization only changes reuse, with prior partial identity/cache provenance preserved.
""")
add('code',"""
EX42=ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1'
ex42_validation=load_json('outputs/stage_c/review_selection_source_length_expanded_v1/validation_report.json')
ex42_findings=load_json('outputs/stage_c/review_selection_source_length_expanded_v1/expanded_findings.json')
assert ex42_validation['validation_passed'] and ex42_validation['businesses']==2200
ex42_summary=checked_csv('outputs/stage_c/review_selection_source_length_expanded_v1/budget_method_summary.csv')
display(ex42_summary);display(pd.DataFrame([ex42_findings]));display(pd.DataFrame([ex42_validation]))
display(checked_csv('outputs/stage_c/review_selection_source_length_expanded_v1/primary_before_after_summary.csv'))
for label in ['split','volume_band','expanded_witness_origin']:
    display(checked_csv(f'outputs/stage_c/review_selection_source_length_expanded_v1/summary_by_{label}.csv'))
display(checked_csv('outputs/stage_c/review_selection_source_length_expanded_v1/token_extreme_businesses.csv'))
display(checked_csv('outputs/stage_c/review_selection_source_length_expanded_v1/example_businesses.csv'))
ex42_pairs=checked_csv('outputs/stage_c/review_selection_source_length_expanded_v1/paired_primary_secondary.csv')
assert len(ex42_pairs)==6600 and ex42_pairs.primary_histograms_preserved.all()
assert (ex42_pairs.length_jsd_final<=ex42_pairs.length_jsd_primary+1e-12).all()
ex42_flags=load_json('outputs/stage_c/review_selection_source_length_expanded_v1/below_ten_attribution.json')
display(pd.DataFrame(ex42_flags));print('Below10 cases:',len(ex42_flags),'No automatic exclusion/source length change.')
ex42_drops=load_json('outputs/stage_c/review_selection_source_length_expanded_v1/v4_2_count_drop_explanations.json')
display(pd.DataFrame(ex42_drops))
print('Cross-budget count drops already present in primary control:',len(ex42_drops),'The secondary length rule preserves the primary count, rating and time histograms.')
ex42_notes=load_json('outputs/stage_c/review_selection_source_length_expanded_v1/qualitative_review_notes.json')
ex42_reads=load_json('outputs/stage_c/review_selection_source_length_expanded_v1/qualitative_reviews_to_read.json')
assert len(ex42_reads)==ex42_notes['read_reviews']
for case in ex42_notes['cases']:
    part=[r for r in ex42_reads if r['business_id']==case['business_id']]
    assert set(case['read_review_ids'])=={r['review_id'] for r in part} and case['status']=='assistant_read_complete'
    display(HTML('<h3>'+html.escape(case['name'])+'</h3><p>'+html.escape(case['observations'])+'</p>'+''.join('<details><summary>'+html.escape(f"{r['version']} | {r['review_id']} | BODY {r['BODY_tokens']}")+'</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape(r['text'])+'</pre></details>' for r in part)))
print(ex42_notes['limitations'])
fig,axes=plt.subplots(1,3,figsize=(14,4))
for method,label,color in [('v3_budget_aware','Accepted v3','#2563eb'),('v4_2_primary_control','Primary control','#64748b'),('v4_2_existing_length_tie','Existing tie','#7c3aed'),('v4_2_length_balanced','Bounded refinement','#059669')]:
    g=ex42_summary[ex42_summary.method.eq(method)].sort_values('review_budget')
    for ax,col,title in zip(axes,['retained_mean','median_mean_ratio','selected_short20_macro'],['Retained complete reviews (mean)','Selected/source BODY ratio (median)','Below20 BODY tokens (macro)']):
        ax.plot(g.review_budget,g[col],marker='o',label=label,color=color);ax.set(title=title,xlabel='Review budget');ax.set_xticks([3500,4000,4500])
axes[0].legend(fontsize=7);save(fig,'v4_2_expanded2200_count_length_diagnostics.png')
print('All2200 independently checked; fresh reversed current v4.2 on32 new cases plus accepted80 equality. No claim of fresh replay all2200. Final allocation/2300 manifest pending.')
""")
add('markdown','''
## 9.6. LED input / token allocation audit — chưa khóa budget

Spec003l đo nguyên chuỗi source bằng tokenizer LED, với đúng review IDs/text/order của v4.2. Ba metadata profiles chỉ là alternatives cho audit: không metadata, name/categories, name/categories/city/state. Review block cũ đã gồm markers/stars/date/separators/BOS/EOS; full source được encode một lần để tránh đếm trùng. Marginal overhead là chênh lệch exact base minus review block, không phải cộng các token counts độc lập.

Ceiling5.000 và caption allowances0/250/500/750/1000/1250/1500 là kịch bản thăm dò. Caption allowance bao gồm toàn bộ phần caption tăng thêm (headers/markers/IDs/separators/boundary effects); **chưa có caption thật**, không suy ra số ảnh/caption hoặc chất lượng. Observed fit dùng selected blocks thực tế; conservative envelope dùng declared review budget+measured marginal overhead+allowance. Final template/allocations/selector/manifest vẫn pending; không đổi/cắt review để làm các kịch bản fit.
''')
add('code','''
TA=ROOT/'outputs/stage_c/led_source_token_allocation_audit_v1'
ta_validation=load_json('outputs/stage_c/led_source_token_allocation_audit_v1/validation_report.json')
ta_findings=load_json('outputs/stage_c/led_source_token_allocation_audit_v1/allocation_findings.json')
assert ta_validation['validation_passed'] and ta_validation['source_rows_verified']==19800
ta_summary=checked_csv('outputs/stage_c/led_source_token_allocation_audit_v1/caption_capacity_summary.csv')
ta_ledger=checked_csv('outputs/stage_c/led_source_token_allocation_audit_v1/source_token_ledger.csv')
display(ta_summary[ta_summary.caption_allowance.eq(0)][['metadata_profile','review_budget','retained_mean','marginal_overhead_median','marginal_overhead_p95','marginal_overhead_max','exact_text_base_max','caption_capacity_min','caption_capacity_median','source_only_overflows']])
ta_focused=ta_summary[ta_summary.metadata_profile.eq(SOURCE_METADATA_PROFILE_TO_VIEW)]
display(ta_focused[['review_budget','caption_allowance','observed_fit','observed_overflow','conservative_envelope_fit','observed_allocated_total_max','conservative_envelope_total_max']])
display(pd.DataFrame([ta_findings['pilot_option_not_adopted']]))
display(pd.DataFrame([ta_validation]))
for dimension in ['split','volume_band']:display(checked_csv(f'outputs/stage_c/led_source_token_allocation_audit_v1/capacity_by_{dimension}.csv'))
ta_rows=ta_ledger[ta_ledger.metadata_profile.eq(SOURCE_METADATA_PROFILE_TO_VIEW)&ta_ledger.review_budget.eq(REVIEW_BUDGET_TO_VIEW)].copy()
ta_rows['hypothetical_allowance_fits']=ta_rows.caption_capacity.ge(CAPTION_ALLOWANCE_TO_VIEW)
display(ta_rows[['business_id','name','retained_reviews','review_block_tokens','marginal_source_overhead','text_only_base_tokens','caption_capacity','hypothetical_allowance_fits']].sort_values(['caption_capacity','business_id']).head(20))
display(ta_rows[ta_rows.business_id.eq(BUSINESS_ID_TO_VIEW)])
print('This capacity test removes no review. Actual caption-conditioned lengths and final parameters are null.')
ta_examples=load_json('outputs/stage_c/led_source_token_allocation_audit_v1/source_examples.json')
display(checked_csv('outputs/stage_c/led_source_token_allocation_audit_v1/example_businesses.csv')[['name','review_budget','retained_reviews','review_block_tokens','marginal_source_overhead','text_only_base_tokens','caption_capacity']])
for example in ta_examples:
    assert not example['final_model_input'] and example['actual_captions'] is None
    display(HTML('<details><summary>'+html.escape(f"{example['name']} | budget {example['review_budget']} | EXACT text-only base {example['text_only_base_tokens']} | reasons {example['reasons']}")+'</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape(example['complete_text_only_source'])+'</pre><p>Provisional source example; no real captions or frozen model input.</p></details>'))
fig,axes=plt.subplots(1,2,figsize=(13,4))
part=ta_ledger[ta_ledger.review_budget.eq(4000)]
profiles=['no_metadata_control','name_categories','name_categories_location']
axes[0].boxplot([part.loc[part.metadata_profile.eq(p),'marginal_source_overhead'] for p in profiles],showfliers=True)
axes[0].set_xticks([1,2,3]);axes[0].set_xticklabels(['No business metadata','Name/categories','Name/categories/location'])
axes[0].set(title='Additional source overhead (review budget: 4,000)',ylabel='Full base minus marked review-block tokens');axes[0].tick_params(axis='x',labelrotation=15)
heat=ta_summary[ta_summary.metadata_profile.eq('name_categories_location')].pivot(index='review_budget',columns='caption_allowance',values='observed_fit_pct').sort_index()
im=axes[1].imshow(heat.values,aspect='auto',vmin=0,vmax=100,cmap='Blues')
axes[1].set_xticks(range(len(heat.columns)),heat.columns);axes[1].set_yticks(range(len(heat.index)),heat.index)
axes[1].set(title='Name/categories/city/state: fit within 5,000 (%)',xlabel='Hypothetical entire caption-block allowance',ylabel='Review-block budget')
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):axes[1].text(j,i,f'{heat.iloc[i,j]:.1f}',ha='center',va='center',color='white' if heat.iloc[i,j]>55 else 'black',fontsize=9)
fig.colorbar(im,ax=axes[1],label='% businesses with sufficient capacity');save(fig,'led_source_token_allocation_capacity.png')
print('Caption allowances are hypothetical; metadata is provisional. The4000+750 pilot option is not an adopted allocation. Real caption pilot and final source re-encoding still required.')
''')
add('markdown','''
## 9.7. Stage D1 — ảnh đọc được, duplicate indexing và overlap giữa splits

Spec004 kiểm tra toàn bộ30.027 ảnh/2.300 businesses bằng cùng quy trình file integrity/hash; test chỉ dùng cho kiểm tra cố định, không tune thresholds hoặc manual policy gallery. Verify container +full decode, raw SHA khớp cohort/Yelp gốc, EXIF/RGB pixel hash và pHash64 cosine32-round6. Không rewrite/xóa ảnh hoặc thay business/split. Shared readable pool giữ toàn bộ valid IDs cho unfiltered; duplicate handling thuộc filtered, policy chưa adopted.

Near pairs quét đủ within-business, thresholds0/4/8/12 chỉ diagnostics. pHash collision không chứng minh cùng pixels/semantic identity; connected components có thể chain nên endpoints vượt threshold. Global approximate-distance scan chưa thực hiện. Cross-split raw/pixel identity là repeated visual evidence cần quyết định xử lý trước final caption/training/evaluation.

Counterfactual dưới đây chỉ tính thử: bảo vệ test, bỏ bản sao train nếu exact pixels có ởvalidation/test, bỏ validation nếu có ởtest. **Chưa áp dụng**; source pools/cohort/splits giữ nguyên. Các business còn0/dưới5 ảnh trong phương án giả định cần handling rule được chốt, không tự thay business hoặc deduplicate unfiltered thành filtered.
''')
add('code','''
PH=ROOT/'outputs/stage_d/photo_integrity_duplicate_audit_v1'
ph_validation=load_json('outputs/stage_d/photo_integrity_duplicate_audit_v1/validation_report.json')
ph_proposal_validation=load_json('outputs/stage_d/photo_integrity_duplicate_audit_v1/overlap_proposal_validation.json')
ph_findings=load_json('outputs/stage_d/photo_integrity_duplicate_audit_v1/photo_audit_findings.json')
assert ph_validation['validation_passed'] and ph_proposal_validation['status']=='PASS'
ph_manifest=checked_csv('outputs/stage_d/photo_integrity_duplicate_audit_v1/image_manifest.csv')
ph_business=checked_csv('outputs/stage_d/photo_integrity_duplicate_audit_v1/business_photo_counts.csv')
ph_groups=checked_csv('outputs/stage_d/photo_integrity_duplicate_audit_v1/within_business_duplicate_groups.csv')
display(checked_csv('outputs/stage_d/photo_integrity_duplicate_audit_v1/split_photo_summary.csv'))
display(pd.DataFrame([ph_validation]));display(pd.DataFrame([ph_findings]))
display(checked_csv('outputs/stage_d/photo_integrity_duplicate_audit_v1/unreadable_images.csv'))
display(ph_manifest.groupby(['image_format','original_mode']).size().rename('photos').reset_index())
display(ph_groups.groupby(['metric','hamming_threshold']).agg(groups=('group_id','size'),photos=('photos','sum'),chained=('chained_beyond_threshold','sum')).reset_index())
display(ph_business[ph_business.below5_pixel_unique][['business_id','split','source_photos','readable_photos','pixel_exact_unique']])
display(checked_csv('outputs/stage_d/photo_integrity_duplicate_audit_v1/cross_split_group_summary.csv'))
ph_impact=checked_csv('outputs/stage_d/photo_integrity_duplicate_audit_v1/hypothetical_overlap_business_impact.csv')
display(pd.DataFrame(ph_findings['hypothetical_overlap_policy_NOT_APPLIED']['by_split']))
display(ph_impact[ph_impact.hypothetical_zero][['business_id','name','split','current_readable','hypothetical_removed','hypothetical_remaining']])
ph_sensitivity=load_json('outputs/stage_d/photo_integrity_duplicate_audit_v1/test_image_disjoint_diagnostic_summary.json')
display(pd.DataFrame([ph_sensitivity]))
overlap_policy=yaml.safe_load((ROOT/'configs/image_overlap_evaluation_policy.yaml').read_text(encoding='utf-8'))
assert overlap_policy['preserve_readable_photo_pool'] and not overlap_policy['cross_split_image_exclusion_applied']
print('D-022 approved: preserve cohort/pools/splits; main100 plus secondary83 exact-pixel-disjoint test businesses. Evaluation not run; near/pretraining overlap not ruled out.')
print('785-image exclusion remains hypothetical and is NOT applied. No source photos, cohort or splits changed.')
fig,axes=plt.subplots(1,2,figsize=(13,4))
axes[0].hist(ph_business.readable_photos,bins=35,color='#2563eb');axes[0].set(title='Readable photos per business (all2300)',xlabel='Readable original IDs',ylabel='Businesses')
labels=['Readable','Exact pixel','pHash0','pHash4','pHash8','pHash12']
counts=[ph_business.below5_readable.sum(),ph_business.below5_pixel_unique.sum(),*[ph_business[f'below5_near_t{t}'].sum() for t in [0,4,8,12]]]
axes[1].bar(labels,counts,color=['#2563eb','#64748b','#7c3aed','#7c3aed','#7c3aed','#7c3aed']);axes[1].set(title='Below5 in diagnostic views (no exclusion applied)',ylabel='Businesses')
for i,count in enumerate(counts):axes[1].text(i,count+.25,str(count),ha='center')
save(fig,'photo_integrity_duplicate_counts.png')
from render_photo_pair_gallery import figures as photo_pair_figures
for fig,name in photo_pair_figures():save(fig,name)
ph_notes=load_json('outputs/stage_d/photo_integrity_duplicate_audit_v1/visual_pair_review_notes.json')
display(pd.DataFrame(ph_notes['pairs']));print(ph_notes['limitations'])
print('All30027 photos valid;227 exact cross-split groups remain documented. D-022 retains original pools; no teacher/LED training in this photo audit.')
''')
add('markdown','''
## 10. Xem originals/filtering/v3/v4/v4.1/v4.2 trong cùng notebook

Ô dưới mặc định hiển thị full text review v4.2 expanded của business đã chọn. Sửa ID/budget/page/stage ở ô cấu hình rồi Run All: `v4_2_expanded_selected` và `v4_2_expanded_control_selected` có dữ liệu cho toàn bộ 2.200 train/validation businesses; các stage `v4_selected`, `v4_1_selected`, `v4_2_selected`, `v4_2_control_selected` vẫn là pilot 80. `v3_selected` là kết quả expanded v3 trước đó. Viewer tương tác (nếu ipywidgets có sẵn) cho phép chọn tên/ID, stage và page, rồi bấm **Xem review**. Không chạy lại detector hoặc selection. `English_removed` là review không vào English pool; review English không được C2 chọn vẫn giữ trong source pool. Stage chưa có selection sẽ báo rõ; không suy ra review bị loại. Test chưa có C2 selections.
''')
add('code','''
render_business(BUSINESS_ID_TO_VIEW,REVIEW_BUDGET_TO_VIEW,VIEW_STAGE_TO_VIEW,VIEW_PAGE,PAGE_SIZE,metadata,cohort)
optional_viewer(metadata,cohort,BUSINESS_ID_TO_VIEW)
print("Manual call: render_business(BUSINESS_ID_TO_VIEW,3500,'English_removed',0,20,metadata,cohort)")
''')
add('markdown','''
## 10.1. BLIP-2 E0 — pilot nhỏ và trạng thái thực tế

D-022 đã chốt giữ nguyên cohort/pool/split và thêm secondary evaluation83testbusiness không có exact pixel reuse, cạnh main100. Không loại785ảnh và không tạo14zero-imagecases. External/generalization và30businesshuman evaluation giữ nguyên.

E0 dùng checkpoint `Salesforce/blip2-opt-2.7b-coco`, beam3,8train/validation businesses ×3ảnh (không phải finalK). Agent quản lý thực thi và kiểm tra; **Run All mặc định đọc kết quả**, không tải/chạy lại model. Nếu chủ động chạy/resume E0, đổi `RUN_CAPTION_PILOT=True` ở ô đầu. Lần đầu sẽ tạo `.venv-caption-pilot`, cài dependencies riêng và tải model vào `assets/model_cache`; không sửa preprocessing environment. FP16 GPU/CPU offload, không quantization, log VRAM/time/device map và pinned commit. Không bảo đảm local6GB feasibility trước kết quả load/generate.

Caption raw giữ nguyên; min5words/punctuation/exact generic phrases là surface diagnostics trong pilot, grammar/information phải xem thủ công. Failure retry một lần bằng prompt đã ghi; không tự thêm dấu câu để tạo pass. Chưa CLIP/relevance/semantic filtering, GPTteacher/LEDtrain hoặc finalcaption sets. Source4000+750 vẫn provisional, actualfullsource sẽ encode một lần.

Kiểm tra tài nguyên trước khi resume 2026-10-06: không có kernel/model job sót; Codex Diff từng giữ~9GiB private committed memory (khác RAM resident), Pylance~1,8GiB resident. Sau đó renderer private giảm còn~0,5GiB và **preflight đã PASS với9,123GiB RAM trống**; chưa xác định nguyên nhân bộ nhớ tăng trước đó. Workspace đã giảm indexing/file watching. Meta probe cũ 7,17GiB weights FP16/Q-Former FP32 và device map là ước lượng tại thời điểm RAM thiếu, không phải kết quả load hiện tại.

**D-024: người dùng chọn thử nhỏ trên local6GiB trước.** Đã ghim commit `cda95b9319b722f79c9451a3d8ff92eea02048dc`,2safetensors shards~15,496GB trên đĩa (khác footprint FP16 runtime); dù chỉ1ảnh vẫn cần đủ weights. Một controller nhẹ tải4MiB chunks có resume/SHA256 trong4I/O workers, không dùng GPU trong lúc tải. Sau khi fullLFS SHA256 PASS, tự kiểm lại RAM/CUDA, chạy `--max-new-photos 1` rồi giải phóng model/dừng để xem ảnh-caption/VRAM/time. Không đổi bộ24ảnh, model/precision/beam/source budgets, không tự thuêGPU/train/fullcorpus. Download deadline24h, partialchunks giữ để resume; resource/model failure được ghi thật, không autoquant. Report/tiến độ live ở `local_smoke_controller_report.json`/`weight_download_progress.json`; **không coi đang tải hoặc lên lịch inference là smoke PASS**. Run All dưới chỉ đọc reports, không khởi động job thứ hai.

**Trạng thái mới nhất D-025:** Người dùng không thể chờ đường tải local. Controller local **đã dừng**,~3GBverifiedchunks giữ lại,0model load/0caption. Gói cloudpilot **2,08MB**,24ảnh/8business/213complete reviews, đã đối chiếu dữ liệu gốc/token counts/ZIPhash PASS. Đây là sourceSLICE, không phải datasettrain/finalmanifest; model/beam/precision/budgets/IDorder giữ nguyên. Chờ account/budget/offer để chạy trên máy chủ, **chưa thuê/upload**. Máy chủ vẫn tải weights ở đó; không hứa thời gian tải. Một ảnh đầu trước, review actualcaption/VRAM/time rồi mới full24. Cloudruntime không chứng minh RTX3050 chạy được. Gói/README/scripts ở `cloud_handoff_20261006/e0_cloud_pilot.zip`; notebook chỉ đọc reports, không cần RunAll để thuê hoặc tạo job.

**Yêu cầu mới — notebook thử trực tiếp:** [08_blip2_caption_smoke_test.ipynb](08_blip2_caption_smoke_test.ipynb) cho thử1ảnh trong kernel, xemảnh/rawcaption/VRAM/time và cache có sẵn. Download mặc địnhOFF, không tạo backgroundjob; MODEL_ID mặc định giữCOCO/beam3, có thể chọn model/path của lần thử cũ nhưdiagnosticriêng. Script lịch sử `_archive_old/blip2_trials` dùngnonCOCO/beam1/max20/FP32, khác currentthesisconfig. Kết quả08 không tự nhậpformalE0/teacher/train. Notebook00 vẫn là index dataset/filtering/progress chính.

**Kết quả mới 07/10 sau reset máy:** Weights15,496GB đã tải đủ vàfullLFS SHA256 PASS. Mẫu30s đo18,87MB/s, không chứngminh nguyênnhân tăng tốc. **Một ảnh COCO thực sự chạy được trên RTX3050 6GiB với GPU+CPUoffload**, peakallocated3,868GiB/reserved3,906GiB. Model load105,656s; caption đầu16,478s, retry46,436s. Rawcaption về wraps/bowl of fruit khớp các vật thể chính trong ảnh; lần đầu thiếu dấu câu, retryprompt cho text rỗng, surfaceeligible=False. Không sửa text/prompt/rules để tạo pass. Model đã giải phóng; mới1/24ảnh, không quality/capacity/productionA2/finalmanifestPASS. Xem output thật và validationmẫu1 bên dưới; bước tiếp là pilot24/review cácflags và allocation, chưateacher/train.

API references: [BLIP-2 v4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/blip-2), [Accelerate offload](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling).
''')
add('code','''
from caption_basic_quality_notebook import configuration as caption_configuration,ensure_environment,run_step
EP=ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'
BQP=ROOT/'outputs/stage_e/blip2_caption_basic_quality_pilot_v2'
if RUN_CAPTION_PILOT:
    caption_python=ensure_environment(allow_install=ALLOW_CAPTION_SETUP_AND_DOWNLOAD)
    display(pd.DataFrame([run_step(caption_python,'prepare')]))
    display(pd.DataFrame([run_step(caption_python,'preflight')]))
    display(pd.DataFrame([run_step(caption_python,'generate')]))
    display(pd.DataFrame([run_step(caption_python,'report')]))
    display(pd.DataFrame([run_step(caption_python,'validate')]))
else:
    print('Cached E0 history + active E1 Basic Quality view. No dependency installation/model download/inference triggered by this Run All.')
print('Historical E0 reports — original punctuation-fail policy:')
for name in ['pilot_preparation_report.json','environment_report.json','download_status.json','weight_download_progress.json','model_load_report.json','generation_report.json','pilot_report.json','validation_report.json','diagnostic_summary.json']:
    path=EP/name
    if path.is_file():
        report=json.loads(path.read_text(encoding='utf-8'))
        display(pd.DataFrame([{'report':name,**report}]))
    else:print(name,': pending / not produced')
for name in ['last_step_failure.json','preflight_failure.json','generate_failure.json','report_failure.json']:
    if (EP/name).is_file():display(pd.DataFrame([{'failure_report':name,**json.loads((EP/name).read_text(encoding='utf-8'))}]))
print('Reports distinguish preparation/preflight/load/actual generation/manual review. No pending step is called PASS.')
print('Active E1 — adopted minimal Basic Quality with fresh EOS evidence:')
for name in ['pilot_preparation_report.json','environment_report.json','model_load_report.json','generation_report.json','pilot_report.json','validation_report.json','actual_caption_review_report.json']:
    path=BQP/name
    if path.is_file():display(pd.DataFrame([{'report':name,**json.loads(path.read_text(encoding='utf-8'))}]))
    else:print('E1',name,': pending')
for name in ['package_preparation_report.json','package_validation_report.json']:
    path=EP/'cloud_handoff_20261006'/name
    if path.is_file():display(pd.DataFrame([{'cloud_handoff_report':name,**json.loads(path.read_text(encoding='utf-8'))}]))
one_photo_path=EP/'after_reboot_20261007/one_photo_validation_report.json'
if one_photo_path.is_file():display(pd.DataFrame([json.loads(one_photo_path.read_text(encoding='utf-8'))]))
resource_path=EP/'resource_audit_20261006/model_memory_estimate.json'
if resource_path.is_file():
    resource=json.loads(resource_path.read_text(encoding='utf-8'))
    resource_fields=['status','method','available_RAM_GiB','effective_weight_GiB','GPU_free_GiB','current_requires_disk_offload','declared_RAM_gate_GiB','gate_requires_disk_offload','model_weights_downloaded','model_weights_loaded','caption_generation_started']
    display(pd.DataFrame([{'E0_memory_estimate':key,'value':resource.get(key)} for key in resource_fields]))
    print(resource.get('placement_estimate_limitations',resource.get('error','')))
''')
add('markdown','''
## 10.2. Caption thật, surface flags và actual LED source tokens

Chỉ hiển thị nếu E0 đã tạo đủ outputs. `all_raw_pilot_captions` dùng để đo capacity; `surface_eligible` chỉ một view basic diagnostics, **không phải filtered condition**. Không suy rộng24ảnh/8business sang toàn corpus hoặc chốtK/sourcebudget trước review. Ảnh/captions cùng IDs và source reviews nguyên vẹn được lưu trong receipts; checkpoint tương thích có thể resume.

Nếu mới hoàn tất smoke1, hiển thị ảnh/rawcaption và cả hai attempt trước; capacity của full24 vẫn pending.
''')
add('code','''
if (EP/'generation_report.json').is_file() and not (EP/'raw_caption_statistics.csv').is_file():
    partial=json.loads((EP/'generation_report.json').read_text())
    if partial.get('completed',0)>0:
        fixed_plan=json.loads((EP/'pilot_photo_plan.json').read_text())
        first_path=EP/'caption_checkpoints'/(fixed_plan[0]['photo_id']+'.json')
        if first_path.is_file():
            first_caption=json.loads(first_path.read_text())
            display(pd.DataFrame(first_caption['attempts']))
            print('ACTUAL RAW CAPTION:',first_caption['raw_caption'],'; full24pilot/sourcecapacity remains pending.')
            from PIL import Image,ImageOps
            import textwrap
            with Image.open(ROOT/first_caption['image_path']) as source_image:preview=ImageOps.exif_transpose(source_image).convert('RGB').copy()
            fig,ax=plt.subplots(figsize=(9,6));ax.imshow(preview);ax.axis('off')
            ax.set_title(textwrap.fill(first_caption['raw_caption'],80))
            save(fig,'blip2_actual_one_photo_smoke.png')
caption_complete=(EP/'pilot_report.json').is_file() and json.loads((EP/'pilot_report.json').read_text(encoding='utf-8')).get('status') in ['generated_outputs_pending_review','pilot_validated']
if caption_complete:
    ep_captions=pd.read_csv(EP/'raw_caption_statistics.csv',keep_default_na=False)
    ep_sources=pd.read_csv(EP/'pilot_source_token_ledger.csv',keep_default_na=False)
    attempt_rows=[]
    for photo in json.loads((EP/'pilot_photo_plan.json').read_text(encoding='utf-8')):
        record=json.loads((EP/'caption_checkpoints'/(photo['photo_id']+'.json')).read_text(encoding='utf-8'))
        for number,attempt in enumerate(record['attempts'],1):
            attempt_rows.append({'name':record['name'],'photo_id':record['photo_id'],'attempt':number,'chosen':number==record['chosen_attempt'],**attempt})
    print('All actual attempts, including empty retries; raw text preserved:')
    display(pd.DataFrame(attempt_rows))
    display(ep_captions[['name','split','photo_id','raw_caption','word_count','LED_body_tokens','attempts','surface_eligible','surface_flags','seconds_all_attempts','peak_GPU_allocated_GiB']])
    display(ep_sources)
    if (EP/'manual_caption_review_notes.json').is_file():display(pd.DataFrame(json.loads((EP/'manual_caption_review_notes.json').read_text())['cases']))
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    axes[0].hist(ep_captions.LED_body_tokens,bins=12,color='#2563eb');axes[0].set(title='Actual raw-caption LED BODY tokens (pilot)',xlabel='Tokens, no source markers/specials',ylabel='Photos')
    raw_sources=ep_sources[ep_sources.condition.eq('all_raw_pilot_captions_capacity_only')].reset_index(drop=True)
    axes[1].bar(range(len(raw_sources)),raw_sources.text_only_base_tokens,label='Review + provisional metadata',color='#64748b')
    axes[1].bar(range(len(raw_sources)),raw_sources.actual_incremental_caption_block_tokens,bottom=raw_sources.text_only_base_tokens,label='Actual incremental caption block',color='#059669')
    axes[1].axhline(5000,linestyle='--',color='#dc2626',label='Diagnostic source ceiling5000')
    axes[1].set(title='Full source encoded once, unchanged whole reviews',xlabel='Eight pilot businesses',ylabel='Exact full-source tokens');axes[1].legend(fontsize=7)
    save(fig,'blip2_caption_pilot_tokens.png')
    from render_caption_pilot_gallery import figures as actual_caption_figures
    for fig,name in actual_caption_figures():save(fig,name)
    draft_dir=ROOT/'outputs/stage_e/basic_caption_quality_draft_v1'
    if (draft_dir/'comparison_report.json').is_file():
        print('HISTORICAL BASIC QUALITY DRAFT — offline replay only. D-026 now adopts minimal Basic Quality for actual E1 validation below; E0 outputs unchanged.')
        print('Whitespace/punctuation normalization; basic failures stay separate from Stage F substantive filtering. Rule developed on these train/validation outputs; not independent quality evaluation.')
        display(pd.DataFrame([json.loads((draft_dir/'comparison_report.json').read_text(encoding='utf-8'))]))
        draft_records=json.loads((draft_dir/'candidate_records.json').read_text(encoding='utf-8'))
        display(pd.DataFrame(draft_records)[['name','photo_id','legacy_raw_caption','raw_caption','normalized_caption','initial_basic_fail','draft_chosen_attempt','draft_basic_eligible']])
        display(pd.DataFrame(json.loads((draft_dir/'normalized_source_token_ledger.json').read_text(encoding='utf-8'))))
        print('Legacy stop/EOS metadata unknown; E1 captures actual evidence under specs/005b_basic_quality_validation_pilot.md.')
else:
    print('No complete caption pilot yet. See preparation/environment/failure reports above; no invented captions or results.')
if (BQP/'caption_statistics.csv').is_file() and (BQP/'pilot_report.json').is_file():
    print('ACTUAL E1 PILOT — raw preserved, normalized separately, EOS/token cap captured; substantive filtering not applied.')
    bq_stats=pd.read_csv(BQP/'caption_statistics.csv',keep_default_na=False)
    bq_sources=pd.DataFrame(json.loads((BQP/'source_token_ledger.json').read_text(encoding='utf-8')))
    display(bq_stats)
    bq_attempts=[]
    for photo in json.loads((BQP/'pilot_photo_plan.json').read_text(encoding='utf-8')):
        record=json.loads((BQP/'caption_checkpoints'/(photo['photo_id']+'.json')).read_text(encoding='utf-8'))
        for number,attempt in enumerate(record['attempts'],1):
            bq_attempts.append({'name':record['name'],'photo_id':record['photo_id'],'attempt':number,'chosen':number==record['chosen_attempt'],**attempt})
    display(pd.DataFrame(bq_attempts));display(bq_sources)
    if (BQP/'actual_caption_review_report.json').is_file():
        display(pd.DataFrame(json.loads((BQP/'actual_caption_review_report.json').read_text(encoding='utf-8'))['cases']))
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    axes[0].hist(bq_stats.normalized_LED_body_tokens,bins=12,color='#2563eb')
    axes[0].set(title='Actual E1 normalized-caption LED BODY tokens',xlabel='Tokens',ylabel='Photos')
    basic_sources=bq_sources[bq_sources.condition.eq('basic_eligible_normalized_NOT_final_caption_set')].reset_index(drop=True)
    axes[1].bar(range(len(basic_sources)),basic_sources.text_only_base_tokens,label='Unchanged review + metadata',color='#64748b')
    axes[1].bar(range(len(basic_sources)),basic_sources.actual_incremental_caption_block_tokens,bottom=basic_sources.text_only_base_tokens,label='Basic-eligible normalized captions',color='#059669')
    axes[1].axhline(5000,linestyle='--',color='#dc2626',label='Diagnostic5000 ceiling')
    axes[1].set(title='Actual E1 full-source capacity, not final allocation',xlabel='Eight pilot businesses',ylabel='Exact tokens');axes[1].legend(fontsize=7)
    save(fig,'blip2_basic_quality_tokens.png')
    from render_basic_quality_caption_gallery import figures as basic_quality_figures
    for fig,name in basic_quality_figures():save(fig,name)
else:print('Actual E1 full outputs pending; no substitute or invented captions.')
k_draft_dir=ROOT/'outputs/stage_f/k_validation_protocol_draft_v1'
if (k_draft_dir/'validation_inventory_report.json').is_file():
    print('NEXT: proposed validation-only K study — metadata inventory only, no K results or filter scores yet.')
    k_inventory=json.loads((k_draft_dir/'validation_inventory_report.json').read_text(encoding='utf-8'))
    k_protocol=yaml.safe_load((ROOT/'configs/k_validation_protocol_draft.yaml').read_text(encoding='utf-8'))
    display(pd.DataFrame([k_inventory]))
    display(pd.DataFrame([{'candidate_K_proposed':K,'raw_validation_businesses_with_fewer_than_K_photos':k_inventory['raw_pool_shortage_counts'][str(K)],'semantic_coverage_NOT_COMPUTED':None,'redundancy_NOT_COMPUTED':None,'caption_or_source_tokens_NOT_COMPUTED':None,'K_frozen':False} for K in k_protocol['candidate_K']]))
    print('Raw2504photos/200validation businesses; current actual pilot only4validation businesses x3photos. Full eligible pools, StageF thresholds/groups/diversity and budgets must be fixed before K sweep. Same group denominator for all K; zero-group coverageNA; budget both conditions. Smallest feasible K >=95% of best candidate coverage; not absolute95% of visual information.')
    print('Protocol proposed only: specs/006_caption_filtering_K_validation_draft.md; final K/thresholds/budgets remain pending, no test/model calls.')
scoring_review_dir=ROOT/'outputs/stage_f/scoring_definition_review_v1'
if (scoring_review_dir/'design_review_report.json').is_file():
    print('HISTORICAL STAGE F SCORING REVIEW — prior to D-027; proposed flags retained as history, active state below.')
    scoring_review=json.loads((scoring_review_dir/'design_review_report.json').read_text(encoding='utf-8'))
    proposed_scoring=yaml.safe_load((ROOT/'configs/stage_f_scoring_review.yaml').read_text(encoding='utf-8'))
    display(pd.DataFrame([scoring_review]))
    display(pd.DataFrame([{'role':'CLIP consistency','proposed_model':'openai/clip-vit-base-patch32','formula':'cosine normalized image-caption features','approved':False},{'role':'Redundancy/diversity','proposed_model':'sentence-transformers/all-mpnet-base-v2','formula':'cosine normalized caption embeddings','approved':False},{'role':'Domain relevance candidate A','proposed_model':'reuse same SBERT','formula':'max positive prototype cosine minus max negative prototype cosine','approved':False}]))
    display(pd.DataFrame([{'raw_photo_count_band':band,'available_validation_businesses':count,'proposed_calibration_businesses':10,'roles_assigned':False} for band,count in scoring_review['raw_photo_count_strata'].items()]))
    print('Proposed40 calibration +160 K-study at business level; no roles assigned. Human calibration labels are separate from final30-business evaluation. Business-disjoint does not guarantee image-disjoint.')
    print('Coverage decision pending: one representative per reference group gives min(K,G)/G, independent of diversity ranking. Disclose count coverage or approve separate coarser reference groups. See specs/006a_stage_f_scoring_and_calibration_review.md.')
SFOUT=ROOT/'outputs/stage_f/validation_calibration_v1'
if (SFOUT/'preparation_report.json').is_file():
    print('ACTIVE D-027 STAGE F — fixed domain relevance/coarser groups/models/roles; actual human thresholds and K pending.')
    for name in ['preparation_report.json','model_files_report.json','setup_validation_report.json','generation_report.json','worker_progress.json','human_label_import_report.json']:
        path=SFOUT/name
        if path.is_file():display(pd.DataFrame([{'report':name,**json.loads(path.read_text(encoding='utf-8'))}]))
    role_frame=pd.DataFrame(json.loads((SFOUT/'business_roles.json').read_text(encoding='utf-8')))
    display(role_frame)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    role_businesses=role_frame.groupby('role').size();role_photos=role_frame.groupby('role').raw_photo_count.sum()
    axes[0].bar(role_businesses.index,role_businesses.values,color=['#64748b','#059669']);axes[0].set(title='Frozen internal roles — original validation200 unchanged',ylabel='Businesses')
    axes[1].bar(role_photos.index,role_photos.values,color=['#64748b','#059669']);axes[1].set(title='Original photo IDs before Basic/substantive filters',ylabel='Raw validation photo IDs')
    save(fig,'stage_f_validation_roles.png')
    for folder in ['scorer_smoke','scorer_smoke_v2','scores_calibration','scores_K_study']:
        path=SFOUT/folder/'scoring_report.json'
        if path.is_file():display(pd.DataFrame([{'actual_scoring_scope':folder,**json.loads(path.read_text(encoding='utf-8'))}]))
    for folder in ['human_check_preview','human_check']:
        path=SFOUT/folder/'queue_report.json'
        if path.is_file():
            display(pd.DataFrame([{'annotation_scope':folder,**json.loads(path.read_text(encoding='utf-8'))}]))
            print('Open local file in browser:',str(SFOUT/folder/'human_check.html'))
    active_sf=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
    display(pd.DataFrame([{'parameter':k,'value':v} for k,v in active_sf['thresholds'].items()]))
    if active_sf.get('decision') in ['D-032','D-033']:
        print('ACTIVE D-033: final K=5 approved/frozen; production input construction authorized with unchanged D026/D030/D031/D032 rules.' if active_sf['decision']=='D-033' else 'ACTIVE D-032: budgets/template frozen; K pending approval.')
        current=ROOT/'outputs/stage_f/methodology_amendment_d032'
        manifest=current/'source_budget_manifest.json'
        if manifest.is_file():
            display(pd.DataFrame([json.loads(manifest.read_text(encoding='utf-8'))['budgets']]))
            print('Current budget authority:',str(manifest),'; old D-031filter manifest budget flag is historical, not this authority.')
        study=ROOT/'outputs/stage_f/k_sensitivity_d032_v1/K_report.json'
        if study.is_file():
            ks=json.loads(study.read_text(encoding='utf-8'))
            display(pd.DataFrame([{'K':r['K'],'macro_coverage':r['macro_coarser_group_coverage'],'relative_to_best':r['relative_coverage_vs_best'],'meets95pct_best':r['meets95pct_best'],'both_condition_feasible':r['budget_feasible_both_conditions']} for r in ks['K_table']]))
            display(pd.DataFrame([{'K':r['K'],'condition':condition,'retained_mean':s['actual_captions_retained']['mean'],'retained_total':s['actual_captions_retained']['total'],'fewer_than_K':s['businesses_fewer_than_K_available'],'empty':s['empty_businesses'],'changed_record_sets':s['changed_caption_record_sets_from_previous_K'],'changed_unique_text_sets':s['changed_unique_caption_text_sets_from_previous_K'],'caption_mean':s['caption_block_tokens']['mean'],'caption_p95':s['caption_block_tokens']['p95'],'caption_max':s['caption_block_tokens']['max'],'source_mean':s['full_source_tokens']['mean'],'source_p95':s['full_source_tokens']['p95'],'source_max':s['full_source_tokens']['max'],'caption_overflows':s['caption_blocks_over750'],'source_overflows':s['full_sources_over5000']} for r in ks['K_table'] for condition,s in r['conditions'].items()]))
            print('Historical sensitivity evidence:',ks['rule_implied_K'],'; historical K_frozen=',ks['K_frozen'],'; CURRENT approved K=',active_sf['K'])
            print('Filtered macro coverage denominator:',ks['coverage_macro_denominator'],'; empty business retained withNAcoverage.')
            print('Complete K table:',str(study.with_suffix('.md')))
        if active_sf['decision']=='D-033':
            production=ROOT/'outputs/stage_f/production_caption_construction_d033_v1'
            for name in ['preparation_report.json','worker_progress.json','completion_report.json']:
                p=production/name
                if p.exists():display(pd.DataFrame([{'report':name,**json.loads(p.read_text(encoding='utf-8'))}]))
            generated=production/'generation_train_test/generation_report.json'
            if generated.exists():
                g=json.loads(generated.read_text(encoding='utf-8'))
                print('New production photo checkpoints:',g['completed'],'/',g['planned'],'; generation status:',g['status'])
            print('Completion report is available only after all2300business manifests validate; no teacher/training in this job.')
            cloud=ROOT/'outputs/stage_f/production_caption_construction_d033_cloud_v1'
            for name in ['preparation_report.json','benchmark_report.json','worker_progress.json','completion_report.json']:
                p=cloud/name
                if p.exists():display(pd.DataFrame([{'cloud_report':name,**json.loads(p.read_text(encoding='utf-8'))}]))
            bundle=ROOT/'outputs/cloud_handoff/stage_f_d033_v1/package_report.json'
            if bundle.exists():
                b=json.loads(bundle.read_text(encoding='utf-8'))
                print('Cloud transfer package:',b['archive'],'bytes:',b['archive_bytes'],'SHA256:',b['archive_sha256'])
                print('Package integrity passed; actual3090GPU smoke remains a separate check.')
            colab_review=ROOT/'outputs/stage_f/colab_smoke_review_20261010/review_report.json'
            if colab_review.exists():
                display(pd.DataFrame([json.loads(colab_review.read_text(encoding='utf-8'))]))
                print('Reviewed userColabnotebook output only; rawcheckpoint files/fullruntime not imported.')
            production_pointer=ROOT/'outputs/stage_f/production_manifest_pointer.json'
            if production_pointer.exists():
                pointer=json.loads(production_pointer.read_text(encoding='utf-8'))
                completion=json.loads((ROOT/pointer['completion_report']).read_text(encoding='utf-8'))
                print('CURRENT StageF COMPLETE independentlyvalidated. Final namespace:',pointer['production_directory'])
                display(pd.DataFrame([{'split':split,'businesses':r['businesses'],'generated':r['caption_pool_generated'],'Basic_pass':r['basic_pass'],'Basic_fail':r['basic_failures'],'Unfiltered_retained':r['conditions']['unfiltered']['retained_captions']['total'],'Filtered_retained':r['conditions']['filtered']['retained_captions']['total'],'zero_Filtered':len(r['conditions']['filtered']['zero_caption_businesses'])} for split,r in completion['split_reports'].items()]))
                validation=ROOT/'outputs/stage_f/colab_result_review_20261010/independent_validation_report.json'
                display(pd.DataFrame([json.loads(validation.read_text(encoding='utf-8'))]))
                print('Humanlabels/method unchanged; STOP beforeteacher/LEDtraining; shuffledassignmentlater.')
    if active_sf.get('decision')=='D-031':
        print('ACTIVE D-031: exact normalized-pixel collapse only; pHash diagnostic-only, CLIP soft-only. Caption thresholds unchanged; no K sensitivity yet.')
        display(pd.DataFrame([active_sf['image_duplicate_policy']]))
        current=ROOT/'outputs/stage_f/methodology_amendment_d031'
        p=current/'pre_k_distribution.json'
        if p.is_file():
            pk=json.loads(p.read_text(encoding='utf-8'))
            display(pd.DataFrame([{'stage':key,**{k:v for k,v in pk['summary'][key].items() if k!='businesses_below_K'},**{'businesses_below_'+k:v for k,v in pk['summary'][key]['businesses_below_K'].items()}} for key in pk['primary_stage_columns']]))
            display(pd.DataFrame(pk['businesses']))
            print('Grid review flag:',pk['grid_adequacy_review_needed'],'groups above10:',len(pk['groups_above_max_K_businesses']))
            print('Report returned for review; no K selected or candidate comparison executed. Caption/source budgets remain TBD.')
            print('Full pre-K report:',str(current/'pre_k_distribution.md'))
            print('Approved amendment/file changes:',str(current/'amendment_summary.md'))
    if active_sf.get('decision')=='D-030':
        print('ACTIVE D-030: CLIP soft-only for existing quality order/ties; no caption hard rejection by CLIP and no CLIP consistency threshold to freeze.')
        display(pd.DataFrame([active_sf['CLIP_consistency_policy']]))
        amended=ROOT/'outputs/stage_f/methodology_amendment_d030'
        p=amended/'frozen_remaining_hard_thresholds.json'
        if p.is_file():
            fr=json.loads(p.read_text(encoding='utf-8'))
            print('Remaining hard-criteria status:',fr['status'])
            display(pd.DataFrame([{'criterion':k,'threshold':v,'frozen':k in fr['frozen_criteria'],**fr['verified_label_balance'][k]} for k,v in fr['thresholds'].items()]))
            display(pd.DataFrame([{'criterion':k,**r['best_candidate'],'accepted_for_freeze':r['accepted_for_freeze']} for k,r in fr['search_tables'].items()]))
            if fr['blockers']:
                print('STOP BEFORE K: a required hard criterion failed its original acceptance condition; no new batch, grid change or acceptance relaxation.')
                display(pd.DataFrame(fr['blockers']))
            print('Amendment evidence and file changes:',str(amended/'amendment_summary.md'))
    assisted=ROOT/'outputs/stage_f/ai_assisted_calibration_v1'
    if (assisted/'ui_build_report.json').is_file():
        print('ACTIVE D-028: AI-assisted pre-annotation with explicit human verification; not independent human annotation.')
        for name in ['ai_annotation_progress.json','initial_selection.json','active_selection.json','ui_build_report.json','human_verification_import_report.json']:
            p=assisted/name
            if p.is_file():
                r=json.loads(p.read_text(encoding='utf-8'))
                display(pd.DataFrame([{'report':name,**{k:v for k,v in r.items() if k not in ['active_case_ids','pending_case_ids','selection','counts']}}]))
                if name=='human_verification_import_report.json' and 'counts' in r:
                    display(pd.DataFrame(r['counts']).T.rename_axis('verified_human_criterion'))
        print('USE THIS ASSISTED UI (old600-case form is historical):',str(assisted/'human_check.html'))
        print('D-029 supersedes the old Consistency minimum10-negative expansion gate. No additional batch required solely for that gate; existing pending cases remain unchanged. Accept/Correct per case; only verified non-uncertain human labels fit cutoffs.')
        print('Unverified AI suggestions are diagnostics only; final30-business human evaluation remains separate. No thresholds/K/budgets frozen yet unless separate real artifacts exist.')
        secondary=assisted/'active_secondary_review.json'
        if secondary.is_file():
            sr=json.loads(secondary.read_text(encoding='utf-8'))
            print('External ChatGPT second review: suggestions only, not human labels; displayed separately in assisted form.')
            display(pd.DataFrame([{'criterion':key,**pd.Series([r[key] for r in sr['cases'].values()]).value_counts().to_dict()} for key in ['consistency','relevance']]))
    stability_policy=yaml.safe_load((ROOT/'configs/stage_f_consistency_stability.yaml').read_text(encoding='utf-8'))
    stability=ROOT/stability_policy['output_directory']/'stability_report.json'
    if stability.is_file():
        st=json.loads(stability.read_text(encoding='utf-8'))
        print('HISTORICAL D-029 Consistency cutoff audit: retained evidence; numeric cutoffs below are not active production gates.' if active_sf.get('decision') in ['D-030','D-031','D-032','D-033'] else 'ACTIVE D-029: Consistency stability report only, researcher review required. No threshold freeze, K selection or automatic expansion.')
        display(pd.DataFrame([st['counts']]))
        display(pd.DataFrame([st['full_fit']]))
        display(pd.DataFrame(st['bootstrap_summary']['threshold_distribution']))
        display(pd.DataFrame([{'metric':k,**st['bootstrap_summary'][k]} for k in ['threshold','resample_fit_balanced_accuracy','original_reference_balanced_accuracy','original_prediction_flips']]))
        display(pd.DataFrame([{'removed_case_id':r['removed']['case_id'],'photo_id':r['removed']['photo_id'],'score':r['removed']['decision_score'],'threshold':r['selected_threshold'],'delta':r.get('threshold_delta'),'fold_BA':r.get('fold_fit',{}).get('balanced_accuracy'),'reference_BA':r.get('original_reference',{}).get('balanced_accuracy'),'prediction_flips':r.get('original_prediction_flips')} for r in st['leave_one_negative_out']]))
        print('These are apparent calibration sensitivity metrics; bootstrap percentiles are not guaranteed population confidence intervals. Actual negatives:',st['counts']['negative'],'; shared-business dependence may remain.')
        print('Read complete report:',str(stability.with_suffix('.md')))
        if (stability.parent/'consistency_audit.html').is_file():
            print('Current read-only image/caption audit:',str(stability.parent/'consistency_audit.html'))
    conservative=ROOT/'outputs/stage_f/consistency_conservative_gate_diagnostic_v1/diagnostic_report.json'
    if conservative.is_file():
        cd=json.loads(conservative.read_text(encoding='utf-8'))
        print('HISTORICAL CONSERVATIVE CLIP DIAGNOSTIC: retained evidence. No CLIP cutoff below is frozen or active.' if active_sf.get('decision') in ['D-030','D-031','D-032','D-033'] else 'USER-REQUESTED CONSERVATIVE CLIP DIAGNOSTIC: compare operating points only; no Stage F protocol amendment, threshold selection/freeze, new batch or K-study.')
        display(pd.DataFrame([cd['counts']]))
        display(pd.DataFrame(cd['threshold_table']))
        display(pd.DataFrame([{'comparison_rule':k,**{key:value for key,value in p.items() if key!='decision_case_ids'}} for k,p in cd['operating_points'].items() if p]))
        for rule,results in cd['bootstrap_summary'].items():
            print('Bootstrap rule:',rule,'; metrics at the fixed original cutoff:')
            display(pd.DataFrame([{'metric':key,**value} for key,value in results['fixed_operating_point_metrics'].items() if isinstance(value,dict)]))
            refit=results['refit_selection_rule']
            print('Refit feasible/infeasible draws:',refit['feasible_draws'],refit['infeasible_draws'])
            display(pd.DataFrame(refit['selected_threshold_distribution']))
        display(pd.DataFrame([{'removed_case_id':fold['removed_case']['case_id'],'rule':rule,'threshold':result['candidate']['threshold'],'delta':result['threshold_delta'],'consistent_recall':result['candidate']['consistent_recall'],'inconsistent_specificity':result['candidate']['inconsistent_recall_specificity'],'BA':result['candidate']['balanced_accuracy'],'original_prediction_flips':result['original_prediction_flips']} for fold in cd['leave_one_negative_out'] if fold['status']=='FITTED' for rule,result in fold['rules'].items() if result['candidate']]))
        print('Read diagnostic table, limitations and all accepted/rejected case IDs:',str(conservative.with_suffix('.md')))
        print('Exact operating-point JSON:',str(conservative.parent/'operating_points.json'))
        print('Conditional bootstrap summaries repeat the five observed negatives, not new independent observations. Frozen scorer/config definitions remain unchanged.')
    phash_diag=ROOT/'outputs/stage_f/phash_pair_diagnostic_v1/diagnostic_report.json'
    if phash_diag.is_file():
        ph=json.loads(phash_diag.read_text(encoding='utf-8'))
        print('HISTORICAL pHash diagnostic: retained evidence; cutoff35below is not a production gate.' if active_sf.get('decision') in ['D-031','D-032','D-033'] else 'pHash 0..64 EXPLORATORY DIAGNOSTIC, not a production-grid amendment. Old grid0/4/8/12 and image threshold pending remain unchanged.')
        display(pd.DataFrame([ph['counts']]))
        display(pd.DataFrame([ph['exploratory_best_not_selected']]))
        with pd.option_context('display.max_rows',None):display(pd.DataFrame(ph['threshold_table']))
        print('These are pair-classification counts, not images removed by the business-level greedy pipeline.')
        print('Read-only viewer of all16existing positive pairs, with separate Codex observations:',str(phash_diag.parent/'positive_pair_audit.html'))
        print('Full diagnostic report:',str(phash_diag.with_suffix('.md')))
        print('Human labels, scores, protocol and frozen remaining thresholds unchanged; no new annotation batch or K-study.')
    actual_budget=ROOT/'outputs/stage_f/actual_source_budget_freeze_audit_v1/audit_report.json'
    if actual_budget.is_file():
        ab=json.loads(actual_budget.read_text(encoding='utf-8'))
        print('ACCEPTED SOURCE-BUDGET AUDIT: D-032now formally freezes its exact budget/template; these historical audit values remain unchanged.' if active_sf.get('decision') in ['D-032','D-033'] else 'ACTUAL SOURCE-BUDGET AUDIT ONLY: Kmax10 stress, no coverage/K selection; D-032 production freeze still requires researcher approval.')
        display(pd.DataFrame([{'scope':scope,'condition':condition,**{'caption_'+k:v for k,v in stats['caption_block_tokens'].items()},**{'source_'+k:v for k,v in stats['complete_source_tokens'].items()},'caption_over750':stats['caption_blocks_over750'],'source_over5000':stats['full_sources_over5000']} for scope,conditions in ab['summary'].items() for condition,stats in conditions.items()]))
        display(pd.DataFrame([{'condition':condition,'extreme':name,'business_id':row['business_id'],'review_block':row['review_block_tokens_with_special'],'metadata_marker_increment':row['marginal_metadata_review_header_overhead'],'caption_increment':row['caption_block_increment_tokens'],'full_source':row['full_source_tokens']} for condition,cases in ab['worst_cases'].items() for name,row in cases.items()]))
        print('Review IDs/text/order unchanged; independent slow tokenizer ID checks:',ab['independent_slow_LEDTokenizer_full_ID_equality_checks'])
        print('No silent caption/review removal/truncation; no real shuffled assignment instantiated; no test data.')
        print('Report:',str(actual_budget.with_suffix('.md')),'; proposal:',str(actual_budget.parent/'D032_PROPOSAL.md'))
    print('The assisted annotation form hides scorer numbers. Caption/scoring pool is reused; no regeneration. Source/scorer config/splits/Unfiltered baseline stay intact. CLI workers never start from cachedRunAll.')
''')
add('markdown','''
## 11. Những bước sẽ làm — cập nhật tiếp vào notebook này

**D-033 Approved/Validated (2026-10-09): final K=5.** K3 fails the predefined rule; K5 reaches97.7479%ofbest and is the smallest qualifying budget-feasible value. The complete3/5/8/10sensitivity remains historical evidence, unchanged.

Production inputs now cover train2000/validation200/test100. Reuse2504validation checkpoints and12compatible training pilot checkpoints; generate27511remaining photos without K-based photo preselection. All D026generator settings, D030CLIPsoft ranking, D031exact-only duplicate collapse and fixed0.08/0.59/0.46thresholds remain unchanged. D0325000/4000/750and exact template remain frozen. Test processing applies the approved method without tuning or teacher targets; human test references are unused.

Current production artifacts/status: `outputs/stage_f/production_caption_construction_d033_v1/`. Caption generation and scoring are resumable. Do not declare StageFcomplete until final Unfiltered/Filtered manifests and whole-source budgets validate for all2300businesses. Stop before GPTteacher and LEDtraining. External/generalization and final30-business human evaluation remain separate and unchanged.

''')
add('code','''
pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text(encoding='utf-8'))
display(pd.DataFrame([{'section':section,'parameter':key,'value':json.dumps(value,ensure_ascii=False)} for section,values in pending.items() for key,value in values.items()]))
assert all(value is None for value in pending['review_selection'].values())
if active_sf.get('decision') in ['D-032','D-033']:
    assert pending['token_budget']['total_source_tokens']==5000
    assert pending['token_budget']['review_tokens']==4000
    assert pending['token_budget']['caption_tokens']==750
    assert pending['token_budget']['metadata_and_marker_tokens'] is None # Explicitly not an independent fixed count.
    assert active_sf['K']==5 if active_sf['decision']=='D-033' else active_sf['K'] is None
else:
    assert all(value is None for value in pending['token_budget'].values())
display(pd.DataFrame([v4_validation]));print('Notebook tổng hợp PASS. D-033K5 and D032source budgets frozen; production status shown above; teacher/training pending. Source data/splits unchanged.')
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.7'}},'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks/00_thesis_dataset_filtering_review_selection.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(f'Generated {path}; {sum(c["cell_type"]=="code" for c in cells)} code cells.')
