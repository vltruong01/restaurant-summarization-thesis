"""Create the reviewable C2 pilot notebook; regeneration clears saved outputs."""
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
cells=[]


def add(kind,source):
    source=textwrap.dedent(source).strip()+'\n'
    cell={'cell_type':kind,'id':f'c2-pilot-{len(cells):03d}','metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':
        compile(source,f'c2-pilot-cell-{len(cells)}','exec')
        cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown','''
# Stage C2 — Pilot chọn review đại diện (draft)

48 businesses **train/validation**; hai phương pháp **random** và **rating + temporal rank segments**, tối đa 30 review/business. Đây là diagnostic stress sample, không phải probability sample để ước lượng toàn cohort. Không có test businesses, final review manifest hay methodology đã khóa.

Hamilton quota giữ gần phân bố sao; mỗi rating có q nhóm thời gian bằng nhau về số review, chọn một review/nhóm. Full-population time bins được giữ cố định. JSD dùng log base 2 và là divergence, không phải căn bậc hai. Date-span coverage và rating là proxies, không thay thế manual aspect/opinion audit.

Budget/model semantic chưa chốt được ghi TBD/null; không thay bằng zero hoặc gọi initial selection là train-ready. Bản draft có nguyên văn review, markers/rating/date và BOS/EOS một lần khi đếm token. Mở bằng Python 3 (ipykernel), Run All; không cần GPU/model weights cho phần đang có.
''')
add('markdown','''
## 1. Cấu hình và bằng chứng pilot

`BUSINESS_ID_TO_VIEW` chỉ dùng xem ví dụ. Pipeline có identity/hash checks; không overwrite một run có config/identity khác. Thay đổi phương pháp cần version output directory.
''')
add('code','''
from pathlib import Path
import sys, json, html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, FileLink
import orjson, yaml

BUSINESS_ID_TO_VIEW=None
PROJECT_ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'configs/review_selection_pilot.yaml').is_file())
sys.path.insert(0,str(PROJECT_ROOT/'scripts'))
from review_selection_pilot import run_pilot
cfg=yaml.safe_load((PROJECT_ROOT/'configs/review_selection_pilot.yaml').read_text(encoding='utf-8'))
OUT=PROJECT_ROOT/cfg['output_directory']
FIGURES=OUT/'figures'; FIGURES.mkdir(exist_ok=True)
completion=run_pilot(PROJECT_ROOT)
display(pd.DataFrame([{k:completion[k] for k in ['businesses','population_reviews','selection_runs','test_businesses','semantic_status','final_review_token_budget','methodology_locked']}]))
sample=pd.read_csv(OUT/'pilot_businesses.csv',keep_default_na=False)
data=pd.read_csv(OUT/'selection_diagnostics.csv')
initial=data.loc[data['scenario'].eq('unadjusted')].copy()
pd.set_option('display.max_columns',25)
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
BANDS=['10-30','31-100','101-300','301+']
METHODS=['random','rating_time']
COLORS={'random':'#64748b','rating_time':'#2563eb'}

def save(fig,name):
    fig.tight_layout(); fig.savefig(FIGURES/name,dpi=150,bbox_inches='tight'); plt.show(); plt.close(fig)
''')
add('markdown','''
## 2. Các businesses pilot và lý do chọn

12/business-count band; ít nhất hai validation businesses mỗi band. Targeted cases: strongly positive, mixed ratings, rare negatives, long review texts, long/short histories. Seeded fill hoàn thiện từng nhóm. Không suy rộng mean của sample này thành mean của toàn bộ 2.300 businesses.
''')
add('code','''
display(pd.crosstab(sample['volume_band'],sample['split']).reindex(BANDS))
display(sample[['business_id','name','split','volume_band','english_reviews','positive_fraction','negative_fraction','rating_entropy','english_review_token_mean','history_days','pilot_reasons']])
assert len(sample)==48 and sample['business_id'].is_unique
assert set(sample['split'])<={'train','validation'}
assert sample.groupby('volume_band').size().eq(12).all()
print('Population là toàn bộ English review của từng business, không phải raw-language pool.')
''')
add('markdown','''
## 3. So sánh paired trước token adjustment

Mỗi business chạy cả hai phương pháp trên cùng population. Cột delta là rating_time − random; giá trị âm ở JSD là tốt hơn. Có businesses ≤30 nên hai phương pháp cùng giữ tất cả, không có chênh lệch. Các kết quả chỉ chứng minh thay đổi phân bố rating/time trong pilot, chưa chứng minh semantic coverage.
''')
add('code','''
metrics=['rating_jsd','temporal_jsd','rating_total_variation','date_span_coverage','selected_reviews','review_tokens_after_budget_adjustment']
display(initial.groupby('method')[metrics].agg(['mean','median','max']).round(5))
display(initial.groupby(['volume_band','method'])[metrics].mean().round(5))
paired=initial.pivot(index='business_id',columns='method',values=['rating_jsd','temporal_jsd','date_span_coverage'])
comparison=[]
for metric in ['rating_jsd','temporal_jsd']:
    delta=paired[(metric,'rating_time')]-paired[(metric,'random')]
    comparison.append({'metric':metric,'rating_time_better':int(delta.lt(-1e-12).sum()),'equal':int(delta.abs().le(1e-12).sum()),'rating_time_worse':int(delta.gt(1e-12).sum()),'mean_delta':delta.mean()})
display(pd.DataFrame(comparison))
worst=initial.loc[initial['method'].eq('rating_time')].sort_values(['temporal_jsd','rating_jsd'],ascending=False)
display(worst[['business_id','name','volume_band','total_reviews','selected_reviews','rating_jsd','temporal_jsd','date_span_coverage']].head(10))
''')
add('markdown','''
## 4. JSD theo số lượng review

Không đặt arbitrary pass/fail thresholds. Boxplots mô tả 12 diagnostic cases mỗi band; không phải confidence intervals.
''')
add('code','''
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric in zip(axes,['rating_jsd','temporal_jsd']):
    values=[]; positions=[]; colors=[]
    for i,band in enumerate(BANDS):
        for j,method in enumerate(METHODS):
            values.append(initial.loc[initial['volume_band'].eq(band)&initial['method'].eq(method),metric])
            positions.append(i+(-.18 if j==0 else .18)); colors.append(COLORS[method])
    boxes=ax.boxplot(values,positions=positions,widths=.28,patch_artist=True)
    for patch,color in zip(boxes['boxes'],colors): patch.set_facecolor(color)
    ax.set(xticks=range(4),xticklabels=BANDS,ylabel='JSD (base 2)',xlabel='English reviews/business',title=metric)
    ax.plot([],[],color=COLORS['random'],label='random'); ax.plot([],[],color=COLORS['rating_time'],label='rating/time'); ax.legend()
save(fig,'pilot_rating_temporal_jsd.png')
''')
add('markdown','''
## 5. Token lengths và phần còn TBD

Đếm exact draft review block có `[REV_i]`, stars/date, separators và BOS/EOS một lần. **Chưa phải full source có business metadata/captions.** Nếu configured exploratory scenarios rỗng, không có corpus token-removal run; chỉ có callable rule đã kiểm tra bằng edge-case tests. Semantic scores null không có nghĩa zero.
''')
add('code','''
display(initial.groupby('method')['review_tokens_after_budget_adjustment'].describe(percentiles=[.5,.9,.95,.99]).round(2))
print('Final review-token budget:',cfg['budget']['final_review_tokens'])
print('Exploratory scenarios actually run:',completion['exploratory_budget_scenarios'])
print('Semantic model:',cfg['semantic']['model_id'])
if completion['exploratory_budget_scenarios']:
    display(data.loc[~data['scenario'].eq('unadjusted')][['business_id','method','scenario','selected_reviews','review_tokens_before_budget_adjustment','review_tokens_after_budget_adjustment','rating_jsd','temporal_jsd','budget_feasible']])
else:
    print('Budget adjustment trên dữ liệu thật chưa chạy; câu hỏi ngân sách còn pending.')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for method in METHODS:
    rows=initial.loc[initial['method'].eq(method)]
    axes[0].scatter(rows['total_reviews'],rows['review_tokens_after_budget_adjustment'],label=method,color=COLORS[method],alpha=.7)
    axes[1].scatter(rows['total_reviews'],rows['date_span_coverage'],label=method,color=COLORS[method],alpha=.7)
axes[0].set(xscale='log',xlabel='Full English reviews/business (log)',ylabel='Exact draft review-block LED tokens',title='30-review cap chưa quyết định token length')
axes[1].set(xscale='log',xlabel='Full English reviews/business (log)',ylabel='Selected date span / full date span',title='Coverage theo ngày thực tế')
for ax in axes: ax.legend()
save(fig,'pilot_tokens_and_date_span.png')
''')
add('markdown','''
## 6. Xem rating và timeline của một business

Default chọn business có temporal JSD rating/time cao nhất để inspect hạn chế; không chọn ví dụ thuận lợi. Đổi BUSINESS_ID_TO_VIEW để xem business khác.
''')
add('code','''
with (OUT/'selection_runs.jsonl').open('rb') as stream: runs=[orjson.loads(line) for line in stream]
with (OUT/'pilot_population.jsonl').open('rb') as stream: pool_rows=[orjson.loads(line) for line in stream]
chosen=BUSINESS_ID_TO_VIEW or worst.iloc[0]['business_id']
assert chosen in set(sample['business_id'])
population=sorted([row['review'] for row in pool_rows if row['review']['business_id']==chosen],key=lambda r:(r['date'],r['review_id']))
chosen_runs=[r for r in runs if r['business_id']==chosen and r['scenario']=='unadjusted']
display(sample.loc[sample['business_id'].eq(chosen)].T)
display(pd.DataFrame(chosen_runs)[['method','total_reviews','selected_reviews','rating_jsd','temporal_jsd','date_span_coverage','review_tokens_after_budget_adjustment']])
fig,axes=plt.subplots(1,2,figsize=(12,4))
x=np.arange(5)
axes[0].bar(x-.25,[chosen_runs[0]['full_rating_distribution'][str(s)] for s in range(1,6)],width=.25,label='full',color='#94a3b8')
for k,run in enumerate(chosen_runs):
    axes[0].bar(x+k*.25,[run['selected_rating_distribution'][str(s)] for s in range(1,6)],width=.25,label=run['method'],color=COLORS[run['method']])
axes[0].set(xticks=x,xticklabels=range(1,6),xlabel='Stars',ylabel='Proportion',title='Full vs selected rating'); axes[0].legend()
axes[1].scatter(pd.to_datetime([r['date'] for r in population]),[0]*len(population),s=6,color='#94a3b8',label='full')
for k,run in enumerate(chosen_runs,1):
    ids=set(run['selected_review_ids']); values=[r for r in population if r['review_id'] in ids]
    axes[1].scatter(pd.to_datetime([r['date'] for r in values]),[k]*len(values),s=18,color=COLORS[run['method']],label=run['method'])
axes[1].set(yticks=[0,1,2],yticklabels=['full','random','rating/time'],xlabel='Review date',title='Actual review dates'); axes[1].tick_params(axis='x',rotation=25)
save(fig,'pilot_example_rating_timeline.png')
''')
add('markdown','''
## 7. Nguyên văn reviews và tám manual-inspection examples

Mở `manual_examples.html` để xem **tất cả** original English review texts của tám businesses (hai mỗi band). Màu xanh biểu thị selected rating/time trước budget. File là tài liệu để đọc, chưa phải manual audit đã hoàn thành. Bảng bên dưới chứa nguyên văn các review rating/time của business đang xem.
''')
add('code','''
example_ids=json.loads((OUT/'detail_example_business_ids.json').read_text(encoding='utf-8'))
display(sample.loc[sample['business_id'].isin(example_ids),['business_id','name','volume_band','split','english_reviews','pilot_reasons']])
display(FileLink(str(OUT/'manual_examples.html')))
selected_ids=set(next(r for r in chosen_runs if r['method']=='rating_time')['selected_review_ids'])
parts=[]
for r in population:
    if r['review_id'] in selected_ids:
        parts.append('<details><summary>'+html.escape(f"{r['stars']} stars | {r['date']} | {r['review_id']}")+'</summary><pre style="white-space:pre-wrap">'+html.escape(r['text'])+'</pre></details>')
display(HTML(''.join(parts)))
print('Manual aspect/opinion coverage audit: chưa thực hiện; cần đọc examples trước khi chốt.')
''')
add('markdown','''
## 8. Acceptance và bước tiếp theo

Implementation/integrity PASS không có nghĩa toàn bộ review đã được đại diện hoàn hảo. Chưa chọn trên toàn cohort; chưa freeze final review manifest. Cần review diagnostics, thử ngân sách/model khi chốt được và manual-inspect examples, rồi mới quyết định methodology cuối.
''')
add('code','''
acceptance=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{k:acceptance[k] for k in ['validation_passed','businesses','population_reviews_verified','selection_runs_verified','budget_removal_steps_verified','test_businesses','semantic_status','methodology_locked']}]))
assert acceptance['validation_passed'] and acceptance['test_businesses']==0
print('Spec:',PROJECT_ROOT/'specs/003b_review_selection_pilot.md')
print('Outputs:',OUT)
print('Overall Stage C complete:',completion['overall_stage_c_complete'])
''')

notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.7','file_extension':'.py','mimetype':'text/x-python','pygments_lexer':'ipython3'}},'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks/04_review_selection_pilot.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(f'Created {path} ({len(cells)} cells)')
