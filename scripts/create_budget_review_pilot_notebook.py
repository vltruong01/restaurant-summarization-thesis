"""Create notebook 05 for versioned budget-aware pilot; resets saved outputs."""
from pathlib import Path
import json
import textwrap

ROOT=Path(__file__).resolve().parents[1];cells=[]


def add(kind,source):
    source=textwrap.dedent(source).strip()+'\n'
    cell={'cell_type':kind,'id':f'budget-pilot-{len(cells):03d}','metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':
        compile(source,f'budget-pilot-cell-{len(cells)}','exec');cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown','''
# C2 — Complete-review selection dưới budget (pilot v3)

56 businesses **train/validation**: 48 diagnostic cases cũ + 8 low-volume long-review stress cases. Population là tất cả original English reviews/business. Các tỷ lệ trong notebook chỉ mô tả stress sample, **không ước lượng toàn cohort 2.300 businesses**.

Review-block scenarios **3.500 / 4.000 / 4.500 tokens**; target total source vẫn ~5.000, allocations cuối chưa khóa. Block gồm markers/stars/date/separators và BOS/EOS một lần; chưa chứa business metadata/captions. No internal truncation, no minimum-ten input, không loại business dưới 10 review.

Bộ chọn mới xem k=1..30 và full-pool replacement/addition/deletion; so sánh với labeled random/rating-time delete-only baselines. Search có giới hạn, không claim global optimum. Draft rating/time weights 0,5/0,5; default comparable JSD tolerance 0,001 mỗi metric, sensitivity 0 / 0,001 / 0,005. Không phải quality PASS thresholds. Semantic model và manual aspect audit còn TBD.
''')
add('markdown','''
## 1. Artifacts và identity

Notebook đọc pilot đã hoàn tất/kiểm chứng; không tự chạy lại heavy search. BUSINESS_ID_TO_VIEW chỉ dùng xem một ví dụ. Nếu thay method/config, version output directory và chạy lại pipeline/validator trước.
''')
add('code','''
from pathlib import Path
import json,html,hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display,HTML,FileLink
import yaml,orjson
BUSINESS_ID_TO_VIEW=None
ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'configs/review_selection_budget_pilot.yaml').is_file())
cfg=yaml.safe_load((ROOT/'configs/review_selection_budget_pilot.yaml').read_text(encoding='utf-8'))
OUT=ROOT/cfg['output_directory'];FIGURES=OUT/'figures';FIGURES.mkdir(exist_ok=True)
completion=json.loads((OUT/'completion_report.json').read_text())
acceptance=json.loads((OUT/'validation_report.json').read_text())
assert acceptance['validation_passed'] and acceptance['test_businesses']==0
for name,digest in completion['artifact_sha256'].items():
    with (OUT/name).open('rb') as stream:
        h=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==digest,name
data=pd.read_csv(OUT/'selection_diagnostics.csv')
summary=pd.read_csv(OUT/'budget_method_summary.csv')
sample=pd.read_csv(OUT/'pilot_businesses.csv',keep_default_na=False)
sensitivity=pd.read_json(OUT/'tolerance_sensitivity.json')
stress=pd.DataFrame([orjson.loads(line) for line in (OUT/'long_review_stress.jsonl').read_bytes().splitlines()])
methods=['random_delete_only','rating_time_delete_only','budget_aware'];budgets=[3500,4000,4500]
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
pd.set_option('display.max_columns',25)
display(pd.DataFrame([{k:completion[k] for k in ['businesses','population_reviews','selection_runs','test_businesses','final_review_budget','semantic_status','methodology_locked']}]))

def save(fig,name):
    fig.tight_layout();fig.savefig(FIGURES/name,dpi=150,bbox_inches='tight');plt.show();plt.close(fig)
''')
add('markdown','''
## 2. Retained review counts và flags <10 / <5

Flag là trường hợp cần review, không phải exclusion. Histogram gồm mọi business; zero phải có budget_infeasible status, không export valid empty input. Kết quả nhóm 48 cũ và 8 stress additions được hiển thị riêng để tránh hiểu nhầm tỷ lệ của toàn cohort.
''')
add('code','''
display(pd.crosstab(sample['sample_group'],sample['split']))
display(summary.sort_values(['review_budget','method']))
new=data.loc[data['method'].eq('budget_aware')].copy()
grouped=new.groupby(['sample_group','review_budget']).agg(businesses=('business_id','size'),below_10=('below_10_retained_flag','sum'),below_5=('below_5_retained_flag','sum'),retained_mean=('final_retained_review_count','mean'),retained_min=('final_retained_review_count','min'))
display(grouped)
display(new.loc[new['below_10_retained_flag'],['business_id','name','sample_group','review_budget','initial_eligible_review_count','final_retained_review_count','tokens_after_budget_enforcement','rating_jsd_after','temporal_jsd_after','selection_status']])
bands=['25-30','20-24','15-19','10-14','5-9','<5']
colors=['#1d4ed8','#60a5fa','#93c5fd','#a7f3d0','#fbbf24','#ef4444']
fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,method in zip(axes,methods):
    table=summary.loc[summary['method'].eq(method)].set_index('review_budget').reindex(budgets)
    bottom=np.zeros(3)
    for band,color in zip(bands,colors):
        values=table['band_'+band].to_numpy()
        ax.bar([str(b) for b in budgets],values,bottom=bottom,color=color,label=band);bottom+=values
    ax.set(title=method,xlabel='Exploratory review budget',ylabel='Businesses')
axes[-1].legend(title='Complete reviews retained',bbox_to_anchor=(1.02,1),loc='upper left')
save(fig,'budget_retained_counts.png')
''')
add('markdown','''
## 3. Rating/time representation trước và sau

Các weighted-quality/tolerance rules có thể đổi count và các JSD theo hướng khác nhau; không claim từng business đều tốt hơn ở cả hai metric. Cùng population/reference được dùng cho paired comparisons. Semantic similarity chưa tính.
''')
add('code','''
display(data.groupby(['method','review_budget'])[['rating_jsd_before','rating_jsd_after','temporal_jsd_before','temporal_jsd_after','final_retained_review_count']].mean().round(6))
paired=data.pivot(index=['business_id','review_budget'],columns='method',values=['final_retained_review_count','rating_jsd_after','temporal_jsd_after'])
for metric in ['final_retained_review_count','rating_jsd_after','temporal_jsd_after']:
    delta=paired[(metric,'budget_aware')]-paired[(metric,'rating_time_delete_only')]
    print(metric,': budget-aware − rating/time deletion; mean=',round(delta.mean(),6),'min=',round(delta.min(),6),'max=',round(delta.max(),6))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric in zip(axes,['rating_jsd_after','temporal_jsd_after']):
    for method,color in zip(methods,['#64748b','#f59e0b','#2563eb']):
        means=data.loc[data['method'].eq(method)].groupby('review_budget')[metric].mean().reindex(budgets)
        ax.plot(budgets,means,marker='o',label=method,color=color)
    ax.set(xlabel='Exploratory review budget',ylabel='Mean JSD (base 2)',title=metric);ax.set_xticks(budgets);ax.legend(fontsize=8)
save(fig,'budget_rating_time_jsd.png')
''')
add('markdown','''
## 4. Sensitivity của comparable-quality tolerance

Cùng searched frontier, thay cách xếp hạng với tolerance 0 / 0,001 / 0,005 mỗi metric. Đây là ranking sensitivity, không phải thử mọi search parameter. “Có thêm review” chưa chứng minh giữ nhiều nội dung hơn; length-aware lựa chọn có thể ưu tiên text ngắn trong cùng rating/time nhóm.
''')
add('code','''
display(sensitivity.groupby(['review_budget','tolerance_per_metric'])[['count','rating_jsd','temporal_jsd','tokens']].agg(['mean','min','max']).round(6))
fig,ax=plt.subplots(figsize=(8,4))
for tolerance,color in zip([0,.001,.005],['#64748b','#2563eb','#f59e0b']):
    rows=sensitivity.loc[sensitivity['tolerance_per_metric'].sub(tolerance).abs().lt(1e-10)]
    means=rows.groupby('review_budget')['count'].mean().reindex(budgets)
    ax.plot(budgets,means,marker='o',label=f'tolerance={tolerance}',color=color)
ax.set(xlabel='Exploratory review budget',ylabel='Mean complete reviews retained',title='Ranking sensitivity on the same searched frontier');ax.set_xticks(budgets);ax.legend()
save(fig,'budget_tolerance_review_counts.png')
''')
add('markdown','''
## 5. Trường hợp chỉ 7–8 review dài đã vượt target tổng

Các stress sets lấy 7/8 review dài nhất của mỗi business chỉ để kiểm tra edge case, không phải input được chọn và không thay population. Bảng cho biết exact draft block lengths; tổng source ~5.000 không bị đổi thành review allocation.
''')
add('code','''
stress['above_5000_total_target']=stress['exact_tokens'].gt(5000)
display(stress.groupby('count').agg(businesses=('business_id','size'),above_5000=('above_5000_total_target','sum'),max_tokens=('exact_tokens','max')))
display(stress.sort_values('exact_tokens',ascending=False).head(12)[['business_id','count','exact_tokens','above_5000_total_target']])
display(new['tokens_after_budget_enforcement'].describe(percentiles=[.5,.9,.95,.99]))
assert new['tokens_after_budget_enforcement'].le(new['review_budget']).all()
print('Final review budget:',cfg['budget']['final_review_tokens'],'; planned TOTAL target:',cfg['budget']['planned_total_source_target_approximate'])
''')
add('markdown','''
## 6. Một business: original reference và budget-aware retained reviews

Default ưu tiên business giữ ít review nhất ở budget 3.500, rồi reference dài nhất. Các trace chỉ ghi actual search moves; reference-to-final differences được tách riêng và không giả vờ là một chuỗi replacements đã thực hiện.
''')
add('code','''
with (OUT/'selection_runs.jsonl').open('rb') as stream:runs=[orjson.loads(line) for line in stream]
with (OUT/'pilot_population.jsonl').open('rb') as stream:population_rows=[orjson.loads(line) for line in stream]
ranking=new.loc[new['review_budget'].eq(3500)].sort_values(['final_retained_review_count','tokens_before_budget_enforcement','business_id'],ascending=[True,False,True])
chosen=BUSINESS_ID_TO_VIEW or ranking.iloc[0]['business_id']
display(sample.loc[sample['business_id'].eq(chosen),['business_id','name','split','sample_group','english_reviews','top8_individual_base_tokens']])
selected=next(r for r in runs if r['business_id']==chosen and r['method']=='budget_aware' and r['review_budget']==3500)
display(pd.DataFrame([{k:selected[k] for k in ['initial_eligible_review_count','initial_candidate_count','final_retained_review_count','tokens_before_budget_enforcement','tokens_after_budget_enforcement','rating_jsd_before','rating_jsd_after','temporal_jsd_before','temporal_jsd_after','below_10_retained_flag']}]))
print('Removed vs reference:',selected['removed_vs_reference_ids'])
print('Added vs reference:',selected['added_vs_reference_ids'])
display(HTML('<details><summary>Accepted local search provenance</summary><pre>'+html.escape(json.dumps(selected['search_provenance'],ensure_ascii=False,indent=2))+'</pre></details>'))
keep=set(selected['selected_review_ids'])
values=sorted([row['review'] for row in population_rows if row['review']['business_id']==chosen],key=lambda r:(r['date'],r['review_id']))
parts=[]
for r in values:
    if r['review_id'] in keep:
        parts.append('<details><summary>'+html.escape(f"{r['stars']} stars | {r['date']} | {r['review_id']}")+'</summary><pre style="white-space:pre-wrap">'+html.escape(r['text'])+'</pre></details>')
display(HTML(''.join(parts)))
''')
add('markdown','''
## 7. Ten full-text inspection examples

File HTML chứa toàn bộ original English text của 10 businesses, ưu tiên các trường hợp count thấp và stress dài. Tạo file không thay cho researcher manual quality audit. Cases <10 không tự bị loại.
''')
add('code','''
examples=json.loads((OUT/'detail_example_business_ids.json').read_text())
display(sample.loc[sample['business_id'].isin(examples),['business_id','name','split','sample_group','english_reviews','pilot_reasons']])
display(FileLink(str(OUT/'manual_examples.html')))
print('Semantic centroid similarity:',None,'(embedding model TBD)')
print('Manual aspect/opinion audit: pending researcher review')
''')
add('markdown','''
## 8. Acceptance và giới hạn

Independent checks + fresh deterministic re-run xác minh implementation, không khẳng định subset đại diện hoàn hảo. Stop sau pilot, chưa freeze 2.300-business manifest, chưa đổi source length/allocation, chưa train/rent GPU. Review results và nội dung trước final methodology.
''')
add('code','''
display(pd.DataFrame([{k:acceptance[k] for k in ['validation_passed','businesses','source_reviews_verified','selection_runs_verified','fresh_deterministic_runs_verified','accepted_local_moves_verified','exact_tokens_jsd_flags_histograms_verified','test_businesses','methodology_locked']}]))
print('Spec:',ROOT/'specs/003c_budget_constrained_pilot.md')
print('Outputs:',OUT)
print('Final methodology locked:',completion['methodology_locked'])
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3 (ipykernel)','name':'python3','language':'python'},'language_info':{'name':'python','version':'3.11.7','file_extension':'.py','mimetype':'text/x-python','pygments_lexer':'ipython3'}},'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks/05_budget_constrained_review_selection_pilot.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(f'Created {path} ({len(cells)} cells)')
