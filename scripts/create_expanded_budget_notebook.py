"""Create read-only inspection notebook for all train/validation diagnostics."""
from pathlib import Path
import json,textwrap
ROOT=Path(__file__).resolve().parents[1];cells=[]


def add(kind,source):
    source=textwrap.dedent(source).strip()+'\n'
    cell={'cell_type':kind,'id':f'expanded-budget-{len(cells):03d}','metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':
        compile(source,f'cell-{len(cells)}','exec');cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown','''
# C2 — Budget diagnostics trên toàn bộ 2.200 train/validation businesses

2.000 train + 200 validation; **575.749 original English reviews**. Không dùng 100 test businesses để tune. Algorithm `complete_review_search_v3`, cùng scientific settings của pilot đã kiểm chứng; ba review-block scenarios 3.500 / 4.000 / 4.500, tối đa 30 complete reviews, không cắt text và không minimum-ten input.

Total source target vẫn ~5.000; final review/metadata/caption allocation và manifest chưa khóa. Notebook chỉ đọc outputs đã chạy/kiểm chứng; không tự chạy lại search. Tokens after = exact marked review block + BOS/EOS, chưa có business metadata/captions. Body-token lengths của review gốc là metric khác.
''')
add('code','''
from pathlib import Path
import json,hashlib,html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,HTML
import yaml,orjson
BUSINESS_ID_TO_VIEW=None
ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'configs/review_selection_expanded.yaml').is_file())
cfg=yaml.safe_load((ROOT/'configs/review_selection_expanded.yaml').read_text(encoding='utf-8'))
OUT=ROOT/cfg['output_directory'];FIGURES=OUT/'figures';FIGURES.mkdir(exist_ok=True)
completion=json.loads((OUT/'completion_report.json').read_text())
validation=json.loads((OUT/'validation_report.json').read_text())
assert validation['validation_passed'] and validation['test_businesses']==0 and not completion['final_manifest']
for name,digest in completion['artifact_sha256'].items():
    with (OUT/name).open('rb') as stream:
        h=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==digest,name
data=pd.read_csv(OUT/'selection_diagnostics.csv')
summary=pd.read_csv(OUT/'budget_method_summary.csv')
profiles=pd.read_csv(OUT/'business_length_profiles.csv')
sample=pd.read_csv(OUT/'businesses.csv',keep_default_na=False)
stress=pd.DataFrame(json.loads((OUT/'long_review_stress.json').read_text()))
new=data.loc[data['method'].eq('budget_aware')].copy()
budgets=[3500,4000,4500]
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
pd.set_option('display.max_columns',25)
def save(fig,name):
    fig.tight_layout();fig.savefig(FIGURES/name,dpi=150,bbox_inches='tight');plt.show();plt.close(fig)
display(pd.DataFrame([{k:completion[k] for k in ['businesses','splits','source_reviews','selection_runs','origins','test_businesses','final_review_budget','final_manifest']}]))
''')
add('markdown','''
## 1. Số review giữ lại, <10/<5 và min/max tokens

Mỗi budget có đủ 2.200 businesses. Infeasible được đếm riêng; không export empty valid input, không loại business. Min/max tokens chỉ xét feasible selected blocks. Review count zero nếu infeasible vẫn nằm trong retained distribution.
''')
add('code','''
columns=['review_budget','businesses','retained_min','retained_max','retained_mean','below_10','below_10_percent','below_5','infeasible','tokens_min','tokens_max','tokens_mean','tokens_p50','tokens_p95','reference_tokens_min','reference_tokens_max','below_ten_probe_fits','below_ten_probe_exceeds']
display(summary.loc[summary['method'].eq('budget_aware'),columns].sort_values('review_budget').round(3))
display(new.groupby(['split','review_budget']).agg(businesses=('business_id','size'),below_10=('below_10_retained_flag','sum'),below_5=('below_5_retained_flag','sum'),retained_min=('final_retained_review_count','min'),retained_mean=('final_retained_review_count','mean'),tokens_min=('tokens_after_budget_enforcement','min'),tokens_max=('tokens_after_budget_enforcement','max')).round(3))
valid=new.loc[new['selection_status'].eq('feasible')]
assert valid['tokens_after_budget_enforcement'].le(valid['review_budget']).all()
''')
add('markdown','''
## 2. Retained-count distribution
''')
add('code','''
bands=['25-30','20-24','15-19','10-14','5-9','<5']
tab=summary.loc[summary['method'].eq('budget_aware')].set_index('review_budget').reindex(budgets)
display(tab[['band_'+b for b in bands]])
fig,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(3)
for band,color in zip(bands,['#1d4ed8','#60a5fa','#93c5fd','#a7f3d0','#fbbf24','#ef4444']):
    vals=tab['band_'+band].to_numpy()
    ax.bar([str(b) for b in budgets],vals,bottom=bottom,label=band,color=color);bottom+=vals
ax.set(xlabel='Exploratory review budget',ylabel='Businesses',title='Complete reviews retained across all train/validation businesses')
ax.legend(title='Retained reviews',bbox_to_anchor=(1.02,1),loc='upper left')
save(fig,'expanded_retained_counts.png')
''')
add('markdown','''
## 3. Token lengths: selected block và individual original review

Selected/reference block có markers/stars/date/separators + special tokens một lần. Individual BODY lengths không gồm header hoặc BOS/EOS. Tổng individual body lengths/business là diagnostic sum, không phải exact assembled model input.
''')
add('code','''
display(valid.groupby('review_budget')['tokens_after_budget_enforcement'].describe(percentiles=[.5,.9,.95,.99]).round(2))
display(pd.DataFrame([{'original_English_reviews':completion['source_reviews'],'individual_BODY_tokens_min':profiles['individual_body_tokens_min'].min(),'individual_BODY_tokens_max':profiles['individual_body_tokens_max'].max(),'individual_BODY_tokens_mean':profiles['individual_body_tokens_sum'].sum()/completion['source_reviews'],'business_BODY_token_sum_min':profiles['individual_body_tokens_sum'].min(),'business_BODY_token_sum_max':profiles['individual_body_tokens_sum'].max(),'complete_singleton_tokens_min':profiles['complete_singleton_tokens_min'].min(),'complete_singleton_tokens_max':profiles['complete_singleton_tokens_max'].max()}]).round(2))
display(profiles.sort_values('individual_body_tokens_max',ascending=False).head(10))
fig,ax=plt.subplots(figsize=(9,4))
ax.boxplot([valid.loc[valid['review_budget'].eq(b),'tokens_after_budget_enforcement'] for b in budgets],labels=[str(b) for b in budgets],showfliers=True)
ax.set(xlabel='Exploratory review budget',ylabel='Exact selected review-block tokens',title='Token distribution, including complete review markers and BOS/EOS')
save(fig,'expanded_selected_tokens.png')
''')
add('markdown','''
## 4. Tất cả cases <10 và kiểm tra áp lực độ dài

Probe gồm mười originals có singleton block cost thấp nhất, sắp xếp chronological rồi encode nguyên block. Nếu probe vừa budget, chắc chắn có input mười complete reviews vừa; count thấp không bị độ dài bắt buộc. Nếu probe không vừa, chỉ ghi observed length pressure: chưa chứng minh mọi subset mười reviews đều không vừa, vì marker-position/boundary effects và bounded search. Không tự đổi phương pháp hoặc loại business.
''')
add('code','''
flags=new.loc[new['below_10_retained_flag']].sort_values(['review_budget','final_retained_review_count','business_id'])
flag_cols=['business_id','name','split','review_budget','initial_eligible_review_count','final_retained_review_count','tokens_before_budget_enforcement','tokens_after_budget_enforcement','individual_body_tokens_min','individual_body_tokens_max','ten_cheapest_singleton_probe_tokens','ten_complete_reviews_demonstrably_fit','below_ten_length_diagnostic','rating_jsd_after','temporal_jsd_after']
display(flags[flag_cols])
display(flags.groupby(['review_budget','below_ten_length_diagnostic']).size().rename('businesses'))
print('Unique budget-aware businesses flagged:',flags['business_id'].nunique())
print('Case file:',OUT/'below_ten_cases_budget_aware.csv')
stress['above_5000']=stress['exact_tokens'].gt(5000)
display(stress.groupby('count').agg(businesses=('business_id','size'),above_5000=('above_5000','sum'),token_min=('exact_tokens','min'),token_max=('exact_tokens','max')))
''')
add('markdown','''
## 5. Rating/time before/after và baselines

JSD log base 2, năm full-population rank bins. Scores phân bố không chứng minh aspect/opinion coverage. Mean improvements có thể đi cùng per-business regressions. Semantic diagnostic/model chưa tính/TBD.
''')
add('code','''
display(summary[['method','review_budget','retained_mean','rating_jsd_before_mean','rating_jsd_after_mean','temporal_jsd_before_mean','temporal_jsd_after_mean']].round(6))
paired=data.pivot(index=['business_id','review_budget'],columns='method',values=['final_retained_review_count','rating_jsd_after','temporal_jsd_after'])
rows=[]
for budget in budgets:
    p=paired.xs(budget,level='review_budget')
    rows.append({'budget':budget,'mean_count_gain_vs_rating_time_delete':(p[('final_retained_review_count','budget_aware')]-p[('final_retained_review_count','rating_time_delete_only')]).mean(),'rating_regressions_vs_baseline':(p[('rating_jsd_after','budget_aware')]>p[('rating_jsd_after','rating_time_delete_only')]+1e-12).sum(),'temporal_regressions_vs_baseline':(p[('temporal_jsd_after','budget_aware')]>p[('temporal_jsd_after','rating_time_delete_only')]+1e-12).sum()})
display(pd.DataFrame(rows).round(4))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric in zip(axes,['rating_jsd_after_mean','temporal_jsd_after_mean']):
    for method,color in zip(['random_delete_only','rating_time_delete_only','budget_aware'],['#64748b','#f59e0b','#2563eb']):
        values=summary.loc[summary['method'].eq(method)].set_index('review_budget').reindex(budgets)[metric]
        ax.plot(budgets,values,marker='o',color=color,label=method)
    ax.set(title=metric,xlabel='Exploratory review budget',ylabel='Mean JSD (base 2)');ax.set_xticks(budgets);ax.legend(fontsize=8)
save(fig,'expanded_rating_time_jsd.png')
''')
add('markdown','''
## 6. Xem business nguyên văn

Đặt BUSINESS_ID_TO_VIEW ở cell đầu nếu cần. Default chọn count thấp nhất tại 3.500 tokens; xem selected texts, IDs/provenance và ten-review diagnostic probe. Đây là inspection, chưa có human semantic/aspect audit theo protocol.
''')
add('code','''
ranking=new.loc[new['review_budget'].eq(3500)].sort_values(['final_retained_review_count','tokens_before_budget_enforcement','business_id'],ascending=[True,False,True])
bid=BUSINESS_ID_TO_VIEW or ranking.iloc[0]['business_id']
payload=json.loads((OUT/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json').read_text(encoding='utf-8'))
display(sample.loc[sample['business_id'].eq(bid),['business_id','name','split','english_reviews','english_review_token_mean','english_review_token_max']])
display(new.loc[new['business_id'].eq(bid),flag_cols])
source_split=sample.loc[sample['business_id'].eq(bid),'split'].iloc[0]
originals={}
with (ROOT/cfg['english_pool_directory']/'reviews'/f'{source_split}.jsonl').open('rb') as stream:
    for line in stream:
        r=orjson.loads(line)
        if r['business_id']==bid:originals[r['review_id']]=r
row=next(r for r in payload['runs'] if r['method']=='budget_aware' and r['review_budget']==3500)
print('Selected IDs:',row['selected_review_ids'])
print('Ten-review diagnostic IDs:',payload['length_profile']['ten_cheapest_singleton_probe_ids'])
parts=[]
for rid in row['selected_review_ids']:
    r=originals[rid]
    parts.append('<details><summary>'+html.escape(f"{rid} | {r['stars']} stars | {r['date']}")+'</summary><pre style="white-space:pre-wrap">'+html.escape(r['text'])+'</pre></details>')
display(HTML(''.join(parts)))
display(HTML('<details><summary>Search provenance</summary><pre>'+html.escape(json.dumps(row['search_provenance'],ensure_ascii=False,indent=2))+'</pre></details>'))
''')
add('markdown','''
## 7. Acceptance và điểm dừng

Expanded diagnostics không phải frozen manifest. Không tune test, semantic diversity hoặc đổi source length/allocation. Review flagged cases và nội dung trước khi chốt final methodology.
''')
add('code','''
display(pd.DataFrame([validation]))
print('Report:',OUT/'expanded_diagnostics_report.md')
print('Final review budget:',completion['final_review_budget'],'; final manifest:',completion['final_manifest'])
print('GPU inventory: local RTX 3050 6 GB; this tokenizer/search remains CPU code.')
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3 (ipykernel)','name':'python3','language':'python'},'language_info':{'name':'python','version':'3.11.7','file_extension':'.py','mimetype':'text/x-python','pygments_lexer':'ipython3'}},'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks/06_train_validation_budget_diagnostics.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(f'Created {path}: {len(cells)} cells')
