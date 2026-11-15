"""Generate read-only notebook 07 for accepted length/content audit outputs."""
from pathlib import Path
import json, textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, text):
    source = textwrap.dedent(text).strip()+'\n'
    cell = {'cell_type': kind, 'id': f'length-audit-{len(cells):03d}', 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        compile(source, f'cell-{len(cells)}', 'exec'); cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''
# C2 — Review length and content audit / Kiểm tra review quá ngắn

All **2,000 train + 200 validation**, no test tuning. Compare body tokens of unchanged originals within each business. This notebook reads diagnostics, does not select reviews or change budgets. Thresholds <20/30/50 body tokens, mean ratios <0.5/0.75 and block tokens <1500/2000 are descriptive only, not quality or exclusion rules. Total ~5000 source target and final review/metadata/caption allocation remain pending.
''')
add('code', '''
from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
ROOT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p/'specs/003e_selected_review_length_audit.md').is_file())
OUT = ROOT/'outputs/stage_c/review_selection_length_audit_v1'
FIG = OUT/'figures'; FIG.mkdir(exist_ok=True)
completion = json.loads((OUT/'completion_report.json').read_text())
validation = json.loads((OUT/'validation_report.json').read_text())
assert validation['validation_passed'] and completion['test_businesses']==0 and not completion['final_manifest']
for name, expected in completion['artifact_sha256'].items():
    with (OUT/name).open('rb') as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    assert h.hexdigest()==expected, name
data = pd.read_csv(OUT/'business_length_comparisons.csv')
summary = pd.read_csv(OUT/'length_summary.csv')
new = data.loc[data.method.eq('budget_aware')].copy()
examples = json.loads((OUT/'inspection_examples.json').read_text(encoding='utf-8'))
annotations = json.loads((OUT/'qualitative_inspection.json').read_text(encoding='utf-8'))
assert hashlib.sha256((OUT/'qualitative_inspection.json').read_bytes()).hexdigest()==validation['qualitative_annotation_sha256']
pd.set_option('display.max_columns', 22)
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.tight_layout(); fig.savefig(FIG/name,dpi=150,bbox_inches='tight'); plt.show(); plt.close(fig)
display(pd.DataFrame([{k:completion[k] for k in ['businesses','source_reviews_verified','selection_runs','unique_selected_originals_fresh_encoded','test_businesses','final_review_budget','final_manifest']}]))
''')
add('markdown', '''
## 1. Selected/source length ratio and within-business percentile

Each business receives equal weight in macro summaries. A ratio 0.4 means selected reviews average 40% of that business's original review length. A midrank percentile near 0.5 is central in that business's length distribution; lower values indicate shorter originals. Ties use their midpoint. Body tokens exclude all review headers/special tokens. The pooled mean counts review occurrences within one method/budget; budgets overlap and are not independent samples.
''')
add('code', '''
cols=['method','review_budget','businesses','source_mean_macro','selected_mean_macro','selected_source_mean_ratio_median','mean_ratio_below_0_5','mean_ratio_below_0_75','selected_midrank_percentile_macro','source_short20_macro','selected_short20_macro','length_jsd_macro']
display(summary[cols].round(4))
fig,axes=plt.subplots(1,2,figsize=(14,4.5))
methods=['budget_aware','rating_time_delete_only','random_delete_only']; colors=['#2563eb','#f59e0b','#64748b']
positions=[];vals=[];labels=[];palette=[]
for i,budget in enumerate([3500,4000,4500]):
    for j,method in enumerate(methods):
        positions.append(i*4+j); vals.append(data.loc[data.review_budget.eq(budget)&data.method.eq(method),'mean_ratio'])
        labels.append(f'{budget}\\n{["new","rating/time","random"][j]}');palette.append(colors[j])
boxes=axes[0].boxplot(vals,positions=positions,showfliers=False,patch_artist=True)
for box,color in zip(boxes['boxes'],palette): box.set_facecolor(color);box.set_alpha(.7)
axes[0].set_xticks(positions,labels,fontsize=8,rotation=25,ha='right');axes[0].axhline(1,color='black',linestyle='--',lw=1)
axes[0].set(ylabel='Selected / source mean BODY tokens',title='Within-business length ratio (2,200 per scenario)')
for method,color in zip(methods,colors):
    s=summary.loc[summary.method.eq(method)].sort_values('review_budget')
    axes[1].plot(s.review_budget,s.selected_midrank_percentile_macro,marker='o',label=method,color=color)
axes[1].axhline(.5,color='black',linestyle='--',lw=1);axes[1].set_xticks([3500,4000,4500])
axes[1].set(xlabel='Exploratory review budget',ylabel='Mean within-business length percentile',title='Position in original length distribution')
axes[1].legend(fontsize=8,loc='center right');save(fig,'selected_source_length_shift.png')
''')
add('markdown', '''
## 2. Short reviews and source-volume/split breakdowns

Short does not automatically mean uninformative. Ratios/thresholds describe length only. Low-volume businesses retaining all originals naturally have ratio 1; a large source pool provides many more short alternatives. These are observational diagnostics, not randomized causal estimates.
''')
add('code', '''
display(pd.read_csv(OUT/'length_summary_by_split.csv').query("method=='budget_aware'").round(4))
bands=pd.read_csv(OUT/'length_summary_by_volume.csv').query("method=='budget_aware'")
display(bands[['review_budget','volume_band','businesses','selected_source_mean_ratio_median','selected_short20_macro','source_short20_macro']].round(4))
fig,axes=plt.subplots(1,2,figsize=(11,4));budgets=[3500,4000,4500]
s=summary.loc[summary.method.eq('budget_aware')].sort_values('review_budget')
x=np.arange(3)
axes[0].bar(x-.18,s.source_short20_macro*100,width=.36,label='Full source',color='#64748b')
axes[0].bar(x+.18,s.selected_short20_macro*100,width=.36,label='Selected',color='#2563eb')
axes[0].set_xticks(x,[str(b) for b in budgets]);axes[0].set(ylabel='Mean fraction of reviews (%)',xlabel='Review budget')
axes[0].set_title('Reviews with <20 BODY tokens',pad=35);axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=9)
for band in ['10-30','31-100','101-300','301+']:
    s= bands.loc[bands.volume_band.eq(band)].sort_values('review_budget')
    axes[1].plot(s.review_budget,s.selected_source_mean_ratio_median,marker='o',label=band)
axes[1].set_xticks(budgets);axes[1].set(ylabel='Median selected/source mean ratio',xlabel='Review budget',title='Source English review count bands');axes[1].legend(title='Source count')
save(fig,'short_reviews_and_source_volume.png')
''')
add('markdown', '''
## 3. Exactly 30 inputs and same-count baseline comparisons

Separate exactly-30 short blocks from businesses with fewer source reviews. The 842-token overall minimum retained 11 originals; it was not a 30-review input. Baseline subsets can differ in count; same-count rows control this arithmetic difference, but are still observational and do not establish equivalent content or rating/time quality.
''')
add('code', '''
display(summary.loc[summary.method.eq('budget_aware'),['review_budget','exactly30','exactly30_under1500','exactly30_under2000']])
low=pd.read_csv(OUT/'exactly30_ranked_by_tokens.csv')
display(low.loc[low.review_budget.eq(3500),['business_id','name','source_count','selected_count','selected_block_tokens','source_mean','selected_mean','mean_ratio','rating_jsd','temporal_jsd']].head(15).round(5))
display(pd.read_csv(OUT/'paired_baseline_comparisons.csv').round(4))
''')
add('markdown', '''
## 4. Counterfactual same-star/time-bin longer originals

Ten deterministic examples only. Up to five replacements, count unchanged; exact same joint star/time-bin histogram means the reported rating/time JSD is identical. Full block tokens are freshly checked. This demonstrates feasibility of some longer alternatives, not their global semantic superiority. They are not new selected evidence or an approved selection method.
''')
add('code', '''
cf=pd.read_csv(OUT/'counterfactual_summary.csv')
display(cf[['business_id','review_budget','retained_count','replacements','original_block_tokens','counterfactual_block_tokens','original_body_sum','counterfactual_body_sum','same_star_time_joint_histogram']])
first=cf.loc[cf.review_budget.eq(3500)].set_index('business_id')
five=[case['business_id'] for case in examples[:5]]
labels=[case['name'] for case in examples[:5]]
fig,ax=plt.subplots(figsize=(10,4));x=np.arange(5);g=first.loc[five]
ax.bar(x-.18,g.original_block_tokens,width=.36,label='Accepted selection',color='#2563eb')
ax.bar(x+.18,g.counterfactual_block_tokens,width=.36,label='5 same-cell replacements',color='#16a34a')
ax.axhline(3500,color='black',linestyle='--',label='3,500 review budget');ax.set_xticks(x,labels,rotation=15,ha='right')
ax.set(ylabel='Exact review-block tokens',title='Low-token 30-review cases: longer originals still fit');ax.legend(fontsize=8)
save(fig,'counterfactual_longer_reviews.png')
''')
add('markdown', '''
## 5. Full-text inspection and qualitative limitations

Assistant read 150 selected originals in the five low-token cases at budget 3500 and ten matched longer originals. This purposive inspection is separate from formal human evaluation or an exhaustive source-aspect audit. The JSON/HTML provide unchanged originals and provenance for all ten examples and three budgets. Review length does not measure semantic informativeness.
''')
add('code', '''
display(pd.DataFrame(annotations['cases'])[['name','observations']])
BUSINESS_ID_TO_VIEW='9PZxjhTIU7OgPIzuGi89Ew'
case=next(x for x in examples if x['business_id']==BUSINESS_ID_TO_VIEW)
run=case['runs'][0]
display(pd.DataFrame([{'id':r['review_id'],'stars':r['stars'],'body_tokens':r['body_tokens'],'text':r['text']} for r in run['selected']]))
display(pd.DataFrame(run['alternative_moves'])[['removed_id','added_id','removed_body_tokens','added_body_tokens','removed_text','added_text']])
display(HTML('<a href="../outputs/stage_c/review_selection_length_audit_v1/inspection_examples.html">All ten full-text examples</a>'))
print(annotations['conclusion']);print(annotations['limitations'])
''')
add('markdown', '''
## 6. Verified diagnostics; final methodology remains pending

Systematic length shift is established; how much semantic quality it loses is not quantified. No new minimum length, semantic selection objective, budget, source length or frozen manifest. Next: review a candidate-construction/tie-breaking revision and run a new pilot before any full-cohort freeze. Current selection artifacts remain untouched.
''')
add('code', '''
assert validation['run_arithmetic_verified']==19800 and validation['counterfactual_sets_verified']==30
assert completion['source_metadata_unchanged'] and completion['new_length_filter'] is None
assert completion['semantic_model'] is None and completion['final_review_budget'] is None
display(pd.DataFrame([validation]));print('Read-only notebook checks PASS; research choices not frozen.')
''')
notebook = {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                  'language_info':{'name':'python','version':'3.11.7'}},'nbformat':4,'nbformat_minor':5}
path = ROOT/'notebooks/07_selected_review_length_and_content_audit.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(f'Generated {path}; {len(cells)} cells.')
