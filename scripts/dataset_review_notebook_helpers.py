"""Read-only helpers for the single dataset/filtering/review notebook."""
from pathlib import Path
import hashlib, html, json, re
import numpy as np
import pandas as pd
import orjson
from IPython.display import display, HTML

ROOT=Path(__file__).resolve().parents[1]


def load_json(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def checked_csv(relative, expected=None):
    path=ROOT/relative
    if expected:
        with path.open('rb') as f:
            h=hashlib.sha256()
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        assert h.hexdigest()==expected,relative
    return pd.read_csv(path,keep_default_na=False)


def stage_status():
    items=[('Raw pool / restaurants / photos','outputs/cohort/validation_report.json','validation_passed','Đã kiểm chứng pool 10.197/cohort 2.300'),
           ('English eligibility trước sampling','outputs/english_eligibility/validation_report.json','validation_passed','Gate >=10 English/5 local photos'),
           ('C1 full English/token filtering','outputs/stage_c/review_diagnostics_v1/validation_report.json','validation_passed','Nhãn tự động, giữ original text'),
           ('C2 v3 pilot 56','outputs/stage_c/review_selection_budget_pilot_v3/validation_report.json','validation_passed','Historical exploratory method'),
           ('C2 v3 expanded 2.200','outputs/stage_c/review_selection_train_validation_v3/validation_report.json','validation_passed','Token/rating/time integrity valid'),
           ('Short-review length/content audit','outputs/stage_c/review_selection_length_audit_v1/validation_report.json','validation_passed','Systematic length shift found'),
           ('C2 v4 revised pilot','outputs/stage_c/review_selection_length_balanced_pilot_v1/validation_report.json','validation_passed','Revision pilot; chưa chốt final methodology'),
           ('V4 tradeoffs / candidate coverage audit','outputs/stage_c/review_selection_tradeoff_audit_v1/validation_report.json','validation_passed','Coverage gaps identified; counterfactuals only, standalone v4 not final'),
           ('C2 v4.1 shared candidate pilot','outputs/stage_c/review_selection_shared_pool_pilot_v1/validation_report.json','validation_passed','Fresh pilot, coverage correction; final method still pending'),
           ('Residual coverage / objective equivalence audit','outputs/stage_c/review_selection_residual_audit_v1/validation_report.json','validation_passed','Missing v4 candidates and same-histogram alternatives; diagnostics only'),
           ('C2 v4.2 source-length secondary-rule pilot','outputs/stage_c/review_selection_source_length_tie_pilot_v1/validation_report.json','validation_passed','Exact primary histograms preserved; source length matching, not quality proof'),
           ('C2 v4.2 expanded2200 train/validation','outputs/stage_c/review_selection_source_length_expanded_v1/validation_report.json','validation_passed','Expanded diagnostics; final allocation/2300 manifest still pending'),
           ('LED source/token allocation audit003l','outputs/stage_c/led_source_token_allocation_audit_v1/validation_report.json','validation_passed','Exact whole-source capacity;4000+750 is a proposal, no actual captions/final allocation'),
           ('D1 photo integrity and duplicate indexing','outputs/stage_d/photo_integrity_duplicate_audit_v1/validation_report.json','validation_passed','All30027 readable;227 cross-split groups recorded; D-022 keeps original pool/main100 plus secondary83')]
    rows=[]
    for label,path,field,note in items:
        exists=(ROOT/path).is_file();passed=exists and bool(load_json(path).get(field))
        rows.append({'step':label,'status':'Đã kiểm chứng' if passed else ('Cần kiểm tra' if exists else 'Đang làm / chưa validated'),'notes':note,'artifact':path})
    for label,note in [('Final review/metadata/caption allocation + manifest','TBD; phải chốt shared reviews cho mọi LED condition'),
                       ('BLIP-2 E0 feasibility/captions','D-022 keeps pool/splits; see preparation/environment/generation reports; preparation alone is not a passed model pilot'),
                       ('Caption filtering + teacher targets','TBD k/protocol; chưa có outputs'),
                       ('LoRA LED training + evaluation','Chưa chạy; local GPU feasibility/Vast.ai A2 pending'),
                       ('External generalization + representative 30-business human evaluation','Kế hoạch thesis; dataset/sampling rules pending')]:
        rows.append({'step':label,'status':'Sẽ làm / pending','notes':note,'artifact':''})
    return pd.DataFrame(rows)


def load_business_originals(bid, metadata, cohort):
    found=cohort.loc[cohort.business_id.eq(bid)]
    if len(found)!=1:raise ValueError('Business ID không nằm trong cohort 2.300.')
    split=found.iloc[0]['split'];rows=[]
    for line in (ROOT/'data/processed/yelp_2300_v1/reviews'/f'{split}.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['business_id']==bid:rows.append(r)
    labels=metadata.loc[metadata.business_id.eq(bid)].set_index('review_id')
    assert len(rows)==len(labels) and {r['review_id'] for r in rows}==set(labels.index)
    for r in rows:
        m=labels.loc[r['review_id']]
        assert hashlib.sha256(r['text'].encode()).hexdigest()==m['text_sha256']
        r.update(language=m['language'],language_reason=m['language_reason'],body_tokens=int(m['led_tokens_without_special']),
                 url_only_flag=bool(re.fullmatch(r'(?:https?://|www\.)\S+\s*',r['text'].strip(),flags=re.I)))
    return sorted(rows,key=lambda r:(r['date'],r['review_id'])),found.iloc[0].to_dict()


def selected_ids(bid,budget):
    result={};metrics=[]
    for version,directory,method in [('v3','review_selection_train_validation_v3','budget_aware'),
                                     ('v4','review_selection_length_balanced_pilot_v1','v4_length_balanced'),
                                     ('v4_1','review_selection_shared_pool_pilot_v1','v4_1_shared_pool'),
                                     ('v4_2','review_selection_source_length_tie_pilot_v1','v4_2_length_balanced'),
                                     ('v4_2_control','review_selection_source_length_tie_pilot_v1','v4_2_primary_control'),
                                     ('v4_2_expanded','review_selection_source_length_expanded_v1','v4_2_length_balanced'),
                                     ('v4_2_expanded_control','review_selection_source_length_expanded_v1','v4_2_primary_control')]:
        if version.startswith('v4_2_expanded'):
            check=ROOT/'outputs/stage_c'/directory/'validation_report.json'
            if not check.is_file() or not json.loads(check.read_text(encoding='utf-8')).get('validation_passed'):result[version]=None;continue
        path=ROOT/'outputs/stage_c'/directory/'business_runs'/f'{hashlib.sha256(bid.encode()).hexdigest()}.json'
        if not path.is_file():result[version]=None;continue
        payload=json.loads(path.read_text(encoding='utf-8'))
        r=next(r for r in payload['runs'] if r['method']==method and r['review_budget']==budget)
        result[version]=set(r['selected_review_ids'])
        metrics.append({'version':version,'count':r['final_retained_review_count'],'review_block_tokens':r['tokens_after_budget_enforcement'],
                        'rating_jsd':r['rating_jsd_after'],'temporal_jsd':r['temporal_jsd_after'],
                        'status':r['selection_status']})
    return result,pd.DataFrame(metrics)


def render_business(bid,budget,view,page,page_size,metadata,cohort):
    originals,info=load_business_originals(bid,metadata,cohort);chosen,metrics=selected_ids(bid,budget)
    display(pd.DataFrame([{k:info.get(k) for k in ['business_id','name','split','reviews','photos','city','state']}]))
    display(metrics)
    source_mean=np.mean([r['body_tokens'] for r in originals if r['language']=='en'])
    display(pd.DataFrame([{'version':version,'available':ids is not None,'selected_BODY_mean':np.mean([r['body_tokens'] for r in originals if ids is not None and r['review_id'] in ids]) if ids else None,'source_English_BODY_mean':source_mean} for version,ids in chosen.items()]))
    versions={'v3_selected':'v3','v4_selected':'v4','v4_1_selected':'v4_1','v4_2_selected':'v4_2','v4_2_control_selected':'v4_2_control','v4_2_expanded_selected':'v4_2_expanded','v4_2_expanded_control_selected':'v4_2_expanded_control'}
    if view in versions and chosen[versions[view]] is None:
        print('No validated selection for this stage/business. Pilot80 and expanded2200 train/validation do not include the held-out test100 or a final2300 manifest.');return
    for r in originals:
        r['kept_English']=r['language']=='en'
        r['v3_selected']=None if chosen.get('v3') is None else r['review_id'] in chosen['v3']
        r['v4_selected']=None if chosen.get('v4') is None else r['review_id'] in chosen['v4']
        r['v4_1_selected']=None if chosen.get('v4_1') is None else r['review_id'] in chosen['v4_1']
        r['v4_2_selected']=None if chosen.get('v4_2') is None else r['review_id'] in chosen['v4_2']
        r['v4_2_control_selected']=None if chosen.get('v4_2_control') is None else r['review_id'] in chosen['v4_2_control']
        r['v4_2_expanded_selected']=None if chosen.get('v4_2_expanded') is None else r['review_id'] in chosen['v4_2_expanded']
        r['v4_2_expanded_control_selected']=None if chosen.get('v4_2_expanded_control') is None else r['review_id'] in chosen['v4_2_expanded_control']
    if view=='English_removed':rows=[r for r in originals if not r['kept_English']]
    elif view=='English_pool':rows=[r for r in originals if r['kept_English']]
    elif view in versions:rows=[r for r in originals if r[view]]
    else:rows=originals
    lo=page*page_size;part=rows[lo:lo+page_size]
    print(f'{len(originals)} raw originals; {sum(r["kept_English"] for r in originals)} en; {sum(not r["kept_English"] for r in originals)} removed by language. View {view}: {len(rows)} reviews; page {page}, rows {lo+1 if part else 0}–{lo+len(part)}. Review text is complete; paging does not truncate originals.')
    blocks=[]
    for r in part:
        label=f"{r['review_id']} | {int(r['stars'])} stars | {r['date']} | {r['body_tokens']} BODY tokens | language={r['language']} ({r['language_reason']}) | v3={r['v3_selected']} v4={r['v4_selected']} v4.1={r['v4_1_selected']} v4.2pilot={r['v4_2_selected']} v4.2expanded={r['v4_2_expanded_selected']}"
        if r['url_only_flag']:label+=' | URL-only diagnostic flag'
        blocks.append('<details open><summary>'+html.escape(label)+'</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape(r['text'])+'</pre></details>')
    display(HTML('\n'.join(blocks) or '<p>Không có review trong trang này.</p>'))


def optional_viewer(metadata,cohort,default_bid):
    try:import ipywidgets as widgets
    except ImportError:
        print('ipywidgets chưa có; dùng render_business(...) hoặc sửa BUSINESS_ID_TO_VIEW rồi Run All.');return
    options={f"{r['name']} | {r['business_id']} | {r['split']}":r['business_id'] for r in cohort.to_dict('records')}
    default=next(label for label,bid in options.items() if bid==default_bid)
    business=widgets.Combobox(value=default,options=sorted(options),description='Business:',layout=widgets.Layout(width='95%'),ensure_option=True)
    budget=widgets.Dropdown(options=[3500,4000,4500],value=3500,description='Budget:')
    view=widgets.Dropdown(options=['v4_2_expanded_selected','v4_2_expanded_control_selected','v4_2_selected','v4_2_control_selected','v4_1_selected','v4_selected','v3_selected','English_pool','English_removed','raw'],value='v4_2_expanded_selected',description='View:')
    page=widgets.IntText(value=0,description='Page:');size=widgets.IntSlider(value=20,min=5,max=50,description='Page size:')
    button=widgets.Button(description='Xem review',button_style='primary');output=widgets.Output()
    def show(_):
        with output:
            output.clear_output(wait=True)
            try:render_business(options[business.value],budget.value,view.value,max(page.value,0),size.value,metadata,cohort)
            except Exception as error:print(type(error).__name__,str(error))
    button.on_click(show);display(widgets.VBox([business,widgets.HBox([budget,view]),widgets.HBox([page,size,button]),output]))
