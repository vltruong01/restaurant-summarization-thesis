"""Read-only pHash0..64 diagnostic on existing verified calibration pairs; no freeze."""
from pathlib import Path
from collections import Counter
import json
import base64
import html
import numpy as np
import yaml
from stage_f_verification_workflow import read,write,sha,CHOICES
from analyze_stage_f_consistency_stability import validate_declared_prefix
from stage_f_math import threshold_search
from stage_f_d030_protocol import validate_compatibility

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
ASSISTED=ROOT/'outputs/stage_f/ai_assisted_calibration_v1'
STATE=ROOT/'outputs/stage_f/methodology_amendment_d030'
OUT=ROOT/'outputs/stage_f/phash_pair_diagnostic_v1'

# These are separate Codex image observations after viewing every source image.
# They are NOT human annotations, not substitutions for labels, and never fit thresholds.
OBSERVATIONS={
 'PAIR_ELF4NKRlR8992VmScH27aQ_PZgRCtjso7KXldgIw0ZvnQ':'Left: three clear cold-drink cups; right: a single white takeaway cup. Shared branding/drink theme, with different cup types, arrangement and background. This does not by itself establish a transformed copy of one photograph.',
 'PAIR_6XqyRFEX8IrSKGUDYEWX-w_OyxDDpgkEOxgYyyepx6TLA':'Left: a masked staff member pouring liquid behind the counter; right: a different visible staff member posed at a doorway/table with a drink. Different people, actions and composition despite shared cafe context.',
 'PAIR_OyxDDpgkEOxgYyyepx6TLA_PZgRCtjso7KXldgIw0ZvnQ':'Left: staff member at a doorway/table holding a drink; right: a close view of one white cup. Different subject scale and scene composition; shared cafe/drink context is not sufficient evidence of image duplication.',
 'PAIR_QYsma8L9GiK20heLvEtUdw_kOSfAxfZJ6Fabmzyr2HHpA':'Both show sushi on long plates, but toppings, plate arrangement, angle and background differ. Similar food topic does not establish the same original photograph or a small crop/resize variant.',
 'PAIR_L9JytFjX8Vogb4XSqN1BRA_ZHhCNUYLoZ5MTI6pcmL91w':'Both show a crowded dance-floor/bar scene with similar ceiling/signage; foreground people and positions differ. Similar venue/event content is visible, but a near-duplicate frame relationship needs an explicit definition.',
 'PAIR_DP9LyhEVEU53MyszSDOIrQ_L9JytFjX8Vogb4XSqN1BRA':'Both show dancing in a similar room; framing, foreground subjects and their positions differ. These may be related scene captures; they are not visibly established as one image with a simple format transformation.',
 'PAIR_DP9LyhEVEU53MyszSDOIrQ_XKJPApOoK5TX2Sns7-VoVg':'Left: dancing under colored lights; right: a brighter seated/standing bar-table scene. Shared venue/social context, with different composition and activity.',
 'PAIR_Jc6miNFPgpYk7FKdEqP8EQ_ZHhCNUYLoZ5MTI6pcmL91w':'Left: close view of a seated patron at a bar; right: a wide crowd/dance-floor view. Different main subject and viewpoint despite related bar context.',
 'PAIR_2CdlyMR8By_QNuYVDuGyrQ_kvUNRlOqXak0O8Tc-vf7bQ':'Left: close dining-table setting with fish wall decor; right: wide restaurant/bar interior. Interior theme may overlap, but scene scale and framing differ.',
 'PAIR_NtfMQtJYK7BzmtuW1Va_Iw_k9v3jV584j39vylCoa2jmQ':'Left: a burger wrapped in lettuce; right: a baked dish with salad on a plate. Different visible dishes and arrangement, although both are food photographs.',
 'PAIR_GlVb0v_8G71G7az6bqPOpw_KZngvlH8PIXvsgA6q9dn_g':'Both exteriors show Red Robin signage; building shapes, camera views and lighting differ. Shared brand/restaurant category is not proof of the same physical scene or photograph.',
 'PAIR_NtfMQtJYK7BzmtuW1Va_Iw_WPvjGnH6G7XLL-vz-_CPVw':'Left: lettuce-wrapped burger on a board; right: bun burger with fries in a basket. Similar food family, with different presentation and composition.',
 'PAIR_bRqWP-RJOwgOrrgX-PriJw_hCx58FQz81-Lqtm-gFoMRw':'Both show a restaurant interior with large windows and columns; viewpoint and visible patrons/table positions differ. Possible related venue captures; deciding near duplication versus repeated venue coverage requires a precise criterion.',
 'PAIR_VSOwSluQDZabtyh5qtKYxQ_tVzxeAOd__eo68fFpIbe0Q':'Both show oyster platters, but plate layout/count, camera angle, table background and surrounding objects differ. Shared dish type does not establish one original photograph.',
 'PAIR_k5Mw-eAq_-Q5aaB8xsauxg_k_0uNYJTY2IRqXgm2eJ8vw':'Left: close pink cocktail with mint; right: overhead arrangement of several cocktails. Similar drink content, but different field of view and object arrangement.',
 'PAIR_YivZMiJW2pgyWKjGNZIP1g_dqe6ICTGFY1egztiYXLPZw':'Both show bar-counter interiors. The second has a layered/reflected or blurred appearance and different framing. Potential scene similarity is visible, but an exact transformation/near-duplicate relationship is not established by these views alone.'
}


def metrics(values,truth,threshold):
    positive=sum(truth);negative=len(truth)-positive
    assert positive and negative
    predicted=[int(v)<=threshold for v in values]
    tp=sum(p and y for p,y in zip(predicted,truth));fp=sum(p and not y for p,y in zip(predicted,truth))
    tn=negative-fp;fn=positive-tp
    return {'threshold':int(threshold),'TP':tp,'TN':tn,'FP':fp,'FN':fn,
            'same_visual_content_recall':tp/positive,'distinct_content_specificity':tn/negative,
            'precision':tp/(tp+fp) if tp+fp else 0.,'balanced_accuracy':(tp/positive+tn/negative)/2,
            'distinct_pairs_marked_duplicate_count':fp,'distinct_pairs_marked_duplicate_pct':100*fp/negative,
            'positive_pairs_missed_count':fn,'positive_pairs_missed_pct':100*fn/positive,
            'predicted_duplicate_pairs':tp+fp,'predicted_duplicate_pairs_pct':100*(tp+fp)/len(truth),
            'positive_total':positive,'negative_total':negative}


def point_ids(rows,threshold):
    result={key:[] for key in ['TP','TN','FP','FN']}
    for row in rows:
        p=row['phash_hamming']<=threshold;y=row['positive']
        result['TP' if p and y else 'FP' if p else 'FN' if y else 'TN'].append(row['case_id'])
    return result


def plot(rows,table,best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for positive,label,color in [(True,'Human same_visual_content (16)','#dc2626'),(False,'Human distinct_content (124)','#2563eb')]:
        values=[r['phash_hamming'] for r in rows if r['positive']==positive]
        axes[0].hist(values,bins=np.arange(-.5,65.5,2),weights=np.ones(len(values))/len(values),alpha=.5,label=label,color=color)
    axes[0].axvline(12,color='black',linestyle='--',label='Production grid maximum 12')
    axes[0].set(xlabel='Stored pHash64 Hamming distance',ylabel='Fraction of class',title='Observed score overlap');axes[0].legend(fontsize=8)
    for key,label in [('same_visual_content_recall','Positive recall'),('distinct_content_specificity','Negative specificity'),('balanced_accuracy','Balanced accuracy'),('precision','Precision')]:
        axes[1].plot([r['threshold'] for r in table],[r[key] for r in table],label=label)
    axes[1].axvline(best['threshold'],color='grey',linestyle=':',label='Exploratory best (not selected)')
    axes[1].set(xlabel='Diagnostic threshold (duplicate if distance <= cutoff)',ylabel='Rate',title='0..64 diagnostic; production grid unchanged',ylim=(-.02,1.03));axes[1].legend(fontsize=8)
    fig.suptitle('140 verified calibration pairs — no labels/config/thresholds changed; no K-study')
    fig.savefig(OUT/'phash_diagnostics.png',dpi=180);plt.close(fig)


def build_viewer(positive_rows,best):
    cards=[];image_hashes={}
    for index,row in enumerate(positive_rows,1):
        e=html.escape;photos=[]
        for pid,path,caption,raw in zip(row['photo_ids'],row['image_paths'],row['captions'],row['raw_captions']):
            source=ROOT/path;image_hashes[path]=sha(source)
            data=base64.b64encode(source.read_bytes()).decode('ascii')
            photos.append(f'<div class="photo"><img loading="lazy" src="data:image/jpeg;base64,{data}" alt="{e(pid)}"><p>{e(caption)}</p><details><summary>Raw caption / photo ID</summary><p>{e(raw)}</p><code>{e(pid)}</code></details></div>')
        cards.append(f'''<article id="{e(row['case_id'])}" data-distance="{row['phash_hamming']}"><h2>{index}. {e(row['business_name'])} — distance {row['phash_hamming']}</h2>
<div class="photos">{''.join(photos)}</div><p><strong>Nhãn human giữ nguyên:</strong> image_duplicate = {e(row['human_final_label']['image_duplicate'])}; redundancy = {e(row['human_final_label']['redundancy'])}; broad_topic = {e(row['human_final_label']['broad_topic'])}.</p>
<p class="prediction"></p><p>Exact pixel match: {row['exact_pixel_match']}; <code>{e(row['case_id'])}</code></p>
<p class="observation"><strong>Codex — ghi chú thị giác riêng, không phải nhãn human:</strong> {e(OBSERVATIONS[row['case_id']])}</p>
<details><summary>AI suggestion / provenance</summary><pre>{e(json.dumps({'ai_suggested_label':row['ai_suggested_label'],'human_verified':row['human_verified'],'verification_action':row['verification_action'],'verified_at':row['verified_at']},ensure_ascii=False,indent=2))}</pre></details></article>''')
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage F — pHash audit 16 positive pairs</title>
<style>body{font:16px system-ui;margin:24px auto;padding:0 18px;max-width:1200px;background:#f5f7fa;color:#182335}article{background:white;margin:22px 0;padding:20px;border-radius:10px}h2{font-size:20px}.photos{display:flex;gap:20px}.photo{flex:1;min-width:0}.photo img{width:100%;height:340px;object-fit:contain;background:#eef1f4}.notice{background:#fff1cc;padding:16px;border-radius:8px}.observation{background:#eef4ff;padding:14px}pre{white-space:pre-wrap}code{overflow-wrap:anywhere}.toolbar{position:sticky;top:0;background:#f5f7fa;padding:14px 0}input,button{font:inherit;padding:8px}a{color:#163f70}@media(max-width:650px){.photos{display:block}}</style>
<h1>pHash — 16 cặp đã xác nhận same_visual_content</h1>
<p class="notice"><strong>Chỉ đọc và diagnostic.</strong> Không phải batch mới, không sửa nhãn. Bảng thống kê dùng toàn bộ140cặp human-verified (16positive/124negative). Trang này hiển thị đủ16positive theo thứ tự frozen queue. Ghi chú Codex tách riêng, không dùng để fit. Grid production vẫn0/4/8/12; threshold image-duplicate vẫn pending; chưa chạy K-study.</p>
<p><a href="diagnostic_report.md">Báo cáo 65 thresholds</a> · <a href="phash_diagnostics.png">Biểu đồ</a></p>
<div class="toolbar"><label>Cutoff chỉ để xem thử: <input id="cutoff" type="number" min="0" max="64" step="1" value="12"></label> <button id="original">Grid cũ: max12</button> <button id="best">Diagnostic best (không freeze)</button><span id="count"></span></div>
'''+'\n'.join(cards)+'''
<script>const BEST=__BEST__;const input=document.getElementById('cutoff');function render(){const t=Math.max(0,Math.min(64,Math.round(Number(input.value)||0)));let n=0;for(const card of document.querySelectorAll('article[data-distance]')){const pass=Number(card.dataset.distance)<=t;if(pass)n++;card.querySelector('.prediction').textContent=(pass?'Diagnostic: được đánh dấu duplicate':'Diagnostic: chưa được đánh dấu duplicate')+' tại cutoff '+t+'. Đây không phải quyết định production.';}document.getElementById('count').textContent=' '+n+'/16positive được phát hiện; không lưu/chọn threshold.';}input.oninput=render;document.getElementById('original').onclick=()=>{input.value=12;render();};document.getElementById('best').onclick=()=>{input.value=BEST;render();};render();</script></html>'''
    page=page.replace('__BEST__',str(best['threshold']))
    assert page.count('src="data:image/jpeg;base64,')==32
    (OUT/'positive_pair_audit.html').write_text(page,encoding='utf-8')
    assert image_hashes=={path:sha(ROOT/path) for path in image_hashes}
    return image_hashes


def render_report(report):
    best=report['exploratory_best_not_selected']
    lines=['# pHash pair diagnostic — 2026-10-09','',
           '**Diagnostic only. Production grid remains {0,4,8,12}; image threshold remains pending. No labels, frozen thresholds or protocol changed; no K-study.**','',
           'Use all140already verified calibration caption-pair cases:16same_visual_content and124distinct_content; no uncertain image-duplicate labels in this verified set. Unverified pairs do not enter the diagnostic. Pair membership and source captions/photos are unchanged.','',
           'Stored distances are independently checked against XOR/bit-count of the frozen photo-plan pHash64 values. No hash algorithm, preprocessing, score, model or human annotation is regenerated. Positive/negative refer to the existing human labels, not new Codex opinions.','',
           '## Results','',
           f"The exploratory maximum is **{best['balanced_accuracy']:.2%} BA at cutoff{best['threshold']}** under the inherited tie rules. Cutoffs34and35give identical predictions on these cases;35wins the existing higher-cutoff tie-break, not a unique better operating point.",'',
           f"At35: TP{best['TP']}/TN{best['TN']}/FP{best['FP']}/FN{best['FN']}; positive recall{best['same_visual_content_recall']:.2%}; negative specificity{best['distinct_content_specificity']:.2%}; precision{best['precision']:.2%}. "
           f"It flags{best['distinct_pairs_marked_duplicate_count']}/124distinct pairs ({best['distinct_pairs_marked_duplicate_pct']:.2f}%) as duplicates. This is pair-classification error, not a measured count of photos removed by the business-level greedy pipeline.",'',
           'Simply widening the grid until BA exceeds0.5does not establish useful duplicate handling. The score ranges overlap: positives22..40, negatives20..44. The inherited grid detects0/16positives at each of0/4/8/12and givesBA0.5. No threshold is adopted from the widened diagnostic.','',
           '## Full0..64 threshold table','',
           'Predict duplicate when pHash Hamming distance <= cutoff. TP=same_visual_content flagged; TN=distinct_content not flagged; FP=distinct_content flagged; FN=same_visual_content missed.','',
           '| Cutoff | TP | TN | FP | FN | Positive recall | Negative specificity | Precision | BA | Distinct pairs falsely flagged n (%) | Positive pairs missed n (%) |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['threshold_table']:
        lines.append(f"| {r['threshold']} | {r['TP']} | {r['TN']} | {r['FP']} | {r['FN']} | {r['same_visual_content_recall']:.2%} | {r['distinct_content_specificity']:.2%} | {r['precision']:.2%} | {r['balanced_accuracy']:.2%} | {r['distinct_pairs_marked_duplicate_count']} ({r['distinct_pairs_marked_duplicate_pct']:.2f}%) | {r['positive_pairs_missed_count']} ({r['positive_pairs_missed_pct']:.2f}%) |")
    lines+=['','![pHash overlap and diagnostic curves](phash_diagnostics.png)','',
            '## Visual audit of all16existing positives','',
            '[Open the read-only side-by-side audit](positive_pair_audit.html). The source images were visually inspected by Codex; observations are separate diagnostic notes, not human annotations or accepted label changes. All existing human_final_label values remain unchanged and are the only targets used in the table.','',
            'Several pairs share a brand/food/venue theme yet show different objects, actions, arrangements or viewpoints. Some bar/interior pairs may depict related captures of a similar physical scene. These observations raise a construct question: are the existing same_visual_content labels intended to mean near-duplicate photographs/variants, or broader repeated semantic content? The source labels are not retroactively reinterpreted here.','',
            '## Limits and next decision','',
            'This is a selected, score-stratified calibration queue, not a representative test of all business photos. Pairs share photos/businesses. Max-BA evaluation on the same labels is apparent calibration performance and can be optimistic. No claim of population accuracy or of incorrectly labeled human ground truth is made.','',
            'Review the16pair images and the intended image-duplicate construct before any grid/criterion amendment. Do not broaden the production grid merely to clear the numerical gate; do not automatically replace labels, disable the criterion or demand a new batch. Any research change requires an explicit researcher decision. K-study remains blocked by the unresolved required image criterion.','',
            'Machine-readable outputs: diagnostic_report.json, threshold_table.json, verified_pair_records.json, positive_pair_audit.json and codex_visual_observations.json. Exact predicted-duplicate/non-duplicate case IDs at the original cutoffs and exploratory best are included in diagnostic_report.json.','']
    (OUT/'diagnostic_report.md').write_text('\n'.join(lines),encoding='utf-8')


def run():
    assert not (OUT/'diagnostic_report.json').exists(),'Use a new namespace for a new diagnostic state'
    OUT.mkdir(parents=True,exist_ok=True);validate_compatibility()
    names=list(read(STATE/'protected_before_sha256.json'))
    names+=['configs/stage_f_validation_protocol.yaml','configs/stage_f_consistency_stability.yaml','DECISIONS.md',
            'specs/006b_stage_f_validation_execution.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']
    names+=[str(p.relative_to(ROOT)).replace('\\','/') for p in STATE.rglob('*') if p.is_file()]
    before={name:sha(ROOT/name) for name in sorted(set(names))}
    write(OUT/'protected_before_sha256.json',before)
    cfg=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
    assert cfg['calibration_search']['image_hamming_grid']==[0,4,8,12]
    write(OUT/'diagnostic_settings.json',{'status':'EXPLORATORY_NOT_PRODUCTION_GRID_CHANGE','diagnostic_grid':list(range(65)),
          'production_grid':[0,4,8,12],'positive_label':'same_visual_content','negative_label':'distinct_content',
          'decision':'distance_at_or_below_cutoff','ties':cfg['calibration_search']['ties'],'threshold_freeze':False,
          'method_amendment':False,'new_annotation_batch':False,'K_study':False,
          'Codex_observations':'separate_notes_not_used_as_labels'})
    labels=read(ASSISTED/'latest_human_export.json');dataset=read(BASE/'human_check/annotation_dataset.json')
    order=read(ASSISTED/'verification_order.json');selection=read(ASSISTED/'active_selection.json');ai=read(ASSISTED/'ai_suggestions.json')
    actual,prefix=validate_declared_prefix(dataset,order,selection,ai,labels)
    source={r['case_id']:r for r in dataset['cases']};private=read(BASE/'human_check/private_calibration_scores.json')
    plan={p['photo_id']:p for p in read(BASE/'validation_photo_plan.json')}
    rows=[];uncertain=[]
    for cid in order['orders']['caption_pair']:
        h=actual[cid]
        if not h['human_verified']:continue
        c=source[cid];assert c['kind']=='caption_pair'
        label=h['human_final_label']['image_duplicate']
        if label=='uncertain':uncertain.append(cid);continue
        assert label in ['same_visual_content','distinct_content']
        p1,p2=[plan[pid] for pid in c['photo_ids']]
        assert p1['role']==p2['role']=='calibration' and p1['split']==p2['split']=='validation'
        value=private[cid]['phash_hamming'];assert 0<=value<=64 and value==int(value)
        assert (int(p1['phash64_hex'],16)^int(p2['phash64_hex'],16)).bit_count()==value
        for path,p in zip(c['image_paths'],[p1,p2]):assert sha(ROOT/path)==p['raw_sha256']
        rows.append({'case_id':cid,'business_id':c['business_id'],'business_name':c['name'],
                     'photo_ids':c['photo_ids'],'image_paths':c['image_paths'],
                     'captions':c['normalized_captions'],'raw_captions':c['raw_captions'],
                     'human_final_label':h['human_final_label'],'human_verified':True,
                     'ai_suggested_label':h['ai_suggested_label'],'verification_action':h['verification_action'],
                     'verified_at':h['verified_at'],'notes':h.get('notes',''),'phash_hamming':int(value),
                     'exact_pixel_match':p1['pixel_sha256']==p2['pixel_sha256'],
                     'positive':label=='same_visual_content'})
    assert len(rows)==140 and sum(r['positive'] for r in rows)==16 and not uncertain
    positive=[r for r in rows if r['positive']];assert {r['case_id'] for r in positive}==set(OBSERVATIONS)
    values=[r['phash_hamming'] for r in rows];truth=[r['positive'] for r in rows]
    best,_=threshold_search(values,truth,list(range(65)),False,cfg['calibration_search']['minimum_each_binary_class'])
    table=[metrics(values,truth,t) for t in range(65)];point=table[int(best['threshold'])]
    points={str(t):{'metrics':table[t],'case_ids':point_ids(rows,t)} for t in [0,4,8,12,int(best['threshold'])]}
    report={'status':'PHASH_DIAGNOSTIC_REVIEW_REQUIRED_NOT_A_THRESHOLD_SELECTION',
            'counts':{'positive':16,'negative':124,'verified_pairs_used':140,'uncertain_excluded':len(uncertain)},
            'fit_business_count':len({r['business_id'] for r in rows}),'source_declared_prefix':prefix,
            'production_grid_unchanged':[0,4,8,12],'diagnostic_grid':list(range(65)),
            'threshold_table':table,'exploratory_best_not_selected':point,
            'exploratory_equal_prediction_cutoffs':[r['threshold'] for r in table if point_ids(rows,r['threshold'])==point_ids(rows,int(best['threshold']))],
            'operating_points_for_comparison_not_frozen':points,'score_support_by_label':{
              name:{'min':min(r['phash_hamming'] for r in rows if r['positive']==flag),'max':max(r['phash_hamming'] for r in rows if r['positive']==flag),'frequencies':dict(sorted(Counter(r['phash_hamming'] for r in rows if r['positive']==flag).items()))}
              for flag,name in [(True,'same_visual_content'),(False,'distinct_content')]},
            'exact_pixel_matching_pairs':sum(r['exact_pixel_match'] for r in rows),
            'source_label_sha256':sha(ASSISTED/'latest_human_export.json'),
            'pHash_source':'unchanged_frozen_photo_plan_phash64_hex_XOR_bit_count_matches_private_calibration_scores',
            'source_configuration_sha256':sha(ROOT/'configs/stage_f_validation_protocol.yaml'),
            'human_labels_modified':False,'AI_or_Codex_observations_used_for_fit':False,
            'image_threshold_frozen':False,'production_grid_modified':False,'protocol_amended':False,'annotation_batch_opened':False,
            'K_study_run':False,'heldout_test_used':False,'integrity':{'protected_files':len(before),'analysis_code_sha256':sha(Path(__file__))}}
    write(OUT/'diagnostic_report.json',report);write(OUT/'threshold_table.json',table)
    write(OUT/'verified_pair_records.json',rows);write(OUT/'positive_pair_audit.json',positive)
    write(OUT/'codex_visual_observations.json',{'role':'Codex_visual_diagnostic_notes_NOT_human_annotations',
          'source_images_actually_viewed':True,'changes_to_source_human_labels':False,'used_for_threshold_fit':False,
          'observations':[{'case_id':cid,'note':OBSERVATIONS[cid]} for cid in [r['case_id'] for r in positive]]})
    image_hashes=build_viewer(positive,point);plot(rows,table,point);render_report(report)
    assert before=={name:sha(ROOT/name) for name in before}
    assert image_hashes=={path:sha(ROOT/path) for path in image_hashes}
    assert read(STATE/'frozen_remaining_hard_thresholds.json')['thresholds']['near_image_duplicate'] is None
    assert not (BASE/'fixed_filter_pools.json').exists() and not (BASE/'K_study_distribution.json').exists()
    write(OUT/'verification_report.json',{'status':'PASS_READ_ONLY_PHASH_DIAGNOSTIC','protected_files_checked':len(before),
          'protected_files_byte_changed':[],'original_source_images_byte_unchanged':True,'all_private_distances_match_frozen_plan':True,
          'all_verified_non_uncertain_pairs_included':140,'positive_pairs_shown':16,'viewer_embedded_image_occurrences':32,
          'unique_positive_source_images':len(image_hashes),'Codex_notes_separate_not_human_labels':True,
          'production_grid_unchanged':[0,4,8,12],'frozen_thresholds_unchanged':True,'labels_unchanged':True,
          'protocol_amended':False,'K_study_run':False,'analysis_code_sha256':sha(Path(__file__))})
    print(json.dumps({'counts':report['counts'],'best_diagnostic_not_frozen':point,'equal_decision_cutoffs':report['exploratory_equal_prediction_cutoffs'],
                      'report':str(OUT/'diagnostic_report.md'),'viewer':str(OUT/'positive_pair_audit.html')}))
    return report


if __name__=='__main__':run()
