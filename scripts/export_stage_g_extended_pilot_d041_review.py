"""Export the predeclared random25 AFTER actual API execution; no model calls or edits."""
import base64
import copy
import html
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from prepare_stage_g_extended_pilot_d041 import FOLDER, CONFIG
from run_stage_g_extended_pilot_d041 import check_inputs

PREFIX='chatgpt_stage_g_d041_random25_review/'
ZIP=FOLDER/'chatgpt_stage_g_d041_random25_review.zip'
RUBRIC='''# Fixed D040 extended pilot: manual review of the preselected random25

Read each original selected review and mapped original photo yourself. Teacher
verification labels are model judgments, not human ground truth. ChatGPT may be
used only as an optional second opinion after the researcher's own inspection.
Final annotation remains an explicitly user-confirmed human decision.

Review consensus: selected reviews only; no photo/caption or external facts.
Check accurate categories, quantities/customer prevalence, supporting REV/source
quotes, meaningful praise AND criticism, qualified/minority contrasting opinions.
Valid evidence IDs do not by themselves prove that a claim follows from the text.

Visual claims: must follow the exact LED-visible caption AND the mapped photo.
No photo-only replacement details. Accept ordinary compatible descriptions and
correct simple content; lack of detail is not a factual error. Inspect specific
ingredients, identities, counts, actions and relationships without requiring
unnecessarily precise taxonomy. Use uncertain for genuine ambiguity. Do not invent
negative examples or treat rejected/uncertain teacher verdicts as automatic truth.
Inspect those verdicts too for excessive strictness and lost useful information.

Flag exact/equivalent claim-caption-photo verdict contradictions across Unfiltered
and Filtered. Different evidence/claims may legitimately differ. Never force
agreement or manually rewrite the API outputs. Zero Filtered captions retain
review consensus and the empty visual header, with no substituted captions.

These 25 IDs were selected before generation. Keep them separate from additional
technical flags; retain failed/incomplete cases in the package. Schema PASS is
structural evidence, not a factual-accuracy certificate. Record what was actually
inspected; if an image/review cannot be opened, mark NOT REVIEWED.

For each issue report business_id, exact claim, condition, REV/review_id and/or
IMG_CAP/PHOTO/photo_id, observation, and whether it is error, unsupported,
uncertain, acceptable, correct-but-simple, overly strict, or missing contrast.
Do not modify source data, K5, 5000/4000/750 budgets, StageF, model or prompts.
'''


def jb(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def page(cases,images):
    esc=lambda x:html.escape(str(x))
    blocks=[]
    for case in cases:
        bid=case['business_id'];targets=case.get('final_targets') or {}
        parts=[f'<article id="b-{esc(bid)}"><h2>{esc(bid)}</h2><p>{esc(case["split"])} · {esc(case["status"])}</p>',
            '<h3>Three student targets</h3>']
        for cond in ['reviews_only','unfiltered','filtered']:
            text=targets.get(cond,'NOT AVAILABLE — preserved failure/incomplete case; no fabricated target')
            parts.append(f'<details><summary>{esc(cond)}</summary><pre>{esc(text)}</pre></details>')
        parts.append('<h3>Review consensus and exact support</h3>')
        cp=case['components']['shared'].get('checkpoint') or {}
        for cl in cp.get('resolved_output',{}).get('claims',[]):
            parts.append('<details><summary>'+esc(cl['text'])+'</summary><pre>'+esc(json.dumps(cl['support'],ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<h3>Complete selected reviews, unchanged order</h3>')
        for review in case['reviews']:
            parts.append(f'<details><summary>{esc(review["review_marker"])} · {esc(review["review_id"])} · {review["stars"]}★ · {esc(review["date"])}</summary><pre>{esc(review["text"])}</pre></details>')
        for cond in ['unfiltered','filtered']:
            component=case['components'][cond];caps=component['captions']
            cp=component.get('checkpoint') or {}
            assessments=cp.get('output',{}).get('assessments',[])
            parts.append(f'<h3>{esc(cond)} captions, original photos and teacher judgments</h3>')
            if not caps:parts.append('<p>Zero captions: empty visual header; no API visual call and no substitute evidence.</p>')
            for cap in caps:
                a=next((x for x in assessments if x['caption_id']==cap['caption_id']),None)
                parts.append('<section class="photo"><a href="'+esc(images[cap['photo_id']])+'" target="_blank" rel="noopener"><img src="'+esc(images[cap['photo_id']])+'" alt="'+esc(cap['photo_id'])+'"></a><div><p><code>'+esc(cap['caption_id']+' / '+cap['photo_marker']+' / '+cap['photo_id'])+'</code></p><p>'+esc(cap['caption'])+'</p>')
                if a is None:parts.append('<p>Assessment unavailable; inspect preserved raw failure evidence separately.</p>')
                else:
                    for claim in a['claims']:
                        parts.append('<div class="claim"><b>'+esc(claim['verdict'])+'</b><p>'+esc(claim['text'])+'</p><p>Exact caption quote: '+esc(claim['caption_quote'])+'</p><small>'+esc(claim['explanation'])+'</small></div>')
                parts.append('</div></section>')
        parts.append('<details><summary>Raw component metadata / validation / API usage</summary><pre>'+esc(json.dumps(case['components'],ensure_ascii=False,indent=2))+'</pre></details></article>')
        blocks.append(''.join(parts))
    nav=''.join('<a href="#b-'+esc(c['business_id'])+'">'+esc(c['business_id'])+'</a>' for c in cases)
    return ('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage G · D041 random25 review</title>'
        '<style>body{margin:0;background:#f4f5f0;color:#26392f;font:16px system-ui;line-height:1.55}main{max-width:1200px;margin:auto;padding:24px}nav{display:flex;flex-wrap:wrap;gap:8px}nav a{font-size:12px}article{background:white;padding:24px;margin:24px 0;border:1px solid #d3ddd5;border-radius:12px}details{border-top:1px solid #ddd;padding:10px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px system-ui}code{overflow-wrap:anywhere}.photo{display:grid;grid-template-columns:40% 1fr;gap:20px;margin:20px 0}.photo img{width:100%;max-height:520px;object-fit:contain}.claim{padding:12px;border-left:3px solid #75977c;margin:10px 0;background:#f8faf6}@media(max-width:720px){.photo{grid-template-columns:1fr}main{padding:12px}}</style><main><h1>D041 · 25 cases selected before generation</h1>'
        '<p>Teacher verdicts are model judgments. Structural checks do not establish factual correctness. You remain the human annotator; optional ChatGPT discussion never automatically changes labels. Click photos to open original images.</p><nav>'+nav+'</nav>'+''.join(blocks)+'</main></html>').encode('utf-8')


def export():
    cfg,pre,jobs=check_inputs();out=FOLDER/'execution_v1'
    if not (FOLDER/'validation_report.json').exists():raise ValueError('Run actual API execution and independent result validation first')
    if ZIP.exists():raise FileExistsError('Preserve existing review export; do not overwrite')
    manual=read(FOLDER/'manual_review_25_manifest.json');chosen={b['business_id']:b for b in manual['businesses']}
    assert len(chosen)==25
    images={};cases=[];files={};protected={}
    def original(path):
        b=path.read_bytes();protected[path.relative_to(ROOT).as_posix()]=digest(b);return b
    for bid,b in chosen.items():
        these=[j for j in jobs if j['business_id']==bid];review=next(j for j in these if j['condition']=='shared')
        case={'case_id':bid,'business_id':bid,'split':b['split'],'selection':'pre-generation random25, not flagged-case selection',
            'human_verified':False,'reviews':review['evidence']['reviews'],'components':{}}
        targetpath=out/'provisional_targets'/(bid+'.json')
        target=json.loads(original(targetpath)) if targetpath.exists() else None
        case['final_targets']=target['targets'] if target else None
        case['status']='STRUCTURAL_ACCEPTED_NOT_HUMAN_VERIFIED' if target else 'INCOMPLETE_OR_FAILED_PRESERVED_NOT_EXCLUDED'
        for job in these:
            comp={'job_id':job['job_id'],'captions':copy.deepcopy(job['evidence'].get('captions',[]))}
            path=out/'checkpoints'/(job['job_id']+'.json')
            comp['checkpoint']=json.loads(original(path)) if path.exists() else None
            raw=out/'raw_responses'/path.name
            comp['raw_API_response_file']='raw_api_responses/'+path.name if raw.exists() else None
            if raw.exists():files[comp['raw_API_response_file']]=original(raw)
            for cap,photo in zip(comp['captions'],job['images']):
                data=original(ROOT/photo['relative_path']);assert digest(data)==photo['raw_sha256']
                name='images/'+cap['photo_id']+'.jpg';images[name]=data;cap['image_file']=name;cap['original_image_sha256']=digest(data)
            case['components'][job['condition']]=comp
        cases.append(case)
        files['businesses/'+bid+'.json']=jb(case)
    files['cases.json']=jb({'schema_version':'stage_g_D041_preselected_random25_review_v1','export_only':True,'businesses':cases,
        'human_verified_businesses':0,'additional_flags_are_separate':True,'sampling_manifest_sha256':file_sha(FOLDER/'sampling_manifest.json')})
    files['rubric.md']=RUBRIC.encode('utf-8')
    files['README.md']=('Extract the WHOLE ZIP and open index.html. All25 randomly preselected cases are retained, including failures. '\
        'Read rubric.md, cases.json, reviews and original photos. Teacher labels are not human labels. '\
        'ChatGPT should return exact business/REV/IMG_CAP/PHOTO IDs and distinguish errors, uncertainty, correct-simple content and excessive strictness. '\
        'additional_flagged_cases.json is separate from the random25 sample. No targets or labels are edited by this export.\n').encode()
    for relative in ['manual_review_25_manifest.json','sampling_manifest.json','validation_report.json','technical_flags.json','pilot_usage_estimate.json']:
        files['reports/'+relative]=original(FOLDER/relative)
    files['additional_flagged_cases.json']=original(FOLDER/'technical_flags.json')
    files['protocol/config.json']=original(CONFIG)
    for path in (FOLDER/'protocol_snapshot').rglob('*'):
        if path.is_file():files['protocol/frozen_D040/'+path.relative_to(FOLDER/'protocol_snapshot').as_posix()]=original(path)
    files['THESIS_PROGRESS.md']=original(ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md')
    files['review_response_template.json']=jb({'role':'researcher_review_after_own_inspection_optional_ChatGPT_comparison',
        'cases':[{'business_id':c['business_id'],'inspected_photo_ids':[],'read_review_ids':[],
            'observations':[],'human_final_decision':None,'human_verified':False} for c in cases]})
    image_map={name[7:-4]:name for name in images}
    files['index.html']=page(cases,image_map)
    for name,data in files.items():assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}',data),'Possible secret in export: '+name
    checks={name:digest(data) for name,data in {**files,**images}.items()}
    files['export_manifest.json']=jb({'export_only':True,'businesses':25,'original_images':len(images),'file_sha256':checks,
        'existing_labels_and_targets_changed':False,'API_calls':0})
    with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in {**files,**images}.items():z.writestr(PREFIX+name,data)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for name,h in checks.items():assert digest(z.read(PREFIX+name))==h
        restored=json.loads(z.read(PREFIX+'cases.json'));assert restored['businesses']==cases
        for c in cases:
            for component in c['components'].values():
                for cap in component['captions']:assert cap['image_file'] in images
    local=FOLDER/'human_review_25_ui/dist/index.html';local.parent.mkdir(parents=True,exist_ok=True)
    local.write_bytes(page(cases,{name[7:-4]:'data:image/jpeg;base64,'+base64.b64encode(data).decode() for name,data in images.items()}))
    (local.parent.parent/'OPEN_REVIEW.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0dist\\index.html"\r\n')
    for rel,h in protected.items():assert file_sha(ROOT/rel)==h
    report={'status':'EXPORT_INTEGRITY_PASS','cases':25,'original_images':len(images),'selected_reviews':sum(len(c['reviews']) for c in cases),
        'complete_businesses':sum(c['final_targets'] is not None for c in cases),'incomplete_cases_preserved':sum(c['final_targets'] is None for c in cases),
        'zip_path':ZIP.relative_to(ROOT).as_posix(),'zip_bytes':ZIP.stat().st_size,'zip_sha256':file_sha(ZIP),
        'viewer_path':local.relative_to(ROOT).as_posix(),'protected_inputs':protected,
        'API_calls':0,'scientific_outputs_or_human_labels_changed':False,'browser_visual_QA':False}
    immutable(FOLDER/'human_review_bundle_report.json',report)
    print({k:report[k] for k in ['status','cases','original_images','zip_path','zip_bytes','viewer_path']})


if __name__=='__main__':export()
