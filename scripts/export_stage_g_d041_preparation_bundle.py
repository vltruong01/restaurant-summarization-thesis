"""Export D041 frozen inputs/readiness only; no teacher calls, targets or annotation edits."""
import argparse
import base64
import copy
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical
from prepare_stage_g_extended_pilot_d041 import FOLDER, CONFIG
from run_stage_g_extended_pilot_d041 import check_inputs

ASSETS=ROOT/'scripts/stage_g_d041_preparation_ui'
VIEW=FOLDER/'preparation_review_ui_v1'
ZIP=FOLDER/'chatgpt_stage_g_d041_preparation_bundle.zip'
PREFIX='chatgpt_stage_g_d041_preparation_bundle/'

README='''# D041 portable PREPARATION review bundle — not generated teacher results

This export contains all100 predeclared unseen businesses, their2848 unchanged
selected reviews, exact Unfiltered/Filtered LED captions and659 original JPEGs.
The random25 manual-review IDs were fixed before generation and are identified.
80train/20validation; no held-out business evidence is included. This is a
stratified diagnostic sample with deliberate validation/zero/minimum-cell
oversampling, not a purely representative random cohort.

D041 API calls=0 and teacher targets=0 at export. There are no fabricated summaries,
model verdicts or new human labels. Existing LED SOURCES are INPUTS, not targets.
The final random25 review package containing actual API outputs will be created
only after separately authorized generation and validation. This export does not
authorize spending or freeze production.

Extract the WHOLE ZIP and open index.html; it uses relative images and no server/API.
cases.json has every business; businesses/<id>.json provides per-case records.
reports/ has the frozen sample/manual25, usage estimate and offline validation.
protocol/ includes exact config/prompts/schema plus active docs. THESIS_PROGRESS.md
is the current consolidated status. API keys/login files are excluded.
Historical D040 report is context only; D040 photos/targets are in the previously
created D040 comparison bundle, not this100-case preparation package.

## Request to paste into ChatGPT
Please read THESIS_PROGRESS.md, rubric.md, cases.json and protocol/active_spec.md.
Review the D041100-case sampling/input preparation and frozen D040 protocol.
No D041 teacher outputs have been generated: do NOT claim factual success/failure
or invent summaries/verdicts for the new pilot. Inspect actual original photos
and selected reviews if discussing particular inputs, and state IDs actually seen.
Discuss sampling coverage and explicit oversampling, the pre-generation random25
selection, source/evidence preservation, fixed prompts/settings, API-only content
responsibility, technical validation limits and readiness/cost prerequisites.
Distinguish existing BLIP caption limitations from errors in nonexistent D041 targets.
The previous D040 known regressions remain unresolved historical concerns; this
larger unseen pilot is an evaluation, not a claim that they have been fixed.
Return questions and suggestions in Vietnamese for researcher discussion only.
Do not alter sample IDs/reviews/captions, teacher prompts, StageF, K5, budgets or
existing human labels. ChatGPT is an optional second opinion, not a human annotator
or alternative target generator. Production teacher is still unfrozen.
'''

RUBRIC='''# D041 preparation discussion rubric

This is pre-generation input/method review. D041 teacher targets and visual verdicts
are absent. Do not score their factual accuracy, label cases as model successes,
or substitute a Codex/ChatGPT summary for a teacher API response.

Sampling: check exact train/validation membership and exclusion of all20prior StageG
IDs. Seed42 and sampling procedure are frozen before outputs. Review-length quartiles
and Filtered caption-count strata improve coverage; validation20%, all8remaining
zero-Filtered cases and minimum1/cell are deliberate oversampling. The25random
manual-review cases were independently preselected from100. Keep later flagged
cases separate; no sample replacement/cherry-picking after seeing results.

Inputs: preserve complete review IDs/text/order in all three conditions; original
photos and exact normalized BLIP captions; frozen StageF/K5/5000–4000–750 controls.
No captions/images are invented or reselected by this export. Simple compatible
BLIP captions are not automatically bad or inconsistent. Input caption errors or
ambiguity are items for the forthcoming fixed teacher verification protocol.

Future teacher review consensus: selected reviews only, supported praise/criticism,
accurate food categories and customer prevalence, meaningful qualified contrasts.
Future visual claims: supported by BOTH exact LED caption and mapped original photo;
no photo-only substitute details. Ordinary categories/qualitative size/simple useful
content remain acceptable; precise identities/ingredients/counts/actions/relations
require visible support. Teacher verified/rejected/uncertain are model judgments,
not human-final labels. Zero-Filtered retains review consensus and empty header.

The exact current prompts/config/schema, not this explanation, govern API generation.
No prompt/model/schema/retry optimization is authorized during this fixed pilot.
Schema/ID/quote/hash/token checks establish structural/source integrity, not factual
truth. Contradictions must be flagged, not reconciled.512target-token length remains
diagnostic rather than a silently invented cap. Failed cases must stay in reports.

The researcher remains primary human decision maker. Optional ChatGPT comparison
does not automatically become a human annotation. Browser notes are discussion
drafts only. Do not manufacture negatives, convert uncertainty, or retune test data.
No paid API call, account/key inspection, training or production approval occurs here.
'''


def jb(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def render(data):
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    page=(ASSETS/'index.html').read_text(encoding='utf-8')
    for key,value in [('__DATA__',encoded),('__STYLES__',(ASSETS/'styles.css').read_text(encoding='utf-8')),
        ('__APP__',(ASSETS/'app.js').read_text(encoding='utf-8'))]:page=page.replace(key,value)
    assert not any(x in page for x in ['__DATA__','__STYLES__','__APP__'])
    assert json.loads(page.split('<script id="review-data" type="application/json">')[1].split('</script>')[0])==data
    return page.encode('utf-8')


def export(rebuild=False):
    if ZIP.exists() or (VIEW/'dist/index.html').exists():
        if not rebuild:raise FileExistsError('Preserve export; choose a new version or verified derived-export rebuild')
        previous=read(FOLDER/'chatgpt_stage_g_d041_preparation_bundle_report.json')
        assert file_sha(ZIP)==previous['zip_sha256']
        assert file_sha(VIEW/'dist/index.html')==previous['viewer_sha256']
    cfg,pre,jobs=check_inputs()
    if (FOLDER/'execution_v1').exists():raise ValueError('This is a pre-API export; execution has begun, use a new result bundle instead')
    protected={}
    def original(path):
        b=path.read_bytes();protected[path.relative_to(ROOT).as_posix()]=digest(b);return b
    businesses=json.loads(original(FOLDER/'pilot_businesses.json'))
    witnesses=json.loads(original(FOLDER/'frozen_source_witness.json'))
    manual=json.loads(original(FOLDER/'manual_review_25_manifest.json'))
    manual_ids={b['business_id'] for b in manual['businesses']}
    assert len(manual_ids)==25
    cases=[];images={};source_index={}
    for source in witnesses:
        assert source.get('teacher_target') is False  # Existing Stage F input-only flag, preserved.
        source_index[source['business_id'],source['condition']]=source
    for business in businesses:
        bid=business['business_id'];these=[j for j in jobs if j['business_id']==bid]
        review=next(j for j in these if j['condition']=='shared')
        assert [r['review_id'] for r in review['evidence']['reviews']]==business['selected_review_ids']
        conditions={}
        for cond in ['unfiltered','filtered']:
            job=next(j for j in these if j['condition']==cond);caps=copy.deepcopy(job['evidence']['captions'])
            assert len(caps)==len(job['images'])
            for cap,photo in zip(caps,job['images']):
                assert cap['photo_id']==photo['photo_id'] and cap['photo_marker']==photo['photo_marker']
                data=original(ROOT/photo['relative_path']);assert digest(data)==photo['raw_sha256']
                name='images/'+cap['photo_id']+'.jpg'
                if name in images:assert images[name]==data
                images[name]=data;cap['image_file']=name;cap['original_image_sha256']=digest(data)
            conditions[cond]={'job_id':job['job_id'],'captions':caps,'teacher_assessments':None,'human_final_labels':None,
                'empty_caption_condition':not caps,'teacher_status':'NOT_GENERATED_NO_API_CALL'}
        sources={cond:copy.deepcopy(source_index[bid,cond]) for cond in ['reviews_only','unfiltered','filtered']}
        metadata={}
        for line in sources['reviews_only']['source'].split('[REVIEWS]')[0].splitlines():
            if '=' in line:
                key,value=line.split('=',1)
                if key in ['name','categories','city','state']:metadata[key]=value
        assert business['manual_review_preselected']==(bid in manual_ids)
        cases.append({'case_id':bid,'business_id':bid,'split':business['split'],'metadata':metadata,
            'manual_review_preselected':bid in manual_ids,'zero_filtered_edge_case':business['zero_filtered_edge_case'],
            'review_tokens':business['review_tokens'],'unfiltered_caption_count':business['unfiltered_caption_count'],
            'filtered_caption_count':business['filtered_caption_count'],'reviews':copy.deepcopy(review['evidence']['reviews']),
            'exact_review_block':review['evidence']['exact_review_block'],'caption_conditions':conditions,'sources':sources,
            'input_binding':copy.deepcopy(business),'teacher_targets':None,'human_verified':False,
            'status':'FROZEN_INPUT_ONLY_TEACHER_API_NOT_RUN'})
    assert len(cases)==100 and len(images)==659
    assert set(manual_ids)<={c['business_id'] for c in cases}
    sampling=read(FOLDER/'sampling_manifest.json');validation=read(FOLDER/'offline_validation_report.json');estimate=read(FOLDER/'pilot_usage_estimate.json')
    data={'schema_version':'stage_g_d041_preparation_portable_inputs_v1','export_only':True,
        'teacher_API_calls':0,'teacher_targets':0,'production_teacher_frozen':False,
        'businesses':cases,'images':{name[7:-4]:name for name in images},'sampling':sampling,'validation':validation,
        'estimate':{k:v for k,v in estimate.items() if k!='prior_observations'},
        'protocol':{'config':cfg,'review_prompt':(ROOT/cfg['prompts']['review_consensus']).read_text(encoding='utf-8'),
            'visual_prompt':(ROOT/cfg['prompts']['visual_evidence']).read_text(encoding='utf-8')}}
    files={'README.md':README.encode('utf-8'),'rubric.md':RUBRIC.encode('utf-8'),'cases.json':jb(data),'index.html':render(data)}
    for case in cases:files['businesses/'+case['business_id']+'.json']=jb(case)
    for name in ['sampling_manifest.json','pilot_businesses.json','manual_review_25_manifest.json','pilot_usage_estimate.json',
        'preflight.json','offline_validation_report.json','readiness_report.json','serializer_compatibility_report.json',
        'notebook_update_report.json','notebook_view_execution_report.json']:
        files['reports/'+name]=original(FOLDER/name)
    if (VIEW/'offline_runtime_check_report.json').exists():
        files['reports/offline_runtime_check_report.json']=original(VIEW/'offline_runtime_check_report.json')
    files['protocol/config.json']=original(CONFIG)
    for path in (FOLDER/'protocol_snapshot').rglob('*'):
        if path.is_file():files['protocol/frozen_D040/'+path.relative_to(FOLDER/'protocol_snapshot').as_posix()]=original(path)
    for name,path in {'THESIS_PROGRESS.md':ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md',
        'protocol/active_spec.md':ROOT/'specs/007_teacher_target_construction.md',
        'protocol/DECISIONS.md':ROOT/'DECISIONS.md','protocol/master_plan.md':ROOT/'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
        'historical_D040_context/comparison_report.json':ROOT/'outputs/stage_g/teacher_pilot_v6_focused_d040_v1/comparison_report.json'}.items():files[name]=original(path)
    files['discussion_response_template.json']=jb({'role':'pre_generation_input_and_protocol_discussion_not_human_final_annotation',
        'sample_and_method_questions':[],'business_notes':[{'business_id':c['business_id'],'photo_ids_actually_opened':[],
            'review_ids_actually_read':[],'input_observations':[],'human_verified':False} for c in cases]})
    for name,b in files.items():assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}',b),'Possible credential pattern in '+name
    checksums={name:digest(b) for name,b in {**files,**images}.items()}
    manifest={'status':'EXPORT_ONLY_D041_PREPARATION_NOT_TEACHER_RESULTS','businesses':100,
        'businesses_by_split':dict(Counter(c['split'] for c in cases)),'manual25_businesses':25,
        'original_images':659,'selected_reviews':sum(len(c['reviews']) for c in cases),
        'caption_occurrences':sum(len(co['captions']) for c in cases for co in c['caption_conditions'].values()),
        'zero_filtered_businesses':sum(c['zero_filtered_edge_case'] for c in cases),
        'images_resized':False,'API_calls':0,'new_teacher_targets':0,'new_human_labels':0,
        'held_out_businesses':0,'scientific_inputs_outputs_or_methodology_changed':False,'file_sha256':checksums}
    files['export_manifest.json']=jb(manifest)
    with zipfile.ZipFile(ZIP,'w' if rebuild else 'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,b in {**files,**images}.items():z.writestr(PREFIX+name,b)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for name,h in checksums.items():assert digest(z.read(PREFIX+name))==h,name
        restored=json.loads(z.read(PREFIX+'cases.json'));assert restored==data
        for case in restored['businesses']:
            review=next(j for j in jobs if j['business_id']==case['business_id'] and j['condition']=='shared')
            assert case['reviews']==review['evidence']['reviews'] and case['exact_review_block']==review['evidence']['exact_review_block']
            assert case['teacher_targets'] is None and case['human_verified'] is False
            for cond,record in case['caption_conditions'].items():
                job=next(j for j in jobs if j['business_id']==case['business_id'] and j['condition']==cond)
                assert [{k:v for k,v in cap.items() if k not in ['image_file','original_image_sha256']} for cap in record['captions']]==job['evidence']['captions']
                for cap in record['captions']:assert cap['image_file'].startswith('images/') and cap['image_file'] in images
    local=copy.deepcopy(data)
    local['images']={name[7:-4]:'data:image/jpeg;base64,'+base64.b64encode(b).decode('ascii') for name,b in images.items()}
    page=VIEW/'dist/index.html';page.parent.mkdir(parents=True,exist_ok=True);page.write_bytes(render(local))
    (VIEW/'OPEN_D041_PREPARATION.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0dist\\index.html"\r\n')
    for rel,h in protected.items():assert file_sha(ROOT/rel)==h,rel
    check_inputs()
    final=dict(manifest,integrity_checks='PASS',all_images_present=True,exact_ID_text_order_and_sources_preserved=True,
        protected_export_input_sha256=protected,zip_path=ZIP.relative_to(ROOT).as_posix(),zip_bytes=ZIP.stat().st_size,
        zip_sha256=file_sha(ZIP),viewer_path=page.relative_to(ROOT).as_posix(),viewer_bytes=page.stat().st_size,
        viewer_sha256=file_sha(page),browser_visual_QA=False)
    (FOLDER/'chatgpt_stage_g_d041_preparation_bundle_report.json').write_bytes(jb(final))
    (VIEW/'build_report.json').write_bytes(jb(final))
    print(json.dumps({k:final[k] for k in ['integrity_checks','businesses','original_images','selected_reviews','caption_occurrences','zip_path','zip_bytes','viewer_path','viewer_bytes']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rebuild-export-viewer-only',action='store_true')
    export(parser.parse_args().rebuild_export_viewer_only)
