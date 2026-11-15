"""Export exact D-039 focused evidence plus v3/D038 comparisons and a standalone viewer; no API."""
from pathlib import Path
from collections import Counter
import base64
import copy
import hashlib
import json
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'outputs/stage_g/teacher_pilot_v5_focused_d039_v1'
EXE=PILOT/'execution_v1'
SOURCE_ZIP=ROOT/'outputs/stage_g/chatgpt_stage_g_d038_comparison_bundle.zip'
ZIP=ROOT/'outputs/stage_g/chatgpt_stage_g_d039_comparison_bundle.zip'
PREFIX='chatgpt_stage_g_d039_comparison_bundle/'
VIEW=ROOT/'outputs/stage_g/pilot_comparison_ui_d039_v1'
ASSETS=ROOT/'scripts/stage_g_d039_comparison_ui'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def jb(d):return (json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


README="""# Stage G D-039: portable independent ChatGPT comparison

Export-only. All9PREDECLARED focused businesses (5train/4validation), not a full20
rerun and not a representative population sample. Same reviews/captions/photos across
three versions: originalv3 (27targets), D038 (24accepted targets;1blocked business),
D039 (27targets;9/9complete). Structural PASS does NOT establish factual correctness.
Seven D039 review components are unchanged D038/v5 imports; only PR and Ez test reviewv6.

Files: cases.json; businesses/<business_id>.json; images/<photo_id>.jpg (50original
unchanged JPEGs); rubric.md; reports/; exact prompts/schema/configs under protocol/.
Extract the WHOLE ZIP and open index.html for a portable offline comparison viewer.
The page defaults to D038 versus D039; choose originalv3 with the version selectors.
Notes are browser-local discussion drafts, not automatically imported annotations.

## Request to paste into ChatGPT

Please independently inspect this D039 StageG teacher-pilot bundle. Read rubric.md
and cases.json; open the actual mapped images and complete selected reviews.
Compare originalv3/D038/D039 on all9listed businesses, in batches if needed.
First form your own evidence-based observations, then compare the existing AI
findings in reports/comparison_report.json. State exact business/photo/review IDs
actually inspected. If you cannot extract or view an image, say NOT REVIEWED rather
than inferring correctness from the caption or model verdict.
Separate factual errors, unsupported verified claims, genuine uncertainty,
review category/prevalence errors and lost contrasts, verdict contradictions,
ordinary acceptable/simple descriptions, and overly strict interpretations.
Cite exact claim, business_id, condition, REV/source-span or IMG_CAP/PHOTO/photo_id,
and relevant source text for every concern. Review rejected/uncertain claims too.
Check whether hotdog/boat/plantain-category fixes preserve pizza/large-hamburger and
simple descriptions; examine remaining asparagus/green-bean/count concerns and both
literal verdict conflicts without forcing desired labels or treating all ambiguity as error.
Do not treat the seven imported review components as new reviewv6 results.
The D038 failed raw parsed response is diagnostic-only, never an accepted target;
D039 fixes quote fidelity through a new response, not manual editing.
Return improvements, remaining problems, regressions and suggested next decisions
in Vietnamese FOR RESEARCHER DISCUSSION ONLY. Do not rewrite targets, change human
labels, K, thresholds, budgets or call a teacher. AI is only a second opinion; user
remains primary human decision maker. All whole-business targets are human-unverified.
No production teacher protocol is approved and no held-out test data are included.
"""

RUBRIC="""# Active Stage G D-039 evidence/rubric

Exact prompts/spec are included. This export creates no new methodology or labels.

## Review consensus
- Selected reviews only; no photos/captions, outside knowledge or aggregate ratings.
- Preserve supported positive/negative/mixed views without manufacturing criticism.
- Avoid inflated counts/customer prevalence; valid IDs do not prove entailment.
- Keep food items/categories distinct unless review text supports their membership.
- Preserve meaningful qualified contrasts; inspect all selected reviews for omissions.
- At least two distinct source reviews support each recurring claim. Minority clauses
  may appear in grounded mixed claims without being represented as recurring complaints.
- Source span IDs resolve original quotes/offsets; claim text/polarity are never edited.
- Same shared consensus in all three conditions of each accepted version.

## Visual evidence
- Every visual claim must be recoverable from EXACT LED caption and supported by its
  mapped original photo. No photo-only replacement details, invented recipes/actions/counts.
- Ordinary compatible categories/qualitative size/simple descriptions are acceptable;
  no ruler or maximally precise taxonomy required. Generic is not automatically false.
- Specific identities need visible support, not merely color/outline/context/caption nouns.
  Do not propagate an unestablished noun into other verified relational clauses.
- verified: caption-supported and reasonably photo-supported; rejected: contradicted;
  uncertain: genuinely ambiguous/hidden/insufficient. Teacher verified is NOT human verified.
- Copy caption_quote literally, preserving punctuation; the full original caption is
  allowed. No fuzzy quote repair, manual target rewrite or failed-response promotion.
- No taste/value/service/popularity or business-wide prevalence inferred from photos.
- Zero Filtered captions preserve the empty visual header and review consensus.

## Comparison/provenance
- Identical photo/hash/caption/claim should normally agree across conditions; flag
  contradictions, do not reconcile them. Different claims/captions may legitimately differ.
- Paraphrases/conjunctions and different claim granularity need separate inspection.
- Seven D039 reviews reused v5 unchanged; only PR/Ez test reviewv6. Source/model/settings/
  schema/validators/Stage F remain fixed. All versions and failed history preserved.
- All9focused cases included; no post-result cherry picking. This diagnostic cohort
  does not measure a population error rate or isolate stochastic variation from prompt effects.
- AI suggestions, optional ChatGPT comparison and human final decisions remain separate.
  ChatGPT comparison is an optional aid after user inspection. No automatic human labels.
- Original one-claim straw confirmation is copied unchanged and scoped ONLY to that
  original claim; no human verification is propagated to D039 or entire businesses.
- No manufactured negatives, uncertain-to-negative conversion, test tuning or calls.
- Source5000/review4000/caption750/K5 unchanged;512target tokens is diagnostic, not frozen cap.
"""


def render(data):
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    page=(ASSETS/'index.html').read_text(encoding='utf-8')
    for key,value in [('__DATA__',encoded),('__STYLES__',(ASSETS/'styles.css').read_text(encoding='utf-8')),
                      ('__APP__',(ASSETS/'app.js').read_text(encoding='utf-8'))]:page=page.replace(key,value)
    assert not any(x in page for x in ['__DATA__','__STYLES__','__APP__'])
    assert json.loads(page.split('<script id="review-data" type="application/json">')[1].split('</script>')[0])==data
    return page.encode('utf-8')


def main(rebuild=False):
    if ZIP.exists() or (VIEW/'dist/index.html').exists():
        if not rebuild:raise FileExistsError('Preserve existing export; choose a new version')
        prev=read(ZIP.with_name(ZIP.stem+'_report.json'))
        assert sha(ZIP.read_bytes())==prev['zip_sha256']
        assert sha((VIEW/'dist/index.html').read_bytes())==prev['viewer_sha256']
    protected={}
    def original(p):
        b=p.read_bytes();protected[p.relative_to(ROOT).as_posix()]=sha(b);return b
    original(SOURCE_ZIP)
    audit=json.loads(original(PILOT/'independent_integrity_report.json'))
    for rel,h in audit['artifact_sha256'].items():assert sha(original(ROOT/rel))==h
    for rel,h in read(PILOT/'preflight.json')['preserved_original_sha256'].items():assert sha(original(ROOT/rel))==h
    cfg=json.loads(original(ROOT/'configs/stage_g_teacher_pilot_v5_focused_d039.json'))
    jobs=json.loads(original(PILOT/'pilot_jobs.json'));cohort=json.loads(original(PILOT/'pilot_businesses.json'))
    report=json.loads(original(PILOT/'comparison_report.json'));wanted=[b['business_id'] for b in cohort]
    images={};cases=[]
    with zipfile.ZipFile(SOURCE_ZIP) as z:
        prefix='chatgpt_stage_g_d038_comparison_bundle/'
        manifest=json.loads(z.read(prefix+'export_manifest.json'))
        for name,h in manifest['file_sha256'].items():assert sha(z.read(prefix+name))==h
        old=json.loads(z.read(prefix+'cases.json'));lookup={c['business_id']:c for c in old['businesses']}
        confirmations=z.read(prefix+'provenance/original_human_claim_verifications.json')
        old_protocol={n[len(prefix):]:z.read(n) for n in z.namelist() if n.startswith(prefix+'protocol/original/') or n.startswith(prefix+'protocol/revised/')}
        for bid in wanted:
            before=copy.deepcopy(lookup[bid]);target=json.loads(original(EXE/'provisional_targets'/(bid+'.json')))
            these=[j for j in jobs if j['business_id']==bid];review_job=next(j for j in these if j['task']=='review_consensus')
            assert review_job['evidence']['reviews']==[{k:v for k,v in r.items() if k!='evidence_spans'} for r in before['reviews']]
            cp=json.loads(original(EXE/'checkpoints'/(review_job['job_id']+'.json')))
            conditions={}
            for cond in ['unfiltered','filtered']:
                job=next(j for j in these if j['condition']==cond)
                condition_cp=json.loads(original(EXE/'checkpoints'/(job['job_id']+'.json')))
                caps=copy.deepcopy(before['versions']['original']['caption_conditions'][cond]['captions'])
                assert job['evidence']['captions']==[{k:v for k,v in c.items() if k not in ['image_file','original_image_sha256']} for c in caps]
                for c,image in zip(caps,job['images']):
                    b=z.read(prefix+c['image_file']);assert sha(b)==c['original_image_sha256']==image['raw_sha256']
                    assert sha(original(ROOT/image['relative_path']))==image['raw_sha256']
                    images[c['image_file']]=b
                conditions[cond]={'job_id':job['job_id'],'captions':caps,'task_status':condition_cp['status'],
                    'accepted_component':True,'teacher_assessments':condition_cp['output']['assessments'],
                    'failed_raw_output_diagnostic_only':None,'failure_reason':None,
                    'serialized_visual_claims_with_all_evidence':target['visual_target_claims_with_all_evidence'][cond],
                    'empty_caption_condition':not caps,'whole_business_human_verified':False}
            versions={'original_v3':before['versions']['original'],'D038':before['versions']['revised']}
            versions['original_v3']['complete_target_set_accepted']=True
            versions['D039']={'complete_target_set_accepted':True,'targets':target['targets'],
                'review_consensus_claims':cp['resolved_output']['claims'],'review_teacher_raw_output':cp['output'],
                'review_source_resolution':cp['evidence_resolution'],'review_citation_metadata_diagnostics':cp['citation_metadata_diagnostics'],
                'caption_conditions':conditions,'review_component_source':target['review_component_source'],
                'review_import_lineage':cp.get('import_lineage'),'whole_business_human_verified':False}
            cases.append({'case_id':bid,'business_id':bid,'split':before['split'],'reviews':before['reviews'],
                'zero_filtered_edge_case':before['zero_filtered_edge_case'],'whole_business_human_verified':False,
                'existing_original_human_confirmations':before['existing_original_human_confirmations'],
                'versions':versions,'ai_second_opinion_findings':[f for f in report['findings'] if f['business_id']==bid]})
        metadata={bid:old['metadata'][bid] for bid in wanted}
    assert len(cases)==9 and len(images)==50
    counts={v:sum(len(c['versions'][v]['targets'] or {}) for c in cases) for v in ['original_v3','D038','D039']}
    assert counts=={'original_v3':27,'D038':24,'D039':27}
    data={'schema_version':'stage_g_d039_portable_three_version_review_v1','export_only':True,
        'production_method_frozen':False,'review_role':'optional_AI_second_opinion_not_human_ground_truth',
        'completion':read(EXE/'completion.json'),'versions':['original_v3','D038','D039'],
        'businesses':cases,'metadata':metadata,'images':{n[7:-4]:n for n in images},
        'existing_human_confirmation_scope':'original_claim_only_not_propagated'}
    files={'README.md':README.encode('utf-8'),'rubric.md':RUBRIC.encode('utf-8'),'cases.json':jb(data),
           'index.html':render(data),'provenance/original_human_claim_verifications.json':confirmations}
    for name,b in old_protocol.items():
        name=name.replace('protocol/original/','protocol/original_v3/').replace('protocol/revised/','protocol/D038/')
        files[name]=b
    for c in cases:files['businesses/'+c['business_id']+'.json']=jb(c)
    copies={'protocol/D039/config.json':ROOT/'configs/stage_g_teacher_pilot_v5_focused_d039.json',
        'protocol/D039/review_v6_prompt.txt':ROOT/cfg['prompts']['review_consensus'],
        'protocol/D039/visual_v3_prompt.txt':ROOT/cfg['prompts']['visual_evidence'],
        'protocol/D039/imported_review_v5_prompt.txt':ROOT/'prompts/stage_g_review_consensus_v5.txt',
        'protocol/D039/schema.json':ROOT/cfg['schema'],
        'protocol/D039/execution_plan.json':PILOT/'execution_plan.json',
        'protocol/active_stage_g_spec.md':ROOT/'specs/007_teacher_target_construction.md',
        'protocol/DECISIONS.md':ROOT/'DECISIONS.md',
        'provenance/THESIS_PROGRESS.md':ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md',
        'provenance/independent_integrity_report.json':PILOT/'independent_integrity_report.json',
        'provenance/completion.json':EXE/'completion.json',
        'provenance/final_preservation_and_protocol_delta_report.json':PILOT/'final_preservation_and_protocol_delta_report.json'}
    for name in ['comparison_report.json','machine_comparison.json','ai_photo_inspection_notes.json']:
        copies['reports/'+name]=PILOT/name
    for name,p in copies.items():files[name]=original(p)
    md=original(PILOT/'comparison_report.md').decode('utf-8')
    files['reports/comparison_report.md']=md.replace('../../../data/processed/yelp_2300_v1/photos/','../images/').encode('utf-8')
    files['review_response_template.json']=jb({'role':'AI_second_opinion_for_discussion_only',
        'business_reviews':[{'business_id':c['business_id'],'actually_inspected':False,'photo_ids_opened':[],
            'review_ids_read':[],'improvements':[],'regressions':[],'uncertain_items':[],'correct_but_simple':[],
            'not_reviewed':[],'human_final_decision':None,'human_verified':False} for c in cases]})
    for name,b in files.items():assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}',b),'Possible credential in '+name
    for c in cases:
        for v in c['versions'].values():
            for cond in v['caption_conditions'].values():
                for cap in cond['captions']:assert cap['image_file'].startswith('images/') and cap['image_file'] in images
    checksums={n:sha(b) for n,b in {**files,**images}.items()}
    manifest={'status':'EXPORT_ONLY_D039_THREE_VERSION_COMPARISON','businesses':9,
        'businesses_by_split':dict(Counter(c['split'] for c in cases)),'target_texts_by_version':counts,
        'original_images':50,'selected_reviews':sum(len(c['reviews']) for c in cases),
        'AI_findings':len(report['findings']),'reused_v5_review_components':7,'fresh_v6_review_components':2,
        'D038_blocked_businesses':1,'D039_blocked_businesses':0,'file_sha256':checksums,
        'images_resized':False,'human_labels_or_teacher_outputs_changed':False,'API_calls':0,'held_out_test_businesses':0}
    files['export_manifest.json']=jb(manifest)
    with zipfile.ZipFile(ZIP,'w' if rebuild else 'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in {**files,**images}.items():z.writestr(PREFIX+n,b)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for n,h in checksums.items():assert sha(z.read(PREFIX+n))==h,n
        restored=json.loads(z.read(PREFIX+'cases.json'));assert restored==data
        for c in restored['businesses']:
            current=read(EXE/'provisional_targets'/(c['business_id']+'.json'))
            assert c['versions']['D039']['targets']==current['targets']
            assert c['existing_original_human_confirmations']==lookup[c['business_id']]['existing_original_human_confirmations']
        assert z.read(PREFIX+'provenance/original_human_claim_verifications.json')==confirmations
    local=copy.deepcopy(data);local['images']={n[7:-4]:'data:image/jpeg;base64,'+base64.b64encode(b).decode('ascii') for n,b in images.items()}
    page=VIEW/'dist/index.html';page.parent.mkdir(parents=True,exist_ok=True);page.write_bytes(render(local))
    (VIEW/'OPEN_STAGE_G_COMPARISON.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0dist\\index.html"\r\n')
    (VIEW/'HOW_TO_OPEN.txt').write_text('Open OPEN_STAGE_G_COMPARISON.cmd or dist/index.html. Standalone/offline. Default D038 vs D039; version selectors include v3. Notes are personal discussion drafts, not final scientific labels.\n',encoding='utf-8')
    for rel,h in protected.items():assert sha((ROOT/rel).read_bytes())==h,rel
    final=dict(manifest,integrity_checks='PASS',all_case_images_present=True,exact_target_caption_review_ID_text_order_preserved=True,
        original_human_confirmations_unchanged=True,protected_input_files=len(protected),
        protected_input_sha256=protected,zip_path=ZIP.relative_to(ROOT).as_posix(),zip_bytes=ZIP.stat().st_size,
        zip_sha256=sha(ZIP.read_bytes()),viewer_path=page.relative_to(ROOT).as_posix(),viewer_bytes=page.stat().st_size,
        viewer_sha256=sha(page.read_bytes()),browser_visual_QA=False)
    ZIP.with_name(ZIP.stem+'_report.json').write_bytes(jb(final));(VIEW/'build_report.json').write_bytes(jb(final))
    print(json.dumps({k:final[k] for k in ['integrity_checks','businesses','target_texts_by_version','original_images',
          'selected_reviews','AI_findings','zip_path','zip_bytes','viewer_path','protected_input_files']},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rebuild-export-viewer-only',action='store_true')
    main(p.parse_args().rebuild_export_viewer_only)
