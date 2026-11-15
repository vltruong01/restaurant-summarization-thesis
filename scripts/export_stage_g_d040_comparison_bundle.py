"""Export exact D-040 focused evidence plus D039 comparison and a standalone viewer; no API."""
from pathlib import Path
from collections import Counter
import base64
import copy
import hashlib
import json
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'outputs/stage_g/teacher_pilot_v6_focused_d040_v1'
EXE=PILOT/'execution_v1'
SOURCE_ZIP=ROOT/'outputs/stage_g/chatgpt_stage_g_d039_comparison_bundle.zip'
ZIP=ROOT/'outputs/stage_g/chatgpt_stage_g_d040_comparison_bundle.zip'
PREFIX='chatgpt_stage_g_d040_comparison_bundle/'
VIEW=ROOT/'outputs/stage_g/pilot_comparison_ui_d040_v1'
ASSETS=ROOT/'scripts/stage_g_d040_comparison_ui'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def jb(d):return (json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


README="""# Stage G D-040: portable independent second review

Export only: all 8 predeclared focused businesses (5 train/3 validation), 24 targets
per version D039 and D040, 45 unchanged original JPEGs and complete selected reviews.
This is a bounded diagnostic sample, not a population error-rate evaluation.
Extract the WHOLE ZIP and open index.html. Local viewer needs no server or API.
Browser notes are discussion drafts only, not automatically confirmed annotations.

## Request to paste into ChatGPT
Please inspect rubric.md and cases.json, actual original images and full selected
reviews. Independently compare D039 versus D040 across all 8 listed businesses.
State the IDs of photos/reviews actually inspected. If unavailable say NOT REVIEWED.
First form your own observations, then consult reports/comparison_report.json.
Report exact business_id, condition, claim, REV / IMG_CAP / PHOTO and original photo_id.
Separate improvements, factual errors, unsupported verified claims, ambiguity,
semantic verdict contradictions, lost review contrasts and excessive strictness.
Correct but simple content and ordinary reasonable descriptions are acceptable.
Inspect rejected/uncertain judgments too; do not manufacture negatives.
Check asparagus/count improvements, fresh UC food-category consensus, remaining
chicken/green-bean and water concerns, pizza/cutlery regressions, and zero captions.
Seven reviews are reused unchanged: only UC is a fresh reviewv7 result. Visualv4 is
new for nonempty conditions. A lower literal contradiction count is not evidence
of better semantic consistency because claim decomposition changed substantially.
Structural PASS is not factual certification and teacher verdicts are not human labels.
Return evidence-grounded Vietnamese findings for researcher discussion only.
Do not edit targets or existing user-confirmed labels, freeze a prompt, retune Stage F,
call the API, or use held-out data. ChatGPT is an optional comparison aid after the
user's own inspection; final annotation remains a user-confirmed human decision.
No production teacher protocol is approved. API credentials are not included.
"""

RUBRIC="""# Active Stage G D-040 evidence/rubric

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
- Seven D040 reviews are unchanged D039 imports retaining actual v5/v6 provenance; only UC tests reviewv7. Source/model/settings/
  schema/validators/Stage F remain fixed. All versions and failed history preserved.
- All8focused cases included; no post-result cherry picking. This diagnostic cohort
  does not measure a population error rate or isolate stochastic variation from prompt effects.
- AI suggestions, optional ChatGPT comparison and human final decisions remain separate.
  ChatGPT comparison is an optional aid after user inspection. No automatic human labels.
- Original one-claim straw confirmation is copied unchanged and scoped ONLY to that
  original claim; no human verification is propagated to D040 or entire businesses.
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
    final_audit=json.loads(original(PILOT/'final_preservation_report.json'))
    for rel,h in final_audit['scientific_output_sha256'].items():assert sha(original(PILOT/rel))==h
    for rel,h in final_audit['protocol_snapshot_sha256'].items():
        assert sha(original(PILOT/'final_protocol_snapshot'/rel))==h
    cfg=json.loads(original(ROOT/'configs/stage_g_teacher_pilot_v6_focused_d040.json'))
    jobs=json.loads(original(PILOT/'pilot_jobs.json'));cohort=json.loads(original(PILOT/'pilot_businesses.json'))
    report=json.loads(original(PILOT/'comparison_report.json'));wanted=[b['business_id'] for b in cohort]
    images={};cases=[]
    with zipfile.ZipFile(SOURCE_ZIP) as z:
        prefix='chatgpt_stage_g_d039_comparison_bundle/'
        manifest=json.loads(z.read(prefix+'export_manifest.json'))
        for name,h in manifest['file_sha256'].items():assert sha(z.read(prefix+name))==h
        old=json.loads(z.read(prefix+'cases.json'));lookup={c['business_id']:c for c in old['businesses']}
        confirmations=z.read(prefix+'provenance/original_human_claim_verifications.json')
        old_protocol={n[len(prefix):]:z.read(n) for n in z.namelist() if n.startswith(prefix+'protocol/D039/')}
        for bid in wanted:
            before=copy.deepcopy(lookup[bid]);target=json.loads(original(EXE/'provisional_targets'/(bid+'.json')))
            these=[j for j in jobs if j['business_id']==bid];review_job=next(j for j in these if j['task']=='review_consensus')
            assert review_job['evidence']['reviews']==[{k:v for k,v in r.items() if k!='evidence_spans'} for r in before['reviews']]
            cp=json.loads(original(EXE/'checkpoints'/(review_job['job_id']+'.json')))
            conditions={}
            for cond in ['unfiltered','filtered']:
                job=next(j for j in these if j['condition']==cond)
                condition_cp=json.loads(original(EXE/'checkpoints'/(job['job_id']+'.json')))
                caps=copy.deepcopy(before['versions']['D039']['caption_conditions'][cond]['captions'])
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
            versions={'D039':before['versions']['D039']}
            versions['D040']={'complete_target_set_accepted':True,'targets':target['targets'],
                'review_consensus_claims':cp['resolved_output']['claims'],'review_teacher_raw_output':cp['output'],
                'review_source_resolution':cp['evidence_resolution'],'review_citation_metadata_diagnostics':cp['citation_metadata_diagnostics'],
                'caption_conditions':conditions,'review_component_source':target['review_component_source'],
                'review_import_lineage':cp.get('import_lineage'),'whole_business_human_verified':False}
            cases.append({'case_id':bid,'business_id':bid,'split':before['split'],'reviews':before['reviews'],
                'zero_filtered_edge_case':before['zero_filtered_edge_case'],'whole_business_human_verified':False,
                'existing_original_human_confirmations':before['existing_original_human_confirmations'],
                'versions':versions,'ai_second_opinion_findings':[f for f in report['findings'] if f.get('business_id')==bid or bid in f.get('business_ids',[])]})
        metadata={bid:old['metadata'][bid] for bid in wanted}
    assert len(cases)==8 and len(images)==45
    counts={v:sum(len(c['versions'][v]['targets'] or {}) for c in cases) for v in ['D039','D040']}
    assert counts=={'D039':24,'D040':24}
    data={'schema_version':'stage_g_d040_portable_two_version_review_v1','export_only':True,
        'production_method_frozen':False,'review_role':'optional_AI_second_opinion_not_human_ground_truth',
        'completion':read(EXE/'completion.json'),'versions':['D039','D040'],
        'businesses':cases,'metadata':metadata,'images':{n[7:-4]:n for n in images},
        'existing_human_confirmation_scope':'original_claim_only_not_propagated'}
    files={'README.md':README.encode('utf-8'),'rubric.md':RUBRIC.encode('utf-8'),'cases.json':jb(data),
           'index.html':render(data),'provenance/original_human_claim_verifications.json':confirmations}
    for name,b in old_protocol.items():
        # Copy D039 exact historical protocol unchanged.
        files[name]=b
    for c in cases:files['businesses/'+c['business_id']+'.json']=jb(c)
    copies={'protocol/D040/config.json':ROOT/'configs/stage_g_teacher_pilot_v6_focused_d040.json',
        'protocol/D040/review_v7_prompt.txt':ROOT/cfg['prompts']['review_consensus'],
        'protocol/D040/visual_v4_prompt.txt':ROOT/cfg['prompts']['visual_evidence'],
        'protocol/D040/imported_review_v5_prompt.txt':ROOT/'prompts/stage_g_review_consensus_v5.txt',
        'protocol/D040/schema.json':ROOT/cfg['schema'],
        'protocol/D040/execution_plan.json':PILOT/'execution_plan.json',
        'protocol/active_stage_g_spec.md':ROOT/'specs/007_teacher_target_construction.md',
        'protocol/DECISIONS.md':ROOT/'DECISIONS.md',
        'provenance/THESIS_PROGRESS.md':ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md',
        'provenance/independent_integrity_report.json':PILOT/'independent_integrity_report.json',
        'provenance/completion.json':EXE/'completion.json',
        'provenance/final_preservation_report.json':PILOT/'final_preservation_report.json',
        'protocol/D040/imported_review_v6_prompt.txt':ROOT/'prompts/stage_g_review_consensus_v6.txt',
        'protocol/D040/preflight.json':PILOT/'preflight.json',
        'protocol/D040/pilot_businesses.json':PILOT/'pilot_businesses.json'}
    for name in ['comparison_report.json','machine_comparison.json','ai_photo_inspection_notes.json']:
        copies['reports/'+name]=PILOT/name
    runtime_report=VIEW/'offline_runtime_check_report.json'
    if runtime_report.exists():copies['reports/offline_runtime_check_report.json']=runtime_report
    for name,p in copies.items():files[name]=original(p)
    files['protocol/current_implementation_plan.md']=original(ROOT/'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md')
    for p in (PILOT/'final_protocol_snapshot').rglob('*'):
        if p.is_file():files['provenance/historical_D040_protocol_snapshot/'+p.relative_to(PILOT/'final_protocol_snapshot').as_posix()]=original(p)
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
    manifest={'status':'EXPORT_ONLY_D040_TWO_VERSION_COMPARISON','businesses':8,
        'businesses_by_split':dict(Counter(c['split'] for c in cases)),'target_texts_by_version':counts,
        'original_images':45,'selected_reviews':sum(len(c['reviews']) for c in cases),
        'AI_findings':len(report['findings']),'unchanged_D039_review_components':7,'fresh_v7_review_components':1,
        'D039_blocked_businesses':0,'D040_blocked_businesses':0,'file_sha256':checksums,
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
            assert c['versions']['D040']['targets']==current['targets']
            assert c['existing_original_human_confirmations']==lookup[c['business_id']]['existing_original_human_confirmations']
        assert z.read(PREFIX+'provenance/original_human_claim_verifications.json')==confirmations
    local=copy.deepcopy(data);local['images']={n[7:-4]:'data:image/jpeg;base64,'+base64.b64encode(b).decode('ascii') for n,b in images.items()}
    page=VIEW/'dist/index.html';page.parent.mkdir(parents=True,exist_ok=True);page.write_bytes(render(local))
    (VIEW/'OPEN_STAGE_G_COMPARISON.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0dist\\index.html"\r\n')
    (VIEW/'HOW_TO_OPEN.txt').write_text('Open OPEN_STAGE_G_COMPARISON.cmd or dist/index.html. Standalone/offline. Default D039 vs D040. Notes are personal discussion drafts, not final scientific labels.\n',encoding='utf-8')
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
