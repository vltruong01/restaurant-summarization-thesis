"""Finalize docs and create an independently reopened/member-hashed review ZIP."""
from pathlib import Path
import ast
import csv
import difflib
import io
import json
import shutil
import tempfile
import zipfile
from s1000_final_eval_core_v1 import read,write,write_csv,file_hash,require
from build_stage_m_reports_v1 import ROOT,OUT,PREP,L,human_status

def main():
    archive=OUT/'STAGE_M_AUTOMATIC_EVALUATION_REVIEW.zip';require(not archive.exists(),'Refuse overwrite review archive')
    final=read(OUT/'FINAL_REGRESSION_TEST_RESULTS.json');require(final['status']=='PASS' and final['tests_run']==12 and final['test_source_sha256']==file_hash(ROOT/'scripts/test_stage_m_evaluation_v1.py'),'Final regression receipt drift')
    # Current actual results supersede the earlier eleven-test iteration.
    for name in ['STAGE_M_REPORT.md']:
        p=OUT/name;t=p.read_text(encoding='utf8').replace('11 synthetic regression tests passed','12 synthetic regression tests passed')
        t=t.replace('Teacher request serialization was checked without network;', 'The full200-request Teacher executor/importer passed on synthetic fixtures with a mocked transport; no real API call occurred. Raw-response tampering was rejected. Live provider access and vision capability remain untested. Teacher request serialization was checked without network;')
        p.write_text(t,encoding='utf8')
    # Source-only Teacher plan was frozen before inference; do not rewrite it.
    current=human_status();write(OUT/'FINAL_READ_ONLY_HUMAN_STATUS.json',current)
    require(current['approved']<100,'Human completion arrived; reassess primary scoring before closing')
    docs=['THESIS_PROGRESS.md','LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','CODEX_THESIS_IMPLEMENTATION_PLAN.md','CHATGPT_NEW_CONVERSATION_HANDOFF.md']
    updates=OUT/'DOCUMENTATION_UPDATES';updates.mkdir();doc_receipts=[]
    paragraph=f'''### Stage M feasible automatic evaluation complete; references pending — 2026-10-29

Current accepted configuration remains Source-Separated V1-S1000. Stage K inference is complete and frozen:100 TEST businesses,400 RO/U/F/S condition records and700 logical components. Stage M independently rechecked source/token/decoder identities and actual local RO/U/F adapter and backbone hashes. Original Stage K diagnostics were replayed byte-identically across all five output artifacts. No new LED inference, training, checkpoint changes or paid calls occurred.

The two reference tracks are implemented separately. Primary human-reference scoring remains PENDING: the read-only Stage L snapshot has {current['approved']}/100 approved and {current['stored_records']} stored records. Full100 genuine human-written/distinct-checker approval is required; no subset or GPT example substitutes. Human Evaluation is researcher-managed concurrently; its completion is not independently verified. Stage M did not edit references, accounts, drafts, TEST30 sampling, rubric or the source-only localhost application.

Supplementary AI Teacher scoring is BLOCKED: no compatible completed TEST100 Teacher references, dedicated credential or paid generation release. A prediction-blind source-only plan is prepared:200 requests (100 Review +100 Visual),2,817 selected reviews,1,136 original-byte photos /41,469,768 bytes, maximum1,638,400 requested output tokens. No defensible USD estimate exists without provider image/reasoning rates. Historical gpt-6.1-sol alias has no immutable snapshot attestation; no live vision capability has been checked. TEST Visual uses all original photos without captions, explicitly differing from historical condition-specific caption-supported TRAIN supervision. AI Teacher references remain supplementary and are never represented as human work.

Twelve synthetic tests passed, including full100 synthetic human approval/export and200 mocked Teacher request/import/tamper checks; no real human annotations or API calls were made. Actual frozen ROUGE/BERTScore CPU checks passed on synthetic text and one existing VALIDATION business/two components. Such technical numbers are not TEST evaluation. Official six-metric tables and700-row matrices are pending, not zero-valued results. Paired bootstrap seed20261028/B20000 and four prespecified BERT F1 sign-flip/Holm tests remain fixed; Review/Visual and Human/Teacher tracks are never merged.

Measured Visual trigram repetition is U0.122986 versus F/S0.060546; U-minus-F difference0.062440,95% CI[0.044258,0.082059]. F is shorter (51.68 vs67.74 words), but lower repetition is not factuality or coverage proof. F/S Review equality and S Visual donor permutation hold100/100. Two distinct frozen F Review generation flags (foreign marker;768-token repetitive output) and two F/S empty Visual cases remain in full100; labeled sensitivities are prepared, not silent exclusions. RO Visual remains N/A. Pipeline ancestry, caption-free Review inference, BERT512-token prefix and original-photo overlap limitations remain explicit.

Reports, ready-to-run commands and integrity-verified review package: outputs/stage_m/s1000_final_automatic_v1/. STOP pending complete approved human references and separately approved paid Teacher access. No final quality ranking or new experiment is implied. Older entries below are historical status snapshots.

'''
    for name in docs:
        p=ROOT/'thesis_sdd_docs'/name;before=p.read_bytes();shutil.copy2(p,updates/(name+'.before'))
        original=before.decode('utf-8-sig');p.write_text(paragraph+original,encoding='utf8')
        diff=''.join(difflib.unified_diff(original.splitlines(True),(paragraph+original).splitlines(True),fromfile=name+'.before',tofile=name))
        (updates/(name+'.diff')).write_text(diff,encoding='utf8')
        doc_receipts.append({'file':str(p.relative_to(ROOT)),'before_sha256':file_hash(updates/(name+'.before')),'after_sha256':file_hash(p),'edit':'Prepend current Stage M status, preserve historical content; no credentials'})
    write(updates/'CHANGE_RECEIPTS.json',doc_receipts)
    # Update implementation hashes after the final regression-only amendments.
    manifest=OUT/'INPUT_REPRODUCIBILITY_MANIFEST.csv';rows=list(csv.DictReader(manifest.open(encoding='utf8')))
    for row in rows:
        if row['role']=='IMPLEMENTATION':
            p=ROOT/row['path'];row['sha256']=file_hash(p);row['bytes']=p.stat().st_size
    for p,role in [(ROOT/'scripts/package_stage_m_review_v1.py','IMPLEMENTATION'),(PREP/'HUMAN_EVALUATION_PROTOCOL.md','PROTECTED_HUMAN_RUBRIC'),(PREP/'TEST100_INFERENCE_PLAN.json','FROZEN_PREPARATION')]:
        rows.append({'path':str(p.relative_to(ROOT)),'role':role,'bytes':p.stat().st_size,'sha256':file_hash(p)})
    manifest.unlink();write_csv(manifest,rows)
    proof=read(OUT/'FROZEN_PREDICTION_VERIFICATION.json')
    require(all(file_hash(PREP/n)==h for n,h in proof['preparation_bindings'].items()),'Frozen preparation changed during Stage M')
    require(file_hash(ROOT/'scripts/stage_l_reference_app_v1.py')=='997ffa021e7ed5cdffef10a16c130757d0fef1fc820a436dcd34301f097ef370','Human application changed')
    write(OUT/'FINAL_SCOPE_VERIFICATION.json',{'status':'PASS','frozen_preparation_hashes_unchanged':True,'stage_l_application_hash_unchanged':True,
        'all_reference_free_artifacts_identical_to_original_stage_k':all(x['byte_identical'] for x in read(OUT/'REFERENCE_FREE_REPLAY_VERIFICATION.json')),
        'primary_TEST_scores_computed':0,'supplementary_TEST_scores_computed':0,'no_actual_Teacher_requests':True,'no_new_LED_generation_or_training':True,
        'human_status_snapshot_only':current,'rubric_sha256':file_hash(PREP/'HUMAN_EVALUATION_PROTOCOL.md'),
        'frozen_inference_plan_sha256':file_hash(PREP/'TEST100_INFERENCE_PLAN.json'),'no_cloud_lifecycle_action':True})
    package=OUT/'REVIEW_PACKAGE';package.mkdir();copies={}
    def add(source,relative):
        source=Path(source);target=package/relative;require(not target.exists(),'Duplicate package path');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);copies[target.relative_to(package).as_posix()]=source
    # Output payload whitelist: no private DB, accounts, credentials, model binaries.
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and not p.is_relative_to(package) and 'DOCUMENTATION_UPDATES' not in p.parts:add(p,Path('REPORTS')/p.relative_to(OUT))
    for p in sorted(updates.glob('*.diff')):add(p,Path('DOC_UPDATES')/p.name)
    add(updates/'CHANGE_RECEIPTS.json','DOC_UPDATES/CHANGE_RECEIPTS.json')
    for name in docs:add(ROOT/'thesis_sdd_docs'/name,Path('thesis_sdd_docs')/name)
    scripts=['stage_m_evaluation_v1.py','stage_m_teacher_references_v1.py','test_stage_m_evaluation_v1.py','verify_stage_m_metric_runtime_v1.py','build_stage_m_reports_v1.py','package_stage_m_review_v1.py','s1000_final_eval_core_v1.py','score_s1000_automatic_prepared_v1.py','validate_s1000_human_references_v1.py','stage_k_reference_free_analysis_v1.py','stage_k_bert_loader_compat_v1.py','stage_g_teacher_credentials.py','stage_l_reference_app_v1.py','test_stage_l_reference_workflow_v1.py']
    for name in scripts:add(ROOT/'scripts'/name,Path('scripts')/name)
    for name in ['COHORT100_FREEZE.json','MODEL_AND_DECODER_LOCK.json','METRIC_LOCK.json','TEST100_INFERENCE_PLAN.json','TEST100_SHUFFLED_INPUT_MANIFEST.json','TEST100_SHUFFLED_MAPPING.csv','SHUFFLED_POLICY.json','AUTOMATIC_EVALUATION_PROTOCOL.md','HUMAN_EVALUATION_PROTOCOL.md','HUMAN_EVALUATION_COHORT30.json']:
        add(PREP/name,Path('FROZEN_PREPARATION')/name)
    package_source=PREP/'TEST100_REFERENCE_ANNOTATION_PACKAGE';add(package_source/'ANNOTATION_PACKAGE_MANIFEST.json','SOURCE_EVIDENCE/ANNOTATION_PACKAGE_MANIFEST.json')
    for r in read(package_source/'ANNOTATION_PACKAGE_MANIFEST.json')['frozen_sources']:add(package_source/r['source_file'],Path('SOURCE_EVIDENCE')/r['source_file'])
    history=ROOT/'configs/stage_g_teacher_production_d042.json';add(history,'configs/stage_g_teacher_production_d042.json')
    for p in read(history)['prompts'].values():add(ROOT/p,p)
    add(ROOT/'schemas/stage_g_teacher_pilot_v3.json','schemas/stage_g_teacher_pilot_v3.json')
    (package/'README.md').write_text('''# Stage M administrator review package

Start with REPORTS/README_VI.md and REPORTS/STAGE_M_REPORT.md. Reference-free tables are measured TEST100 diagnostics. HUMAN_PRIMARY and TEACHER_SUPPLEMENTARY contain explicitly pending tables with blank scores. TECHNICAL_RUNTIME_VERIFICATION uses synthetic/old VALIDATION text only, not TEST reference scores. The final12 regression tests use synthetic temporary workspaces and a mocked API transport.

This is NOT an annotator package. It contains model identities,400 frozen condition records and descriptive diagnostics. Keep it separate from blinded human reference authors/checkers and Human Evaluation raters. It contains no real human drafts/accounts/approvals or credentials. No AI reference text has been generated for TEST. Source-only Teacher preparation is not a reference dataset.

SOURCE_EVIDENCE preserves100 exact source bundles and original-photo identifiers/hashes; raw JPEGs and large scorer/backbone/adapter binaries are not duplicated in this compact review ZIP. They were rehashed locally and remain in the original project paths. FROZEN_PREPARATION preserves cohort, decoder/model/metric locks and Shuffled plan. Execution commands assume the original project workspace and pinned assets; this ZIP is review evidence, not a self-contained model runtime. Complete implementation, methodology, test sources and documentation diffs are included.

PACKAGE_MANIFEST.csv hashes every payload except itself. The external integrity receipt verifies ZIP CRC, reopening, safe extraction, all payload SHA256s and JSON/CSV/JSONL/Python parsing. No predictions or source evidence were rewritten. Primary references remain pending all100 human approvals; Teacher access/spending remains blocked. STOP before any new paid call, inference/training or final ranking.
''',encoding='utf8')
    payload=[p for p in sorted(package.rglob('*')) if p.is_file()]
    for p in payload:
        rel=p.relative_to(package).as_posix();require('PRIVATE_STATE' not in rel and p.suffix.lower() not in ['.sqlite3','.bin','.safetensors','.jpg'],'Forbidden private/large artifact in compact review ZIP')
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
    write_csv(package/'PACKAGE_MANIFEST.csv',[{'filename':p.relative_to(package).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in payload])
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_m_zip_verify_') as d,zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names))==len(payload)+1 and z.testzip() is None,'ZIP count/duplicate/CRC failure')
        require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),'Unsafe ZIP member')
        z.extractall(d);extracted=Path(d);entries=list(csv.DictReader((extracted/'PACKAGE_MANIFEST.csv').open(encoding='utf8')))
        require({x['filename'] for x in entries}==set(names)-{'PACKAGE_MANIFEST.csv'},'ZIP manifest membership differs')
        for r in entries:
            p=extracted/r['filename'];require(p.stat().st_size==int(r['bytes']) and file_hash(p)==r['sha256'],'Extracted payload SHA256 mismatch')
        for p in extracted.rglob('*'):
            if not p.is_file():continue
            if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8-sig'))
            elif p.suffix=='.jsonl':
                for line in p.read_text(encoding='utf8').splitlines():json.loads(line)
            elif p.suffix=='.csv':list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
            elif p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
    checksum=file_hash(archive);(OUT/(archive.name+'.sha256')).write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    write(OUT/'PACKAGE_INTEGRITY_REPORT.json',{'status':'PASS_ACTUALLY_REOPENED_EXTRACTED_MEMBER_SHA256_VERIFIED','archive':archive.name,'bytes':archive.stat().st_size,'sha256':checksum,'payload_files':len(payload),'ZIP_members':len(payload)+1,
        'CRC_checked':True,'safe_extraction_checked':True,'every_payload_SHA256_and_size_checked':True,'every_JSON_CSV_JSONL_Python_payload_parsed':True,
        'no_private_DB_credentials_weights_or_raw_photos_packaged':True,'TEST_reference_scores':'NOT_AVAILABLE_PENDING','final_regression_tests':final})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':checksum,'payload_files':len(payload),'status':'PASS'},indent=2))

if __name__=='__main__':main()
