"""Integrity verification and portable packaging only; zero production scoring."""
from pathlib import Path
import json,hashlib,shutil,zipfile,subprocess,tempfile,collections,datetime,re,sys,concurrent.futures
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1';P=D/'PACKAGE';APP=P/'EVALUATOR_APP'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def members(root,exclude=()):
    paths=[f for f in sorted(root.rglob('*')) if f.is_file() and f.name not in exclude]
    def row(f):return dict(file=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=sha(f))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(row,paths))
def check_prior(rows):
    def check(row):
        f=ROOT/row['file'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],row['file']
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,rows))
def archive(root,path):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(root))
def unpack(path,dest,manifest):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
        z.extractall(dest)
    for r in manifest:
        f=dest/r['file'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],r['file']
def main():
    assert read(D/'VALIDATION/SOFTWARE_TEST_RESULTS.json')['status']=='PASS' and read(D/'VALIDATION/BROWSER_E2E_RESULTS.json')['status']=='PASS'
    assert 'Ran 14 tests' in (D/'VALIDATION/BACKEND_AND_ANALYSIS_TESTS.stderr.log').read_text()
    w=APP/'WORKSPACE';cfg=read(w/'CONFIG.json');key=read(P/'COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json');items={r['alias']:r for r in read(w/'BLINDED_ITEMS.json')};frozen={(r['business_id'],r['system']):r for r in read(P/'COORDINATOR_ONLY/FROZEN_OUTPUTS_45.json')};assert len(key)==90 and len(items)==90 and len(frozen)==45
    balances={};textchecks=[]
    for rater in ['A01','A02']:
        rows=[r for r in key if r['annotator']==rater];counts=collections.Counter((r['system'],r['within_restaurant_position']) for r in rows);assert len(counts)==9 and set(counts.values())=={5};assert len(rows)==45
        balances[rater]=[dict(system=s,position=p,count=n) for (s,p),n in sorted(counts.items())]
    for r in key:
        item=items[r['alias']];f=frozen[r['business_id'],r['system']];assert item['review_text']==f['review_text'] and item['visual_text']==f['visual_text'];assert item['review_text_sha256']==f['review_text_sha256'] and item['visual_text_sha256']==f['visual_text_sha256'];textchecks.append(dict(alias=r['alias'],system=r['system'],review_sha256=item['review_text_sha256'],visual_sha256=item['visual_text_sha256'],status='PASS'))
    assert all(frozen[b,'Qwen-Caption']['review_text']==frozen[b,'Qwen-Image']['review_text'] for b in cfg['sources'])
    assert sha(w/'RUBRIC.json')==sha(ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP/WORKSPACE/RUBRIC.json')
    for f in [APP/'STATE',w/'PRIVATE_STATE',w/'COLLECTION_OPEN.json']:assert not f.exists(),f
    assert not [f for f in APP.rglob('*') if f.is_file() and f.suffix in ['.sqlite3','.db','.safetensors','.pt','.bin']]
    # Source files and all photo bytes remain exact copies of the old common source evidence.
    original=ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP/WORKSPACE';photo_checks=0
    for b,v in cfg['sources'].items():
        s=read(w/v['file']);assert (w/v['file']).read_bytes()==(original/v['file']).read_bytes()
        for ph in s['photos']:
            assert sha(w/'PHOTOS'/(ph['photo_id']+'.jpg'))==ph['sha256'];photo_checks+=1
    assert photo_checks==5143
    prior=read(D/'PROTECTED_PRIOR_ARTIFACTS.json')
    check_prior(prior)
    # Explicit model-family words must not appear in neutral application code or instructions.
    for f in APP.rglob('*'):
        if f.is_file() and f.suffix in ['.py','.html','.js','.md','.txt','.json'] and 'PYTHON' not in f.parts:
            assert not re.search(r'(?i:\bQwen\b)|\bLED\b|\bS1000\b',f.read_text(encoding='utf-8-sig')),'Identity term in blinded bundle: '+str(f)
    for name in ['stage_p_human_evaluation_v1.py','launch_stage_p_human_v1.py','human_http_common_v1.py','human_app_core_v1.py']:assert sha(APP/'APP'/name)==sha(ROOT/'scripts'/name)
    put(D/'VALIDATION/FROZEN_DATA_INTEGRITY.json',dict(status='PASS',new_study_id=cfg['study_id'],new_assignments=90,new_outputs=45,restaurants=15,original_photo_file_checks=photo_checks,source_reviews=247,original_LED_not_benchmark=True,Qwen_identical_Review_pairs=15,old_scores_reused=0,prior_files_rehashed_unchanged=len(prior),production_STATE_or_scores_created=False,collection_open=False,rubric_exact_sha256=sha(w/'RUBRIC.json'),position_balance=balances,model_identity_metadata_in_app=False))
    put(D/'VALIDATION/OUTPUT_BINDINGS_90.json',textchecks);put(D/'VALIDATION/PRIOR_PRESERVATION.json',dict(status='PASS',files=prior,old_Stage_O_90_records_unchanged=True))
    repro=P/'REPRODUCIBILITY';repro.mkdir(exist_ok=True)
    for name in ['prepare_stage_p_human_study_v1.py','package_stage_p_human_ready_v1.py','write_stage_p_human_instructions_v1.py']:shutil.copy2(ROOT/'scripts'/name,repro/name)
    for name in ['test_stage_p_human_session_v1.py','browser_stage_p_human_session_v1.py']:shutil.copy2(ROOT/'tests'/name,repro/name)
    shutil.copy2(Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/79f8fb11-634a-412b-aa4d-45140b1eead2/Pasted text.txt'),repro/'RESEARCHER_REQUEST.md');shutil.copy2(D/'ADAPTATION_LOG.md',P/'ADAPTATION_LOG.md')
    import importlib.metadata
    (repro/'TEST_REQUIREMENTS.txt').write_text('Backend/statistics tests: NumPy and Pillow.\nBrowser tests: Playwright1.51.0,Pillow,installed Chrome.\nThese are optional developer-test dependencies;the evaluator application requires no pip packages.\nRun tests with python -B REPRODUCIBILITY/test_stage_p_human_session_v1.py and python -B REPRODUCIBILITY/browser_stage_p_human_session_v1.py. They use only temporary SYNTHETIC fixtures.\nBuild scripts retain original repository paths/provenance and are not prerequisites for running the distributed app.\n',encoding='utf8')
    # Immutable rater-only manifest, excluding mutable folders which do not yet exist.
    put(APP/'IMMUTABLE_MANIFEST.json',dict(format='stage_p_blinded_evaluator_app_manifest_v1',files=members(APP,exclude=['IMMUTABLE_MANIFEST.json']),mutable_paths_created_later=['STATE','WORKSPACE/COLLECTION_OPEN.json'],no_rating_records_or_passwords_included=True))
    raterzip=D/'STAGE_P_BLINDED_EVALUATOR_WINDOWS.zip';archive(APP,raterzip)
    with tempfile.TemporaryDirectory(prefix='STAGE_P_EXTRACTED_BLINDED_') as td:
        extracted=Path(td)/'EVALUATOR_APP';unpack(raterzip,extracted,read(APP/'IMMUTABLE_MANIFEST.json')['files'])
        p=subprocess.run([str(extracted/'PYTHON/python.exe'),'-B','-X','utf8',str(extracted/'APP/verify_stage_p_blinded_app_v1.py'),'--app',str(extracted)],cwd=extracted,capture_output=True,text=True,encoding='utf8');assert p.returncode==0,p.stdout+p.stderr;portable=json.loads(p.stdout)
        assert not (extracted/'STATE').exists()
    put(D/'VALIDATION/EXTRACTED_WINDOWS_RUNTIME_VERIFICATION.json',dict(status='PASS',official_bundled_python_executed=True,standalone_verifier=portable,system_Python_required=False,GPU_required=False,rater_ZIP_crc_and_all_member_hashes=True,production_state_created=False))
    # Return an explicit validation table, not synthetic human results.
    report='''# Readiness validation — PASS, pending researcher review

| Check | Actual result |
| --- | --- |
| New session coverage |15 restaurants,45 unique frozen combined outputs,90 assignments,45/evaluator |
| Original LED identity |All15 Review/Visual pairs match accepted Stage O F outputs;benchmark reruns excluded |
| Qwen Review reuse |15/15 identical Review pairs;explicit analysis limitation |
| Randomization |Separate seeds/orders;exact5 appearances per system in each position per evaluator |
| Rubric |Original RUBRIC.json bytes/SHA256 unchanged;all six criteria and anchors retained |
| Sources |247 selected reviews,5,143 individual original photos;common exact sources for all three systems |
| Backend/statistical tests |14 tests PASS on isolated SYNTHETIC fixtures,including ownership,CSRF,save/resume,locking,export validation,journal tamper checks and restaurant-cluster statistics |
| Browser E2E |Headless Chrome156 on SYNTHETIC data;both raters login,save,restart,resume,submit,lock and download independent exports;zero JavaScript errors |
| Large gallery |2,040-photo synthetic stress case;24-thumbnail DOM bound,page85,ID jump,zoom,keyboard,desktop/mobile |
| Portable Windows runtime |Extracted ZIP executes bundled official Python3.11.9 and offline verifier;no system Python or GPU |
| Previous research |All protected prior files rehashed unchanged;Stage O90 ratings kept separate |
| Production session |No database,accounts,passwords,ratings or collection marker included;not opened for collection |
| Archive integrity |CRC,duplicate/traversal protection and every extracted-member SHA256 verified;final receipt adjacent to ZIP |

Screenshots and any test record counts refer only to visibly marked synthetic software tests. No test databases or test rating exports are delivered. No real human scores,model-quality ranking or final statistical conclusion is created.

Source hashes establish preservation,not correctness of every claim or independent participant authentication. True model identity mapping is coordinator-only and excluded from the separate evaluator ZIP. Styles may still be recognized;returning evaluators have prior exposure. The intended new session is concurrent across three systems,with no substitution of old Stage O scores.

Use START_PREVIEW.cmd for review. Only the operator's deliberate later START_COLLECTION.cmd opens data collection. Share only the separate STAGE_P_BLINDED_EVALUATOR_WINDOWS.zip after review;the master review ZIP contains the true identity key and must stay private.
'''
    (P/'VALIDATION_REPORT.md').write_text(report,encoding='utf8');shutil.copytree(D/'VALIDATION',P/'VALIDATION',dirs_exist_ok=True)
    # Copy the current implementation status, not old preparation metadata.
    status=P/'THESIS_STATUS';status.mkdir(exist_ok=True);shutil.copy2(ROOT/'thesis_sdd_docs/STAGE_P_THREE_SYSTEM_HUMAN_EVALUATION.md',status/'STAGE_P_THREE_SYSTEM_HUMAN_EVALUATION.md')
    put(P/'PACKAGE_MANIFEST.json',dict(format='stage_p_three_system_human_eval_ready_v1',files=members(P,exclude=['PACKAGE_MANIFEST.json']),rater_archive_external_name=raterzip.name,rater_archive_sha256=sha(raterzip),production_ratings=0,released_for_collection=False))
    master=D/'STAGE_P_THREE_SYSTEM_HUMAN_EVAL_READY.zip';archive(P,master)
    with tempfile.TemporaryDirectory(prefix='STAGE_P_MASTER_FINAL_CHECK_') as td:
        extracted=Path(td);unpack(master,extracted,read(P/'PACKAGE_MANIFEST.json')['files'])
        assert len(read(extracted/'COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json'))==90 and not (extracted/'EVALUATOR_APP/STATE').exists()
        # Run the portable verifier again from the master extraction, then prove future validator rejects absent real results.
        p=subprocess.run([str(extracted/'EVALUATOR_APP/PYTHON/python.exe'),'-B','-X','utf8',str(extracted/'EVALUATOR_APP/APP/verify_stage_p_blinded_app_v1.py')],cwd=extracted,capture_output=True,text=True,encoding='utf8');assert p.returncode==0,p.stdout+p.stderr
    check_prior(prior)
    result=dict(status='PASS_READY_FOR_RESEARCHER_REVIEW_NOT_RELEASED',master_ZIP=master.name,master_bytes=master.stat().st_size,master_sha256=sha(master),master_members=len(read(P/'PACKAGE_MANIFEST.json')['files'])+1,evaluator_ZIP=raterzip.name,evaluator_bytes=raterzip.stat().st_size,evaluator_sha256=sha(raterzip),new_real_ratings=0,old_records_reused=0,old_ratings_preserved=True,production_collection_open=False,businesses=15,outputs=45,assignments=90,per_evaluator=45,photos=5143,backend_tests=14,browser_E2E='PASS_SYNTHETIC_ONLY',extracted_bundled_Windows_runtime='PASS',every_extracted_member_hash='PASS',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    put(D/'DELIVERY_VALIDATION.json',result)
    for f in [master,raterzip]:(D/(f.name+'.sha256')).write_text(sha(f)+'  '+f.name+'\n',encoding='ascii')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
