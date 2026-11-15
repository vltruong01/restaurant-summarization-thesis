"""Package only the validated Stage O application; never releases a real study."""
from pathlib import Path
import collections,datetime,hashlib,http.cookiejar,json,os,shutil,socket,subprocess,sys,tarfile,tempfile,time,urllib.error,urllib.request,zipfile

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/stage_o/tripadvisor_external_v1';APP=O/'HUMAN_APP'
D=ROOT/'outputs/stage_o/human_evaluation_ready_20261105_v1';P=D/'PACKAGE'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')

def portable_smoke(package):
    """Fresh interpreter in an extracted bundle; private test state stays outside it."""
    with tempfile.TemporaryDirectory(prefix='stage_o_PORTABLE_READ_ONLY_') as td:
        t=Path(td);state=t/'private_state'
        base=[sys.executable,'-X','utf8','-B',str(package/'scripts/stage_o_human_evaluation_v1.py')]
        env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'}
        setup=subprocess.run(base+['setup-accounts','--workspace',str(package/'WORKSPACE'),'--state-dir',str(state)],cwd=package,env=env,capture_output=True,text=True,check=True)
        credentials=dict(line.split(' | ',1) for line in setup.stdout.splitlines() if ' | ' in line)
        assert set(credentials)=={'A01','A02','COORDINATOR'}
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        origin='http://127.0.0.1:'+str(port)
        process=subprocess.Popen(base+['serve','--workspace',str(package/'WORKSPACE'),'--state-dir',str(state),'--preview','--port',str(port)],cwd=package,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        try:
            for _ in range(80):
                try:
                    with urllib.request.urlopen(origin+'/api/status',timeout=1) as response:status=json.load(response)
                    break
                except (OSError,urllib.error.URLError):
                    assert process.poll() is None,'Packaged runtime exited unexpectedly';time.sleep(.1)
            else:raise AssertionError('Packaged app did not start')
            assert status['preview'] is True and status['released'] is False and status['expected_records']==90
            def login(identity):
                client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
                req=urllib.request.Request(origin+'/api/login',data=json.dumps(dict(identity=identity,password=credentials[identity])).encode(),headers={'Origin':origin,'Content-Type':'application/json'})
                with client.open(req) as response:session=json.load(response)
                return client,session
            c,s=login('A01')
            with c.open(origin+'/api/items') as response:items=json.load(response)['items']
            assert len(items)==45
            with c.open(origin+'/api/item/'+items[0]['alias']) as response:item=json.load(response)
            assert item['read_only'] and item['reviews'] and item['photos']
            with c.open(origin+item['photos'][0]['url']) as response:image=response.read()
            assert hashlib.sha256(image).hexdigest()==sha(package/'WORKSPACE/PHOTOS'/(item['photos'][0]['photo_id']+'.jpg'))
            for action in ['save','submit']:
                req=urllib.request.Request(origin+'/api/rating/'+item['alias']+'/'+action,data=b'{}',headers={'Origin':origin,'Content-Type':'application/json','X-CSRF-Token':s['csrf']})
                try:c.open(req);raise AssertionError('Packaged preview allowed writes')
                except urllib.error.HTTPError as error:assert error.code==400
            c,s=login('COORDINATOR')
            with c.open(origin+'/api/coordinator/export') as response:export=json.load(response)
            assert export['records']==[] and export['audit']['stored_rating_records']==0 and export['reference_free'] is True
            # A live-start command must fail before setup or a release can be invented.
            blocked=subprocess.run([sys.executable,'-B',str(package/'scripts/launch_stage_o_review_v1.py'),'--live','--no-browser'],cwd=package,env=env,capture_output=True,text=True)
            assert blocked.returncode!=0 and 'not released' in blocked.stderr
            return dict(status='PASS',fresh_interpreter=True,original_repository_on_pythonpath=False,portable_rater_items=45,original_photo_HTTP_bytes_verified=True,preview_writes_blocked=True,exported_rating_records=0,live_start_without_release_blocked=True)
        finally:
            process.terminate();process.communicate(timeout=10);credentials.clear()

def main():
    unit=read(D/'SOFTWARE_TEST_RESULTS.json');browser=read(D/'VALIDATION/BROWSER_E2E_RESULTS.json');integrity=read(D/'FROZEN_INPUT_INTEGRITY.json')
    assert unit['status']==browser['status']==integrity['status']=='PASS'
    assert all(sha(APP/path)==h for path,h in browser['application_source_sha256'].items()),'App source changed after browser validation'
    assert not (APP/'WORKSPACE/PRIVATE_STATE').exists() and not (APP/'WORKSPACE/LIVE_RELEASE.json').exists()
    config=read(APP/'WORKSPACE/CONFIG.json');assert config['synthetic'] is False and config['expected_records']==90 and len(config['sources'])==15
    private=read(APP/'WORKSPACE/COORDINATOR_ONLY/PRIVATE_ITEMS.json');assert len(private)==90
    frozen=read(O/'COHORT15_FREEZE.json');assert set(config['sources'])=={r['business_id'] for r in frozen['restaurants']}
    for rater in ['A01','A02']:
        items=[r for r in private if r['annotator']==rater]
        assert len(items)==45 and collections.Counter(r['condition'] for r in items)=={'RO':15,'U':15,'F':15}
    native={(r['condition'],r['business_id']):r for r in read(O/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json')}
    for row in private:
        expected=native[row['condition'],row['business_id']]['component_texts']
        assert row['review_text']==expected['review'] and row['visual_text']==expected.get('visual')
    assert not P.exists(),'Do not overwrite a review package; inspect any partial build first'
    shutil.copytree(APP,P,ignore=shutil.ignore_patterns('__pycache__','*.pyc','PRIVATE_STATE','*.sqlite3','*.sqlite3-wal','*.sqlite3-shm','LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json'))
    V=P/'VALIDATION';V.mkdir(exist_ok=True)
    for name in ['SOFTWARE_TEST_RESULTS.json','FROZEN_INPUT_INTEGRITY.json','FROZEN_INPUTS_BEFORE.json']:shutil.copyfile(D/name,V/name)
    for p in (D/'VALIDATION').iterdir():
        if p.is_file():shutil.copyfile(p,V/p.name)
    reproduction=V/'REPRODUCIBILITY';reproduction.mkdir()
    shutil.copyfile(Path(__file__),reproduction/Path(__file__).name)
    bindings=dict(cohort_freeze_sha256=sha(O/'COHORT15_FREEZE.json'),native_review_zip_sha256=read(O/'RTX3080_EXECUTION_V1/FINAL_COMPLETED_PACKAGE_RECEIPT.json')['sha256'],
        assembled_45_sha256=sha(O/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json'),call_plan_75_sha256=sha(O/'LED_INFERENCE_CALL_PLAN_75.json'),
        workspace_config_sha256=sha(APP/'WORKSPACE/CONFIG.json'),private_items_sha256=sha(APP/'WORKSPACE/COORDINATOR_ONLY/PRIVATE_ITEMS.json'),rubric_sha256=sha(APP/'WORKSPACE/RUBRIC.json'),
        randomization_seeds_sha256=sha(APP/'WORKSPACE/COORDINATOR_ONLY/RANDOMIZATION_SEEDS.json'),source_businesses=sorted(config['sources']),
        note='Captions and full native generation evidence remain frozen in the previously delivered native review ZIP. They are not shown to evaluators.')
    put(P/'PROVENANCE/NATIVE_INFERENCE_BINDINGS.json',bindings)
    report=dict(status='PASS_EVALUATOR_READY_PENDING_RESEARCHER_RELEASE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),restaurants=15,real_model_outputs=45,planned_assignments=90,assignments_per_evaluator=45,source_photographs=5143,largest_gallery=2040,
        backend_tests=unit['tests'],backend_result='PASS',browser_result='PASS',browser_version=browser['browser'],all_90_item_payloads_and_photo_ID_order='PASS',photo_HTTP_hash_checks=30,
        exhaustive_photo_HTTP_claim=False,frozen_files_rehashed=integrity['frozen_files_rehashed'],frozen_file_changes=0,production_database_included=False,production_accounts_included=False,genuine_ratings_created=0,
        temporary_software_fixture_records=6,fixture_records_included=False,live_release_included=False,Qwen_inference=0,training=0,
        default_mode='Authenticated read-only preview, localhost only',limitations=['Software tests are not human evaluation.','30 first/last photo HTTP samples; all photo bytes separately bound by frozen-file/package hashes.','All 5,143 photographs retained; large-set inspection workload remains.','Private hosting bundle must not be forwarded to raters; RO structure limits perfect blinding.'])
    put(V/'VALIDATION_REPORT.json',report)
    # No passwords, API credentials, mutable state or actual/synthetic rating exports.
    forbidden=[p for p in P.rglob('*') if p.is_file() and (p.suffix in ['.sqlite3','.db','.env'] or p.name in ['LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json'] or 'PRIVATE_STATE' in p.parts)]
    assert not forbidden
    files=[p for p in sorted(P.rglob('*')) if p.is_file()]
    manifest=dict(format='stage_o_evaluator_ready_package_v1',files=[dict(file=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],credentials_or_rating_database_included=False)
    put(P/'PACKAGE_MANIFEST.json',manifest)
    archive=D/'STAGE_O_TRIPADVISOR15_HUMAN_EVALUATION_READY_FOR_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(P))
    with tempfile.TemporaryDirectory(prefix='stage_o_ARCHIVE_VERIFY_') as td:
        extracted=Path(td)/'bundle';extracted.mkdir()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
            for name in z.namelist():assert (extracted/name).resolve().is_relative_to(extracted.resolve())
            z.extractall(extracted)
        for r in manifest['files']:
            p=extracted/r['file'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['file']
        smoke=portable_smoke(extracted)
        # Independent packaged verification entry point, no original repo access.
        check=subprocess.run([sys.executable,'-B',str(extracted/'VERIFY_PACKAGE.py')],cwd=extracted,capture_output=True,text=True,check=True)
        assert json.loads(check.stdout)['status']=='PASS'
    portable=dict(**smoke,zip_reopened=True,CRC='PASS',all_extracted_member_SHA256='PASS',standalone_verifier='PASS',
        scope='Fresh interpreter tested the extracted app; final repack adds only validation reports, not application/source changes.')
    put(D/'PORTABLE_EXTRACTION_TEST.json',portable);put(V/'PORTABLE_EXTRACTION_TEST.json',portable)
    report['portable_extracted_runtime']='PASS';report['release_without_approval_blocked']=True
    put(V/'VALIDATION_REPORT.json',report)
    # Include the actual extraction/runtime evidence in the review archive, then
    # reverify the final archive whose only changes are these validation reports.
    files=[p for p in sorted(P.rglob('*')) if p.is_file() and p.name!='PACKAGE_MANIFEST.json']
    manifest['files']=[dict(file=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    put(P/'PACKAGE_MANIFEST.json',manifest)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(P))
    with tempfile.TemporaryDirectory(prefix='stage_o_FINAL_ARCHIVE_VERIFY_') as td:
        final=Path(td)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
            z.extractall(final)
        for r in manifest['files']:
            p=final/r['file'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        parsed=read(final/'WORKSPACE/CONFIG.json');assert parsed['expected_records']==90 and parsed['synthetic'] is False
    receipt=dict(status='PASS_READY_FOR_REVIEW_NOT_RELEASED',ZIP=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),files=len(manifest['files'])+1,extracted_and_every_member_verified=True,portable_runtime='PASS',businesses=15,outputs=45,planned_records=90,genuine_ratings=0,live_release=False)
    put(D/'PACKAGE_RECEIPT.json',receipt);(D/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
