"""Release archive and read-only readiness only. Never creates/submits a rating."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,tempfile,urllib.request,zipfile

ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP'
D=ROOT/'outputs/stage_o/human_evaluation_release_20261105_v1';P=D/'RELEASE_PACKAGE'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def main():
    report=read(D/'FINAL_READINESS.json');assert report['status'].startswith('PASS_STAGE_O_RELEASED')
    assert report['before']==report['after']==dict(ratings=0,events=0,participants=0)
    assert not P.exists(),'Do not overwrite a release archive'
    # Preserve the release/authorization; exclude all mutable private state.
    shutil.copytree(APP,P,ignore=shutil.ignore_patterns('__pycache__','*.pyc','PRIVATE_STATE','PRIVATE_BACKUPS','PRIVATE_EXPORTS','*.sqlite3','*.sqlite3-wal','*.sqlite3-shm','*.log'))
    for name in ['FINAL_READINESS.json','FROZEN_WORKSPACE_BEFORE.json','RELEASE_RECORD.json']:shutil.copyfile(D/name,P/'VALIDATION'/name) if (P/'VALIDATION').exists() else None
    (P/'VALIDATION').mkdir(exist_ok=True)
    for name in ['FINAL_READINESS.json','FROZEN_WORKSPACE_BEFORE.json','RELEASE_RECORD.json']:shutil.copyfile(D/name,P/'VALIDATION'/name)
    old=ROOT/'outputs/stage_o/human_evaluation_ready_20261105_v1'
    history=P/'VALIDATION/HISTORICAL_PRE_RELEASE_SOFTWARE_VALIDATION';history.mkdir()
    for file in [old/'SOFTWARE_TEST_RESULTS.json',old/'VALIDATION/BROWSER_E2E_RESULTS.json',old/'PORTABLE_EXTRACTION_TEST.json']:
        shutil.copyfile(file,history/file.name)
    reproduce=P/'REPRODUCIBILITY';reproduce.mkdir()
    for name in ['release_stage_o_readiness_v1.py','package_stage_o_release_v1.py']:shutil.copyfile(ROOT/'scripts'/name,reproduce/name)
    shutil.copyfile(ROOT/'thesis_sdd_docs/STAGE_O_HUMAN_EVALUATION_RELEASE.md',P/'REPRODUCIBILITY/STAGE_O_HUMAN_EVALUATION_RELEASE.md')
    # Code and active deployment file ownership are explicit in the receipt.
    note=dict(status='STAGE_O_RELEASED_PARTICIPANT_ONBOARDING_PENDING',canonical_workspace=str((APP/'WORKSPACE').relative_to(ROOT)),
        passwords_bound_to='Existing canonical local installation only, not a fresh extraction',private_database_included=False,
        fresh_extraction='Deployment source only; creates its own new credentials at first setup. Do not run a parallel collection.',
        no_new_artificial_ratings=True,new_readiness_tests='Login/logout, real read-only payload/photo checks, empty export and final-export rejection only',
        historical_tests='Earlier disposable software-fixture tests are retained as historical receipts, not rerun or represented as human ratings',
        release_authority='Actual current researcher instruction; no independent supervisor/participant claims')
    put(P/'VALIDATION/RELEASE_SCOPE.json',note)
    # Rehash actual frozen study again after all readiness steps and docs edits.
    before=read(D/'FROZEN_WORKSPACE_BEFORE.json')['files']
    assert all((APP/'WORKSPACE'/r['file']).stat().st_size==r['bytes'] and sha(APP/'WORKSPACE'/r['file'])==r['sha256'] for r in before)
    sys.path.insert(0,str(APP/'scripts'))
    from stage_o_human_evaluation_v1 import Study
    study=Study(APP/'WORKSPACE');audit=study.audit()
    with study.connect() as c:
        counts={n:c.execute('SELECT COUNT(*) FROM '+n).fetchone()[0] for n in ['ratings','events','participants']}
    assert counts==dict(ratings=0,events=0,participants=0) and study.active()
    with urllib.request.urlopen('http://127.0.0.1:8767/api/status',timeout=10) as f:status=json.load(f)
    assert status==dict(released=True,preview=False,synthetic=False,planned_raters=2,expected_records=90,businesses=15)
    report['latest_live_status']=status;report['latest_audit_counts']=counts
    report['packaged_app_sha256']={p.relative_to(APP).as_posix():sha(p) for p in (APP/'scripts').rglob('*') if p.is_file() and p.suffix in ['.py','.js','.html','.css']}
    put(D/'FINAL_READINESS.json',report);put(P/'VALIDATION/FINAL_READINESS.json',report)
    # No credential/token material is copied; hashed accounts stay in the local DB.
    files=[p for p in sorted(P.rglob('*')) if p.is_file()]
    assert not [p for p in files if p.suffix in ['.sqlite3','.db','.env'] or 'PRIVATE_STATE' in p.parts]
    manifest=dict(format='stage_o_released_application_v1',files=[dict(file=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        private_state_or_credentials_included=False,ratings_included=0)
    put(P/'PACKAGE_MANIFEST.json',manifest)
    archive=D/'STAGE_O_TRIPADVISOR15_HUMAN_EVALUATION_RELEASED.zip';assert not archive.exists()
    def pack():
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(P.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(P))
    def unpack_check(path):
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
            assert all((path/n).resolve().is_relative_to(path.resolve()) for n in z.namelist());z.extractall(path)
        for r in manifest['files']:
            p=path/r['file'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        assert len(list((path/'WORKSPACE/PHOTOS').glob('*.jpg')))==5143
    pack()
    with tempfile.TemporaryDirectory(prefix='stage_o_RELEASE_ARCHIVE_ZERO_RATINGS_') as td:
        temp=Path(td);extracted=temp/'app';extracted.mkdir();unpack_check(extracted)
        env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'}
        verification=subprocess.run([sys.executable,'-X','utf8','-B',str(extracted/'VERIFY_PACKAGE.py')],cwd=extracted,env=env,capture_output=True,text=True,check=True)
        assert json.loads(verification.stdout)['status']=='PASS'
        # Fresh interpreter parses real source/config/aliases and the release.
        # Empty temporary state only: no test accounts or assessments created.
        code='''from pathlib import Path
import json,sys
sys.path.insert(0,str(Path('scripts').resolve()))
from stage_o_human_evaluation_v1 import Study,live_release_valid
s=Study(Path('WORKSPACE'),preview=True,state_dir=Path(sys.argv[1]))
assert live_release_valid(s.root) and len(s.key)==90 and len(s.sources)==15
assert s.audit()['stored_rating_records']==0
assert all(len(s.list_items({'id':r,'role':'rater'}))==45 for r in ['A01','A02'])
assert len(s.public_item({'id':'A01','role':'rater'},s.list_items({'id':'A01','role':'rater'})[0]['alias'])['reviews'])>0
assert s.export({'id':'COORDINATOR','role':'coordinator'})['records']==[]
with s.connect() as c:assert c.execute('SELECT COUNT(*) FROM users').fetchone()[0]==0 and c.execute('SELECT COUNT(*) FROM participants').fetchone()[0]==0 and c.execute('SELECT COUNT(*) FROM events').fetchone()[0]==0
print(json.dumps({'status':'PASS','release_valid':True,'items':90,'businesses':15,'ratings_or_test_accounts_created':0}))
'''
        q=subprocess.run([sys.executable,'-X','utf8','-B','-c',code,str(temp/'temporary_empty_state')],cwd=extracted,env=env,capture_output=True,text=True,check=True)
        portable=read(P/'VALIDATION/RELEASE_SCOPE.json')
        portable.update(portable_read_only_checks=json.loads(q.stdout),ZIP_CRC='PASS',all_extracted_members_sha256='PASS',standalone_verifier='PASS',
            source_or_ratings_modified=False,final_repack='Adds validation evidence only; no application/source change')
    put(P/'VALIDATION/PORTABLE_RELEASE_VALIDATION.json',portable)
    files=[p for p in sorted(P.rglob('*')) if p.is_file() and p.name!='PACKAGE_MANIFEST.json']
    manifest['files']=[dict(file=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    put(P/'PACKAGE_MANIFEST.json',manifest);pack()
    with tempfile.TemporaryDirectory(prefix='stage_o_RELEASE_FINAL_MEMBER_CHECK_') as td:unpack_check(Path(td))
    receipt=dict(status='PASS_STAGE_O_RELEASED_FOR_REAL_PARTICIPANT_ONBOARDING',ZIP=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),members=len(manifest['files'])+1,
        every_extracted_member_verified=True,URL='http://127.0.0.1:8767/',canonical_app=str(APP.relative_to(ROOT)),real_rating_records=0,artificial_ratings_created_or_submitted=0,
        participant_declarations_and_coordinator_review='Required from actual participants/coordinator; not fabricated',Qwen='DEFERRED')
    put(D/'PACKAGE_RECEIPT.json',receipt);(D/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
