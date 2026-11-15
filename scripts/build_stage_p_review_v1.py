"""Package actual Stage P preparation evidence; do not imply inference happened."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'outputs/stage_p/qwen_comparison_v1'
ARCHIVE=P/'STAGE_P_QWEN_COMPARISON_REVIEW.zip'
EXCLUDED={ARCHIVE.name,ARCHIVE.name+'.sha256','PACKAGE_MANIFEST.json','PACKAGE_RECEIPT.json'}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def members():return [p for p in sorted(P.rglob('*')) if p.is_file() and p.name not in EXCLUDED and '__pycache__' not in p.parts and p.suffix!='.pyc']
def pack():
    files=members();forbidden=[p for p in files if p.suffix in ['.sqlite3','.db','.env'] or p.name in ['LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json'] or 'PRIVATE_STATE' in p.parts or p.suffix=='.safetensors']
    assert not forbidden,forbidden
    manifest=dict(format='stage_p_preparation_review_v1',status='PREPARATION_ONLY_QWEN_AND_HUMAN_RATINGS_PENDING',
        files=[dict(file=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],private_coordinator_bundle=True,ratings_or_credentials_included=False)
    put(P/'PACKAGE_MANIFEST.json',manifest)
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[P/'PACKAGE_MANIFEST.json']:z.write(p,p.relative_to(P))
    return manifest
def extract_check(destination,manifest):
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))==len(manifest['files'])+1
        assert all((destination/n).resolve().is_relative_to(destination.resolve()) for n in names)
        z.extractall(destination)
    for row in manifest['files']:
        p=destination/row['file'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['file']
    assert len(list((destination/'HUMAN_APP/WORKSPACE/PHOTOS').glob('*.jpg')))==5143
    assert len(read(destination/'HUMAN_APP/WORKSPACE/COORDINATOR_ONLY/ASSIGNMENT_SLOTS_120.json'))==120
    assert not (destination/'HUMAN_APP/WORKSPACE/CONFIG.json').exists()

def main():
    assert not ARCHIVE.exists(),'Do not silently overwrite a delivered archive'
    results={n:read(P/'VALIDATION'/n) for n in ['SOFTWARE_TEST_RESULTS.json','FULL120_LAYOUT_BROWSER.json','INFERENCE_CONTRACT_TESTS.json','INPUT_ALIGNMENT.json','CPU_COMPATIBILITY.json','ORIGINAL_AND_SOURCE_INTEGRITY.json']}
    assert all(v['status'].startswith('PASS') for v in results.values())
    browser=read(P/'VALIDATION/BROWSER/BROWSER_E2E_RESULTS.json');assert browser['status']=='PASS' and browser['real_four_condition_E2E']=='PENDING_QWEN_OUTPUTS'
    assert all(sha(P/'HUMAN_APP'/path)==h for path,h in browser['application_source_sha256'].items()),'App changed after browser tests'
    assert not (P/'HUMAN_APP/WORKSPACE/CONFIG.json').exists() and not (P/'HUMAN_APP/WORKSPACE/PRIVATE_STATE').exists()
    # Current program source and progress report accompany the frozen input evidence.
    shutil.copyfile(Path(__file__),P/'REPRODUCIBILITY'/Path(__file__).name)
    shutil.copyfile(ROOT/'thesis_sdd_docs/STAGE_P_QWEN_PREPARATION.md',P/'REPRODUCIBILITY/STAGE_P_QWEN_PREPARATION.md')
    report=dict(status='PASS_PREPARATION_ONLY_WITH_EXPLICIT_PENDING_ITEMS',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        existing_LED_outputs=45,Qwen_outputs=0,restaurants=15,planned_rating_records=120,planned_records_per_evaluator=60,genuine_ratings=0,
        preferred_revision=read(P/'FROZEN_SPEC.json')['models']['Qwen3.5-9B']['revision'],
        PASS=['30 exact LED F evidence blocks/native token-ID alignments','Both official model configs/tokenizers and 60 non-thinking prompt renderings on CPU',
              '11 backend tests on disposable four-condition fixtures','6 inference/finalizer contract tests','15-business/120-item synthetic browser layout with zero score writes',
              'Four-condition synthetic submit/restart/export E2E; 8 temporary test records deleted',
              '90 actual LED item API regression and 30 original-photo HTTP hashes, no ratings','120 unique balanced randomized slots before Qwen output',
              '10501 original files unchanged, 5158 exact source/photo copies verified'],
        PENDING=['Authorized compatible GPU and actual cost/runtime estimate','CUDA weight load/kernel/peak-memory verification','30 native Qwen generations and 15 genuine outputs',
                 'Final 60-output/120-real-item application freeze and full real-data E2E','Researcher review/release','120 genuine human ratings'],
        no_weights_downloaded=True,no_paid_APIs=True,no_new_rental=True,no_training=True,no_live_release=True,
        software_test_disclosure='Synthetic fixtures are disposable software inputs, never independent human evaluation; no fixture database/export is shipped.',
        previous_instance_status='Provider destroy confirmed; old endpoint not contacted',
        caveats=['Local 6 GiB GPU and currently free RAM insufficient for fixed BF16 plan.','Model and decoding differ from LED; no architecture/size causal inference.','Perfect blinding/source-inspection/participant identity cannot be guaranteed.','No external workspace/copy can be certified by this repository-only scan.'])
    put(P/'VALIDATION/VALIDATION_REPORT.json',report)
    manifest=pack()
    env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    with tempfile.TemporaryDirectory(prefix='stage_p_PORTABLE_PACKAGE_CHECK_') as td:
        e=Path(td)/'bundle';e.mkdir();extract_check(e,manifest)
        def call(python,args,expected=0):
            q=subprocess.run([str(python),'-X','utf8','-B']+args,cwd=e,env=env,capture_output=True,text=True,timeout=120)
            assert q.returncode==expected,(args,q.returncode,q.stdout[-1800:],q.stderr[-1800:]);return q
        verified=call(sys.executable,['VERIFY_PACKAGE.py']);assert read(e/'PACKAGE_MANIFEST.json')['files']
        assert json.loads(verified.stdout)['status']=='PASS'
        for file in ['scripts/stage_p_qwen_inference_v1.py','scripts/finalize_stage_p_human_workspace_v1.py']:
            call(sys.executable,[file,'--help'])
        cpu=call(ROOT/'.venv-stage-p-preflight/Scripts/python.exe',['scripts/stage_p_qwen_inference_v1.py','check','--destination',str(Path(td)/'CPU_REPORT.json')])
        assert read(Path(td)/'CPU_REPORT.json')['status'].startswith('PASS')
        call(ROOT/'.venv-stage-o-app-test/Scripts/python.exe',['tests/test_inference_contract.py'])
        blocked=call(sys.executable,['HUMAN_APP/scripts/launch_stage_o_review_v1.py','--no-browser'],expected=1)
        assert '30 genuine Qwen' in blocked.stderr and not (e/'HUMAN_APP/WORKSPACE/PRIVATE_STATE').exists()
        require_help=call(sys.executable,['scripts/stage_p_qwen_inference_v1.py','run'],expected=1)
        assert 'execution approval receipt' in require_help.stderr and not (e/'QWEN_EXECUTION').exists()
        portable=dict(status='PASS_EXTRACTED_PREPARATION_RUNTIME',all_members_SHA256_verified=True,ZIP_CRC='PASS',fresh_interpreter=True,original_repository_on_PYTHONPATH=False,
            offline_pinned_config_and_tokenizer_check='PASS',contract_tests=6,unfinalized_app_start_blocked=True,unapproved_model_load_blocked=True,
            production_state_created=False,ratings_created=0,CUDA_or_real_Qwen_E2E='PENDING_NOT_TESTED',
            note='Fresh extracted preparation tested; final repack adds reports only, no source/app change.')
    put(P/'VALIDATION/PORTABLE_EXTRACTION_TEST.json',portable)
    report['portable_preparation_runtime']='PASS';put(P/'VALIDATION/VALIDATION_REPORT.json',report)
    manifest=pack()
    with tempfile.TemporaryDirectory(prefix='stage_p_FINAL_ZIP_MEMBER_CHECK_') as td:extract_check(Path(td),manifest)
    receipt=dict(status='VERIFIED_PREPARATION_PACKAGE_QWEN_EXECUTION_PENDING',zip=str(ARCHIVE.relative_to(ROOT)),bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),
        members=len(manifest['files'])+1,every_member_extracted_and_verified=True,standalone_runtime_check='PASS_PREPARATION_ONLY',Qwen_outputs=0,real_human_ratings=0,live_release=False)
    put(P/'PACKAGE_RECEIPT.json',receipt);(P/(ARCHIVE.name+'.sha256')).write_text(receipt['sha256']+'  '+ARCHIVE.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
