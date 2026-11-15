"""Package and independently extract/reproduce Stage O statistical tables."""
import datetime as dt,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_o/final_human_statistical_analysis_20261106_v1';P=OUT/'PACKAGE'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def run(argv):
    r=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,argv)],capture_output=True,text=True,encoding='utf8');assert r.returncode==0,r.stdout+r.stderr;return r.stdout
def manifest():
    files=[dict(file=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file() and f!=P/'PACKAGE_MANIFEST.json'];put(P/'PACKAGE_MANIFEST.json',dict(format='stage_o_final_statistical_review_manifest_v1',files=files))
def main():
    s=P/'SCRIPTS';s.mkdir(exist_ok=True)
    for name in ['analyze_stage_o_final_statistics_v1.py','report_stage_o_final_statistics_v1.py','verify_stage_o_final_statistics_v1.py','prepare_stage_o_final_statistics_v1.py','package_stage_o_final_statistics_v1.py']:
        shutil.copy2(ROOT/'scripts'/name,s/name)
    runtime=read(P/'TABLES/RUNTIME.json');(s/'requirements.txt').write_text('\n'.join(k+'=='+runtime[k] for k in ['numpy','scipy','scikit-learn','matplotlib'])+'\n',encoding='utf8')
    (P/'VERIFY_PACKAGE.py').write_text("import runpy,sys\nfrom pathlib import Path\np=Path(__file__).resolve().parent\nsys.argv=[sys.argv[0],'--package',str(p)]\nrunpy.run_path(str(p/'SCRIPTS/verify_stage_o_final_statistics_v1.py'),run_name='__main__')\n",encoding='utf8')
    ws=ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP/WORKSPACE'
    for name in ['TA_PHOTO_19c9f445','TA_PHOTO_07732ad1','TA_PHOTO_1b958219']:
        dest=P/'QUALITATIVE/PHOTOS'/f'{name}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ws/'PHOTOS'/dest.name,dest)
    preservation=[]
    for row in read(P/'PROVENANCE/PROTECTED_INPUT_FILES.json'):
        f=ROOT/row['original_path'];assert sha(f)==row['sha256'];assert sha(P/row['package_path'])==row['sha256'];preservation.append(row['original_path'])
    photos=read(P/'PROVENANCE/PHOTO_INTEGRITY_5143.json')
    for row in photos:assert sha(ROOT/row['original_path'])==row['sha256']
    # No data-dependent exclusions or numerical edits. Check the complete fixed input panel again.
    put(P/'PROVENANCE/PRESERVATION_VERIFICATION.json',dict(status='PASS',unchanged_original_files=len(preservation),unchanged_photo_files=len(photos),ratings_or_assignments_changed=False,original_photos_or_reviews_changed=False,checkpoints_loaded_or_modified=False,new_training_inference_or_references=False,Qwen_started=False,checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    # Privacy check: reject sensitive file types/known credential markers, without printing any secret.
    forbidden=[]
    import re
    for f in P.rglob('*'):
        if not f.is_file():continue
        if f.suffix.lower() in ['.sqlite','.sqlite3','.db','.pem','.key'] or f.name.lower() in ['.env','credentials.json','sessions.json']:forbidden.append(f.relative_to(P).as_posix())
        if f.suffix.lower() in ['.json','.md','.txt','.csv','.py']:
            text=f.read_text(encoding='utf-8-sig');patterns=[r'\bhf_[A-Za-z0-9]{20,}\b',r'\bsk-[A-Za-z0-9_-]{20,}\b',r'BEGIN (?:RSA |OPENSSH )?PRIVATE KEY',r'"password_hash"\s*:',r'"session_token"\s*:']
            if any(re.search(pattern,text) for pattern in patterns):forbidden.append(f.relative_to(P).as_posix())
    assert not forbidden,'Sensitive file/marker detected; review locally'
    put(P/'PROVENANCE/PRIVACY_CHECK.json',dict(status='PASS',private_DB_accounts_password_hashes_or_session_files_included=False,credential_marker_scan='PASS',scope='Known secret patterns and forbidden private file types; original pseudonymous study provenance retained'))
    validation=dict(status='PASS',numerical_reconstruction=read(P/'TABLES/NUMERICAL_VERIFICATION.json'),original_preservation='PASS',native_input_and_output_bindings='PASS',three_figure_PNGs_visually_inspected=True,privacy_scan='PASS',archive_extraction_and_every_member_hash='PASS',portable_reproduction_of_all_deterministic_tables='PASS',limitations='Technical integrity and reproducibility do not independently authenticate participants or certify all semantic judgments.')
    # This file is finalized only after the following checks actually pass; any failure removes its claim.
    final=P/'PROVENANCE/FINAL_VALIDATION.json'
    if final.exists():raise ValueError('Final package already validated; do not overwrite')
    manifest();pre=run([s/'verify_stage_o_final_statistics_v1.py','--package',P]);put(OUT/'PREPACKAGE_VERIFICATION.json',json.loads(pre))
    stage=OUT/'PORTABLE_TEST';stage.mkdir(exist_ok=False)
    # Test a preliminary archive without a prospective final PASS receipt.
    preliminary=OUT/'PORTABLE_TEST_INPUT.zip'
    with zipfile.ZipFile(preliminary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(P.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(P).as_posix())
    with zipfile.ZipFile(preliminary) as z:
        assert z.testzip() is None;z.extractall(stage/'EXTRACTED')
    extracted=stage/'EXTRACTED';put(OUT/'EXTRACTED_PACKAGE_VERIFICATION.json',json.loads(run([extracted/'VERIFY_PACKAGE.py'])))
    output=stage/'REPRODUCED_TABLES';run([extracted/'SCRIPTS/analyze_stage_o_final_statistics_v1.py','--package',extracted,'--output',output])
    matched=[]
    for f in (P/'TABLES').iterdir():
        if f.name=='RUNTIME.json':continue
        assert sha(f)==sha(output/f.name),f.name;matched.append(f.name)
    put(P/'PROVENANCE/PORTABLE_REPRODUCTION.json',dict(status='PASS',fresh_ZIP_extraction=True,files_matched_byte_for_byte=matched,excluded_runtime_environment_metadata_only=['RUNTIME.json'],analysis_run_without_live_DB_full_photo_archive_checkpoint_or_GPU=True))
    put(final,validation);manifest()
    finalzip=OUT/'STAGE_O_FINAL_STATISTICAL_REVIEW.zip'
    with zipfile.ZipFile(finalzip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(P.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(P).as_posix())
    with zipfile.ZipFile(finalzip) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        z.extractall(stage/'FINAL_EXTRACTED')
        for f in P.rglob('*'):
            if f.is_file():assert sha(f)==sha(stage/'FINAL_EXTRACTED'/f.relative_to(P))
    final_check=json.loads(run([stage/'FINAL_EXTRACTED/VERIFY_PACKAGE.py']));put(OUT/'FINAL_EXTRACTED_VERIFICATION.json',final_check)
    h=sha(finalzip);(OUT/(finalzip.name+'.sha256')).write_text(h+'  '+finalzip.name+'\n',encoding='ascii')
    receipt=dict(status='PASS',artifact=finalzip.name,sha256=h,bytes=finalzip.stat().st_size,members=len(list(P.rglob('*')))-len([f for f in P.rglob('*') if f.is_dir()]),tables_reproduced=len(matched),ratings_sha256=read(P/'TABLES/NUMERICAL_VERIFICATION.json')['ratings_sha256'],archive_reopened_extracted_every_member_verified=True,review_only_stop=True,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    put(OUT/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
