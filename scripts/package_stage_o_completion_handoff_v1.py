"""Compact completion report for independent review, without analysis or edits."""
from pathlib import Path
import collections,contextlib,hashlib,json,shutil,sqlite3,subprocess,sys,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[1];O=ROOT/'outputs/stage_o/tripadvisor_external_v1'
C=ROOT/'outputs/stage_o/human_collection_complete_20261106_v1'
D=ROOT/'outputs/stage_o/completed_human_evaluation_handoff_20261106_v1';P=D/'PACKAGE'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not P.exists(),'Do not overwrite a delivered handoff'
    P.mkdir(parents=True);r=read(C/'EXPORT_AND_BACKUP_RECEIPT.json');assert sha(C/r['export_file'])==r['export_sha256']
    for name in ['STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json','COMPLETION_VERIFICATION.json','EXPORT_AND_BACKUP_RECEIPT.json']:shutil.copyfile(C/name,P/name)
    w=O/'HUMAN_APP/WORKSPACE'
    for name in ['CONFIG.json','RUBRIC.json','TWO_RATER_PROTOCOL_AMENDMENT.md','LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json']:
        (P/'PROVENANCE').mkdir(exist_ok=True);shutil.copyfile(w/name,P/'PROVENANCE'/name)
    assert sha(P/'PROVENANCE/CONFIG.json')==r['config_sha256'] and sha(P/'PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md')==r['protocol_sha256']
    for record in ['LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json']:
        x=read(w/record);source=w/x['authorization_record'];assert sha(source)==x['authorization_record_sha256']
        shutil.copyfile(source,P/'PROVENANCE'/source.name)
    assignments=[{k:i[k] for k in ['alias','annotator','position','business_id','condition']} for i in read(w/'COORDINATOR_ONLY/PRIVATE_ITEMS.json')]
    put(P/'ASSIGNMENT_MANIFEST_90.json',assignments)
    db=C/'PRIVATE_BACKUP/ratings.sqlite3';assert sha(db)==r['backup_sha256']
    with contextlib.closing(sqlite3.connect('file:'+db.resolve().as_posix()+'?mode=ro',uri=True)) as c:
        c.row_factory=sqlite3.Row
        journal=[dict(row) for row in c.execute('SELECT seq,event_json,event_hash FROM events ORDER BY seq')]
        rows=[dict(row) for row in c.execute('SELECT alias,rater,version,payload FROM ratings ORDER BY rater,alias')]
    assert len(journal)==138 and len(rows)==90
    put(P/'RATING_EVENT_JOURNAL_138.json',journal);put(P/'RAW_RATING_ROWS_90.json',rows)
    shutil.copyfile(ROOT/'scripts/verify_stage_o_handoff_v1.py',P/'VERIFY_HANDOFF.py')
    for filename in ['README.md','PROMPT_FOR_CHATGPT.txt']:
        assert (D/filename).is_file();shutil.copyfile(D/filename,P/filename)
    assert not list(P.rglob('*.sqlite3'))
    files=[p for p in sorted(P.rglob('*')) if p.is_file()]
    manifest=dict(format='stage_o_completed_collection_handoff_v1',files=[dict(file=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        accounts_passwords_sessions_SQLite_included=False,statistical_analysis_included=False)
    put(P/'PACKAGE_MANIFEST.json',manifest)
    check=subprocess.run([sys.executable,'-X','utf8','-B',str(P/'VERIFY_HANDOFF.py')],cwd=P,capture_output=True,text=True,check=True)
    verification=json.loads(check.stdout);put(D/'VERIFICATION.json',verification)
    archive=D/'STAGE_O_COMPLETED_HUMAN_EVALUATION_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(P))
    with tempfile.TemporaryDirectory(prefix='stage_o_COMPLETION_HANDOFF_VERIFY_') as td:
        out=Path(td)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
            assert all((out/n).resolve().is_relative_to(out.resolve()) for n in z.namelist());z.extractall(out)
        q=subprocess.run([sys.executable,'-X','utf8','-B',str(out/'VERIFY_HANDOFF.py')],cwd=out,capture_output=True,text=True,check=True)
        assert json.loads(q.stdout)==verification
    receipt=dict(status='PASS',ZIP=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),members=len(manifest['files'])+1,
        all_extracted_member_hashes_and_journal_reverified=True,ratings_changed=0,statistical_analysis_started=False,Qwen_started=False)
    put(D/'PACKAGE_RECEIPT.json',receipt);(D/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
