"""Build A02-only offline revision app from the verified current snapshot."""
from pathlib import Path
import hashlib,json,shutil,sys,tempfile,zipfile,subprocess

ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT/'outputs/stage_p/three_system_human_evaluation_live_20261106_v1'
SNAPSHOT=ROOT/'outputs/stage_p/human_collection_complete_20261106T094354Z_v1'
DEST=ROOT/'outputs/stage_p/a02_portable_revision_20261106_v1'
APP=DEST/'STAGE_P_A02_REVISION_APP'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    assert not DEST.exists(),'Do not overwrite a delivered revision app'
    before=sha(LIVE/'STATE/ratings.sqlite3')
    APP.mkdir(parents=True)
    for name in ['PYTHON','WEB','WORKSPACE']:shutil.copytree(LIVE/name,APP/name)
    (APP/'APP').mkdir()
    for name in ['stage_p_human_evaluation_v1.py','human_app_core_v1.py','human_http_common_v1.py']:
        shutil.copy2(LIVE/'APP'/name,APP/'APP'/name)
    for name in ['stage_p_a02_portable_revision_v1.py','launch_stage_p_a02_revision_v1.py']:
        shutil.copy2(ROOT/'scripts'/name,APP/'APP'/name)
    shutil.copy2(SNAPSHOT/'A02_FINAL_45.json',APP/'WORKSPACE/A02_BASELINE_FINAL_45.json')
    baseline=read(APP/'WORKSPACE/A02_BASELINE_FINAL_45.json')
    assert len(baseline['records'])==45 and baseline['rater']=='A02' and all(r['annotator']=='A02' for r in baseline['records'])
    save(APP/'WORKSPACE/REVISION_BINDING.json',dict(format='stage_p_a02_revision_binding_v1',baseline_source=SNAPSHOT.relative_to(ROOT).as_posix()+'/A02_FINAL_45.json',baseline_file_sha256=sha(APP/'WORKSPACE/A02_BASELINE_FINAL_45.json'),study_id=baseline['study_id'],purpose='Researcher requested portable source-complete A02 revision on a separate computer. Only proposed revisions are recorded locally; no coordinator records are changed.'))
    js=APP/'WEB/app.js';s=js.read_text(encoding='utf8')
    s=s.replace("'Submitted '+d.items.filter(x=>x.status==='SUBMITTED').length+' / '+d.items.length","'New revision submissions '+d.items.filter(x=>x.status==='SUBMITTED').length+' / '+d.items.length+' | Unedited original ratings remain retained'")
    s=s.replace("'Not started; all applicable scores are blank.'","'Original A02 scores loaded. Edit only what you need; untouched records retain their original submission.'")
    s=s.replace("'New independent session open | 15 restaurants | 45 items per evaluator. Work independently; do not share judgments.'","'A02 REVISION COPY | 15 restaurants | 45 existing ratings. Untouched original records are retained. Review source reviews, Visual Evidence and photos before editing.'")
    s=s.replace('"NEW_SESSION_"+session.actor.id+"_PROGRESS.json"','"A02_REVISION_PROGRESS.json"').replace('"NEW_SESSION_"+session.actor.id+"_RESULTS_FINAL.json"','"A02_REVISED_RESULTS_FINAL.json"')
    js.write_text(s,encoding='utf8')
    html=APP/'WEB/index.html';s=html.read_text(encoding='utf8').replace('New session | Independent output review','A02 | Portable revision').replace('INDEPENDENT SESSION · TRIPADVISOR15','A02 REVISION · TRIPADVISOR15').replace('<input id="identity"','<input value="A02" id="identity"')
    s=s.replace('Download my completed results','Export revised results to send back').replace('Download my progress','Export revision progress')
    s=s.replace('New session: 15 restaurants, 45 items per evaluator. Work independently using your own account and the displayed sources. Save drafts to resume later; submitted items are locked.','A02 revision copy: existing scores are prefilled. Untouched original records remain unchanged. Save then submit edits; export revised results and send the JSON to the coordinator. Keep STATE to resume.')
    html.write_text(s,encoding='utf8')
    (APP/'START_A02_REVISION.cmd').write_text('@echo off\ncd /d "%~dp0"\n"%~dp0PYTHON\\python.exe" -B -X utf8 "%~dp0APP\\launch_stage_p_a02_revision_v1.py"\nif errorlevel 1 pause\n',encoding='ascii')
    (APP/'HUONG_DAN_A02_VI.txt').write_text('''HUONG DAN A02 - SUA DANH GIA TREN MAY WINDOWS RIENG

1. Giai nen TOAN BO ZIP vao mot thu muc. Khong chay truc tiep ben trong ZIP.
2. Bam dup START_A02_REVISION.cmd. Khong can cai Python, GPU hay ket noi may cua nguoi dieu phoi.
3. Lan dau, cua so terminal hien tai khoan A02 va mat khau moi. Ghi lai rieng. Mat khau khong phai mat khau trang cu va chi duoc hien mot lan. Khong dong cua so nay khi dang lam.
4. Mo http://127.0.0.1:8769/ tren CHINH may cua ban. Dang nhap A02 voi mat khau vua hien. Neu cong8769 bi chiem, dong ban cu cua UNG DUNG NAY truoc; khong tat chuong trinh la.
5. Co45 muc cua15 nha hang. Diem va nhan xet A02 hien tai da duoc dien san. Ca Review Consensus va Visual Evidence deu co; reviews nguon o ben trai va anh nguon ben duoi. Anh co nhieu trang; nhap anh de phong to/chuyen anh.
6. Sua cac muc can thiet theo rubric cu. Save draft de luu tam; Submit and lock this item de xac nhan ban sua. Muc da submit se khoa. Chi submit khi da kiem tra xong. Muc khong sua GIU NGUYEN diem va thoi gian nop goc, khong can nop lai45 muc.
7. Khi tat ca ban sua da submit, bam Export revised results to send back. Gui file A02_REVISED_RESULTS_FINAL.json cho nguoi dieu phoi. Neu con draft, nut xuat ban cuoi se bao can submit draft. Khong can gui database, mat khau hay toan bo anh lai.
8. Truoc khi nghi, Save draft va doi thong bao thanh cong. Chay lai START_A02_REVISION.cmd de tiep tuc. Giu nguyen thu muc STATE (tao sau lan dau), khong giai nen de len no va khong xoa no.

Tien do 'New revision submissions' chi dem cac muc ban moi xac nhan trong ban sua nay; khong phai so muc goc da nop. Diem goc chua cham lai se duoc danh dau retained original trong file xuat. Ung dung nay chi tao de xuat sua; KHONG tu thay doi du lieu tren may nguoi dieu phoi. Neu gui lai nhieu lan, gui file moi nhat va noi ro lan nao can dung.

Moi output duoc giu nguyen; khong co diem cua A01 hay khoa ten mo hinh trong goi nay. Su dung dung reviews va anh cua nha hang dang mo. Khong can ket noi Internet, khong co inference hay training.
''',encoding='utf8')
    save(APP/'OFFLINE_PACKAGE_SCOPE.json',dict(rater='A02',records=45,restaurants=15,review_source_count=247,photos=5143,visual_applicable=45,contains_A01_scores=False,contains_true_model_key=False,contains_original_database=False,proposed_revisions_only=True,baseline_file_sha256=sha(APP/'WORKSPACE/A02_BASELINE_FINAL_45.json')))
    # Verify all source bytes, including actual individual JPEGs, before delivery.
    config=read(APP/'WORKSPACE/CONFIG.json');reviews=0;photos={}
    for binding in config['sources'].values():
        source=read(APP/'WORKSPACE'/binding['file']);reviews+=len(source['reviews'])
        for photo in source['photos']:
            p=APP/'WORKSPACE/PHOTOS'/(photo['photo_id']+'.jpg')
            assert sha(p)==photo['sha256']==photo['original_source_sha256'] and p.stat().st_size==photo['bytes']
            photos[photo['photo_id']]=photo['sha256']
    assert reviews==247 and len(photos)==5143
    assert not (APP/'STATE').exists()
    files=sorted(p for p in APP.rglob('*') if p.is_file())
    save(APP/'PACKAGE_MANIFEST.json',dict(files=[dict(file=p.relative_to(APP).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    archive=DEST/'STAGE_P_A02_PORTABLE_REVISION_WITH_PHOTOS.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(APP.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(APP).as_posix())
    with tempfile.TemporaryDirectory(prefix='a02_portable_verify_') as tmp:
        extraction=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert len(z.namelist())==len(set(z.namelist()))
            z.extractall(extraction)
        manifest=read(extraction/'PACKAGE_MANIFEST.json')
        for row in manifest['files']:assert sha(extraction/row['file'])==row['sha256']
        # Run the extracted bundled interpreter; test state is outside delivery.
        testfile=ROOT/'scripts/test_stage_p_a02_portable_revision_v1.py'
        result=subprocess.run([str(extraction/'PYTHON/python.exe'),'-B','-X','utf8',str(testfile),'--app',str(extraction)],capture_output=True,text=True,encoding='utf8')
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        tests=json.loads(result.stdout)
    assert before==sha(LIVE/'STATE/ratings.sqlite3'),'Original database changed during packaging'
    receipt=dict(status='PASS',archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,source_reviews=reviews,photos=len(photos),all_photo_sha256='PASS',zip_extract_all_member_sha256='PASS',tests=tests,original_database_unchanged=True,delivered_artificial_ratings=0,delivered_accounts=0)
    save(DEST/'VALIDATION_REPORT.json',receipt)
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
