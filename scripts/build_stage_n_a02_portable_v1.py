"""Build a blinded Windows A02-only package from a consistent read-only snapshot."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import urllib.request
import zipfile

from s1000_final_eval_core_v1 import read, digest, file_hash

HERE = Path(__file__).resolve().parent
RUNTIME_URL = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip'
RUNTIME_MD5 = '6d9aa08531d48fcc261ba667e2df17c4'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def build(workspace, output):
    root = Path(workspace).resolve(); out = Path(output).resolve()
    if out.exists(): raise ValueError('Fresh output directory required')
    out.mkdir(parents=True); package = out/'STAGE_N_A02_PORTABLE'; package.mkdir()
    (package/'DATA').mkdir(); (package/'APP').mkdir(); (package/'WEB').mkdir(); (package/'PHOTOS').mkdir()
    cfg = read(root/'CONFIG.json'); original = read(root/'COORDINATOR_ONLY/PRIVATE_ITEMS.json')
    for rel, h in cfg['file_bindings'].items():
        assert file_hash(root/rel) == h, rel
    items = sorted((x for x in original if x['annotator']=='A02'),key=lambda x:x['position'])
    assert len(items)==120 and len({x['business_id'] for x in items})==30
    fields = ['alias','annotator','position','business_id','review_text','review_text_sha256','visual_text','visual_text_sha256']
    blinded = [{k:x[k] for k in fields} for x in items]
    put(package/'DATA/ITEMS.json', blinded)
    sources = {b:read(root/v['file']) for b,v in cfg['sources'].items()}
    photos = {}
    for b,s in sources.items():
        assert digest(s)==cfg['sources'][b]['canonical_sha256']
        for p in s['photos']:
            actual = root/'PHOTOS'/(p['photo_id']+'.jpg')
            assert file_hash(actual)==p['sha256']==p['original_source_sha256']
            assert actual.stat().st_size==p['bytes']
            photos[p['photo_id']] = p
    assert len(photos)==343
    put(package/'DATA/SOURCES.json',sources)
    shutil.copy2(root/'RUBRIC.json',package/'DATA/RUBRIC.json')
    for photo_id in photos: shutil.copy2(root/'PHOTOS'/(photo_id+'.jpg'), package/'PHOTOS'/(photo_id+'.jpg'))
    conn=sqlite3.connect((root/'PRIVATE_STATE/ratings.sqlite3').as_uri()+'?mode=ro',uri=True,timeout=15)
    conn.row_factory=sqlite3.Row
    try:
        conn.execute('BEGIN')
        rows=[{'alias':r['alias'],'rater':r['rater'],'version':r['version'],'payload':json.loads(r['payload'])}
              for r in conn.execute("SELECT * FROM ratings WHERE rater='A02' ORDER BY alias")]
        events=[json.loads(r['event_json']) for r in conn.execute('SELECT * FROM events ORDER BY seq')
                if json.loads(r['event_json'])['rater']=='A02']
        auth=dict(conn.execute("SELECT * FROM users WHERE id='A02' AND role='rater'").fetchone())
        a01=[dict(r) for r in conn.execute("SELECT * FROM ratings WHERE rater='A01' ORDER BY alias")]
    finally: conn.rollback(); conn.close()
    put(package/'DATA/ACCOUNT.json',auth)  # Salted hash only; never plaintext.
    previous=None; seed=[]
    for e in events:
        event={**e,'previous_event_sha256':previous,'baseline_source_event_sha256':digest(e)}
        previous=digest(event);seed.append(event)
    baseline={'ratings':rows,'events':seed,'original_a02_states_sha256':digest(rows)}
    put(package/'DATA/BASELINE.json',baseline)
    binding={'format':'stage_n_A02_portable_binding_v1','rater':'A02','expected_records':120,'expected_businesses':30,
        'synthetic':False,'built_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'original_config_sha256':file_hash(root/'CONFIG.json'),'original_assignment_key_sha256':file_hash(root/'COORDINATOR_ONLY/PRIVATE_ITEMS.json'),
        'original_rubric_sha256':file_hash(root/'RUBRIC.json'),'original_cohort_sha256':file_hash(root/'COHORT30.json'),
        'original_release_sha256':file_hash(root/'LIVE_RELEASE.json'),'original_direct_access_sha256':file_hash(root/'DIRECT_RATER_ACCESS.json'),
        'blinded_items_sha256':file_hash(package/'DATA/ITEMS.json'),'sources_sha256':file_hash(package/'DATA/SOURCES.json'),
        'baseline_sha256':file_hash(package/'DATA/BASELINE.json'),'baseline_rating_count':len(rows),'baseline_event_count':len(seed),
        'baseline_submitted_count':sum(r['payload']['status']=='SUBMITTED' for r in rows),
        'source_canonical_sha256':{b:digest(s) for b,s in sources.items()},
        'authority':'Researcher requested an A02-only portable continuation of the already released study; no participant declaration or score is invented.',
        'authorship_limitation':'Account actions and local timestamps are recorded, not independently authenticated human identity or source inspection.'}
    put(package/'DATA/BINDING.json',binding)
    for name in ['stage_n_a02_portable_v1.py','stage_n_human_evaluation_v1.py','stage_l_reference_app_v1.py','s1000_final_eval_core_v1.py']:
        shutil.copy2(HERE/name,package/'APP'/name)
    for name in ['app.js','app.css','index.html']:
        shutil.copy2(HERE/'stage_n_human_web_v1'/name,package/'WEB'/name)
    html=(package/'WEB/index.html').read_text(encoding='utf8')
    html=html.replace('<nav><h2 id="progress">','<div class="actions"><button id="portableBackup">Download progress backup</button><button id="portableFinal">Download completed A02 results</button></div><p>Only A02 work is included. Keep this folder to resume. After Submitted 120 / 120, send RESULTS/A02_RESULTS_FINAL.json to the coordinator.</p><nav><h2 id="progress">')
    html=html.replace('No actual study scoring is authorized until the release is approved.','A02 portable continuation of the researcher-released study. No model identities or other rater scores are supplied.')
    (package/'WEB/index.html').write_text(html,encoding='utf8')
    with (package/'WEB/app.js').open('a',encoding='utf8') as f:
        f.write("\n$('portableBackup').onclick=()=>download('/api/export','A02_PROGRESS_BACKUP.json').catch(error);\n$('portableFinal').onclick=()=>download('/api/export?final=1','A02_RESULTS_FINAL.json').catch(error);\n")
    runtime=out/'python-3.11.9-embed-amd64.zip'
    with urllib.request.urlopen(RUNTIME_URL,timeout=60) as response, runtime.open('xb') as target:
        shutil.copyfileobj(response,target)
    assert hashlib.md5(runtime.read_bytes()).hexdigest()==RUNTIME_MD5, 'Official release MD5 mismatch'
    with zipfile.ZipFile(runtime) as z:
        assert z.testzip() is None
        z.extractall(package/'PYTHON')
    (package/'PYTHON/python311._pth').write_text('python311.zip\n.\n..\\APP\n',encoding='ascii')
    put(package/'RUNTIME_PROVENANCE.json',{'source_url':RUNTIME_URL,'release_page':'https://www.python.org/downloads/release/python-3119/',
        'upstream_archive_sha256':file_hash(runtime),'upstream_archive_md5':RUNTIME_MD5,
        'distribution':'Official Windows x64 embeddable Python 3.11.9; license included in PYTHON/LICENSE.txt',
        'local_change':'python311._pth adds the relative APP directory; no pip packages or system installation.'})
    (package/'START_A02.cmd').write_bytes(b'@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\n"%~dp0PYTHON\\python.exe" -B -X utf8 "%~dp0APP\\stage_n_a02_portable_v1.py"\r\nif errorlevel 1 pause\r\n')
    (package/'VERIFY_PACKAGE.cmd').write_bytes(b'@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\n"%~dp0PYTHON\\python.exe" -B -X utf8 "%~dp0APP\\stage_n_a02_portable_v1.py" --verify\r\npause\r\n')
    guide='''# A02 — Offline Windows review package

1. Copy the entire ZIP to a Windows 10/11 x64 computer. Right-click **Extract All**. Do not run inside the ZIP, a read-only directory, or a shared/synchronised folder. Use a local writable folder.
2. Double-click **START_A02.cmd**. Python is bundled; no installation, internet or administrator rights are needed. Keep its terminal open while working. Your browser opens the local address shown there, usually http://127.0.0.1:8766/.
3. Sign in as **A02**, using your existing password sent separately by the coordinator. Passwords are not printed in this guide. Do not use another person's account.
4. Continue your own 120 items. Any previously submitted A02 work remains locked and counts toward progress. Do not continue A02 work on the coordinator's original site after switching to this package.
5. Select an item; the correct sources/output open automatically. Read selected reviews and inspect original photos (click for full size). Judge independently; no rewriting of summaries is required.
6. Choose 1–5 using the exact rubric shown under each criterion. 2 and 4 are intermediate. Review Faithfulness, Aspect Coverage and Usefulness apply to Review only; Coherence and Non-Redundancy apply to all available output; Visual Grounding applies to the Visual output against this business's photos. Preserve Visual N/A. An empty applicable Visual still requires a score.
7. **Save draft** to pause, or **Submit and lock this item** when finished. Wait for success. Submitted items cannot be edited. General comments are optional. If reporting a serious grounding issue, provide the exact output span, recipient review/photo IDs, explanation and certainty.
8. You may close the browser and terminal after saving. Resume using START_A02.cmd from the SAME extracted folder. Scores are in STATE/ratings.sqlite3, with automatic JSON backups in RESULTS/. Do not delete or replace those directories. Copy the whole folder for your own backup only while the program is closed.
9. At **Submitted 120 / 120**, click **Download completed A02 results**. Send the single **A02_RESULTS_FINAL.json** to the coordinator. The same final file is saved automatically under RESULTS/. Do not send the account file, STATE database or entire program back. Do not manually edit the results JSON.
10. Work independently without consulting the other rater's scores or using AI to score the study. If something fails, send the error text and your progress backup to the coordinator; do not reset the study or fill scores merely to finish.

## Included data

- 30 frozen businesses, 120 original anonymous outputs in A02's original order.
- 343 individual original JPEGs and all selected reviews. A business's sources recur across its four outputs.
- Original six-criterion rubric and native N/A/empty-output handling.
- Only A02's own existing records; no A01 scores, condition identity key, Teacher targets, human references or automatic rankings.

## Folder layout

START_A02.cmd starts the local application. VERIFY_PACKAGE.cmd checks immutable files and stored journal. WEB/ is the interface; DATA/ contains blinded source/output data and A02's salted login hash; PHOTOS/ contains individual photos; PYTHON/ is the bundled runtime; APP/ contains code; STATE/ and RESULTS/ are created on first launch.

This is a Windows x64 package, not an Android/iPhone/macOS app. It runs on your own computer's localhost and does not connect to the coordinator's computer. Device clocks, actual identity and source inspection are not independently authenticated by this program. Returned results are checked against the original freeze and imported without overwriting other rater records.
'''
    (package/'READ_ME_FIRST_EN.md').write_text(guide,encoding='utf8')
    vi='''A02 — CÁCH DÙNG NHANH

1. Giải nén toàn bộ ZIP trên máy Windows 10/11 64-bit.
2. Mở START_A02.cmd, giữ cửa sổ đó mở. Trình duyệt tự hiện trang chấm trên máy bạn.
3. Đăng nhập A02 bằng mật khẩu đã nhận riêng. Không cần cài Python hay kết nối máy người điều phối.
4. Chấm tiếp các mục còn lại; các mục A02 đã nộp được giữ nguyên. Từ đây chỉ dùng bản này, không chấm song song trên trang cũ.
5. Save draft để lưu nháp; Submit and lock this item để nộp. Mục đã nộp sẽ khóa.
6. Có thể nghỉ rồi mở lại START_A02.cmd trong cùng thư mục. Không xóa STATE hoặc RESULTS.
7. Khi Submitted 120 / 120, bấm Download completed A02 results và gửi A02_RESULTS_FINAL.json. File cũng tự có trong RESULTS.

Hướng dẫn chấm chi tiết bằng tiếng Anh nằm trong READ_ME_FIRST_EN.md và ngay dưới từng tiêu chí trên trang. Không trao đổi điểm với A01 và không dùng AI để chấm thay.
'''
    (package/'BAT_DAU_VI.txt').write_text(vi,encoding='utf-8-sig')
    members={p.relative_to(package).as_posix():file_hash(p) for p in sorted(package.rglob('*')) if p.is_file()}
    put(package/'MANIFEST.json',{'format':'stage_n_A02_package_manifest_v1','files':members})
    zip_path=out/'STAGE_N_A02_PORTABLE_WINDOWS.zip'
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    checks=out/'EXTRACTED_VERIFICATION'
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        z.extractall(checks)
    extracted=checks/package.name
    for rel,h in read(extracted/'MANIFEST.json')['files'].items():assert file_hash(extracted/rel)==h
    assert not (extracted/'STATE').exists() and not (extracted/'PRIVATE_STATE').exists()
    assert not any('condition' in x for x in read(extracted/'DATA/ITEMS.json'))
    receipt={'status':'PASS','zip':zip_path.name,'zip_bytes':zip_path.stat().st_size,'zip_sha256':file_hash(zip_path),
        'businesses':30,'items':120,'selected_reviews':sum(len(s['reviews']) for s in sources.values()),'unique_original_photos':len(photos),
        'native_output_texts_and_aliases_unchanged':True,'rubric_exact_original_sha256':file_hash(package/'DATA/RUBRIC.json'),
        'A02_baseline_rating_count':len(rows),'A02_baseline_submitted_count':binding['baseline_submitted_count'],
        'A01_records_in_recipient_package':0,'condition_mapping_in_recipient_data':False,
        'A01_state_digest_at_snapshot':digest(a01),'A02_baseline_state_digest':digest(rows),
        'binding_sha256':file_hash(package/'DATA/BINDING.json'),'zip_CRC_extract_and_every_member_sha256':'PASS',
        'runtime_execution_and_HTTP_tests':'PENDING - separate verification required',
        'source_workspace_read_only':True}
    put(out/'PACKAGE_BUILD_RECEIPT.json',receipt)
    put(out/'COORDINATOR_ONLY_DISTRIBUTION_SNAPSHOT.json',{'binding':binding,'baseline':baseline,'original_a02_events':events,'A01_state_digest':digest(a01)})
    (out/(zip_path.name+'.sha256')).write_text(file_hash(zip_path)+'  '+zip_path.name+'\n',encoding='ascii')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(build(a.workspace,a.output),indent=2))
