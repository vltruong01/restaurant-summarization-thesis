"""Compact Stage L.1 launch handoff. No reference prose or human events written."""
import csv
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from stage_l_reference_app_v1 import Store,read,new_json,file_hash,now

def main():
    project=Path.cwd().resolve();workspace=project/'outputs/stage_l/test100_human_references_v1';out=workspace/'LAUNCH_L1_20261029_v1'
    r=read(out/'STAGE_L1_VERIFICATION_RECEIPT.json');first=read(out/'FIRST_TWO_FROZEN_BINDINGS.json')['cases']
    tests=read(workspace/'L1_TECHNICAL_TEST_LOGS/TEST_RESULTS.json')
    if r['status']!='PASS_SOURCE_AND_HTTP_BACKEND' or not all(x['exit_code']==0 for x in tests):raise ValueError('Verification incomplete')
    store=Store(workspace)
    if store.audit()!=r['actual_progress']:raise ValueError('Human progress changed; refresh the report')
    with store.connect() as c:
        if any(c.execute('SELECT count(*) FROM '+t).fetchone()[0]!=r['real_database_counts_after'][t] for t in ['users','refs','events']):raise ValueError('Real participant state changed')
    for test in tests:shutil.copy2(workspace/'L1_TECHNICAL_TEST_LOGS'/test['test_log'],out/test['test_log'])
    new_json(out/'TECHNICAL_TEST_RECEIPT.json',tests)
    launch=r'''# Bắt đầu hai reference thật

Ứng dụng hiện chạy tại http://127.0.0.1:8765 . Mở địa chỉ này bằng trình duyệt trên chính máy có project. Không chạy thêm một server cùng cổng nếu trang đã mở được.

## Việc cần làm ngay

Chưa có tài khoản người tham gia. Người điều phối đăng ký từng người thật bằng CMD sau, chạy lệnh `call` hai lần: lần đầu chọn vai trò `author`, lần sau `checker`. ID là mã tự chọn của người thật; mỗi người tự nhập mật khẩu ít nhất 12 ký tự tại máy, không gửi mật khẩu vào chat. Không dùng một người đóng cả hai vai trò.

```bat
cd /d C:\thesis_proposal\yelp_dataset
call outputs\stage_l\test100_human_references_v1\REGISTER_PARTICIPANT.cmd
```

Người viết/checker phải chưa xem output RO/U/F/S, điểm Stage K hoặc Teacher answers dùng để xây reference. Nếu đã xem, phải khai báo; không tick xác nhận sai. Giao diện tách nguồn và có đăng nhập, nhưng không thể ngăn người có quyền đọc cả ổ đĩa mở các thư mục khác. Dùng workspace nguồn riêng/máy hoặc tài khoản hệ điều hành riêng cho người tham gia; công cụ portable đã có từ Stage L.

## Hai hồ sơ đầu, theo thứ tự manifest

1. **Mungo's Italian Eatery** — `-78Va3llKcPbL6SW4is8lw`: 30 reviews, 8 ảnh gốc.
2. **The Bungalow Tampa** — `0-M3vGd6IveJh77-Bwc8EQ`: 30 reviews, 16 ảnh gốc.

Cả hai vẫn NOT_STARTED; việc xếp vào đợt đầu không được tính là đã viết. Không có lựa chọn nào dựa trên output/điểm TEST.

Tác giả đăng nhập -> nhập ID hoặc tên vào **Find business** -> chọn hồ sơ. Đọc toàn bộ **Selected reviews**; dùng tìm kiếm để điều hướng nhưng không bỏ qua các review không khớp từ khóa. Mở **Original photos**, bấm từng ảnh để xem ảnh gốc đầy đủ và photo ID. Viết hai phần tiếng Anh vào `[REVIEW_CONSENSUS]` và `[VISUAL_EVIDENCE]`; không tự gõ marker vào nội dung.

Liên kết bằng chứng: chọn đoạn review, bấm **Link excerpt from this review**, điền **Exact span from your reference**, kiểm tra review ID/đoạn trích rồi **Add evidence link**. Với Visual, chọn photo ID, xác nhận đã trực tiếp xem ảnh rồi thêm span có căn cứ. Một span có thể có nhiều nguồn. Bất định/ảnh khó xem ghi vào notes, không biến suy đoán thành sự thật trong reference. Không có gợi ý summary do AI viết.

Chờ **Saved / server version** trước khi rời hồ sơ. **Save draft** để lưu; **Submit for independent checking** sau khi tác giả đã thực sự hoàn tất các việc trong ba xác nhận. Đăng xuất.

Checker khác người đăng nhập -> chọn hồ sơ SUBMITTED -> đối chiếu toàn bộ review/ảnh và từng link bằng chứng, kiểm tra cân bằng ý kiến, thiếu sót quan trọng và quan hệ trong ảnh. Ghi feedback rồi **Request revision** hoặc **Reject this version** nếu cần; tác giả sửa và gửi lại. Chỉ tick các xác nhận và **Approve exact version** sau khi thực sự kiểm độc lập và giải quyết bất đồng. Các phiên bản, phản hồi và hash được giữ trong lịch sử.

Sau khi hai hồ sơ được viết/duyệt thật, báo lại để kiểm provenance, hash và định dạng xuất. Không tự nhận đã có hai reference hợp lệ chỉ vì demo test thành công. `ACTUAL_PROGRESS_AND_BATCH_ORDER.csv` là snapshot lúc kiểm tra; tracker trong ứng dụng phản ánh tiến độ mới nhất. Sau pilot có 98 hồ sơ trong các đợt 10, đợt cuối 8; tổng công việc thật hiện còn đủ 100 hồ sơ.

## Nếu server đã dừng

Mở CMD và giữ cửa sổ chạy:

```bat
cd /d C:\thesis_proposal\yelp_dataset
.venv-caption-pilot\Scripts\python.exe -X utf8 -B scripts\stage_l_reference_app_v1.py serve --workspace outputs\stage_l\test100_human_references_v1
```

Sau đó mở http://127.0.0.1:8765 . Nếu báo cổng đang dùng, kiểm tra trang đang chạy trước; không tắt tiến trình không rõ danh tính. Server chỉ lắng nghe loopback, không mở cổng internet hay dịch vụ cloud.

Lệnh tự chuyển sang tiến trình nền bị công cụ thực thi chặn (`blocked by policy`); server mở trước đó vẫn chạy tại lúc bàn giao. Việc này không phải lỗi model hoặc nguồn ảnh. Nếu phiên chạy hiện tại kết thúc, dùng lệnh CMD trên.

## Kiểm tra hiển thị thủ công trước khi viết hàng loạt

Do công cụ browser không kết nối, người dùng cần đăng nhập và xác nhận: đủ 100 hồ sơ; mở đúng ID đầu tiên; review/ID đọc được; cả 8 ảnh hồ sơ đầu và 16 ảnh hồ sơ sau hiển thị; bấm ảnh mở được ảnh lớn. Không điền reference giả vào TEST để thử nút lưu. Các bước lưu/gửi/yêu cầu sửa/duyệt đã được kiểm bằng fixture tổng hợp riêng. Khi viết thật, chỉ lưu nội dung do chính tác giả viết.

Ứng dụng hiện không cung cấp nút xuất trên web. Dùng CLI đã có để xuất/kiểm tra. Sau các phiên làm việc, sao lưu bằng `backup` trong README_SETUP_VI.md. Không đưa database mật khẩu hoặc bản nháp cá nhân vào gói gửi ChatGPT công khai.

```bat
.venv-caption-pilot\Scripts\python.exe -X utf8 -B scripts\stage_l_reference_app_v1.py verify --workspace outputs\stage_l\test100_human_references_v1
```

Chưa chạy Stage M hay Human Evaluation. Chưa có reference freeze cuối.
'''
    (out/'START_FIRST_TWO_VI.md').write_text(launch,encoding='utf8')
    rows=[
        ('Application opens','PASS HTTP 200 for live HTML/JS/CSS','UNVERIFIED actual page rendering'),
        ('All 100 businesses listed','PASS exact manifest order through source API','UNVERIFIED list layout'),
        ('Selected reviews and IDs','PASS exact source JSON for all 100 / 2817 reviews','UNVERIFIED visual review navigation'),
        ('Original photos display','PASS 1136 original-file hashes + exact HTTP bytes + full image decoding','UNVERIFIED browser image display'),
        ('Photo ID/image correspondence','PASS every same-business route and original SHA256','UNVERIFIED visually opening each image'),
        ('Full-size viewing','PASS route returns unchanged original JPEG bytes','UNVERIFIED modal/zoom rendering'),
        ('Both text fields','PASS synthetic DOM editor contracts','UNVERIFIED browser typing'),
        ('Save and reopen','PASS existing synthetic backend restart/save and DOM recovery tests','UNVERIFIED browser UI flow'),
        ('Evidence attachment','PASS synthetic review quote/photo ID validation and DOM evidence builder','UNVERIFIED browser UI flow'),
        ('Submit and independent check','PASS synthetic distinct-role/identity and approval checks','UNVERIFIED browser UI; no real submissions'),
        ('Revision request/rejection','PASS synthetic request/revise/resubmit/history tests','UNVERIFIED browser UI; no real requests'),
        ('Export and validation','PASS synthetic 100-record freeze against existing Stage M contract','CLI feature; no final real TEST export')]
    browser='''# Stage L.1 interface verification

The actual local server is running at http://127.0.0.1:8765 and returns HTTP 200 for HTML, JavaScript and CSS. Its source API requires login. No real account was created and no actual TEST reference was saved/submitted/approved during technical tests.

Real-browser visual testing could not be performed: Windows Computer Use reports `Computer Use native pipe is unavailable ... The system cannot find the file specified. (os error 2)`; browser inventory contains zero connected browsers. This is a test-access limitation, not a confirmed application defect. No fallback browser automation or invented screenshot was used.

The 100-case source/1136-image GET audit used a SEPARATE ephemeral instance of the unchanged server class with a technical read-only in-memory session, not a participant login on the actual listener. It issued GET requests only. The real database had 0 users / 0 reference rows / 0 events before and after. This verifies routing and content, not browser display or a human's actions. All app/frontend hashes remained unchanged; no rebuild or scientific criteria changes were needed.

| Requirement | Executed technical check | Browser / human limitation |
|---|---|---|
'''
    browser+=''.join('| '+a+' | '+b+' | '+c+' |\n' for a,b,c in rows)
    browser+='\nAll 16 existing synthetic Python workflow tests and synthetic DOM/save tests passed again. They are not actual human annotations. See raw logs and FIRST_TWO_FROZEN_BINDINGS.json. Required next manual browser checks and real participant actions are in START_FIRST_TWO_VI.md.\n'
    (out/'BROWSER_INTERFACE_VERIFICATION_REPORT.md').write_text(browser,encoding='utf8')
    (out/'ORIGINAL_PHOTO_AVAILABILITY_REPORT.md').write_text('''# Original photo availability

PASS: all 1,136 frozen TEST100 photo records resolve to the exact declared local image file, match the preserved original-source SHA256 and size, return byte-identical JPEGs through the same source-serving handler, and decode fully. Missing/corrupt/substituted photos: 0. Selected reviews: 2,817 across exactly 100 frozen businesses. The per-photo ledger PHOTO_AVAILABILITY_1136.csv gives business ID, photo ID, manifest path, actual path, hashes, byte count and dimensions.

Served files are under `C:/thesis_proposal/yelp_dataset/data/processed/yelp_2300_v1/photos/`. These retained local copies match the original Yelp-byte hashes already frozen in the source manifests. The historical `original_source_path` may point to the old raw-image directory; no substitution is needed because the declared active `path` resolves correctly. The common photo scope remains ALL original indexed photos for each business, not U/F caption subsets. Nothing was regenerated, renamed or rebound.

No recovery is required. If future damage occurs, use the exact photo ID from the pinned original Yelp archive identified in SOURCE_PACKAGE/ANNOTATION_PACKAGE_MANIFEST.json and require the original_source_sha256 before restoring to its declared path. Do not change the frozen manifest, select a different photo or use a generated image. MISSING_OR_BAD_PHOTOS.json currently contains an empty list.

This is file/source/API validation. Actual browser display remains unverified because no browser is connected. No image-content summary or human visual verification was fabricated.
''',encoding='utf8')
    (out/'ACTUAL_HUMAN_PROGRESS_REPORT.md').write_text('''# Actual human progress

100 NOT_STARTED, 0 DRAFT, 0 SUBMITTED, 0 NEEDS_REVISION, 0 APPROVED. No genuine reference has yet been written or independently checked. No participant accounts, signatures or approvals were created by the assistant. The first two queued records remain NOT_STARTED. The other 98 are only scheduled; they are not completed work.

Selection is the first two entries of the original frozen annotation manifest. Schedule then follows the same manifest order in groups of 10 (last group 8). No model prediction or TEST performance informed this order. Use ACTUAL_PROGRESS_AND_BATCH_ORDER.csv for the frozen order and current snapshot; read the live application tracker or run the existing `verify` command for subsequent actual progress.

Human actions needed: coordinator verifies eligible distinct author/checker identities and blindness, participants choose their own credentials, author reads all selected reviews/photos and writes both English components with evidence, checker independently reviews the exact submitted version, disagreements are resolved and logged, checker approves only after a real check. Anyone previously exposed to evaluated TEST outputs/Teacher answers must disclose it and must not falsely attest blindness. A separate source-only OS account/machine remains the appropriate deployment boundary.

After authentic first-two approval, run the existing source/workflow validator and check each approved record's original content/approval/history bindings before scaling. Draft export is explicitly non-final. A final freeze requires all 100 independently approved references; two pilot approvals would not complete Stage L. No official Stage M scores or model-output Human Evaluation were produced.
''',encoding='utf8')
    (out/'ISSUES_AND_NEXT_ACTIONS.md').write_text('''# Issues and next actions

No blocking application/source defect was found. Application code, source manifests, photo identities and reference state are unchanged. No application patch or new annotation framework was introduced.

1. Browser verification is blocked by unavailable Computer Use native pipe (os error 2) and no connected browser. Open http://127.0.0.1:8765 manually and check source navigation/images. Backend and synthetic tests are separately verified; real browser behavior is not claimed.
2. Real human work has not started; there are no participant accounts. Register eligible real authors/checkers using the existing local CLI/helper. The assistant does not invent identities, passwords, author prose or approvals. Two different account IDs alone cannot prove two different eligible people.
3. A request to detach/restart the local server was automatically rejected by the execution layer with reason `blocked by policy`. It did not execute. The earlier live server was kept unchanged and still responded HTTP 200. If that session stops, use the foreground CMD launch sequence; keep its window open. This is a local process-launch restriction, not a model/GPU/photo failure. No permission bypass or alternate hidden launcher was attempted.

The only new utility is verify_stage_l_launch_l1_v1.py: a read-only source/HTTP audit and queue/progress ledger generator. It neither modifies scientific criteria nor writes a reference. package_stage_l_l1_handoff_v1.py writes compact reports and documentation snapshots. No new large source ZIP is needed.
''',encoding='utf8')
    snapshots=out/'DOCUMENTATION_BEFORE';snapshots.mkdir()
    progress='''### Stage L.1 annotation application launched — 2026-10-29

The accepted Stage L application is now launched locally at http://127.0.0.1:8765. All 100 frozen cases/2,817 selected reviews and all 1,136 original photo records were rechecked through the unchanged source-serving code; every photo matched its original SHA256, returned exact JPEG bytes and decoded. The actual listener serves HTML/JS/CSS and keeps source access login-protected. No source recovery or application patch was necessary. A detached-background launch was blocked by the execution policy; the original live server remains available and a foreground CMD restart sequence is supplied.

The first two entries of the frozen annotation manifest are queued without performance-based selection: Mungo's Italian Eatery (-78Va3llKcPbL6SW4is8lw; 30 reviews/8 photos) and The Bungalow Tampa (0-M3vGd6IveJh77-Bwc8EQ; 30 reviews/16 photos). Actual status remains 100 NOT_STARTED / 0 drafts / 0 submitted / 0 needs revision / 0 approved; both pilot records are unstarted. No real participant account or human action was fabricated. The remaining 98 are queued in manifest order in groups of 10 (last 8).

Sixteen existing synthetic workflow tests and synthetic DOM tests passed again. No real-browser visual smoke test was possible: Computer Use native pipe is unavailable (os error 2), and no browser is connected. This is disclosed separately from verified backend/image routing, not presented as browser success. Real authors/checkers must register their own accounts, inspect the sources, write/check both components and record authentic approvals. No model output, score or Teacher answer is served in the annotation interface; actual blindness additionally requires source-only OS/machine access and eligible people.

Reports and first-two instructions: `outputs/stage_l/test100_human_references_v1/LAUNCH_L1_20261029_v1/`. Stage L remains incomplete until all 100 human-written references are independently approved and integrity-validated. Stage M Automatic Evaluation remains pending, followed later by separate Human Evaluation. No model training, TEST regeneration, final metrics or new paid compute occurred. STOP for the actual human author/checker actions.

'''
    docs=['THESIS_PROGRESS.md','LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','CODEX_THESIS_IMPLEMENTATION_PLAN.md','CHATGPT_NEW_CONVERSATION_HANDOFF.md']
    changes=[]
    for name in docs:
        path=project/'thesis_sdd_docs'/name;oldbytes=path.read_bytes();(snapshots/name).write_bytes(oldbytes)
        old=oldbytes.decode('utf8');updated=progress+old;path.write_text(updated,encoding='utf8',newline='')
        (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),updated.splitlines(True),fromfile=name+'.before',tofile=name)),encoding='utf8')
        changes.append({'file':'thesis_sdd_docs/'+name,'before_sha256':hashlib.sha256(oldbytes).hexdigest(),'after_sha256':file_hash(path),'prior_text_retained':True})
    new_json(out/'DOCUMENTATION_UPDATE_SUMMARY.json',changes)
    (out/'DOCUMENTATION_UPDATE_SUMMARY.md').write_text('# Documentation update\n\nUpdated existing progress, roadmap, implementation plan and conversation handoff with the actual local launch, photo checks, first-two IDs, zero real human progress, browser-access limitation and pending Stage M/Human Evaluation. Prior text and exact before snapshots are retained locally. No failed-training history or new scientific annotation criterion was introduced.\n',encoding='utf8')
    (out/'STAGE_L1_PROGRESS_UPDATE.md').write_text(progress,encoding='utf8')
    (out/'LAUNCH_OR_RESTART_IN_CMD.txt').write_text('cd /d C:\\thesis_proposal\\yelp_dataset\r\n.venv-caption-pilot\\Scripts\\python.exe -X utf8 -B scripts\\stage_l_reference_app_v1.py serve --workspace outputs\\stage_l\\test100_human_references_v1\r\n\r\nOpen http://127.0.0.1:8765 . Keep CMD open. If the existing page works, do not start a duplicate server.\r\n',encoding='utf8')
    # Small reports/scripts only. Never include original reviews/images, model
    # predictions, historical scientific outcomes or the credential database.
    stage=out/'COMPACT_HANDOFF';stage.mkdir()
    for f in out.iterdir():
        if f.is_file() and f.suffix in ['.md','.json','.csv','.txt'] and not f.name.endswith('.diff'):
            shutil.copy2(f,stage/f.name)
    for name in ['verify_stage_l_launch_l1_v1.py','package_stage_l_l1_handoff_v1.py']:
        (stage/'scripts').mkdir(exist_ok=True);shutil.copy2(project/'scripts'/name,stage/'scripts'/name)
    (stage/'README.md').write_text('# Stage L.1 compact handoff\n\nStart with START_FIRST_TWO_VI.md. Actual human references/approvals remain zero. The server/source checks passed; browser visuals are unverified. This bundle contains only new audit/reporting scripts, compact evidence and queue/progress snapshots, not the unchanged source dataset or app. Existing Stage L workspace/scripts remain required. Credentials, private state, photos, predictions and scores are excluded.\n',encoding='utf8')
    manifest={f.relative_to(stage).as_posix():{'bytes':f.stat().st_size,'sha256':file_hash(f)} for f in sorted(stage.rglob('*')) if f.is_file()}
    new_json(stage/'PACKAGE_MANIFEST.json',manifest)
    archive=out/'STAGE_L1_LAUNCH_FIRST_TWO_COMPACT_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(stage.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(stage).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_l1_zip_') as td:
        dest=Path(td)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('ZIP corruption/duplicates')
            z.extractall(dest)
        if {f.relative_to(dest).as_posix() for f in dest.rglob('*') if f.is_file()}!=set(manifest)|{'PACKAGE_MANIFEST.json'}:raise ValueError('ZIP members')
        if read(dest/'PACKAGE_MANIFEST.json')!=manifest:raise ValueError('ZIP manifest')
        for rel,row in manifest.items():
            if file_hash(dest/rel)!=row['sha256'] or (dest/rel).stat().st_size!=row['bytes']:raise ValueError('ZIP member hash')
    receipt={'status':'PASS_REOPEN_EXTRACT_EVERY_MEMBER_SHA256','archive':archive.name,'bytes':archive.stat().st_size,'sha256':file_hash(archive),
        'payload_files':len(manifest),'zip_members':len(manifest)+1,'actual_human_approved':0,'browser_visual_verified':False,'verified_at_utc':now()}
    new_json(out/'COMPACT_PACKAGE_INTEGRITY_REPORT.json',receipt)
    (out/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
