"""Write technical receipts/docs and a verified small source-only review ZIP."""
import difflib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import zipfile
from stage_l_reference_app_v1 import Store,new_json,file_hash,read,now

def main():
    project=Path.cwd();out=project/'outputs/stage_l/test100_human_references_v1';evidence=out/'EVIDENCE'
    tests=read(evidence/'TEST_RUN_RECEIPT.json')
    if not all(x['exit_code']==0 for x in tests):raise ValueError('Technical tests did not pass')
    store=Store(out);sources=store.verify_sources(True);workflow=store.audit()
    with store.connect() as c:
        counts={t:c.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ['users','refs','events']}
    if any(counts.values()):raise ValueError('Real human state changed; do not package private participant state')
    before=read(evidence/'PROTECTED_ARTIFACTS_BEFORE.json');changes=[p for p,h in before.items() if not (project/p).is_file() or file_hash(project/p)!=h]
    if changes:raise ValueError('Protected originals changed: '+str(changes))
    new_json(evidence/'PROTECTED_ARTIFACTS_AFTER_VERIFICATION.json',{'status':'PASS','files_individually_rehashed':len(before),'changed_or_missing':changes,
        'covers':'Original preparation sources/templates/locks and Stage K small outputs/evidence. No checkpoint or large runtime file was opened for modification.', 'checked_at_utc':now()})
    new_json(out/'REFERENCE_APPROVAL_TRACKER.json',workflow)
    new_json(out/'CURRENT_SOURCE_AND_REFERENCE_VALIDATION.json',{'sources':sources,'workflow':workflow,'private_state_row_counts':counts})
    report='''# Stage L technical workflow validation

Status: PASS for source integrity and tested functional contracts. Real-browser visual/layout and interactive keyboard QA remain UNVERIFIED because the available Computer Use surface reported no browser and no native pipe. Do not represent the API/DOM tests as a real-browser end-to-end run. The local HTTP server did serve the actual interface successfully; its source API required login.

## Actual source and human progress

100 frozen TEST businesses; 2,817 selected reviews; 1,136 indexed original photos. All source bundles match frozen canonical hashes. All 1,136 image byte sizes/SHA256 values match their preserved original-source SHA256; Pillow image verification succeeded for every image. Frozen photo scope and identifiers were not changed. Source-only workspace bytes match the previously approved package. No captions, Teacher targets, evaluated predictions or automatic scores enter the annotator interface.

Actual human-authored references: 0. Actual submitted references: 0. Actual independently approved references: 0. All 100 remain NOT_STARTED. No real participant identities/accounts are precreated. No human actions were fabricated. All workflow fixtures and simulated approvals used synthetic data in separate temporary workspaces, never TEST drafts; those are not counted as human work.

## Executed checks

16 Python tests passed: exact source/photo load; draft persistence and reopened state; submission, independent revision request and approval; valid evidence; stale-version conflicts; approved locking; role/same-author rejection; missing human attestations; invalid review/photo IDs and review quotes; embedded headings/empty fields; missing/malformed claim data; missing direct-image confirmation; tainted blank template; duplicate-reference detection; journal/state corruption detection; rejection history; incomplete final-freeze refusal; consistent database backup/reopen; source tampering; authenticated HTTP source/photo access and refusal of filesystem/Stage K paths; CSRF enforcement. The 100-case synthetic freeze additionally passed the existing original Stage M reference validator and every export member hash. Some methods cover multiple guards; these are 16 test methods, not 16 actual human annotations.

Synthetic JavaScript DOM contract tests passed: review/photo controls populate; exact evidence builder; save receipt/version; edits typed while a save is in flight remain recoverable; invalid JSON retains browser recovery; submitted text locks; CSRF header. JavaScript syntax check passed. This harness has a simulated DOM and HTTP responses, not a browser engine.

Source verification, live HTTP smoke, test output, original-preservation rehashes and package hashes are supplied in EVIDENCE/. No GPU process, inference, model update, automatic reference prose or final TEST metric was invoked. Existing source/checkpoint/decoder/mapping and Stage K files are unchanged.

## Implementation fixes made during synthetic testing

Windows cleanup tests revealed that sqlite context-manager commit did not close database handles; explicit close was added and backup handles were fixed. A save-in-flight regression test detected potential browser draft loss; sequence tracking now retains later edits, prevents navigation before their save and keeps recovery on failure. Failed development-test evidence is retained separately; final tests pass. No genuine annotations were used or rewritten to test fixes.

## Human and deployment limitations

An account and a checked checkbox do not authenticate a real person or prove semantic correctness. A coordinator must register and verify eligible distinct authors/checkers, maintain the private roster and ensure neither has seen evaluated outputs/Teacher answers. The source-only web routes prevent interface exposure; they do not stop a person with full administrator disk access from opening other project folders. Use the portable source-only deployment or a separate OS account/machine for actual blind work. Keep credentials, database backups and human drafts private.

Before scaling to all 100 cases, a researcher should open the supplied local interface and verify normal display/navigation/photo zoom with their browser; this unperformed visual check is explicit. All writing and checking remain genuine human tasks. Once all 100 are approved, the final export revalidates sources, evidence IDs/quotes, version/approval hashes, distinct human IDs, chronology, duplicate full references, journal consistency and original Stage M compatibility. It does not certify factuality, complete coverage or identity authenticity. No final frozen reference manifest exists yet. Stage M and separate Human Evaluation remain pending.
'''
    (out/'TECHNICAL_WORKFLOW_VALIDATION_REPORT.md').write_text(report,encoding='utf8')
    (out/'STAGE_M_HANDOFF.md').write_text('''# Stage M handoff preparation

Stage L exports one shared reference per original TEST business. Final export is blocked until all 100 records have genuine independent human approval and pass integrity checks. No final freeze exists yet. The existing scorer `scripts/score_s1000_automatic_prepared_v1.py` and Stage K compatibility wrapper are unchanged and were not executed on TEST.

Use the final export's ANNOTATIONS directory as the existing scorer's --references argument. It retains original field names (`review_consensus`, `visual_evidence`, claim evidence, author/checker provenance, history and source binding). APPROVED_REFERENCES.json, REFERENCE_FREEZE.json and the clean REFERENCES_INSPECTION.json retain content and exact file hashes. Human prose is never rewritten. Readable export adds required section markers outside the plain component fields. The two original components and common human reference are reused for all applicable comparisons.

Separate researcher release is required before Stage M metrics. Review comparisons cover RO/U/F/S; Visual covers U/F/S only. RO has no missing-Visual penalty. Preserve pre-outcome metric/statistical pins, BERTScore's disclosed 512-token prefix, full-text ROUGE, paired business comparisons and confidence intervals. Human model-output evaluation starts only after Stage M, with the previously frozen design. Neither metric code nor evaluated output files are distributed to reference participants.
''',encoding='utf8')
    quick='''# Bắt đầu Stage L

Đã có giao diện và nguồn thật: 100 nhà hàng, 2.817 review, 1.136 ảnh gốc đã kiểm hash. Chưa có reference do người viết hoặc duyệt: 0/100.

1. Chạy `REGISTER_PARTICIPANT.cmd` trong thư mục này để đăng ký từng người thật: một tài khoản vai trò `author`, một tài khoản `checker`, khác người. Mỗi người chọn ID và mật khẩu riêng (ít nhất 12 ký tự). Không có tài khoản mẫu được tạo sẵn.
2. Chạy `START_ANNOTATION.cmd`, rồi mở http://127.0.0.1:8765 . Nếu server đang chạy từ phiên kiểm tra, dừng bằng Ctrl+C trước khi chạy lại.
3. Tác giả chọn nhà hàng, đọc tất cả review, mở ảnh gốc, viết hai phần tiếng Anh, liên kết từng nhận định với review/ảnh và lưu. Bản nháp tự lưu; chỉ rời trang sau khi thấy thông báo đã lưu. Nhấn Submit khi hoàn tất.
4. Người kiểm đăng nhập bằng tài khoản khác, đối chiếu nguồn, ghi yêu cầu sửa hoặc duyệt đúng phiên bản đã kiểm. Không được đánh dấu xác nhận nếu chưa thực sự làm.
5. Sao lưu database định kỳ bằng lệnh trong README_SETUP_VI.md. Xuất bản nháp được ngay; xuất reference cuối chỉ được khi đủ 100 bản đã duyệt hợp lệ.

Người viết và người kiểm không được xem output RO/U/F/S, điểm Stage K hoặc Teacher targets. Tốt nhất dùng máy/tài khoản hệ điều hành riêng, chỉ nhận bộ nguồn Stage L. Nếu đã xem output thì phải khai báo, không xác nhận sai về tính độc lập. Hai tài khoản không tự chứng minh hai người khác nhau: người điều phối phải kiểm tra.

Đã test luồng chức năng qua API và logic giao diện bằng dữ liệu tổng hợp. Công cụ browser của phiên này không kết nối được, nên kiểm tra hiển thị/zoom bằng trình duyệt thật vẫn cần thực hiện trước khi viết hàng loạt. Không có reference TEST giả nào được tạo để test. Chưa chạy ROUGE/BERTScore TEST hay Human Evaluation.
'''
    (out/'BAT_DAU_TAI_DAY_VI.md').write_text(quick,encoding='utf8')
    current='''### Stage L reference workflow ready for human annotation — 2026-10-29

Stage K TEST100 inference is complete and frozen. Stage L source-only annotation workspace is implemented at `outputs/stage_l/test100_human_references_v1/`: 100 businesses, 2,817 selected reviews and all 1,136 original-byte-equivalent photos were verified against frozen identifiers/hashes. Local UI supports source navigation, separate English Review/Visual drafting, exact evidence links, durable versioned saves, author submission, independent checker revision requests/rejection/approval and progress/export. No predictions, Teacher answers or automatic scores are served to annotators. Real independence requires eligible distinct humans and a separate source-only machine/OS profile, not merely different account IDs.

Actual human progress remains 0 authored / 0 submitted / 0 independently approved; 100 NOT_STARTED. No real participant accounts or final reference freeze have been fabricated. Sixteen synthetic Python test methods and synthetic DOM/save regression tests pass, including the original Stage M export contract. Actual source/HTTP checks pass. Real-browser visual/layout QA is unverified because the browser/native UI connection is unavailable; a researcher must check display/navigation before full annotation. Automated integrity is not factuality or coverage certification.

Next: register eligible human author(s) and different checker(s), independently write/check all 100 common photo-grounded references, resolve disagreements and freeze the validated export. Stage M reference-based Automatic Evaluation remains pending a separate release; RO has Review only, U/F/S have both applicable components. Human Evaluation of model outputs remains a separate later stage after Stage M. No training, new inference, final metrics, source changes or cloud charges occurred during Stage L implementation. Existing methodological limitations/provenance remain unchanged. STOP for researcher review of the implemented workflow.

'''
    docdir=evidence/'DOCUMENTATION_BEFORE';docdir.mkdir()
    docs=['THESIS_PROGRESS.md','LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','CODEX_THESIS_IMPLEMENTATION_PLAN.md','CHATGPT_NEW_CONVERSATION_HANDOFF.md']
    doc_changes=[]
    for name in docs:
        path=project/'thesis_sdd_docs'/name;raw=path.read_bytes();(docdir/name).write_bytes(raw)
        old=raw.decode('utf8');updated=current+old;path.write_text(updated,encoding='utf8',newline='')
        diff=''.join(difflib.unified_diff(old.splitlines(True),updated.splitlines(True),fromfile=name+'.before',tofile=name))
        (evidence/(name+'.diff')).write_text(diff,encoding='utf8')
        doc_changes.append({'file':str(path.relative_to(project)),'before_sha256':__import__('hashlib').sha256(raw).hexdigest(),'after_sha256':file_hash(path),'original_text_retained':True})
    decisions=project/'DECISIONS.md';raw=decisions.read_bytes();(docdir/'DECISIONS.md').write_bytes(raw)
    addition='\n\n## 2026-10-29 Stage L human reference construction\n\nImplemented the source-only TEST100 author/checker workbench and synthetic technical tests, reusing frozen sources and original reference validators. Actual human-authored/submitted/approved counts remain 0/0/0. Browser visual QA remains unverified; all sources and functional API/DOM contracts passed. Different eligible real humans must write/check references without TEST predictions, scores or Teacher answers. No final reference freeze, Stage M metric or human output scoring was created. See outputs/stage_l/test100_human_references_v1/TECHNICAL_WORKFLOW_VALIDATION_REPORT.md. STOP for researcher review.\n'
    with decisions.open('ab') as f:f.write(addition.encode('utf8'))
    doc_changes.append({'file':'DECISIONS.md','before_sha256':__import__('hashlib').sha256(raw).hexdigest(),'after_sha256':file_hash(decisions),'original_bytes_preserved_as_prefix':decisions.read_bytes().startswith(raw)})
    new_json(evidence/'DOCUMENTATION_CHANGE_MANIFEST.json',doc_changes)
    package_review(project,out,current)

def package_review(project,out,current):
    evidence=out/'EVIDENCE'
    stage=out/'REVIEW_PACKAGE';stage.mkdir()
    def copy_file(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    scripts=['stage_l_reference_app_v1.py','stage_l_reference_web_v1/index.html','stage_l_reference_web_v1/app.js','stage_l_reference_web_v1/app.css',
        'prepare_stage_l_reference_workspace_v1.py','configure_stage_l_portable_v1.py','test_stage_l_reference_workflow_v1.py','test_stage_l_reference_frontend_v1.cjs',
        's1000_final_eval_core_v1.py','validate_s1000_human_references_v1.py','snapshot_s1000_reference_version_v1.py','materialize_reference_photos_v1.py','finalize_stage_l_reference_delivery_v1.py']
    for rel in scripts:copy_file(project/'scripts'/rel,stage/'scripts'/rel)
    for item in out.iterdir():
        if item.name in ['PRIVATE_STATE','REVIEW_PACKAGE','PACKAGE_INTEGRITY_REPORT.json'] or item.name.endswith(('.zip','.sha256')):continue
        if item.is_file():copy_file(item,stage/'WORKSPACE'/item.name)
        elif item.name in ['SOURCE_PACKAGE','INITIAL_DRAFT_EXPORT']:shutil.copytree(item,stage/'WORKSPACE'/item.name)
        elif item.name=='EVIDENCE':
            # Historical thesis snapshots contain Stage K outcomes. Retain those
            # locally; never distribute them in this source-only review bundle.
            for f in item.iterdir():
                if f.is_file() and not f.name.endswith('.diff'):copy_file(f,stage/'WORKSPACE/EVIDENCE'/f.name)
    (stage/'UPDATED_THESIS_DOCS').mkdir()
    (stage/'UPDATED_THESIS_DOCS/STAGE_L_PROGRESS_UPDATE.md').write_text(current,encoding='utf8')
    copy_file(evidence/'DOCUMENTATION_CHANGE_MANIFEST.json',stage/'UPDATED_THESIS_DOCS/DOCUMENTATION_CHANGE_MANIFEST.json')
    (stage/'README.md').write_text('''# Stage L source-only human reference workflow review

Start with WORKSPACE/BAT_DAU_TAI_DAY_VI.md and WORKSPACE/README_SETUP_VI.md. Current real references: 0 authored, 0 approved. Technical test fixtures are synthetic only. No model output, Teacher answer, score, image archive or credential database is included.

For independent participant deployment, transfer verified original photos separately, then from this extracted root run:
```
python scripts/configure_stage_l_portable_v1.py --bundle WORKSPACE --photos PATH_TO_VERIFIED_PHOTOS --destination NEW_PARTICIPANT_WORKSPACE
python scripts/stage_l_reference_app_v1.py add-user --workspace NEW_PARTICIPANT_WORKSPACE --id REAL_AUTHOR_ID --role author
python scripts/stage_l_reference_app_v1.py add-user --workspace NEW_PARTICIPANT_WORKSPACE --id REAL_CHECKER_ID --role checker
python scripts/stage_l_reference_app_v1.py serve --workspace NEW_PARTICIPANT_WORKSPACE
```
Use Python 3.11+ and Pillow. See the guidelines for actual human independence and evidence rules. This package has the full 100-case exact review/source manifest and blank template workspace; photos are linked and must be supplied privately, never inferred from captions. Read WORKSPACE/TECHNICAL_WORKFLOW_VALIDATION_REPORT.md for the explicit real-browser QA limitation. Stage M and human model-output evaluation were not run. PACKAGE_MANIFEST.json binds every payload; archive integrity receipt is delivered separately.
''',encoding='utf8')
    manifest={str(p.relative_to(stage)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(stage.rglob('*')) if p.is_file()}
    new_json(stage/'PACKAGE_MANIFEST.json',manifest)
    archive=out/'STAGE_L_TEST100_HUMAN_REFERENCE_WORKFLOW_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage).as_posix())
    with tempfile.TemporaryDirectory() as td:
        dest=Path(td)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC failure')
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('ZIP duplicates')
            z.extractall(dest)
        actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
        if actual!=set(manifest)|{'PACKAGE_MANIFEST.json'}:raise ValueError('Missing/extra members')
        for rel,row in manifest.items():
            if file_hash(dest/rel)!=row['sha256'] or (dest/rel).stat().st_size!=row['bytes']:raise ValueError('Member hash mismatch '+rel)
        if read(dest/'PACKAGE_MANIFEST.json')!=manifest:raise ValueError('Manifest changed')
        # Prove the source-only package can reconstruct the actual frozen source frame.
        extracted=read(dest/'WORKSPACE/SOURCE_PACKAGE/ANNOTATION_PACKAGE_MANIFEST.json')
        if len(extracted['frozen_sources'])!=100:raise ValueError('Portable cohort mismatch')
    receipt={'status':'PASS_ZIP_CRC_EXTRACT_AND_EVERY_MEMBER_SHA256','archive':archive.name,'archive_bytes':archive.stat().st_size,
        'archive_sha256':file_hash(archive),'payload_files':len(manifest),'zip_members':len(manifest)+1,'businesses':100,'selected_reviews':2817,'original_photos_indexed_not_embedded':1136,
        'private_state_excluded':True,'model_predictions_teacher_scores_excluded':True,'actual_human_approved':0,'real_browser_visual_qa':'UNVERIFIED','verified_at_utc':now()}
    new_json(out/'PACKAGE_INTEGRITY_REPORT.json',receipt)
    (out/'STAGE_L_TEST100_HUMAN_REFERENCE_WORKFLOW_REVIEW.zip.sha256').write_text(receipt['archive_sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
