"""Offline Drive-folder delivery wrapper; frozen GPU runner/config remain byte-identical."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / 'outputs/cloud_handoff'
DEST = HANDOFF / 'LED-Large_multimodal'
ZIP_NAME = 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip'
ZIP_SHA = '91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f'
APPROVAL_NAME = 'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json'
APPROVAL_SHA = 'ca3b06f9647f9cfa1888ee3fd29ba11cc1c9b91db08cdca33a234c3535491ef2'
NOTEBOOK_NAME = 'LED_LARGE_MULTIMODAL_F_FULL_TRAIN.ipynb'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
        f.write('\n')


def build():
    assert not DEST.exists(), 'Preserve existing delivery; do not overwrite'
    assert sha(HANDOFF / ZIP_NAME) == ZIP_SHA
    assert sha(HANDOFF / APPROVAL_NAME) == APPROVAL_SHA
    DEST.mkdir()
    for name in (ZIP_NAME, ZIP_NAME + '.sha256', APPROVAL_NAME,
                 'LED_LARGE_F_V2_RESEARCHER_APPROVAL_VALIDATION.json'):
        shutil.copy2(HANDOFF / name, DEST / name)
    nb = json.loads((HANDOFF / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb').read_text(encoding='utf-8'))
    nb['metadata']['colab']['name'] = NOTEBOOK_NAME
    codes = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(codes) == 11
    codes[1]['source'] = '''from pathlib import Path, PurePosixPath
import hashlib, zipfile, stat, json

# All upload files belong in this single existing Drive folder.
DRIVE_DIR = Path('/content/drive/MyDrive/LED-Large_multimodal')
archive = DRIVE_DIR / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip'
receipt = DRIVE_DIR / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip.sha256'
APPROVAL = DRIVE_DIR / 'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json'
RUN_OUT = DRIVE_DIR / 'runs/F_run_v2'
EXPECTED_ZIP_SHA256 = '91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f'
EXPECTED_APPROVAL_SHA256 = 'ca3b06f9647f9cfa1888ee3fd29ba11cc1c9b91db08cdca33a234c3535491ef2'
missing = [p.name for p in (archive, receipt, APPROVAL) if not p.is_file()]
assert not missing, f'Thieu file trong {DRIVE_DIR}: {missing}. Upload cac file sau khi giai nen goi UPLOAD; khong chi upload ZIP ngoai.'
parts = receipt.read_text(encoding='utf-8').split()
assert len(parts) == 2 and parts[0] == EXPECTED_ZIP_SHA256 and parts[1] == archive.name, 'Wrong V2 checksum receipt; STOP'
with archive.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == EXPECTED_ZIP_SHA256, 'ZIP hash mismatch; STOP'
with APPROVAL.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == EXPECTED_APPROVAL_SHA256, 'Approval bytes differ from delivered researcher approval; STOP'
approval = json.loads(APPROVAL.read_text(encoding='utf-8'))
assert approval.get('GPU_execution_approved') is True, 'Researcher approval missing'
ROOT = Path('/content/led_large_f_multimodal_v2')
assert not ROOT.exists(), 'Existing extracted directory: STOP and inspect before reuse; no overwrite or silent restart'
with zipfile.ZipFile(archive) as z:
    seen = set()
    for info in z.infolist():
        p = PurePosixPath(info.filename)
        assert info.filename not in seen, 'Duplicate ZIP member'
        seen.add(info.filename)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in info.filename
        assert chr(92) not in info.filename and not stat.S_ISLNK(info.external_attr >> 16)
    z.extractall(ROOT)
print('Upload files and hashes PASS. Workspace:', ROOT)
print('Persistent outputs:', RUN_OUT)
print('Existing approval deadline (not reset):', approval['execution_deadline_utc'])
'''.splitlines(True)
    src = ''.join(codes[5]['source'])
    assert "APPROVAL=Path('/content/drive/MyDrive/LED_LARGE_F_V2_RESEARCHER_APPROVAL.json')" in src
    src = src.replace("APPROVAL=Path('/content/drive/MyDrive/LED_LARGE_F_V2_RESEARCHER_APPROVAL.json')",
                      "APPROVAL=DRIVE_DIR/'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json'")
    src = src.replace("RUN_OUT=Path('/content/drive/MyDrive/thesis_led_large/F_run_v2')",
                      "RUN_OUT=DRIVE_DIR/'runs/F_run_v2'")
    # Approval checker is import-only: confirms the unchanged execution hashes and clock before GPU.
    src = src.replace("LAUNCH_LOGS=", "import importlib.util\n_spec=importlib.util.spec_from_file_location('led_large_approval_check',ROOT/'scripts/stage_i_led_large_runtime_v2.py')\n_runtime=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_runtime)\nfor _mode in ('feasibility','train','evaluate'):\n    _runtime.authorize(APPROVAL,_mode,True)\nprint('Approval, original deadline and execution hashes PASS')\nLAUNCH_LOGS=", 1)
    codes[5]['source'] = src.splitlines(True)
    nb['cells'][0]['source'] = '''# LED-large Filtered — complete notebook for LED-Large_multimodal
Upload ALL files extracted from LED-Large_multimodal_UPLOAD_V1.zip into the existing
MyDrive/LED-Large_multimodal folder, then open this notebook in Colab and select T4.
Do not upload the outer ZIP alone or create a second nested LED-Large_multimodal folder.
This notebook uses the unchanged, hash-bound v2 runner/config/data and actual researcher approval.
Setup -> offline preflight -> exact public model download -> feasibility -> conditional fresh F
training -> 48 validation outputs -> HTML/ChatGPT review bundle. No RO/U or held-out access.
The previous approval deadline is unchanged; it is never reset by running this notebook.
Stop on any failure. This notebook has been checked offline; actual T4 feasibility is unmeasured.
'''.splitlines(True)
    for c in nb['cells']:
        if c['cell_type'] == 'markdown':
            s = ''.join(c['source'])
            if s.startswith('## 4.'):
                s = '''## 4. Existing researcher approval and persistent output directory
The actual conversation approval is included next to the data ZIP in LED-Large_multimodal.
No need to create or edit an approval JSON. The historical false template inside the frozen
payload is preserved for provenance; the separate signed-by-provenance approval is used here.
This cell checks its original deadline and hash bindings; no clock extension or GPU operation.
Outputs/checkpoints go to MyDrive/LED-Large_multimodal/runs/F_run_v2.
If an earlier run has already started elsewhere, STOP and report its logs before launching
another run: this delivery does not authorize duplicate training or an automatic restart.
'''
            c['source'] = s.splitlines(True)
        else:
            compile(''.join(c['source']), NOTEBOOK_NAME, 'exec')
            c['outputs'] = []; c['execution_count'] = None
    save(DEST / NOTEBOOK_NAME, nb)
    save(DEST / 'README_START_HERE_VI.md', '''# LED-large F — gói upload đầy đủ

## Chỉ cần làm như sau

1. Giải nén LED-Large_multimodal_UPLOAD_V1.zip trên máy tính.
2. Mở thư mục LED-Large_multimodal vừa giải nén. Upload toàn bộ các FILE bên trong
   vào thư mục Drive đã tạo: MyDrive/LED-Large_multimodal.
   Không tạo thêm một thư mục LED-Large_multimodal lồng bên trong.
3. Không giải nén ZIP dữ liệu V2 bên trong trên Drive. Notebook sẽ kiểm tra và giải nén
   nó vào bộ nhớ cục bộ Colab, tránh đọc 9.898 file dữ liệu nhỏ trực tiếp từ Drive.
4. Mở LED_LARGE_MULTIMODAL_F_FULL_TRAIN.ipynb bằng Google Colab.
   Chọn Runtime > Change runtime type > T4 GPU. Chạy từng cell từ đầu.
5. Quy trình: mount Drive -> kiểm tra đủ ZIP/checksum/approval -> cài Python 3.11 riêng
   -> 42 offline tests và full-data preflight -> tải đúng public LED-large
   -> kiểm tra approval -> GPU feasibility -> chỉ khi PASS mới training -> evaluation -> report.
6. Khi có lỗi, dừng, giữ log và gửi traceback. Không tự sửa precision/batch/token budget,
   bỏ kiểm tra, bấm lại gate/training hoặc xóa output để chạy lại.

## Các file bắt buộc cùng thư mục trên Drive

- LED_LARGE_MULTIMODAL_F_FULL_TRAIN.ipynb
- LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip (dữ liệu và toàn bộ runner v2, giữ nguyên)
- LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip.sha256 (checksum V2, không dùng V1)
- LED_LARGE_F_V2_RESEARCHER_APPROVAL.json (phê duyệt thực tế đã ghi nhận)

Các file JSON kiểm chứng và README còn lại cũng nên upload cùng để đối chiếu.
Không cần API key Teacher, API key Codex hoặc config.toml.

## Phạm vi và thời hạn đã duyệt

Giữ nguyên v2: 228 modules /4.915.200 trainable parameters; F 2.000 train/200 validation;
gate <=3 updates/24 microbatches, <=2 train-input generation memory probes, >=15% VRAM
headroom; chỉ khi PASS mới fresh train <=5 epochs/1.250 updates, warmup63, batch1/accum8,
FP32 parameters/FP16 AMP; lưu cả5 checkpoints; sinh48 validation outputs từ epoch3/5.
Không train RO/U, không dùng test, không Teacher calls. Không thay frozen data hay method.
Phê duyệt ghi nhận: 2026-10-19T17:11:48+00:00.
Hạn dừng giữ nguyên: 2026-10-21T17:11:48+00:00 = 00:11:48 ngày22/10/2026 Việt Nam.
Notebook không reset đồng hồ. Nếu hết hạn, dừng và báo; không tự sửa approval JSON.
Nếu gate/training đã khởi chạy ở notebook cũ, gửi log trước khi dùng gói này;
gói upload không cho phép chạy trùng hoặc tự restart.

## Dữ liệu và kết quả

Bao gồm frozen F JSONL/NPZ cho 2.200 businesses, tokenizer, Teacher evidence records,
7.578 ảnh gốc, references D053,24 validation cases và historical benchmarks.
Public LED-large weights ~1,84GB được tải bằng downloader kiểm chứng trong notebook;
không đính kèm lại trọng số tải từ Hugging Face.
Checkpoint/log: MyDrive/LED-Large_multimodal/runs/F_run_v2.
Kết thúc: gửi LED_LARGE_F_QUALITY_REVIEW.zip trong run folder cho GPT/researcher.
Chất lượng thực tế và khả năng fit T4 chưa được xác nhận bằng lần chạy GPU này.

## Tính toàn vẹn

Đây chỉ là bản đóng gói/đường dẫn mới. Scientific config/preflight/runner/helper v2 và
approval giữ nguyên bytes/hashes. Manifest ngoài kiểm chứng notebook, ZIP và approval.
False templates bên trong là lịch sử chuẩn bị, không thay thế phê duyệt riêng.
Không sửa D051 strict replay FAIL hoặc bất kỳ evidence lịch sử nào.
''')
    files = {p.name: sha(p) for p in sorted(DEST.iterdir()) if p.is_file()}
    save(DEST / 'DELIVERY_SHA256_MANIFEST.json', dict(
        files=files, scientific_payload_unchanged=True, GPU_executed=False,
        delivery_only=True, notebook=NOTEBOOK_NAME,
        Drive_directory='/content/drive/MyDrive/LED-Large_multimodal',
        original_execution_deadline_utc='2026-10-21T17:11:48+00:00'))
    out = HANDOFF / 'LED-Large_multimodal_UPLOAD_V1.zip'
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(DEST.iterdir()):
            z.write(p, 'LED-Large_multimodal/' + p.name)
    digest = sha(out)
    save(Path(str(out) + '.sha256'), digest + '  ' + out.name)
    print(json.dumps(dict(archive=str(out), bytes=out.stat().st_size,
                          sha256=digest, files=len(list(DEST.iterdir()))), indent=2))


if __name__ == '__main__':
    build()
