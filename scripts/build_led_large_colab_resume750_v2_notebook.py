"""Build one self-contained notebook; keep v5 model runner and previous notebooks unchanged."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cloud_handoff/LED_LARGE_F_RESUME_CHECKPOINT750_V2.ipynb'

def build():
    original=json.loads((ROOT/'outputs/cloud_handoff/LED_LARGE_F_FULL_TRAIN_V5.ipynb').read_text(encoding='utf-8'))
    code=[c for c in original['cells'] if c['cell_type']=='code']
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def py(s):
        compile(s,str(OUT),'exec')
        cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('''# LED-large Colab — tiếp tục checkpoint750, bản sửa launcher v2

**Dùng notebook này thay notebook resume250 cũ. Không cần ZIP/data mới.**
Chạy lần lượt từ cell0. Các cell0–8 chỉ setup/kiểm tra CPU; không chạy gate hay train mới.
Notebook tự khôi phục workspace từ ZIP `LED_LARGE_V3_ALL_IN_ONE.zip` đã có trên Drive.

Đã nhìn thấy checkpoint250/500/750 trong Google Drive:
`runs/F_full_train_v5_resume_checkpoint250_v1/train/`.
Tên nhánh vẫn chứa250 nhưng đã có checkpoint mới. Notebook kiểm tra hash TẤT CẢ
checkpoint, optimizer/scheduler/scaler/RNG và receipts trước khi chuẩn bị phục hồi.
Nếu thiếu/hỏng file, log khác nhau, có checkpoint mới hơn750 hoặc numerical failure,
notebook dừng; không fallback về250, không sửa/xóa evidence.

Lượt cũ được báo dừng ở890. Phục hồi750 đồng nghĩa tính lại140 updates751–890;
28 updates của lần phục hồi trước được giữ trong lineage. Quỹ đạo cuối vẫn1250
steps/5epochs; số updates đã thực hiện trên các nhánh, nếu dừng đúng890, tối đa1418.
Trạng thái chính xác được đọc từ log890–999 khi chuẩn bị và in rõ trước cell train.
Output mới: `runs/F_full_train_v5_resume_checkpoint750_v2/`.

Giữ nguyên runner/config/preflight v5, T4, FP16/AMP ban đầu1024, dữ liệu, thứ tự,
LoRA, LR/scheduler và beam4. Không dùng runner dynamic65536 của Vast.
Restore đầy đủ adapter+optimizer+scheduler+scaler+RNG, không train từ đầu.
Deadline cũ: **2026-10-21T17:11:48UTC**, không gia hạn. D051 strict replay FAIL giữ nguyên.

Cell train/evaluate tự gọi launcher độc lập, không cần biến/hàm `run_logged`.
Nếu runtime reset, chạy setup rồi prepare lại: nhánh đã launch sẽ bị chặn,
không tự retry. Đóng notebook training cũ; không chạy hai trajectory đồng thời.
''')
    titles=('Mount đúng tài khoản Google Drive','Workspace từ ZIP cũ, không cần giải nén thủ công',
            'Python3.11 và dependencies pinned','Dependency lịch sử, không chạy gate',
            'Code v4 đã duyệt, không chạy gate','Runner v5 không đổi',
            'Offline tests và integrity preflight','Backbone chính xác, không GPU')
    for i,title in enumerate(titles):
        md(f'## {i}. {title}')
        c=copy.deepcopy(code[i]);c['outputs']=[];c['execution_count']=None;cells.append(c)
    files={}
    for name in ('scripts/stage_i_led_large_colab_resume750_v2.py','tests/test_led_large_colab_resume750_v2.py'):
        raw=(ROOT/name).read_bytes()
        files[name]=dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    payload=json.dumps(files,separators=(',',':')).encode()
    encoded=base64.b64encode(zlib.compress(payload,9)).decode()
    md('## 8. Cài helper phục hồi và chạy regression tests CPU — không sửa runner v5')
    py('''from pathlib import Path,PurePosixPath
import base64,zlib,json,hashlib,subprocess,os
ROOT=Path('/content/led_large_f_multimodal_v2')
PYTHON=ROOT/'.venv/bin/python'
assert PYTHON.is_file(), 'Chạy các cell0–7 của notebook này trước; chưa có workspace/runtime.'
raw=zlib.decompress(base64.b64decode(__PAYLOAD__))
assert hashlib.sha256(raw).hexdigest()==__DIGEST__
for name,item in json.loads(raw).items():
    p=PurePosixPath(name);data=base64.b64decode(item['data']);dest=ROOT/name
    assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
    assert hashlib.sha256(data).hexdigest()==item['sha256']
    assert not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256'], 'Helper khác hash; STOP'
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(data)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
subprocess.run([str(PYTHON),'-m','unittest','discover','-s','tests','-p','test_led_large_colab_resume750_v2.py','-v'],cwd=ROOT,env=env,check=True)
print('Helper và CPU regression PASS; v5 runner/config không thay đổi.')
'''.replace('__PAYLOAD__',repr(encoded)).replace('__DIGEST__',repr(hashlib.sha256(payload).hexdigest())))
    # Each later execution cell is self-contained: no run_logged or notebook closure dependency.
    def invocation(mode,cpu=False):
        return '''from pathlib import Path
import subprocess,os
ROOT=Path('/content/led_large_f_multimodal_v2')
PYTHON=ROOT/'.venv/bin/python'
DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')
assert PYTHON.is_file() and (ROOT/'scripts/stage_i_led_large_colab_resume750_v2.py').is_file(), 'Chạy các cell setup0–8 trước; không gọi training cũ.'
assert DRIVE.is_dir() and (DRIVE/'runs').is_dir(), 'Drive/runs chưa mount đúng; không tạo folder thay thế.'
env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
'''+("env['CUDA_VISIBLE_DEVICES']=''\n" if cpu else '')+f"subprocess.run([str(PYTHON),'-u','scripts/stage_i_led_large_colab_resume750_v2.py','{mode}','--drive',str(DRIVE)],cwd=ROOT,env=env,check=True)\n"
    md('''## 9. Xác minh gate/checkpoint750 và tạo nhánh CPU riêng
Copy nội dung file, không copy quyền/xattrs của Drive FUSE. Không model/GPU execution.
Hash source được kiểm tra lại sau copy. Giữ snapshot đầy đủ và lineage cũ.
Nếu cell lỗi: giữ folder/log; chưa chạy cell10. Không dùng cell9 resume250 cũ.
''')
    py(invocation('prepare',True))
    md('''## 10. Tiếp tục TRAIN — checkpoint750 →1250, còn2epochs
Chạy cell này khi cell9 PASS và đã đọc số updates cần tính lại được in ra.
Đây là tiếp tục lượt F đã được yêu cầu, không phải thí nghiệm/model mới.
Giữ checkpoint250/500/750; lưu thêm1000/1250. Không reset LR/warmup/scaler.
Cell tự gọi launcher; không cần `run_logged`. Chỉ chạy một lần.
Nếu lỗi/ngắt, chạy backup cuối; không chạy lại cell train.
''')
    py(invocation('train'))
    md('## 11. Sinh đúng48 validation summaries epoch3/5 — chỉ sau training complete')
    py(invocation('evaluate'))
    md('## 12. Báo cáo và portable reviewZIP kèm provenance phục hồi — CPU only')
    py('''from pathlib import Path
import subprocess,os,json,hashlib,zipfile,shutil
ROOT=Path('/content/led_large_f_multimodal_v2');PYTHON=ROOT/'.venv/bin/python'
DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')
RUN_OUT=(DRIVE/'runs/F_full_train_v5_resume_checkpoint250_v1').resolve().parent/'F_full_train_v5_resume_checkpoint750_v2'
assert RUN_OUT.is_dir(), 'Chưa có recovery branch'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
subprocess.run([str(PYTHON),'scripts/report_stage_i_led_large_full_train_v5.py','--output',str(RUN_OUT)],cwd=ROOT,env=env,check=True)
review=RUN_OUT/'quality_review';proof=review/'recovery_provenance';proof.mkdir(exist_ok=False)
for name in ('RECOVERY_LINEAGE.json','RECOVERY_APPROVAL.json'):
    with (RUN_OUT/name).open('rb') as a,(proof/name).open('xb') as b:shutil.copyfileobj(a,b)
ledger=json.loads((RUN_OUT/'RECOVERY_LINEAGE.json').read_text())
note=f"Restored checkpoint750 after logged step{ledger['original_last_logged_step']}; recomputed updates max={ledger['additional_recomputed_updates_max']}, prior recovery28 preserved. Deadline unchanged; no exact CUDA replay claim; D051 strict replay FAIL unchanged."
with (review/'EXECUTION_AND_QUALITY_REPORT.md').open('a',encoding='utf-8') as f:f.write(chr(10)+note+chr(10))
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest={p.relative_to(review).as_posix():digest(p) for p in review.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}
(review/'SHA256_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2),encoding='utf-8')
archive=RUN_OUT/'LED_LARGE_F_RESUME750_QUALITY_REVIEW.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(review.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(review).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
Path(str(archive)+'.sha256').write_text(digest(archive)+'  '+archive.name+chr(10))
print('HTML:',review/'RESULTS_REVIEW.html');print('ReviewZIP:',archive)
from google.colab import files
files.download(str(archive))
''')
    md('''## 13. Backup sau hoàn tất hoặc khi có lỗi/ngắt — không GPU
Backup ghi vào `/content` rồi download, không phụ thuộc quyền ghi Drive.
Bao gồm recovery/source snapshot, gate và helper; không sửa/xóa checkpoint.
Nếu mất runtime trước khi backup, toàn bộ checkpoint vẫn nằm trong nhánh trên Drive.
''')
    py('''from pathlib import Path
from datetime import datetime,timezone
import tarfile,hashlib,json
DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')
ROOT=Path('/content/led_large_f_multimodal_v2')
source=DRIVE/'runs/F_full_train_v5_resume_checkpoint250_v1'
output=source.resolve().parent/'F_full_train_v5_resume_checkpoint750_v2'
gate=DRIVE/'runs/F_gradient_gate_v4'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive=Path('/content')/('LED_LARGE_COLAB_RECOVERY750_'+stamp+'.tar.gz')
assert output.is_dir() or source.is_dir(), 'Drive chưa có run; remount trước'
with tarfile.open(archive,'x:gz') as t:
    if output.is_dir():t.add(output,arcname='runs/'+output.name,recursive=True)
    else:t.add(source,arcname='runs/'+source.name,recursive=True)
    if gate.is_dir():t.add(gate,arcname='runs/F_gradient_gate_v4',recursive=True)
    helper=ROOT/'scripts/stage_i_led_large_colab_resume750_v2.py'
    if helper.is_file():t.add(helper,arcname='scripts/'+helper.name)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+chr(10))
print(archive,digest)
from google.colab import files
files.download(str(archive))
''')
    for i,c in enumerate(cells):c['id']=f'resume750-v2-{i:02d}'
    original['cells']=cells
    original['metadata'].setdefault('colab',{})['name']=OUT.name
    OUT.write_text(json.dumps(original,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(OUT)

if __name__=='__main__':build()
