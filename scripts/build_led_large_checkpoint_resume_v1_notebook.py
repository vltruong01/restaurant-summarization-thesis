"""Deliver checkpoint recovery as one notebook, reusing the unchanged v5 GPU runner."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_checkpoint250_resume_v1'
HANDOFF=ROOT/'outputs/cloud_handoff'
NAME='LED_LARGE_F_RESUME_CHECKPOINT250.ipynb'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build():
    original=json.loads((HANDOFF/'LED_LARGE_F_FULL_TRAIN_V5.ipynb').read_text(encoding='utf-8'))
    old=[c for c in original['cells'] if c['cell_type']=='code']
    approval=json.loads((PREP/'RECOVERY_AUTHORIZATION.json').read_text(encoding='utf-8'))
    rels=['scripts/stage_i_led_large_checkpoint_resume_v1.py','tests/test_led_large_checkpoint_resume_v1.py',
          'outputs/stage_i/led_large_checkpoint250_resume_v1/SOURCE_BINDINGS.json',
          'outputs/stage_i/led_large_checkpoint250_resume_v1/RECOVERY_AUTHORIZATION.json']
    raw=json.dumps(dict(files={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in rels},
                        hashes={n:sha(ROOT/n) for n in rels}),separators=(',',':')).encode()
    encoded=base64.b64encode(zlib.compress(raw,9)).decode()
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,NAME,'exec')
        cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('''# LED-large F — RESUME từ checkpoint-250

Notebook này tiếp tục lần train đã bị ngắt, **không train từ đầu**.
Dùng đúng T4 và folder MyDrive/LED-Large_multimodal hiện có. Giữ ZIP cũ
LED_LARGE_V3_ALL_IN_ONE.zip; không cần ZIP/JSON mới hay tự giải nén.

Khôi phục model + optimizer + scheduler + AMP + RNG từ checkpoint-250;
chạy đến tổng5epochs/1.250updates, không phải thêm5epochs.
28updates251–278 chưa có checkpoint sẽ được tính lại và ghi rõ trong ledger.
Run/log cũ giữ nguyên; kết quả tiếp tục vào runs/F_full_train_v5_resume_checkpoint250_v1.
Không chạy lại gate, không đổi data/LoRA/LR/beam4. Sau train sinh48outputs epoch3/5.

Yêu cầu tiếp tục checkpoint trong chat đã được ghi thành phê duyệt phục hồi riêng;
không cần tự tạo JSON hay duyệt lại checkbox. Đây không phải phê duyệt UI giả lập.
Deadline gốc giữ nguyên:2026-10-21T17:11:48UTC (=00:11:48ngày22/10giờViệtNam).
Nếu originalDrive đã có checkpoint mới hơn250 hoặc log thêm bước, notebook sẽ dừng
để tránh lùi về checkpoint cũ. Không chạy đồng thời notebook full-train cũ.
''')
    for i,title in enumerate(('Mount Drive','Khôi phục workspace từ ZIP cũ','Môi trường pinnedPython3.11',
                              'Khôi phục dependency lịch sử, không gate','Khôi phục v4code, không gate',
                              'Cài lại đúng v5 runner đã chạy trước đó')):
        md(f'## {i}. {title}');cells.append(old[i])
    md('## 6. Cài phần phục hồi nhỏ được nhúng — không sửa v5 runner/config')
    code('''import base64,zlib
raw=zlib.decompress(base64.b64decode(__PAYLOAD__))
assert hashlib.sha256(raw).hexdigest()==__HASH__
overlay=json.loads(raw)
for name,value in overlay['files'].items():
    p=PurePosixPath(name);data=base64.b64decode(value)
    assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
    assert hashlib.sha256(data).hexdigest()==overlay['hashes'][name]
    dest=ROOT/name
    assert not dest.exists() or checksum(dest)==overlay['hashes'][name], 'Recovery code differs: '+name
for name,value in overlay['files'].items():
    dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with dest.open('xb') as f:f.write(base64.b64decode(value))
print('Recovery installed; original v5 runner/config/bindings unchanged.')
'''.replace('__PAYLOAD__',repr(encoded)).replace('__HASH__',repr(hashlib.sha256(raw).hexdigest())))
    md('## 7. Kiểm tra offline và bindings cũ')
    checks=''.join(old[6]['source'])
    checks=checks.replace("'test_led_large_full_train_v5.py')", "'test_led_large_full_train_v5.py','test_led_large_checkpoint_resume_v1.py')")
    code(checks)
    md('## 8. Kiểm tra/tải backbone đúng SHA — không GPU/model execution')
    cells.append(old[7])
    md('''## 9. Kiểm tra checkpoint và tạo nhánh phục hồi riêng
Xác minh snapshot278updates và đủ9filecheckpoint250. Giữ nguyên bản gốc.
Log đang hoạt động của nhánh mới bắt đầu với prefix250; log đầy đủ278 bước và
28 bước bị superseded vẫn được giữ riêng. Đó là rollback đã công bố, không giấu log.
Cell này chỉ đọc/copy file; chưa chạy GPU. Không chạy lại nếu nhánh đã bắt đầu train.
''')
    code('''SOURCE_RUN=DRIVE/'runs/F_full_train_v5'
GATE_OUT=DRIVE/'runs/F_gradient_gate_v4'
RUN_OUT=DRIVE/'runs/F_full_train_v5_resume_checkpoint250_v1'
subprocess.run([str(PYTHON),'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import stage_i_led_large_full_train_v5 as v;import stage_i_led_large_checkpoint_resume_v1 as r;v.validate_gate(Path(sys.argv[1]));r.verify_source(Path(sys.argv[2]));print('Verified accepted gate and exact interrupted checkpoint250 source')",str(GATE_OUT),str(SOURCE_RUN)],cwd=ROOT,env=OFFLINE_ENV,check=True)
if not RUN_OUT.exists():
    subprocess.run([str(PYTHON),'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import stage_i_led_large_checkpoint_resume_v1 as r;print(r.prepare(Path(sys.argv[1]),Path(sys.argv[2])))",str(SOURCE_RUN),str(RUN_OUT)],cwd=ROOT,env=OFFLINE_ENV,check=True)
else:
    ledger=json.loads((RUN_OUT/'RECOVERY_LINEAGE.json').read_text())
    rows=[json.loads(x) for x in (RUN_OUT/'train/progress.jsonl').read_text().splitlines()]
    assert ledger['restored_checkpoint_step']==250 and ledger['original_last_logged_step']==278
    assert len(rows)==250 and rows[-1]['step']==250, 'Recovery đã tiến triển; không tự rollback/lặp lại'
    assert not (RUN_OUT/'train/RESUME_AUTHORIZATION_STEP_250.json').exists(), 'Recovery đã bắt đầu; không tự retry'
print('Verified recovery branch:',RUN_OUT)
''')
    md('''## 10. Ghi lại phê duyệt phục hồi đã có từ cuộc trò chuyện
Nguồn là yêu cầu trực tiếp của researcher tiếp tục từ checkpoint sau khi đã được
thông báo28updatesphải chạy lại. Không sửa timestamp/deadlinecũ và không gán nhãn UI.
V5runner/config/preflight và generation giữ nguyên; scope mới chỉ là phục hồi một lần.
''')
    code('''APPROVAL=DRIVE/'LED_LARGE_F_CHECKPOINT250_RESUME_APPROVAL.json'
record=json.loads(__APPROVAL__)
if APPROVAL.exists():
    assert json.loads(APPROVAL.read_text())==record, 'Existing recovery approval differs; STOP'
else:
    with APPROVAL.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write(chr(10))
subprocess.run([str(PYTHON),'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import stage_i_led_large_full_train_v5 as m;m.authorize(Path(sys.argv[1]),'train',True);print('Checkpoint250 recovery authorization PASS; original deadline unchanged')",str(APPROVAL)],cwd=ROOT,env=OFFLINE_ENV,check=True)
LAUNCH_LOGS=DRIVE/'runs/F_full_train_v5_resume_checkpoint250_launcher_logs';LAUNCH_LOGS.mkdir(parents=True,exist_ok=True)
RESUME_CHECKPOINT=RUN_OUT/'train/checkpoint-250'
def run_logged(mode):
    log=LAUNCH_LOGS/(mode+'.log')
    assert not log.exists(), 'Mode đã chạy; giữ log, không tự retry'
    cmd=[str(PYTHON),'scripts/run_stage_i_led_large_full_train_v5.py',mode,'--execute','--approval',str(APPROVAL),'--gate-output',str(GATE_OUT),'--output',str(RUN_OUT)]
    if mode=='train':cmd+=['--resume',str(RESUME_CHECKPOINT)]
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with log.open('x',encoding='utf-8') as f:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        status=process.wait()
    assert status==0, f'{mode} FAILED ({status}); dừng và chạy cellbackup cuối, không tự retry.'
'''.replace('__APPROVAL__',repr(json.dumps(approval,ensure_ascii=False))))
    md('''## 11. RESUME training từ250đến1.250 — còn4epochs
Trainer dùng resume_from_checkpoint: khôi phục adapter, AdamWmoments, scheduler,
AMPscale/growthtracker và RNG; epoch order bắt đầu ở epoch2. Không resetwarmup/LR.
Checkpoint250giữ nguyên; các checkpoint500/750/1000/1250 sẽ lưu trong nhánh recovery.
Nếu có lỗi/ngắt mới, giữ evidence và dừng, không chạy lại cell này.
''')
    cells.append(old[10])
    md('## 12. Đánh giá epoch3/5 — đúng48outputs, originalbeam4/max_length768')
    cells.append(old[11])
    md('## 13. Report/HTML và reviewZIP có thêm provenance phục hồi — CPU only')
    code('''subprocess.run([str(PYTHON),'scripts/report_stage_i_led_large_full_train_v5.py','--output',str(RUN_OUT)],cwd=ROOT,env=OFFLINE_ENV,check=True)
import shutil
review=RUN_OUT/'quality_review'
evidence=review/'recovery_provenance';evidence.mkdir(exist_ok=False)
for p in [RUN_OUT/'RECOVERY_LINEAGE.json',RUN_OUT/'train/RESUME_AUTHORIZATION_STEP_250.json',
          ROOT/'outputs/stage_i/led_large_checkpoint250_resume_v1/SOURCE_BINDINGS.json',
          RUN_OUT/'recovery_evidence/ORIGINAL_progress.jsonl']:
    shutil.copy2(p,evidence/p.name)
note='Interrupted original trajectory logged278updates;restored exactcheckpoint250;28uncheckpointedupdates recomputed in an explicitly authorized separate recovery branch. Original evidence preserved; deadline unchanged. D051strict replayFAIL unchanged. No exactCUDAreplay claim.'
with (review/'EXECUTION_AND_QUALITY_REPORT.md').open('a',encoding='utf-8') as f:f.write(chr(10)+chr(10)+'## Checkpoint recovery provenance'+chr(10)+note+chr(10))
htmlfile=review/'RESULTS_REVIEW.html'
text=htmlfile.read_text();text=text.replace('<h1>','<p>'+note+'</p><h1>',1);htmlfile.write_text(text,encoding='utf-8')
manifest={p.relative_to(review).as_posix():checksum(p) for p in review.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}
(review/'SHA256_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2),encoding='utf-8')
archive=RUN_OUT/'LED_LARGE_F_RESUMED_QUALITY_REVIEW.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(review.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(review).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
Path(str(archive)+'.sha256').write_text(checksum(archive)+'  '+archive.name+chr(10))
print('Review HTML:',htmlfile)
print('Gửi GPT:',archive)
from google.colab import files
files.download(str(archive))
''')
    md('''## 14. Backup sau hoàn tất hoặc khi lỗi — không GPU
Chỉ backup các file của recovery, không sửa/xóa run gốc. Chạy cell này nếu resume
bị ngắt; không tự chạy lại training hoặc sinh thêm outputs.
''')
    backup=''.join(old[13]['source'])
    backup=backup.replace("RUN_OUT=DRIVE/'runs/F_full_train_v5'", "RUN_OUT=DRIVE/'runs/F_full_train_v5_resume_checkpoint250_v1'")
    backup=backup.replace("LAUNCH_LOGS=DRIVE/'runs/F_full_train_v5_launcher_logs'", "LAUNCH_LOGS=DRIVE/'runs/F_full_train_v5_resume_checkpoint250_launcher_logs'")
    backup=backup.replace('LED_LARGE_F_FULL_TRAIN_V5_BACKUP_','LED_LARGE_F_RESUME_CHECKPOINT250_BACKUP_')
    backup=backup.replace("for rel in ('scripts/stage_i_led_large_full_train_v5.py'", "for rel in ('scripts/stage_i_led_large_checkpoint_resume_v1.py','outputs/stage_i/led_large_checkpoint250_resume_v1/SOURCE_BINDINGS.json','outputs/stage_i/led_large_checkpoint250_resume_v1/RECOVERY_AUTHORIZATION.json','scripts/stage_i_led_large_full_train_v5.py'")
    code(backup)
    for i,c in enumerate(cells):c['id']=f'resume-cp250-{i:02d}'
    original['cells']=cells;original['metadata']['colab']['name']=NAME
    path=HANDOFF/NAME
    path.write_text(json.dumps(original,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    receipt=dict(notebook=NAME,sha256=sha(path),existing_v5_GPU_runner_unchanged=True,
        source_checkpoint_integrity_sha256=approval['resume_checkpoint_integrity_sha256'],
        research_message_approval_embedded=True,original_deadline_unchanged=True,
        no_GPU_execution_during_preparation=True,code_cells=sum(x['cell_type']=='code' for x in cells))
    (PREP/'NOTEBOOK_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':build()
