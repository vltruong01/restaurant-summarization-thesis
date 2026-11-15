"""One notebook reuses the existing single ZIP and accepted v4 gate; no GPU execution."""
import base64
import difflib
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
H = ROOT/'outputs/cloud_handoff'
PREP = ROOT/'outputs/stage_i/led_large_full_train_v5_preparation'
WORK = H/'led_large_full_train_colab_verification_v2/extracted'
NAME = 'LED_LARGE_F_FULL_TRAIN_V5.ipynb'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False))
        f.write('\n')


def build():
    # Unpublished offline preparation may be rebuilt; no accepted or executed files are touched.
    PREP.mkdir(parents=True,exist_ok=True)
    files = ['scripts/stage_i_led_large_full_train_v5.py','scripts/run_stage_i_led_large_full_train_v5.py',
             'scripts/report_stage_i_led_large_full_train_v5.py','configs/stage_i_led_large_full_train_v5.json',
             'tests/test_led_large_full_train_v5.py']
    inherited = ['scripts/stage_i_led_large_runtime_v2.py','scripts/run_stage_i_led_large_colab_v2.py',
                 'scripts/report_stage_i_led_large_colab_v2.py','configs/stage_i_led_large_colab_v2.json',
                 'outputs/stage_i/led_large_colab_preparation_v2/PREFLIGHT_MANIFEST.json',
                 'scripts/run_stage_i_led_large_gradient_gate_v4.py','configs/stage_i_led_large_gradient_gate_v4.json',
                 'outputs/stage_i/led_large_gradient_gate_v4_preparation/PREFLIGHT_MANIFEST.json']
    manifest = {rel:sha(ROOT/rel if (ROOT/rel).exists() else WORK/rel) for rel in files+inherited}
    save(PREP/'PREFLIGHT_MANIFEST.json',dict(files=manifest,original_frozen_preflight_retained=True,
                                          accepted_v4_gate_reused_read_only=True,new_GPU_gate=False))
    bindings = dict(config_sha256=sha(ROOT/files[3]),runner_sha256=sha(ROOT/files[1]),
                    preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'))
    save(PREP/'EXECUTION_BINDINGS.json',bindings)
    plan = '''# LED-large F full training v5 — ready notebook

Use LED_LARGE_F_FULL_TRAIN_V5.ipynb with the SAME LED_LARGE_V3_ALL_IN_ONE.zip
in MyDrive/LED-Large_multimodal. No additional ZIP, receipt or approval JSON upload.
The notebook installs its small versioned code/config overlay automatically.

## Exact proposed GPU scope
Read and verify existing F_gradient_gate_v4 PASS, full gate checkpoint hashes,
original source/model/runtime bindings and the SAME T4 hardware/runtime profile.
No repeat gate. Fresh official LED-large/fresh Expanded-LoRA; never gate adapters.
F2000train/200validation;228modules/4,915,200parameters;r8/alpha16/dropout0;
FP32 stored weights/FP16 AMP, init_scale1024/growth2/backoff0.5/interval2000.
All optimization, order, labels, token budgets and generation settings unchanged.
Five epochs/1250updates,warmup63,AdamW2e-4,physicalbatch1/accum8,all5checkpoints.
No early stopping or best-loss reload. Exactly48 validation predictions:epochs3/5
each24fixedcases(random20,targeted4),beam4/max_length768. Original sources,
captions/photos, Teacher targets and approved D053 references remain frozen.

## Approval and clocks
Preparation is not GPU authorization. In the notebook, the researcher reads the
printed exact hashes/scope and changes RESEARCHER_APPROVES_F_TRAINING to True.
This explicit action records a notebook-cell authorization with actual UTC time,
not a fabricated historical conversation approval. No external JSON file needed.
Original authorization2026-10-19T17:11:48Z and execution deadline
2026-10-21T17:11:48Z remain unchanged. Expired deadline stops before GPU execution;
renewal never silently extends the original clock. No paid infrastructure launched.

## Minimal differences from v2
Versioned bindings, read-only acceptance of exact v4 gate instead of rerunning v2;
public GradScaler.load_state_dict sets the existing Trainer/Accelerate scaler to
1024 before any backward. Same object, optimizer integration and checkpoint API.
Callback retains finite-gradient/skip/update/checkpoint/memory guards and uses v5
bindings. Extra AMP initialization and accepted gate receipts recorded.
Generation loop and optimizer/scheduler implementation unchanged; no decoding change.
Hardware is checked against the gate before model loading. No automatic resume.

## Use
Select Colab T4 and run cells in order. Existing Drive gate directory must still
contain all four checkpoint files; they are verified directly, so another gate
evidence export is not a prerequisite. Reuse the existing data ZIP without manual
unzip. Fresh Colab kernels restore the isolated pinned Python3.11 workspace.
Run CPU checks, public-model hash check and gate validation; approve exact scope
in the clearly marked cell; train; generate48; export HTML/metrics/reviewZIP.
Checkpoints and logs are stored directly on Drive. The final backup cell works
after success or failure and includes every file suffix, including safetensors.
On a failure, stop and preserve evidence; do not rerun train/generation cells.
An interruption does not authorize rollback or automatic resumption.

## Planning and interpretation
Gate stress-update timestamps suggest ~13hours for1250updates alone; full200-case
validation at each epoch,48beam4generations and overhead add time. This is a coarse
extrapolation, not a runtime guarantee; Colab disconnect limits remain a practical risk.
Actual training with this versioned Trainer path has NOT run during preparation.
The accepted v4 gate is evidence of bounded feasibility, not a quality GO.
After48outputs,stop for independent source-based research review. Semantic repetition,
Corecoverage,opinionbalance,factual/visual grounding and completeness require review;
software flags/loss/header improvements alone are insufficient. Human labels remain blank.
No RO/U/test/Teacher calls or automatic follow-up. D051strict replayFAIL preserved.
'''
    save(PREP/'READINESS.md',plan)
    diff=''.join(difflib.unified_diff((ROOT/'scripts/run_stage_i_led_large_colab_v2.py').read_text().splitlines(True),
                                  (ROOT/files[1]).read_text().splitlines(True),fromfile='frozen_v2_runner',tofile='full_train_v5_runner'))
    save(PREP/'RUNNER_DIFF.patch',diff)
    overlay_files = files+[str((PREP/n).relative_to(ROOT)).replace('\\','/') for n in
                           ('PREFLIGHT_MANIFEST.json','EXECUTION_BINDINGS.json','READINESS.md')]
    payload=dict(files={rel:base64.b64encode((ROOT/rel).read_bytes()).decode() for rel in overlay_files},
                 hashes={rel:sha(ROOT/rel) for rel in overlay_files})
    raw=json.dumps(payload,separators=(',',':')).encode()
    encoded=base64.b64encode(zlib.compress(raw,9)).decode()
    old=json.loads((H/'LED_LARGE_V4_ALL_IN_ONE_APPROVED.ipynb').read_text(encoding='utf-8'))
    oldcode=[c for c in old['cells'] if c['cell_type']=='code']
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,NAME,'exec')
        cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('''# LED-large — FULL F training and evaluation v5

Notebook đầy đủ: setup → kiểm tra dữ liệu/model và gate v4 đã PASS → train F
5 epochs → sinh48 summaries epoch3/5 → HTML/ZIPreview → backup.
**Dùng lại LED_LARGE_V3_ALL_IN_ONE.zip trong MyDrive/LED-Large_multimodal.**
Không cần tải thêm ZIP/JSON hay tự giải nén. Các bản code nhỏ được nhúng trong notebook.
Gate v4 KHÔNG chạy lại. Training khởi tạo model/LoRA mới, không dùng adapter gate.
Chọn T4; chạy lần lượt. Cell phê duyệt có một biến cần bạn xác nhận trước training.
AMPscale1024;giữ nguyên phương pháp/data/beam4. Không train RO/U hay dùng held-out.
Deadline gốc:2026-10-21 17:11:48UTC (=2026-10-22 00:11:48 giờ Việt Nam),không tự gia hạn.
Thời gian có thể trên13giờ cộng validation/generation; giữ đủ thời gian trước deadline.
''')
    for i in range(5):
        md(('## 0. Mount Drive','## 1. Tự khôi phục workspace từ ZIP hiện có',
            '## 2. Môi trường Python3.11 đã ghim','## 3. Giữ nguyên các dependency lịch sử — không chạy gate',
            '## 4. Kiểm tra/khôi phục code gate v4 — không thực thi GPU')[i])
        cells.append(oldcode[i])
    md('## 5. Cài code training v5 được nhúng, giữ nguyên các file khoa học')
    code('''import base64,zlib
raw=zlib.decompress(base64.b64decode(__PAYLOAD__))
assert hashlib.sha256(raw).hexdigest()==__SHA__, 'Embedded training overlay differs'
overlay=json.loads(raw)
decoded={name:base64.b64decode(data) for name,data in overlay['files'].items()}
assert set(decoded)==set(overlay['hashes'])
for name,data in decoded.items():
    p=PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
    assert hashlib.sha256(data).hexdigest()==overlay['hashes'][name]
    assert not (ROOT/name).exists() or checksum(ROOT/name)==overlay['hashes'][name], 'Changed v5 file: '+name
for name,data in decoded.items():
    dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with dest.open('xb') as f:f.write(data)
print('Training code ready; original data, model and old runners preserved.')
'''.replace('__PAYLOAD__',repr(encoded)).replace('__SHA__',repr(hashlib.sha256(raw).hexdigest())))
    md('## 6. Offline tests + full integrity preflight; không GPU')
    code('''OFFLINE_ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for pattern in ('test_led_large_v4_amp_candidate.py','test_led_large_full_train_v5.py'):
    subprocess.run([str(PYTHON),'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=ROOT,env=OFFLINE_ENV,check=True)
subprocess.run([str(PYTHON),'-c',"import sys;sys.path.insert(0,'scripts');import stage_i_led_large_full_train_v5 as m;m.preflight();print('Full training preflight PASS')"],cwd=ROOT,env=OFFLINE_ENV,check=True)
print((ROOT/'outputs/stage_i/led_large_full_train_v5_preparation/EXECUTION_BINDINGS.json').read_text())
''')
    md('## 7. Tải/kiểm tra đúng backbone công khai (nếu đã có thì kiểm tra SHA, không tải lại)')
    cells.append(oldcode[6])
    md('''## 8. Dùng lại gate v4 PASS trên Drive — không chạy lại
Xác minh trực tiếp cả safetensors/state/config/README của checkpoint, report và scope.
ZIP export trước bị thiếu safetensors không ảnh hưởng bước này nếu Drive vẫn giữ file gốc.
Nếu thư mục gate gốc mất/đổi, dừng; không tự chạy gate mới.
''')
    code('''GATE_OUT=DRIVE/'runs/F_gradient_gate_v4'
RUN_OUT=DRIVE/'runs/F_full_train_v5'
subprocess.run([str(PYTHON),'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import stage_i_led_large_full_train_v5 as m;r=m.validate_gate(Path(sys.argv[1]));print('Accepted v4 gate PASS:',r['GPU'],'headroom=',r['headroom_fraction'])",str(GATE_OUT)],cwd=ROOT,env=OFFLINE_ENV,check=True)
print('Fresh training output:',RUN_OUT)
''')
    md('''## 9. Xác nhận bắt đầu training F — chỉ sửa một biến
Đọc scope và hashes in ở trên. Đổi **RESEARCHER_APPROVES_F_TRAINING=False** thành
**True** khi bạn đồng ý train/evaluate đúng bản v5 này. Đây là hành động phê duyệt
trực tiếp của bạn; notebook ghi nguồn `researcher_notebook_cell` và UTC thực tế,
không giả lập phê duyệt cũ hoặc yêu cầu bạn tạo file JSON. Cấu hình chuẩn bị vẫn
giữ trạng thái chưa duyệt. Deadline gốc không được reset/gia hạn.
Scope:1freshF/2000train+200val/5epochs1250updates,48outputs24×epochs3/5,
AMP1024,>=15%headroom,không gate mới/RO/U/Teacher/held-out. Không tự retry.
''')
    code('''RESEARCHER_APPROVES_F_TRAINING=False  # Bạn đổi thành True để duyệt scope/hashes đã in.
assert RESEARCHER_APPROVES_F_TRAINING is True, 'Chờ bạn đặt biến xác nhận thành True; chưa chạy GPU'
from datetime import datetime,timezone
end=datetime.fromisoformat('2026-10-21T17:11:48+00:00')
assert datetime.now(timezone.utc)<end, 'Deadline gốc đã hết; dừng, không tự gia hạn'
APPROVAL=DRIVE/'LED_LARGE_F_FULL_TRAIN_V5_APPROVAL.json'
bindings=json.loads((ROOT/'outputs/stage_i/led_large_full_train_v5_preparation/EXECUTION_BINDINGS.json').read_text())
record=dict(experiment='LED_LARGE_F_FULL_TRAIN_V5',GPU_execution_approved=True,training_authorized=True,
    approval_source='researcher_notebook_cell',researcher_explicit_authorization='Researcher set RESEARCHER_APPROVES_F_TRAINING=True for the printed hash-bound full F scope',
    execution_authorized_at_utc=datetime.now(timezone.utc).isoformat(),execution_deadline_utc=end.isoformat(),
    original_authorization_utc='2026-10-19T17:11:48+00:00',defense_reserve_at_least_14_days_confirmed=True,
    scopes=['train','evaluate'],max_training_updates=1250,evaluation_outputs_max=48,new_gate_updates=0,
    new_gate_generation_calls=0,accepted_gate_report_sha256='3bfbf2248b676c3fa568af77f205905370390b946d33af15aa2b49301d6d51cc',
    amp_scaler=dict(init_scale=1024.0,growth_factor=2.0,backoff_factor=.5,growth_interval=2000),
    held_out_access=False,Teacher_calls=False,RO_U_training_authorized=False,resume_after_interruption_approved=False,**bindings)
if not APPROVAL.exists():
    assert not RUN_OUT.exists(), 'Output đã tồn tại; không cho phép tự ghi đè/restart'
    with APPROVAL.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write(chr(10))
else:
    print('Retaining original v5 approval timestamp; no clock reset.')
subprocess.run([str(PYTHON),'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import stage_i_led_large_full_train_v5 as m;m.authorize(Path(sys.argv[1]),'train',True);print('Full F approval verified')",str(APPROVAL)],cwd=ROOT,env=OFFLINE_ENV,check=True)
LAUNCH_LOGS=DRIVE/'runs/F_full_train_v5_launcher_logs';LAUNCH_LOGS.mkdir(parents=True,exist_ok=True)
def run_logged(mode):
    log=LAUNCH_LOGS/(mode+'.log')
    assert not log.exists(), 'Mode đã chạy trước đó; giữ log và dừng, không tự retry'
    cmd=[str(PYTHON),'scripts/run_stage_i_led_large_full_train_v5.py',mode,'--execute','--approval',str(APPROVAL),'--gate-output',str(GATE_OUT),'--output',str(RUN_OUT)]
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with log.open('x',encoding='utf-8') as f:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        status=process.wait()
    assert status==0, f'{mode} FAILED ({status}). Dừng; chạy cell backup cuối. Không retry/fallback.'
''')
    md('''## 10. FULL TRAIN F — 5 epochs/1.250 updates
Không tiếp tục từ gate. Model và optimizer/scheduler/scaler được khởi tạo mới.
Checkpoint-250/500/750/1000/1250 nằm trên Drive. Giữ notebook/runtime hoạt động;
không chạy lại cell nếu Colab ngắt hoặc báo lỗi. Loss/repetition chưa phải qualityGO.
''')
    code('''run_logged('train')
training=json.loads((RUN_OUT/'train/COMPLETION.json').read_text())
assert training['steps']==1250 and training['epochs']==5
print('Training complete:',training)
''')
    md('## 11. Sinh đúng48 validation summaries — epoch3 và5, cùng24cases, originalbeam4')
    code('''run_logged('evaluate')
evaluation=json.loads((RUN_OUT/'evaluate/COMPLETION.json').read_text())
assert evaluation['generation_calls']==48
print('48 predictions saved:',evaluation)
''')
    md('## 12. Báo cáo/HTML và ChatGPT reviewZIP — CPU only, không tự gán nhãn human')
    code('''subprocess.run([str(PYTHON),'scripts/report_stage_i_led_large_full_train_v5.py','--output',str(RUN_OUT)],cwd=ROOT,env=OFFLINE_ENV,check=True)
print('HTML:',RUN_OUT/'quality_review/RESULTS_REVIEW.html')
print('Review ZIP:',RUN_OUT/'LED_LARGE_F_QUALITY_REVIEW.zip')
from google.colab import files
files.download(str(RUN_OUT/'LED_LARGE_F_QUALITY_REVIEW.zip'))
''')
    md('''## 13. Backup đầy đủ sau khi xong HOẶC khi lỗi — không GPU
Nếu có lỗi ở cell train/generate, chỉ chạy cell này để giữ evidence. Không gọi train lại.
Backup tất cả suffixes và checkpointweights, optimizer, scaler, RNG, logs, code/config
bindings và gate receipts. Data/photos có sẵn trong ZIP cũ nên không nhân bản ở backup.
''')
    code('''from datetime import datetime,timezone
RUN_OUT=DRIVE/'runs/F_full_train_v5'
LAUNCH_LOGS=DRIVE/'runs/F_full_train_v5_launcher_logs'
assert RUN_OUT.exists() or LAUNCH_LOGS.exists(), 'Chưa có run/log để backup'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=DRIVE/f'LED_LARGE_F_FULL_TRAIN_V5_BACKUP_{stamp}.zip'
manifest={}
with zipfile.ZipFile(backup,'x',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for base,prefix in [(RUN_OUT,'run'),(LAUNCH_LOGS,'launcher_logs')]:
        for p in sorted(base.rglob('*')):
            if p.is_file():
                name=prefix+'/'+p.relative_to(base).as_posix();z.write(p,name);manifest[name]=checksum(p)
    for p in sorted((ROOT/'outputs/stage_i/led_large_full_train_v5_preparation').rglob('*')):
        if p.is_file():
            name='preparation/'+p.name;z.write(p,name);manifest[name]=checksum(p)
    for rel in ('scripts/stage_i_led_large_full_train_v5.py','scripts/run_stage_i_led_large_full_train_v5.py','scripts/report_stage_i_led_large_full_train_v5.py','configs/stage_i_led_large_full_train_v5.json'):
        z.write(ROOT/rel,rel);manifest[rel]=checksum(ROOT/rel)
    if 'APPROVAL' in globals() and APPROVAL.is_file():
        z.write(APPROVAL,'RESEARCHER_APPROVAL.json');manifest['RESEARCHER_APPROVAL.json']=checksum(APPROVAL)
    z.writestr('SHA256_MANIFEST.json',json.dumps(dict(files=manifest),indent=2))
with zipfile.ZipFile(backup) as z:assert z.testzip() is None
Path(str(backup)+'.sha256').write_text(checksum(backup)+'  '+backup.name+chr(10))
print('Backup saved:',backup)
print('Đã dừng. Gửi quality reviewZIP/evidence cho researcher; không tự train RO/U.')
''')
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),colab=dict(name=NAME)))
    for i,c in enumerate(cells):c['id']=f'full-f-v5-{i:02d}'
    save(H/NAME,nb)
    save(PREP/'NOTEBOOK_RECEIPT.json',dict(notebook=NAME,notebook_sha256=sha(H/NAME),same_existing_data_zip=True,
                                        gate_reexecution=False,training_executed_during_preparation=False,
                                        explicit_notebook_cell_approval_required=True,bindings=bindings))
    print(json.dumps(dict(notebook=str(H/NAME),bindings=bindings,code_cells=sum(c['cell_type']=='code' for c in cells)),indent=2))


if __name__=='__main__':
    build()
