"""Package the complete prepared F notebook/runner; offline tests only, never GPU execution."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/cloud_handoff/led_large_f_colab_v1'
DEST = ROOT/'outputs/cloud_handoff/led_large_f_colab_v2'
NEWPREP = 'outputs/stage_i/led_large_colab_preparation_v2'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False));f.write('\n')


def notebook():
    cells = []
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,'<full_colab_cell>','exec')
        cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('''# LED-large — FULL Filtered training notebook v2
Đây là notebook đầy đủ: setup → preflight → tải checkpoint → bounded GPU gate →
train F 5 epochs → sinh 48 outputs epoch3/5 → quality-review ZIP + backup.
**Chưa được thực thi/kiểm tra GPU. Không train RO/U.**
Phần GPU chỉ chạy khi có researcher approval JSON đúng hash bindings, timestamp và deadline.
Không dùng approval cũ D-059/D-060/UL. Không thay đổi precision/batch/độ dài để vượt lỗi.
Tải ZIP + `.zip.sha256` lên MyDrive. Chọn GPU T4 trong Colab khi đã được duyệt.
Đọc README_FIRST.md và EXECUTION_READINESS.md trước khi chạy.''')
    code('''from google.colab import drive
drive.mount('/content/drive')
''')
    code('''from pathlib import Path, PurePosixPath
import hashlib, zipfile, stat
archive = Path('/content/drive/MyDrive/LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip')
receipt = Path(str(archive)+'.sha256')
with archive.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest() == receipt.read_text().split()[0], 'ZIP hash mismatch; STOP'
ROOT = Path('/content/led_large_f_colab_v2')
assert not ROOT.exists(), 'Existing extracted directory: inspect before reuse; no overwrite'
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        p=PurePosixPath(info.filename)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in info.filename
        assert chr(92) not in info.filename and not stat.S_ISLNK(info.external_attr >> 16)
    z.extractall(ROOT)
print(ROOT)
''')
    md('''## 1. Môi trường riêng Python 3.11
Kernel Colab không bị thay Torch. Chỉ chạy các script bằng venv bên dưới.
Nếu cài/import/pin lỗi, dừng và gửi log; không tự nâng dependencies.''')
    code('''import subprocess, sys, os, json
subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
subprocess.run([sys.executable,'-m','uv','venv','--python','3.11','--seed',str(ROOT/'.venv')],check=True)
PYTHON=ROOT/'.venv/bin/python'
subprocess.run([str(PYTHON),'-m','pip','install','torch==2.1.0+cu118','--index-url','https://download.pytorch.org/whl/cu118'],check=True)
subprocess.run([str(PYTHON),'-m','pip','install','-r',str(ROOT/'requirements/led_large_colab_core.txt')],check=True)
subprocess.run([str(PYTHON),'-m','pip','check'],check=True)
(ROOT/'transfer_runs').mkdir(exist_ok=True)
with (ROOT/'transfer_runs/installed_freeze.txt').open('w') as f:
    subprocess.run([str(PYTHON),'-m','pip','freeze'],stdout=f,check=True)
''')
    md('''## 2. Kiểm tra toàn bộ dữ liệu và phần mềm — CPU/meta only
2.000 train/200 validation, nguồn/nhãn/token IDs, references, ảnh và runtime pins.
Kiểm tra 228 modules /4.915.200 parameters dùng META, không forward model.''')
    code('''OFFLINE_ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for pattern in ('test_led_large_colab_transfer_v1.py','test_stage_i_led_large_preparation_v1.py','test_led_large_full_notebook_v2.py'):
    subprocess.run([str(PYTHON),'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=ROOT,env=OFFLINE_ENV,check=True)
subprocess.run([str(PYTHON),'scripts/verify_led_large_colab_bundle_v2.py','--runtime','--tokenize','--report','transfer_runs/COLAB_PREFLIGHT_V2.json'],cwd=ROOT,env=OFFLINE_ENV,check=True)
''')
    md('''## 3. Tải đúng trọng số công khai — không GPU/model execution
Khoảng 1,84GB. Revision và SHA-256 cố định. Lỗi checksum hoặc partial download thì dừng.
Không cần Teacher API key; không tải Amazon checkpoint hoặc adapter LED-base.''')
    code('''subprocess.run([str(PYTHON),'scripts/download_led_large_pinned_v1.py','--download'],cwd=ROOT,check=True)
''')
    md('''## 4. Phê duyệt và nơi lưu bền vững
Mẫu chưa duyệt: outputs/stage_i/led_large_colab_preparation_v2/GPU_AUTHORIZATION_TEMPLATE_NOT_APPROVED.json.
Gửi EXECUTION_BINDINGS.json + EXECUTION_READINESS.md để researcher duyệt scope/hashes/deadline.
Chỉ sau khi được duyệt, lưu approval JSON thực tế tại đường dẫn dưới đây.
Giữ timestamp thực và deadline tối đa48h, không sau2026-10-26T00:00:00Z; không reset lịch cũ.
Đây không phải checkbox tự phê duyệt; Codex chưa gán human authorization.
Checkpoints/logs sẽ ghi trực tiếp vào MyDrive để giữ lại khi Colab ngắt.''')
    code('''APPROVAL=Path('/content/drive/MyDrive/LED_LARGE_F_V2_RESEARCHER_APPROVAL.json')
RUN_OUT=Path('/content/drive/MyDrive/thesis_led_large/F_run_v2')
assert APPROVAL.is_file(), 'Chưa có phê duyệt GPU đúng scope/hash; STOP trước GPU'
approval=json.loads(APPROVAL.read_text())
assert approval.get('GPU_execution_approved') is True, 'Chưa được duyệt GPU'
LAUNCH_LOGS=RUN_OUT.parent/(RUN_OUT.name+'_launcher_logs')
LAUNCH_LOGS.mkdir(parents=True,exist_ok=True)
def run_logged(mode, extra=()):
    command=[str(PYTHON),'scripts/run_stage_i_led_large_colab_v2.py',mode,'--execute','--approval',str(APPROVAL),'--output',str(RUN_OUT),*extra]
    log=LAUNCH_LOGS/(mode+'.log')
    assert not log.exists(), 'Log trước đã tồn tại; không tự retry/ghi đè'
    with log.open('x') as f:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        status=process.wait()
    assert status==0, f'{mode} FAILED ({status}); STOP, preserve logs; no fallback'
''')
    md('''## 5. Gate GPU — chạy trước train
3 updates/24 microbatches CE, checkpoint roundtrip (không replay update), kiểm tra gradients/
AMP/masks/frozen backbone và2 train-input beam4 memory traces. Yêu cầu>=15% memory headroom.
Gate thất bại: không chạy cell train. Gate adapters không dùng làm khởi tạo train.
Không chạy lại gate trên output cũ. Khi quay lại một gate đã PASS sau ngắt Colab,
cần kiểm tra report/hash/hardware và phê duyệt, không dùng Run all để tự lặp.''')
    code('''run_logged('feasibility')
gate=json.loads((RUN_OUT/'feasibility/REPORT.json').read_text())
assert gate['status']=='PASS' and gate['headroom_fraction']>=0.15
print('Gate PASS:',gate['GPU'],gate['headroom_fraction'])
''')
    md('''## 6. Full-data F training
Fresh pretrained LED-large + fresh Expanded-LoRA;2.000/200;5 epochs/1.250 updates;
warmup63;batch1/accum8;AdamW2e-4;FP32 parameters + FP16 AMP;save cả5 epoch checkpoints.
Không early stop/best-loss reload; không train RO/U. In-memory model gate đã được bỏ
vì từng mode chạy trong subprocess riêng.
Colab disconnect không tự cho phép retry. Mặc định training mới. Resume chỉ từ epoch
checkpoint nguyên vẹn, đúng hash và phê duyệt riêng, không rollback updates đã ghi log.''')
    code('''RESUME_CHECKPOINT=None  # mặc định fresh. Không đặt đường dẫn nếu chưa được duyệt resume.
extra=() if RESUME_CHECKPOINT is None else ('--resume',str(RESUME_CHECKPOINT))
run_logged('train',extra)
trained=json.loads((RUN_OUT/'train/COMPLETION.json').read_text())
assert trained['steps']==1250 and trained['epochs']==5
print(trained)
''')
    md('''## 7. Sinh đúng48 outputs đánh giá
Epoch3:24 cases. Epoch5:cùng24 cases. Originalbeam4/do_sample=False/max_length768.
Chỉ validation development20random+4targeted, không held-out. Giữ nguyên raw output,
không repetition guard/NLI/MMR hoặc sửa câu thủ công.''')
    code('''run_logged('evaluate')
evaluation=json.loads((RUN_OUT/'evaluate/COMPLETION.json').read_text())
assert evaluation['generation_calls']==48
print(evaluation)
''')
    md('''## 8. Báo cáo và gói review — CPU only
Xuất software metrics, HTML nguồn/Teacher/references/ảnh/outputs/benchmarks, annotations
trống cho48 outputs và portable ChatGPT quality ZIP. Không tự gán nhãn human/quality GO.
Semantic repetition, Core coverage, opinion balance, grounding/visual usefulness cần review.''')
    code('''subprocess.run([str(PYTHON),'scripts/report_stage_i_led_large_colab_v2.py','--output',str(RUN_OUT)],cwd=ROOT,env=OFFLINE_ENV,check=True)
print('Gửi GPT:',RUN_OUT/'LED_LARGE_F_QUALITY_REVIEW.zip')
print('Xem:',RUN_OUT/'quality_review/RESULTS_REVIEW.html')
''')
    md('''## 9. Backup toàn bộ run (tùy chọn, chỉ sau khi kết thúc)
Quality-review ZIP nhỏ không chứa optimizer checkpoints. Run folder trên Drive đã
giữ checkpoint đầy đủ. Cell này lưu backup riêng cùng SHA-256, không train thêm.''')
    code('''MAKE_FULL_BACKUP=False
if MAKE_FULL_BACKUP:
    backup=RUN_OUT.parent/(RUN_OUT.name+'_COMPLETE_BACKUP.zip')
    assert not backup.exists(), 'Preserve existing backup'
    with zipfile.ZipFile(backup,'x',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for p in sorted(RUN_OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(RUN_OUT).as_posix())
    with backup.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    Path(str(backup)+'.sha256').write_text(digest+'  '+backup.name+chr(10))
    print(backup,digest)
''')
    md('''## DỪNG
Gửi quality-review ZIP cho researcher. Không tự train RO/U hoặc thử model/decoding khác.
GO cần cải thiện thực chất, được researcher xác nhận; loss/header/repetition alone chưa đủ.
Giữ LED-base matched RO/U/F baseline đến khi đủ matched LED-large family.
D051 strict replay FAIL và UL numerical feasibility failure không bị sửa/diễn giải lại.''')
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),
        colab=dict(name='LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb')),cells=cells)


def payload_files():
    return [p for p in sorted(DEST.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']


def main():
    assert not DEST.exists(), 'Fresh versioned namespace required'
    DEST.mkdir()
    replaced={'README_FIRST.md','CODEX_HANDOFF.md','TRANSFER_CONTRACT.json','SHA256_MANIFEST.json','LED_LARGE_F_COLAB_V1.ipynb'}
    original=read(SOURCE/'SHA256_MANIFEST.json')['files']
    def cp_old(rel):
        if rel in replaced:return
        src=SOURCE/rel;dest=DEST/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        assert sha(src)==original[rel], 'Original v1 payload changed'
        shutil.copy2(src,dest)
    with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(cp_old,original))
    added=['scripts/stage_i_led_large_runtime_v2.py','scripts/run_stage_i_led_large_colab_v2.py',
           'scripts/report_stage_i_led_large_colab_v2.py','scripts/verify_led_large_colab_bundle_v2.py',
           'scripts/build_led_large_full_train_colab_v2.py','scripts/stage_i_decoding_diagnostics.py',
           'tests/test_led_large_full_notebook_v2.py','configs/stage_i_led_large_colab_v2.json',
           NEWPREP+'/INITIALIZATION_HASHES.json']
    for rel in added:
        p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
    # Live project logs are copied as documentation snapshots, not treated as frozen scientific data.
    for rel in ('DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md'):
        shutil.copy2(ROOT/rel,DEST/rel)
    contract=read(SOURCE/'TRANSFER_CONTRACT.json')
    contract.update(package='LED_LARGE_F_FULL_TRAIN_COLAB_V2',created_utc=datetime.now(timezone.utc).isoformat(),
        training_runner_status='COMPLETE_PREPARED_LARGE_F_RUNNER_GPU_EXECUTION_NOT_TESTED_OR_AUTHORIZED',
        notebook='LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb',historical_training_code='Reference only; notebook invokes the new large-specific runner',
        full_training_notebook=True,training_authorized=False,GPU_execution_approved=False)
    write(DEST/'TRANSFER_CONTRACT.json',contract)
    write(DEST/'LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb',notebook())
    write(DEST/'README_FIRST.md','''# LED-large F — FULL training notebook v2

V1 had only setup/preflight. V2 adds the actual prepared large-specific feasibility,
training, evaluation, persistent checkpoints and quality-bundle export code.
No GPU execution has occurred; GPU compatibility is NOT claimed from offline tests.

## Included
Unchanged F2000train/200validation JSONL/NPZ,7578original photos,2200whole Teacher
provenance records, frozen tokenizer, approved D053 references,24fixed development
cases and saved historicalF/D058/D059 benchmarks. No held-out contents or credentials.
Whole Teacher records retain historical RO/U text; no RO/U execution dataset is present.
The public1.84GB official pretrained model is downloaded/hash-checked on Colab.

## Use
1. Upload this ZIP and .zip.sha256 to MyDrive; open LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb.
2. Run setup + CPU/meta tests and full preflight. These cells do not run the GPU model.
3. Download the exact public checkpoint with the supplied command (no model inference).
4. Review EXECUTION_READINESS.md and hash bindings. Obtain explicit researcher approval
   for the prepared scope/deadline; save its actual JSON at the notebook's approval path.
5. Run GPU gate; proceed to fresh F train only if PASS. Save all5checkpoints to Drive.
6. Generate only epoch3/5×24 outputs; export report/HTML/blank annotations/review ZIP.
7. Stop for substantive researcher review. No automatic RO/U training or test evaluation.

GPU gate:3updates/24microbatches;2train-input beam4 memory traces;>=15%measured VRAM
headroom, real gradients/AMP/masks/frozen backbone/checkpoint roundtrip. No update replay.
Full train:5epochs/1250updates,warmup63,batch1/accum8,originalD059optimizer/schedule,
FP32parameters+FP16AMP,228modules/4915200params,beam4/max_length768. No quantization.
Fresh training does not load feasibility adapters or LED-base checkpoints.

Colab T4 is the candidate, not a verified fit. Stop on OOM/numerical/integrity/mask/
driver or precision failures; no fallback or silent retry. Keep all historical clocks.
New proposal needs<=48h and a deadline no later than2026-10-26T00:00:00Z with>=14days
reserved for thesis evaluation/writing; researcher must confirm the actual date.

Checkpoints are directly persistent in Drive. Resume is NEVER automatic. Only a
separately approved intact epoch checkpoint with no later logged optimizer updates
is eligible; no rollback/repeated optimization, no silent resume after numerical failure.
Checkpoint roundtrip is not a guarantee of exact resumed CUDA replay: preserve D051FAIL.

The full notebook is prepared and tested offline, not executed on Colab or a GPU.
See EXECUTION_READINESS.md for verification scope and remaining runtime uncertainties.
''')
    write(DEST/'CODEX_HANDOFF.md','''# Use the prepared FULL notebook v2

The large-specific runner is already included. Do not invoke old D059 base runners,
reimplement a different training loop or weaken guards. Read README_FIRST.md and
EXECUTION_READINESS.md; run all offline tests/preflight in the actual Colab environment.
No GPU work is authorized by this handoff. Bind new researcher approval to CONFIG,
PREFLIGHT and RUNNER hashes in EXECUTION_BINDINGS.json, actual UTC timestamp/deadline,
conditional F scope, confirmed14day defense reserve. No old experiment approval reuse.
Then follow the exact gate -> freshFtrain ->48outputs ->quality-review notebook cells.
Stop on any failure. Do not modify frozen data/targets/references, add experiments,
use held-out contents, call Teacher API or start RO/U. Preserve D051FAIL and all history.
''')
    write(DEST/'EXECUTION_READINESS.md','''# Complete prepared F notebook — GPU approval still required

## Scientific scope unchanged from the LED-large proposal
Official revision cd59d11c3528415b7dda4dfc95cc8f138aceda2e; fresh Expanded-LoRA,
228modules/456tensors/4915200params;r8/alpha16/dropout0;frozenbackbone459801600params.
F2000/200;source5000/reviews4000/captions750/K5/labels768;exact frozen token IDs.
5epochs/1250updates,warmup63;D059AdamW2e-4/WD.01,batch1/accum8,linear,FP16AMP with
FP32parameterstorage,non-reentrantcheckpointing,seed42/orderunchanged;saveall5epochs;
noearlystop/bestlossreload. Beam4/do_sampleFalse/max_length768/no-repeat0,epochs3/5
only:48validationdevelopmentoutputs;random20/targeted4 separate. NoRO/U/test/Teacher.

## Implemented operational steps
CPU checksum of the pinned public binary; strict tensor key/shape/dtype inventory;
actual pretrained CPU state load without remote code; exact LoRA attachment and
namedseedA/zeroB hashes; frozenbackbone fingerprints use historical D050 convention.
Gate3updates/24microbatches +2train-only fullbeam memorytraces;gradient/AMP/mask/EOS/
sourceandlabelchecks;optimizer/scheduler/scaler/RNGandadapterroundtrip,noextra replay;
>=15%measured headroom, allocated/reserved and non-Torch occupancy recorded.
Training starts fresh in a separate subprocess. Actual driver/runtime/hardware must
match gate. Train/validationloss finite and all456adaptertensor gradients checked.
All5epoch checkpoints include model/optimizer/scheduler/scaler/RNG/trainer state and
SHA256 receipts. No overwrites/automatic retries. Full generation leaves rawoutputs
unchanged. CPU export provides source/photo/reference/benchmark review with no human
labels or automaticqualityGO. Fullbackup is optional; Drive retains checkpoints.

## Disclosed technical loader correction / recovery boundary
The pinned Torch2.1 Trainer RNG loader uses weights_only=True for NumPy RNG state,
which can reject that own-checkpoint pickle. The prepared subclass explicitly loads
only a hash-verified locally generated rng_state.pth with weights_only=False and
restores the same Python/NumPy/CPU/CUDA RNG components. This changes loader mechanics,
not training settings; it must be reviewed with the runner. No arbitrary downloaded
state pickle is accepted for resume. The public backbone uses weights_only=True.
Manual resume requires a separate approved checkpoint-integrity SHA and matching
feasibility/deadline; last logged update must equal the checkpoint step. Any later
uncheckpointed optimization blocks resume rather than silently repeating it. No claim
of exact CUDA replay is made; D051 strict replayFAIL is preserved. The gate tests
same-process full-state roundtrip, not a complete multi-session Trainer resume.

## Verification limits and required authorization
Offline tests invoke real callback JSON serialization/checkpoint receipt code and actual
META LED/PEFT attachment; config/generation/budget drift and no-approval executions are
negative controls. No pretrained model values have been loaded locally, no real GPU
forward/backward/generation/optimizer update, no T4 VRAM or throughput measurement,
no actual Colab installation and no successful resumed training are claimed.
Binary-value/inventory compatibility, gradients, numerical behavior, resources and
actual checkpoint roundtrip remain mandatory execution checks; any failure STOPS.
No quantization, batch/precision/length change or automatic model fallback is permitted.

Review full hashes and authorize bounded gate + conditional fresh F train +48outputs
with actual UTC timestamp, <=48h deadline capped at2026-10-26T00:00:00Z and confirmed
>=14day thesis reserve. Historical D059/D060/UL clocks are not modified. Authorization
template remainsfalse. This is complete executable preparation, not a GPU PASS.
''')
    prep=DEST/NEWPREP;log=prep/'OFFLINE_TESTS.log'
    env=__import__('os').environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONIOENCODING='utf-8',TOKENIZERS_PARALLELISM='false')
    with log.open('x',encoding='utf-8') as f:
        for pattern in ('test_led_large_colab_transfer_v1.py','test_stage_i_led_large_preparation_v1.py','test_led_large_full_notebook_v2.py'):
            f.write('\n'+pattern+'\n');f.flush()
            p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=DEST,env=env,stdout=f,stderr=subprocess.STDOUT)
            assert p.returncode==0, 'Offline test failure; preserve log and stop'
    write(prep/'OFFLINE_TEST_RESULTS.json',dict(status='PASS',tests=42,model_forward_backward_generation=0,GPU_operations=0,
        actual_callback_JSON_serialization=True,actual_META_228_module_attachment=True,Colab_installation_tested=False))
    # Prepare immutable bindings after all test output is written. Exclude generated Python caches.
    immutable={p.relative_to(DEST).as_posix():sha(p) for p in payload_files()}
    write(prep/'PREFLIGHT_MANIFEST.json',dict(files=immutable,GPU_execution_approved=False,
          scope='COMPLETE_PREPARED_LARGE_F_ONLY_RUNTIME_AND_FROZEN_PAYLOAD',pretrained_binary_verified_at_execution=True))
    bindings=dict(config_sha256=sha(DEST/'configs/stage_i_led_large_colab_v2.json'),
        preflight_manifest_sha256=sha(prep/'PREFLIGHT_MANIFEST.json'),
        runner_sha256=sha(DEST/'scripts/run_stage_i_led_large_colab_v2.py'),
        helper_sha256=sha(DEST/'scripts/stage_i_led_large_runtime_v2.py'),GPU_execution_approved=False)
    write(prep/'EXECUTION_BINDINGS.json',bindings)
    approval=dict(experiment='LED_LARGE_F_COLAB_V2',GPU_execution_approved=False,
        approval_source='researcher_message',researcher_explicit_authorization=None,
        execution_authorized_at_utc=None,execution_deadline_utc=None,
        defense_reserve_at_least_14_days_confirmed=False,scopes=['feasibility','train','evaluate'],
        conditional_train_after_feasibility_pass=True,max_training_updates=1250,feasibility_updates_max=3,
        feasibility_microbatches_max=24,technical_generation_calls_max=2,evaluation_outputs_max=48,
        held_out_access=False,Teacher_calls=False,RO_U_training_authorized=False,
        resume_after_interruption_approved=False,resume_checkpoint_integrity_sha256=None,
        approved_feasibility_report_sha256=None,**{k:v for k,v in bindings.items() if k.endswith('sha256')})
    write(prep/'GPU_AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',approval)
    for p in payload_files():
        if p.suffix in ('.json','.jsonl','.py','.md','.txt','.ipynb'):
            assert not re.search(rb'sk-[A-Za-z0-9]{32,}',p.read_bytes()), 'Credential-like string; stop'
    paths=payload_files()
    with ThreadPoolExecutor(max_workers=12) as pool:
        manifest=dict(zip((p.relative_to(DEST).as_posix() for p in paths),pool.map(sha,paths)))
    write(DEST/'SHA256_MANIFEST.json',dict(files=manifest,payload_files=len(manifest)))
    archive=DEST.parent/'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in payload_files():z.write(p,p.relative_to(DEST).as_posix())
    digest=sha(archive)
    write(Path(str(archive)+'.sha256'),digest+'  '+archive.name)
    shutil.copy2(DEST/'LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb',DEST.parent/'LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb')
    write(DEST.parent/'LED_LARGE_F_FULL_TRAIN_COLAB_V2_RECEIPT.json',dict(archive=archive.name,bytes=archive.stat().st_size,
        sha256=digest,files=len(manifest),tests=42,execution_bindings=bindings,GPU_execution_approved=False))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest,bindings=bindings),indent=2),flush=True)


if __name__=='__main__':main()
